"""Meta-learnable Adam optimizer with CompletedP parameterization support.

This module implements a learned optimizer that uses the standard Adam update rule
but with meta-learnable per-tensor-type hyperparameter multipliers. Unlike the
MLP-based MuCompletedPAdafacMLPLOpt, this optimizer's meta-parameters are simply
scalar offsets in log-space for each (tensor_type, hyperparameter) pair.

Meta-learned hyperparameters (per tensor type):
- Learning rate multiplier
- Epsilon multiplier
- Weight decay multiplier
- β₁ (first moment decay)
- β₂ (second moment decay)

The meta-learned multipliers are applied on top of CompletedP scaling factors
from model_state, enabling hyperparameter transfer across model scales.

The scaling factors should be passed via model_state:
- mup_lr_scales: LR scales
- mup_eps_scales: Epsilon scales
- mup_wd_scales: Weight decay scales
- mup_one_minus_beta1_scales: (1-β₁) scales
- mup_one_minus_beta2_scales: (1-β₂) scales
- mup_tensor_type_indices: Integer index per parameter (into TensorType enum)
"""
from typing import Any, Optional

import flax
import gin
import jax
import jax.numpy as jnp
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
import numpy as onp
import optax

PRNGKey = jnp.ndarray


# ============================================================================
# Log-space parameterization helpers
# ============================================================================

def decay_to_param(x):
    """Convert decay value to learnable parameter space: log(1-x) / 5."""
    return jnp.log(1 - x) / 5.0


def param_to_decay(x):
    """Convert learnable parameter to decay value: 1 - exp(x * 5)."""
    return 1 - jnp.exp(x * 5.0)


def mult_to_param(x):
    """Convert positive multiplier to learnable parameter space: log(x) / 5."""
    return jnp.log(x) / 5.0


def param_to_mult(x):
    """Convert learnable parameter to positive multiplier: exp(x * 5."""
    return jnp.exp(x * 5.0)


# ============================================================================
# State
# ============================================================================

@flax.struct.dataclass
class CompletedPAdamLOptState:
    """State for the CompletedP Adam learned optimizer.

    Attributes:
        params: Current model parameters
        state: Model state (includes CompletedP scales and tensor type indices)
        m: First moment estimate (pytree matching params)
        v: Second moment estimate (pytree matching params)
        num_steps: Total number of inner training steps
        iteration: Current inner iteration count
    """
    params: Any
    state: Any
    m: Any
    v: Any
    num_steps: jnp.ndarray
    iteration: jnp.ndarray


# ============================================================================
# Main learned optimizer
# ============================================================================

@gin.configurable
class MuCompletedPAdamLOpt(lopt_base.LearnedOptimizer):
    """Adam-based learned optimizer with CompletedP parameterization support.

    Meta-learns per-tensor-type hyperparameter offsets (in log-space) on top of
    CompletedP scaling factors. Initialized from known-good global HP values.

    Includes a built-in warmup + cosine decay LR schedule.

    Args:
        initial_lr: Initial base learning rate (from sweep optimal).
        initial_b1: Initial β₁ (from sweep optimal).
        initial_b2: Initial β₂ (from sweep optimal).
        initial_eps: Initial epsilon for Adam.
        initial_wd: Initial weight decay (from sweep optimal).
        clip_grad: Whether to clip gradients by global norm.
        clip_norm: Max gradient norm for clipping.
        num_tensor_types: Number of tensor types (19 covers all TensorType enum values).
        warmup_steps: Number of warmup steps for built-in LR schedule.
        decay_steps: Number of decay steps after warmup.
        end_lr_ratio: Ratio of end LR to peak LR (0.5 = decay to half).
    """

    def __init__(
        self,
        initial_lr=3.410952e-03,
        initial_b1=0.95484,
        initial_b2=0.9908,
        initial_eps=1e-8,
        initial_wd=0.093198,
        clip_grad=True,
        clip_norm=1.0,
        num_tensor_types=19,
        warmup_steps=100,
        decay_steps=1900,
        end_lr_ratio=0.5,
    ):
        super().__init__()
        self._initial_lr = initial_lr
        self._initial_b1 = initial_b1
        self._initial_b2 = initial_b2
        self._initial_eps = initial_eps
        self._initial_wd = initial_wd
        self._clip_grad = clip_grad
        self._clip_norm = clip_norm
        self._num_tensor_types = num_tensor_types
        self._warmup_steps = warmup_steps
        self._decay_steps = decay_steps
        self._end_lr_ratio = end_lr_ratio

    def init(self, key: PRNGKey) -> lopt_base.MetaParams:
        """Initialize the learned optimizer's meta-parameters.

        All offsets are initialized to 0.0, meaning the optimizer starts with
        the sweep-optimal hyperparameters exactly.

        Returns:
            Dictionary of meta-parameters (5 global + 5*N per-tensor + 2 schedule).
        """
        n = self._num_tensor_types
        return {
            # Global HP offsets (in log-space, 0.0 = no change from initial)
            "base_lr_offset": jnp.zeros([]),
            "base_b1_offset": jnp.zeros([]),
            "base_b2_offset": jnp.zeros([]),
            "base_eps_offset": jnp.zeros([]),
            "base_wd_offset": jnp.zeros([]),
            # Per-tensor-type offsets (in log-space, 0.0 = multiplier of 1.0)
            "per_tensor_lr_offsets": jnp.zeros([n]),
            "per_tensor_b1_offsets": jnp.zeros([n]),
            "per_tensor_b2_offsets": jnp.zeros([n]),
            "per_tensor_eps_offsets": jnp.zeros([n]),
            "per_tensor_wd_offsets": jnp.zeros([n]),
            # Schedule offsets
            "schedule_warmup_offset": jnp.zeros([]),
            "schedule_end_lr_offset": jnp.zeros([]),
        }

    def opt_fn(
        self,
        theta: lopt_base.MetaParams,
        is_training: Optional[bool] = False,
    ) -> opt_base.Optimizer:
        """Create an optimizer instance from the meta-parameters."""
        parent = self

        class _Opt(opt_base.Optimizer):
            """Inner optimizer with meta-learned per-tensor-type Adam HPs."""

            def __init__(self, theta):
                self.theta = theta

            def _compute_schedule(self, iteration, num_steps):
                """Warmup + cosine decay schedule with meta-learnable end ratio.

                Returns a scalar multiplier in [end_ratio, 1.0].
                """
                # Meta-learnable warmup and end_lr adjustments
                warmup = jnp.float32(parent._warmup_steps)
                total = jnp.float32(parent._warmup_steps + parent._decay_steps)
                end_ratio = jnp.clip(
                    parent._end_lr_ratio * param_to_mult(self.theta["schedule_end_lr_offset"]),
                    0.0, 1.0,
                )

                t = jnp.float32(iteration)

                # Warmup phase: linear ramp from 0 to 1
                warmup_mult = jnp.where(warmup > 0, jnp.minimum(t / warmup, 1.0), 1.0)

                # Cosine decay phase
                progress = jnp.clip((t - warmup) / jnp.maximum(total - warmup, 1.0), 0.0, 1.0)
                decay_mult = end_ratio + (1.0 - end_ratio) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))

                return jnp.where(t < warmup, warmup_mult, decay_mult)

            def init(
                self,
                params: opt_base.Params,
                model_state: Optional[opt_base.ModelState] = None,
                num_steps: Optional[int] = None,
                key: Optional[PRNGKey] = None,
            ) -> CompletedPAdamLOptState:
                """Initialize optimizer state with zero moments."""
                if num_steps is None:
                    raise ValueError("Must specify number of steps for this lopt!")

                m = jax.tree_util.tree_map(jnp.zeros_like, params)
                v = jax.tree_util.tree_map(jnp.zeros_like, params)

                return CompletedPAdamLOptState(
                    params=params,
                    state=model_state,
                    m=m,
                    v=v,
                    num_steps=jnp.asarray(num_steps),
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                )

            def update(
                self,
                opt_state: CompletedPAdamLOptState,
                grad: opt_base.Gradient,
                loss: jnp.ndarray,
                model_state: Optional[opt_base.ModelState] = None,
                is_valid: bool = False,
                key: Optional[PRNGKey] = None,
            ) -> CompletedPAdamLOptState:
                """Update parameters using Adam with meta-learned per-tensor HPs.

                1. Extract CompletedP scales and tensor_type_indices from model_state
                2. Compute effective per-param HPs (base + per-tensor offset + CP scaling)
                3. Standard Adam update with bias correction
                4. Apply built-in LR schedule
                """
                # Optional gradient clipping
                if parent._clip_grad:
                    clipping = optax.clip_by_global_norm(parent._clip_norm)
                    grad, _ = clipping.update(grad, None)

                # Handle NaN gradients
                grad = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grad)

                # Extract CompletedP scales from model_state
                ones = jax.tree_util.tree_map(lambda x: jnp.ones(()), grad)
                zeros_int = jax.tree_util.tree_map(
                    lambda x: jnp.zeros((), dtype=jnp.int32), grad
                )
                lr_scales = model_state.get("mup_lr_scales", ones)
                eps_scales = model_state.get("mup_eps_scales", ones)
                wd_scales = model_state.get("mup_wd_scales", ones)
                b1_scales = model_state.get("mup_one_minus_beta1_scales", ones)
                b2_scales = model_state.get("mup_one_minus_beta2_scales", ones)
                tt_indices = model_state.get("mup_tensor_type_indices", zeros_int)

                # Current step (1-indexed for bias correction)
                t = jnp.float32(opt_state.iteration + 1)

                # ---- Compute base HPs with meta-learned global offsets ----
                base_lr = param_to_mult(
                    mult_to_param(jnp.float32(parent._initial_lr))
                    + self.theta["base_lr_offset"]
                )
                base_eps = param_to_mult(
                    mult_to_param(jnp.float32(parent._initial_eps))
                    + self.theta["base_eps_offset"]
                )
                base_wd = param_to_mult(
                    mult_to_param(jnp.float32(parent._initial_wd))
                    + self.theta["base_wd_offset"]
                )
                base_b1_param = decay_to_param(jnp.float32(parent._initial_b1)) + self.theta["base_b1_offset"]
                base_b2_param = decay_to_param(jnp.float32(parent._initial_b2)) + self.theta["base_b2_offset"]

                # Per-tensor-type offset arrays
                per_tt_lr = self.theta["per_tensor_lr_offsets"]
                per_tt_eps = self.theta["per_tensor_eps_offsets"]
                per_tt_wd = self.theta["per_tensor_wd_offsets"]
                per_tt_b1 = self.theta["per_tensor_b1_offsets"]
                per_tt_b2 = self.theta["per_tensor_b2_offsets"]

                # ---- Compute effective per-param HP trees ----

                # Effective β₁ per parameter (with CompletedP (1-β) scaling)
                def compute_eff_b1(b1_scale, tt_idx):
                    b1_raw = param_to_decay(base_b1_param + per_tt_b1[tt_idx])
                    eff_one_minus_b1 = (1.0 - b1_raw) * b1_scale
                    return 1.0 - eff_one_minus_b1

                eff_b1_tree = jax.tree_util.tree_map(
                    compute_eff_b1, b1_scales, tt_indices
                )

                # Effective β₂ per parameter
                def compute_eff_b2(b2_scale, tt_idx):
                    b2_raw = param_to_decay(base_b2_param + per_tt_b2[tt_idx])
                    eff_one_minus_b2 = (1.0 - b2_raw) * b2_scale
                    return 1.0 - eff_one_minus_b2

                eff_b2_tree = jax.tree_util.tree_map(
                    compute_eff_b2, b2_scales, tt_indices
                )

                # Effective LR per parameter (before schedule)
                def compute_eff_lr(lr_scale, tt_idx):
                    return base_lr * lr_scale * param_to_mult(per_tt_lr[tt_idx])

                eff_lr_tree = jax.tree_util.tree_map(
                    compute_eff_lr, lr_scales, tt_indices
                )

                # Effective epsilon per parameter
                def compute_eff_eps(eps_scale, tt_idx):
                    return base_eps * eps_scale * param_to_mult(per_tt_eps[tt_idx])

                eff_eps_tree = jax.tree_util.tree_map(
                    compute_eff_eps, eps_scales, tt_indices
                )

                # Effective weight decay per parameter
                def compute_eff_wd(wd_scale, tt_idx):
                    return base_wd * wd_scale * param_to_mult(per_tt_wd[tt_idx])

                eff_wd_tree = jax.tree_util.tree_map(
                    compute_eff_wd, wd_scales, tt_indices
                )

                # ---- Adam moment updates ----
                next_m = jax.tree_util.tree_map(
                    lambda m, g, eb1: eb1 * m + (1.0 - eb1) * g,
                    opt_state.m, grad, eff_b1_tree,
                )
                next_v = jax.tree_util.tree_map(
                    lambda v, g, eb2: eb2 * v + (1.0 - eb2) * (g ** 2),
                    opt_state.v, grad, eff_b2_tree,
                )

                # ---- Bias correction ----
                m_hat = jax.tree_util.tree_map(
                    lambda m, eb1: m / (1.0 - eb1 ** t),
                    next_m, eff_b1_tree,
                )
                v_hat = jax.tree_util.tree_map(
                    lambda v, eb2: v / (1.0 - eb2 ** t),
                    next_v, eff_b2_tree,
                )

                # ---- LR schedule ----
                schedule_mult = self._compute_schedule(
                    opt_state.iteration, opt_state.num_steps
                )

                # ---- Parameter update: p = p - schedule * lr * (adam_step + wd * p) ----
                next_params = jax.tree_util.tree_map(
                    lambda p, mh, vh, e_eps, e_wd, e_lr: (
                        p - schedule_mult * e_lr * (
                            mh / (jnp.sqrt(vh) + e_eps) + e_wd * p
                        )
                    ),
                    opt_state.params, m_hat, v_hat,
                    eff_eps_tree, eff_wd_tree, eff_lr_tree,
                )

                # ---- Merge model state ----
                if model_state is not None and opt_state.state is not None:
                    merged_state = dict(opt_state.state)
                    merged_state.update(model_state)
                elif model_state is not None:
                    merged_state = model_state
                else:
                    merged_state = opt_state.state

                next_opt_state = CompletedPAdamLOptState(
                    params=next_params,
                    state=merged_state,
                    m=next_m,
                    v=next_v,
                    iteration=opt_state.iteration + 1,
                    num_steps=opt_state.num_steps,
                )

                return tree_utils.match_type(next_opt_state, opt_state)

        return _Opt(theta)
