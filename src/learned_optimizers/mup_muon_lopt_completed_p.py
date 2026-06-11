"""Meta-learnable MuP Muon optimizer with CompletedP parameterization support.

Counterpart to ``mup_adam_lopt_completed_p.py`` for the hybrid Muon/Adam
optimizer in ``src/opt/mup_muon.py``. Hidden 2D/3D weights (``ModuleType.HIDDEN_WEIGHT``)
get a momentum + Newton-Schulz update; embeddings, biases, norms, and the
unembedding head get an Adam update. Both sides share a single base learning
rate but have independent momentum / epsilon / weight-decay HPs that are
**meta-learned per tensor type** on top of the CompletedP scaling factors
delivered through ``model_state``.

Validated reference for the Muon scaling rules:
- Qiu et al. 2025, "Hyperparameter Transfer Enables Consistent Gains of
  Matrix-Preconditioned Optimizers Across Scales" (arXiv:2512.05620).
- Dey et al. 2025, "Completed Hyperparameter Transfer across Modules, Width,
  Depth, Batch and Duration" (arXiv:2512.22382).

Meta-learned hyperparameters per tensor type (8 = LR + 3 Muon + 4 Adam):
- Learning rate multiplier (shared across both sides)
- Muon momentum β
- Muon Newton-Schulz damping ε
- Muon weight decay
- Adam β₁
- Adam β₂
- Adam ε
- Adam weight decay

Scales consumed from ``model_state`` (populated by
``mu_task_base._compute_completed_p_scales`` when configured for muon):
- ``mup_muon_lr_scales``       — per-param Muon LR scale (carries √(d_out/d_in))
- ``mup_muon_eps_scales``      — per-param Muon Newton-Schulz ε scale
- ``mup_muon_shape_scales``    — per-param √(d_out/d_in) shape factor (matches
  ``mup_muon._compute_shape_scales`` so the lopt is equivalent to the
  validated reference rather than the paper's once-applied form)
- ``mup_lr_scales``            — per-param Adam LR scale (for non-Muon params)
- ``mup_eps_scales``           — per-param Adam ε scale
- ``mup_wd_scales``            — per-param weight-decay scale (shared)
- ``mup_one_minus_beta1_scales``— Adam (1-β₁) SDE scale (Adam side only)
- ``mup_one_minus_beta2_scales``— Adam (1-β₂) SDE scale (Adam side only)
- ``mup_tensor_type_indices``  — int32 per param, index into the TensorType enum
- ``mup_is_muon_mask``         — bool per param, True for Muon-side leaves

Design notes:
- Muon momentum is intentionally **not** scaled by ``(1-β)`` SDE factors.
  Those scales are derived from the Adam SDE limit and have no published
  Muon counterpart (Qiu et al. do not derive batch/duration corrections for
  Muon momentum). Per-tensor offsets remain meta-learnable.
- The √(d_out/d_in) shape factor is applied **twice** along the Muon update
  path to match the validated reference ``src/opt/mup_muon.py`` exactly:
  once via the per-leaf ``mup_muon_shape_scales`` pytree (analogous to the
  reference's internal ``_compute_shape_scales`` cache) and again via
  ``mup_muon_lr_scales`` (which carries ``√(d_out/d_in)·√(m_B/m_D)`` from
  ``MuonCompletedPParameterization``). Net contribution to the orth update is
  ``(d_out/d_in)·√(m_B/m_D)``. This double-counting is empirically the
  HP-transfer regime the reference was validated in; preserving it lets the
  meta-learned optimizer's optimal HPs transfer to the validated reference.
- Moment buffers (``muon_mu``, ``adam_m``, ``adam_v``) default to float32
  regardless of param dtype to match the precision of the validated
  reference (overridable via ``mu_dtype`` constructor arg).
- Newton-Schulz coefficients and ndim==3 reshape logic are reused verbatim
  from ``src/opt/mup_muon.py``.
"""
from typing import Any, Optional, Tuple

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
# Log-space parameterization helpers (identical to mup_adam_lopt_completed_p)
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
# Newton-Schulz orthogonalization (verbatim port from src/opt/mup_muon.py)
# ============================================================================

def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: jax.Array = 1e-8,
) -> jax.Array:
    """Newton-Schulz orthogonalization for Muon optimizer.

    Mirrors ``orthogonalize_via_newton_schulz`` in ``src/opt/mup_muon.py`` but
    accepts a JAX-array epsilon so per-tensor epsilon can be threaded through
    a tree_map without recompilation.
    """
    if x.ndim < 2:
        raise ValueError(f"Input must have >= 2 dims, got {x.shape}")
    if ns_coeffs.shape != (3,):
        raise ValueError(f"ns_coeffs must have shape (3,), got {ns_coeffs.shape}")

    def newton_schulz_iterator(x_inner, coeffs):
        x_mT = jnp.swapaxes(x_inner, -2, -1)
        a = x_inner @ x_mT
        b = coeffs[1] * a + coeffs[2] * a @ a
        return coeffs[0] * x_inner + b @ x_inner

    transposed = False
    if x.shape[-2] > x.shape[-1]:
        x = jnp.swapaxes(x, -2, -1)
        transposed = True
    x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(
        0, ns_steps,
        lambda _, x_inner: newton_schulz_iterator(x_inner, ns_coeffs),
        x,
    )
    if transposed:
        x = jnp.swapaxes(x, -2, -1)
    return x


def _muon_step_for_leaf(
    grad: jax.Array,
    eff_eps: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int,
) -> jax.Array:
    """Apply Newton-Schulz to a single param leaf.

    Handles ndim==2 directly, reshapes ndim==3 (attention 3D kernels) using the
    same heuristic as ``src/opt/mup_muon.py``: collapse to a 2-D matrix where
    the larger leading axis becomes the row count. For ndim<2 (1-D / scalar
    biases / norms — these only ever appear on the Adam side, so this branch
    should be unreachable from the Muon side), return zeros to match the
    enclosing ``_maybe_ns`` short-circuit semantics.
    """
    original_shape = grad.shape
    if grad.ndim == 2:
        grad_2d = grad
    elif grad.ndim == 3:
        if grad.shape[0] >= grad.shape[-1]:
            grad_2d = grad.reshape(grad.shape[0], -1)
        else:
            grad_2d = grad.reshape(-1, grad.shape[-1])
    else:
        return jnp.zeros_like(grad)
    orth = orthogonalize_via_newton_schulz(grad_2d, ns_coeffs, ns_steps, eff_eps)
    return orth.reshape(original_shape)


# ============================================================================
# State
# ============================================================================

@flax.struct.dataclass
class CompletedPMuonLOptState:
    """State for the CompletedP Muon learned optimizer.

    Attributes:
        params: Current model parameters.
        state: Model state (carries CompletedP scales + tensor_type_indices
            + is_muon mask).
        muon_mu: Muon momentum buffer (pytree matching params).
        adam_m: Adam first moment (pytree matching params).
        adam_v: Adam second moment (pytree matching params).
        num_steps: Total number of inner training steps.
        iteration: Current inner iteration count.
    """
    params: Any
    state: Any
    muon_mu: Any
    adam_m: Any
    adam_v: Any
    num_steps: jnp.ndarray
    iteration: jnp.ndarray


# ============================================================================
# Main learned optimizer
# ============================================================================

@gin.configurable
class MuCompletedPMuonLOpt(lopt_base.LearnedOptimizer):
    """MuP Muon learned optimizer with CompletedP parameterization support.

    Meta-learns per-tensor-type hyperparameter offsets (in log-space) on top of
    the CompletedP scaling factors emitted by
    ``MuonCompletedPParameterization``. Hidden 2D/3D weights are updated with
    Muon (momentum + Newton-Schulz); all other params are updated with Adam.

    Includes a built-in warmup + cosine decay LR schedule with a meta-learnable
    end-LR ratio (mirrors ``MuCompletedPAdamLOpt``).

    Args:
        initial_lr: Initial base learning rate (shared across Muon and Adam
            sides — matches ``mup_muon`` factory default 0.02).
        initial_muon_beta: Initial Muon momentum (default 0.9).
        initial_muon_eps: Initial Muon Newton-Schulz damping ε (default 1e-8).
        initial_muon_wd: Initial Muon weight decay (default 1e-4).
        initial_adam_b1: Initial Adam β₁ (default 0.9).
        initial_adam_b2: Initial Adam β₂ (default 0.99).
        initial_adam_eps: Initial Adam ε (default 1e-8).
        initial_adam_wd: Initial Adam weight decay (default 1e-4).
        ns_coeffs: Newton-Schulz polynomial coefficients (default Jordan et al.).
        ns_steps: Newton-Schulz iterations (default 5).
        nesterov: Whether to use Nesterov momentum on the Muon side
            (default True).
        clip_grad: Whether to clip gradients by global norm.
        clip_norm: Max gradient norm for clipping.
        num_tensor_types: Number of tensor types (19 covers all TensorType enum
            values).
        warmup_steps: Number of warmup steps for built-in LR schedule.
        decay_steps: Number of decay steps after warmup.
        end_lr_ratio: Ratio of end LR to peak LR (0.5 = decay to half).
        mu_dtype: Optional dtype override for the moment buffers
            (``muon_mu``, ``adam_m``, ``adam_v``). Defaults to ``jnp.float32``
            so meta-training and meta-test get full precision regardless of
            param dtype (matches the validated reference's recommended
            precision). Set to ``None`` to inherit param dtype, or to
            ``jnp.bfloat16`` for memory savings at the cost of precision.
    """

    def __init__(
        self,
        initial_lr: float = 0.0221,
        initial_muon_beta: float = 0.95,
        initial_muon_eps: float = 1e-8,
        initial_muon_wd: float = 0.125,
        initial_adam_b1: float = 0.95484,
        initial_adam_b2: float = 0.9908,
        initial_adam_eps: float = 1e-8,
        initial_adam_wd: float = 0.093198,
        ns_coeffs: Tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        ns_steps: int = 5,
        nesterov: bool = True,
        clip_grad: bool = True,
        clip_norm: float = 1.0,
        num_tensor_types: int = 19,
        warmup_steps: int = 100,
        decay_steps: int = 1900,
        end_lr_ratio: float = 0.1,
        mu_dtype: Any = jnp.float32,
        use_global_offsets: bool = False,
    ):
        super().__init__()
        self._initial_lr = initial_lr
        self._initial_muon_beta = initial_muon_beta
        self._initial_muon_eps = initial_muon_eps
        self._initial_muon_wd = initial_muon_wd
        self._initial_adam_b1 = initial_adam_b1
        self._initial_adam_b2 = initial_adam_b2
        self._initial_adam_eps = initial_adam_eps
        self._initial_adam_wd = initial_adam_wd
        self._ns_coeffs = tuple(ns_coeffs)
        self._ns_steps = int(ns_steps)
        self._nesterov = bool(nesterov)
        self._clip_grad = clip_grad
        self._clip_norm = clip_norm
        self._num_tensor_types = num_tensor_types
        self._warmup_steps = warmup_steps
        self._decay_steps = decay_steps
        self._end_lr_ratio = end_lr_ratio
        self._mu_dtype = mu_dtype
        self._use_global_offsets = use_global_offsets

    def init(self, key: PRNGKey) -> lopt_base.MetaParams:
        """Initialize meta-parameters.

        All offsets start at 0.0, meaning the optimizer begins with the
        ``initial_*`` constructor values exactly. Per-tensor offsets are dense
        across all 19 ``TensorType`` slots; the per-leaf ``is_muon`` mask
        decides which side actually contributes per parameter, so unused
        offsets simply receive zero gradient through that step.
        """
        n = self._num_tensor_types
        meta_params = {}
        if self._use_global_offsets:
            meta_params.update({
                # Global HP offsets (log-space, 0.0 = no change from initial)
                "base_lr_offset": jnp.zeros([]),
                "base_muon_beta_offset": jnp.zeros([]),
                "base_muon_eps_offset": jnp.zeros([]),
                "base_muon_wd_offset": jnp.zeros([]),
                "base_adam_b1_offset": jnp.zeros([]),
                "base_adam_b2_offset": jnp.zeros([]),
                "base_adam_eps_offset": jnp.zeros([]),
                "base_adam_wd_offset": jnp.zeros([]),
                # Schedule
                "schedule_end_lr_offset": jnp.zeros([]),
            })
        meta_params.update({
            # Per-tensor-type offsets (length n; 0.0 = identity)
            "per_tensor_lr_offsets": jnp.zeros([n]),
            "per_tensor_muon_beta_offsets": jnp.zeros([n]),
            "per_tensor_muon_eps_offsets": jnp.zeros([n]),
            "per_tensor_muon_wd_offsets": jnp.zeros([n]),
            "per_tensor_adam_b1_offsets": jnp.zeros([n]),
            "per_tensor_adam_b2_offsets": jnp.zeros([n]),
            "per_tensor_adam_eps_offsets": jnp.zeros([n]),
            "per_tensor_adam_wd_offsets": jnp.zeros([n]),
        })
        return meta_params

    def opt_fn(
        self,
        theta: lopt_base.MetaParams,
        is_training: Optional[bool] = False,
    ) -> opt_base.Optimizer:
        parent = self
        ns_coeffs_arr = jnp.array(parent._ns_coeffs, dtype=jnp.float32)

        class _Opt(opt_base.Optimizer):
            """Inner optimizer with meta-learned per-tensor-type Muon+Adam HPs."""

            def __init__(self, theta):
                self.theta = theta

            def _compute_schedule(self, iteration, num_steps):
                """Warmup + cosine decay schedule with meta-learnable end ratio."""
                warmup = jnp.float32(parent._warmup_steps)
                total = jnp.float32(parent._warmup_steps + parent._decay_steps)
                end_ratio = jnp.clip(
                    parent._end_lr_ratio * (param_to_mult(self.theta["schedule_end_lr_offset"]) if parent._use_global_offsets else 1.0),
                    0.0, 1.0,
                )
                t = jnp.float32(iteration)
                warmup_mult = jnp.where(warmup > 0, jnp.minimum(t / warmup, 1.0), 1.0)
                progress = jnp.clip(
                    (t - warmup) / jnp.maximum(total - warmup, 1.0),
                    0.0, 1.0,
                )
                decay_mult = end_ratio + (1.0 - end_ratio) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
                return jnp.where(t < warmup, warmup_mult, decay_mult)

            def init(
                self,
                params: opt_base.Params,
                model_state: Optional[opt_base.ModelState] = None,
                num_steps: Optional[int] = None,
                key: Optional[PRNGKey] = None,
            ) -> CompletedPMuonLOptState:
                if num_steps is None:
                    raise ValueError("Must specify number of steps for this lopt!")

                # Moment buffers default to float32 for full precision regardless
                # of param dtype (override via ``mu_dtype=None`` to inherit param
                # dtype). Matches the validated reference recommendation.
                def _zeros(p):
                    return jnp.zeros_like(p, dtype=parent._mu_dtype or p.dtype)

                muon_mu = jax.tree_util.tree_map(_zeros, params)
                adam_m = jax.tree_util.tree_map(_zeros, params)
                adam_v = jax.tree_util.tree_map(_zeros, params)

                return CompletedPMuonLOptState(
                    params=params,
                    state=model_state,
                    muon_mu=muon_mu,
                    adam_m=adam_m,
                    adam_v=adam_v,
                    num_steps=jnp.asarray(num_steps),
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                )

            def update(
                self,
                opt_state: CompletedPMuonLOptState,
                grad: opt_base.Gradient,
                loss: jnp.ndarray,
                model_state: Optional[opt_base.ModelState] = None,
                is_valid: bool = False,
                key: Optional[PRNGKey] = None,
            ) -> CompletedPMuonLOptState:
                """Hybrid Muon + Adam update with meta-learned per-tensor HPs."""
                # ---- NaN scrub then global-norm clip ----
                # Order matters: scrubbing NaNs first means a single bad leaf
                # becomes a zero leaf, rather than poisoning the *entire*
                # update via the global norm being NaN (which would zero
                # everything via clip's NaN propagation followed by the
                # subsequent ``nan_to_num``).
                grad = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grad)
                if parent._clip_grad:
                    clipping = optax.clip_by_global_norm(parent._clip_norm)
                    grad, _ = clipping.update(grad, None)

                # Allow ``model_state=None`` (the type signature claims this is
                # supported even though the Adam reference would crash on it).
                if model_state is None:
                    model_state = {}

                # ---- Extract scales / mask from model_state ----
                # Default fallbacks built from ``opt_state.params`` (NOT
                # ``grad``) so the pytree structure is guaranteed to match the
                # real scale pytrees emitted by ``mu_task_base`` (which are
                # also derived from ``params``). Building from ``grad`` would
                # be fragile if any wrapping code returned grads with a
                # different pytree structure.
                ones_like_params = jax.tree_util.tree_map(
                    lambda x: jnp.ones((), dtype=jnp.float32), opt_state.params
                )
                zeros_int_like_params = jax.tree_util.tree_map(
                    lambda x: jnp.zeros((), dtype=jnp.int32), opt_state.params
                )
                false_like_params = jax.tree_util.tree_map(
                    lambda x: jnp.zeros((), dtype=jnp.bool_), opt_state.params
                )

                muon_lr_scales = model_state.get("mup_muon_lr_scales", ones_like_params)
                muon_eps_scales = model_state.get("mup_muon_eps_scales", ones_like_params)
                muon_shape_scales = model_state.get("mup_muon_shape_scales", ones_like_params)
                adam_lr_scales = model_state.get("mup_lr_scales", ones_like_params)
                adam_eps_scales = model_state.get("mup_eps_scales", ones_like_params)
                wd_scales = model_state.get("mup_wd_scales", ones_like_params)
                b1_scales = model_state.get("mup_one_minus_beta1_scales", ones_like_params)
                b2_scales = model_state.get("mup_one_minus_beta2_scales", ones_like_params)
                tt_indices = model_state.get("mup_tensor_type_indices", zeros_int_like_params)
                # If is_muon mask is missing (e.g., non-muon meta-test task), default to
                # all-False so the optimizer degrades to pure Adam.
                is_muon = model_state.get("mup_is_muon_mask", false_like_params)

                # Current step (1-indexed for Adam bias correction)
                t = jnp.float32(opt_state.iteration + 1)

                # ---- Compute base HPs with meta-learned global offsets ----
                base_lr = param_to_mult(
                    mult_to_param(jnp.float32(parent._initial_lr))
                    + (self.theta["base_lr_offset"] if parent._use_global_offsets else 0.0)
                )
                base_muon_eps = param_to_mult(
                    mult_to_param(jnp.float32(parent._initial_muon_eps))
                    + (self.theta["base_muon_eps_offset"] if parent._use_global_offsets else 0.0)
                )
                base_muon_wd = param_to_mult(
                    mult_to_param(jnp.float32(parent._initial_muon_wd))
                    + (self.theta["base_muon_wd_offset"] if parent._use_global_offsets else 0.0)
                )
                base_adam_eps = param_to_mult(
                    mult_to_param(jnp.float32(parent._initial_adam_eps))
                    + (self.theta["base_adam_eps_offset"] if parent._use_global_offsets else 0.0)
                )
                base_adam_wd = param_to_mult(
                    mult_to_param(jnp.float32(parent._initial_adam_wd))
                    + (self.theta["base_adam_wd_offset"] if parent._use_global_offsets else 0.0)
                )
                base_muon_beta_param = (
                    decay_to_param(jnp.float32(parent._initial_muon_beta))
                    + (self.theta["base_muon_beta_offset"] if parent._use_global_offsets else 0.0)
                )
                base_adam_b1_param = (
                    decay_to_param(jnp.float32(parent._initial_adam_b1))
                    + (self.theta["base_adam_b1_offset"] if parent._use_global_offsets else 0.0)
                )
                base_adam_b2_param = (
                    decay_to_param(jnp.float32(parent._initial_adam_b2))
                    + (self.theta["base_adam_b2_offset"] if parent._use_global_offsets else 0.0)
                )

                # Per-tensor-type offset arrays
                per_tt_lr = self.theta["per_tensor_lr_offsets"]
                per_tt_muon_beta = self.theta["per_tensor_muon_beta_offsets"]
                per_tt_muon_eps = self.theta["per_tensor_muon_eps_offsets"]
                per_tt_muon_wd = self.theta["per_tensor_muon_wd_offsets"]
                per_tt_adam_b1 = self.theta["per_tensor_adam_b1_offsets"]
                per_tt_adam_b2 = self.theta["per_tensor_adam_b2_offsets"]
                per_tt_adam_eps = self.theta["per_tensor_adam_eps_offsets"]
                per_tt_adam_wd = self.theta["per_tensor_adam_wd_offsets"]

                # ---- Effective per-param HP trees ----

                # Effective LR per parameter: branch on is_muon to pick the right scale.
                def compute_eff_lr(muon_lr_scale, adam_lr_scale, tt_idx, is_m):
                    lr_scale = jnp.where(is_m, muon_lr_scale, adam_lr_scale)
                    return base_lr * lr_scale * param_to_mult(per_tt_lr[tt_idx])

                eff_lr_tree = jax.tree_util.tree_map(
                    compute_eff_lr,
                    muon_lr_scales, adam_lr_scales, tt_indices, is_muon,
                )

                # Effective Muon momentum β per parameter (decay-space, no SDE scale).
                def compute_eff_muon_beta(tt_idx):
                    return param_to_decay(base_muon_beta_param + per_tt_muon_beta[tt_idx])

                eff_muon_beta_tree = jax.tree_util.tree_map(
                    compute_eff_muon_beta, tt_indices
                )

                # Effective Muon ε per parameter
                def compute_eff_muon_eps(muon_eps_scale, tt_idx):
                    return base_muon_eps * muon_eps_scale * param_to_mult(per_tt_muon_eps[tt_idx])

                eff_muon_eps_tree = jax.tree_util.tree_map(
                    compute_eff_muon_eps, muon_eps_scales, tt_indices
                )

                # Effective Adam β₁ / β₂ per parameter (with (1-β) SDE scaling).
                def compute_eff_adam_b1(b1_scale, tt_idx):
                    b1_raw = param_to_decay(base_adam_b1_param + per_tt_adam_b1[tt_idx])
                    eff_one_minus_b1 = (1.0 - b1_raw) * b1_scale
                    return 1.0 - eff_one_minus_b1

                eff_adam_b1_tree = jax.tree_util.tree_map(
                    compute_eff_adam_b1, b1_scales, tt_indices
                )

                def compute_eff_adam_b2(b2_scale, tt_idx):
                    b2_raw = param_to_decay(base_adam_b2_param + per_tt_adam_b2[tt_idx])
                    eff_one_minus_b2 = (1.0 - b2_raw) * b2_scale
                    return 1.0 - eff_one_minus_b2

                eff_adam_b2_tree = jax.tree_util.tree_map(
                    compute_eff_adam_b2, b2_scales, tt_indices
                )

                # Effective Adam ε per parameter
                def compute_eff_adam_eps(eps_scale, tt_idx):
                    return base_adam_eps * eps_scale * param_to_mult(per_tt_adam_eps[tt_idx])

                eff_adam_eps_tree = jax.tree_util.tree_map(
                    compute_eff_adam_eps, adam_eps_scales, tt_indices
                )

                # Effective weight decay per parameter — branch on is_muon to pick
                # base + per-tensor offset, then scale by the (shared) wd_scales.
                def compute_eff_wd(wd_scale, tt_idx, is_m):
                    muon_wd = base_muon_wd * param_to_mult(per_tt_muon_wd[tt_idx])
                    adam_wd = base_adam_wd * param_to_mult(per_tt_adam_wd[tt_idx])
                    base = jnp.where(is_m, muon_wd, adam_wd)
                    return base * wd_scale

                eff_wd_tree = jax.tree_util.tree_map(
                    compute_eff_wd, wd_scales, tt_indices, is_muon
                )

                # ---- Muon side: momentum + (optional) Nesterov + Newton-Schulz ----
                # Cast grad to the moment buffer dtype so the EMA stays in
                # the buffer's precision (matches the validated reference at
                # ``mup_muon.py:~432``).
                new_muon_mu = jax.tree_util.tree_map(
                    lambda m, g, eb, is_m: jnp.where(
                        is_m, eb * m + (1.0 - eb) * g.astype(m.dtype), m
                    ),
                    opt_state.muon_mu, grad, eff_muon_beta_tree, is_muon,
                )

                if parent._nesterov:
                    ns_input_tree = jax.tree_util.tree_map(
                        lambda m, g, eb: eb * m + (1.0 - eb) * g.astype(m.dtype),
                        new_muon_mu, grad, eff_muon_beta_tree,
                    )
                else:
                    ns_input_tree = new_muon_mu

                # Newton-Schulz per leaf — only meaningful for ndim >= 2 leaves.
                # The is_muon mask guarantees that only HIDDEN_WEIGHT (2-D / 3-D)
                # leaves should reach the Muon path. tree_map traverses *every*
                # leaf, so for 1-D bias/norm leaves we short-circuit to zeros to
                # avoid an NS shape error; the result is masked off when we
                # build muon_step_tree below. The ndim check is on a static
                # shape (Python int) so it is resolved at trace time per leaf.
                def _maybe_ns(grad_leaf, eff_eps_leaf):
                    if grad_leaf.ndim < 2:
                        return jnp.zeros_like(grad_leaf)
                    return _muon_step_for_leaf(
                        grad_leaf.astype(jnp.float32),
                        eff_eps_leaf,
                        ns_coeffs_arr,
                        parent._ns_steps,
                    )

                orth_tree = jax.tree_util.tree_map(
                    _maybe_ns, ns_input_tree, eff_muon_eps_tree,
                )

                # Apply the per-leaf √(d_out/d_in) shape factor to the
                # orthogonalized result. This matches the validated reference
                # ``src/opt/mup_muon.py`` which applies the factor *internally*
                # via ``_compute_shape_scales`` (mup_muon.py:295-320), in
                # addition to the factor that is also baked into
                # ``mup_muon_lr_scales`` by ``MuonCompletedPParameterization``.
                # Net contribution to the orth update is (d_out/d_in)·√(m_B/m_D),
                # which is what the empirically-validated reference produces.
                orth_tree = jax.tree_util.tree_map(
                    lambda orth, ss: orth * ss, orth_tree, muon_shape_scales,
                )

                # Muon parameter step (decoupled weight decay added before LR scaling)
                # mup_muon.py uses: u = orth + wd * p ; final_step = -lr * u
                muon_step_tree = jax.tree_util.tree_map(
                    lambda orth, p, e_lr, e_wd: -e_lr * (orth + e_wd * p),
                    orth_tree, opt_state.params, eff_lr_tree, eff_wd_tree,
                )

                # ---- Adam side: standard moment EMAs + AdamW step ----
                new_adam_m = jax.tree_util.tree_map(
                    lambda m, g, eb1, is_m: jnp.where(
                        is_m, m, eb1 * m + (1.0 - eb1) * g.astype(m.dtype)
                    ),
                    opt_state.adam_m, grad, eff_adam_b1_tree, is_muon,
                )
                new_adam_v = jax.tree_util.tree_map(
                    lambda v, g, eb2, is_m: jnp.where(
                        is_m, v, eb2 * v + (1.0 - eb2) * (g.astype(v.dtype) ** 2)
                    ),
                    opt_state.adam_v, grad, eff_adam_b2_tree, is_muon,
                )

                m_hat = jax.tree_util.tree_map(
                    lambda m, eb1: m / (1.0 - eb1 ** t),
                    new_adam_m, eff_adam_b1_tree,
                )
                v_hat = jax.tree_util.tree_map(
                    lambda v, eb2: v / (1.0 - eb2 ** t),
                    new_adam_v, eff_adam_b2_tree,
                )

                adam_step_tree = jax.tree_util.tree_map(
                    lambda p, mh, vh, e_eps, e_wd, e_lr: (
                        -e_lr * (mh / (jnp.sqrt(vh) + e_eps) + e_wd * p)
                    ),
                    opt_state.params, m_hat, v_hat,
                    eff_adam_eps_tree, eff_wd_tree, eff_lr_tree,
                )

                # ---- Schedule ----
                schedule_mult = self._compute_schedule(
                    opt_state.iteration, opt_state.num_steps
                )

                # ---- Merge muon / adam steps based on is_muon mask ----
                next_params = jax.tree_util.tree_map(
                    lambda p, mu_step, ad_step, is_m: (
                        p + schedule_mult * jnp.where(is_m, mu_step, ad_step)
                    ),
                    opt_state.params, muon_step_tree, adam_step_tree, is_muon,
                )

                # ---- Merge model state ----
                # Use type-preserving merge so a Flax FrozenDict (or similar
                # immutable container) survives the round-trip — passing a
                # plain dict where the previous state was a FrozenDict would
                # break ``tree_utils.match_type`` later because Flax registers
                # FrozenDict as a distinct pytree node type.
                if model_state and opt_state.state is not None:
                    container_t = type(opt_state.state)
                    try:
                        merged_state = container_t({**opt_state.state, **model_state})
                    except TypeError:
                        # Fallback for containers that can't be constructed
                        # from a dict (rare); plain-dict merge.
                        merged_state = {**opt_state.state, **model_state}
                elif model_state:
                    merged_state = model_state
                else:
                    merged_state = opt_state.state

                next_opt_state = CompletedPMuonLOptState(
                    params=next_params,
                    state=merged_state,
                    muon_mu=new_muon_mu,
                    adam_m=new_adam_m,
                    adam_v=new_adam_v,
                    iteration=opt_state.iteration + 1,
                    num_steps=opt_state.num_steps,
                )

                return tree_utils.match_type(next_opt_state, opt_state)

        return _Opt(theta)
