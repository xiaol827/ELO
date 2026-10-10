# coding=utf-8
# Wrapper for the original Celo2 optax implementation (celo2_optax.py).
# This file bridges the original optax API into the scaling_l2o framework's
# lopt_base.LearnedOptimizer interface.
"""Celo2 optimizer wrapper for the scaling_l2o framework."""

from typing import Optional
import os
import pickle

import chex
import flax
import gin
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
from learned_optimization import tree_utils
import optax

from .celo2_optax import (
    Celo2Transformation,
    load_checkpoint,
    scale_by_celo2,
    vec_bias_correction,
)

PRNGKey = jnp.ndarray


@flax.struct.dataclass
class AdamState:
    """Optax state for Adam."""
    rms_rolling: chex.ArrayTree
    mom_rolling: chex.ArrayTree
    iteration: jnp.ndarray


class AdamTransformation:
    """Adam optimizer optax transformation.

    Ported verbatim from the official celo repo (celo/optimizers/celo2.py).
    bias_correction=False: step = m * rsqrt(v + eps) (eps inside the sqrt).
    bias_correction=True: bias-correct m and v first, same step formula.
    """

    def __init__(self, b1=0.9, b2=0.999, epsilon=1e-8, bias_correction=True):
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.bias_correction = bias_correction

    def accumulators_for_decays(self):
        mom_decay = jnp.asarray((self.b1,))
        rms_decay = jnp.asarray((self.b2,))
        return common.vec_rolling_mom(mom_decay), common.vec_rolling_rms(rms_decay)

    def _compute_step(self, m, rms):
        rsqrt = lax.rsqrt(rms + self.epsilon)
        norm_g = m * rsqrt
        return norm_g[..., 0]

    # def _compute_step(self, m, rms):
    #     denom = jnp.sqrt(rms) + self.epsilon
    #     norm_g = m / denom
    #     return norm_g[..., 0]

    def init(self, params: chex.ArrayTree) -> AdamState:
        mom_roll, rms_roll = self.accumulators_for_decays()
        return AdamState(
            rms_rolling=rms_roll.init(params),
            mom_rolling=mom_roll.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
        )

    def update(self, grads, state: AdamState, params=None) -> tuple:
        """Returns (step, new_state). Step is the raw Adam update direction."""
        iteration = optax.safe_increment(state.iteration)
        mom_roll, rms_roll = self.accumulators_for_decays()
        next_mom_rolling = mom_roll.update(state.mom_rolling, grads)
        next_rms_rolling = rms_roll.update(state.rms_rolling, grads)

        if self.bias_correction:
            m = vec_bias_correction(next_mom_rolling.m, (self.b1,), iteration)
            rms = vec_bias_correction(next_rms_rolling.rms, (self.b2,), iteration)
        else:
            m = next_mom_rolling.m
            rms = next_rms_rolling.rms

        step = jax.tree_util.tree_map(self._compute_step, m, rms)

        new_state = AdamState(
            mom_rolling=next_mom_rolling,
            rms_rolling=next_rms_rolling,
            iteration=iteration,
        )
        return step, new_state


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    epsilon: float = 1e-8,
    bias_correction: bool = True,
) -> optax.GradientTransformation:
    """Create an optax GradientTransformation for the official celo Adam."""
    transformation = AdamTransformation(
        b1=b1, b2=b2, epsilon=epsilon, bias_correction=bias_correction,
    )

    def init_fn(params):
        return transformation.init(params)

    def update_fn(updates, state, params=None, **kwargs):
        return transformation.update(updates, state, params)

    return optax.GradientTransformation(init_fn, update_fn)


@flax.struct.dataclass
class Celo2State:
    params: opt_base.Params
    state: opt_base.ModelState
    optax_state: optax.OptState
    iteration: jnp.ndarray
    num_steps: jnp.ndarray
    scheduled_lr: jnp.ndarray

@gin.configurable
class Celo2LOpt(lopt_base.LearnedOptimizer):
    """Wrapper around the original Celo2 optax implementation.

    Supports both meta-training (theta flows through opt_fn for gradient)
    and meta-testing (load pretrained weights from checkpoint).
    """

    def __init__(
        self,
        checkpoint_path="FILL_IN_LOCAL_CHECKPOINT_PATH",
        # LR schedule params (warmup_cosine_decay_schedule)
        init_lr=0.0,
        peak_lr=1e-3,
        warmup_steps=0,
        warmup_fraction=0.0,
        end_lr=1e-5,
        # Weight decay
        weight_decay=0.0,
        # Adam for 1D params
        adam_lr_mult=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        adam_bias_correction=True,
        adam_weight_decay=None,
        use_adamw_for_1d=True,
        # Celo2 model config (forwarded to scale_by_celo2)
        orthogonalize=True,
        # Optional gradient clipping (applied to raw grads before Celo2 preprocessing)
        clip_grad=False,
        clip_norm=1.0,
        **celo2_kwargs,
    ):
        self.checkpoint_path = checkpoint_path
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.warmup_fraction = warmup_fraction
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.adam_lr_mult = adam_lr_mult
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.adam_bias_correction = adam_bias_correction
        self.adam_weight_decay = adam_weight_decay if adam_weight_decay is not None else weight_decay
        self.use_adamw_for_1d = use_adamw_for_1d
        self.orthogonalize = orthogonalize
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm
        self.celo2_kwargs = celo2_kwargs

    def _has_checkpoint(self):
        """Check if a valid checkpoint path is configured."""
        return (
            self.checkpoint_path
            and self.checkpoint_path != "FILL_IN_LOCAL_CHECKPOINT_PATH"
            and os.path.isfile(self.checkpoint_path)
        )

    def _load_from_checkpoint(self, path=None):
        """Load Celo2 MLP params from a flax checkpoint."""
        p = path if path is not None else self.checkpoint_path
        return load_checkpoint(
            p, orthogonalize=self.orthogonalize, **self.celo2_kwargs
        )

    def _random_init(self, key):
        """Randomly initialize Celo2 MLP params (for meta-training from scratch)."""
        return Celo2Transformation(
            orthogonalize=self.orthogonalize, **self.celo2_kwargs
        ).init_meta_params(key)

    def load_meta_params(self, path):
        """Load meta-params from checkpoint file (for meta-testing / resume).

        Auto-detects format:
          - .pickle → framework meta-train checkpoint (pickle)
          - otherwise → original paper checkpoint (flax msgpack)
        """
        if path is not None and os.path.isfile(str(path)):
            path = str(path)
            if path.endswith('.pickle'):
                with open(path, "rb") as f:
                    return pickle.load(f)
            return self._load_from_checkpoint(path)
        return self._load_from_checkpoint()

    def init(self, key: PRNGKey) -> lopt_base.MetaParams:
        """Initialize meta-params. Loads from checkpoint if available, else random init."""
        if self._has_checkpoint():
            return self._load_from_checkpoint()
        return self._random_init(key)

    def opt_fn(
        self,
        theta: lopt_base.MetaParams,
        is_training: Optional[bool] = False,
    ) -> opt_base.Optimizer:
        parent = self
        # theta IS the Celo2 MLP params — gradient flows through here
        celo2_params = theta

        class _Opt(opt_base.Optimizer):

            def __init__(self):
                self._celo2_params = celo2_params

            @staticmethod
            def _make_lr_schedule(init_lr, peak_lr, warmup, decay_steps, end_lr):
                """Pure-JAX warmup + cosine decay schedule (tracer-safe)."""
                def schedule(step):
                    step = jnp.asarray(step, dtype=jnp.float32)
                    warmup_f = jnp.maximum(jnp.asarray(warmup, jnp.float32), 1.0)
                    decay_f = jnp.maximum(jnp.asarray(decay_steps, jnp.float32), 1.0)
                    # Warmup: linear from init_lr to peak_lr
                    warmup_lr = init_lr + (peak_lr - init_lr) * jnp.minimum(step / warmup_f, 1.0)
                    # Cosine decay: from peak_lr to end_lr
                    progress = jnp.clip((step - warmup_f) / jnp.maximum(decay_f - warmup_f, 1.0), 0.0, 1.0)
                    cosine_lr = end_lr + (peak_lr - end_lr) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
                    return jnp.where(step < warmup_f, warmup_lr, cosine_lr)
                return schedule

            def _build_optimizer(self, params, num_steps):
                """Build optax.multi_transform matching the paper's usage."""
                # warmup_fraction takes priority if > 0, otherwise use warmup_steps
                warmup = parent.warmup_steps
                if parent.warmup_fraction > 0:
                    warmup = jnp.float32(parent.warmup_fraction * num_steps)

                self._lr_schedule = self._make_lr_schedule(
                    parent.init_lr, parent.peak_lr, warmup, num_steps, parent.end_lr,
                )

                celo2_chain = optax.chain(
                    scale_by_celo2(
                        self._celo2_params,
                        orthogonalize=parent.orthogonalize,
                        **parent.celo2_kwargs,
                    ),
                    optax.add_decayed_weights(parent.weight_decay),
                    optax.scale_by_learning_rate(self._lr_schedule),
                )

                if parent.use_adamw_for_1d:
                    scaled_lr_schedule = lambda step: parent.adam_lr_mult * self._lr_schedule(step)
                    adam_opt = optax.chain(
                        scale_by_adam(
                            b1=parent.adam_beta1,
                            b2=parent.adam_beta2,
                            epsilon=parent.adam_epsilon,
                            bias_correction=parent.adam_bias_correction,
                        ),
                        optax.add_decayed_weights(parent.adam_weight_decay),
                        optax.scale_by_learning_rate(scaled_lr_schedule),
                    )

                    param_labels = jax.tree.map_with_path(
                        lambda path, val: (
                            'adam'
                            if val.ndim <= 1
                            or 'embed' in jax.tree_util.keystr(path)
                            else 'celo2'
                        ),
                        params,
                    )

                    optimizer = optax.multi_transform(
                        transforms={'celo2': celo2_chain, 'adam': adam_opt},
                        param_labels=param_labels,
                    )
                else:
                    optimizer = celo2_chain

                return optimizer

            def init(self, params, model_state=None, num_steps=None, key=None):
                if num_steps is None:
                    raise ValueError("Must specify num_steps for Celo2LOpt!")

                optimizer = self._build_optimizer(params, num_steps)
                optax_state = optimizer.init(params)

                return Celo2State(
                    params=params,
                    state=model_state,
                    optax_state=optax_state,
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                    num_steps=jnp.asarray(num_steps, dtype=jnp.int32),
                    scheduled_lr=jnp.asarray(self._lr_schedule(0), dtype=jnp.float32),
                )

            def update(self, opt_state, grad, loss=None,
                       model_state=None, is_valid=False, key=None):
                params = opt_state.params

                if parent.clip_grad:
                    clipping = optax.clip_by_global_norm(parent.clip_norm)
                    grad, _ = clipping.update(grad, None)

                # Rebuild optimizer (opt_fn may create a new _Opt each call)
                optimizer = self._build_optimizer(params, opt_state.num_steps)
                updates, new_optax_state = optimizer.update(
                    grad, opt_state.optax_state, params,
                )
                next_params = optax.apply_updates(params, updates)

                if model_state is not None and opt_state.state is not None:
                    merged_state = dict(opt_state.state)
                    merged_state.update(model_state)
                elif model_state is not None:
                    merged_state = model_state
                else:
                    merged_state = opt_state.state

                next_iteration = opt_state.iteration + 1
                current_lr = jnp.asarray(
                    self._lr_schedule(next_iteration), dtype=jnp.float32)

                next_opt_state = Celo2State(
                    params=next_params,
                    state=merged_state,
                    optax_state=new_optax_state,
                    iteration=next_iteration,
                    num_steps=opt_state.num_steps,
                    scheduled_lr=current_lr,
                )

                return tree_utils.match_type(next_opt_state, opt_state)

        return _Opt()
