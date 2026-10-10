# coding=utf-8
"""Standalone Muon optimizer for big_vision (meta-test baseline).

Routing matches `muon_expert`'s ndim==2 gate (NOT CELO2's batched ndim>=2
path, which orthogonalizes a 3D attention kernel (in, heads, head_dim) as
`in` independent (heads, head_dim) blocks and shape-scales on the sub-block
dims -> ~3-8x oversized updates that diverge ViT training as LR ramps up):

  * Genuine 2D matrices only (both dims > 1, path has no 'embed') -> Muon:
    (Nesterov) momentum -> Newton-Schulz orthogonalization -> shape scaling.
  * Everything else -- 1D (LN/bias), >=3D attention / 4D conv kernels,
    embeddings, and vector-like 2D params (a dim == 1, e.g. cls token) --
    -> AdamW (`adam_expert`).

Muon has no meta-parameters: `opt_fn(theta)` ignores theta. It plugs into
big_vision `train_lo` through the same lopt_base.LearnedOptimizer /
opt_base.Optimizer interface as Celo2LOpt, exposing `iteration` and
`scheduled_lr` on its state for the train loop's logging.
"""

from typing import Any, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as onp
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
import optax

# Reused numerical cores (src/ on PYTHONPATH). Import only — not registered in
# the learned_optimizers package __init__.
from learned_optimizers.celo2_optax import orthogonalize_via_newton_schulz
from learned_optimizers.elo_adfac_mlp_lopt import adam_expert

PRNGKey = jnp.ndarray


@flax.struct.dataclass
class MuonState:
  params: Any
  state: Any
  iteration: jnp.ndarray
  num_steps: jnp.ndarray
  scheduled_lr: jnp.ndarray
  m: Any  # momentum EMA (raw, not bias-corrected)
  v: Any  # second-moment EMA (raw), used by the 1D/embed AdamW path


class MuonLOpt(lopt_base.LearnedOptimizer):
  """Hand-designed Muon dressed as a (theta-free) LearnedOptimizer."""

  def __init__(
      self,
      # LR schedule (warmup -> cosine decay), same shape as Celo2LOpt.
      init_lr=0.0,
      peak_lr=1e-3,
      warmup_steps=0,
      warmup_fraction=0.05,
      end_lr=1e-5,
      # Weight decay (coupled, applied to the Muon matrix params).
      weight_decay=0.0,
      # Muon (>=2D params).
      muon_beta=0.95,
      ns_coeffs=(3.4445, -4.7750, 2.0315),
      ns_steps=5,
      ns_eps=1e-8,
      nesterov=True,
      # AdamW fallback for 1D / embedding params.
      adam_beta1=0.9,
      adam_beta2=0.95,
      adam_lr_mult=1.0,
      adam_weight_decay=None,
      use_adamw_for_1d=True,
      # Optional gradient clipping (raw grads, before momentum).
      clip_grad=True,
      clip_norm=1.0,
  ):
    self.init_lr = init_lr
    self.peak_lr = peak_lr
    self.warmup_steps = warmup_steps
    self.warmup_fraction = warmup_fraction
    self.end_lr = end_lr
    self.weight_decay = weight_decay
    self.muon_beta = muon_beta
    # Host (numpy) array, NOT a device array: orthogonalize_via_newton_schulz
    # needs .shape/.astype/indexing on it, but building a device array here
    # (outside jit) trips train_lo's host-to-device transfer guard. As a numpy
    # constant it is folded into the trace inside the jitted update instead.
    self.ns_coeffs = onp.asarray(ns_coeffs, dtype=onp.float32)
    self.ns_steps = ns_steps
    self.ns_eps = ns_eps
    self.nesterov = nesterov
    self.adam_beta1 = adam_beta1
    self.adam_beta2 = adam_beta2
    self.adam_lr_mult = adam_lr_mult
    self.adam_weight_decay = (
        adam_weight_decay if adam_weight_decay is not None else weight_decay)
    self.use_adamw_for_1d = use_adamw_for_1d
    self.clip_grad = clip_grad
    self.clip_norm = clip_norm

  def init(self, key: PRNGKey) -> lopt_base.MetaParams:
    # No meta-params. Return a dummy scalar so reshard/opt_fn never see an
    # empty pytree; opt_fn ignores it entirely.
    del key
    return {"_unused": jnp.zeros((1,), jnp.float32)}

  def opt_fn(
      self,
      theta: lopt_base.MetaParams,
      is_training: Optional[bool] = False,
  ) -> opt_base.Optimizer:
    parent = self
    del theta  # Muon is hand-designed; no meta-params flow through.

    def _is_muon_matrix(path, p):
      # Muon only for genuine 2D weight matrices (both dims > 1). Everything
      # else -> AdamW: 1D (LN/bias), >=3D (attention/conv kernels), embeddings,
      # biases (flax multi-head biases are 2D (heads, head_dim) but must NOT be
      # orthogonalized), and vector-like 2D params (a singleton dim, e.g. cls).
      keystr = jax.tree_util.keystr(path)
      if 'embed' in keystr or 'bias' in keystr:
        return False
      return p.ndim == 2 and p.shape[0] > 1 and p.shape[1] > 1

    class _Opt(opt_base.Optimizer):

      @staticmethod
      def _make_lr_schedule(init_lr, peak_lr, warmup, decay_steps, end_lr):
        """Pure-JAX warmup + cosine decay (matches Celo2LOpt)."""
        def schedule(step):
          step = jnp.asarray(step, dtype=jnp.float32)
          warmup_f = jnp.maximum(jnp.asarray(warmup, jnp.float32), 1.0)
          decay_f = jnp.maximum(jnp.asarray(decay_steps, jnp.float32), 1.0)
          warmup_lr = init_lr + (peak_lr - init_lr) * jnp.minimum(
              step / warmup_f, 1.0)
          progress = jnp.clip(
              (step - warmup_f) / jnp.maximum(decay_f - warmup_f, 1.0), 0.0, 1.0)
          cosine_lr = end_lr + (peak_lr - end_lr) * 0.5 * (
              1.0 + jnp.cos(jnp.pi * progress))
          return jnp.where(step < warmup_f, warmup_lr, cosine_lr)
        return schedule

      def _compute_lr(self, step, num_steps):
        warmup = parent.warmup_steps
        if parent.warmup_fraction > 0:
          warmup = jnp.float32(parent.warmup_fraction * num_steps)
        return self._make_lr_schedule(
            parent.init_lr, parent.peak_lr, warmup, num_steps, parent.end_lr)(
                step)

      def init(self, params, model_state=None, num_steps=None, key=None):
        if num_steps is None:
          raise ValueError("Must specify num_steps for MuonLOpt!")
        zeros = lambda p: jnp.zeros_like(p)
        return MuonState(
            params=params,
            state=model_state,
            iteration=jnp.asarray(0, dtype=jnp.int32),
            num_steps=jnp.asarray(num_steps, dtype=jnp.int32),
            scheduled_lr=jnp.asarray(
                self._compute_lr(0, num_steps), dtype=jnp.float32),
            m=jax.tree_util.tree_map(zeros, params),
            v=jax.tree_util.tree_map(zeros, params),
        )

      def update(self, opt_state, grad, loss=None,
                 model_state=None, is_valid=False, key=None):
        params = opt_state.params

        if parent.clip_grad:
          grad, _ = optax.clip_by_global_norm(parent.clip_norm).update(
              grad, None)

        t = opt_state.iteration + 1
        lr = self._compute_lr(t, opt_state.num_steps)
        mb = parent.muon_beta
        ab1 = parent.adam_beta1
        ab2 = parent.adam_beta2

        # Raw (non-bias-corrected) EMAs; bias correction done downstream.
        # First moment uses the per-leaf-appropriate decay: muon_beta for the
        # 2D matrices that go to Muon, adam_beta1 for the AdamW-routed rest, so
        # each path's bias correction downstream is consistent with how its
        # buffer was accumulated.
        def _accum_m(path, m, p, g):
          beta = mb if _is_muon_matrix(path, p) else ab1
          return beta * m + (1.0 - beta) * g
        new_m = jax.tree_util.tree_map_with_path(
            _accum_m, opt_state.m, params, grad)
        new_v = jax.tree_util.tree_map(
            lambda v, g: ab2 * v + (1.0 - ab2) * g * g, opt_state.v, grad)

        def _apply(path, p, m_leaf, v_leaf, g):
          if not _is_muon_matrix(path, p):
            return adam_expert(
                p, m_leaf, v_leaf, parent.adam_beta1, parent.adam_beta2, t,
                parent.adam_lr_mult * lr, parent.adam_weight_decay)

          # ---- Muon: (Nesterov) momentum -> NS orthogonalization -> scale ----
          if parent.nesterov:
            # Bias-corrected Nesterov, matching muon_expert (elo_adfac:159-161).
            bc_m = m_leaf / (1.0 - mb ** (t + 1))
            bc_g = g / (1.0 - mb ** t)
            mu_hat = mb * bc_m + (1.0 - mb) * bc_g
          else:
            mu_hat = m_leaf / (1.0 - mb ** t)

          ortho = orthogonalize_via_newton_schulz(
              mu_hat, parent.ns_coeffs, parent.ns_steps, parent.ns_eps)
          # Muon shape scaling on the last two dims (reduction=-2, output=-1).
          n = jnp.float32(ortho.shape[-2])
          k = jnp.float32(ortho.shape[-1])
          update = ortho * jnp.sqrt(jnp.maximum(1.0, k / n))
          return p - lr * (update + parent.weight_decay * p)

        new_params = jax.tree_util.tree_map_with_path(
            _apply, params, new_m, new_v, grad)

        if model_state is not None and opt_state.state is not None:
          merged_state = dict(opt_state.state)
          merged_state.update(model_state)
        elif model_state is not None:
          merged_state = model_state
        else:
          merged_state = opt_state.state

        next_opt_state = MuonState(
            params=new_params,
            state=merged_state,
            iteration=t,
            num_steps=opt_state.num_steps,
            scheduled_lr=jnp.asarray(lr, dtype=jnp.float32),
            m=new_m,
            v=new_v,
        )
        return tree_utils.match_type(next_opt_state, opt_state)

    return _Opt()
