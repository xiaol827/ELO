# coding=utf-8
"""ELO + CELO2 fusion: CELO2 MLP backbone with ELO expert mechanisms.

Accumulator layout mirrors the official celo2 1D/2D routing:
  - 2D+ params: multi-channel mom/rms/fac accumulators shared between the
    CELO2 MLP (input features) and the expert optimizer.
  - 1D/embed params: official celo AdamTransformation (single-channel
    mom + rms); LO and expert share this exact update, so no blending.
No optax chain — manages accumulators directly.
"""

import functools
import os
import pickle
from typing import Any, Optional
import chex

import flax
from jax import lax
import gin
import jax
import jax.numpy as jnp
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
import optax

from .celo2_lopt import AdamState
from .celo2_optax import (
    Celo2Transformation,
    load_checkpoint,
    vec_bias_correction,
)
from .elo_adfac_mlp_lopt import adam_expert, sgdm_expert, muon_expert, cosine_scheduler

PRNGKey = jnp.ndarray


class AdamTransformation:
    """Adam optimizer optax transformation (ELO_Celo2LOpt-local variant).

    NOTE: this does NOT match ``celo2_lopt.AdamTransformation`` (official celo).
    Official celo puts eps INSIDE the sqrt: step = m * rsqrt(v + eps).
    The active formula here is the optax.adam form, step = m / (sqrt(v) + eps),
    with the official form kept commented out below. The two differ most when
    v is tiny (the first steps of an inner problem): rsqrt(v + 1e-8) caps the
    step at 1e4*m, m / (sqrt(v) + 1e-8) caps it at 1e8*m.
    bias_correction=True: bias-correct m and v first, same step formula.

    This is a standalone copy on purpose: edits to celo2_lopt's Adam do NOT
    propagate here, so keep the two in sync manually when that is intended.
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
        denom = jnp.sqrt(rms) + self.epsilon
        norm_g = m / denom
        return norm_g[..., 0]

    # def _compute_step(self, m, rms):
    #     rsqrt = lax.rsqrt(rms + self.epsilon)
    #     norm_g = m * rsqrt
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


def _is_1d_leaf(path, val):
  return val.ndim <= 1 or 'embed' in jax.tree_util.keystr(path)


def _mask_1d(params, use_1d=True):
  if not use_1d:
    return jax.tree_util.tree_map(lambda _: False, params)
  return jax.tree_util.tree_map_with_path(_is_1d_leaf, params)


def _split_1d_2d(tree, params, use_1d=True):
  """Split pytree into (1D, 2D+) trees, non-group leaves replaced by None."""
  flat, treedef = jax.tree_util.tree_flatten(tree)
  flat_mask = jax.tree_util.tree_leaves(_mask_1d(params, use_1d))
  flat_1d = [x if m else None for x, m in zip(flat, flat_mask)]
  flat_2d = [x if not m else None for x, m in zip(flat, flat_mask)]
  return (jax.tree_util.tree_unflatten(treedef, flat_1d),
          jax.tree_util.tree_unflatten(treedef, flat_2d))


def _merge_1d_2d(tree_1d, tree_2d, params, use_1d=True):
  """Merge 1D and 2D pytrees back using the same mask."""
  is_none = lambda x: x is None
  flat_1d = jax.tree_util.tree_flatten(tree_1d, is_leaf=is_none)[0]
  flat_2d = jax.tree_util.tree_flatten(tree_2d, is_leaf=is_none)[0]
  flat_mask = jax.tree_util.tree_leaves(_mask_1d(params, use_1d))
  merged = [x1 if m else x2 for x1, x2, m in zip(flat_1d, flat_2d, flat_mask)]
  assert all(x is not None for x in merged), "Merged tree should not contain None"
  treedef = jax.tree_util.tree_flatten(params)[1]
  return jax.tree_util.tree_unflatten(treedef, merged)


@flax.struct.dataclass
class ELOCelo2State:
  params: Any
  state: Any
  iteration: jnp.ndarray
  num_steps: jnp.ndarray
  scheduled_lr: jnp.ndarray
  expert_weight: jnp.ndarray
  expert_lr: jnp.ndarray
  # 2D+ accumulators (CELO2 MLP + expert)
  mom_rolling: Any
  rms_rolling: Any
  fac_rolling: Any
  # 1D accumulators (Adam + expert)
  mom_rolling_1d: Any
  rms_rolling_1d: Any


@gin.configurable
class ELO_Celo2LOpt(lopt_base.LearnedOptimizer):
  """CELO2 backbone with ELO expert-guided meta-training.

  Accumulator sharing:
    - 2D+ params: expert reads decay channels from the CELO2 MLP accumulators.
    - 1D/embed params: single shared official-celo Adam update
      (beta1 = momentum_decays[exp_adam_mom_idx],
      beta2 = rms_decays[exp_adam_rms_idx]); no expert branch, no blending.
  """

  def __init__(
      self,
      # CELO2 backbone
      orthogonalize=True,
      ff_hidden_size=8,
      ff_hidden_layers=2,
      initial_momentum_decays=(0.9, 0.99, 0.999),
      initial_rms_decays=(0.95,),
      initial_adafactor_decays=(0.9, 0.99, 0.999),
      # LR schedule
      init_lr=0.0,
      peak_lr=1e-3,
      warmup_steps=0,
      warmup_fraction=0.0,
      end_lr=1e-5,
      # Weight decay
      weight_decay=0.0,
      # Adam for 1D params (shares the dedicated 1D accumulator with the expert)
      adam_lr_mult=1.0,
      adam_epsilon=1e-8,
      adam_bias_correction=False,
      adam_weight_decay=None,
      use_adamw_for_1d=True,
      # Gradient clipping
      clip_grad=False,
      clip_norm=1.0,
      # elementwise grad clip magnitude (official celo2: 1000.0)
      clip_value=1000.0,
      # Expert params (uses momentum[0]/rms[-1] from shared accumulators)
      expert_lr_max=0.001,
      expert_lr_min=1e-5,
      expert_lr_decay_steps=10000,
      expert_weight_decay=0.0,
      expert_optim="adamw",
      muon_expert_adamlr_scaler=0.3,
      # Expert accumulator channel indices (which decay channel each expert reads)
      exp_sgdm_mom_idx=0,
      exp_adam_mom_idx=0,
      exp_muon_mom_idx=1,
      exp_adam_rms_idx=-1,
      # Mode
      meta_train=True,
      expert_only=False,
      # Extra CELO2 kwargs
      **celo2_kwargs,
  ):
    super().__init__()
    self.orthogonalize = orthogonalize
    self.ff_hidden_size = ff_hidden_size
    self.ff_hidden_layers = ff_hidden_layers
    self.initial_momentum_decays = initial_momentum_decays
    self.initial_rms_decays = initial_rms_decays
    self.initial_adafactor_decays = initial_adafactor_decays
    self.init_lr = init_lr
    self.peak_lr = peak_lr
    self.warmup_steps = warmup_steps
    self.warmup_fraction = warmup_fraction
    self.end_lr = end_lr
    self.weight_decay = weight_decay
    self.adam_lr_mult = adam_lr_mult
    self.adam_epsilon = adam_epsilon
    self.adam_bias_correction = adam_bias_correction
    self.adam_weight_decay = adam_weight_decay if adam_weight_decay is not None else weight_decay
    self.use_adamw_for_1d = use_adamw_for_1d
    self.clip_grad = clip_grad
    self.clip_norm = clip_norm
    self.clip_value = clip_value
    self.expert_lr_max = expert_lr_max
    self.expert_lr_min = expert_lr_min
    self.expert_lr_decay_steps = expert_lr_decay_steps
    self.expert_weight_decay = expert_weight_decay
    self.expert_optim = expert_optim
    self.muon_expert_adamlr_scaler = muon_expert_adamlr_scaler
    self.exp_sgdm_mom_idx = exp_sgdm_mom_idx
    self.exp_adam_mom_idx = exp_adam_mom_idx
    self.exp_muon_mom_idx = exp_muon_mom_idx
    self.exp_adam_rms_idx = exp_adam_rms_idx
    self.meta_train = meta_train
    self.expert_only = expert_only
    self.celo2_kwargs = celo2_kwargs

    assert expert_optim in ("adamw", "sgdm", "muon"), \
        f"expert_optim must be one of 'adamw', 'sgdm', 'muon', got '{expert_optim}'"

  def _celo2_config(self):
    return dict(
        orthogonalize=self.orthogonalize,
        ff_hidden_size=self.ff_hidden_size,
        ff_hidden_layers=self.ff_hidden_layers,
        initial_momentum_decays=self.initial_momentum_decays,
        initial_rms_decays=self.initial_rms_decays,
        initial_adafactor_decays=self.initial_adafactor_decays,
        **self.celo2_kwargs,
    )

  def init(self, key: PRNGKey) -> lopt_base.MetaParams:
    return Celo2Transformation(**self._celo2_config()).init_meta_params(key)

  def load_meta_params(self, path):
    """Load meta-params from checkpoint (pickle or flax format)."""
    if path is not None and os.path.isfile(str(path)):
      path = str(path)
      if path.endswith('.pickle'):
        with open(path, "rb") as f:
          return pickle.load(f)
      return load_checkpoint(path, **self._celo2_config())
    return load_checkpoint(path, **self._celo2_config())

  def opt_fn(
      self,
      theta: lopt_base.MetaParams,
      is_training: Optional[bool] = False,
  ) -> opt_base.Optimizer:
    parent = self
    celo2_params = theta

    class _Opt(opt_base.Optimizer):

      def __init__(self):
        self._celo2_params = celo2_params
        self._transformation = Celo2Transformation(**parent._celo2_config())
        self.meta_train = parent.meta_train

      @staticmethod
      def _make_lr_schedule(init_lr, peak_lr, warmup, decay_steps, end_lr):
        def schedule(step):
          step = jnp.asarray(step, dtype=jnp.float32)
          warmup_f = jnp.maximum(jnp.asarray(warmup, jnp.float32), 1.0)
          decay_f = jnp.maximum(jnp.asarray(decay_steps, jnp.float32), 1.0)
          warmup_lr = init_lr + (peak_lr - init_lr) * jnp.minimum(step / warmup_f, 1.0)
          progress = jnp.clip(
              (step - warmup_f) / jnp.maximum(decay_f - warmup_f, 1.0), 0.0, 1.0)
          cosine_lr = end_lr + (peak_lr - end_lr) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
          return jnp.where(step < warmup_f, warmup_lr, cosine_lr)
        return schedule

      def _compute_lr(self, step, num_steps):
        warmup = parent.warmup_steps
        if parent.warmup_fraction > 0:
          warmup = jnp.float32(parent.warmup_fraction * num_steps)
        return self._make_lr_schedule(
            parent.init_lr, parent.peak_lr, warmup, num_steps, parent.end_lr)(step)

      @staticmethod
      def _adam_transformation_1d():
        """Official celo AdamTransformation for the 1D/embed group (shared
        between LO and expert — both apply the exact same update)."""
        return AdamTransformation(
            b1=parent.initial_momentum_decays[parent.exp_adam_mom_idx],
            b2=parent.initial_rms_decays[parent.exp_adam_rms_idx],
            epsilon=parent.adam_epsilon,
            bias_correction=parent.adam_bias_correction,
        )

      def init(self, params, model_state=None, num_steps=None, key=None):
        if num_steps is None:
          raise ValueError("Must specify num_steps for ELO_Celo2LOpt!")

        p_1d, p_2d = _split_1d_2d(params, params, parent.use_adamw_for_1d)
        mom_roll, rms_roll, fac_roll = self._transformation.accumulators_for_decays()
        mom_roll_1d, rms_roll_1d = self._adam_transformation_1d().accumulators_for_decays()

        return ELOCelo2State(
            params=params,
            state=model_state,
            iteration=jnp.asarray(0, dtype=jnp.int32),
            num_steps=jnp.asarray(num_steps, dtype=jnp.int32),
            scheduled_lr=jnp.asarray(
                self._compute_lr(0, num_steps), dtype=jnp.float32),
            expert_weight=jnp.asarray(1.0, dtype=jnp.float32),
            expert_lr=jnp.asarray(parent.expert_lr_max, dtype=jnp.float32),
            mom_rolling=mom_roll.init(p_2d),
            rms_rolling=rms_roll.init(p_2d),
            fac_rolling=fac_roll.init(p_2d),
            mom_rolling_1d=mom_roll_1d.init(p_1d),
            rms_rolling_1d=rms_roll_1d.init(p_1d),
        )

      def update(self, opt_state, grad, loss=None,
                 model_state=None, is_valid=False, key=None):
        params = opt_state.params

        # elementwise clip (official Celo2Transformation uses 1000.0;
        # applied before the optional global-norm clip so it still bites)
        grad = jax.tree_util.tree_map(
            lambda x: jnp.clip(x, -parent.clip_value, parent.clip_value), grad)

        if parent.clip_grad:
          clipping = optax.clip_by_global_norm(parent.clip_norm)
          grad, _ = clipping.update(grad, None)
        # any NaN leaf poisons the global norm above, zeroing the whole step
        grad = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grad)

        use_1d = parent.use_adamw_for_1d
        g_1d, g_2d = _split_1d_2d(grad, params, use_1d)
        p_1d, p_2d = _split_1d_2d(params, params, use_1d)

        # --- Update 2D accumulators (CELO2 MLP + expert) ---
        mom_roll, rms_roll, fac_roll = self._transformation.accumulators_for_decays()
        next_mom = mom_roll.update(opt_state.mom_rolling, g_2d)
        next_rms = rms_roll.update(opt_state.rms_rolling, g_2d)
        next_fac, fac_g = fac_roll.update(opt_state.fac_rolling, g_2d)

        m = next_mom.m
        rms_val = next_rms.rms
        v_col = next_fac.v_col
        v_row = next_fac.v_row

        # --- 1D/embed group: official celo AdamTransformation ---
        # Single shared update — the LO uses it directly and the expert does
        # not touch this group (no blending on 1D/embed params).
        adam_1d = self._adam_transformation_1d()
        step_1d, next_adam_state_1d = adam_1d.update(
            g_1d,
            AdamState(
                rms_rolling=opt_state.rms_rolling_1d,
                mom_rolling=opt_state.mom_rolling_1d,
                iteration=opt_state.iteration,
            ))

        next_iteration = opt_state.iteration + 1

        # --- CELO2 MLP forward pass (2D+ params only) ---
        # Celo2Transformation.update optionally bias-corrects the MLP input
        # features (m/rms/v_col/v_row; fac_g stays raw). Only the features are
        # corrected — the expert reads the raw accumulators and applies its
        # own bias correction internally.
        if self._transformation.bias_correction:
          m_feat = vec_bias_correction(
              m, parent.initial_momentum_decays, next_iteration)
          rms_feat = vec_bias_correction(
              rms_val, parent.initial_rms_decays, next_iteration)
          v_col_feat = vec_bias_correction(
              v_col, parent.initial_adafactor_decays, next_iteration)
          v_row_feat = vec_bias_correction(
              v_row, parent.initial_adafactor_decays, next_iteration)
        else:
          m_feat, rms_feat = m, rms_val
          v_col_feat, v_row_feat = v_col, v_row

        apply_fn = functools.partial(
            self._transformation.ff_mod.apply,
            self._celo2_params["ff_mod_stack"])
        step_celo2 = jax.tree_util.tree_map(
            apply_fn, p_2d, g_2d, m_feat, rms_feat, fac_g, v_col_feat, v_row_feat)

        # step index must match what init() stored (_compute_lr(0, ...)),
        # otherwise the first update already runs one step into the warmup.
        lr = self._compute_lr(opt_state.iteration, opt_state.num_steps)

        # --- Apply updates: 2D+ → CELO2 step, 1D → Adam ---
        exp_sgdm_mom_idx = parent.exp_sgdm_mom_idx
        exp_adam_mom_idx = parent.exp_adam_mom_idx
        exp_muon_mom_idx = parent.exp_muon_mom_idx
        exp_adam_rms_idx = parent.exp_adam_rms_idx

        next_lo_2d = jax.tree_util.tree_map(
            lambda p, s: p - lr * (s + parent.weight_decay * p),
            p_2d, step_celo2)

        # when use_adamw_for_1d=False the 1D mask is empty and all params
        # flow through the CELO2 branch above; next_lo_1d is then all-None
        adam_lr = parent.adam_lr_mult * lr

        next_lo_1d = jax.tree_util.tree_map(
            lambda p, s: p - adam_lr * (s + parent.adam_weight_decay * p),
            p_1d, step_1d)

        next_params_lo = _merge_1d_2d(next_lo_1d, next_lo_2d, params, use_1d)

        # --- Expert LR (cosine decay) ---
        current_expert_lr = cosine_scheduler(
            opt_state.iteration, parent.expert_lr_decay_steps,
            parent.expert_lr_max, parent.expert_lr_min)

        # --- Model state merging ---
        if model_state is not None and opt_state.state is not None:
          merged_state = dict(opt_state.state)
          merged_state.update(model_state)
        elif model_state is not None:
          merged_state = model_state
        else:
          merged_state = opt_state.state

        current_lr = jnp.asarray(lr, dtype=jnp.float32)

        if self.meta_train or parent.expert_only:
          # --- Expert trajectory: 2D (CELO2) group only ---
          # The 1D/embed group has no expert branch: LO and expert would both
          # apply the exact same shared Adam update, so those params take
          # next_lo_1d directly (no blending).
          if parent.expert_optim == "adamw":
            expert_2d = jax.tree_util.tree_map(
                lambda p, ml, rl: adam_expert(
                    p, ml, rl,
                    parent.initial_momentum_decays[exp_adam_mom_idx],
                    parent.initial_rms_decays[exp_adam_rms_idx],
                    next_iteration,
                    current_expert_lr,
                    parent.expert_weight_decay),
                p_2d,
                jax.tree_util.tree_map(lambda x: x[..., exp_adam_mom_idx], m),
                jax.tree_util.tree_map(lambda x: x[..., exp_adam_rms_idx], rms_val))
          elif parent.expert_optim == "sgdm":
            expert_2d = jax.tree_util.tree_map(
                lambda p, ml: sgdm_expert(
                    p, ml, current_expert_lr, parent.expert_weight_decay),
                p_2d,
                jax.tree_util.tree_map(lambda x: x[..., exp_sgdm_mom_idx], m))
          elif parent.expert_optim == "muon":
            # The 2D group excludes embed tables (routed to Adam by the mask),
            # so everything here goes down the muon path; muon_expert's
            # internal non-2D fallback still covers e.g. 3D+ conv kernels.
            expert_2d = jax.tree_util.tree_map(
                lambda p, m_muon, m_adam, rl, g: muon_expert(
                    p, m_muon, m_adam, rl, g,
                    next_iteration,
                    current_expert_lr,
                    parent.expert_weight_decay,
                    beta=parent.initial_momentum_decays[exp_muon_mom_idx],
                    adam_beta1=parent.initial_momentum_decays[exp_adam_mom_idx],
                    adam_beta2=parent.initial_rms_decays[exp_adam_rms_idx],
                    ns_steps=5,
                    adamlr_scaler=parent.muon_expert_adamlr_scaler),
                p_2d,
                jax.tree_util.tree_map(lambda x: x[..., exp_muon_mom_idx], m),
                jax.tree_util.tree_map(lambda x: x[..., exp_adam_mom_idx], m),
                jax.tree_util.tree_map(lambda x: x[..., exp_adam_rms_idx], rms_val),
                g_2d)

        if self.meta_train:
          # Blend on the 2D group only; 1D/embed takes the shared Adam update
          blend_2d = jax.tree_util.tree_map(
              lambda p1, p2: opt_state.expert_weight * p1 + (1.0 - opt_state.expert_weight) * p2,
              expert_2d, next_lo_2d)
          next_params = _merge_1d_2d(next_lo_1d, blend_2d, params, use_1d)

          # IMT losses: 2D (CELO2) group only — None leaves of the 2D trees
          # are skipped by tree_leaves automatically
          update_expert = jax.tree_util.tree_map(lambda n, c: n - c, expert_2d, p_2d)
          update_lo = jax.tree_util.tree_map(lambda n, c: n - c, next_lo_2d, p_2d)
          flat_expert = jnp.concatenate(
              [x.ravel() for x in jax.tree_util.tree_leaves(update_expert)])
          flat_lo = jnp.concatenate(
              [x.ravel() for x in jax.tree_util.tree_leaves(update_lo)])

          dot = jnp.sum(flat_expert * flat_lo)
          norm_expert = jnp.linalg.norm(flat_expert)
          norm_lo = jnp.linalg.norm(flat_lo)
          cosine_loss = 1.0 - dot / (norm_expert * norm_lo + 1e-30)
          # Pure scale, decoupled from direction: the per-coordinate form
          # mean(||e_i| - |l_i||) also scores the coordinate pattern, which is
          # part of the direction cosine_loss already measures.
          magnitude_loss = jnp.abs(norm_lo - norm_expert)
          # Plain MSE between the two updates -- the coupled baseline the
          # (cosine, magnitude) pair is meant to improve on. Always computed so
          # both objectives are logged side by side; which one actually drives
          # the meta-gradient is picked by the estimator's imt_loss_type.
          mse_loss = jnp.mean(jnp.square(flat_expert - flat_lo))

          next_opt_state = ELOCelo2State(
              params=next_params,
              state=merged_state,
              iteration=next_iteration,
              num_steps=opt_state.num_steps,
              scheduled_lr=current_lr,
              expert_weight=opt_state.expert_weight,
              expert_lr=current_expert_lr,
              mom_rolling=next_mom,
              rms_rolling=next_rms,
              fac_rolling=next_fac,
              mom_rolling_1d=next_adam_state_1d.mom_rolling,
              rms_rolling_1d=next_adam_state_1d.rms_rolling,
          )
          return (tree_utils.match_type(next_opt_state, opt_state),
                  cosine_loss, magnitude_loss, mse_loss)
        else:
          # expert_only: 2D takes the expert trajectory; 1D/embed has no expert
          # branch (LO and expert share the same Adam update) so it keeps next_lo_1d
          next_params_test = (
              _merge_1d_2d(next_lo_1d, expert_2d, params, use_1d)
              if parent.expert_only else next_params_lo)
          next_opt_state = ELOCelo2State(
              params=next_params_test,
              state=merged_state,
              iteration=next_iteration,
              num_steps=opt_state.num_steps,
              scheduled_lr=current_lr,
              expert_weight=opt_state.expert_weight,
              expert_lr=current_expert_lr,
              mom_rolling=next_mom,
              rms_rolling=next_rms,
              fac_rolling=next_fac,
              mom_rolling_1d=next_adam_state_1d.mom_rolling,
              rms_rolling_1d=next_adam_state_1d.rms_rolling,
          )
          return tree_utils.match_type(next_opt_state, opt_state)

    return _Opt()
