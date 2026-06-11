# coding=utf-8
"""ELO + CELO2 fusion: CELO2 MLP backbone with ELO expert mechanisms.

Shared accumulators between CELO2 MLP and expert optimizer to minimize
GPU memory overhead. No optax chain — manages accumulators directly.
"""

import functools
import os
import pickle
from typing import Any, Optional

import flax
import gin
import jax
import jax.numpy as jnp
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
import optax

from .celo2_optax import Celo2Transformation, load_checkpoint
from .elo_adfac_mlp_lopt import adam_expert, sgdm_expert, muon_expert, cosine_scheduler

PRNGKey = jnp.ndarray


@flax.struct.dataclass
class ELOCelo2State:
  params: Any
  state: Any
  iteration: jnp.ndarray
  num_steps: jnp.ndarray
  scheduled_lr: jnp.ndarray
  expert_weight: jnp.ndarray
  expert_lr: jnp.ndarray
  # Shared accumulators (CELO2 MLP + expert)
  mom_rolling: Any
  rms_rolling: Any
  fac_rolling: Any


@gin.configurable
class ELO_Celo2LOpt(lopt_base.LearnedOptimizer):
  """CELO2 backbone with ELO expert-guided meta-training.

  Uses shared accumulators between CELO2 MLP and expert optimizer.
  momentum[0] and rms[-1] are used by expert and 1D Adam;
  all decays are used by the CELO2 MLP as input features.
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
      # Adam for 1D params (uses momentum[0] and rms[-1] from shared accumulators)
      adam_lr_mult=1.0,
      adam_weight_decay=None,
      use_adamw_for_1d=True,
      # Gradient clipping
      clip_grad=False,
      clip_norm=1.0,
      # Expert params (uses momentum[0]/rms[-1] from shared accumulators)
      expert_lr_max=0.001,
      expert_lr_min=1e-5,
      expert_lr_decay_steps=10000,
      expert_weight_decay=0.0,
      expert_optim="adamw",
      muon_expert_adamlr_scaler=0.3,
      # Mode
      meta_train=True,
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
    self.adam_weight_decay = adam_weight_decay if adam_weight_decay is not None else weight_decay
    self.use_adamw_for_1d = use_adamw_for_1d
    self.clip_grad = clip_grad
    self.clip_norm = clip_norm
    self.expert_lr_max = expert_lr_max
    self.expert_lr_min = expert_lr_min
    self.expert_lr_decay_steps = expert_lr_decay_steps
    self.expert_weight_decay = expert_weight_decay
    self.expert_optim = expert_optim
    self.muon_expert_adamlr_scaler = muon_expert_adamlr_scaler
    self.meta_train = meta_train
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

      def init(self, params, model_state=None, num_steps=None, key=None):
        if num_steps is None:
          raise ValueError("Must specify num_steps for ELO_Celo2LOpt!")

        mom_roll, rms_roll, fac_roll = self._transformation.accumulators_for_decays()

        return ELOCelo2State(
            params=params,
            state=model_state,
            iteration=jnp.asarray(0, dtype=jnp.int32),
            num_steps=jnp.asarray(num_steps, dtype=jnp.int32),
            scheduled_lr=jnp.asarray(
                self._compute_lr(0, num_steps), dtype=jnp.float32),
            expert_weight=jnp.asarray(1.0, dtype=jnp.float32),
            expert_lr=jnp.asarray(parent.expert_lr_max, dtype=jnp.float32),
            mom_rolling=mom_roll.init(params),
            rms_rolling=rms_roll.init(params),
            fac_rolling=fac_roll.init(params),
        )

      def update(self, opt_state, grad, loss=None,
                 model_state=None, is_valid=False, key=None):
        params = opt_state.params

        if parent.clip_grad:
          clipping = optax.clip_by_global_norm(parent.clip_norm)
          grad, _ = clipping.update(grad, None)
        grad = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grad)

        # --- Update shared accumulators ---
        mom_roll, rms_roll, fac_roll = self._transformation.accumulators_for_decays()
        next_mom = mom_roll.update(opt_state.mom_rolling, grad)
        next_rms = rms_roll.update(opt_state.rms_rolling, grad)
        next_fac, fac_g = fac_roll.update(opt_state.fac_rolling, grad)

        m = next_mom.m
        rms_val = next_rms.rms
        v_col = next_fac.v_col
        v_row = next_fac.v_row

        # --- CELO2 MLP forward pass (all params) ---
        apply_fn = functools.partial(
            self._transformation.ff_mod.apply,
            self._celo2_params["ff_mod_stack"])
        step_celo2 = jax.tree_util.tree_map(
            apply_fn, params, grad, m, rms_val, fac_g, v_col, v_row)

        next_iteration = opt_state.iteration + 1
        lr = self._compute_lr(next_iteration, opt_state.num_steps)

        # --- Apply updates: 2D+ → CELO2 step, 1D → AdamW ---
        if parent.use_adamw_for_1d:
          adam_lr = parent.adam_lr_mult * lr
          mom_beta1 = jnp.asarray(parent.initial_momentum_decays[0])
          rms_beta2 = jnp.asarray(parent.initial_rms_decays[-1])

          def _apply_lo(path, p, step_c, m_leaf, rms_leaf):
            is_1d = p.ndim <= 1 or 'embed' in jax.tree_util.keystr(path)
            if is_1d:
              m_bc = m_leaf[..., 0] / (1.0 - mom_beta1 ** next_iteration)
              v_bc = rms_leaf[..., -1] / (1.0 - rms_beta2 ** next_iteration)
              adam_step = m_bc / (jnp.sqrt(v_bc) + 1e-8)
              return p - adam_lr * (adam_step + parent.adam_weight_decay * p)
            else:
              return p - lr * (step_c + parent.weight_decay * p)

          next_params_lo = jax.tree.map_with_path(
              _apply_lo, params, step_celo2, m, rms_val)
        else:
          next_params_lo = jax.tree_util.tree_map(
              lambda p, s: p - lr * (s + parent.weight_decay * p),
              params, step_celo2)

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

        if self.meta_train:
          # --- Expert trajectory (shared accumulators: mom[0], rms[-1]) ---
          rms_beta2_val = parent.initial_rms_decays[-1]

          if parent.expert_optim == "adamw":
            next_params_expert = jax.tree_util.tree_map(
                lambda p, ml, rl: adam_expert(
                    p, ml, rl,
                    parent.initial_momentum_decays[0],
                    rms_beta2_val,
                    next_iteration,
                    current_expert_lr,
                    parent.expert_weight_decay),
                params,
                jax.tree_util.tree_map(lambda x: x[..., 0], m),
                jax.tree_util.tree_map(lambda x: x[..., -1], rms_val))
          elif parent.expert_optim == "sgdm":
            next_params_expert = jax.tree_util.tree_map(
                lambda p, ml: sgdm_expert(
                    p, ml, current_expert_lr, parent.expert_weight_decay),
                params,
                jax.tree_util.tree_map(lambda x: x[..., 0], m))
          elif parent.expert_optim == "muon":
            next_params_expert = jax.tree_util.tree_map(
                lambda p, m_muon, m_adam, rl, g: muon_expert(
                    p, m_muon, m_adam, rl, g,
                    next_iteration,
                    current_expert_lr,
                    parent.expert_weight_decay,
                    beta1=parent.initial_momentum_decays[1],
                    adam_beta1=parent.initial_momentum_decays[0],
                    beta2=rms_beta2_val,
                    ns_steps=5,
                    adamlr_scaler=parent.muon_expert_adamlr_scaler),
                params,
                jax.tree_util.tree_map(lambda x: x[..., 1], m),
                jax.tree_util.tree_map(lambda x: x[..., 0], m),
                jax.tree_util.tree_map(lambda x: x[..., -1], rms_val),
                grad)

          # Blend all params
          next_params = jax.tree_util.tree_map(
              lambda p1, p2: opt_state.expert_weight * p1 + (1.0 - opt_state.expert_weight) * p2,
              next_params_expert, next_params_lo)

          # IMT losses: only on CELO2 (2D+) params
          if parent.use_adamw_for_1d:
            is_celo2 = jax.tree.map_with_path(
                lambda path, val: val.ndim > 1 and 'embed' not in jax.tree_util.keystr(path),
                params)
          else:
            is_celo2 = jax.tree_util.tree_map(lambda _: True, params)

          update_expert = jax.tree_util.tree_map(lambda n, c: n - c, next_params_expert, params)
          update_lo = jax.tree_util.tree_map(lambda n, c: n - c, next_params_lo, params)
          flat_expert = jnp.concatenate([
              x.ravel() for x, ic in zip(
                  jax.tree_util.tree_leaves(update_expert),
                  jax.tree_util.tree_leaves(is_celo2)) if ic])
          flat_lo = jnp.concatenate([
              x.ravel() for x, ic in zip(
                  jax.tree_util.tree_leaves(update_lo),
                  jax.tree_util.tree_leaves(is_celo2)) if ic])

          dot = jnp.sum(flat_expert * flat_lo)
          norm_expert = jnp.linalg.norm(flat_expert)
          norm_lo = jnp.linalg.norm(flat_lo)
          cosine_loss = 1.0 - dot / (norm_expert * norm_lo + 1e-30)
          magnitude_loss = jnp.mean(jnp.abs(jnp.abs(flat_expert) - jnp.abs(flat_lo)))
          # magnitude_loss = jnp.abs(norm_lo - norm_expert)

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
          )
          return tree_utils.match_type(next_opt_state, opt_state), cosine_loss, magnitude_loss
        else:
          next_opt_state = ELOCelo2State(
              params=next_params_lo,
              state=merged_state,
              iteration=next_iteration,
              num_steps=opt_state.num_steps,
              scheduled_lr=current_lr,
              expert_weight=opt_state.expert_weight,
              expert_lr=current_expert_lr,
              mom_rolling=next_mom,
              rms_rolling=next_rms,
              fac_rolling=next_fac,
          )
          return tree_utils.match_type(next_opt_state, opt_state)

    return _Opt()
