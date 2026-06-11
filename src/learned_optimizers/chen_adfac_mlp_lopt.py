# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chen-style learned optimizer: AdamW expert + per-parameter MSE imitation loss."""
import functools
from typing import Any, Optional
from learned_optimization.optimizers import optax_opts
import flax
import gin
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
import numpy as onp
import optax

PRNGKey = jnp.ndarray


def second_moment_normalizer(x, axis, eps=1e-5):
  return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def tanh_embedding(x):
  f32 = jnp.float32

  def one_freq(timescale):
    return jnp.tanh(x / (f32(timescale)) - 1.0)

  timescales = jnp.asarray(
      [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
      dtype=jnp.float32)
  return jax.vmap(one_freq)(timescales)


@flax.struct.dataclass
class AdafacMLPLOptState:
  params: Any
  state: Any
  mom_rolling: common.MomAccumulator
  rms_rolling: common.RMSAccumulator
  fac_rolling_features: common.FactoredAccum
  num_steps: jnp.ndarray
  iteration: jnp.ndarray
  scheduled_lr: jnp.ndarray
  expert_weight: jnp.ndarray
  expert_lr: jnp.ndarray


def decay_to_param(x):
  return jnp.log(1 - x) / 10.


def param_to_decay(x):
  return 1 - jnp.exp(x * 10.)


def cosine_scheduler(step, num_steps, lr_init, lr_final=0.0):
  fraction = jnp.clip(step / (num_steps - 1), 0.0, 1.0)
  cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * fraction))
  return lr_final + (lr_init - lr_final) * cosine_decay


def adam_expert(param, m, v, beta1, beta2, t, lr, wd, eps=1e-8):
  m_hat = m / (1.0 - beta1**t)
  v_hat = v / (1.0 - beta2**t)
  update = m_hat / (jnp.sqrt(v_hat) + eps)
  return param - lr * (update + wd * param)


@gin.configurable
class ChenAdafacMLPLOpt(lopt_base.LearnedOptimizer):
  """Chen-style MLP learned optimizer with adafactor features.

  Differences from ELO_AdafacMLPLOpt:
    * expert trajectory is AdamW only (no sgdm/muon variants);
    * imitation loss is a single per-parameter MSE on the update vector,
      not split into direction/magnitude;
    * update() returns (opt_state, imt_loss) instead of a (opt_state, cos, mag) triple.
  """

  def __init__(self,
               exp_mult=0.001,
               step_mult=0.001,
               weight_decay=0.0,
               expert_lr_max=0.001,
               expert_lr_min=1e-5,
               expert_lr_decay_steps=10000,
               expert_weight_decay=0.0,
               hidden_size=4,
               hidden_layers=2,
               initial_momentum_decays=(0.9, 0.99, 0.999),
               initial_rms_decays=(0.999,),
               initial_adafactor_decays=(0.9, 0.99, 0.999),
               concat_weights=True,
               clip_grad=False,
               clip_norm=1.0,
               make_separate_weights=False,
               split_weights=False,
               meta_train=True,
               use_lo_cosine_scheduler=False,
               step_mult_min=1e-4,
               init_lr=0.0,
               warmup_fraction=0.05,
               warmup_steps=0):
    super().__init__()
    self._exp_mult = exp_mult
    self._step_mult = step_mult
    self.expert_lr_max = expert_lr_max
    self.expert_lr_min = expert_lr_min
    self.expert_lr_decay_steps = expert_lr_decay_steps
    self._hidden_size = hidden_size
    self._hidden_layers = hidden_layers
    self._initial_momentum_decays = initial_momentum_decays
    self._initial_rms_decays = initial_rms_decays
    self._initial_adafactor_decays = initial_adafactor_decays
    self._concat_weights = concat_weights
    self._make_separate_weights = make_separate_weights
    self._split_weights = split_weights
    self.clip_grad = clip_grad
    self.clip_norm = clip_norm
    self.weight_decay = weight_decay
    self.expert_weight_decay = expert_weight_decay
    self.meta_train = meta_train
    self.use_lo_cosine_scheduler = use_lo_cosine_scheduler
    self.step_mult_min = step_mult_min
    self.init_lr = init_lr
    self.warmup_fraction = warmup_fraction
    self.warmup_steps = warmup_steps

    self._mod_init, self._mod_apply = hk.without_apply_rng(
        hk.transform(self._mod))

  def _mod(self, global_feat, scheduled_lr, p, g, m, rms, fac_g, fac_vec_col, fac_vec_row,
           fac_vec_v):
    if not p.shape:
      p = jnp.expand_dims(p, 0)
      g = jnp.expand_dims(g, 0)
      m = jnp.expand_dims(m, 0)
      rms = jnp.expand_dims(rms, 0)
      fac_g = jnp.expand_dims(fac_g, 0)
      fac_vec_v = jnp.expand_dims(fac_vec_v, 0)
      fac_vec_col = jnp.expand_dims(fac_vec_col, 0)
      fac_vec_row = jnp.expand_dims(fac_vec_row, 0)
      did_reshape = True
    else:
      did_reshape = False
    inps = []

    inps.append(jnp.expand_dims(g, axis=-1))
    inps.append(jnp.expand_dims(p, axis=-1))
    inps.append(m)
    inps.append(rms)
    rsqrt = lax.rsqrt(rms + 1e-6)
    inps.append(m * rsqrt)
    inps.append(rsqrt)
    inps.append(fac_g)

    factored_dims = common.factored_dims(g.shape)
    if factored_dims is not None:
      d1, d0 = factored_dims

      to_tile = [1] * (1 + len(g.shape))
      to_tile[d0] = g.shape[d0]
      row_feat = jnp.tile(jnp.expand_dims(fac_vec_row, axis=d0), to_tile)

      to_tile = [1] * (1 + len(g.shape))
      to_tile[d1] = g.shape[d1]
      col_feat = jnp.tile(jnp.expand_dims(fac_vec_col, axis=d1), to_tile)

      inps.append(row_feat)
      inps.append(col_feat)

      inps.append(lax.rsqrt(row_feat + 1e-8))
      inps.append(lax.rsqrt(col_feat + 1e-8))

      reduced_d1 = d1 - 1 if d1 > d0 else d1
      row_col_mean = jnp.mean(fac_vec_row, axis=reduced_d1, keepdims=True)

      row_factor = common.safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
      col_factor = common.safe_rsqrt(fac_vec_col)
      fac_mom_mult = (
          m * jnp.expand_dims(row_factor, axis=d0) *
          jnp.expand_dims(col_factor, axis=d1))
      inps.append(fac_mom_mult)
    else:
      inps.append(fac_vec_v)
      inps.append(fac_vec_v)
      inps.append(lax.rsqrt(fac_vec_v + 1e-8))
      inps.append(lax.rsqrt(fac_vec_v + 1e-8))
      fac_mom_mult = m * (fac_vec_v + 1e-6)**-0.5
      inps.append(fac_mom_mult)

    last_size = jnp.concatenate(inps, axis=-1).shape[-1]
    last_size += global_feat["training_step_feature"].shape[-1]

    weights = []
    biases = []

    for wi, w in enumerate([self._hidden_size] * self._hidden_layers + [2]):
      stddev = 1. / onp.sqrt(last_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)

      make_full_weights = self._concat_weights or (
          not self._make_separate_weights)
      if make_full_weights:
        weights.append(
            hk.get_parameter(
                f"w{wi}", shape=(last_size, w), dtype=jnp.float32, init=w_init))
        biases.append(
            hk.get_parameter(
                f"b{wi}", shape=(w,), dtype=jnp.float32, init=jnp.zeros))
      else:
        if self._make_separate_weights:
          weights.append([])
          for vi in range(last_size):
            ww = []
            for oi in range(w):
              wij = hk.get_parameter(
                  f"w{wi}_{vi}_{oi}", shape=[], dtype=jnp.float32, init=w_init)
              ww.append(wij)
            weights[-1].append(ww)
          biases.append([])
          for oi in range(w):
            b = hk.get_parameter(
                f"b{wi}_{oi}", shape=[], dtype=jnp.float32, init=jnp.zeros)
            biases[-1].append(b)
        elif self._split_weights:
          f = list(x for x in weights[-1].ravel())
          weights[-1] = [[None] * w for i in range(last_size)]
          for fi, ff in enumerate(f):
            i = fi % last_size
            j = fi // last_size
            weights[-1][i][j] = ff
            biases[-1] = list(b for b in biases[-1])
      last_size = w

    if self._concat_weights:
      inp_stack = jnp.concatenate(inps, axis=-1)
      axis = list(range(len(p.shape)))
      inp_stack = second_moment_normalizer(inp_stack, axis=axis)

      training_step_feature = global_feat["training_step_feature"]
      stacked = jnp.reshape(training_step_feature, [1] * len(axis) +
                            list(training_step_feature.shape[-1:]))
      stacked = jnp.tile(stacked, list(p.shape) + [1])
      inp_stack = jnp.concatenate([inp_stack, stacked], axis=-1)

      net = inp_stack
      for wi, (w, b) in enumerate(zip(weights, biases)):
        o_tmp = net @ w
        net = o_tmp + jnp.broadcast_to(b, list(net.shape[0:-1]) + [w.shape[-1]])

        if wi != len(weights) - 1:
          net = jax.nn.relu(net)

      direction = net[..., 0]
      magnitude = net[..., 1]
    else:
      flat_features = []
      for i in inps:
        flat_features.extend(
            [jnp.squeeze(x, -1) for x in jnp.split(i, i.shape[-1], axis=-1)])

      inp = [
          x * lax.rsqrt(1e-5 + jnp.mean(jnp.square(x), keepdims=True))
          for x in flat_features
      ]
      for wi, (w, b) in enumerate(zip(weights, biases)):
        grids = []

        for oi in range(len(w[0])):
          outs = []
          for vi, v in enumerate(inp):
            if type(w) == list:
              outs.append(v * w[vi][oi])
            else:
              outs.append(v * w[vi, oi])

          if wi == 0:
            training_step_feature = global_feat["training_step_feature"]
            for i, vi in enumerate(
                range(vi + 1, vi + 1 + len(training_step_feature))):
              if type(w) == list:
                outs.append(training_step_feature[i] * w[vi][oi])
              else:
                outs.append(training_step_feature[i] * w[vi, oi])

          grids.append(outs)

        out_mul = [sum(g) for g in grids]

        inp = []
        for oi, net in enumerate(out_mul):
          inp.append(net + b[oi])

        if wi != len(weights) - 1:
          inp = [jax.nn.relu(x) for x in inp]

      direction = inp[0]
      magnitude = inp[1]

    step = direction * jnp.exp(magnitude * self._exp_mult)
    step = step.reshape(p.shape)
    new_p = p - (step + self.weight_decay * p) * scheduled_lr

    if did_reshape:
      new_p = jnp.squeeze(new_p, 0)

    avg_step_size = jnp.mean(jnp.abs(step))
    summary.summary("chen_adfac_mlp_lopt/avg_step_size", avg_step_size)
    summary.summary(
        "chen_adfac_mlp_lopt/avg_step_size_hist",
        avg_step_size,
        aggregation="collect")
    summary.summary("chen_adfac_mlp_lopt/direction/mean_abs",
                    jnp.mean(jnp.abs(direction)))
    summary.summary("chen_adfac_mlp_lopt/magnitude/mean_abs",
                    jnp.mean(jnp.abs(magnitude)))
    summary.summary("chen_adfac_mlp_lopt/magnitude/mean", jnp.mean(magnitude))
    summary.summary("chen_adfac_mlp_lopt/grad/mean_abs", jnp.mean(jnp.abs(g)))

    return new_p

  def init(self, key: PRNGKey) -> lopt_base.MetaParams:
    training_step_feature = tanh_embedding(1)
    global_features = {
        "iterations": 0,
        "num_steps": 10,
        "training_step_feature": training_step_feature,
    }
    r = 10
    c = 10
    p = jnp.ones([r, c])
    g = jnp.ones([r, c])

    m = jnp.ones([r, c, len(self._initial_momentum_decays)])
    rms = jnp.ones([r, c, len(self._initial_rms_decays)])

    fac_g = jnp.ones([r, c, len(self._initial_adafactor_decays)])
    fac_vec_row = jnp.ones([r, len(self._initial_adafactor_decays)])
    fac_vec_col = jnp.ones([c, len(self._initial_adafactor_decays)])
    fac_vec_v = jnp.ones([len(self._initial_adafactor_decays)])
    scheduled_lr = 1.0
    mod_theta = self._mod_init(key, global_features, scheduled_lr, p, g, m, rms, fac_g,
                               fac_vec_col, fac_vec_row, fac_vec_v)
    return hk.data_structures.to_haiku_dict({
        "nn": mod_theta
    })

  def opt_fn(self,
             theta: lopt_base.MetaParams,
             is_training: Optional[bool] = False) -> opt_base.Optimizer:

    mod_apply = self._mod_apply
    parent = self

    class _Opt(opt_base.Optimizer):
      """Optimizer capturing the meta params."""

      def __init__(self, theta):
        self.theta = theta
        self.use_lo_cosine_scheduler = parent.use_lo_cosine_scheduler
        self.meta_train = parent.meta_train

      def _get_rolling(self, learned_accums=False):
        if not learned_accums:
          mom_decay = jnp.asarray(parent._initial_momentum_decays)
        else:
          mom_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_momentum_decays)) +
            self.theta["momentum_decays"])
        mom_roll = common.vec_rolling_mom(mom_decay)

        if not learned_accums:
          rms_decay = jnp.asarray(parent._initial_rms_decays)
        else:
          rms_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_rms_decays)) +
            self.theta["rms_decays"])
        rms_roll = common.vec_rolling_rms(rms_decay)

        if not learned_accums:
          adafactor_decay = jnp.asarray(parent._initial_adafactor_decays)
        else:
          adafactor_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_adafactor_decays)) +
            self.theta["adafactor_decays"])
        fac_vec_roll = common.vec_factored_rolling(adafactor_decay)
        return mom_roll, rms_roll, fac_vec_roll

      def init(
          self,
          params: opt_base.Params,
          model_state: Optional[opt_base.ModelState] = None,
          num_steps: Optional[int] = None,
          key: Optional[PRNGKey] = None,
      ) -> AdafacMLPLOptState:
        if num_steps is None:
          raise ValueError("Must specify number of steps for this lopt!")

        mom_roll, rms_roll, fac_vec_roll = self._get_rolling()
        return AdafacMLPLOptState(
            params=params,
            state=model_state,
            rms_rolling=rms_roll.init(params),
            mom_rolling=mom_roll.init(params),
            fac_rolling_features=fac_vec_roll.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            expert_weight=jnp.asarray(1.0, dtype=jnp.float32),
            num_steps=jnp.asarray(num_steps),
            scheduled_lr=jnp.asarray(parent._step_mult, dtype=jnp.float32),
            expert_lr=jnp.asarray(parent.expert_lr_max, dtype=jnp.float32))

      def update(
          self,
          opt_state: AdafacMLPLOptState,
          grad: opt_base.Gradient,
          loss: jnp.ndarray,
          model_state: Optional[opt_base.ModelState] = None,
          key: Optional[PRNGKey] = None,
      ) -> AdafacMLPLOptState:
        if parent.clip_grad:
          clip_norm = parent.clip_norm
          clipping = optax.clip_by_global_norm(clip_norm)
          grad, _ = clipping.update(grad, None)

        grad = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grad)
        current_params = opt_state.params

        mom_roll, rms_roll, fac_vec_roll = self._get_rolling()
        next_mom_rolling = mom_roll.update(opt_state.mom_rolling, grad)
        next_rms_rolling = rms_roll.update(opt_state.rms_rolling, grad)
        next_fac_rolling_features, fac_g = fac_vec_roll.update(
            opt_state.fac_rolling_features, grad)

        training_step_feature = tanh_embedding(opt_state.iteration)

        global_features = {
            "iterations": opt_state.iteration,
            "num_steps": opt_state.num_steps,
            "training_step_feature": training_step_feature,
        }
        total_step = 10000 if self.meta_train else opt_state.num_steps
        scheduled_lr = lax.cond(
            self.use_lo_cosine_scheduler,
            lambda _: cosine_scheduler(opt_state.iteration, total_step, parent._step_mult, parent.step_mult_min),
            lambda _: jnp.array(parent._step_mult),
            None,
        )
        # Linear warmup: init_lr → step_mult over warmup_n steps. After warmup,
        # the existing scheduled_lr (constant or cosine to step_mult_min) takes over.
        if parent.warmup_fraction > 0:
          warmup_n_raw = parent.warmup_fraction * opt_state.num_steps
        else:
          warmup_n_raw = parent.warmup_steps
        warmup_n = jnp.maximum(jnp.asarray(warmup_n_raw, jnp.float32), 1.0)
        warmup_lr = parent.init_lr + (parent._step_mult - parent.init_lr) * jnp.minimum(
            opt_state.iteration.astype(jnp.float32) / warmup_n, 1.0)
        scheduled_lr = jnp.where(
            opt_state.iteration.astype(jnp.float32) < warmup_n,
            warmup_lr,
            scheduled_lr,
        )

        fun = functools.partial(mod_apply, self.theta["nn"], global_features, scheduled_lr)

        next_params_lo = jax.tree_util.tree_map(fun, current_params, grad,
                                             next_mom_rolling.m,
                                             next_rms_rolling.rms, fac_g,
                                             next_fac_rolling_features.v_col,
                                             next_fac_rolling_features.v_row,
                                             next_fac_rolling_features.v_diag)

        current_expert_lr = cosine_scheduler(
            opt_state.iteration, parent.expert_lr_decay_steps,
            parent.expert_lr_max, parent.expert_lr_min)

        if self.meta_train:
          next_params_expert = jax.tree_util.tree_map(
                  lambda p, m, v: adam_expert(
                      p, m, v,
                      parent._initial_momentum_decays[0],
                      parent._initial_rms_decays[0],
                      opt_state.iteration + 1,
                      current_expert_lr,
                      parent.expert_weight_decay),
                  current_params,
                  jax.tree_util.tree_map(lambda x: x[..., 0], next_mom_rolling.m),
                  jax.tree_util.tree_map(lambda x: x[..., 0], next_rms_rolling.rms))

          next_params = jax.tree_util.tree_map(
              lambda p1, p2: opt_state.expert_weight * p1 + (1.0 - opt_state.expert_weight) * p2,
              next_params_expert, next_params_lo)

          # Chen imitation loss: per-parameter MSE on the update vector.
          update_expert = jax.tree_util.tree_map(lambda n, c: n - c, next_params_expert, current_params)
          update_lo = jax.tree_util.tree_map(lambda n, c: n - c, next_params_lo, current_params)
          per_leaf_mse = [
              jnp.mean((ue - ul) ** 2)
              for ue, ul in zip(jax.tree_util.tree_leaves(update_expert),
                                jax.tree_util.tree_leaves(update_lo))
          ]
          imt_loss = jnp.mean(jnp.stack(per_leaf_mse))

          next_opt_state = AdafacMLPLOptState(
              params=next_params,
              mom_rolling=next_mom_rolling,
              rms_rolling=next_rms_rolling,
              fac_rolling_features=next_fac_rolling_features,
              iteration=opt_state.iteration + 1,
              expert_weight=opt_state.expert_weight,
              state=model_state,
              num_steps=opt_state.num_steps,
              scheduled_lr=scheduled_lr,
              expert_lr=current_expert_lr)

          return tree_utils.match_type(next_opt_state, opt_state), imt_loss
        else:
          next_opt_state = AdafacMLPLOptState(
              params=next_params_lo,
              mom_rolling=next_mom_rolling,
              rms_rolling=next_rms_rolling,
              fac_rolling_features=next_fac_rolling_features,
              iteration=opt_state.iteration + 1,
              expert_weight=opt_state.expert_weight,
              state=model_state,
              num_steps=opt_state.num_steps,
              scheduled_lr=scheduled_lr,
              expert_lr=current_expert_lr)

          return tree_utils.match_type(next_opt_state, opt_state)

    return _Opt(theta)
