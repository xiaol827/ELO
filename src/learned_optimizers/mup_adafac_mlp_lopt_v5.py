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

"""MLP learned optimizer with adafactor features."""
import functools
from typing import Any, Optional

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
from helpers import cast_to_bf16, cast_to_fp8

from functools import reduce
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

import numpy as np
is_leaf = lambda x : reduce(np.logical_and, [type(x1) != dict for x1 in x.values()])

def add_prefix(prefix,s):
    if prefix != '':
        prefix = prefix + '/'
    return prefix + s

def get_mup_lrs(state,prefix):
  d = {}
  for k,v in state.items():
      if is_leaf(v):
          d[add_prefix(prefix,k)] = v
      else:
          for kk,vv in get_mup_lrs(v,k).items():
              d[add_prefix(prefix,kk)] = vv
  
  d = {k.replace('/mup_lrs',''):v for k,v in d.items()}
  return d


@flax.struct.dataclass
class AdafacMLPLOptState:
  params: Any
  state: Any
  mom_rolling: common.MomAccumulator
  rms_rolling: common.RMSAccumulator
  fac_rolling_features: common.FactoredAccum
  num_steps: jnp.ndarray
  iteration: jnp.ndarray


def decay_to_param(x):
  return jnp.log(1 - x) / 10.


def param_to_decay(x):
  return 1 - jnp.exp(x * 10.)


@gin.configurable
class MuAdafacMLPLOptV4(lopt_base.LearnedOptimizer):
  """MLP based learned optimizer with adafactor style accumulators."""

  def __init__(self,
               exp_mult=0.001,
               step_mult=0.001,
               hidden_size=4,
               hidden_layers=2,
               initial_momentum_decays=(0.9, 0.99, 0.999),
               initial_rms_decays=(0.999,),
               initial_adafactor_decays=(0.9, 0.99, 0.999),
               concat_weights=True,
               make_separate_weights=False,
               split_weights=False,
               clip_norm=None,
               mup_lrs=None,
               zero_training_step_feature=False,
               ):
    super().__init__()
    self._exp_mult = exp_mult
    self._step_mult = step_mult
    self._hidden_size = hidden_size
    self._hidden_layers = hidden_layers
    self._initial_momentum_decays = initial_momentum_decays
    self._initial_rms_decays = initial_rms_decays
    self._initial_adafactor_decays = initial_adafactor_decays
    self._concat_weights = concat_weights
    self._make_separate_weights = make_separate_weights
    self._split_weights = split_weights
    self.clip_norm = clip_norm
    self.mup_lrs = mup_lrs
    self.zero_training_step_feature = zero_training_step_feature
    self.non_linearity = lambda x: x * jax.nn.sigmoid(x)  # swish activation

    self._mod_init, self._mod_apply = hk.without_apply_rng(
        hk.transform(self._mod))
    


  @jax.default_matmul_precision("bfloat16")
  def _mod(self, 
           global_feat,  # Contains time features, iteration count, and total steps
           p,  # Parameter values
           g,  # Gradient values
           m,  # Momentum accumulators with different decay rates
           rms,  # Second moment accumulators with different decay rates
           fac_g,  # Adafactor gradient features
           fac_vec_col,  # Adafactor column accumulator
           fac_vec_row,  # Adafactor row accumulator
           fac_vec_v,  # Adafactor accumulator for non-factored case
           mup_lr_scale):  # Learning rate scale from muP (width aware scaling)
    """LO forward pass"""
    # this doesn't work with scalar parameters, so instead lets just reshape.
    if not p.shape:
      p = jnp.expand_dims(p, 0)
      g = jnp.expand_dims(g, 0)
      m = jnp.expand_dims(m, 0)
      rms = jnp.expand_dims(rms, 0)
      fac_g = jnp.expand_dims(fac_g, 0)
      fac_vec_v = jnp.expand_dims(fac_vec_v, 0)
      fac_vec_col = jnp.expand_dims(fac_vec_col, 0)
      fac_vec_row = jnp.expand_dims(fac_vec_row, 0)
      mup_lr_scale = jnp.expand_dims(mup_lr_scale, 0)
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
      # Construct features for
      d1, d0 = factored_dims

      # add 2 dims: 1 for batch of decay, one because low rank
      to_tile = [1] * (1 + len(g.shape))
      to_tile[d0] = g.shape[d0]

      row_feat = jnp.tile(jnp.expand_dims(fac_vec_row, axis=d0), to_tile)

      to_tile = [1] * (1 + len(g.shape))
      to_tile[d1] = g.shape[d1]
      col_feat = jnp.tile(jnp.expand_dims(fac_vec_col, axis=d1), to_tile)

      # 3 possible kinds of adafactor style features.
      # Raw values
      inps.append(row_feat)
      inps.append(col_feat)

      # 1/sqrt
      inps.append(lax.rsqrt(row_feat + 1e-8))
      inps.append(lax.rsqrt(col_feat + 1e-8))

      # multiplied by momentum
      reduced_d1 = d1 - 1 if d1 > d0 else d1
      row_col_mean = jnp.mean(fac_vec_row, axis=reduced_d1, keepdims=True)

      row_factor = common.safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
      col_factor = common.safe_rsqrt(fac_vec_col)
      fac_mom_mult = (
          m * jnp.expand_dims(row_factor, axis=d0) *
          jnp.expand_dims(col_factor, axis=d1))
      inps.append(fac_mom_mult)
    else:
      # In the non-factored case, match what RMSProp does.
      inps.append(fac_vec_v)
      inps.append(fac_vec_v)

      inps.append(lax.rsqrt(fac_vec_v + 1e-8))
      inps.append(lax.rsqrt(fac_vec_v + 1e-8))

      fac_mom_mult = m * (fac_vec_v + 1e-6)**-0.5
      inps.append(fac_mom_mult)

    # Build the weights of the NN
    last_size = jnp.concatenate(inps, axis=-1).shape[-1]
    last_size += global_feat["training_step_feature"].shape[-1]



    ############################################################
    # create NN weights and biases
    ############################################################
    weights = []
    biases = []

    for wi, w in enumerate([self._hidden_size] * self._hidden_layers + [2]):
      stddev = 1. / onp.sqrt(last_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)

      weights.append(
          hk.get_parameter(
              f"w{wi}", shape=(last_size, w), dtype=jnp.float32, init=w_init))
      biases.append(
          hk.get_parameter(
              f"b{wi}", shape=(w,), dtype=jnp.float32, init=jnp.zeros))
      last_size = w


    

    ############################################################
    # concat the inputs, normalize
    ############################################################
    inp_stack = jnp.concatenate(inps, axis=-1)
    axis = list(range(len(p.shape)))
    inp_stack = second_moment_normalizer(inp_stack, axis=axis)

    # add features that should not be normalized
    training_step_feature = global_feat["training_step_feature"]
    stacked = jnp.reshape(training_step_feature, [1] * len(axis) +
                          list(training_step_feature.shape[-1:]))
    stacked = jnp.tile(stacked, list(p.shape) + [1])
    inp_stack = jnp.concatenate([inp_stack, stacked], axis=-1)

    # with Timer("lopt forward") as t:
    # Manually run the neural network.
    net = inp_stack
    for wi, (w, b) in enumerate(zip(weights, biases)):
      o_tmp = net @ w
      net = o_tmp + jnp.broadcast_to(b, list(net.shape[0:-1]) + [w.shape[-1]])  # pytype: disable=attribute-error

      if wi != len(weights) - 1:
        net =  self.non_linearity(net)

    direction = net[..., 0]
    magnitude = net[..., 1]

    step = direction * jnp.exp(magnitude * self._exp_mult) * self._step_mult
    step = step.reshape(p.shape)
    new_p = p - step * mup_lr_scale

    if did_reshape:
      new_p = jnp.squeeze(new_p, 0)

    # Finally, log some metrics out
    avg_step_size = jnp.mean(jnp.abs(step))
    summary.summary("adafac_mlp_lopt/avg_step_size", avg_step_size)
    summary.summary(
        "adafac_mlp_lopt/avg_step_size_hist",
        avg_step_size,
        aggregation="collect")
    summary.summary("adafac_mlp_lopt/direction/mean_abs",
                    jnp.mean(jnp.abs(direction)))
    summary.summary("adafac_mlp_lopt/magnitude/mean_abs",
                    jnp.mean(jnp.abs(magnitude)))
    summary.summary("adafac_mlp_lopt/magnitude/mean", jnp.mean(magnitude))
    summary.summary("adafac_mlp_lopt/grad/mean_abs", jnp.mean(jnp.abs(g)))

    return new_p

  def init(self, key: PRNGKey) -> lopt_base.MetaParams:
    # We meta-learn:
    # * weights of the MLP
    # * decays of momentum, RMS, and adafactor style accumulators

    training_step_feature = tanh_embedding(1)
    global_features = {
        "iterations": 0,
        "num_steps": 10,
        "training_step_feature": training_step_feature,
    }
    # fake weights with 2 dimension
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
    mup_lr_scale = jnp.ones([r, c])
    mod_theta = self._mod_init(key, global_features, p, g, m, rms, fac_g,
                               fac_vec_col, fac_vec_row, fac_vec_v, mup_lr_scale)
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
        self.mup_lrs = None

      def _get_rolling(self):
        mom_decay = jnp.asarray(parent._initial_momentum_decays)
        mom_roll = common.vec_rolling_mom(mom_decay)

        rms_decay = jnp.asarray(parent._initial_rms_decays)
        rms_roll = common.vec_rolling_rms(rms_decay)

        adafactor_decay = jnp.asarray(parent._initial_adafactor_decays)
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
            num_steps=jnp.asarray(num_steps))

      def update(self,
                 opt_state: AdafacMLPLOptState,
                 grad: opt_base.Gradient,
                 loss: jnp.ndarray,
                 model_state: Optional[opt_base.ModelState] = None,
                 is_valid: bool = False,
                 key: Optional[PRNGKey] = None) -> AdafacMLPLOptState:


        lrs = model_state['mup_lrs_to_use']
        
        if parent.clip_norm:
          clipping = optax.clip_by_global_norm(parent.clip_norm)

          # Apply gradient clipping
          grad, _ = clipping.update(grad, None)


        grad = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grad)

        mom_roll, rms_roll, fac_vec_roll = self._get_rolling()
        next_mom_rolling = mom_roll.update(opt_state.mom_rolling, grad)
        next_rms_rolling = rms_roll.update(opt_state.rms_rolling, grad)
        next_fac_rolling_features, fac_g = fac_vec_roll.update(
            opt_state.fac_rolling_features, grad)

        # compute some global features
        training_step_feature = tanh_embedding(opt_state.iteration)

        if parent.zero_training_step_feature:
          training_step_feature = jnp.zeros_like(training_step_feature)
          
        global_features = {
            "iterations": opt_state.iteration,
            "num_steps": opt_state.num_steps,
            "training_step_feature": training_step_feature,
        }

        fun = functools.partial(mod_apply, self.theta["nn"], global_features)

        next_params = jax.tree_util.tree_map(fun, opt_state.params, grad,
                                             next_mom_rolling.m,
                                             next_rms_rolling.rms, fac_g,
                                             next_fac_rolling_features.v_col,
                                             next_fac_rolling_features.v_row,
                                             next_fac_rolling_features.v_diag,
                                             lrs)

        next_opt_state = AdafacMLPLOptState(
            params=next_params,
            mom_rolling=next_mom_rolling,
            rms_rolling=next_rms_rolling,
            fac_rolling_features=next_fac_rolling_features,
            iteration=opt_state.iteration + 1,
            state=model_state,
            num_steps=opt_state.num_steps)

        return tree_utils.match_type(next_opt_state, opt_state)

    return _Opt(theta)
