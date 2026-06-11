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
from opt import AnyOptimizer
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
  bc_opt_state: Any


def decay_to_param(x):
  return jnp.log(1 - x) / 10.


def param_to_decay(x):
  return 1 - jnp.exp(x * 10.)


@gin.configurable
class MuAdafacMLPLOptBC(lopt_base.LearnedOptimizer):
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
               clip_grad=False,
               clip_norm=1.0,
               mup_lrs=None,
               zero_training_step_feature=False,
               adam_lr=0.044173,
               teacher_force=False,
               quantized=None,
               train=True,
               bc_optimizer_args = None,
               
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
    self.clip_grad = clip_grad
    self.clip_norm = clip_norm
    self.mup_lrs = mup_lrs
    self.quantized = quantized
    self.zero_training_step_feature = zero_training_step_feature
    self.adam_lr = adam_lr
    self.teacher_force = teacher_force
    self.bc_optimizer_args = bc_optimizer_args
    self._mod_init, self._mod_apply = hk.without_apply_rng(
        hk.transform(self._mod))
    self.train = train

  def cast_by_args(self, x):
    if self.quantized == 'bf16':
      return cast_to_bf16(x)
    elif self.quantized == 'fp8':
      return cast_to_fp8(x)
    else:
      return x

  @jax.default_matmul_precision("bfloat16")
  def _mod(self, global_feat, p, g, m, rms, fac_g, fac_vec_col, fac_vec_row,
           fac_vec_v,mup_lr_scale):
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
        # Otherwise weights will be stored as scalars.
        # these scalars could be made from scratch, split from weights made
        # above
        if self._make_separate_weights:
          # Manually make the weight matrix in scalars.
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
          # split up the weights first before running computation.
          f = list(x for x in weights[-1].ravel())
          weights[-1] = [[None] * w for i in range(last_size)]
          for fi, ff in enumerate(f):
            i = fi % last_size
            j = fi // last_size
            weights[-1][i][j] = ff
            biases[-1] = list(b for b in biases[-1])
      last_size = w


    

    # 2 different methods to compute the learned optimizer weight update are
    # provided. First, using matmuls (like a standard NN). Second, with the
    # computation unpacked using only scalar math. This uses a different path
    # in hardware and can be much faster for small learned optimizer hidden
    # sizes.
    if self._concat_weights:
      # concat the inputs, normalize
      inp_stack = jnp.concatenate(inps, axis=-1)
      axis = list(range(len(p.shape)))
      inp_stack = second_moment_normalizer(inp_stack, axis=axis)

      # add features that should not be normalized
      training_step_feature = global_feat["training_step_feature"]
      stacked = jnp.reshape(training_step_feature, [1] * len(axis) +
                            list(training_step_feature.shape[-1:]))
      stacked = jnp.tile(stacked, list(p.shape) + [1])
      inp_stack = jnp.concatenate([inp_stack, stacked], axis=-1)

      weights = self.cast_by_args(weights)
      biases = self.cast_by_args(biases)
      inp = self.cast_by_args(inp_stack)

      # print(jax.tree_util.tree_map(lambda x: x.dtype, weights))
      # exit(0)

      # with Timer("lopt forward") as t:
      # Manually run the neural network.
      net = inp_stack
      for wi, (w, b) in enumerate(zip(weights, biases)):
        o_tmp = net @ w
        net = o_tmp + jnp.broadcast_to(b, list(net.shape[0:-1]) + [w.shape[-1]])  # pytype: disable=attribute-error

        if wi != len(weights) - 1:
          net = jax.nn.relu(net)

      direction = net[..., 0]
      magnitude = net[..., 1]
    else:
      # The scalar math path.
      flat_features = []
      for i in inps:
        flat_features.extend(
            [jnp.squeeze(x, -1) for x in jnp.split(i, i.shape[-1], axis=-1)])

      # match the second moment normalize calculation but applied to each scalar
      inp = [
          x * lax.rsqrt(1e-5 + jnp.mean(jnp.square(x), keepdims=True))
          for x in flat_features
      ]
      
      weights = self.cast_by_args(weights)
      biases = self.cast_by_args(biases)
      inp = self.cast_by_args(inp)
      for wi, (w, b) in enumerate(zip(weights, biases)):
        grids = []

        # hidden layer wi
        for oi in range(len(w[0])):
          outs = []
          for vi, v in enumerate(inp):
            if type(w) == list:  # pylint: disable=unidiomatic-typecheck
              outs.append(v * w[vi][oi])
            else:
              outs.append(v * w[vi, oi])  # pytype: disable=unsupported-operands

          if wi == 0:
            training_step_feature = global_feat["training_step_feature"]
            for i, vi in enumerate(
                range(vi + 1, vi + 1 + len(training_step_feature))):
              if type(w) == list:  # pylint: disable=unidiomatic-typecheck
                outs.append(training_step_feature[i] * w[vi][oi])
              else:
                outs.append(training_step_feature[i] * w[vi, oi])  # pytype: disable=unsupported-operands

          grids.append(outs)

        out_mul = [sum(g) for g in grids]

        # bias
        inp = []
        for oi, net in enumerate(out_mul):
          inp.append(net + b[oi])

        # activation
        if wi != len(weights) - 1:
          inp = [jax.nn.relu(x) for x in inp]

      direction = inp[0]
      magnitude = inp[1]

    step = direction * jnp.exp(magnitude * self._exp_mult) * self._step_mult
    step = step.reshape(p.shape)
    new_p = p - step * mup_lr_scale
    # print(mup_lr_scale)

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
        # "momentum_decays": jnp.zeros([len(self._initial_momentum_decays)]),
        # "rms_decays": jnp.zeros([len(self._initial_rms_decays)]),
        # "adafactor_decays": jnp.zeros([len(self._initial_adafactor_decays)]),
        "nn": mod_theta
    })


  def get_bc_optimizer(self, mup_lrs=None):
    if self.bc_optimizer_args.use_mup:
        assert mup_lrs is not None, "mup_lrs must be provided if use_mup is True"

    return AnyOptimizer(
        optimizer=self.bc_optimizer_args.optimizer_args,
        schedule=self.bc_optimizer_args.schedule,
        gradient_transform_before_optim=self.bc_optimizer_args.gradient_transform_before_optim,
        gradient_transform_after_optim=self.bc_optimizer_args.gradient_transform_after_optim,
        mup_lrs=mup_lrs if self.bc_optimizer_args.use_mup else None,
        local_optimizer_args=self.bc_optimizer_args,
    )
  
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
        # param_to_decay(
        #     decay_to_param(jnp.asarray(parent._initial_momentum_decays)) +  # pylint: disable=protected-access
        #     self.theta["momentum_decays"])
        mom_roll = common.vec_rolling_mom(mom_decay)

        rms_decay = jnp.asarray(parent._initial_rms_decays)
        # param_to_decay(
        #     decay_to_param(jnp.asarray(parent._initial_rms_decays)) +  # pylint: disable=protected-access
        #     self.theta["rms_decays"])
        rms_roll = common.vec_rolling_rms(rms_decay)

        adafactor_decay = jnp.asarray(parent._initial_adafactor_decays)
        # param_to_decay(
        #     decay_to_param(jnp.asarray(parent._initial_adafactor_decays)) +  # pylint: disable=protected-access
        #     self.theta["adafactor_decays"])
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

        if parent.train:
          bc_opt = parent.get_bc_optimizer(model_state['mup_lrs_to_use'])
          bc_opt_state = bc_opt.init(params, model_state, num_steps, key)
        else:
          bc_opt_state = None

        return AdafacMLPLOptState(
            params=params,
            state=model_state,
            rms_rolling=rms_roll.init(params),
            mom_rolling=mom_roll.init(params),
            fac_rolling_features=fac_vec_roll.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            num_steps=jnp.asarray(num_steps),
            bc_opt_state=bc_opt_state
            )

      def update(self,
                 opt_state: AdafacMLPLOptState,
                 grad: opt_base.Gradient,
                 loss: jnp.ndarray,
                 model_state: Optional[opt_base.ModelState] = None,
                 is_valid: bool = False,
                 key: Optional[PRNGKey] = None) -> AdafacMLPLOptState:

        lrs = model_state['mup_lrs_to_use']
        if parent.train:
          bc_opt = parent.get_bc_optimizer(lrs)
          bc_opt_state = bc_opt.update(opt_state=opt_state.bc_opt_state, 
                                        grad=grad, 
                                        loss=loss, 
                                        model_state=model_state, 
                                        key=key)
        
        if parent.clip_grad:
          clip_norm = parent.clip_norm
          clipping = optax.clip_by_global_norm(clip_norm)
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

        # print("mom rolling",jax.tree_util.tree_map(lambda x: x.shape, next_mom_rolling.m))
        # print("rms rolling",jax.tree_util.tree_map(lambda x: x.shape, next_rms_rolling.rms))
        # print("param",jax.tree_util.tree_map(lambda x: x.shape, opt_state.params))
        # print("fac rolling",jax.tree_util.tree_map(lambda x: x.shape, next_fac_rolling_features.v_col))
        # exit(0)

        if parent.train:
          # Define the loss function (MSE between output and target)
          def mse_loss(params1, params2):
              squared_diff = jax.tree_util.tree_map(lambda x, y: jnp.square(x - y), params1, params2)
              leaves = jax.tree_util.tree_leaves(squared_diff)
              return jnp.mean(jnp.array([jnp.mean(leaf) for leaf in leaves]))
          
          # Define a function to compute gradients for a single step
          def compute_bc_gradients(net_params):
            # Define a function that computes loss for a single step with given network parameters
            def loss_fn(params):
              
              # Apply the network to get the update
              fun = functools.partial(mod_apply, params['nn'], global_features)
              next_params = jax.tree_util.tree_map(fun, opt_state.params, grad,
                                            next_mom_rolling.m,
                                            next_rms_rolling.rms, fac_g,
                                            next_fac_rolling_features.v_col,
                                            next_fac_rolling_features.v_row,
                                            next_fac_rolling_features.v_diag,
                                            lrs)
              
              # Compute loss against target
              loss = mse_loss( jax.tree_util.tree_map(lambda x,y: x - y, next_params, opt_state.params), 
                              jax.tree_util.tree_map(lambda x,y: x - y, bc_opt_state.params, opt_state.params))
              
              return loss, (loss, next_params)
            
            # Compute gradient and auxiliary values with respect to network parameters
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (aux_loss, next_params)), grads = grad_fn( net_params)
            
            return grads, loss, next_params
            
          grads, loss, next_params = compute_bc_gradients(self.theta)
          grads['bc_loss'] = loss
          if not parent.teacher_force:
            bc_opt_state = bc_opt.resume_init(opt_state=bc_opt_state,
                                              params=next_params,
                                              model_state=model_state,
                                              key=key)

          next_opt_state = AdafacMLPLOptState(
              params=bc_opt_state.params if parent.teacher_force else next_params,
              mom_rolling=next_mom_rolling,
              rms_rolling=next_rms_rolling,
              fac_rolling_features=next_fac_rolling_features,
              iteration=opt_state.iteration + 1,
              state=model_state,
              num_steps=opt_state.num_steps,
              bc_opt_state=bc_opt_state)

          return tree_utils.match_type(next_opt_state, opt_state), grads
        else:
          fun = functools.partial(mod_apply, self.theta['nn'], global_features)
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
              num_steps=opt_state.num_steps,
              bc_opt_state=None)

          return tree_utils.match_type(next_opt_state, opt_state)

    return _Opt(theta)
