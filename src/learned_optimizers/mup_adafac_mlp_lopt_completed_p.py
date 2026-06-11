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

"""MLP learned optimizer with adafactor features and CompletedP parameterization support.

This module implements a learned optimizer that is compatible with the CompletedP
parameterization for hyperparameter transfer across model scales (width, depth,
batch size, and dataset size).

Key features:
- Per-parameter learning rate scaling (from CompletedP)
- Per-parameter epsilon scaling (from CompletedP)  
- Per-parameter (1-β₁) and (1-β₂) scaling (from CompletedP)
- Decoupled weight decay with per-parameter scaling (from CompletedP)
- Learnable decays and weight decay coefficient

The scaling factors should be passed via model_state:
- mup_lr_scales: LR scales
- mup_eps_scales: Epsilon scales
- mup_wd_scales: Weight decay scales
- mup_one_minus_beta1_scales: (1-β₁) scales  
- mup_one_minus_beta2_scales: (1-β₂) scales
"""
import functools
from typing import Any, Optional
import collections

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

from functools import reduce
PRNGKey = jnp.ndarray


def second_moment_normalizer(x, axis, eps=1e-5):
  return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def second_moment_normalizer_scaled(x, axis, base_eps=1e-5, eps_scale=1.0):
  """Second moment normalizer with scaled epsilon for CompletedP."""
  scaled_eps = base_eps * eps_scale
  return x * lax.rsqrt(scaled_eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


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


def add_prefix(prefix, s):
    if prefix != '':
        prefix = prefix + '/'
    return prefix + s


def get_mup_lrs(state, prefix):
  d = {}
  for k, v in state.items():
      if is_leaf(v):
          d[add_prefix(prefix, k)] = v
      else:
          for kk, vv in get_mup_lrs(v, k).items():
              d[add_prefix(prefix, kk)] = vv
  
  d = {k.replace('/mup_lrs', ''): v for k, v in d.items()}
  return d


# Custom accumulator types for CompletedP scaling
ScaledMomAccumulator = collections.namedtuple("ScaledMomAccumulator", ["m", "t"])
ScaledRMSAccumulator = collections.namedtuple("ScaledRMSAccumulator", ["rms", "t"])


@flax.struct.dataclass
class CompletedPAdafacMLPLOptState:
  """State for the CompletedP learned optimizer.
  
  Attributes:
    params: Current model parameters
    state: Model state (includes mup scales)
    mom_rolling: Momentum accumulator (first moment)
    rms_rolling: RMS accumulator (second moment)  
    fac_rolling_features: Factored accumulator for Adafactor-style features
    num_steps: Total number of training steps
    iteration: Current iteration count
  """
  params: Any
  state: Any
  mom_rolling: ScaledMomAccumulator
  rms_rolling: ScaledRMSAccumulator
  fac_rolling_features: common.FactoredAccum
  num_steps: jnp.ndarray
  iteration: jnp.ndarray


def decay_to_param(x):
  """Convert decay value to learnable parameter space."""
  return jnp.log(1 - x) / 10.


def param_to_decay(x):
  """Convert learnable parameter to decay value."""
  return 1 - jnp.exp(x * 10.)


def wd_to_param(x):
  """Convert weight decay value to learnable parameter space (logspace).
  
  Maps weight decay λ to log(λ) / 10 for stable learning.
  Default weight decay of 0.001 maps to approximately -0.69.
  """
  return jnp.log(x) / 10.


def param_to_wd(x):
  """Convert learnable parameter to weight decay value.
  
  Inverse of wd_to_param: exp(x * 10).
  """
  return jnp.exp(x * 10.)


def scaled_rolling_mom_init(params, num_decays):
  """Initialize scaled momentum accumulator.
  
  Args:
    params: Parameter pytree to create momentum for
    num_decays: Number of decay values to track
    
  Returns:
    ScaledMomAccumulator with zero-initialized momentum
  """
  def init_one(p):
    return jnp.zeros(list(p.shape) + [num_decays], dtype=jnp.float32)
  
  return ScaledMomAccumulator(
      m=jax.tree_util.tree_map(init_one, params),
      t=jnp.asarray(0, dtype=jnp.int32))


def scaled_rolling_rms_init(params, num_decays):
  """Initialize scaled RMS accumulator.
  
  Args:
    params: Parameter pytree to create RMS for
    num_decays: Number of decay values to track
    
  Returns:
    ScaledRMSAccumulator with zero-initialized RMS
  """
  def init_one(p):
    return jnp.zeros(list(p.shape) + [num_decays], dtype=jnp.float32)
  
  return ScaledRMSAccumulator(
      rms=jax.tree_util.tree_map(init_one, params),
      t=jnp.asarray(0, dtype=jnp.int32))


def scaled_rolling_mom_update(state, grad, base_decays, one_minus_beta_scales):
  """Update momentum with per-parameter (1-β) scaling from CompletedP.
  
  Implements: m_new = effective_decay * m_old + effective_one_minus_decay * g
  where: effective_one_minus_decay = (1 - base_decay) * scale
         effective_decay = 1 - effective_one_minus_decay
  
  Args:
    state: Current ScaledMomAccumulator
    grad: Gradient pytree
    base_decays: Array of base decay values [num_decays]
    one_minus_beta_scales: Pytree of per-parameter (1-β) scaling factors
    
  Returns:
    Updated ScaledMomAccumulator
  """
  base_one_minus_decay = 1 - base_decays  # [num_decays]
  
  def update_one(m, g, scale):
    # scale is a scalar for this parameter
    # m has shape [param_shape..., num_decays]
    # g has shape [param_shape...]
    
    # Effective (1-decay) = base (1-decay) * scale
    effective_one_minus_decay = base_one_minus_decay * scale  # [num_decays]
    effective_decay = 1 - effective_one_minus_decay  # [num_decays]
    
    # Expand g to match m's shape
    g_expanded = jnp.expand_dims(g, axis=-1)  # [param_shape..., 1]
    
    # Update: m_new = decay * m + (1-decay) * g
    new_m = effective_decay * m + effective_one_minus_decay * g_expanded
    return new_m
  
  new_m = jax.tree_util.tree_map(update_one, state.m, grad, one_minus_beta_scales)
  return ScaledMomAccumulator(m=new_m, t=state.t + 1)


def scaled_rolling_rms_update(state, grad, base_decays, one_minus_beta_scales):
  """Update RMS (second moment) with per-parameter (1-β) scaling from CompletedP.
  
  Implements: rms_new = effective_decay * rms_old + effective_one_minus_decay * g²
  where: effective_one_minus_decay = (1 - base_decay) * scale
         effective_decay = 1 - effective_one_minus_decay
  
  Args:
    state: Current ScaledRMSAccumulator
    grad: Gradient pytree
    base_decays: Array of base decay values [num_decays]
    one_minus_beta_scales: Pytree of per-parameter (1-β₂) scaling factors
    
  Returns:
    Updated ScaledRMSAccumulator
  """
  base_one_minus_decay = 1 - base_decays  # [num_decays]
  
  def update_one(rms, g, scale):
    # scale is a scalar for this parameter
    # rms has shape [param_shape..., num_decays]
    # g has shape [param_shape...]
    
    # Effective (1-decay) = base (1-decay) * scale
    effective_one_minus_decay = base_one_minus_decay * scale  # [num_decays]
    effective_decay = 1 - effective_one_minus_decay  # [num_decays]
    
    # Clip decays for numerical stability
    effective_decay = jnp.clip(effective_decay, 0.0, 1.0)
    effective_one_minus_decay = 1 - effective_decay
    
    # Expand g² to match rms's shape
    g_sq_expanded = jnp.expand_dims(g * g, axis=-1)  # [param_shape..., 1]
    
    # Update: rms_new = decay * rms + (1-decay) * g²
    new_rms = effective_decay * rms + effective_one_minus_decay * g_sq_expanded
    return new_rms
  
  new_rms = jax.tree_util.tree_map(update_one, state.rms, grad, one_minus_beta_scales)
  return ScaledRMSAccumulator(rms=new_rms, t=state.t + 1)


def scaled_factored_rolling_update(state, grad, base_decays, one_minus_beta_scales, 
                                    epsilon=1e-30, local_epsilon=1e-9):
  """Update factored accumulator with per-parameter (1-β) scaling.
  
  This is the Adafactor-style factored second moment estimator with CompletedP scaling.
  
  Args:
    state: Current FactoredAccum
    grad: Gradient pytree
    base_decays: Array of base decay values [num_decays]
    one_minus_beta_scales: Pytree of per-parameter (1-β₂) scaling factors
    epsilon: Small constant for numerical stability in gradient squaring
    local_epsilon: Small constant for safe_rsqrt
    
  Returns:
    Tuple of (updated FactoredAccum, preconditioned gradients)
  """
  base_one_minus_decay = 1 - base_decays  # [num_decays]
  num_decays = len(base_decays)
  
  def update_one(v_col, v_row, v_full, g, scale):
    # scale is a scalar for this parameter
    grad_sqr = g * g + epsilon
    f_dims = common.factored_dims(g.shape)
    
    # Compute effective decays with CompletedP scaling
    effective_one_minus_decay = base_one_minus_decay * scale
    effective_decay = jnp.clip(1 - effective_one_minus_decay, 0.0, 1.0)
    mixing_rate = 1 - effective_decay
    
    if f_dims is not None:
      # Precondition with factored dimensions
      d1, d0 = f_dims
      
      # v_row has shape [reduced_shape, num_decays], need to update each decay
      def update_factored_single_decay(v_r, v_c, eff_decay, mix_rate):
        new_v_row = eff_decay * v_r + mix_rate * jnp.mean(grad_sqr, axis=d0)
        new_v_col = eff_decay * v_c + mix_rate * jnp.mean(grad_sqr, axis=d1)
        
        reduced_d1 = d1 - 1 if d1 > d0 else d1
        row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
        
        row_factor = common.safe_rsqrt(new_v_row / (row_col_mean + local_epsilon), epsilon=local_epsilon)
        col_factor = common.safe_rsqrt(new_v_col, epsilon=local_epsilon)
        y = (g * jnp.expand_dims(row_factor, axis=d0) * jnp.expand_dims(col_factor, axis=d1))
        
        return new_v_col, new_v_row, y
      
      # Vectorize over decays
      new_v_col, new_v_row, y = jax.vmap(
          update_factored_single_decay, in_axes=(-1, -1, 0, 0), out_axes=(-1, -1, -1)
      )(v_row, v_col, effective_decay, mixing_rate)
      
      # Return empty v_diag with correct shape (0, num_decays) to match init
      empty_v_diag = jnp.zeros((0, num_decays), dtype=jnp.float32)
      return new_v_col, new_v_row, empty_v_diag, y
      
    else:
      # Diagonal style preconditioner
      def update_diag_single_decay(v, eff_decay, mix_rate):
        new_v = eff_decay * v + mix_rate * grad_sqr
        y = g * common.safe_rsqrt(new_v + local_epsilon, epsilon=local_epsilon)
        return new_v, y
      
      # Vectorize over decays
      new_v, y = jax.vmap(
          update_diag_single_decay, in_axes=(-1, 0, 0), out_axes=(-1, -1)
      )(v_full, effective_decay, mixing_rate)
      
      # Return empty v_col/v_row with correct shape (0, num_decays) to match init
      empty_v = jnp.zeros((0, num_decays), dtype=jnp.float32)
      return empty_v, empty_v, new_v, y
  
  f_v_col, tree = jax.tree_util.tree_flatten(state.v_col)
  f_v_row = jax.tree_util.tree_leaves(state.v_row)
  f_v = jax.tree_util.tree_leaves(state.v_diag)
  f_g = jax.tree_util.tree_leaves(grad)
  f_scales = jax.tree_util.tree_leaves(one_minus_beta_scales)
  
  f_v_col_new, f_v_row_new, f_v_new, outs = zip(
      *[update_one(*args) for args in zip(f_v_col, f_v_row, f_v, f_g, f_scales)])
  
  next_state = common.FactoredAccum(
      v_col=jax.tree_util.tree_unflatten(tree, f_v_col_new),
      v_row=jax.tree_util.tree_unflatten(tree, f_v_row_new),
      v_diag=jax.tree_util.tree_unflatten(tree, f_v_new))
  
  return next_state, jax.tree_util.tree_unflatten(tree, outs)


@gin.configurable
class MuCompletedPAdafacMLPLOpt(lopt_base.LearnedOptimizer):
  """MLP based learned optimizer with CompletedP parameterization support.
  
  This optimizer extends the base Adafactor-style MLP learned optimizer with
  full CompletedP scaling support for hyperparameter transfer across:
  - Width (via per-layer fan_in based scaling)
  - Depth (via m_L^α scaling)
  - Batch size (via √(m_B/m_D) scaling)
  - Dataset size (via √(m_D/m_B) scaling)
  
  The scaling factors are passed via model_state and include:
  - mup_lr_scales: Per-parameter learning rate scales
  - mup_eps_scales: Per-parameter epsilon scales
  - mup_wd_scales: Per-parameter weight decay scales
  - mup_one_minus_beta1_scales: Per-parameter (1-β₁) scales
  - mup_one_minus_beta2_scales: Per-parameter (1-β₂) scales
  
  Args:
    exp_mult: Multiplier for magnitude output (default: 0.001)
    step_mult: Multiplier for step size (default: 0.001)
    hidden_size: Hidden layer size in the MLP (default: 4)
    hidden_layers: Number of hidden layers (default: 2)
    initial_momentum_decays: Initial momentum decay values (default: (0.9, 0.99, 0.999))
    initial_rms_decays: Initial RMS decay values (default: (0.999,))
    initial_adafactor_decays: Initial adafactor decay values (default: (0.9, 0.99, 0.999))
    initial_weight_decay: Initial weight decay value (default: 0.001)
    concat_weights: Whether to use concatenated weights path (default: True)
    make_separate_weights: Whether to make separate scalar weights (default: False)
    split_weights: Whether to split weights (default: False)
    clip_grad: Whether to clip gradients (default: False)
    mup_lrs: Deprecated, use model_state instead (default: None)
    zero_training_step_feature: Zero out training step feature (default: False)
    quantized: Quantization mode ('bf16', 'fp8', or None) (default: None)
    base_epsilon: Base epsilon value for numerical stability (default: 1e-6)
  """

  def __init__(self,
               exp_mult=0.001,
               step_mult=0.001,
               hidden_size=4,
               hidden_layers=2,
               initial_momentum_decays=(0.9, 0.99, 0.999),
               initial_rms_decays=(0.999,),
               initial_adafactor_decays=(0.9, 0.99, 0.999),
               initial_weight_decay=0.001,
               concat_weights=True,
               make_separate_weights=False,
               split_weights=False,
               clip_grad=False,
               clip_norm=1.0,
               mup_lrs=None,
               zero_training_step_feature=False,
               quantized=None,
               base_epsilon=1e-6):
    super().__init__()
    self._exp_mult = exp_mult
    self._step_mult = step_mult
    self._hidden_size = hidden_size
    self._hidden_layers = hidden_layers
    self._initial_momentum_decays = initial_momentum_decays
    self._initial_rms_decays = initial_rms_decays
    self._initial_adafactor_decays = initial_adafactor_decays
    self._initial_weight_decay = initial_weight_decay
    self._concat_weights = concat_weights
    self._make_separate_weights = make_separate_weights
    self._split_weights = split_weights
    self.clip_grad = clip_grad
    self.clip_norm = clip_norm
    self.mup_lrs = mup_lrs
    self.quantized = quantized
    self.zero_training_step_feature = zero_training_step_feature
    self._base_epsilon = base_epsilon

    self._mod_init, self._mod_apply = hk.without_apply_rng(
        hk.transform(self._mod))
    


  @jax.default_matmul_precision("bfloat16")
  def _mod(self, global_feat, p, g, m, rms, fac_g, fac_vec_col, fac_vec_row,
           fac_vec_v, mup_lr_scale, mup_eps_scale, mup_wd_scale, weight_decay_value):
    """Compute the parameter update using an MLP.
    
    This function computes the update for a single parameter using:
    1. Features from gradients, momentum, RMS, and factored accumulators
    2. An MLP to predict direction and magnitude
    3. CompletedP scaling for LR, epsilon, and weight decay
    
    Args:
      global_feat: Dictionary of global features (iteration, num_steps, training_step_feature)
      p: Parameter value
      g: Gradient value
      m: Momentum values [param_shape..., num_momentum_decays]
      rms: RMS values [param_shape..., num_rms_decays]
      fac_g: Factored gradient features
      fac_vec_col: Factored column vectors
      fac_vec_row: Factored row vectors
      fac_vec_v: Factored diagonal vectors (for non-factored params)
      mup_lr_scale: Learning rate scale from CompletedP (scalar)
      mup_eps_scale: Epsilon scale from CompletedP (scalar)
      mup_wd_scale: Weight decay scale from CompletedP (scalar)
      weight_decay_value: Base weight decay value (scalar, same for all params)
      
    Returns:
      Updated parameter value
    """
    # Handle scalar parameters by reshaping
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
      mup_eps_scale = jnp.expand_dims(mup_eps_scale, 0)
      mup_wd_scale = jnp.expand_dims(mup_wd_scale, 0)
      did_reshape = True
    else:
      did_reshape = False
      
    # Scale epsilon for CompletedP
    # Base epsilon scaled by per-parameter eps_scale
    scaled_eps = self._base_epsilon * mup_eps_scale
    
    inps = []

    inps.append(jnp.expand_dims(g, axis=-1))
    inps.append(jnp.expand_dims(p, axis=-1))
    inps.append(m)
    inps.append(rms)
    
    # Use scaled epsilon for rsqrt operations
    rsqrt = lax.rsqrt(rms + scaled_eps[..., None])  # Broadcast eps_scale to match rms shape
    inps.append(m * rsqrt)
    inps.append(rsqrt)
    inps.append(fac_g)

    factored_dims = common.factored_dims(g.shape)
    if factored_dims is not None:
      # Construct features for factored case
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

      # 1/sqrt with scaled epsilon
      inps.append(lax.rsqrt(row_feat + scaled_eps[..., None]))
      inps.append(lax.rsqrt(col_feat + scaled_eps[..., None]))

      # multiplied by momentum
      reduced_d1 = d1 - 1 if d1 > d0 else d1
      row_col_mean = jnp.mean(fac_vec_row, axis=reduced_d1, keepdims=True)

      row_factor = common.safe_rsqrt(fac_vec_row / (row_col_mean + scaled_eps[..., None]))
      col_factor = common.safe_rsqrt(fac_vec_col)
      fac_mom_mult = (
          m * jnp.expand_dims(row_factor, axis=d0) *
          jnp.expand_dims(col_factor, axis=d1))
      inps.append(fac_mom_mult)
    else:
      # In the non-factored case, match what RMSProp does.
      inps.append(fac_vec_v)
      inps.append(fac_vec_v)

      # Use scaled epsilon
      inps.append(lax.rsqrt(fac_vec_v + scaled_eps[..., None]))
      inps.append(lax.rsqrt(fac_vec_v + scaled_eps[..., None]))

      fac_mom_mult = m * (fac_vec_v + scaled_eps[..., None])**-0.5
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

    # Compute MLP output
    if self._concat_weights:
      # concat the inputs, normalize
      inp_stack = jnp.concatenate(inps, axis=-1)
      axis = list(range(len(p.shape)))
      
      # Use scaled epsilon in second moment normalizer
      inp_stack = second_moment_normalizer_scaled(inp_stack, axis=axis, 
                                                   base_eps=1e-5, 
                                                   eps_scale=jnp.mean(mup_eps_scale))

      # add features that should not be normalized
      training_step_feature = global_feat["training_step_feature"]
      stacked = jnp.reshape(training_step_feature, [1] * len(axis) +
                            list(training_step_feature.shape[-1:]))
      stacked = jnp.tile(stacked, list(p.shape) + [1])
      inp_stack = jnp.concatenate([inp_stack, stacked], axis=-1)



      # Manually run the neural network.
      net = inp_stack
      for wi, (w, b) in enumerate(zip(weights, biases)):
        o_tmp = net @ w
        net = o_tmp + jnp.broadcast_to(b, list(net.shape[0:-1]) + [w.shape[-1]])

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
      # Use scaled epsilon
      mean_eps_scale = jnp.mean(mup_eps_scale)
      inp = [
          x * lax.rsqrt(1e-5 * mean_eps_scale + jnp.mean(jnp.square(x), keepdims=True))
          for x in flat_features
      ]
      
      for wi, (w, b) in enumerate(zip(weights, biases)):
        grids = []

        # hidden layer wi
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

        # bias
        inp = []
        for oi, net in enumerate(out_mul):
          inp.append(net + b[oi])

        # activation
        if wi != len(weights) - 1:
          inp = [jax.nn.relu(x) for x in inp]

      direction = inp[0]
      magnitude = inp[1]

    # Compute step from MLP output
    step = direction * jnp.exp(magnitude * self._exp_mult) * self._step_mult
    step = step.reshape(p.shape)
    
    # Apply weight decay with CompletedP scaling
    # Decoupled weight decay: new_p = p - lr_scale * (step + wd * wd_scale * p)
    # This follows the AdamW formulation where WD is applied after the adaptive step
    effective_wd = weight_decay_value * mup_wd_scale
    wd_term = effective_wd * p
    
    # Final update with LR scaling
    new_p = p - (step + wd_term) * mup_lr_scale

    if did_reshape:
      new_p = jnp.squeeze(new_p, 0)

    # Log metrics
    avg_step_size = jnp.mean(jnp.abs(step))
    summary.summary("completedp_adafac_mlp_lopt/avg_step_size", avg_step_size)
    summary.summary(
        "completedp_adafac_mlp_lopt/avg_step_size_hist",
        avg_step_size,
        aggregation="collect")
    summary.summary("completedp_adafac_mlp_lopt/direction/mean_abs",
                    jnp.mean(jnp.abs(direction)))
    summary.summary("completedp_adafac_mlp_lopt/magnitude/mean_abs",
                    jnp.mean(jnp.abs(magnitude)))
    summary.summary("completedp_adafac_mlp_lopt/magnitude/mean", jnp.mean(magnitude))
    summary.summary("completedp_adafac_mlp_lopt/grad/mean_abs", jnp.mean(jnp.abs(g)))
    summary.summary("completedp_adafac_mlp_lopt/effective_wd/mean", jnp.mean(effective_wd))

    return new_p

  def init(self, key: PRNGKey) -> lopt_base.MetaParams:
    """Initialize the learned optimizer's meta-parameters.
    
    We meta-learn:
    - Weights of the MLP
    - Decays of momentum, RMS, and adafactor style accumulators
    - Weight decay coefficient (in logspace)
    
    Returns:
      Dictionary of meta-parameters
    """
    training_step_feature = tanh_embedding(1)
    global_features = {
        "iterations": 0,
        "num_steps": 10,
        "training_step_feature": training_step_feature,
    }
    # fake weights with 2 dimensions
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
    
    # Dummy scales (all ones for init)
    # NOTE: Scales are per-parameter scalars, not per-element arrays
    mup_lr_scale = jnp.array(1.0)
    mup_eps_scale = jnp.array(1.0)
    mup_wd_scale = jnp.array(1.0)
    weight_decay_value = jnp.array(self._initial_weight_decay)
    
    mod_theta = self._mod_init(key, global_features, p, g, m, rms, fac_g,
                               fac_vec_col, fac_vec_row, fac_vec_v,
                               mup_lr_scale, mup_eps_scale, mup_wd_scale, 
                               weight_decay_value)
    
    return hk.data_structures.to_haiku_dict({
        "momentum_decays": jnp.zeros([len(self._initial_momentum_decays)]),
        "rms_decays": jnp.zeros([len(self._initial_rms_decays)]),
        "adafactor_decays": jnp.zeros([len(self._initial_adafactor_decays)]),
        "weight_decay": jnp.zeros([]),  # Learnable WD offset in logspace
        "nn": mod_theta
    })
  
  def opt_fn(self,
             theta: lopt_base.MetaParams,
             is_training: Optional[bool] = False) -> opt_base.Optimizer:
    """Create an optimizer instance from the meta-parameters.
    
    Args:
      theta: Learned meta-parameters
      is_training: Whether in training mode
      
    Returns:
      Optimizer instance
    """
    mod_apply = self._mod_apply
    parent = self

    class _Opt(opt_base.Optimizer):
      """Optimizer capturing the meta params with CompletedP scaling support."""

      def __init__(self, theta):
        self.theta = theta
        self.mup_lrs = None

      def _get_base_decays(self):
        """Get the base decay values with learned adjustments."""
        mom_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_momentum_decays)) +
            self.theta["momentum_decays"])

        rms_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_rms_decays)) +
            self.theta["rms_decays"])

        adafactor_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_adafactor_decays)) +
            self.theta["adafactor_decays"])
        
        return mom_decay, rms_decay, adafactor_decay
      
      def _get_weight_decay(self):
        """Get the weight decay value with learned adjustment."""
        # WD is learned in logspace similar to decays
        base_wd = parent._initial_weight_decay
        learned_offset = self.theta["weight_decay"]
        # Convert: base_wd in logspace + learned offset, then back to linear
        return param_to_wd(wd_to_param(base_wd) + learned_offset)

      def init(
          self,
          params: opt_base.Params,
          model_state: Optional[opt_base.ModelState] = None,
          num_steps: Optional[int] = None,
          key: Optional[PRNGKey] = None,
      ) -> CompletedPAdafacMLPLOptState:
        """Initialize optimizer state.
        
        Args:
          params: Model parameters
          model_state: Model state (should contain mup scales)
          num_steps: Total number of training steps
          key: Random key (unused)
          
        Returns:
          Initial optimizer state
        """
        if num_steps is None:
          raise ValueError("Must specify number of steps for this lopt!")

        num_mom_decays = len(parent._initial_momentum_decays)
        num_rms_decays = len(parent._initial_rms_decays)
        num_adafactor_decays = len(parent._initial_adafactor_decays)

        # Initialize accumulators
        mom_rolling = scaled_rolling_mom_init(params, num_mom_decays)
        rms_rolling = scaled_rolling_rms_init(params, num_rms_decays)
        
        # Use BASE decay values for init (NOT self._get_base_decays() which accesses
        # self.theta and would capture tracers when init is called inside JIT).
        # The learned adjustments to decays are zero at init time anyway.
        base_adafactor_decays = jnp.asarray(parent._initial_adafactor_decays)
        fac_vec_roll = common.vec_factored_rolling(base_adafactor_decays)
        fac_rolling = fac_vec_roll.init(params)

        return CompletedPAdafacMLPLOptState(
            params=params,
            state=model_state,
            rms_rolling=rms_rolling,
            mom_rolling=mom_rolling,
            fac_rolling_features=fac_rolling,
            iteration=jnp.asarray(0, dtype=jnp.int32),
            num_steps=jnp.asarray(num_steps))
            
      def update(self,
                 opt_state: CompletedPAdafacMLPLOptState,
                 grad: opt_base.Gradient,
                 loss: jnp.ndarray,
                 model_state: Optional[opt_base.ModelState] = None,
                 is_valid: bool = False,
                 key: Optional[PRNGKey] = None) -> CompletedPAdafacMLPLOptState:
        """Update parameters using the learned optimizer with CompletedP scaling.
        
        Args:
          opt_state: Current optimizer state
          grad: Gradients
          loss: Current loss value
          model_state: Model state containing mup scaling factors:
            - mup_lr_scales: LR scales
            - mup_eps_scales: Epsilon scales  
            - mup_wd_scales: Weight decay scales
            - mup_one_minus_beta1_scales: (1-β₁) scales
            - mup_one_minus_beta2_scales: (1-β₂) scales
          is_valid: Whether the update is valid
          key: Random key (unused)
          
        Returns:
          Updated optimizer state
        """
        if parent.clip_grad:
          clip_norm = parent.clip_norm
          clipping = optax.clip_by_global_norm(clip_norm)
          grad, _ = clipping.update(grad, None)

        # Extract CompletedP scales from model_state (fall back to scalar ones if not using MuP)
        ones = jax.tree_util.tree_map(lambda x: jnp.ones(()), grad)
        lr_scales = model_state.get('mup_lr_scales', ones)
        eps_scales = model_state.get('mup_eps_scales', ones)
        wd_scales = model_state.get('mup_wd_scales', ones)
        one_minus_beta1_scales = model_state.get('mup_one_minus_beta1_scales', ones)
        one_minus_beta2_scales = model_state.get('mup_one_minus_beta2_scales', ones)

        # Handle NaN gradients
        grad = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grad)

        # Get base decays (with learned adjustments)
        mom_decay, rms_decay, adafactor_decay = self._get_base_decays()
        
        # Get weight decay value
        weight_decay_value = self._get_weight_decay()

        # Update accumulators with CompletedP beta scaling
        # Momentum uses (1-β₁) scaling
        next_mom_rolling = scaled_rolling_mom_update(
            opt_state.mom_rolling, grad, mom_decay, one_minus_beta1_scales)
        
        # RMS uses (1-β₂) scaling
        next_rms_rolling = scaled_rolling_rms_update(
            opt_state.rms_rolling, grad, rms_decay, one_minus_beta2_scales)
        
        # Factored accumulator uses (1-β₂) scaling
        next_fac_rolling_features, fac_g = scaled_factored_rolling_update(
            opt_state.fac_rolling_features, grad, adafactor_decay, 
            one_minus_beta2_scales)

        # Compute training step features
        training_step_feature = tanh_embedding(opt_state.iteration)

        if parent.zero_training_step_feature:
          training_step_feature = jnp.zeros_like(training_step_feature)
          
        global_features = {
            "iterations": opt_state.iteration,
            "num_steps": opt_state.num_steps,
            "training_step_feature": training_step_feature,
        }

        # Create partial function for tree_map
        fun = functools.partial(mod_apply, self.theta["nn"], global_features)

        # Apply the learned optimizer to each parameter
        # Pass all CompletedP scales and weight decay value
        next_params = jax.tree_util.tree_map(
            fun, 
            opt_state.params, 
            grad,
            next_mom_rolling.m,
            next_rms_rolling.rms, 
            fac_g,
            next_fac_rolling_features.v_col,
            next_fac_rolling_features.v_row,
            next_fac_rolling_features.v_diag,
            lr_scales,
            eps_scales,
            wd_scales,
            jax.tree_util.tree_map(lambda x: weight_decay_value, lr_scales)  # Broadcast WD
        )

        # Preserve the original state structure while updating with new model_state
        # This ensures all keys from opt_state.state are preserved
        if model_state is not None and opt_state.state is not None:
          # Merge: start with old state, update with new model_state values
          merged_state = dict(opt_state.state)
          merged_state.update(model_state)
        elif model_state is not None:
          merged_state = model_state
        else:
          merged_state = opt_state.state

        next_opt_state = CompletedPAdafacMLPLOptState(
            params=next_params,
            mom_rolling=next_mom_rolling,
            rms_rolling=next_rms_rolling,
            fac_rolling_features=next_fac_rolling_features,
            iteration=opt_state.iteration + 1,
            state=merged_state,
            num_steps=opt_state.num_steps)

        return tree_utils.match_type(next_opt_state, opt_state)

    return _Opt(theta)
