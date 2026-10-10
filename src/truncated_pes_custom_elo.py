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

"""A vetorized, truncated, PES based gradient estimator."""
import dataclasses
import functools
from typing import Any, Mapping, Optional, Sequence, Tuple

import flax
from flax import jax_utils as flax_jax_utils
import gin
import haiku as hk
import jax
from jax import lax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from learned_optimization import jax_utils
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod
import pprint
import flax.serialization as serialization
from jax.experimental import io_callback
from lopt_truncated_step_elo import TruncatedUnrollOut
import os
import numpy as np

PRNGKey = jnp.ndarray
MetaParams = Any
TruncatedUnrollState = Any

def second_moment_normalizer(x, eps=1e-8):
  return x * lax.rsqrt(eps+jnp.mean(jnp.square(x), keepdims=True))


def masked_second_moment_normalizer(x, mask, eps=1e-8):
  """RMS-normalize x using only the entries where mask is set.

  The unmasked variant averages over the whole [steps, num_tasks] tensor,
  including the reset steps that carry a hard 0. Those zeros shrink the mean
  square, so the resulting scale drifts with the mask density -- which itself
  moves with the inner-length schedule (a log-uniform length makes it move a
  lot more than a fixed one). eps must stay > 0: with eps=0 an all-zero tensor
  (e.g. the very first unroll, where every step is a reset) gives
  rsqrt(0) = inf and 0 * inf = NaN, which poisons the whole meta-gradient.
  """
  m = mask.astype(jnp.float32)
  denom = jnp.maximum(jnp.sum(m), jnp.float32(1.0))
  mean_sq = jnp.sum(jnp.square(x) * m) / denom
  return x * lax.rsqrt(mean_sq + eps)


def _meta_grad_snr(vec_es_grad, es_grad):
  """Global vector SNR of the meta-gradient (fp32 accumulation).

  signal = Σ_leaves ||ḡ_leaf||²
  noise  = Σ_leaves E_t[||g_t - ḡ||²]      (mean over task axis)
  snr_sample    = signal / noise           (per-sample SNR; N-independent)
  snr_estimator = N · snr_sample           (SNR of the mean-of-N estimator)
  """
  per_task_leaves = jax.tree_util.tree_leaves(vec_es_grad)
  mean_leaves = jax.tree_util.tree_leaves(es_grad)
  N = per_task_leaves[0].shape[0]

  def leaf_sig(mean_g):
    return jnp.sum(jnp.square(mean_g.astype(jnp.float32)))

  def leaf_noise(per_task, mean_g):
    pt = per_task.astype(jnp.float32)
    mg = mean_g.astype(jnp.float32)
    diff = pt - mg[None]
    return jnp.mean(jnp.sum(jnp.square(diff.reshape(N, -1)), axis=-1))

  sigs = [leaf_sig(m) for m in mean_leaves]
  noises = [leaf_noise(p, m) for p, m in zip(per_task_leaves, mean_leaves)]
  sig_sum = functools.reduce(jnp.add, sigs)
  noise_sum = functools.reduce(jnp.add, noises)

  snr_sample = sig_sum / (noise_sum + jnp.float32(1e-12))
  snr_estimator = jnp.float32(N) * snr_sample
  return snr_sample, snr_estimator


def _delta_loss_snr(delta_losses, mask):
  """Mask-weighted SNR of delta_losses on the full [steps, num_tasks] tensor.

  abs_mean / std — scale-invariant, picks up signal-shape stability rather
  than absolute Δℓ magnitude. Masked-out positions are zeroed so upstream
  NaN/garbage at reset steps doesn't poison the statistics.
  """
  m = mask.astype(jnp.float32)
  dl = jnp.where(m > 0, delta_losses, jnp.float32(0.0)).astype(jnp.float32)
  denom = jnp.maximum(jnp.sum(m), jnp.float32(1.0))

  abs_mean = jnp.sum(jnp.abs(dl)) / denom
  mean = jnp.sum(dl) / denom
  var = jnp.sum(jnp.square(dl - mean) * m) / denom
  std = jnp.sqrt(var + jnp.float32(1e-12))
  return abs_mean / (std + jnp.float32(1e-12))


def _write_per_task_if(buffer_tree, value_tree, slot_vec, push_flag_vec):
  """Per-task conditional write into buffer[t, slot[t]] (vmap over task axis).

  Non-push tasks write back the current slot value — a XLA no-op update.
  buffer leaves: (num_tasks, buffer_size, ...); value leaves: (num_tasks, ...).
  """
  def per_task(buf, val, slot, flag):
    def one_leaf(b, v):
      curr = jax.lax.dynamic_index_in_dim(b, slot, axis=0, keepdims=False)
      new = jnp.where(flag, v, curr)
      return jax.lax.dynamic_update_index_in_dim(b, new, slot, axis=0)
    return jax.tree_util.tree_map(one_leaf, buf, val)
  return jax.vmap(per_task)(buffer_tree, value_tree, slot_vec, push_flag_vec)


def _push_buffers_core(
    p_state, n_state,
    snap_pos_opt, snap_pos_tp, snap_neg_opt, snap_neg_tp,
    snap_mean_max_cumsum,
    any_done,
    buffer_size,
):
  """Post-truncation push decision + buffer write (pure jnp/vmap).

  V_n accumulates positive loss improvement (no clamp), so within a trajectory
  max V is reached exactly at the loss minimum and is monotone non-decreasing.
  improved = post mean(pos/neg max_cumulative_sum) > pre snapshot mean
           <=> a new loss low occurred inside this truncation window.
  On improved: overwrite the CURRENT slot with the pre-window snapshot.
  The slot only advances when the trajectory ends (any_done), so all pushes of
  one full unroll target one slot; its final content is the snapshot taken one
  window before the trajectory's loss minimum, and a buffer of size B holds
  states from B distinct full unrolls.
  """
  post_mean = 0.5 * (p_state.max_cumulative_sum + n_state.max_cumulative_sum)
  push = post_mean > snap_mean_max_cumsum

  slots = p_state.next_write_slot
  new_pos_sb  = _write_per_task_if(p_state.state_buffer,      snap_pos_opt, slots, push)
  new_pos_tpb = _write_per_task_if(p_state.task_param_buffer, snap_pos_tp,  slots, push)
  new_neg_sb  = _write_per_task_if(n_state.state_buffer,      snap_neg_opt, slots, push)
  new_neg_tpb = _write_per_task_if(n_state.task_param_buffer, snap_neg_tp,  slots, push)

  # Advance the write slot on trajectory end, not on push: the next full
  # unroll writes into a fresh slot.
  new_slots = jnp.where(any_done, (slots + 1) % buffer_size, slots)

  return (
      p_state.replace(state_buffer=new_pos_sb, task_param_buffer=new_pos_tpb,
                      next_write_slot=new_slots),
      n_state.replace(state_buffer=new_neg_sb, task_param_buffer=new_neg_tpb,
                      next_write_slot=new_slots),
  )


_push_buffers_jit = jax.jit(_push_buffers_core, static_argnames=("buffer_size",))
_push_buffers_pmap = jax.pmap(
    _push_buffers_core,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None),
    static_broadcasted_argnums=(8,))


@flax.struct.dataclass
class PESWorkerState(gradient_learner.GradientEstimatorState):
  pos_state: TruncatedUnrollState
  neg_state: TruncatedUnrollState
  accumulator: MetaParams

@functools.partial(jax.jit, static_argnames=("std", "timer_obj", "sign_delta_loss_scalar", "samples_per_device", "device_idx", "imt_loss_type"))
def compute_pes_grad(
    dirloss_weight: jnp.ndarray,
    magloss_weight: jnp.ndarray,
    p_yses: Sequence[TruncatedUnrollOut],
    n_yses: Sequence[TruncatedUnrollOut],
    accumulator: MetaParams,
    vec_pos: MetaParams,
    std: float,
    timer_obj: Any,
    sign_delta_loss_scalar: Optional[float] = None,
    delta_loss_scalar_afsnm: Optional [float] = 0.01,
    samples_per_device: Optional[int] = None,
    device_idx: Optional[int] = None,
    imt_loss_type: str = "decoupled",
):
  """Compute the PES gradient estimate from the outputs of many unrolls.

  Args:
    p_yses: Sequence of PES outputs from the positive perturbation.
    n_yses: Sequence of PES outputs from the negative perturbation.
    accumulator: Current PES accumulator from the last iteration.
    vec_pos: Positive perturbations used to compute the current unroll.
    std: Standard deviation of pertrubations used.
    sign_delta_loss_scalar: Optional, if specified the sign of the delta loss
      multiplied by this value is used instead of the real delta_loss

  Returns:
    loss: the mean loss.
    es_grad: the grad estimate.
    new_accumulator: the new accumulator value.
    delta_loss: the difference in positive and negative losses.

  """
  

  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  with timer_obj("PES Gather", []):
    pass

  p_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  delta_direction_ori = p_ys.imt_cosine_loss - n_ys.imt_cosine_loss
  delta_magnitude_ori = p_ys.imt_magnitude_loss - n_ys.imt_magnitude_loss
  delta_mse_ori = p_ys.imt_mse_loss - n_ys.imt_mse_loss
  delta_direction = masked_second_moment_normalizer(
      delta_direction_ori, p_ys.mask) * delta_loss_scalar_afsnm
  delta_magnitude = masked_second_moment_normalizer(
      delta_magnitude_ori, p_ys.mask) * delta_loss_scalar_afsnm
  delta_mse = masked_second_moment_normalizer(
      delta_mse_ori, p_ys.mask) * delta_loss_scalar_afsnm
  delta_task_losses = p_ys.task_loss - n_ys.task_loss
  if imt_loss_type == "mse":
    # Plain-MSE baseline. The imitation-vs-task split is kept identical to the
    # decoupled objective by reusing the sum of the two (already scheduled)
    # weights, so the two runs differ only in the shape of the imitation term.
    imt_weight = dirloss_weight + magloss_weight
    delta_losses = imt_weight * delta_mse + (1.0 - imt_weight) * delta_task_losses
  elif imt_loss_type == "decoupled":
    delta_losses = dirloss_weight * delta_direction + magloss_weight * delta_magnitude + (1.0 - dirloss_weight - magloss_weight) * delta_task_losses
  else:
    raise ValueError(f"imt_loss_type must be 'decoupled' or 'mse', got {imt_loss_type!r}")

  if sign_delta_loss_scalar:
    sign_per_task = jnp.sign(jnp.mean(delta_losses * p_ys.mask, axis=0))
    delta_losses = jnp.ones_like(delta_losses) * sign_per_task * sign_delta_loss_scalar

  has_finished = lax.cumsum(jnp.asarray(p_ys.is_done, dtype=jnp.int32)) > 0
  # p_ys is of the form [sequence, n_tasks]
  denom = jnp.sum(p_ys.mask, axis=0)

  #initially zero for ONLY RNN and NOT MLP because delta_losses is zero
  last_unroll_loss = jnp.sum(
      delta_losses * (1.0 - has_finished) * p_ys.mask, axis=0) / denom

  new_unroll_loss = jnp.sum(
      delta_losses * has_finished * p_ys.mask, axis=0) / denom

  factor = 1.0 / (2 * std**2)


  # initially NON-ZERO for RNN and MLP
  accumulator = tree_utils.tree_add(vec_pos, accumulator)

  num_tasks = last_unroll_loss.shape[0]

  def reshape_to(loss, p):
    return loss.reshape((num_tasks,) + (1,) * (len(p.shape) - 1)) * factor * p

  #initially zero for ONLY RNN and NOT MLP
  es_grad_from_accum = jax.tree_util.tree_map(
      functools.partial(reshape_to, last_unroll_loss), accumulator)

  #initially zero for both RNN and MLP
  es_grad_from_new_perturb = jax.tree_util.tree_map(
      functools.partial(reshape_to, new_unroll_loss), vec_pos)

  #initially zero for ONLY RNN and NOT MLP
  vec_es_grad = jax.tree_util.tree_map(lambda a, b: a + b, es_grad_from_accum,
                                       es_grad_from_new_perturb)

  #initially zero for ONLY RNN and NOT MLP
  es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  snr_sample, snr_estimator = _meta_grad_snr(vec_es_grad, es_grad)
  snr_delta_loss = _delta_loss_snr(delta_losses, p_ys.mask)

  def _switch_one_accum(a, b):
    shape = [num_tasks] + [1] * (len(a.shape) - 1)
    return jnp.where(jnp.reshape(has_finished[-1], shape), a, b)

  new_accumulator = jax.tree_util.tree_map(_switch_one_accum, vec_pos,
                                           accumulator)

  pos_task_loss = jnp.sum(p_ys.task_loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
  neg_task_loss = jnp.sum(n_ys.task_loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)

  return (jnp.mean((pos_task_loss + neg_task_loss) / 2.0), es_grad,
          new_accumulator, p_ys, delta_losses, delta_task_losses,
          delta_direction_ori, delta_magnitude_ori, delta_mse_ori,
          snr_sample, snr_estimator, snr_delta_loss)

@functools.partial(jax.jit, static_argnames=("std", "sign_delta_loss_scalar", "replicated", "imt_loss_type"))
def _pes_grad_sharded_inner_elo(dirloss_weight, magloss_weight, p_ys, n_ys, accumulator, vec_pos, std,
                                sign_delta_loss_scalar, replicated, delta_loss_scalar_afsnm=0.01,
                                imt_loss_type="decoupled"):
  """JIT-compiled inner function: all-gather via sharding constraint, then PES gradient (ELO variant)."""
  # All-gather: replicate all sharded arrays across devices
  p_ys = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), p_ys)
  n_ys = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), n_ys)
  accumulator = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), accumulator)
  vec_pos = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), vec_pos)

  # Separate direction and magnitude IMT losses, normalize each independently
  delta_direction_ori = p_ys.imt_cosine_loss - n_ys.imt_cosine_loss
  delta_magnitude_ori = p_ys.imt_magnitude_loss - n_ys.imt_magnitude_loss
  delta_mse_ori = p_ys.imt_mse_loss - n_ys.imt_mse_loss
  delta_direction = masked_second_moment_normalizer(
      delta_direction_ori, p_ys.mask) * delta_loss_scalar_afsnm
  delta_magnitude = masked_second_moment_normalizer(
      delta_magnitude_ori, p_ys.mask) * delta_loss_scalar_afsnm
  delta_mse = masked_second_moment_normalizer(
      delta_mse_ori, p_ys.mask) * delta_loss_scalar_afsnm
  delta_task_losses = p_ys.task_loss - n_ys.task_loss
  if imt_loss_type == "mse":
    # Plain-MSE baseline. The imitation-vs-task split is kept identical to the
    # decoupled objective by reusing the sum of the two (already scheduled)
    # weights, so the two runs differ only in the shape of the imitation term.
    imt_weight = dirloss_weight + magloss_weight
    delta_losses = imt_weight * delta_mse + (1.0 - imt_weight) * delta_task_losses
  elif imt_loss_type == "decoupled":
    delta_losses = dirloss_weight * delta_direction + magloss_weight * delta_magnitude + (1.0 - dirloss_weight - magloss_weight) * delta_task_losses
  else:
    raise ValueError(f"imt_loss_type must be 'decoupled' or 'mse', got {imt_loss_type!r}")

  if sign_delta_loss_scalar:
    sign_per_task = jnp.sign(jnp.mean(delta_losses * p_ys.mask, axis=0))
    delta_losses = jnp.ones_like(
        delta_losses) * sign_per_task * sign_delta_loss_scalar

  has_finished = lax.cumsum(jnp.asarray(p_ys.is_done, dtype=jnp.int32)) > 0

  denom = jnp.sum(p_ys.mask, axis=0)

  last_unroll_loss = jnp.sum(
      delta_losses * (1.0 - has_finished) * p_ys.mask, axis=0) / denom

  new_unroll_loss = jnp.sum(
      delta_losses * has_finished * p_ys.mask, axis=0) / denom

  factor = 1.0 / (2 * std**2)

  accumulator = tree_utils.tree_add(vec_pos, accumulator)

  num_tasks = last_unroll_loss.shape[0]

  def reshape_to(loss, p):
    return loss.reshape((num_tasks,) + (1,) * (len(p.shape) - 1)) * factor * p

  es_grad_from_accum = jax.tree_util.tree_map(
      functools.partial(reshape_to, last_unroll_loss), accumulator)

  es_grad_from_new_perturb = jax.tree_util.tree_map(
      functools.partial(reshape_to, new_unroll_loss), vec_pos)

  vec_es_grad = jax.tree_util.tree_map(lambda a, b: a + b, es_grad_from_accum,
                                       es_grad_from_new_perturb)

  es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  # SNR computed inside the jit (and inside the all-gather replication) so the
  # task axis is global. Outside this function vec_es_grad is sliced per device.
  snr_sample, snr_estimator = _meta_grad_snr(vec_es_grad, es_grad)
  snr_delta_loss = _delta_loss_snr(delta_losses, p_ys.mask)

  def _switch_one_accum(a, b):
    shape = [num_tasks] + [1] * (len(a.shape) - 1)
    return jnp.where(jnp.reshape(has_finished[-1], shape), a, b)

  new_accumulator = jax.tree_util.tree_map(_switch_one_accum, vec_pos,
                                           accumulator)

  pos_task_loss = jnp.sum(p_ys.task_loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
  neg_task_loss = jnp.sum(n_ys.task_loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)

  return (
      jnp.mean((pos_task_loss + neg_task_loss) / 2.0),
      es_grad,
      new_accumulator,
      p_ys,
      delta_losses,
      delta_task_losses,
      delta_direction_ori,
      delta_magnitude_ori,
      delta_mse_ori,
      snr_sample,
      snr_estimator,
      snr_delta_loss,
  )




def compute_pes_grad_sharded_elo(
    dirloss_weight: jnp.ndarray,
    magloss_weight: jnp.ndarray,
    p_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    n_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    accumulator: MetaParams,
    vec_pos: MetaParams,
    std: float,
    timer_obj: Any,
    sign_delta_loss_scalar: Optional[float] = None,
    samples_per_device: int = 1,
    device_idx: int = 0,
    baseline_losses: Optional[list[float]] = None,
    delta_loss_scalar_afsnm: Optional[float] = 0.01,
    imt_loss_type: str = "decoupled",
    mesh: Optional[Mesh] = None,
):
  """Compute PES gradient using JAX mesh-based sharding for multi-GPU all-gather (ELO variant).

  Replaces the deprecated compute_pes_grad_pmap which used jax.pmap + jax.lax.all_gather
  and breaks with modern JAX's jax.distributed.initialize().
  """
  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  # Stack and flatten locally
  p_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  # Set up shardings
  replicated = NamedSharding(mesh, P())
  sharding_axis1 = NamedSharding(mesh, P(None, 'devices'))
  sharding_axis0 = NamedSharding(mesh, P('devices'))

  num_devices = jax.device_count()
  local_device = jax.local_devices()[0]

  def _to_single_device(arr):
    """Ensure arr is a single-device array on the local device."""
    if isinstance(arr, jax.Array) and not arr.is_fully_addressable:
      return arr.addressable_data(0)
    return jax.device_put(arr, local_device)

  def make_global_axis1(local_arr):
    """[steps, local_tasks] -> global [steps, total_tasks] sharded on axis 1."""
    global_shape = list(local_arr.shape)
    global_shape[1] = global_shape[1] * num_devices
    local_arr = _to_single_device(local_arr)
    return jax.make_array_from_single_device_arrays(
        tuple(global_shape), sharding_axis1, [local_arr])

  def make_global_axis0(local_arr):
    """[local_tasks, ...] -> global [total_tasks, ...] sharded on axis 0."""
    global_shape = list(local_arr.shape)
    global_shape[0] = global_shape[0] * num_devices
    local_arr = _to_single_device(local_arr)
    return jax.make_array_from_single_device_arrays(
        tuple(global_shape), sharding_axis0, [local_arr])
  
  def make_global_replicated_scalar(x):
      x = _to_single_device(x)
      return jax.make_array_from_single_device_arrays(
          x.shape, replicated, [x]
      )

  # Create globally-sharded arrays from local per-process data
  with timer_obj("PES Gather", []):
    p_ys = jax.tree_util.tree_map(make_global_axis1, p_ys)
    n_ys = jax.tree_util.tree_map(make_global_axis1, n_ys)
    accumulator = jax.tree_util.tree_map(make_global_axis0, accumulator)
    vec_pos = jax.tree_util.tree_map(make_global_axis0, vec_pos)
    dirloss_weight = make_global_replicated_scalar(dirloss_weight)
    magloss_weight = make_global_replicated_scalar(magloss_weight)

  # JIT-compiled all-gather + PES gradient computation
  (loss, es_grad, new_accumulator, p_ys_out, delta_losses,
   delta_task_losses, delta_direction_ori, delta_magnitude_ori, delta_mse_ori,
   snr_sample, snr_estimator, snr_delta_loss) = _pes_grad_sharded_inner_elo(
      dirloss_weight, magloss_weight, p_ys, n_ys, accumulator, vec_pos,
      std=std,
      sign_delta_loss_scalar=sign_delta_loss_scalar,
      replicated=replicated,
      delta_loss_scalar_afsnm=delta_loss_scalar_afsnm,
      imt_loss_type=imt_loss_type,
  )

  # Per-device slicing: extract this process's portion of per-task outputs
  start_idx = device_idx * samples_per_device
  end_idx = start_idx + samples_per_device

  def slice_first_dim(x):
    if hasattr(x, 'shape'):
      return x[start_idx:end_idx]
    return x

  new_accumulator = jax.tree_util.tree_map(slice_first_dim, new_accumulator)
  p_ys_out = jax.tree_util.tree_map(
      lambda x: x[:, start_idx:end_idx] if hasattr(x, 'shape') else x, p_ys_out)
  delta_losses = delta_losses[:, start_idx:end_idx]
  delta_task_losses = delta_task_losses[:, start_idx:end_idx]
  delta_direction_ori = delta_direction_ori[:, start_idx:end_idx]
  delta_magnitude_ori = delta_magnitude_ori[:, start_idx:end_idx]
  delta_mse_ori = delta_mse_ori[:, start_idx:end_idx]

  return (loss, es_grad, new_accumulator, p_ys_out, delta_losses,
          delta_task_losses, delta_direction_ori, delta_magnitude_ori,
          delta_mse_ori, snr_sample, snr_estimator, snr_delta_loss)


@gin.configurable
class TruncatedPES_ELO(gradient_learner.GradientEstimator):
  """GradientEstimator for computing PES gradient estimates.

  Persistent Evolution Strategies (PES) is a gradient estimation technique
  for computing unbiased gradients in a unrolled computation graph. It does this
  by building of of Evolutionary Strategies but additionally keeping a running
  buffer of all the previously used perturbations. See the paper for more
  details (http://proceedings.mlr.press/v139/vicol21a.html).

  In practice, PES is higher variance than pure truncated ES but lower bias.
  """

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      trunc_length=10,
      std=0.01,
      steps_per_jit=10,
      stack_antithetic_samples: bool = False,
      sign_delta_loss_scalar: Optional[float] = None,
      trunc_schedule = None,
      pmap_across_devices: bool = False,
      timer_obj: Any = None,
      expert_wd_sp = 100,
      expert_wd_ep = 4999,
      expert_traj_wmin = 0.0,
      bc_type: str = 'elo',
      use_bc_grads: bool = False,
      expert_dirloss_weight: float = 0.7,
      expert_magloss_weight: float = 0.3,
      expert_dirloss_wmin: float = 0.0,
      expert_magloss_wmin: float = 0.0,
      imt_loss_type: str = "decoupled",
  ):
    # "decoupled" = dirloss_weight * (1 - cos) + magloss_weight * |norm gap|
    # "mse"       = (dirloss_weight + magloss_weight) * mean((d_e - d_lo)^2)
    # Both forms are always computed and logged; this only picks which one the
    # meta-gradient follows, and keeps the imitation-vs-task split identical.
    if imt_loss_type not in ("decoupled", "mse"):
      raise ValueError("imt_loss_type must be 'decoupled' or 'mse', got "
                       f"{imt_loss_type!r}")
    self.imt_loss_type = imt_loss_type
    self.truncated_step = truncated_step
    self.std = std
    self.bc_type = bc_type
    self.expert_wd_sp = expert_wd_sp
    self.expert_wd_ep = expert_wd_ep
    self.expert_traj_wmin = expert_traj_wmin
    self.trunc_length = trunc_length
    self.steps_per_jit = steps_per_jit
    self.stack_antithetic_samples = stack_antithetic_samples
    self.sign_delta_loss_scalar = sign_delta_loss_scalar
    self.trunc_schedule = trunc_schedule
    self.timer_obj = timer_obj
    self.samples_per_device = self.truncated_step.num_tasks
    if pmap_across_devices:
      devices = mesh_utils.create_device_mesh((jax.device_count(),))
      self.mesh = Mesh(devices, axis_names=('devices',))
      self.grad_fn = functools.partial(compute_pes_grad_sharded_elo,
                                       mesh=self.mesh,
                                       imt_loss_type=imt_loss_type)
    else:
      self.mesh = None
      self.grad_fn = functools.partial(compute_pes_grad,
                                       imt_loss_type=imt_loss_type)
    self.use_bc_grads = use_bc_grads
    self.expert_dirloss_weight = expert_dirloss_weight
    self.expert_magloss_weight = expert_magloss_weight
    self.expert_dirloss_wmin = expert_dirloss_wmin
    self.expert_magloss_wmin = expert_magloss_wmin

    if self.trunc_length % self.steps_per_jit != 0:
      raise ValueError("Pass a trunc_length and steps_per_jit that are"
                       " multiples of each other.")
    assert self.timer_obj is not None, "timer_obj must be provided"

    @jax.jit
    def _update_state_jit(state, outer_iter):
      f32 = jnp.float32
      sp = f32(self.expert_wd_sp)
      ep = f32(self.expert_wd_ep)
      it = outer_iter.astype(jnp.float32)
      decayed_base = 1.0 - (it - sp) / (ep - sp)
      active = it > sp

      def schedule(min_w):
        mw = f32(min_w)
        return jnp.where(active, jnp.clip(decayed_base, mw, f32(1.0)), f32(1.0))

      ew_scalar = schedule(self.expert_traj_wmin)
      dirloss_weight = f32(self.expert_dirloss_weight) * schedule(self.expert_dirloss_wmin)
      magloss_weight = f32(self.expert_magloss_weight) * schedule(self.expert_magloss_wmin)

      old_ew = state.pos_state.inner_opt_state.expert_weight
      new_ew = jnp.broadcast_to(ew_scalar.astype(old_ew.dtype), old_ew.shape)

      new_pos_ios = dataclasses.replace(state.pos_state.inner_opt_state, expert_weight=new_ew)
      new_neg_ios = dataclasses.replace(state.neg_state.inner_opt_state, expert_weight=new_ew)
      new_state = dataclasses.replace(
          state,
          pos_state=dataclasses.replace(state.pos_state, inner_opt_state=new_pos_ios),
          neg_state=dataclasses.replace(state.neg_state, inner_opt_state=new_neg_ios),
      )
      return new_state, dirloss_weight, magloss_weight

    self._update_state_jit = _update_state_jit

  def task_name(self) -> str:
    return self.truncated_step.task_name()

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> PESWorkerState:
    theta = worker_weights.theta

    pos_unroll_state = self.truncated_step.init_step_state(
        theta, worker_weights.outer_state, key, theta_is_vector=False)
    neg_unroll_state = pos_unroll_state

    accumulator = jax.tree_util.tree_map(
        lambda x: jnp.zeros([self.truncated_step.num_tasks] + list(x.shape)),
        theta)

    return PESWorkerState(
        pos_state=pos_unroll_state,
        neg_state=neg_unroll_state,
        accumulator=accumulator)

  @profile.wrap()
  def get_datas(self):
    return [
        self.truncated_step.get_batch(self.steps_per_jit)
        for _ in range(self.trunc_length // self.steps_per_jit)
    ]

  @profile.wrap()
  def compute_gradient_estimate(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state: PESWorkerState,
      with_summary: bool = False,
      datas_list: Optional[Sequence[Any]] = None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    DEBUG = False
    # jax.debug.print('cur_outer_iteration: {}, total_outer_iteration: {}', worker_weights.outer_state.outer_iteration, worker_weights.outer_state.num_outer_iteration)
    # jax.debug.print('state.iteration: {}', state.pos_state.inner_opt_state.iteration)
    # jax.debug.print('state.inner_step: {}', state.pos_state.inner_step)
    # jax.debug.print('state.truncation_length: {}', state.pos_state.truncation_state)

    state, dirloss_weight, magloss_weight = self._update_state_jit(
        state, worker_weights.outer_state.outer_iteration)

    p_state = state.pos_state
    n_state = state.neg_state
    accumulator = state.accumulator
    rng = hk.PRNGSequence(key)

    # Snapshot pre-truncation inner_opt_state / task_param / mean(max_cumsum)
    # for post-truncation push decision.
    snap_pos_opt = p_state.inner_opt_state
    snap_pos_tp = p_state.task_param
    snap_neg_opt = n_state.inner_opt_state
    snap_neg_tp = n_state.task_param
    snap_mean_max_cumsum = 0.5 * (p_state.max_cumulative_sum
                                  + n_state.max_cumulative_sum)

    theta = worker_weights.theta

    vec_pos, vec_p_theta, vec_n_theta = common.vector_sample_perturbations(
        theta, next(rng), self.std, self.truncated_step.num_tasks)
    if DEBUG:
      for k,v in p_state.inner_opt_state.__dict__.items():
        print("p_state.inner_opt_state."+k)
        pprint.pprint(jax.tree_map(lambda x:x.sum(),v))
        print()

    p_yses = []
    n_yses = []
    p_bc_grads_list = []
    n_bc_grads_list = []
    metrics = []

    for i in range(self.trunc_length // self.steps_per_jit):
      if datas_list is None:
        if jax_utils.in_jit():
          raise ValueError("Must pass data in when using a jit gradient est.")
        datas = self.truncated_step.get_batch(self.steps_per_jit)
      else:
        datas = datas_list[i]

      # force all to be non weak type. This is for cache hit reasons.
      p_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       p_state)
      n_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       n_state)

      key = next(rng)

      p_state, n_state, p_ys, n_ys, p_bc_grads, n_bc_grads, m = common.maybe_stacked_es_unroll(
          self.truncated_step,
          self.steps_per_jit,
          self.stack_antithetic_samples,
          vec_p_theta,
          vec_n_theta,
          p_state,
          n_state,
          key,
          datas,
          worker_weights.outer_state,
          with_summary=with_summary,
          sample_rng_key=next(rng))

      metrics.append(m)
      p_yses.append(p_ys)
      n_yses.append(n_ys)
      p_bc_grads_list.append(p_bc_grads)
      n_bc_grads_list.append(n_bc_grads)

    # Post-truncation push: write the pre-window snapshot to buffer when a new
    # loss low occurred inside this window (improvement-CUSUM max rose). The
    # write slot advances only when the trajectory ends (any_done).
    buffer_size = jax.tree_util.tree_leaves(p_state.state_buffer)[0].shape[1]
    any_done = (jnp.any(jnp.stack([y.is_done for y in p_yses]), axis=(0, 1))
                | jnp.any(jnp.stack([y.is_done for y in n_yses]), axis=(0, 1)))
    p_state, n_state = _push_buffers_jit(
        p_state, n_state,
        snap_pos_opt, snap_pos_tp, snap_neg_opt, snap_neg_tp,
        snap_mean_max_cumsum,
        any_done,
        buffer_size)

    if self.use_bc_grads:
      # Convert lists of gradients to stacked arrays
      stacked_bc_grads = jax.tree_util.tree_map(
          lambda *xs: jnp.stack(xs),
          *(p_bc_grads_list + n_bc_grads_list)
      )
      # print(jax.tree_util.tree_map(lambda x: x.shape, stacked_bc_grads))
      
      # Take the mean along the first two dimensions
      mean_bc_grads = jax.tree_util.tree_map(
          lambda x: jnp.mean(x, axis=(0, 1, 2)),
          stacked_bc_grads
      )
      # print(jax.tree_util.tree_map(lambda x: x.shape, mean_bc_grads))
      # exit(0)
    else:
      mean_bc_grads = None


    (loss, es_grad, new_accumulator, p_ys, delta_losses, delta_task_losses,
     delta_direction_ori, delta_magnitude_ori, delta_mse_ori,
     snr_sample, snr_estimator, snr_delta_loss) = self.grad_fn(
        dirloss_weight,
        magloss_weight,
        p_yses,
        n_yses,
        accumulator,
        vec_pos,
        self.std,
        self.timer_obj,
        sign_delta_loss_scalar=self.sign_delta_loss_scalar,
        samples_per_device=self.samples_per_device,
        device_idx=jax.process_index())

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.task_loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        bc_grad=mean_bc_grads,
        unroll_state=PESWorkerState(p_state, n_state, new_accumulator),
        unroll_info=unroll_info)

    metrics = summary.aggregate_metric_list(
        metrics, use_jnp=jax_utils.in_jit(), key=next(rng))
    metrics["mean||snr_meta_grad_sample"] = snr_sample
    metrics["mean||snr_meta_grad_estimator"] = snr_estimator
    metrics["mean||snr_delta_loss"] = snr_delta_loss
    if with_summary:
      # metrics["sample||delta_loss_sample"] = summary.sample_value(
      #     key, jnp.abs(delta_losses))
      metrics["mean||delta_loss_mean"] = jnp.abs(delta_losses)
      metrics["mean||delta_task_loss_mean"] = jnp.abs(delta_task_losses)
      metrics["mean||delta_direction_ori_mean"] = jnp.mean(jnp.abs(delta_direction_ori))
      metrics["mean||delta_magnitude_ori_mean"] = jnp.mean(jnp.abs(delta_magnitude_ori))
      metrics["mean||delta_mse_ori_mean"] = jnp.mean(jnp.abs(delta_mse_ori))
      metrics["mean||expert_weight"] = state.pos_state.inner_opt_state.expert_weight
      metrics["mean||dirloss_weight"] = dirloss_weight
      metrics["mean||magloss_weight"] = magloss_weight
      metrics["mean||imt_cosine_loss"] = jnp.mean(p_ys.imt_cosine_loss)
      metrics["mean||imt_magnitude_loss"] = jnp.mean(p_ys.imt_magnitude_loss)

      if hasattr(p_state, "inner_step"):
        metrics["sample||inner_step"] = p_state.inner_step[0]
        # metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics


@functools.partial(jax.pmap, axis_name="dev")
def _pmap_reduce(vals):
  return jax.lax.pmean(vals, axis_name="dev")


@jax.pmap
def vec_key_split(key):
  key1, key2 = jax.random.split(key)
  return key1, key2


@gin.configurable
class TruncatedPESPMAP_ELO(TruncatedPES_ELO):
  """GradientEstimator for computing PES gradient estimates leveraging pmap.
  See TruncatedPES documentation for information on PES. This estimator
  additionally makes use of multiple TPU devices via jax's pmap.
  """

  def __init__(self,
               *args,
               num_devices=8,
               replicate_data_across_devices=False,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.num_devices = num_devices
    self.replicate_data_across_devices = replicate_data_across_devices
    if len(jax.local_devices()) != self.num_devices:
      raise ValueError("Mismatch in device count!"
                       f" Found: {jax.local_devices()}."
                       f" Expected {num_devices} devices.")

    self.pmap_init_step_state = jax.pmap(
        self.truncated_step.init_step_state, in_axes=(None, None, 0))

    # NOTE: pmap version is legacy code; dirloss/magloss weights are fixed (no decay)
    self.pmap_compute_pes_grad = jax.pmap(
        functools.partial(compute_pes_grad,
                          jnp.asarray(self.expert_dirloss_weight),
                          jnp.asarray(self.expert_magloss_weight),
                          std=self.std,
                          imt_loss_type=self.imt_loss_type))

    self.pmap_vector_sample_perturbations = jax.pmap(
        functools.partial(
            common.vector_sample_perturbations,
            std=self.std,
            num_samples=self.truncated_step.num_tasks),
        in_axes=(None, 0),
    )

  @functools.partial(
      jax.pmap,
      in_axes=(None, 0, 0, 0, 0, None, None),
      static_broadcasted_argnums=(
          0,
          6,
      ))
  def pmap_unroll_next_state(self, vec_theta, key, state, datas, outer_state,
                             with_summary):
    theta_is_vector = True
    key1, key2 = jax.random.split(key)
    override_num_steps = None
    (p_state, p_ys), m = common.truncated_unroll(  # pylint: disable=unbalanced-tuple-unpacking
        self.truncated_step,
        self.steps_per_jit,
        theta_is_vector,
        vec_theta,
        key1,
        state,
        datas,
        outer_state,
        override_num_steps,
        with_summary=with_summary,
        sample_rng_key=key2)
    return (p_state, p_ys), m

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> PESWorkerState:
    theta = worker_weights.theta

    keys = jax.random.split(key, self.num_devices)
    # Note this doesn't use sampled theta for the first init.
    # I believe this is fine most of the time.
    # TODO(lmetz) consider init-ing at an is_done state instead.
    pos_unroll_state = self.pmap_init_step_state(worker_weights.theta,
                                                 worker_weights.outer_state,
                                                 keys)
    neg_unroll_state = pos_unroll_state

    accumulator = jax.tree_util.tree_map(
        lambda x: jnp.zeros([self.truncated_step.num_tasks] + list(x.shape)),
        theta)
    accumulator = flax_jax_utils.replicate(accumulator)

    return PESWorkerState(
        pos_state=pos_unroll_state,
        neg_state=neg_unroll_state,
        accumulator=accumulator)

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state: PESWorkerState,
      with_summary: bool = False
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:

    p_state = state.pos_state
    n_state = state.neg_state
    accumulator = state.accumulator
    vec_key = jax.random.split(key, self.num_devices)

    # Snapshot pre-truncation state for post-truncation push decision
    # (leaves carry an extra leading device axis under pmap).
    snap_pos_opt = p_state.inner_opt_state
    snap_pos_tp = p_state.task_param
    snap_neg_opt = n_state.inner_opt_state
    snap_neg_tp = n_state.task_param
    snap_mean_max_cumsum = 0.5 * (p_state.max_cumulative_sum
                                  + n_state.max_cumulative_sum)

    theta = worker_weights.theta

    vec_key1, vec_key = vec_key_split(vec_key)
    vec_pos, vec_p_theta, vec_n_theta = self.pmap_vector_sample_perturbations(
        theta, vec_key1)

    p_yses = []
    n_yses = []
    metrics = []

    def get_batch():
      """Get batch with leading dims [num_devices, steps_per_jit, num_tasks]."""
      if self.replicate_data_across_devices:
        b = self.truncated_step.get_batch(self.steps_per_jit)
        return flax_jax_utils.replicate(b)
      else:
        # Use different data across the devices
        batches = [
            self.truncated_step.get_batch(self.steps_per_jit)
            for _ in range(self.num_devices)
        ]
        return tree_utils.tree_zip_onp(batches)

    for _ in range(self.trunc_length // self.steps_per_jit):
      datas = get_batch()

      # force all to be non weak type. This is for cache hit reasons.
      # TODO(lmetz) consider instead just setting the weak type flag?
      p_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       p_state)
      n_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       n_state)

      vec_key1, vec_key = vec_key_split(vec_key)
      (p_state, p_ys), m = self.pmap_unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
          vec_p_theta, vec_key, p_state, datas, worker_weights.outer_state,
          with_summary)
      metrics.append(m)

      p_yses.append(p_ys)
      (n_state, n_ys), _ = self.pmap_unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
          vec_n_theta, vec_key, n_state, datas, worker_weights.outer_state,
          False)
      n_yses.append(n_ys)

    # Post-truncation push (pmap variant): state_buffer leaves are
    # (num_devices, num_tasks, buffer_size, ...). Push the pre-window snapshot
    # when a new loss low occurred inside this window (improvement-CUSUM max
    # rose); the write slot advances only on trajectory end. is_done is
    # (num_devices, steps_per_jit, num_tasks) -> reduce over the step axis.
    buffer_size = jax.tree_util.tree_leaves(p_state.state_buffer)[0].shape[2]
    any_done = (jnp.any(jnp.concatenate([y.is_done for y in p_yses], axis=1), axis=1)
                | jnp.any(jnp.concatenate([y.is_done for y in n_yses], axis=1), axis=1))
    p_state, n_state = _push_buffers_pmap(
        p_state, n_state,
        snap_pos_opt, snap_pos_tp, snap_neg_opt, snap_neg_tp,
        snap_mean_max_cumsum,
        any_done,
        buffer_size)

    (loss, es_grad, new_accumulator, p_ys, delta_losses, delta_task_losses,
     delta_direction_ori, delta_magnitude_ori, delta_mse_ori, snr_sample,
     snr_estimator, snr_delta_loss) = self.pmap_compute_pes_grad(
        p_yses, n_yses, accumulator, vec_pos)

    # NOTE pmap-path SNR is mean-of-ratios across devices (not exact global
    # ratio-of-means). Acceptable for this legacy code path.
    es_grad, loss, snr_sample, snr_estimator, snr_delta_loss = (
        flax_jax_utils.unreplicate(
            _pmap_reduce((es_grad, loss, snr_sample, snr_estimator,
                          snr_delta_loss))))

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        unroll_state=PESWorkerState(p_state, n_state, new_accumulator),
        unroll_info=unroll_info)

    metrics = summary.aggregate_metric_list(metrics)
    metrics["mean||snr_meta_grad_sample"] = snr_sample
    metrics["mean||snr_meta_grad_estimator"] = snr_estimator
    metrics["mean||snr_delta_loss"] = snr_delta_loss
    if with_summary:
      # metrics["sample||delta_loss_sample"] = summary.sample_value(
      #     key, jnp.abs(delta_losses))
      metrics["mean||delta_loss_mean"] = jnp.abs(delta_losses)
      metrics["mean||delta_task_loss_mean"] = jnp.abs(delta_task_losses)
      metrics["mean||delta_direction_ori_mean"] = jnp.mean(jnp.abs(delta_direction_ori))
      metrics["mean||delta_magnitude_ori_mean"] = jnp.mean(jnp.abs(delta_magnitude_ori))
      metrics["mean||delta_mse_ori_mean"] = jnp.mean(jnp.abs(delta_mse_ori))
      if hasattr(p_state, "inner_step"):
        metrics["sample||inner_step"] = p_state.inner_step[0]
        # metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics
