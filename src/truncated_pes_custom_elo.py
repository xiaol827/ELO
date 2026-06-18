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


def _select_per_task(post_tree, pre_tree, done_flag):
  """Per-task select: done_flag[t] True -> post[t], else pre[t]. Broadcasts flag."""
  return jax.tree_util.tree_map(
      lambda po, pr: jnp.where(
          done_flag.reshape((-1,) + (1,) * (po.ndim - 1)), po, pr),
      post_tree, pre_tree)


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
    pos_is_done_stack, neg_is_done_stack,
    buffer_size,
    snap_accumulator=None, accumulator_buffer=None,
):
  """Post-truncation push decision + FIFO buffer write (pure jnp/vmap).

  struggled  = post mean(pos/neg max_cumulative_sum) > pre snapshot mean.
  any_done   = OR of is_done across the truncation on either side.
  push       = struggled | any_done.
  On any_done: write post-truncation state; else on struggled: write pre snapshot.
  Uses FIFO next_write_slot from p_state (pos/neg kept in sync).

  When `accumulator_buffer` is provided, the pre-truncation PES accumulator
  snapshot (`snap_accumulator`, shared across pos/neg) is written into the same
  slot, so a later buffer warm-restart can recover the perturbation history of
  the stored inner state. Returns the updated accumulator_buffer as a third
  element (None on the legacy pmap path where it is not threaded through).
  """
  post_mean = 0.5 * (p_state.max_cumulative_sum + n_state.max_cumulative_sum)
  struggled = post_mean > snap_mean_max_cumsum
  any_done = jnp.any(pos_is_done_stack, axis=0) | jnp.any(neg_is_done_stack, axis=0)
  push = struggled | any_done

  slots = p_state.next_write_slot
  # sel_pos_opt = _select_per_task(p_state.inner_opt_state, snap_pos_opt, any_done)
  # sel_pos_tp  = _select_per_task(p_state.task_param,      snap_pos_tp,  any_done)
  # sel_neg_opt = _select_per_task(n_state.inner_opt_state, snap_neg_opt, any_done)
  # sel_neg_tp  = _select_per_task(n_state.task_param,      snap_neg_tp,  any_done)

  new_pos_sb  = _write_per_task_if(p_state.state_buffer,      snap_pos_opt, slots, push)
  new_pos_tpb = _write_per_task_if(p_state.task_param_buffer, snap_pos_tp,  slots, push)
  new_neg_sb  = _write_per_task_if(n_state.state_buffer,      snap_neg_opt, slots, push)
  new_neg_tpb = _write_per_task_if(n_state.task_param_buffer, snap_neg_tp,  slots, push)

  # Write the matching PES accumulator snapshot into the same slot. The
  # pre-truncation accumulator pairs with snap_*_opt (the pre-truncation inner
  # state currently written above); if push is ever switched to write the
  # post-truncation state, snap_accumulator must switch to the post accumulator.
  if accumulator_buffer is not None:
    # Push is rare (struggled | any_done); skip the full num_tasks*theta write
    # on the common no-push step. Equivalent to _write_per_task_if's masked
    # no-op write, but avoids touching the buffer memory at all when nothing
    # pushes.
    new_acc_buf = jax.lax.cond(
        jnp.any(push),
        lambda: _write_per_task_if(accumulator_buffer, snap_accumulator, slots,
                                   push),
        lambda: accumulator_buffer)
  else:
    new_acc_buf = None

  new_slots = jnp.where(push, (slots + 1) % buffer_size, slots)
  # Realign pos/neg max_cumulative_sum to a shared baseline: post_mean if push,
  # else snap_mean (keeps next-truncation comparison well-defined).
  new_max_cumsum = jnp.where(push, post_mean, snap_mean_max_cumsum)

  return (
      p_state.replace(state_buffer=new_pos_sb, task_param_buffer=new_pos_tpb,
                      next_write_slot=new_slots,
                      max_cumulative_sum=new_max_cumsum),
      n_state.replace(state_buffer=new_neg_sb, task_param_buffer=new_neg_tpb,
                      next_write_slot=new_slots,
                      max_cumulative_sum=new_max_cumsum),
      new_acc_buf,
  )


_push_buffers_jit = jax.jit(_push_buffers_core, static_argnames=("buffer_size",))
_push_buffers_pmap = jax.pmap(
    _push_buffers_core,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None),
    static_broadcasted_argnums=(9,))


@jax.jit
def _resolve_restart_accum(reset_slot_stack, accumulator_buffer):
  """Resolve the per-task buffer warm-restart accumulator (single jit dispatch).

  reset_slot_stack: [steps, num_tasks] int32 — -1 on train / random-init steps,
    the restored slot index on a buffer warm-restart. For each task we take the
    last warm-restart this truncation and read that slot's stored accumulator;
    tasks that never warm-restarted get 0, so the downstream reset reduces to
    vec_pos exactly (original behavior).
  accumulator_buffer: pytree, leaves [num_tasks, buffer_size, *theta_leaf].
  Returns restart_accum: pytree, leaves [num_tasks, *theta_leaf].

  Folded into one jit so the per-leaf gather is a single fused dispatch instead
  of one eager kernel per theta leaf, and `lax.cond` skips the gather entirely
  on the common step where no task warm-restarted from the buffer.
  """
  num_tasks = reset_slot_stack.shape[1]
  buffer_mask = reset_slot_stack >= 0
  any_buffer = jnp.any(buffer_mask, axis=0)  # [num_tasks]
  step_ids = jnp.arange(reset_slot_stack.shape[0])[:, None]
  last_step = jnp.maximum(
      jnp.max(jnp.where(buffer_mask, step_ids, -1), axis=0), 0)
  last_slot = jnp.where(
      any_buffer, reset_slot_stack[last_step, jnp.arange(num_tasks)],
      0).astype(jnp.int32)

  def _gather_leaf(buf_leaf):  # buf_leaf: [num_tasks, buffer_size, *]
    g = jax.vmap(
        lambda b, s: jax.lax.dynamic_index_in_dim(b, s, axis=0,
                                                  keepdims=False))(
            buf_leaf, last_slot)  # [num_tasks, *]
    mask = any_buffer.reshape((num_tasks,) + (1,) * (g.ndim - 1))
    return jnp.where(mask, g, jnp.zeros_like(g))

  def _zeros_leaf(buf_leaf):
    return jnp.zeros((num_tasks,) + buf_leaf.shape[2:], buf_leaf.dtype)

  return jax.lax.cond(
      jnp.any(any_buffer),
      lambda: jax.tree_util.tree_map(_gather_leaf, accumulator_buffer),
      lambda: jax.tree_util.tree_map(_zeros_leaf, accumulator_buffer))


@flax.struct.dataclass
class PESWorkerState(gradient_learner.GradientEstimatorState):
  pos_state: TruncatedUnrollState
  neg_state: TruncatedUnrollState
  accumulator: MetaParams
  # Per-task, per-slot snapshot of the PES accumulator, written slot-for-slot
  # alongside state_buffer/task_param_buffer at push time. On a buffer
  # warm-restart, the matching slot's accumulator is restored so PES keeps
  # attributing the loss difference to the full perturbation history that
  # produced the restored inner state (instead of resetting it to zero).
  # Leaves: (num_tasks, buffer_size, *theta_leaf). None on the legacy pmap path.
  accumulator_buffer: Any = None

@functools.partial(jax.jit, static_argnames=("std", "timer_obj", "sign_delta_loss_scalar", "samples_per_device", "device_idx"))
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
    restart_accum: Optional[MetaParams] = None,
):
  """Compute the PES gradient estimate from the outputs of many unrolls.

  Args:
    p_yses: Sequence of PES outputs from the positive perturbation.
    n_yses: Sequence of PES outputs from the negative perturbation.
    restart_accum: Optional per-task accumulator inherited from a buffer
      warm-restart this truncation (0 for tasks that did not warm-restart from
      the buffer). When provided, a reset trajectory continues from
      restart_accum + vec_pos rather than vec_pos, so the post-restart loss is
      attributed to the full perturbation history of the restored state. None
      reproduces the original reset-to-vec_pos behavior exactly.
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
  delta_direction = second_moment_normalizer(delta_direction_ori, eps=0.0) * delta_loss_scalar_afsnm
  delta_magnitude = second_moment_normalizer(delta_magnitude_ori, eps=0.0) * delta_loss_scalar_afsnm
  delta_task_losses = p_ys.task_loss - n_ys.task_loss
  delta_losses = dirloss_weight * delta_direction + magloss_weight * delta_magnitude + (1.0 - dirloss_weight - magloss_weight) * delta_task_losses

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

  # Perturbation history a reset trajectory carries this truncation: vec_pos
  # for a fresh (random) init, restart_accum + vec_pos for a buffer
  # warm-restart. restart_accum is 0 for tasks that did not warm-restart, so
  # this reduces to vec_pos and matches the original behavior when None.
  if restart_accum is None:
    post_reset_accum = vec_pos
  else:
    post_reset_accum = tree_utils.tree_add(restart_accum, vec_pos)

  num_tasks = last_unroll_loss.shape[0]

  def reshape_to(loss, p):
    return loss.reshape((num_tasks,) + (1,) * (len(p.shape) - 1)) * factor * p

  #initially zero for ONLY RNN and NOT MLP
  es_grad_from_accum = jax.tree_util.tree_map(
      functools.partial(reshape_to, last_unroll_loss), accumulator)

  #initially zero for both RNN and MLP
  es_grad_from_new_perturb = jax.tree_util.tree_map(
      functools.partial(reshape_to, new_unroll_loss), post_reset_accum)

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

  new_accumulator = jax.tree_util.tree_map(_switch_one_accum, post_reset_accum,
                                           accumulator)

  pos_task_loss = jnp.sum(p_ys.task_loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
  neg_task_loss = jnp.sum(n_ys.task_loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)

  return (jnp.mean((pos_task_loss + neg_task_loss) / 2.0), es_grad,
          new_accumulator, p_ys, delta_losses, delta_task_losses,
          delta_direction_ori, delta_magnitude_ori, snr_sample, snr_estimator,
          snr_delta_loss)

@functools.partial(jax.jit, static_argnames=("std", "sign_delta_loss_scalar", "replicated"))
def _pes_grad_sharded_inner_elo(dirloss_weight, magloss_weight, p_ys, n_ys, accumulator, vec_pos, std,
                                sign_delta_loss_scalar, replicated, delta_loss_scalar_afsnm=0.01,
                                restart_accum=None):
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
  if restart_accum is not None:
    restart_accum = jax.tree_util.tree_map(
        lambda x: lax.with_sharding_constraint(x, replicated), restart_accum)

  # Separate direction and magnitude IMT losses, normalize each independently
  delta_direction_ori = p_ys.imt_cosine_loss - n_ys.imt_cosine_loss
  delta_magnitude_ori = p_ys.imt_magnitude_loss - n_ys.imt_magnitude_loss
  delta_direction = second_moment_normalizer(delta_direction_ori, eps=0.0) * delta_loss_scalar_afsnm
  delta_magnitude = second_moment_normalizer(delta_magnitude_ori, eps=0.0) * delta_loss_scalar_afsnm
  delta_task_losses = p_ys.task_loss - n_ys.task_loss
  delta_losses = dirloss_weight * delta_direction + magloss_weight * delta_magnitude + (1.0 - dirloss_weight - magloss_weight) * delta_task_losses

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

  # See compute_pes_grad: buffer warm-restart inherits restart_accum (0 for
  # tasks that did not warm-restart), so this reduces to vec_pos when None.
  if restart_accum is None:
    post_reset_accum = vec_pos
  else:
    post_reset_accum = tree_utils.tree_add(restart_accum, vec_pos)

  num_tasks = last_unroll_loss.shape[0]

  def reshape_to(loss, p):
    return loss.reshape((num_tasks,) + (1,) * (len(p.shape) - 1)) * factor * p

  es_grad_from_accum = jax.tree_util.tree_map(
      functools.partial(reshape_to, last_unroll_loss), accumulator)

  es_grad_from_new_perturb = jax.tree_util.tree_map(
      functools.partial(reshape_to, new_unroll_loss), post_reset_accum)

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

  new_accumulator = jax.tree_util.tree_map(_switch_one_accum, post_reset_accum,
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
    mesh: Optional[Mesh] = None,
    restart_accum: Optional[MetaParams] = None,
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
    if restart_accum is not None:
      restart_accum = jax.tree_util.tree_map(make_global_axis0, restart_accum)
    dirloss_weight = make_global_replicated_scalar(dirloss_weight)
    magloss_weight = make_global_replicated_scalar(magloss_weight)

  # JIT-compiled all-gather + PES gradient computation
  (loss, es_grad, new_accumulator, p_ys_out, delta_losses,
   delta_task_losses, delta_direction_ori, delta_magnitude_ori,
   snr_sample, snr_estimator, snr_delta_loss) = _pes_grad_sharded_inner_elo(
      dirloss_weight, magloss_weight, p_ys, n_ys, accumulator, vec_pos,
      std=std,
      sign_delta_loss_scalar=sign_delta_loss_scalar,
      replicated=replicated,
      delta_loss_scalar_afsnm=delta_loss_scalar_afsnm,
      restart_accum=restart_accum,
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

  return (loss, es_grad, new_accumulator, p_ys_out, delta_losses,
          delta_task_losses, delta_direction_ori, delta_magnitude_ori,
          snr_sample, snr_estimator, snr_delta_loss)


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
  ):
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
      self.grad_fn = functools.partial(compute_pes_grad_sharded_elo, mesh=self.mesh)
    else:
      self.mesh = None
      self.grad_fn = compute_pes_grad
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

    num_tasks = self.truncated_step.num_tasks
    accumulator = jax.tree_util.tree_map(
        lambda x: jnp.zeros([num_tasks] + list(x.shape)),
        theta)
    # Per-slot accumulator snapshots, mirroring state_buffer's slot layout.
    buffer_size = jax.tree_util.tree_leaves(
        pos_unroll_state.state_buffer)[0].shape[1]
    accumulator_buffer = jax.tree_util.tree_map(
        lambda x: jnp.zeros([num_tasks, buffer_size] + list(x.shape)),
        theta)

    return PESWorkerState(
        pos_state=pos_unroll_state,
        neg_state=neg_unroll_state,
        accumulator_buffer=accumulator_buffer,
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
    # Pre-truncation accumulator buffer: any buffer warm-restart during this
    # unroll read inner states stored here (written by an earlier truncation's
    # push), so the matching accumulator must also come from this pre-push copy.
    accumulator_buffer = state.accumulator_buffer
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

    # Resolve the accumulator inherited by any buffer warm-restart this
    # truncation. reset_slot (per step, from the inner step) is -1 on train /
    # random-init steps and the restored slot index on a buffer warm-restart.
    # For each task we take the last warm-restart this truncation and read that
    # slot's stored accumulator; tasks that never warm-restarted get 0, so the
    # downstream reset reduces to vec_pos exactly (original behavior).
    # Single fused/jitted dispatch (see _resolve_restart_accum): collapses the
    # former per-leaf eager gather and skips it entirely when no task
    # warm-restarted this truncation.
    reset_slot_stack = jnp.concatenate([y.reset_slot for y in p_yses], axis=0)
    restart_accum = _resolve_restart_accum(reset_slot_stack, accumulator_buffer)

    # Post-truncation push: write pos/neg inner states to buffer when the
    # trajectory struggled (CUSUM mean-max rose) or any reset fired this segment.
    # The pre-truncation accumulator is stored slot-for-slot alongside.
    buffer_size = jax.tree_util.tree_leaves(p_state.state_buffer)[0].shape[1]
    pos_is_done_stack = jnp.concatenate([y.is_done for y in p_yses], axis=0)
    neg_is_done_stack = jnp.concatenate([y.is_done for y in n_yses], axis=0)
    p_state, n_state, accumulator_buffer = _push_buffers_jit(
        p_state, n_state,
        snap_pos_opt, snap_pos_tp, snap_neg_opt, snap_neg_tp,
        snap_mean_max_cumsum,
        pos_is_done_stack, neg_is_done_stack,
        buffer_size,
        snap_accumulator=accumulator, accumulator_buffer=accumulator_buffer)

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
     delta_direction_ori, delta_magnitude_ori, snr_sample, snr_estimator,
     snr_delta_loss) = self.grad_fn(
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
        device_idx=jax.process_index(),
        restart_accum=restart_accum)

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.task_loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        bc_grad=mean_bc_grads,
        unroll_state=PESWorkerState(p_state, n_state, new_accumulator,
                                    accumulator_buffer=accumulator_buffer),
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
                          std=self.std))

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
    # (num_devices, num_tasks, buffer_size, ...); is_done is
    # (num_devices, steps_per_jit, num_tasks), so concat on step axis (axis=1).
    buffer_size = jax.tree_util.tree_leaves(p_state.state_buffer)[0].shape[2]
    pos_is_done_stack = jnp.concatenate([y.is_done for y in p_yses], axis=1)
    neg_is_done_stack = jnp.concatenate([y.is_done for y in n_yses], axis=1)
    # Legacy pmap path: accumulator_buffer is not threaded here (snap_accumulator
    # / accumulator_buffer default to None), so the third return is unused.
    p_state, n_state, _ = _push_buffers_pmap(
        p_state, n_state,
        snap_pos_opt, snap_pos_tp, snap_neg_opt, snap_neg_tp,
        snap_mean_max_cumsum,
        pos_is_done_stack, neg_is_done_stack,
        buffer_size)

    (loss, es_grad, new_accumulator, p_ys, delta_losses, delta_task_losses,
     delta_direction_ori, delta_magnitude_ori, snr_sample,
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
      if hasattr(p_state, "inner_step"):
        metrics["sample||inner_step"] = p_state.inner_step[0]
        # metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics
