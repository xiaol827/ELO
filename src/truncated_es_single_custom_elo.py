# coding=utf-8
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""ES-Single ELO: accumulator-free unbiased ES gradient estimator with ELO machinery.

Ports the ELO additions from `truncated_pes_custom_elo.py` (failure-aware resume
buffer push, direction/magnitude/task loss blend, scheduled expert/dir/mag weights,
per-particle/cross-device collective contract) onto the `ESSingle` base in
`learned_optimization/outer_trainers/es_single.py`.

Key invariants preserved from ES-Single:
  - `vec_pos` is fixed across all truncation windows of one inner problem.
  - `vec_pos` is resampled only when `is_done` fires (per-task).
  - No PES accumulator; the gradient is `mean_over_tasks(vec_pos * delta_loss / (2 σ²))`.

Key additions from ELO (mirrored from `truncated_pes_custom_elo.py`):
  - `_update_state_jit` linearly ramps `expert_weight` (written into
    `pos_state.inner_opt_state.expert_weight` and `neg_state.inner_opt_state.expert_weight`),
    `dirloss_weight` and `magloss_weight` between `expert_wd_sp` and `expert_wd_ep`.
  - `delta_losses` is the weighted blend of direction, magnitude, and task losses.
  - Multi-device path replicates `dirloss_weight`, `magloss_weight` scalars and
    `vec_pos`/`prev_delta_loss` across devices; mirrors PES_ELO's all-gather contract.
  - Reuses `_push_buffers_jit` from `truncated_pes_custom_elo` for the
    failure-aware resume buffer (estimator-agnostic).
"""

import dataclasses
import functools
from typing import Any, Mapping, Optional, Sequence, Tuple

import flax
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
from learned_optimization.outer_trainers.es_single import (
    ESSingle,
    _aggregate_delta_losses,
)

from lopt_truncated_step_elo import TruncatedUnrollOut
from truncated_pes_custom_elo import (
    _push_buffers_jit,
    _meta_grad_snr,
    _delta_loss_snr,
    second_moment_normalizer,
)

PRNGKey = jnp.ndarray
MetaParams = Any
TruncatedUnrollState = Any


@flax.struct.dataclass
class ESSingleWorkerState_ELO(gradient_learner.GradientEstimatorState):
  """Worker state for ESSingle_ELO.

  Same shape as upstream `ESSingleWorkerState` (pos/neg unroll states, persistent
  perturbation `vec_pos`, telescoping cross-window field `prev_delta_loss`).
  Expert-supervision schedule values (`expert_weight`, `dirloss_weight`,
  `magloss_weight`) are NOT stored here — `expert_weight` is written into the
  inner LO state's `expert_weight` field (matching PES_ELO), and the two loss
  weights are passed inline to the gradient function each window.
  """
  pos_state: TruncatedUnrollState
  neg_state: TruncatedUnrollState
  vec_pos: MetaParams
  prev_delta_loss: jnp.ndarray  # [num_tasks], float32 (telescoping mode)


# ---------------------------------------------------------------------------
# Single-machine ES-Single ELO gradient
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("std", "timer_obj", "sign_delta_loss_scalar",
                     "samples_per_device", "device_idx", "loss_type"))
def compute_es_single_grad_elo(
    dirloss_weight: jnp.ndarray,
    magloss_weight: jnp.ndarray,
    p_yses: Sequence[TruncatedUnrollOut],
    n_yses: Sequence[TruncatedUnrollOut],
    vec_pos: MetaParams,
    std: float,
    timer_obj: Any,
    sign_delta_loss_scalar: Optional[float] = None,
    delta_loss_scalar_afsnm: float = 0.01,
    samples_per_device: Optional[int] = None,  # unused; kept for API parity
    device_idx: Optional[int] = None,           # unused; kept for API parity
    loss_type: str = "mean",
    prev_delta_loss: Optional[jnp.ndarray] = None,
    final_loss_weight: float = 0.0,
):
  """ES-Single gradient with ELO direction+magnitude+task loss blend.

  Mirrors `compute_pes_grad` from truncated_pes_custom_elo.py but:
    - DROPS the PES accumulator and the `has_finished` pre/post-reset split.
    - Uses `vec_pos` directly (single fixed perturbation per inner problem).
    - Aggregates per-task deltas via `_aggregate_delta_losses(loss_type)`.

  Returns the same rich tuple as PES_ELO (loss, es_grad, p_ys, delta_losses,
  delta_task_losses, delta_direction_ori, delta_magnitude_ori,
  snr_sample, snr_estimator, snr_delta_loss) plus `new_prev_delta_loss`
  for the telescoping mode.
  """
  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  with timer_obj("ES-Single ELO Gather", []):
    pass  # No cross-device communication in single-machine mode.

  p_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  # ELO direction + magnitude + task-loss blend (identical to PES_ELO).
  delta_direction_ori = p_ys.imt_cosine_loss - n_ys.imt_cosine_loss
  delta_magnitude_ori = p_ys.imt_magnitude_loss - n_ys.imt_magnitude_loss
  delta_direction = (
      second_moment_normalizer(delta_direction_ori, eps=0.0) * delta_loss_scalar_afsnm)
  delta_magnitude = (
      second_moment_normalizer(delta_magnitude_ori, eps=0.0) * delta_loss_scalar_afsnm)
  delta_task_losses = p_ys.task_loss - n_ys.task_loss
  delta_losses = (
      dirloss_weight * delta_direction
      + magloss_weight * delta_magnitude
      + (1.0 - dirloss_weight - magloss_weight) * delta_task_losses)

  if sign_delta_loss_scalar:
    sign_per_task = jnp.sign(jnp.mean(delta_losses * p_ys.mask, axis=0))
    delta_losses = (
        jnp.ones_like(delta_losses) * sign_per_task * sign_delta_loss_scalar)

  # Aggregate per-task deltas — single-perturbation ES, so a SINGLE scalar
  # delta per task multiplies a SINGLE vec_pos per task. No pre/post split.
  delta_loss, new_prev_delta_loss = _aggregate_delta_losses(
      delta_losses, p_ys.mask, loss_type, prev_delta_loss, final_loss_weight)

  factor = 1.0 / (2 * std**2)
  num_tasks = delta_loss.shape[0]

  def reshape_to(loss, p):
    return loss.reshape((num_tasks,) + (1,) * (len(p.shape) - 1)) * factor * p

  vec_es_grad = jax.tree_util.tree_map(
      functools.partial(reshape_to, delta_loss), vec_pos)
  es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  # Cross-particle SNR (jointly over the local task axis — same as PES_ELO
  # in the single-machine path).
  snr_sample, snr_estimator = _meta_grad_snr(vec_es_grad, es_grad)
  snr_delta_loss = _delta_loss_snr(delta_losses, p_ys.mask)

  pos_task_loss = jnp.sum(p_ys.task_loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
  neg_task_loss = jnp.sum(n_ys.task_loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)
  loss = jnp.mean((pos_task_loss + neg_task_loss) / 2.0)

  return (loss, es_grad, p_ys, delta_losses, delta_task_losses,
          delta_direction_ori, delta_magnitude_ori,
          snr_sample, snr_estimator, snr_delta_loss,
          new_prev_delta_loss)


# ---------------------------------------------------------------------------
# Multi-GPU (sharded) ES-Single ELO gradient
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("std", "sign_delta_loss_scalar", "replicated", "loss_type"))
def _es_single_grad_sharded_inner_elo(
    dirloss_weight, magloss_weight, p_ys, n_ys, vec_pos, std,
    sign_delta_loss_scalar, replicated,
    delta_loss_scalar_afsnm=0.01,
    loss_type="mean",
    prev_delta_loss=None,
    final_loss_weight=0.0,
):
  """JIT-compiled all-gather (via sharding constraint) + ES-Single ELO grad.

  Mirrors `_pes_grad_sharded_inner_elo` exactly except:
    - Drops the `accumulator` gather and update (no accumulator in ES-Single).
    - Adds `prev_delta_loss` to the replicated set (for telescoping mode).
    - Computes a single `delta_loss` per task (no pre/post-reset split) and
      one gradient term `delta_loss · vec_pos / (2 σ²)`.
  """
  # All-gather: replicate sharded arrays across devices via constraint.
  p_ys = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), p_ys)
  n_ys = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), n_ys)
  vec_pos = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), vec_pos)
  prev_delta_loss = lax.with_sharding_constraint(prev_delta_loss, replicated)

  delta_direction_ori = p_ys.imt_cosine_loss - n_ys.imt_cosine_loss
  delta_magnitude_ori = p_ys.imt_magnitude_loss - n_ys.imt_magnitude_loss
  delta_direction = (
      second_moment_normalizer(delta_direction_ori, eps=0.0) * delta_loss_scalar_afsnm)
  delta_magnitude = (
      second_moment_normalizer(delta_magnitude_ori, eps=0.0) * delta_loss_scalar_afsnm)
  delta_task_losses = p_ys.task_loss - n_ys.task_loss
  delta_losses = (
      dirloss_weight * delta_direction
      + magloss_weight * delta_magnitude
      + (1.0 - dirloss_weight - magloss_weight) * delta_task_losses)

  if sign_delta_loss_scalar:
    sign_per_task = jnp.sign(jnp.mean(delta_losses * p_ys.mask, axis=0))
    delta_losses = (
        jnp.ones_like(delta_losses) * sign_per_task * sign_delta_loss_scalar)

  delta_loss, new_prev_delta_loss = _aggregate_delta_losses(
      delta_losses, p_ys.mask, loss_type, prev_delta_loss, final_loss_weight)

  factor = 1.0 / (2 * std**2)
  num_tasks = delta_loss.shape[0]

  def reshape_to(loss, p):
    return loss.reshape((num_tasks,) + (1,) * (len(p.shape) - 1)) * factor * p

  vec_es_grad = jax.tree_util.tree_map(
      functools.partial(reshape_to, delta_loss), vec_pos)
  es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  # SNR over the GLOBAL task axis (inside the all-gather), then sliced
  # downstream — same pattern as PES_ELO's sharded inner.
  snr_sample, snr_estimator = _meta_grad_snr(vec_es_grad, es_grad)
  snr_delta_loss = _delta_loss_snr(delta_losses, p_ys.mask)

  pos_task_loss = jnp.sum(p_ys.task_loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
  neg_task_loss = jnp.sum(n_ys.task_loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)
  loss = jnp.mean((pos_task_loss + neg_task_loss) / 2.0)

  return (loss, es_grad, p_ys, delta_losses, delta_task_losses,
          delta_direction_ori, delta_magnitude_ori,
          snr_sample, snr_estimator, snr_delta_loss,
          new_prev_delta_loss)


def compute_es_single_grad_sharded_elo(
    dirloss_weight: jnp.ndarray,
    magloss_weight: jnp.ndarray,
    p_yses: Sequence[TruncatedUnrollOut],
    n_yses: Sequence[TruncatedUnrollOut],
    vec_pos: MetaParams,
    std: float,
    timer_obj: Any,
    sign_delta_loss_scalar: Optional[float] = None,
    samples_per_device: int = 1,
    device_idx: int = 0,
    mesh: Optional[Mesh] = None,
    delta_loss_scalar_afsnm: float = 0.01,
    loss_type: str = "mean",
    prev_delta_loss: Optional[jnp.ndarray] = None,
    final_loss_weight: float = 0.0,
):
  """Multi-GPU ES-Single ELO gradient via JAX mesh + sharding constraint.

  Replicates the all-gather pattern of `compute_pes_grad_sharded_elo`:
  each process holds a local task shard; we promote each shard to a globally-
  sharded array, then run the JIT'd inner that uses
  `lax.with_sharding_constraint(..., replicated)` to trigger NCCL all-gather.

  Difference vs PES_ELO sharded path:
    - NO `accumulator` global-axis0 setup or replication (ES-Single has none).
    - `prev_delta_loss` IS promoted+replicated and sliced back per device.
  """
  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  p_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  replicated = NamedSharding(mesh, P())
  sharding_axis1 = NamedSharding(mesh, P(None, 'devices'))
  sharding_axis0 = NamedSharding(mesh, P('devices'))

  num_devices = jax.device_count()
  local_device = jax.local_devices()[0]

  def _to_single_device(arr):
    if isinstance(arr, jax.Array) and not arr.is_fully_addressable:
      return arr.addressable_data(0)
    return jax.device_put(arr, local_device)

  def make_global_axis1(local_arr):
    """[steps, local_tasks, ...] -> global [steps, total_tasks, ...] sharded on axis 1."""
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
    return jax.make_array_from_single_device_arrays(x.shape, replicated, [x])

  with timer_obj("ES-Single ELO Gather", []):
    p_ys = jax.tree_util.tree_map(make_global_axis1, p_ys)
    n_ys = jax.tree_util.tree_map(make_global_axis1, n_ys)
    vec_pos = jax.tree_util.tree_map(make_global_axis0, vec_pos)
    prev_delta_loss = make_global_axis0(prev_delta_loss)
    dirloss_weight = make_global_replicated_scalar(jnp.asarray(dirloss_weight))
    magloss_weight = make_global_replicated_scalar(jnp.asarray(magloss_weight))

  (loss, es_grad, p_ys_out, delta_losses, delta_task_losses,
   delta_direction_ori, delta_magnitude_ori,
   snr_sample, snr_estimator, snr_delta_loss,
   new_prev_delta_loss) = _es_single_grad_sharded_inner_elo(
      dirloss_weight, magloss_weight, p_ys, n_ys, vec_pos,
      std=std,
      sign_delta_loss_scalar=sign_delta_loss_scalar,
      replicated=replicated,
      delta_loss_scalar_afsnm=delta_loss_scalar_afsnm,
      loss_type=loss_type,
      prev_delta_loss=prev_delta_loss,
      final_loss_weight=final_loss_weight,
  )

  # Per-device slicing of per-task outputs.
  start_idx = device_idx * samples_per_device
  end_idx = start_idx + samples_per_device

  p_ys_out = jax.tree_util.tree_map(
      lambda x: x[:, start_idx:end_idx] if hasattr(x, 'shape') else x, p_ys_out)
  delta_losses = delta_losses[:, start_idx:end_idx]
  delta_task_losses = delta_task_losses[:, start_idx:end_idx]
  delta_direction_ori = delta_direction_ori[:, start_idx:end_idx]
  delta_magnitude_ori = delta_magnitude_ori[:, start_idx:end_idx]
  new_prev_delta_loss = new_prev_delta_loss[start_idx:end_idx]

  return (loss, es_grad, p_ys_out, delta_losses, delta_task_losses,
          delta_direction_ori, delta_magnitude_ori,
          snr_sample, snr_estimator, snr_delta_loss,
          new_prev_delta_loss)


# ---------------------------------------------------------------------------
# Main class: ESSingle_ELO
# ---------------------------------------------------------------------------

@gin.configurable
class ESSingle_ELO(ESSingle):
  """GradientEstimator combining unbiased ES-Single with ELO expert supervision.

  Per-window flow (mirrors `TruncatedPES_ELO.compute_gradient_estimate`):
    1. `_update_state_jit` writes the scheduled `expert_weight` into both
       inner_opt_states and returns the scheduled (dirloss_weight, magloss_weight).
    2. Snapshot pre-truncation `(inner_opt_state, task_param, mean_max_cumsum)`
       for the buffer-push decision.
    3. Run truncation windows via `common.maybe_stacked_es_unroll`, with the
       **same** `vec_pos` across all windows of the inner problem.
    4. `_push_buffers_jit` pushes the post-truncation state into the resume
       buffer when struggle or any_done fires.
    5. Call `self.grad_fn` (single-device or sharded) to compute the ES-Single
       ELO gradient with the dir/mag/task blend.
    6. Resample `vec_pos` per task on `is_done`, mirroring upstream ESSingle.
  """

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      trunc_length: int = 10,
      std: float = 0.01,
      steps_per_jit: int = 10,
      stack_antithetic_samples: bool = False,
      sign_delta_loss_scalar: Optional[float] = None,
      trunc_schedule=None,
      pmap_across_devices: bool = False,
      timer_obj: Any = None,
      use_bc_grads: bool = False,
      std_schedule=None,
      loss_type: str = "mean",
      final_loss_weight: float = 0.0,
      # ELO additions
      expert_wd_sp=100,
      expert_wd_ep=4999,
      expert_traj_wmin=0.0,
      bc_type: str = 'elo',
      expert_dirloss_weight: float = 0.7,
      expert_magloss_weight: float = 0.3,
      expert_dirloss_wmin: float = 0.0,
      expert_magloss_wmin: float = 0.0,
      delta_loss_scalar_afsnm: float = 0.01,
  ):
    # Initialize the ESSingle base; this also picks the loss-type validation,
    # std_schedule, and default grad_fn. We override grad_fn below.
    super().__init__(
        truncated_step=truncated_step,
        trunc_length=trunc_length,
        std=std,
        steps_per_jit=steps_per_jit,
        stack_antithetic_samples=stack_antithetic_samples,
        sign_delta_loss_scalar=sign_delta_loss_scalar,
        trunc_schedule=trunc_schedule,
        pmap_across_devices=pmap_across_devices,
        timer_obj=timer_obj,
        use_bc_grads=use_bc_grads,
        std_schedule=std_schedule,
        loss_type=loss_type,
        final_loss_weight=final_loss_weight,
    )

    # ELO config
    self.bc_type = bc_type
    self.expert_wd_sp = expert_wd_sp
    self.expert_wd_ep = expert_wd_ep
    self.expert_traj_wmin = expert_traj_wmin
    self.expert_dirloss_weight = expert_dirloss_weight
    self.expert_magloss_weight = expert_magloss_weight
    self.expert_dirloss_wmin = expert_dirloss_wmin
    self.expert_magloss_wmin = expert_magloss_wmin
    self.delta_loss_scalar_afsnm = delta_loss_scalar_afsnm

    # Override grad_fn with the ELO variants (the base set it to ESSingle's).
    if pmap_across_devices:
      self.grad_fn = functools.partial(
          compute_es_single_grad_sharded_elo, mesh=self.mesh)
    else:
      self.grad_fn = compute_es_single_grad_elo

    # JITted weight-schedule helper. Same algebra as PES_ELO `_update_state_jit`.
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

      new_pos_ios = dataclasses.replace(
          state.pos_state.inner_opt_state, expert_weight=new_ew)
      new_neg_ios = dataclasses.replace(
          state.neg_state.inner_opt_state, expert_weight=new_ew)
      new_state = dataclasses.replace(
          state,
          pos_state=dataclasses.replace(state.pos_state, inner_opt_state=new_pos_ios),
          neg_state=dataclasses.replace(state.neg_state, inner_opt_state=new_neg_ios),
      )
      return new_state, dirloss_weight, magloss_weight

    self._update_state_jit = _update_state_jit

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> ESSingleWorkerState_ELO:
    """Initialize pos/neg inner states, the initial `vec_pos`, and `prev_delta_loss`."""
    theta = worker_weights.theta
    rng = hk.PRNGSequence(key)
    pos_unroll_state = self.truncated_step.init_step_state(
        theta, worker_weights.outer_state, next(rng), theta_is_vector=False)
    neg_unroll_state = pos_unroll_state
    vec_pos, _, _ = common.vector_sample_perturbations(
        theta, next(rng), self.std, self.truncated_step.num_tasks)
    return ESSingleWorkerState_ELO(
        pos_state=pos_unroll_state,
        neg_state=neg_unroll_state,
        vec_pos=vec_pos,
        prev_delta_loss=jnp.zeros(
            self.truncated_step.num_tasks, dtype=jnp.float32))

  @profile.wrap()
  def compute_gradient_estimate(  # pytype: disable=signature-mismatch
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state: ESSingleWorkerState_ELO,
      with_summary: bool = False,
      datas_list: Optional[Sequence[Any]] = None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:

    # 1) Schedule expert weight (into inner_opt_states) + dir/mag loss weights.
    state, dirloss_weight, magloss_weight = self._update_state_jit(
        state, worker_weights.outer_state.outer_iteration)

    p_state = state.pos_state
    n_state = state.neg_state
    rng = hk.PRNGSequence(key)

    # 2) Snapshot pre-truncation state for buffer-push decision.
    snap_pos_opt = p_state.inner_opt_state
    snap_pos_tp = p_state.task_param
    snap_neg_opt = n_state.inner_opt_state
    snap_neg_tp = n_state.task_param
    snap_mean_max_cumsum = 0.5 * (
        p_state.max_cumulative_sum + n_state.max_cumulative_sum)

    theta = worker_weights.theta

    # 3) Pin vec_pos and prev_delta_loss to local device — protects against
    # globally-sharded outputs from a prior sharded-grad call leaking back into
    # state and triggering all-gather on every step (the documented
    # `_to_local` pitfall from upstream ESSingle).
    _local_dev = jax.local_devices()[0]

    def _to_local(x):
      if isinstance(x, jax.Array) and not x.is_fully_addressable:
        return x.addressable_data(0)
      return jax.device_put(x, _local_dev)

    vec_pos = jax.tree_util.tree_map(_to_local, state.vec_pos)
    prev_delta_loss = _to_local(state.prev_delta_loss)
    vec_p_theta = jax.tree_util.tree_map(lambda t, p: t + p, theta, vec_pos)
    vec_n_theta = jax.tree_util.tree_map(lambda t, p: t - p, theta, vec_pos)

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

      # Force strong types (avoids JIT retracing).
      p_state = jax.tree_util.tree_map(
          lambda x: jnp.asarray(x, dtype=x.dtype), p_state)
      n_state = jax.tree_util.tree_map(
          lambda x: jnp.asarray(x, dtype=x.dtype), n_state)

      k = next(rng)
      p_state, n_state, p_ys, n_ys, p_bc_grads, n_bc_grads, m = (
          common.maybe_stacked_es_unroll(
              self.truncated_step,
              self.steps_per_jit,
              self.stack_antithetic_samples,
              vec_p_theta,
              vec_n_theta,
              p_state,
              n_state,
              k,
              datas,
              worker_weights.outer_state,
              with_summary=with_summary,
              sample_rng_key=next(rng)))

      metrics.append(m)
      p_yses.append(p_ys)
      n_yses.append(n_ys)
      p_bc_grads_list.append(p_bc_grads)
      n_bc_grads_list.append(n_bc_grads)

    # 4) Post-truncation buffer push (same logic as PES_ELO; buffer is
    # estimator-agnostic so we reuse the helper directly).
    buffer_size = jax.tree_util.tree_leaves(p_state.state_buffer)[0].shape[1]
    pos_is_done_stack = jnp.concatenate([y.is_done for y in p_yses], axis=0)
    neg_is_done_stack = jnp.concatenate([y.is_done for y in n_yses], axis=0)
    p_state, n_state = _push_buffers_jit(
        p_state, n_state,
        snap_pos_opt, snap_pos_tp, snap_neg_opt, snap_neg_tp,
        snap_mean_max_cumsum,
        pos_is_done_stack, neg_is_done_stack,
        buffer_size)

    if self.use_bc_grads:
      stacked_bc_grads = jax.tree_util.tree_map(
          lambda *xs: jnp.stack(xs),
          *(p_bc_grads_list + n_bc_grads_list))
      mean_bc_grads = jax.tree_util.tree_map(
          lambda x: jnp.mean(x, axis=(0, 1, 2)), stacked_bc_grads)
    else:
      mean_bc_grads = None

    # 5) ELO ES-Single gradient computation.
    (loss, es_grad, p_ys, delta_losses, delta_task_losses,
     delta_direction_ori, delta_magnitude_ori,
     snr_sample, snr_estimator, snr_delta_loss,
     new_prev_delta_loss) = self.grad_fn(
        dirloss_weight,
        magloss_weight,
        p_yses,
        n_yses,
        vec_pos,
        self.std,
        self.timer_obj,
        sign_delta_loss_scalar=self.sign_delta_loss_scalar,
        delta_loss_scalar_afsnm=self.delta_loss_scalar_afsnm,
        samples_per_device=self.samples_per_device,
        device_idx=jax.process_index(),
        loss_type=self.loss_type,
        prev_delta_loss=prev_delta_loss,
        final_loss_weight=self.final_loss_weight)

    # 6) Resample vec_pos for tasks whose inner problem reset this window.
    is_done_local = _to_local(p_ys.is_done)
    did_reset = jnp.any(is_done_local, axis=0)  # [num_tasks]
    new_vec_pos, _, _ = common.vector_sample_perturbations(
        theta, next(rng), self.std, self.truncated_step.num_tasks)

    def _select(old, new):
      mask = did_reset.reshape((did_reset.shape[0],) + (1,) * (len(old.shape) - 1))
      return jnp.where(mask, new, old)

    updated_vec_pos = jax.tree_util.tree_map(_select, vec_pos, new_vec_pos)

    new_prev_delta_loss = _to_local(new_prev_delta_loss)
    updated_prev_delta_loss = jnp.where(did_reset, 0.0, new_prev_delta_loss)

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.task_loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        bc_grad=mean_bc_grads,
        unroll_state=ESSingleWorkerState_ELO(
            p_state, n_state, updated_vec_pos, updated_prev_delta_loss),
        unroll_info=unroll_info)

    metrics = summary.aggregate_metric_list(
        metrics, use_jnp=jax_utils.in_jit(), key=next(rng))
    metrics["mean||snr_meta_grad_sample"] = snr_sample
    metrics["mean||snr_meta_grad_estimator"] = snr_estimator
    metrics["mean||snr_delta_loss"] = snr_delta_loss
    if with_summary:
      metrics["mean||delta_loss_mean"] = jnp.abs(delta_losses)
      metrics["mean||delta_task_loss_mean"] = jnp.abs(delta_task_losses)
      metrics["mean||delta_direction_ori_mean"] = jnp.mean(jnp.abs(delta_direction_ori))
      metrics["mean||delta_magnitude_ori_mean"] = jnp.mean(jnp.abs(delta_magnitude_ori))
      metrics["mean||expert_weight"] = state.pos_state.inner_opt_state.expert_weight
      metrics["mean||dirloss_weight"] = dirloss_weight
      metrics["mean||magloss_weight"] = magloss_weight
      metrics["mean||imt_cosine_loss"] = jnp.mean(p_ys.imt_cosine_loss)
      metrics["mean||imt_magnitude_loss"] = jnp.mean(p_ys.imt_magnitude_loss)
      metrics["sample||baseline_loss"] = 0.0
      if hasattr(p_state, "inner_step"):
        metrics["sample||inner_step"] = p_state.inner_step[0]

    return output, metrics
