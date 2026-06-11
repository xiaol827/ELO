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

"""Chen-style vectorized truncated PES gradient estimator.

Differences from the ELO variant:
  * no buffer / CUSUM push logic;
  * a single ``imt_loss`` field per step (no direction/magnitude split);
  * ``expert_weight`` is sourced from ``state.pos_state.truncation_state.N_unroll``
    (BC on/off alternation: ``expert_weight = 1 - N_unroll % 2``) and written
    into the inner opt state each outer step.
"""
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
from lopt_truncated_step_chen import TruncatedUnrollOut
import numpy as np

PRNGKey = jnp.ndarray
MetaParams = Any
TruncatedUnrollState = Any


def second_moment_normalizer(x, eps=1e-8):
  return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), keepdims=True))


@flax.struct.dataclass
class PESWorkerState(gradient_learner.GradientEstimatorState):
  pos_state: TruncatedUnrollState
  neg_state: TruncatedUnrollState
  accumulator: MetaParams


@functools.partial(
    jax.jit,
    static_argnames=("std", "timer_obj", "sign_delta_loss_scalar",
                     "samples_per_device", "device_idx"))
def compute_pes_grad(
    expert_weight: jnp.ndarray,
    p_yses: Sequence[TruncatedUnrollOut],
    n_yses: Sequence[TruncatedUnrollOut],
    accumulator: MetaParams,
    vec_pos: MetaParams,
    std: float,
    timer_obj: Any,
    sign_delta_loss_scalar: Optional[float] = None,
    delta_loss_scalar_afsnm: Optional[float] = 0.01,
    samples_per_device: Optional[int] = None,
    device_idx: Optional[int] = None,
):
  """Compute the PES gradient estimate (chen: single imt_loss, expert_weight blend)."""

  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  with timer_obj("PES Gather", []):
    pass

  p_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  delta_imt_ori = p_ys.imt_loss - n_ys.imt_loss
  delta_imt = second_moment_normalizer(delta_imt_ori, eps=0.0) * delta_loss_scalar_afsnm
  delta_task_losses = p_ys.task_loss - n_ys.task_loss
  delta_losses = expert_weight * delta_imt + (1.0 - expert_weight) * delta_task_losses

  if sign_delta_loss_scalar:
    sign_per_task = jnp.sign(jnp.mean(delta_losses * p_ys.mask, axis=0))
    delta_losses = jnp.ones_like(delta_losses) * sign_per_task * sign_delta_loss_scalar

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

  def _switch_one_accum(a, b):
    shape = [num_tasks] + [1] * (len(a.shape) - 1)
    return jnp.where(jnp.reshape(has_finished[-1], shape), a, b)

  new_accumulator = jax.tree_util.tree_map(_switch_one_accum, vec_pos, accumulator)

  pos_task_loss = jnp.sum(p_ys.task_loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
  neg_task_loss = jnp.sum(n_ys.task_loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)

  return (jnp.mean((pos_task_loss + neg_task_loss) / 2.0),
          es_grad, new_accumulator, p_ys, delta_losses, delta_task_losses, delta_imt_ori)


@functools.partial(jax.jit, static_argnames=("std", "sign_delta_loss_scalar", "replicated"))
def _pes_grad_sharded_inner_chen(expert_weight, p_ys, n_ys, accumulator, vec_pos, std,
                                 sign_delta_loss_scalar, replicated,
                                 delta_loss_scalar_afsnm=0.01):
  """JIT-compiled inner function: all-gather via sharding constraint, then PES gradient (chen)."""
  p_ys = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), p_ys)
  n_ys = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), n_ys)
  accumulator = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), accumulator)
  vec_pos = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), vec_pos)

  delta_imt_ori = p_ys.imt_loss - n_ys.imt_loss
  delta_imt = second_moment_normalizer(delta_imt_ori, eps=0.0) * delta_loss_scalar_afsnm
  delta_task_losses = p_ys.task_loss - n_ys.task_loss
  delta_losses = expert_weight * delta_imt + (1.0 - expert_weight) * delta_task_losses

  if sign_delta_loss_scalar:
    sign_per_task = jnp.sign(jnp.mean(delta_losses * p_ys.mask, axis=0))
    delta_losses = jnp.ones_like(delta_losses) * sign_per_task * sign_delta_loss_scalar

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

  def _switch_one_accum(a, b):
    shape = [num_tasks] + [1] * (len(a.shape) - 1)
    return jnp.where(jnp.reshape(has_finished[-1], shape), a, b)

  new_accumulator = jax.tree_util.tree_map(_switch_one_accum, vec_pos, accumulator)

  pos_task_loss = jnp.sum(p_ys.task_loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
  neg_task_loss = jnp.sum(n_ys.task_loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)

  return (
      jnp.mean((pos_task_loss + neg_task_loss) / 2.0),
      es_grad,
      new_accumulator,
      p_ys,
      delta_losses,
      delta_task_losses,
      delta_imt_ori,
  )


def compute_pes_grad_sharded_chen(
    expert_weight: jnp.ndarray,
    p_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    n_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    accumulator: MetaParams,
    vec_pos: MetaParams,
    std: float,
    timer_obj: Any,
    sign_delta_loss_scalar: Optional[float] = None,
    samples_per_device: int = 1,
    device_idx: int = 0,
    baseline_losses: Optional[list] = None,
    delta_loss_scalar_afsnm: Optional[float] = 0.01,
    mesh: Optional[Mesh] = None,
):
  """Compute PES gradient using JAX mesh-based sharding for multi-GPU all-gather (chen)."""
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
    global_shape = list(local_arr.shape)
    global_shape[1] = global_shape[1] * num_devices
    local_arr = _to_single_device(local_arr)
    return jax.make_array_from_single_device_arrays(
        tuple(global_shape), sharding_axis1, [local_arr])

  def make_global_axis0(local_arr):
    global_shape = list(local_arr.shape)
    global_shape[0] = global_shape[0] * num_devices
    local_arr = _to_single_device(local_arr)
    return jax.make_array_from_single_device_arrays(
        tuple(global_shape), sharding_axis0, [local_arr])

  def make_global_replicated_scalar(x):
    x = _to_single_device(x)
    return jax.make_array_from_single_device_arrays(x.shape, replicated, [x])

  with timer_obj("PES Gather", []):
    p_ys = jax.tree_util.tree_map(make_global_axis1, p_ys)
    n_ys = jax.tree_util.tree_map(make_global_axis1, n_ys)
    accumulator = jax.tree_util.tree_map(make_global_axis0, accumulator)
    vec_pos = jax.tree_util.tree_map(make_global_axis0, vec_pos)
    expert_weight = make_global_replicated_scalar(expert_weight)

  (loss, es_grad, new_accumulator, p_ys_out, delta_losses,
   delta_task_losses, delta_imt_ori) = _pes_grad_sharded_inner_chen(
      expert_weight, p_ys, n_ys, accumulator, vec_pos,
      std=std,
      sign_delta_loss_scalar=sign_delta_loss_scalar,
      replicated=replicated,
      delta_loss_scalar_afsnm=delta_loss_scalar_afsnm,
  )

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
  delta_imt_ori = delta_imt_ori[:, start_idx:end_idx]

  return (loss, es_grad, new_accumulator, p_ys_out, delta_losses,
          delta_task_losses, delta_imt_ori)


@gin.configurable
class TruncatedPES_CHEN(gradient_learner.GradientEstimator):
  """Chen-style PES gradient estimator.

  Expert weight alternates BC on/off (1 <-> 0) each unroll, derived from the
  truncation_state's N_unroll counter. Gradient combination:
      delta_losses = ew * delta_imt + (1 - ew) * delta_task
  where delta_imt is second-moment normalized across the [steps, tasks] tensor.
  """

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      trunc_length=10,
      std=0.01,
      steps_per_jit=10,
      stack_antithetic_samples: bool = False,
      sign_delta_loss_scalar: Optional[float] = None,
      trunc_schedule=None,
      pmap_across_devices: bool = False,
      timer_obj: Any = None,
      use_bc_grads: bool = False,
      delta_loss_scalar_afsnm: float = 0.01,
  ):
    self.truncated_step = truncated_step
    self.std = std
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
      self.grad_fn = functools.partial(compute_pes_grad_sharded_chen, mesh=self.mesh)
    else:
      self.mesh = None
      self.grad_fn = compute_pes_grad
    self.use_bc_grads = use_bc_grads
    self.delta_loss_scalar_afsnm = delta_loss_scalar_afsnm

    if self.trunc_length % self.steps_per_jit != 0:
      raise ValueError("Pass a trunc_length and steps_per_jit that are"
                       " multiples of each other.")
    assert self.timer_obj is not None, "timer_obj must be provided"

    @jax.jit
    def _write_expert_weight(state):
      """Derive expert_weight from truncation_state.current_unroll and write into inner_opt_state."""
      N = state.pos_state.truncation_state.current_unroll
      ew_scalar = jnp.where(jnp.mod(N, jnp.int32(2)) == 0, 1.0, 0.0).astype(jnp.float32)

      old_ew = state.pos_state.inner_opt_state.expert_weight
      new_ew = jnp.broadcast_to(ew_scalar.astype(old_ew.dtype), old_ew.shape)

      new_pos_ios = dataclasses.replace(state.pos_state.inner_opt_state, expert_weight=new_ew)
      new_neg_ios = dataclasses.replace(state.neg_state.inner_opt_state, expert_weight=new_ew)
      new_state = dataclasses.replace(
          state,
          pos_state=dataclasses.replace(state.pos_state, inner_opt_state=new_pos_ios),
          neg_state=dataclasses.replace(state.neg_state, inner_opt_state=new_neg_ios),
      )
      return new_state, ew_scalar

    self._write_expert_weight = _write_expert_weight

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
  def compute_gradient_estimate(
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state: PESWorkerState,
      with_summary: bool = False,
      datas_list: Optional[Sequence[Any]] = None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:

    state, expert_weight = self._write_expert_weight(state)

    p_state = state.pos_state
    n_state = state.neg_state
    accumulator = state.accumulator
    rng = hk.PRNGSequence(key)

    theta = worker_weights.theta

    vec_pos, vec_p_theta, vec_n_theta = common.vector_sample_perturbations(
        theta, next(rng), self.std, self.truncated_step.num_tasks)

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

    if self.use_bc_grads:
      stacked_bc_grads = jax.tree_util.tree_map(
          lambda *xs: jnp.stack(xs),
          *(p_bc_grads_list + n_bc_grads_list)
      )
      mean_bc_grads = jax.tree_util.tree_map(
          lambda x: jnp.mean(x, axis=(0, 1, 2)),
          stacked_bc_grads
      )
    else:
      mean_bc_grads = None

    (loss, es_grad, new_accumulator, p_ys, delta_losses, delta_task_losses,
     delta_imt_ori) = self.grad_fn(
        expert_weight,
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
    if with_summary:
      metrics["mean||delta_loss_mean"] = jnp.abs(delta_losses)
      metrics["mean||delta_task_loss_mean"] = jnp.abs(delta_task_losses)
      metrics["mean||delta_imt_ori_mean"] = jnp.mean(jnp.abs(delta_imt_ori))
      metrics["mean||expert_weight"] = state.pos_state.inner_opt_state.expert_weight
      metrics["mean||imt_loss"] = jnp.mean(p_ys.imt_loss)
      metrics["mean||current_unroll"] = state.pos_state.truncation_state.current_unroll.astype(jnp.float32)
      metrics["mean||unroll_length"] = state.pos_state.truncation_state.length.astype(jnp.float32)

      if hasattr(p_state, "inner_step"):
        metrics["sample||inner_step"] = p_state.inner_step[0]

    return output, metrics
