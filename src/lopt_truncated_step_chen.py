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

"""Chen-style vectorized truncated step: no buffer, no CUSUM, single imt_loss."""
import dataclasses
import os
import functools
from typing import Any, Callable, Optional, Tuple, TypeVar, List
import chex
import flax
import gin
import jax
import jax.numpy as jnp
import time
from learned_optimization import summary
from learned_optimization import training
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import truncated_step
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import base as tasks_base

PRNGKey = jnp.ndarray
MetaParams = Any
InnerState = Any
InnerBatch = Any
OuterBatch = Any
OuterState = Any
T = TypeVar("T")
G = TypeVar("G")


@flax.struct.dataclass
class TruncatedUnrollOut:
  task_loss: jnp.ndarray
  is_done: jnp.ndarray
  task_param: Any
  iteration: jnp.ndarray
  mask: jnp.ndarray
  imt_loss: jnp.ndarray


@flax.struct.dataclass
class TruncatedUnrollState:
  inner_opt_state: Any
  inner_step: jnp.ndarray
  truncation_state: Any
  task_param: Any
  is_done: jnp.ndarray
  bc_grad: Optional[Any] = None


@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "trunc_sched"))
@functools.partial(jax.vmap, in_axes=(None, None, None, None, None, 0, None))
def init_truncation_state(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    outer_state: Any,
    key: PRNGKey,
    num_steps_override: Optional[int] = None) -> TruncatedUnrollState:
  """Init inner state without vectorized theta."""
  return _init_truncation_state(task_family, learned_opt, trunc_sched, theta,
                                outer_state, key, num_steps_override)


@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "trunc_sched"))
@functools.partial(jax.vmap, in_axes=(None, None, None, 0, None, 0, None))
def init_truncation_state_vec_theta(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    outer_state: Any,
    key: PRNGKey,
    num_steps_override: Optional[int] = None) -> TruncatedUnrollState:
  """Init inner state with vectorized theta."""
  return _init_truncation_state(task_family, learned_opt, trunc_sched, theta,
                                outer_state, key, num_steps_override)


def _init_truncation_state(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    outer_state: Any,
    key: PRNGKey,
    num_steps_override: Optional[int] = None) -> TruncatedUnrollState:
  """Initialize a single inner problem state."""

  key1, key2, key3, key4 = jax.random.split(key, 4)
  task_param = task_family.sample(key1)
  inner_param, inner_state = task_family.task_fn(task_param).init_with_state(
      key2)
  trunc_state = trunc_sched.init(key3, outer_state)
  num_steps = trunc_state.length if num_steps_override is None else num_steps_override
  opt_state = learned_opt.opt_fn(
      theta, is_training=True).init(
          inner_param, inner_state, num_steps=num_steps, key=key4)
  return TruncatedUnrollState(
      inner_opt_state=opt_state,
      inner_step=jnp.asarray(0, dtype=jnp.int32),
      truncation_state=trunc_state,
      task_param=task_param,
      is_done=jnp.asarray(False),
  )


def progress_or_reset_inner_opt_state(
    task_family: tasks_base.TaskFamily,
    opt: opt_base.Optimizer,
    num_steps: int,
    trunc_sched: truncation_schedule.TruncationSchedule,
    truncation_state: Any,
    outer_state: Any,
    key: PRNGKey,
    inner_opt_state: T,
    task_param: G,
    inner_step: int,
    is_done: bool,
    data: Any,
    cond_fn: Callable[[bool, Any, Any, Any], Any] = jax.lax.cond,
    axis_name: Optional[str] = None,
    meta_loss_with_aux_key: Optional[str] = None,
) -> Tuple[T, G, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
  """Train a single step, or reset the current inner problem (fresh sample)."""

  def reset_fn(key):
    """Reset: always sample a fresh task + fresh opt_state (no buffer)."""
    k1, k2, k3 = jax.random.split(key, 3)
    if axis_name:
      k1 = jax.lax.all_gather(k1, axis_name)[0]
    tp = task_family.sample(k1)
    p, s = task_family.task_fn(tp).init_with_state(k2)
    next_state = opt.init(p, s, num_steps=num_steps, key=k3)
    zero = jnp.asarray(0., dtype=jnp.float32)
    return (next_state, tp, jnp.asarray(0, dtype=jnp.int32),
            zero, zero, truncation_state)

  def train_fn(key):
    """Train one inner step."""
    key1, key2 = jax.random.split(key, 2)
    p = opt.get_params(inner_opt_state)
    s = opt.get_state(inner_opt_state)

    task = task_family.task_fn(task_param)
    if meta_loss_with_aux_key:
      def loss_fn(p, s, k, d):
        l, s, aux = task.loss_with_state_and_aux(p, s, k, d)
        return l, (s, aux)
      (l, (s, aux)), g = jax.value_and_grad(loss_fn, has_aux=True)(p, s, key1, data)
      if meta_loss_with_aux_key not in aux:
        raise ValueError(f"Aux key: {meta_loss_with_aux_key} not found in "
                         f"task family {task_family}. Found keys are "
                         f" {aux.keys()}")
      meta_loss = aux[meta_loss_with_aux_key]
    else:
      (l, s), g = jax.value_and_grad(
          task.loss_with_state, has_aux=True)(p, s, key1, data)
      meta_loss = l

    if axis_name:
      g = jax.lax.pmean(g, axis_name=axis_name)
      l = jax.lax.pmean(l, axis_name=axis_name)

    next_inner_opt_state, imt_loss = opt.update(
        inner_opt_state, g, loss=l, model_state=s, key=key2)
    next_inner_step = inner_step + 1

    return (next_inner_opt_state, task_param, next_inner_step,
            jnp.asarray(meta_loss, dtype=jnp.float32),
            jnp.asarray(imt_loss, dtype=jnp.float32),
            truncation_state)

  return cond_fn(jnp.logical_not(is_done), train_fn, reset_fn, key)


@functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0))
def vectorized_loss_and_aux(task_family: tasks_base.TaskFamily,
                            learned_opt: lopt_base.LearnedOptimizer,
                            theta: lopt_base.MetaParams, inner_opt_state: Any,
                            task_param: Any, key: PRNGKey,
                            data: Any) -> jnp.ndarray:
  """Vectorized computation of the task loss given data."""
  task = task_family.task_fn(task_param)
  opt = learned_opt.opt_fn(theta, is_training=True)
  p, s = opt.get_params_state(inner_opt_state)
  l, _, aux = task.loss_with_state_and_aux(p, s, key, data)
  return l, aux


def _truncated_unroll_one_step(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: TruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key,
    override_num_steps: Optional[int] = None,
) -> Tuple[TruncatedUnrollState, TruncatedUnrollOut]:
  """Train a given inner problem state a single step or reset it when done."""
  key1, key2 = jax.random.split(key)
  if override_num_steps is not None:
    num_steps = override_num_steps
  else:
    num_steps = state.truncation_state.length

  (next_inner_opt_state, task_param, next_inner_step, task_loss,
   imt_loss, next_truncation_state) = (
       progress_or_reset_inner_opt_state(
           task_family=task_family,
           opt=learned_opt.opt_fn(theta),
           num_steps=num_steps,
           trunc_sched=trunc_sched,
           truncation_state=state.truncation_state,
           outer_state=outer_state,
           key=key1,
           inner_opt_state=state.inner_opt_state,
           task_param=state.task_param,
           inner_step=state.inner_step,
           is_done=state.is_done,
           data=data,
           meta_loss_with_aux_key=meta_loss_with_aux_key,
       )
   )
  new_truncation_state, is_done = trunc_sched.next_state(
      next_truncation_state, next_inner_step, key2, outer_state)

  output_state = TruncatedUnrollState(
      inner_opt_state=next_inner_opt_state,
      inner_step=next_inner_step,
      truncation_state=new_truncation_state,
      task_param=task_param,
      is_done=is_done,
  )

  # Mask out the fresh-sample reset step: it should not contribute to
  # meta-loss / imt loss.
  step_mask = jnp.logical_and(jnp.logical_not(state.is_done), next_inner_step != 0)

  out = TruncatedUnrollOut(
      is_done=is_done,
      task_loss=task_loss,
      mask=step_mask,
      iteration=next_inner_step,
      task_param=state.task_param,
      imt_loss=imt_loss,
  )

  return output_state, out


@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "trunc_sched",
                     "meta_loss_with_aux_key"))
@functools.partial(
    jax.vmap, in_axes=(None, None, None, None, 0, 0, 0, None, None, None))
def truncated_unroll_one_step(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: TruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key: Optional[str],
    override_num_steps: Optional[int],
) -> Tuple[TruncatedUnrollState, TruncatedUnrollOut]:
  """Perform one step of inner training without vectorized theta."""
  return _truncated_unroll_one_step(
      task_family=task_family,
      learned_opt=learned_opt,
      trunc_sched=trunc_sched,
      theta=theta,
      key=key,
      state=state,
      data=data,
      outer_state=outer_state,
      meta_loss_with_aux_key=meta_loss_with_aux_key,
      override_num_steps=override_num_steps)


@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "trunc_sched",
                     "meta_loss_with_aux_key"))
@functools.partial(
    jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, None, None, None))
def truncated_unroll_one_step_vec_theta(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: TruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key: Optional[str],
    override_num_steps: Optional[int],
) -> Tuple[TruncatedUnrollState, TruncatedUnrollOut]:
  """Perform one step of inner training with vectorized theta."""
  return _truncated_unroll_one_step(
      task_family=task_family,
      learned_opt=learned_opt,
      trunc_sched=trunc_sched,
      theta=theta,
      key=key,
      state=state,
      data=data,
      outer_state=outer_state,
      meta_loss_with_aux_key=meta_loss_with_aux_key,
      override_num_steps=override_num_steps)


@gin.configurable
class VectorizedLOptTruncatedStep_CHEN(truncated_step.VectorizedTruncatedStep,
                                  full_es.OverrideStepVectorizedTruncatedStep):
  """Chen-style vectorized truncated step (no buffer, single imt_loss)."""

  def __init__(
      self,
      task_family: tasks_base.TaskFamily,
      learned_opt: lopt_base.LearnedOptimizer,
      trunc_sched: truncation_schedule.TruncationSchedule,
      num_tasks: int,
      meta_loss_split: Optional[str] = None,
      random_initial_iteration_offset: int = 0,
      outer_data_split="train",
      meta_loss_with_aux_key: Optional[str] = None,
      task_name: Optional[str] = None,
      random_initial_iteration_offset_linspace: bool = False,
      global_num_particles: int = 1,
  ):
    self.task_family = task_family
    self.learned_opt = learned_opt
    self.trunc_sched = trunc_sched
    self.num_tasks = num_tasks
    self.meta_loss_split = meta_loss_split
    self.random_initial_iteration_offset = random_initial_iteration_offset
    self.outer_data_split = outer_data_split
    self.meta_loss_with_aux_key = meta_loss_with_aux_key
    self._task_name = task_name
    self.timings = []
    self.random_initial_iteration_offset_linspace = random_initial_iteration_offset_linspace
    self.global_num_particles = global_num_particles

    self.data_shape = jax.tree_util.tree_map(
        lambda x: jax.core.ShapedArray(shape=x.shape, dtype=x.dtype),
        training.vec_get_batch(
            task_family, num_tasks, split="train", numpy=True))

  def outer_init(self, key):
    return self.learned_opt.init(key)

  def task_name(self):
    if self._task_name is None:
      return self.task_family.name
    else:
      return self._task_name

  def cfg_name(self):
    return self.learned_opt.name

  def init_step_state(self,
                      theta,
                      outer_state,
                      key,
                      theta_is_vector=False,
                      num_steps_override=None):
    if theta_is_vector:
      init_fn = init_truncation_state_vec_theta
    else:
      init_fn = init_truncation_state

    key1, key2 = jax.random.split(key)
    unroll_state = init_fn(self.task_family, self.learned_opt, self.trunc_sched, theta, outer_state, jax.random.split(key1, self.num_tasks), num_steps_override)
    # When initializing, we want to keep the trajectories not all in sync.
    if self.random_initial_iteration_offset:
      if self.random_initial_iteration_offset_linspace:
        curr_num_particles = unroll_state.inner_step.shape[0]

        if self.global_num_particles > curr_num_particles:
          all_offsets = jnp.linspace(0,
                                    self.random_initial_iteration_offset-1,
                                    num=self.global_num_particles,
                                    dtype=unroll_state.inner_step.dtype)
          idx = jax.process_index()
          start_idx = idx * curr_num_particles
          end_idx = (idx + 1) * curr_num_particles
          offsets = all_offsets[start_idx:end_idx]
          inner_step = jnp.asarray(offsets, dtype=unroll_state.inner_step.dtype)
        else:
          offsets = jnp.linspace(0,
                                self.random_initial_iteration_offset-1,
                                num=unroll_state.inner_step.shape[0],
                                dtype=unroll_state.inner_step.dtype)
          inner_step = jnp.asarray(offsets, dtype=unroll_state.inner_step.dtype)
      else:
        inner_step = jax.random.randint(
            key2,
            unroll_state.inner_step.shape,
            0,
            self.random_initial_iteration_offset,
            dtype=unroll_state.inner_step.dtype)
      unroll_state = unroll_state.replace(inner_step=inner_step)

    return unroll_state

  def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        diff = end_time - start_time
        args[0].timings.append(diff)
        return result
    return wrapper

  @timing_decorator
  def get_batch(self, steps: Optional[int] = None):
    if steps is not None:
      data_shape = (steps, self.num_tasks)
    else:
      data_shape = (self.num_tasks,)
    tr_batch = training.get_batches(
        self.task_family,
        data_shape,
        numpy=True,
        split="train")

    if self.meta_loss_split == "same_data" or self.meta_loss_split is None:
      return tr_batch
    else:
      outer_batch = training.get_batches(
          self.task_family, data_shape, numpy=True, split=self.meta_loss_split)
      return (tr_batch, outer_batch)

  def get_outer_batch(self, steps: Optional[int] = None):
    if steps is not None:
      data_shape = (steps, self.num_tasks)
    else:
      data_shape = (self.num_tasks,)
    return training.get_batches(
        self.task_family, data_shape, numpy=True, split=self.outer_data_split)

  def unroll_step(self,
                  theta,
                  unroll_state,
                  key,
                  data,
                  outer_state,
                  theta_is_vector=False,
                  override_num_steps: Optional[int] = None):
    if self.meta_loss_split == "same_data":
      tr_data = data
      meta_data = data
    elif self.meta_loss_split is None:
      tr_data = data
      meta_data = None
    else:
      tr_data, meta_data = data

    key1, key2 = jax.random.split(key)

    num_tasks_in_state = tree_utils.first_dim(unroll_state)
    if num_tasks_in_state == self.num_tasks * 2:
      stack_antithetic_samples = True
    else:
      stack_antithetic_samples = False

    vec_keys = jax.random.split(key1, self.num_tasks)
    if stack_antithetic_samples:
      vec_keys = jax.tree_util.tree_map(
          lambda a: jnp.concatenate([a, a], axis=0), vec_keys)

    fn = truncated_unroll_one_step_vec_theta if theta_is_vector else truncated_unroll_one_step
    next_unroll_state_, ys = fn(self.task_family, self.learned_opt,
                                self.trunc_sched, theta, vec_keys, unroll_state,
                                tr_data, outer_state,
                                self.meta_loss_with_aux_key, override_num_steps)

    if meta_data is not None:
      vec_keys = jax.random.split(key2, self.num_tasks)
      if stack_antithetic_samples:
        vec_keys = jax.tree_util.tree_map(
            lambda a: jnp.concatenate([a, a], axis=0), vec_keys)
      loss, aux = vectorized_loss_and_aux(self.task_family, self.learned_opt,
                                          theta,
                                          next_unroll_state_.inner_opt_state,
                                          next_unroll_state_.task_param,
                                          vec_keys, meta_data)
      if self.meta_loss_with_aux_key:
        ys = ys.replace(task_loss=aux[self.meta_loss_with_aux_key])
      else:
        ys = ys.replace(task_loss=loss)

    @jax.vmap
    def norm(loss, task_param):
      return self.task_family.task_fn(task_param).normalizer(loss)
    ys = ys.replace(task_loss=norm(ys.task_loss, unroll_state.task_param))
    return next_unroll_state_, ys

  def meta_loss_batch(self,
                      theta: Any,
                      unroll_state: Any,
                      key: Any,
                      data: Any,
                      outer_state: Any,
                      theta_is_vector: bool = False):
    keys = jax.random.split(key, self.num_tasks)
    loss, aux_metrics = vectorized_loss_and_aux(self.task_family,
                                                self.learned_opt, theta,
                                                unroll_state.inner_opt_state,
                                                unroll_state.task_param, keys,
                                                data)

    if self.meta_loss_with_aux_key:
      return aux_metrics[self.meta_loss_with_aux_key]
    else:

      @jax.vmap
      def norm(loss, task_param):
        return self.task_family.task_fn(task_param).normalizer(loss)

      loss = norm(loss, unroll_state.task_param)

      return loss
