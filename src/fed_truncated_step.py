# Adapted from https://github.com/google/learned_optimization/blob/main/learned_optimization/outer_trainers/lopt_truncated_step.py

import functools
from typing import Any, Callable, Optional, Tuple

import globals
import time
import gin
import jax
import optax
import flax
import jax.numpy as jnp
import numpy as np
from functools import partial
import flax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

from haiku._src.data_structures import FlatMap
from learned_optimization import summary, training, tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.optimizers import optax_opts
from learned_optimization.outer_trainers import (
    full_es,
    truncated_step,
    truncation_schedule,
)
from learned_optimization.outer_trainers.lopt_truncated_step import (
    G,
    PRNGKey,
    T,
    # TruncatedUnrollState,
    # init_truncation_state,
    # init_truncation_state_vec_theta,
    vectorized_loss_and_aux,
)
from learned_optimization.tasks import base as tasks_base

from opt import AnyOptimizer
# @flax.struct.dataclass
# class TruncatedUnrollState:
#   inner_opt_state: Any
#   inner_step: jnp.ndarray
#   truncation_state: Any
#   task_param: Any
#   is_done: jnp.ndarray
#   local_opt_state: Any

@flax.struct.dataclass
class LocalTruncatedUnrollState:
  inner_opt_state: Any
  local_inner_opt_state: Any
  inner_step: jnp.ndarray
  truncation_state: Any
  task_param: Any
  is_done: jnp.ndarray
  bc_grad: Optional = None

@functools.partial(
    jax.jit, static_argnames=("task_family", 
    "learned_opt", 
    "trunc_sched", 
    "local_optimizer",
    "use_bc_grads"))
@functools.partial(jax.vmap, in_axes=(None, None, None, None, None, 0, None, None, None, None))
def init_truncation_state(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    outer_state: Any,
    key: PRNGKey,
    local_optimizer: str = "sgd",
    local_learning_rate: float = 1e-1,
    num_steps_override: Optional[int] = None,
    use_bc_grads: bool = False) -> LocalTruncatedUnrollState:
  """Init inner state without vectorized theta."""
  return _init_truncation_state(task_family, learned_opt, trunc_sched, theta,
                                outer_state, key, local_optimizer, local_learning_rate, num_steps_override,
                                use_bc_grads)


@functools.partial(
    jax.jit, static_argnames=(
        "task_family", 
        "learned_opt", 
        "trunc_sched", 
        "local_optimizer",
        "use_bc_grads"))
@functools.partial(jax.vmap, in_axes=(None, None, None, 0, None, 0, None, None, None, None))
def init_truncation_state_vec_theta(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    outer_state: Any,
    key: PRNGKey,
    local_optimizer: str = "sgd",
    local_learning_rate: float = 1e-1,
    num_steps_override: Optional[int] = None,
    use_bc_grads: bool = False) -> LocalTruncatedUnrollState:
  """Init inner state with vectorized theta."""
  return _init_truncation_state(task_family, learned_opt, trunc_sched, theta,
                                outer_state, key, local_optimizer, local_learning_rate, num_steps_override,
                                use_bc_grads)




def _init_truncation_state(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    outer_state: Any,
    key: PRNGKey,
    local_optimizer: str = "sgd",
    local_learning_rate: float = 1e-1,
    num_steps_override: Optional[int] = None,
    use_bc_grads: bool = False) -> LocalTruncatedUnrollState:
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
  try: 
    local_opt = learned_opt.get_local_optimizer(
        inner_state['mup_lrs_to_use']
    )
  except KeyError:
    # if mup_lrs is not provided, use the default optimizer
    local_opt = learned_opt.get_local_optimizer(None)

  local_inner_opt_state = local_opt.init(inner_param, model_state=inner_state)
  local_inner_opt_state = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * globals.num_grads),
        local_inner_opt_state)


  if use_bc_grads:
    grad_init = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), theta)
    grad_init['bc_loss'] = jnp.array(0.0)
  else:
    grad_init = jnp.array(0.0)

  return LocalTruncatedUnrollState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      inner_opt_state=opt_state,
      local_inner_opt_state=local_inner_opt_state,
      inner_step=jnp.asarray(0, dtype=jnp.int32),
      truncation_state=trunc_state,
      task_param=task_param,
      is_done=False,
      bc_grad=grad_init
  )



def progress_or_reset_inner_opt_state_fedlopt(
    task_family: tasks_base.TaskFamily,
    opt: opt_base.Optimizer,
    num_steps: int,
    key: PRNGKey,
    inner_opt_state: T,
    local_inner_opt_state: T,
    task_param: G,
    inner_step: int,
    is_done: bool,
    data: Any,
    cond_fn: Callable[[bool, Any, Any, Any], Any] = jax.lax.cond,
    axis_name: Optional[str] = None,
    meta_loss_with_aux_key: Optional[str] = None,
    local_optimizer: str = "sgd",
    local_learning_rate: float = 1e-1,
    num_local_steps: int = 4,
    use_bc_grads: bool = False,
) -> Tuple[T, G, int, jnp.ndarray]:
    """Train a single step, or reset the current inner problem."""
    summary.summary(
        "num_steps", num_steps, aggregation="sample"
    )  # pytype: disable=wrong-arg-types  # jax-ndarray

    task = task_family.task_fn(task_param)
  
    try:
        local_opt = opt.get_local_optimizer(task.get_mup_state({})['mup_lrs_to_use'])
    except KeyError:
        # if mup_lrs is not provided, use the default optimizer
        local_opt = opt.get_local_optimizer(None)


    @functools.partial(jax.jit)
    def local_step(local_opt_state_and_key, local_batch):
        local_opt_state, key = local_opt_state_and_key
        params = local_opt.get_params(local_opt_state)
        key, key1 = jax.random.split(key)

        if globals.needs_state:
            state = local_opt.get_state(local_opt_state)
            (l, model_s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, local_batch)
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key1, local_batch)
            model_s = None

        return (local_opt.update(local_opt_state, grad, loss=l, model_state=model_s), key), l

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def vmap_local_updates_k(init_local_opt_state, key, client_batch):
        return jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)
    
    devices = mesh_utils.create_device_mesh((globals.num_devices,1))
    mesh = Mesh(devices, ('i', 'j'))
    @functools.partial(jax.jit)
    @functools.partial(shard_map, 
                        mesh=mesh, 
                        in_specs=(P(),P('i',None),P('i',None),), 
                        out_specs=(P('i'),
                                    P('i'),
                                    P('i'),
                                    )
                        )
    def shard_map_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = vmap_local_updates_k(init_local_opt_state, key, client_batch)

        delta = jax.tree_util.tree_map(
            lambda new_p, old_p: new_p - old_p,
            local_opt.get_params(final_local_opt_state),
            local_opt.get_params(init_local_opt_state),
        )
        return (
            local_losses, 
            delta, 
            local_opt.get_state(final_local_opt_state) if globals.needs_state else s, 
            final_local_opt_state
        )

    @functools.partial(jax.pmap, in_axes=(None, 0, 0), out_axes=(None, 0, None, None), axis_name="num_grads")
    def pmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)
        delta = jax.tree_util.tree_map(
            lambda new_p, old_p: new_p - old_p,
            local_opt.get_params(final_local_opt_state),
            local_opt.get_params(init_local_opt_state),
        )
        return (
            jax.lax.pmean(jnp.mean(local_losses), axis_name="num_grads"),
            delta,
            jax.lax.pmean(delta, axis_name="num_grads"),
            jax.lax.pmean(local_opt.get_state(final_local_opt_state), axis_name="num_grads") if globals.needs_state else s,
            final_local_opt_state
        )

    @functools.partial(jax.vmap, in_axes=(0, 0, 0))
    def vmap_local_updates(init_local_opt_state, key, client_batch):

        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        # print("len final_local_opt_state", len(final_local_opt_state))
        # print("nu shape inside",jax.tree_util.tree_map(lambda x: x.shape, final_local_opt_state.optax_opt_state[0].nu))

        return (
            jnp.mean(local_losses),
            jax.tree_util.tree_map(
                lambda new_p, old_p: new_p - old_p,
                local_opt.get_params(final_local_opt_state),
                local_opt.get_params(init_local_opt_state),
            ),
            local_opt.get_state(final_local_opt_state) if globals.needs_state else s,
            final_local_opt_state
        )
            


    def true_fn(args):
        key, local_inner_opt_state = args

        """Reset the state of the inner-problem."""
        # When training with pmap, we want to sync keys over the axis
        # to ensure they are all in sync.
        if axis_name:
            key = jax.lax.all_gather(key, axis_name)[0]

        key1, key2, key3 = jax.random.split(key, 3)
        task_param = task_family.sample(key1)
        p, s = task_family.task_fn(task_param).init_with_state(key2)

        next_inner_opt_state = opt.init(p, s, num_steps=num_steps, key=key3)

        next_local_opt_state = local_opt.init(p, model_state=s)
        next_local_opt_state = jax.tree_util.tree_map(lambda x: jnp.stack([x] * globals.num_grads), next_local_opt_state)
        summary.summary(
            "opt_init_num_steps", num_steps
        )  # pytype: disable=wrong-arg-types  # jax-ndarray
        if use_bc_grads:
            bc_grad = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), opt.theta)
            bc_grad['bc_loss'] = jnp.array(0.0)
        else:
            bc_grad = jnp.array(0.0)

        local_inner_opt_state = local_opt.init(p, model_state=s)
        #TODO(benjamin): remove  globals
        local_inner_opt_state = jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * globals.num_grads),
            local_inner_opt_state
        )

        return next_inner_opt_state, local_inner_opt_state, task_param, jnp.asarray(0), jnp.asarray(0.0), bc_grad

    def false_fn(args):
        key, init_local_opt_state = args
        """Train one step of the inner-problem."""
        p = opt.get_params(inner_opt_state)
        s = opt.get_state(inner_opt_state)
        key1, key2 = jax.random.split(key)
        if meta_loss_with_aux_key:
            raise NotImplementedError("Meta-training with an auxiliary metric is not implemented for FedLOptTruncatedStep.")
            # If we are meta-training with an auxiliary metric, we must compute them.

            def loss_fn(p, s, key, data):
                """Wrapper around loss_with_state_and_aux to return 2 values."""
                l, s, aux = task.loss_with_state_and_aux(p, s, key, data)
                return l, (s, aux)

            (l, (s, aux)), g = jax.value_and_grad(loss_fn, has_aux=True)(
                p, s, key1, data
            )

            if meta_loss_with_aux_key:
                if meta_loss_with_aux_key not in aux:
                    raise ValueError(
                        f"Aux key: {meta_loss_with_aux_key} not found in "
                        f"task family {task_family}. Found keys are "
                        f" {aux.keys()}"
                    )
            meta_loss = aux[meta_loss_with_aux_key]
        else:
            ######################################################################################################################################################
            # FIX
            # global_p = jax.tree_util.tree_map(lambda x: jnp.stack([x] * globals.num_grads), p)
            # global_s = jax.tree_util.tree_map(lambda x: jnp.stack([x] * globals.num_grads), s)

            # init_local_opt_state = local_opt.resume_init(opt_state=init_local_opt_state, params=global_p, model_state=global_s)
            ##################################################


            splitted_batches = jax.tree_util.tree_map(lambda x : x.reshape(
                (globals.num_grads, globals.num_local_steps, globals.local_batch_size) + x.shape[1:]), data)

            keys = jax.random.split(key2, globals.num_grads)
            if globals.use_pmap:
                ####################################################################################################
                # Distributed local steps
                ####################################################################################################
                losses, deltas, new_state, final_local_opt_state = shard_map_local_updates(local_inner_opt_state, keys, splitted_batches)

                l = jnp.mean(losses)
                avg_delta = jax.tree_util.tree_map(
                    lambda ds: jnp.mean(ds, axis=0), deltas
                )
                if globals.needs_state:
                    avg_state = jax.tree_util.tree_map(lambda ns: jnp.mean(ns, axis=0),new_state)
                else:
                    avg_state = s

            else:
                ####################################################################################################
                # single device local steps
                ####################################################################################################
                params = local_opt.get_params(local_inner_opt_state)
                print("type of params>>>>:", type(params))
                print("params keys (if dict)>>>>:", params.keys() if isinstance(params, dict) else "not a dict")
                print(jax.tree_util.tree_map(lambda x: x.shape, splitted_batches))
                losses, deltas, new_state, final_local_opt_state = vmap_local_updates(local_inner_opt_state, keys, splitted_batches)
                l = jnp.mean(losses)
                avg_delta = jax.tree_util.tree_map(
                        lambda ds: jnp.mean(ds, axis=0), deltas
                )
                # avg_next_local_opt_state = jax.tree_util.tree_map(lambda nnlos: jnp.mean(nnlos, axis=0, dtype=nnlos.dtype), next_local_opt_state)
                if globals.needs_state:
                    avg_state = jax.tree_util.tree_map(
                        lambda ns: jnp.mean(ns, axis=0),
                        new_state,
                    )
                else:
                    avg_state = s
            meta_loss = l
        
        # print("nu shape after vmap",jax.tree_util.tree_map(lambda x: x.shape, final_local_opt_state.optax_opt_state[0].nu))
        
        if axis_name:
            g = jax.lax.pmean(g, axis_name=axis_name)
            l = jax.lax.pmean(l, axis_name=axis_name)

        summary.summary("task_loss", l)
        if use_bc_grads:
            next_inner_opt_state, bc_grad = opt.update(inner_opt_state, deltas, avg_delta, loss=l, model_state=avg_state, key=key2)            
        else:
            next_inner_opt_state = opt.update(inner_opt_state, deltas, avg_delta, loss=l, model_state=avg_state, key=key2)
            bc_grad = jnp.array(0.0)

        ####################################################################################################
        # update local_inner_opt_state
        ####################################################################################################\
        global_s = jax.tree_util.tree_map(lambda x: jnp.stack([x] * globals.num_grads), avg_state)
        global_p = jax.tree_util.tree_map(lambda x: jnp.stack([x] * globals.num_grads), next_inner_opt_state.params)
        final_local_opt_state = local_opt.resume_init(opt_state=final_local_opt_state, params=global_p, model_state=global_s)

        next_inner_step = inner_step + 1

        return (
            next_inner_opt_state,
            final_local_opt_state,
            task_param,
            next_inner_step,
            jnp.asarray(meta_loss, dtype=jnp.float32),
            bc_grad
        )

    # jax.debug.print("is_done shape: {x}", x=is_done)
    next_inner_opt_state, local_inner_opt_state, task_param, next_inner_step, meta_loss, bc_grad = cond_fn(
        jnp.logical_not(is_done), false_fn, true_fn, (key, local_inner_opt_state)
    )

    return next_inner_opt_state, local_inner_opt_state,task_param, next_inner_step, meta_loss, bc_grad



def _truncated_unroll_one_step_fedlopt(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: LocalTruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key,
    override_num_steps: Optional[int] = None,
    local_optimizer: str = "sgd",
    local_learning_rate: float = 1e-1,
    num_local_steps: int = 4,
    use_bc_grads: bool = False,
) -> Tuple[LocalTruncatedUnrollState, truncated_step.TruncatedUnrollOut]:
    """Train a given inner problem state a single step or reset it when done."""
    key1, key2 = jax.random.split(key)

    if override_num_steps is not None:
        num_steps = override_num_steps
    else:
        num_steps = state.truncation_state.length

    (
        next_inner_opt_state,
        local_inner_opt_state,
        task_param,
        next_inner_step,
        l,
        bc_grad,
    ) = progress_or_reset_inner_opt_state_fedlopt(  # pytype: disable=wrong-arg-types  # jax-ndarray
        task_family=task_family,
        opt=learned_opt.opt_fn(theta),
        num_steps=num_steps,
        key=key1,
        inner_opt_state=state.inner_opt_state,
        local_inner_opt_state=state.local_inner_opt_state,
        task_param=state.task_param,
        inner_step=state.inner_step,
        is_done=state.is_done,
        data=data,
        meta_loss_with_aux_key=meta_loss_with_aux_key,
        local_optimizer=local_optimizer,
        local_learning_rate=local_learning_rate,
        num_local_steps=num_local_steps,
        use_bc_grads=use_bc_grads,
    )

    next_truncation_state, is_done = trunc_sched.next_state(
        state.truncation_state, next_inner_step, key2, outer_state
    )

    # summaries
    opt = learned_opt.opt_fn(theta, is_training=True)
    summary.summarize_inner_params(opt.get_params(next_inner_opt_state))

    output_state = (
        LocalTruncatedUnrollState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            inner_opt_state=next_inner_opt_state,
            local_inner_opt_state=local_inner_opt_state,
            inner_step=next_inner_step,
            truncation_state=next_truncation_state,
            task_param=task_param,
            is_done=is_done,
            bc_grad=bc_grad
        )
    )

    out = truncated_step.TruncatedUnrollOut(  # pytype: disable=wrong-arg-types  # jax-ndarray
        is_done=is_done,
        loss=l,
        mask=(next_inner_step != 0),
        iteration=next_inner_step,
        task_param=state.task_param,
    )

    return output_state, out


@functools.partial(
    jax.jit,
    static_argnames=(
        "task_family",
        "learned_opt",
        "trunc_sched",
        "meta_loss_with_aux_key",
        "local_optimizer",
        "local_learning_rate",
        "num_local_steps",
    ),
)
@functools.partial(
    jax.vmap, in_axes=(None, None, None, None, 0, 0, 0, None, None, None, None, None, None, None)
)
def truncated_unroll_one_step_fedlopt(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: LocalTruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key: Optional[str],
    override_num_steps: Optional[int],
    local_optimizer: str = "sgd",
    local_learning_rate: float = 1e-1,
    num_local_steps: int = 4,
    use_bc_grads: bool = False,
) -> Tuple[LocalTruncatedUnrollState, truncated_step.TruncatedUnrollOut]:
    """Perform one step of inner training without vectorized theta."""
    return _truncated_unroll_one_step_fedlopt(
        task_family=task_family,
        learned_opt=learned_opt,
        trunc_sched=trunc_sched,
        theta=theta,
        key=key,
        state=state,
        data=data,
        outer_state=outer_state,
        meta_loss_with_aux_key=meta_loss_with_aux_key,
        override_num_steps=override_num_steps,
        local_optimizer=local_optimizer,
        local_learning_rate=local_learning_rate,
        num_local_steps=num_local_steps,
        use_bc_grads=use_bc_grads,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "task_family",
        "learned_opt",
        "trunc_sched",
        "meta_loss_with_aux_key",
        "local_optimizer",
        "local_learning_rate",
        "num_local_steps",
        "use_bc_grads",
    ),
)
@functools.partial(
    jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None)
)
def truncated_unroll_one_step_vec_theta_fedlopt(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: LocalTruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key: Optional[str],
    override_num_steps: Optional[int],
    local_optimizer: str = "sgd",
    local_learning_rate: float = 1e-1,
    num_local_steps: int = 4,
    use_bc_grads: bool = False,
) -> Tuple[LocalTruncatedUnrollState, truncated_step.TruncatedUnrollOut]:
    """Perform one step of inner training with vectorized theta."""
    return _truncated_unroll_one_step_fedlopt(
        task_family=task_family,
        learned_opt=learned_opt,
        trunc_sched=trunc_sched,
        theta=theta,
        key=key,
        state=state,
        data=data,
        outer_state=outer_state,
        meta_loss_with_aux_key=meta_loss_with_aux_key,
        override_num_steps=override_num_steps,
        local_optimizer=local_optimizer,
        local_learning_rate=local_learning_rate,
        num_local_steps=num_local_steps,
        use_bc_grads=use_bc_grads,
    )


@functools.partial(jax.jit)
def shuffle_batch(batch, key):
    return jax.tree_util.tree_map(partial(reshape_permute_reshape, key), batch)

def reshape_idx_reshape(idx, x):
    old_shape = x.shape
    new_shape = (np.prod(old_shape[:3]),) + old_shape[3:]
    return x.reshape(new_shape)[idx].reshape(old_shape)

@functools.partial(jax.jit)
def reshape_permute_reshape(key, x):
    old_shape = x.shape
    new_shape = (np.prod(old_shape[:3]),) + old_shape[3:]
    return jax.random.permutation(key,x.reshape(new_shape),axis=0).reshape(old_shape)

@gin.configurable
class VectorizedFedLOptTruncatedStep(
    truncated_step.VectorizedTruncatedStep, full_es.OverrideStepVectorizedTruncatedStep
):
    """VectorizedTruncatedStep for learned optimizer inner training.
    This is more fully featured than VectorizedLOptTruncated step allowing for
    both task_family (rather than a single task), and truncation schedules.
    """

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
        local_optimizer: str = "sgd",
        local_learning_rate: float = 1e-1,
        num_local_steps: int = 4,
        keep_batch_in_gpu_memory = False,
        use_bc_grads: bool = False,
    ):
        """Initializer.
        Args:
          task_family: task family to do unrolls on.
          learned_opt: learned optimizer instance.
          trunc_sched: truncation schedule to use.
          num_tasks: number of tasks to vmap over.
          meta_loss_split: This can take 3 values: None, 'same_data', or a
            dataset split: {"train", "outer_valid", "inner_valid", "test"}.
            If set to a dataset split we use a new batch of data to compute the
            meta-loss which is evaluated on the newly created inner state (after
            applying the lopt.). If set to 'same_data', the same data is reused to
            evaluate the meta-loss. If None no additional computation is performed
            and the previous state's loss evaluated on the training batch is used.
          random_initial_iteration_offset: An initial offset for the inner-steps of
            each task. This is to prevent all tasks running in lockstep. This should
            be set to the max number of steps the truncation schedule.
          outer_data_split: Split of data to use when computing meta-losses.
          meta_loss_with_aux_key: Instead of using the loss, use a value from the
            returned auxiliary data.
          task_name: Optional string used to prefix summary.
            If not set, the name of the task family is used.
        """
        self.task_family = task_family
        self.learned_opt = learned_opt
        self.trunc_sched = trunc_sched
        self.num_tasks = num_tasks
        self.meta_loss_split = meta_loss_split
        self.random_initial_iteration_offset = random_initial_iteration_offset
        self.outer_data_split = outer_data_split
        self.meta_loss_with_aux_key = meta_loss_with_aux_key
        self._task_name = task_name
        self.local_optimizer = local_optimizer
        self.local_learning_rate = local_learning_rate
        self.num_local_steps = num_local_steps

        # for in memory batch
        self.keep_batch_in_gpu_memory = keep_batch_in_gpu_memory
        self.batch = None
        self.idx = None
        jax.debug.print('>>>>>>>>>>using: {}<<<<<<<<<<', self.local_optimizer)
        self.use_bc_grads = use_bc_grads
        # self.data_shape = jax.tree_util.tree_map(
        #     lambda x: jax.core.ShapedArray(shape=x.shape, dtype=x.dtype),
        #     training.vec_get_batch(task_family, num_tasks, split="train", numpy=True)
        # )
        self.timings = []

    def outer_init(self, key):
        return self.learned_opt.init(key)

    def task_name(self):
        if self._task_name is None:
            return self.task_family.name
        else:
            return self._task_name

    def cfg_name(self):
        return self.learned_opt.name

    def init_step_state(
        self, theta, outer_state, key, theta_is_vector=False, num_steps_override=None
    ):
        if theta_is_vector:
            init_fn = init_truncation_state_vec_theta
        else:
            init_fn = init_truncation_state

        key1, key2 = jax.random.split(key)
        # jax.debug.print('self.local_learning_rate: {}, num_steps_override: {}', self.local_learning_rate, num_steps_override)
        unroll_state = init_fn(
            self.task_family,
            self.learned_opt,
            self.trunc_sched,
            theta,
            outer_state,
            jax.random.split(key1, self.num_tasks),
            self.local_optimizer,
            self.local_learning_rate,
            num_steps_override,
            self.use_bc_grads
        )
        # When initializing, we want to keep the trajectories not all in sync.
        # To do this, we can initialize with a random offset on the inner-step.
        if self.random_initial_iteration_offset:
            inner_step = jax.random.randint(
                key2,
                unroll_state.inner_step.shape,
                0,
                self.random_initial_iteration_offset,
                dtype=unroll_state.inner_step.dtype,
            )
            unroll_state = unroll_state.replace(inner_step=inner_step)

        return unroll_state
    
    def timing_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            diff = end_time - start_time
            # print(f"{func.__name__} took {diff} seconds to complete.")
            args[0].timings.append(diff)
            return result

        return wrapper

    def shuffle_batch(self):

        assert self.batch != None, 'self.batch is None, cannot shuffle None'

        self.batch = jax.tree_util.tree_map(partial(reshape_permute_reshape, jax.random.key()), self.batch)
        return self.batch


        # if self.idx == None:
        #     self.idx = jnp.arange(np.prod(self.batch['label'].shape))
        # self.idx = jax.random.permutation(jax.random.key(0),self.idx)

        # return jax.tree_util.tree_map(partial(reshape_idx_reshape, self.idx), self.batch)

    @timing_decorator
    def get_batch(self, steps: Optional[int] = None):
        """Get a batch of data for training. This is used within the gradient estimator
        to sample data for the inner training loop."""
        if steps is not None:
            data_shape = (steps, self.num_tasks)
        else:
            data_shape = (self.num_tasks,)

        if self.keep_batch_in_gpu_memory:
            
            if self.batch is None:
                tr_batch = next(self.task_family.datasets.split("train"))
                self.batch = tr_batch
            else:
                self.batch = shuffle_batch(self.batch, jax.random.key(0))
                tr_batch = self.batch
        else:
            tr_batch = next(self.task_family.datasets.split("train"))


        if self.meta_loss_split == "same_data" or self.meta_loss_split is None:
            return tr_batch
        else:
            # print('in outer abtch')
            outer_batch = next(self.task_family.datasets.split(self.meta_loss_split))
            # training.get_batches(
            #     self.task_family, data_shape, numpy=True, split=self.meta_loss_split
            # )
            return (tr_batch, outer_batch)
        



    def get_outer_batch(self, steps: Optional[int] = None):
        if steps is not None:
            data_shape = (steps, self.num_tasks)
        else:
            data_shape = (self.num_tasks,)

        return training.get_batches(
            self.task_family, data_shape, numpy=True, split=self.outer_data_split
        )

    def unroll_step(
        self,
        theta,
        unroll_state,
        key,
        data,
        outer_state,
        theta_is_vector=False,
        override_num_steps: Optional[int] = None,
    ):
    
        # per-step data changes depending on if we use a extra eval batch per step.
        if self.meta_loss_split == "same_data":
            # use same batch of data
            tr_data = data
            meta_data = data
        elif self.meta_loss_split is None:
            tr_data = data
            meta_data = None
        else:
            # Otherwise assume we passed a valid data split.
            tr_data, meta_data = data

        key1, key2 = jax.random.split(key)

        # This function is designed to be called with the unroll_state having the
        # same number of tasks as created initially. One can, however, call it with
        # with a bigger batchsize representing 2 perturbations stacked together.
        # When doing this, we want to share randomness across these 2 batches
        # as they are antithetic samples.
        # TODO(lmetz) consider passing stack_antithetic_samples in some capacity
        # rather than guessing it here.
        num_tasks_in_state = tree_utils.first_dim(unroll_state)
        if num_tasks_in_state == self.num_tasks * 2:
            stack_antithetic_samples = True
        else:
            stack_antithetic_samples = False

        # If stacking the antithetic samples, we want to share random keys across
        # the antithetic samples.
        vec_keys = jax.random.split(key1, self.num_tasks)
        if stack_antithetic_samples:
            vec_keys = jax.tree_util.tree_map(
                lambda a: jnp.concatenate([a, a], axis=0), vec_keys
            )

        fn = (
            truncated_unroll_one_step_vec_theta_fedlopt
            if theta_is_vector
            else truncated_unroll_one_step_fedlopt
        )
        next_unroll_state_, ys = fn(
            self.task_family,
            self.learned_opt,
            self.trunc_sched,
            theta,
            vec_keys,
            unroll_state,
            tr_data,
            outer_state,
            self.meta_loss_with_aux_key,
            override_num_steps,
            self.local_optimizer,
            self.local_learning_rate,
            self.num_local_steps,
            self.use_bc_grads
        )
        # print("after first unroll")
        # exit(0)

        # Should we evaluate resulting state on potentially new data?
        if meta_data is not None:
            vec_keys = jax.random.split(key2, self.num_tasks)
            if stack_antithetic_samples:
                vec_keys = jax.tree_util.tree_map(
                    lambda a: jnp.concatenate([a, a], axis=0), vec_keys
                )
            loss, aux = vectorized_loss_and_aux(
                self.task_family,
                self.learned_opt,
                theta,
                next_unroll_state_.inner_opt_state,
                next_unroll_state_.task_param,
                vec_keys,
                meta_data,
            )
            if self.meta_loss_with_aux_key:
                ys = ys.replace(loss=aux[self.meta_loss_with_aux_key])
            else:
                ys = ys.replace(loss=loss)

        @jax.vmap
        def norm(loss, task_param):
            return self.task_family.task_fn(task_param).normalizer(loss)

        ys = ys.replace(loss=norm(ys.loss, unroll_state.task_param))

        return next_unroll_state_, ys

    def meta_loss_batch(
        self,
        theta: Any,
        unroll_state: Any,
        key: Any,
        data: Any,
        outer_state: Any,
        theta_is_vector: bool = False,
    ):
        keys = jax.random.split(key, self.num_tasks)
        loss, aux_metrics = vectorized_loss_and_aux(
            self.task_family,
            self.learned_opt,
            theta,
            unroll_state.inner_opt_state,
            unroll_state.task_param,
            keys,
            data,
        )

        if self.meta_loss_with_aux_key:
            return aux_metrics[self.meta_loss_with_aux_key]
        else:

            @jax.vmap
            def norm(loss, task_param):
                return self.task_family.task_fn(task_param).normalizer(loss)

            # Then normalize the losses to a sane meta-training range.
            loss = norm(loss, unroll_state.task_param)

            return loss
