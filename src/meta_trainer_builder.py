import jax
import optax
import chex
import flax
import gin
import jax.numpy as jnp
import numpy as onp
import time
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from learned_optimization.tasks.task_augmentation import ReparamWeightsFamily
from lopt_truncated_step_elo import VectorizedLOptTruncatedStep_ELO
from truncated_pes_custom_elo import TruncatedPES_ELO
from truncated_es_single_custom_elo import ESSingle_ELO
from lopt_truncated_step_chen import VectorizedLOptTruncatedStep_CHEN
from truncated_pes_custom_chen import TruncatedPES_CHEN
from learned_optimization.learned_optimizers.adafac_mlp_lopt import AdafacMLPLOpt
from learned_optimization.learned_optimizers.rnn_mlp_lopt import RNNMLPLOpt
from fed_truncated_step import VectorizedFedLOptTruncatedStep
from helpers import Timing, convert_config_to_dict
from optimizers import AdamWLinearCosine, AdamW
from opt import AnyOptimizer
from opt.new_optimizers import get_optax_schedule, DoubleAdam
from learned_optimization.outer_trainers.lopt_truncated_step import VectorizedLOptTruncatedStep
from learned_optimization.outer_trainers.truncation_schedule import TruncationSchedule, ConstantTruncationState, ConstantTruncationSchedule
from learned_optimization.research.general_lopt.hyper_v2 import HyperV2
from learned_optimization.outer_trainers import (
    truncated_pes,
    truncation_schedule,
    full_es,
    es_single,
)
import gin
from baseline_trajectories import load_trajectories
from learned_optimization.tasks import base as tasks_base
from learned_optimization.optimizers import learning_rate_schedules
from learned_optimizers import build_learned_optimizer
############################################################################
# Inner Problem Schedules 
############################################################################  
import inspect

def build_from_dict(cls, kwargs):
    sig = inspect.signature(cls)
    valid_keys = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
    return cls(**filtered)

GRADIENT_ESTIMATORS = {
    "FullES".lower(): full_es.FullES,
    "TruncatedPES".lower(): truncated_pes.TruncatedPES,
    "TruncatedPES_ELO".lower(): TruncatedPES_ELO,
    "TruncatedPES_CHEN".lower(): TruncatedPES_CHEN,
    "ESSingle".lower(): es_single.ESSingle,
    "ESSingle_ELO".lower(): ESSingle_ELO,
}

TRUNCATED_STEPS = {
    "VectorizedFedLOptTruncatedStep".lower(): VectorizedFedLOptTruncatedStep,
    "VectorizedLOptTruncatedStep_ELO".lower(): VectorizedLOptTruncatedStep_ELO,
    "VectorizedLOptTruncatedStep_CHEN".lower(): VectorizedLOptTruncatedStep_CHEN,
    "VectorizedLOptTruncatedStep".lower(): VectorizedLOptTruncatedStep,
}

class TruncationRampUp(learning_rate_schedules.ScalarSchedule):
    def __init__(self, 
                 max_unroll_length: int,
                 outer_steps: int,):
        self.max_unroll_length = max_unroll_length
        self.outer_steps = outer_steps
        # heristic to make the ramp up last for 1/2 of the outer steps
        self.increment_rate = 2 * ( max_unroll_length / outer_steps )
        self.max_init_steps = 1000

    def __call__(
      self,
      step: Union[int, chex.Array],
      max_steps: Optional[Union[int, chex.Array]] = None) -> chex.Array:
        
        num1 = self.max_init_steps + ( step * self.increment_rate)

        # jax.debug.print('step={x} curr={y} max={z}',x=step, y=num1, z=self.max_unroll_length)

        return jnp.min(
            jnp.array([num1, self.max_unroll_length],)
        )

class PESTruncationRampUp(learning_rate_schedules.ScalarSchedule):
    def __init__(self, 
                 trunc_ramp_up,
                 truncation_inner_problem_ratio=50):
        self.trunc_ramp_up = trunc_ramp_up
        self.truncation_inner_problem_ratio = truncation_inner_problem_ratio

    def __call__(
      self,
      step: Union[int, chex.Array]) -> chex.Array:
        curr = self.trunc_ramp_up(step)
        return int( curr / self.truncation_inner_problem_ratio )

@gin.configurable
class EnhancedLogUniformLengthSchedule(TruncationSchedule):
  """Sample unroll length from a log uniform distribution.

  This creates more samples with shorter unrolls.
  """

  def __init__(self, min_length, max_length):
    print(min_length, max_length)
    self._max_length = max_length
    self._min_length = min_length

  def init(self, key, outer_state):
    max_length = self._max_length(outer_state.outer_iteration)#.astype(jnp.int32)
    min_length = self._min_length(outer_state.outer_iteration)#.astype(jnp.int32)

    log_length = jax.random.uniform(
        key, [],
        jnp.float32,
        minval=jnp.log(min_length),
        maxval=jnp.log(max_length))
    length = jnp.asarray(jnp.exp(log_length), dtype=jnp.int64)
    return ConstantTruncationState(length=length)

  def next_state(self, state, step, key, outer_state):
    is_done = (step >= state.length)
    state = jax.lax.cond(is_done, lambda ss: self.init(*ss), lambda ss: state,
                     (key, outer_state))
    return state, is_done

@gin.configurable
class Curriculum_ConstantTruncationState(TruncationSchedule):
  def __init__(self, curriculum_lengths: Union[int, learning_rate_schedules.ScalarSchedule]):
    self._curriculum_lengths = jnp.asarray(curriculum_lengths)

  def init(self, key, outer_state, max_length: int = 100):
    return ConstantTruncationState(length=max_length)

  def update_state(self, key, outer_state, state=None, max_length: int = 100):
    curriculum_idx = jnp.minimum(
    state.curriculum_idx + 1,
    self._curriculum_lengths.shape[0]-1
    )
    return ConstantTruncationState(length=max_length, curriculum_idx=curriculum_idx)

  def next_state(self, state, step, key, outer_state):
    is_done = (step >= state.length-1)
    max_length = self._curriculum_lengths[state.curriculum_idx]
    state = jax.lax.cond(is_done, lambda ss: self.update_state(*ss), lambda ss: state,
                     (key, outer_state, state, max_length))
    return state, is_done


@flax.struct.dataclass
class Chen_ConstantTruncationState:
  # Target unroll length for the current truncation window.
  length: jnp.ndarray
  # Count of unrolls that have finished (incremented once per is_done).
  # Drives BC on/off alternation via expert_weight = 1 - (current_unroll % 2)
  # written into inner_opt_state by the PES worker each outer step.
  current_unroll: jnp.ndarray


@gin.configurable
class Chen_Curriculum_ConstantTruncationState(TruncationSchedule):
  """Chen-style curriculum: length grows by `increment` every `N_period` unrolls, capped at `max_length`."""

  def __init__(self, init_length: int, increment: int, N_period: int, max_length: int):
    self._init_length = int(init_length)
    self._increment = int(increment)
    self._N_period = int(N_period)
    self._max_length = int(max_length)

  def init(self, key, outer_state, max_length: int = 100):
    return Chen_ConstantTruncationState(
        length=jnp.asarray(self._init_length, jnp.int32),
        current_unroll=jnp.asarray(0, jnp.int32),
    )

  def _advance(self, state):
    new_N = state.current_unroll + jnp.int32(1)
    grow = jnp.equal(jnp.mod(new_N, jnp.int32(self._N_period)), jnp.int32(0))
    proposed = jnp.minimum(state.length + jnp.int32(self._increment), jnp.int32(self._max_length))
    new_length = jnp.where(grow, proposed, state.length)
    return Chen_ConstantTruncationState(length=new_length, current_unroll=new_N)

  def next_state(self, state, step, key, outer_state):
    is_done = (step >= state.length - 1)
    state = jax.lax.cond(is_done, self._advance, lambda s: s, state)
    return state, is_done

@gin.configurable
class UniformLengthSchedule(TruncationSchedule):
  def __init__(self, min_length: Union[int,
                                       learning_rate_schedules.ScalarSchedule],
               max_length: Union[int, learning_rate_schedules.ScalarSchedule]):
    self._max_length = max_length
    self._min_length = min_length

  def init(self, key, outer_state):
    if isinstance(self._max_length, learning_rate_schedules.ScalarSchedule):
      max_length = self._max_length(outer_state.outer_iteration)
    else:
      max_length = self._max_length

    if isinstance(self._min_length, learning_rate_schedules.ScalarSchedule):
      min_length = self._min_length(outer_state.outer_iteration)
    else:
      min_length = self._min_length

    length = jax.random.uniform(
        key, [],
        jnp.float32,
        minval=min_length,
        maxval=max_length)
    return ConstantTruncationState(length=length)

  def next_state(self, state, step, key, outer_state):
    is_done = (step >= state.length)
    state = jax.lax.cond(is_done, lambda ss: self.init(*ss), lambda ss: state,
                     (key, outer_state))
    return state, is_done


def build_meta_optimizer(args):
    """Build meta optimizer (theta_opt)."""
    if args.optimizer_args["class_"] == "DoubleAdam":
        kw = args.optimizer_args["kwargs"]
        return DoubleAdam(
            learning_rate=kw["learning_rate"],
            merging_rate=kw["merging_rate"],
            adam_bc=kw["adam_bc"],
            adam_es=kw["adam_es"],
            clip_norm=kw["clip_norm"],
        )

    return AnyOptimizer(
        optimizer=args.optimizer_args,
        schedule=args.schedule,
        gradient_transform_before_optim=args.gradient_transform_before_optim,
        gradient_transform_after_optim=args.gradient_transform_after_optim,
        mup_lrs=None,  # important: set None if optimizer not startswith "mu"
    )

@gin.configurable
def build_inner_length_schedule(args):
    if args.use_es:
        return truncation_schedule.NeverEndingTruncationSchedule()
    min_length = get_optax_schedule(args.inner_problem_length_schedule['min']['class_'], args.inner_problem_length_schedule['min']['kwargs'])
    max_length = get_optax_schedule(args.inner_problem_length_schedule['max']['class_'], args.inner_problem_length_schedule['max']['kwargs'])
    schedulers = {
        "constant": lambda: ConstantTruncationSchedule(
            total_length=args.inner_problem_length_schedule['max']['kwargs']['value']
        ),
        "uniform": lambda: UniformLengthSchedule(
            min_length=min_length,
            max_length=max_length,
        ),
        "log_uniform": lambda: EnhancedLogUniformLengthSchedule(
            min_length=min_length,
            max_length=max_length,
        ),
        "curriculum_constant": lambda: Curriculum_ConstantTruncationState(
            curriculum_lengths=args.inner_problem_length_schedule['curriculum_lengths']
        ),
        "chen_curriculum_constant": lambda: Chen_Curriculum_ConstantTruncationState(
            init_length=args.inner_problem_length_schedule['init_length'],
            increment=args.inner_problem_length_schedule['increment'],
            N_period=args.inner_problem_length_schedule['N_period'],
            max_length=args.inner_problem_length_schedule['max_length'],
        ),
    }

    try:
        return schedulers[args.inner_problem_length_schedule['sample_choice']]()
    except KeyError as e:
        raise ValueError(f"Unknown sample_choice: {args.inner_problem_length_schedule['sample_choice']}") from e

def build_truncated_step(args, task_family, learned_opt, inner_length_schedule):
    ts_class_name = args.truncated_step_args['class_']
    ts_kwargs = dict(args.truncated_step_args['kwargs'])  # shallow copy

    # Inject runtime objects that can't live in cfg
    ts_kwargs['task_family'] = task_family
    ts_kwargs['learned_opt'] = learned_opt
    ts_kwargs['trunc_sched'] = inner_length_schedule
    ts_kwargs['task_name'] = task_family.datasets.extra_info["name"]
    ts_kwargs['num_tasks'] = args.num_tasks
    ts_kwargs['num_local_steps'] = args.num_local_steps
    ts_kwargs['gradient_accumulation_steps'] = getattr(args, 'gradient_accumulation_steps', 1)
    ts_kwargs['random_initial_iteration_offset'] = args.num_inner_steps
    ts_kwargs['use_bc_grads'] = getattr(args, 'bc_grad_weight', None)
    ts_kwargs['global_num_particles'] = (
        args.world_size * ts_kwargs['num_tasks']
        if args.gradient_estimator_args['kwargs']['pmap_across_devices']
        else ts_kwargs['num_tasks']
    )

    ts_cls = TRUNCATED_STEPS[ts_class_name.lower()]
    return build_from_dict(ts_cls, ts_kwargs)
    # return ts_cls(**ts_kwargs)

def build_gradient_estimator(args, truncated_step, inner_length_schedule, baseline_losses=None):
    ge_class_name = args.gradient_estimator_args['class_']
    ge_kwargs = dict(args.gradient_estimator_args['kwargs'])  # shallow copy

    # Always inject truncated_step
    ge_kwargs['truncated_step'] = truncated_step
    ge_kwargs['baseline_losses'] = baseline_losses
    ge_kwargs['timer_obj'] = Timing
    ge_kwargs['truncation_schedule'] = inner_length_schedule
    ge_cls = GRADIENT_ESTIMATORS[ge_class_name.lower()]
    return build_from_dict(ge_cls, ge_kwargs)
    # return ge_cls(**ge_kwargs)