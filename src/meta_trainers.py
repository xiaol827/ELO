from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from absl import logging
import functools
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import time
from helpers import Timing
from learned_optimization import profile, summary, tree_utils
from learned_optimization.optimizers import base as opt_base
from meta_trainer_builder import (
  build_gradient_estimator,
  build_truncated_step,
  build_inner_length_schedule,
  build_meta_optimizer
)

from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import es_single

from learned_optimization.outer_trainers.gradient_learner import (
    AggregatedGradient,
    GradientEstimator,
    GradientEstimatorState,
    GradientLearnerState,
    MetaInitializer,
    MetaParams,
    OuterState,
    SingleMachineState,
    ThetaModelState,
    WorkerComputeOut,
    WorkerWeights,
    _get_theta_update_fn,
    _nan_to_num,
    _tree_mean_onp,
    _tree_zeros_on_device,
)

from learned_optimization.tasks import base as tasks_base
from learned_optimization.tasks.task_augmentation import ReparamWeightsFamily, ReparamWeights
from learned_optimization.tasks.parametric.cfgobject import LogFeature as LogFeat
from baseline_trajectories import load_trajectories


class _ReparamWeightsCompat(ReparamWeights):
  """ReparamWeights compatible with tasks that implement loss_with_state but not loss_with_state_and_aux."""
  def loss_with_state_and_aux(self, params, state, key, data):
    scales = self._match_param_scale_to_pytree(params)
    scaled_params = jax.tree_util.tree_map(lambda x, scale: x * scale, params, scales)
    loss, new_state = self.task.loss_with_state(scaled_params, state, key, data)
    return loss, new_state, {}


class _ReparamWeightsFamilyCompat(ReparamWeightsFamily):
  """ReparamWeightsFamily that uses _ReparamWeightsCompat to support custom tasks."""
  def task_fn(self, cfg):
    reparam = super().task_fn(cfg)
    return _ReparamWeightsCompat(reparam.task, reparam._param_scale)
from learned_optimizers import build_learned_optimizer

from tasks import get_task


PRNGKey = jnp.ndarray

@gin.configurable
class CustomGradientLearner:
  """Learner is responsible for training the weights of the learned opt."""

  def __init__(
      self,
      meta_init: MetaInitializer,
      theta_opt: opt_base.Optimizer,
      init_theta_from_path: Optional[str] = None,
      init_outer_state_from_path: Optional[str] = None,
      reset_outer_iteration: bool = False,
      num_steps: Optional[int] = None,
      init_seed: Optional[int] = None,
      bc_grad_weight: Optional[float] = None,
  ):
    self._theta_opt = theta_opt
    self._meta_init = meta_init
    self._init_theta_from_path = init_theta_from_path
    self._init_outer_state_from_path = init_outer_state_from_path
    self._reset_outer_iteration = reset_outer_iteration
    self._num_steps = num_steps
    self._init_seed = init_seed
    self._bc_grad_weight = bc_grad_weight

  def get_meta_params(self, state: GradientLearnerState) -> MetaParams:
    return self._theta_opt.get_params(state.theta_opt_state)

  def get_meta_model_state(self,
                           state: GradientLearnerState) -> ThetaModelState:
    return self._theta_opt.get_state(state.theta_opt_state)

  def get_state_for_worker(self, state: GradientLearnerState) -> WorkerWeights:
    return WorkerWeights(
        theta=self.get_meta_params(state),
        theta_model_state=self.get_meta_model_state(state),
        outer_state=OuterState(state.theta_opt_state.iteration))

  def init(self, key: PRNGKey) -> GradientLearnerState:
    """Initial state of the GradientLearner.

    This can be constructed from a random distribution, or loaded from a path.

    Args:
      key: jax rng key

    Returns:
      gradient_learner_state: A new initial state of the gradient learner.
    """
    if self._init_seed is not None:
      key = jax.random.PRNGKey(self._init_seed)

    theta_init = self._meta_init.init(key)
    # TODO(lmetz) hook up model state for learned optimizers
    model_state = None

    if self._init_theta_from_path:
      logging.info(  # pylint: disable=logging-fstring-interpolation
          f"Got a init from params path {self._init_theta_from_path}."
          " Using this instead of random initialization.")

      import pickle
      with open(self._init_theta_from_path, "rb") as f:
        theta_init = pickle.load(f)

    theta_opt_state = self._theta_opt.init(
        theta_init, model_state, num_steps=self._num_steps)

    if self._init_outer_state_from_path:
      logging.info(  # pylint: disable=logging-fstring-interpolation
          f"Got a init from outer state path {self._init_outer_state_from_path}."
          " Using this instead of randomly initializing.")
      fake_checkpoint = OptCheckpoint(
          gradient_learner_state=GradientLearnerState(theta_opt_state),
          elapsed_time=0.0,
          total_inner_steps=1)
      real_checkpoint = checkpoints.load_state(self._init_outer_state_from_path,
                                               fake_checkpoint)
      theta_opt_state = real_checkpoint.gradient_learner_state.theta_opt_state
      if self._reset_outer_iteration:
        theta_opt_state = theta_opt_state.replace(iteration=0)

    return GradientLearnerState(theta_opt_state)

  def update(
      self,
      state: GradientLearnerState,
      grads_list: Sequence[AggregatedGradient],
      with_metrics: bool = False,
      key: Optional[PRNGKey] = None
  ) -> Tuple[GradientLearnerState, Mapping[str, float]]:
    """Update the state of the outer-trainer using grads_list.

    This performs one outer weight update by aggregating the gradients in
    `grads_list`.

    Args:
      state: The state of the outer-trainer.
      grads_list: A list of gradients to be aggregated and applied.
      with_metrics: To compute metrics, or not.
      key: Jax PRNGKey.

    Returns:
      next_state: The next outer-training state.
      metrics: The computed metrics from this update.
    """

    metrics = {}
    theta_opt_state = state.theta_opt_state

    with profile.Profile("stack_grad"):
      grads_stack = tree_utils.tree_zip_onp([t.theta_grads for t in grads_list])
      
    with profile.Profile("mean_grad"):
      grads = _tree_mean_onp(grads_stack)

    with profile.Profile("stack_state"):
      model_state_stack = tree_utils.tree_zip_onp(
          [t.theta_model_state for t in grads_list])
      next_model_state = _tree_mean_onp(model_state_stack)

    with profile.Profile("stack_loss"):
      losses = jnp.asarray([t.mean_loss for t in grads_list])
      mean_loss = jnp.mean(losses)
      min_loss = jnp.min(losses)

    fn = _get_theta_update_fn(self._theta_opt)
    key1, key2 = jax.random.split(key)
    # print("theta_opt_state", theta_opt_state)
    theta_opt_state, theta_update_metrics = fn(
        theta_opt_state,
        grads,
        mean_loss,
        key1,
        next_model_state,
        sample_rng_key=key2,
        with_summary=with_metrics)
    metrics = summary.aggregate_metric_list([metrics, theta_update_metrics])

    # Create fast summaries for all steps, and slower summaries occasionally
    # metrics["none||mean_loss"] = mean_loss
    # metrics["none||best_of_mean_loss"] = min_loss

    if with_metrics:
      # metrics["none||theta_grad_norm"] = tree_utils.tree_norm(grads)
      metrics["none||theta_grad_abs_mean"] = tree_utils.tree_mean_abs(grads)

    return GradientLearnerState(theta_opt_state), metrics  # pytype: disable=bad-return-type  # jax-ndarray


@gin.configurable
class CustomGradientLearnerBC(CustomGradientLearner):
  """Learner that supports behavioral cloning during training."""
  
  def update(
      self,
      state: GradientLearnerState,
      grads_list: Sequence[AggregatedGradient],
      with_metrics: bool = False,
      key: Optional[PRNGKey] = None
  ) -> Tuple[GradientLearnerState, Mapping[str, float]]:
    """Update the state of the outer-trainer using grads_list with BC support.

    This performs one outer weight update by aggregating the gradients in
    `grads_list` and applying behavioral cloning if configured.

    Args:
      state: The state of the outer-trainer.
      grads_list: A list of gradients to be aggregated and applied.
      with_metrics: To compute metrics, or not.
      key: Jax PRNGKey.

    Returns:
      next_state: The next outer-training state.
      metrics: The computed metrics from this update.
    """

    with profile.Profile("stack_grad"):
      grads_stack_es = tree_utils.tree_zip_onp([t.theta_grads[0] for t in grads_list])
      grads_stack_bc = tree_utils.tree_zip_onp([t.theta_grads[1] for t in grads_list])

    metrics = {}
    theta_opt_state = state.theta_opt_state
      
    with profile.Profile("mean_grad"):
      grads_es = _tree_mean_onp(grads_stack_es)
      grads_bc = _tree_mean_onp(grads_stack_bc)

    if with_metrics:
      metrics["none||bc_grad_norm"] = tree_utils.tree_norm(grads_bc)
      metrics["none||bc_grad_abs_mean"] = tree_utils.tree_mean_abs(grads_bc)
      metrics["none||theta_grad_norm"] = tree_utils.tree_norm(grads_es)
      metrics["none||theta_grad_abs_mean"] = tree_utils.tree_mean_abs(grads_es)

    # print("grads_bc norm", tree_utils.tree_norm(grads_bc))

    with profile.Profile("stack_state"):
      model_state_stack = tree_utils.tree_zip_onp(
          [t.theta_model_state for t in grads_list])
      next_model_state = _tree_mean_onp(model_state_stack)

    with profile.Profile("stack_loss"):
      losses = jnp.asarray([t.mean_loss for t in grads_list])
      mean_loss = jnp.mean(losses)
      min_loss = jnp.min(losses)


    key1, key2 = jax.random.split(key)
    theta_opt_state = self._theta_opt.update(
        opt_state=theta_opt_state, 
        grad=None,
        grad_bc=grads_bc, 
        grad_es=grads_es, 
        loss=mean_loss, 
        key=key1, 
        model_state=next_model_state)

    # Create fast summaries for all steps, and slower summaries occasionally
    metrics["none||mean_loss"] = mean_loss
    metrics["none||best_of_mean_loss"] = min_loss

    return GradientLearnerState(theta_opt_state), metrics  # pytype: disable=bad-return-type  # jax-ndarray



@gin.configurable
@profile.wrap()
def gradient_worker_compute_distributed(
    worker_weights: WorkerWeights,
    gradient_estimators: Sequence[GradientEstimator],
    unroll_states: Sequence[GradientEstimatorState],
    key: PRNGKey,
    with_metrics: bool,
    clip_nan_loss_to_value: Optional[float] = 20.0,
    extra_metrics: bool = True,
    device: Optional[jax.Device] = None,
    global_task_size: float = 1.0,
    pmap_across_devices: bool = False,
    bc_grad_weight: Optional[float] = None) -> WorkerComputeOut:
  """Compute a gradient signal to meta-train with.

  This function performs unrolls for each of the unroll_states with the
  corresponding gradient_estimator. The results from each of the gradient
  estimators get's merged into a single gradient. This aggregation is done
  to save bandwidth when collecting gradients from workers.

  Args:
    worker_weights: Weights created by the GradientLearner and represent the
      current parameters and model state of the learned optimizer.
    gradient_estimators: The gradient estimators used to update the unroll state
    unroll_states: state of the gradient estimator (e.g. inner problem weights)
    key: jax rng
    with_metrics: compute with summary metrics or not
    clip_nan_loss_to_value: float, value to set nan losses to
    extra_metrics: log out additional metrics.
    device: The jax device to run the computation on

  Returns:
    worker_compute_out: The results of the computation.
      This contains a gradient estimate, the next unroll states, metrics.
      A subset of which get passed to the GradientLearner.
  """
  #   print("in distributed gradient worker compute")
  if device is None:
    # device = jax.local_devices(0)[0]
    device = jax.devices(device)[jax.process_index()]

  theta = worker_weights.theta
  theta_model_state = worker_weights.theta_model_state

  theta_shape = jax.tree_util.tree_map(
      lambda x: jax.core.ShapedArray(x.shape, x.dtype), theta)
  grads_accum = _tree_zeros_on_device(theta_shape, device)
  bc_grads_accum = _tree_zeros_on_device(theta_shape, device)
  bc_grads_accum['bc_loss'] = jnp.array(0.0,device=device)

  metrics_list = []
  unroll_states_out = []
  losses = []
  event_info = []
  
  assert len(gradient_estimators) == len(unroll_states)


  with Timing('meta train unroll', []):
    for si, (estimator,
            unroll_state) in enumerate(zip(gradient_estimators, unroll_states)):

            
      with profile.Profile(f"estimator{si}"):
        stime = time.time()
        key, rng = jax.random.split(key)

        cfg_name = estimator.cfg_name()

        logging.info(
            "compute_gradient_estimate for estimator name %s and cfg name %s",
            estimator.task_name(), estimator.cfg_name())
        with profile.Profile(f"unroll__metrics{with_metrics}"):
          # print("\n\n before estimator.compute_gradient_estimate()\n\n")
          estimator_out, metrics = estimator.compute_gradient_estimate(
              worker_weights, rng, unroll_state, with_summary=with_metrics)
          # print("\n\n after estimator.compute_gradient_estimate()\n\n")

        unroll_states_out.append(estimator_out.unroll_state)
        losses.append(estimator_out.mean_loss)
        with profile.Profile("tree_add"):
          grads_accum = tree_utils.tree_add(grads_accum, estimator_out.grad)
          if bc_grad_weight is not None:
            bc_grads_accum = tree_utils.tree_add(bc_grads_accum, estimator_out.bc_grad)

        # grab a random iteration from the trajectory
        if estimator_out.unroll_info:
          idx = onp.random.randint(0, len(estimator_out.unroll_info.loss))

          def extract_one(idx, x):
            return x[idx] if x is not None else None

          fn = functools.partial(extract_one, idx)
          onp_task_params = jax.tree_util.tree_map(
              onp.asarray, estimator_out.unroll_info.task_param)
          iteration = estimator_out.unroll_info.iteration[
              idx] if estimator_out.unroll_info.iteration is not None else None
          event_info.append({
              "loss": estimator_out.unroll_info.loss[idx, :],
              "task_param": jax.tree_util.tree_map(fn, onp_task_params),
              "iteration": iteration,
              "outer_iteration": worker_weights.outer_state.outer_iteration,
          })
        else:
          logging.warn("No out specified by learner. "
                      "Not logging any events data.")

        metrics = {k: v for k, v in metrics.items()}
        if extra_metrics:
          family_name = estimator.task_name()
          cfg_name = estimator.cfg_name()
          if with_metrics:
            # Metrics don't take into account which task they are comming from.
            # Let's add additional metrics with the task name pulled out.
            with profile.Profile("metric_computation"):
              keys = list(metrics.keys())
              for k in keys:
                v = metrics[k]
                assert "||" in k, f"bad metric format? Got: {k}"
                agg, name = k.split("||")
                metrics[f"{agg}||{family_name}/{name}"] = v
                metrics[f"{agg}||{cfg_name}/{name}"] = v

              mean_abs = tree_utils.tree_mean_abs(estimator_out.grad)
              metrics[f"mean||{family_name}/grad_mean_abs"] = mean_abs
              metrics[f"mean||{cfg_name}/grad_mean_abs"] = mean_abs

              norm = tree_utils.tree_norm(estimator_out.grad)
              metrics[f"mean||{family_name}/grad_norm"] = norm
              metrics[f"mean||{cfg_name}/grad_norm"] = norm
          metrics[f"mean||{family_name}/mean_loss"] = estimator_out.mean_loss
          metrics[f"mean||{cfg_name}/mean_loss"] = estimator_out.mean_loss
          metrics[f"sample||{family_name}/time"] = time.time() - stime
          metrics[f"sample||{cfg_name}/time"] = time.time() - stime

        metrics_list.append(metrics)


  # Function to compute the sum across devices
  def reduce_across_devices(x):
    return jax.lax.psum(x, axis_name='i')

  # jax.experimental.multihost_utils.sync_global_devices('sync')
  with Timing('meta train all reduce', []):
    with profile.Profile("mean_grads"):

      #assumes that there will only be one task per process when using distributed

      if jax.process_count() == 1 or pmap_across_devices:
        print(
            f"[PARALLEL|RANK {jax.process_index()}] gradient_worker_compute_distributed: "
            f"{'pmap_across_devices=True — gradient already globally consistent from PES all-gather; skipping cross-process grad all-reduce.' if pmap_across_devices else 'Single process — no cross-process communication needed.'} "
            f"local loss={[float(l) for l in losses]}"
        )
        grads_accum = tree_utils.tree_div(grads_accum, len(gradient_estimators))
        mean_loss = jnp.mean(jnp.asarray(losses))
      else:
        print(
            f"[PARALLEL|RANK {jax.process_index()}] gradient_worker_compute_distributed: "
            f"pmap_across_devices=False — performing pmap all-reduce of gradients "
            f"across {jax.process_count()} processes (global_task_size={global_task_size})."
        )
        grads_accum = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x,axis=0),grads_accum)
        # print("grads_accum shape",grads_accum.shape)
        grads_accum = jax.tree_util.tree_map(lambda x: jax.pmap(reduce_across_devices, axis_name='i')(x), grads_accum)
        grads_accum = jax.tree_util.tree_map(lambda x: jnp.squeeze(x,axis=0),grads_accum)
        grads_accum = tree_utils.tree_div(grads_accum, global_task_size)
        # print("losses shape",jnp.asarray(losses).sum().shape)
        losses = jnp.sum(jnp.asarray(losses))
        losses = jnp.expand_dims(losses, axis=0)  # Add batch dimension for pmap
        # losses = jnp.sum(losses, axis=-1, keepdims=True)  # Sum while keeping batch dim
        # print("losses shape", losses.shape)
        mean_loss = jax.pmap(reduce_across_devices, axis_name='i')(losses) / global_task_size
        mean_loss = jnp.squeeze(mean_loss)  # Remove extra dimensions after reduction

      if bc_grad_weight is not None:
        if jax.process_count() == 1:
          #  or pmap_across_devices:   
          bc_grads_accum = tree_utils.tree_div(bc_grads_accum, len(gradient_estimators))
          bc_losses = bc_grads_accum['bc_loss']
          del bc_grads_accum['bc_loss']
        else:
          # still need all reduce exept 
          bc_grads_accum = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x,axis=0),bc_grads_accum)
          bc_grads_accum = jax.tree_util.tree_map(lambda x: jax.pmap(reduce_across_devices, axis_name='i')(x), bc_grads_accum)
          bc_grads_accum = jax.tree_util.tree_map(lambda x: jnp.squeeze(x,axis=0),bc_grads_accum)
          bc_grads_accum = tree_utils.tree_div(bc_grads_accum, global_task_size)
          bc_losses = bc_grads_accum['bc_loss']
          # print(bc_losses.shape)
          del bc_grads_accum['bc_loss']
        # print("bc_grads_accum",jax.tree_util.tree_map(lambda x: x.shape, bc_grads_accum))
        # print("grads_accum",jax.tree_util.tree_map(lambda x: x.shape, grads_accum))


        # grads_accum = jax.tree_util.tree_map(lambda x, y: x * bc_grad_weight + y * (1 - bc_grad_weight), bc_grads_accum, grads_accum)
      else:
        bc_losses = jnp.array(0.0)

      

      # metrics["mean||bc_loss"] = bc_losses
    
    # block here to better acco unt for costs with profile profiling.
    with profile.Profile("blocking"):
      stime = time.time()
      mean_loss.block_until_ready()
      block_time = time.time() - stime

  with profile.Profile("summary_aggregation"):
    metrics = summary.aggregate_metric_list(metrics_list)
  # metrics["mean||block_time"] = block_time

  with profile.Profile("strip_nan"):
    # this should ideally never be NAN
    # TODO(lmetz) check if we need these checks.
    grads_accum = _nan_to_num(grads_accum, 0.0, use_jnp=True)
    if clip_nan_loss_to_value:
      mean_loss = _nan_to_num(mean_loss, clip_nan_loss_to_value, use_jnp=True)


  with profile.Profile("grads_to_onp"):
    if bc_grad_weight is not None:
      to_put = AggregatedGradient(
          theta_grads=(grads_accum, bc_grads_accum),
          theta_model_state=theta_model_state,
          mean_loss=mean_loss)
    else:
      to_put = AggregatedGradient(
          theta_grads=grads_accum,
          theta_model_state=theta_model_state,
          mean_loss=mean_loss)

    return WorkerComputeOut(
        to_put=jax.tree_util.tree_map(onp.asarray, to_put),
        unroll_states=unroll_states_out,
        metrics=metrics,
        event_info=event_info)

class EnhancedSingleMachineGradientLearner(gradient_learner.SingleMachineGradientLearner):
  """Train with gradient estimators on a single machine.

  This is a convience wrapper calling the multi-worker interface -- namley
  both `GradientLearner` and `gradient_worker_compute`.
  """

  def __init__(self,
               meta_init: gradient_learner.MetaInitializer,
               gradient_estimators: Sequence[gradient_learner.GradientEstimator],
               theta_opt: opt_base.Optimizer,
               init_theta_from_path: Optional[str] = None,
               num_steps: Optional[int] = None,
               device: Optional[jax.Device] = None,
               global_task_size = 1.0,
               pmap_across_devices: bool = False,
               bc_grad_weight: Optional[float] = None):
    """Initializer.

    Args:
      meta_init: Class containing an init function to construct outer params.
      gradient_estimators: Sequence of gradient estimators used to calculate
        gradients.
      theta_opt: The optimizer used to train the weights of the learned opt.
      num_steps: Number of meta-training steps used by optimizer for schedules.
    """
    if bc_grad_weight is not None:
      grad_learner = CustomGradientLearnerBC
    else:
      grad_learner = CustomGradientLearner
    self.gradient_learner = grad_learner(
        meta_init, theta_opt, num_steps=num_steps, init_theta_from_path=init_theta_from_path)

    self.gradient_estimators = gradient_estimators
    self.global_task_size = global_task_size
    self.pmap_across_devices = pmap_across_devices
    self.bc_grad_weight = bc_grad_weight
    

  def update(
      self,
      state,
      key: PRNGKey,
      with_metrics: Optional[bool] = False
  ) -> Tuple[SingleMachineState, jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Perform one outer-update to train the learned optimizer.

    Args:
      state: State of this class
      key: jax rng
      with_metrics: To compute metrics or not

    Returns:
      state: The next state from this class
      loss: loss from the current iteration
      metrics: dictionary of metrics computed
    """
    key1, key2 = jax.random.split(key)
    worker_weights = self.gradient_learner.get_state_for_worker(
        state.gradient_learner_state)

    worker_compute_out = gradient_worker_compute_distributed(
        worker_weights,
        self.gradient_estimators,
        list(state.gradient_estimator_states),
        key=key1,
        with_metrics=with_metrics,
        global_task_size=self.global_task_size,
        pmap_across_devices=self.pmap_across_devices,
        bc_grad_weight=self.bc_grad_weight)

    next_gradient_estimator_states = worker_compute_out.unroll_states

    next_theta_state, metrics = self.gradient_learner.update(
        state.gradient_learner_state, [worker_compute_out.to_put],
        key=key2,
        with_metrics=with_metrics)

    metrics = summary.aggregate_metric_list(
        [worker_compute_out.metrics, metrics])

    return (SingleMachineState(
        gradient_learner_state=next_theta_state,
        gradient_estimator_states=next_gradient_estimator_states),
            worker_compute_out.to_put.mean_loss, metrics)



def default_meta_trainer(args):

    tasks = get_task(args)
    lopt = build_learned_optimizer(args)
    meta_opt = build_meta_optimizer(args)


    baseline_losses_dict = load_trajectories()
    print("args.inner_problem_length_schedule", args.inner_problem_length_schedule)

    # build per-task gradient estimators
    gradient_estimators = []
    for task, task_name in zip(tasks, args.task):
        task_family = tasks_base.single_task_to_family(task)
        if getattr(args, 'use_task_augmentation', False):
            task_family = _ReparamWeightsFamilyCompat(
                task_family,
                level=getattr(args, 'task_aug_level', 'global'),
                param_scale_range=tuple(getattr(args, 'task_aug_scale_range', (0.001, 1000.))),
            )

        baseline_losses = baseline_losses_dict.get(task_name, None)
        if args.gradient_estimator_args['kwargs']['use_baseline_losses']:
            assert baseline_losses is not None, f"Baseline trajectory for task {task_name} not found"

        inner_length_schedule = build_inner_length_schedule(args)
        truncated_step = build_truncated_step(args, task_family, lopt, inner_length_schedule)
        grad_est = build_gradient_estimator(args, truncated_step, inner_length_schedule, baseline_losses=baseline_losses)

        gradient_estimators.append(grad_est)

    meta_trainer = EnhancedSingleMachineGradientLearner(
        meta_init=lopt,
        gradient_estimators=gradient_estimators,
        init_theta_from_path=args.test_checkpoint,
        theta_opt=meta_opt,
        device=jax.devices()[jax.process_index()],
        global_task_size=args.global_task_size,
        pmap_across_devices=args.gradient_estimator_args['kwargs']['pmap_across_devices'],
        bc_grad_weight=args.gradient_estimator_args['kwargs']['bc_grad_weight'],
    )


    return meta_trainer, meta_opt



def get_meta_trainer(args):
  return default_meta_trainer(args)

