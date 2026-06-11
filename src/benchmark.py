import argparse
import os
import pickle
import sys
import time
from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np
import pprint
import wandb
from tqdm import tqdm

import globals
from helpers import (
    Timing,
    set_non_hashable_args,
    delete_old_checkpoints,
)
from optimizers import get_optimizer
from opt.new_optimizers import get_optax_schedule
from config_utils import config_to_dict
from tasks import get_task
from helpers import print_rank_0, safe_block_until_ready

from parameterization import CompletedPParameterization, MuonCompletedPParameterization

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


def rename_batch(batch):
    label_map = {'obs':'image',
                    'target':'label',
                    'image':'image',
                    'attention_mask':'attention_mask',
                    'label':'label'}
    
    return {label_map[k]:v for k,v in batch.items()}

def count_parameters(params):
    return sum(jnp.size(param) for param in jax.tree_util.tree_leaves(params))

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key+'_mean', jnp.mean(v).item()))
            items.append((new_key+'_std', jnp.std(v).item()))
            items.append((new_key+'_max', jnp.max(v).item()))
            items.append((new_key+'_min', jnp.min(v).item()))
            items.append((new_key+'_2norm', jnp.linalg.norm(v,ord=2).item()))

    return dict(items)

def get_params_and_state(needs_state, task, key):
    if needs_state:
        print_rank_0("callling init_with_state in get_params_and_state")
        return task.init_with_state(key)
    else:
        return task.init(key), None


def setup_completed_p_parameterization(args, task, params, state, key):
    """
    Set up the CompletedP parameterization for muP scaling.

    Creates the parameterization object, computes all scaling factors,
    initializes parameters, and sets up multipliers on the model.

    For mup_muon optimizer, uses MuonCompletedPParameterization which computes
    separate Muon and Adam scaling factors (Qiu et al. 2025).

    Args:
        args: Argument namespace containing parameterization_args, optimizer settings, etc.
        task: The task object containing the flax_module and datasets.
        params: Initial model parameters to be re-initialized with proper scaling.
        state: Model state dictionary (will be updated with mup_lrs_to_use).
        key: JAX random key for parameter initialization.

    Returns:
        params: Re-initialized parameters with proper muP scaling.
        state: Updated state with mup_lrs_to_use.
        mup_update_dict: Dictionary of scaling factors for muP optimizer (empty if not using mup_adamw/mup_muon).
        parameterization: The parameterization object (needed for re-init in run loop).
        tensor_types: Tensor type annotations from the model (needed for re-init in run loop).
        key: Updated JAX random key after split.
    """
    # Compute current batch size accounting for gradient accumulation and sequence length
    current_bs = (
        args.gradient_accumulation_steps
        * args.local_batch_size
        * task.datasets.extra_info.get('sequence_length', 1)
    )

    # Select parameterization class based on optimizer OR parameterization_class routing key.
    # The routing key is set by Muon-aware configs (e.g. complete_p_bs100k_steps2000_muon.py)
    # and is required for learned-Muon optimizers like MuCompletedPMuonLOpt that don't have
    # 'mup_muon' in args.optimizer.
    _param_class_key = (args.parameterization_args.get('parameterization_class') or '').lower()
    use_muon_param = (args.optimizer.lower() == 'mup_muon') or (_param_class_key == 'muon_completedp')
    ParamClass = MuonCompletedPParameterization if use_muon_param else CompletedPParameterization

    # ``parameterization_class`` is a routing key consumed by
    # ``mu_task_base._compute_completed_p_scales`` (which is the meta-train path).
    # The benchmark / meta-test path uses ``args.optimizer`` to make the same
    # choice, so we drop the routing key here before forwarding to the
    # parameterization constructor (which doesn't accept it).
    pargs = dict(args.parameterization_args)
    pargs.pop('parameterization_class', None)

    parameterization = ParamClass(
        current_width=task.flax_module.docfg.width,
        current_depth=task.flax_module.docfg.depth,
        current_batch_size=current_bs,
        current_dataset_size=current_bs * args.num_inner_steps,
        **pargs,
    )

    # Get tensor type annotations from the model
    tensor_types = task.flax_module.get_tensor_types(params)
    device = jax.devices()[jax.process_index()]

    # Get scaling factors — shared across all optimizers
    wd_scales = parameterization.get_wd_scales_pytree(tensor_types, device=device)
    one_minus_beta1_scales = parameterization.get_one_minus_beta1_scales_pytree(tensor_types, device=device)
    one_minus_beta2_scales = parameterization.get_one_minus_beta2_scales_pytree(tensor_types, device=device)

    # Re-initialize parameters with proper muP scaling
    key, key1 = jax.random.split(key)
    params = parameterization.init_params_pytree(params, tensor_types, key=key1)

    # Set forward pass multipliers on the model
    task.flax_module.set_multipliers_from_dict(
        parameterization.get_multipliers(device=device),
        device=device
    )

    # Build optimizer update dict based on optimizer type
    if use_muon_param:
        # Muon-specific: separate LR/eps scales for Muon and Adam params
        muon_lr_scales = parameterization.get_muon_lr_scales_pytree(tensor_types, device=device)
        muon_eps_scales = parameterization.get_muon_eps_scales_pytree(tensor_types, device=device)
        adam_lr_scales = parameterization.get_adam_lr_scales_pytree(tensor_types, device=device)
        adam_eps_scales = parameterization.get_adam_eps_scales_pytree(tensor_types, device=device)

        mup_update_dict = {
            'muon_lr_scales': muon_lr_scales,
            'muon_eps_scales': muon_eps_scales,
            'muon_wd_scales': wd_scales,
            'adam_lr_scales': adam_lr_scales,
            'adam_eps_scales': adam_eps_scales,
            'adam_wd_scales': wd_scales,
            'one_minus_beta1_scales': one_minus_beta1_scales,
            'one_minus_beta2_scales': one_minus_beta2_scales,
        }

        # Standard mup_* keys (Adam-side fallback for learned optimizers)
        state['mup_lr_scales'] = adam_lr_scales
        state['mup_eps_scales'] = adam_eps_scales
        state['mup_wd_scales'] = wd_scales
        state['mup_one_minus_beta1_scales'] = one_minus_beta1_scales
        state['mup_one_minus_beta2_scales'] = one_minus_beta2_scales

        # Legacy hand-tuned mup_muon keys (no mup_ prefix) — kept for backward compat
        state['muon_lr_scales'] = muon_lr_scales
        state['muon_eps_scales'] = muon_eps_scales

        # New keys consumed by MuCompletedPMuonLOpt (with mup_ prefix).
        # Mirrors mu_task_base._compute_completed_p_scales (the meta-train path).
        state['mup_muon_lr_scales'] = muon_lr_scales
        state['mup_muon_eps_scales'] = muon_eps_scales

        from opt.mup_muon import _compute_shape_scales
        from opt.new_optimizers import make_muon_weight_dimension_numbers
        from optax.contrib._muon import MuonDimensionNumbers
        from optax.transforms._masking import MaskedNode

        dim_nums = make_muon_weight_dimension_numbers(params)

        def _is_leaf(x):
            return x is None or isinstance(x, MaskedNode) or isinstance(x, MuonDimensionNumbers)

        is_muon_mask = jax.tree_util.tree_map(
            lambda dn: jax.device_put(
                jnp.asarray(
                    not (dn is None or isinstance(dn, MaskedNode)),
                    dtype=jnp.bool_,
                ),
                device,
            ),
            dim_nums,
            is_leaf=_is_leaf,
        )
        muon_shape_scales_raw = _compute_shape_scales(
            params, make_muon_weight_dimension_numbers
        )
        muon_shape_scales = jax.tree_util.tree_map(
            lambda s: jax.device_put(jnp.asarray(s, dtype=jnp.float32), device),
            muon_shape_scales_raw,
        )
        state['mup_muon_shape_scales'] = muon_shape_scales
        state['mup_is_muon_mask'] = is_muon_mask

        # Tensor type indices for per-tensor learned offsets
        from parameterization import TENSOR_TYPE_INDEX
        def _tt_index_scale_fn(module_type, fan_in, fan_out, tensor_type, layer_idx, device):
            return jax.device_put(
                jnp.array(TENSOR_TYPE_INDEX[tensor_type.value], dtype=jnp.int32),
                device
            )
        state['mup_tensor_type_indices'] = parameterization._transform_tensor_types_tree_with_depth(
            tensor_types, _tt_index_scale_fn, device)

    elif args.optimizer.lower() == 'mup_adamw':
        lr_scales = parameterization.get_lr_scales_pytree(tensor_types, device=device)
        eps_scales = parameterization.get_eps_scales_pytree(tensor_types, device=device)

        mup_update_dict = {
            'lr_scales': lr_scales,
            'eps_scales': eps_scales,
            'wd_scales': wd_scales,
            'one_minus_beta1_scales': one_minus_beta1_scales,
            'one_minus_beta2_scales': one_minus_beta2_scales,
        }

        state['mup_lr_scales'] = lr_scales
        state['mup_eps_scales'] = eps_scales
        state['mup_wd_scales'] = wd_scales
        state['mup_one_minus_beta1_scales'] = one_minus_beta1_scales
        state['mup_one_minus_beta2_scales'] = one_minus_beta2_scales
    else:
        # Non-MuP optimizers or learned optimizers that read from state
        lr_scales = parameterization.get_lr_scales_pytree(tensor_types, device=device)
        eps_scales = parameterization.get_eps_scales_pytree(tensor_types, device=device)

        state['mup_lr_scales'] = lr_scales
        state['mup_eps_scales'] = eps_scales
        state['mup_wd_scales'] = wd_scales
        state['mup_one_minus_beta1_scales'] = one_minus_beta1_scales
        state['mup_one_minus_beta2_scales'] = one_minus_beta2_scales

        mup_update_dict = {}

    pprint.pprint(mup_update_dict)

    return params, state, mup_update_dict, parameterization, tensor_types, key


def evaluate_test(args, task, opt, opt_state, params, key):
    """
    Evaluate the model on test data with optional accumulation over multiple batches.
    
    Args:
        args: Argument namespace containing:
            - needs_state: Whether the task needs state
            - world_size: Number of processes for distributed training
            - test_accumulate_steps: Number of test batches to accumulate over
        task: The task object with datasets and loss functions
        opt: The optimizer object
        opt_state: Current optimizer state
        params: Model parameters
        key: JAX random key
    
    Returns:
        test_loss: Averaged test loss (scalar)
        test_acc: Averaged test accuracy (scalar, or 0 if not available)
        test_log: Dictionary with "test loss" and optionally "test accuracy"
    """
    num_accumulate_steps = args.test_accumulate_steps
    
    accumulated_loss = 0.0
    accumulated_acc = 0.0
    has_accuracy = False
    
    for step_idx in range(num_accumulate_steps):
        test_batch = rename_batch(next(task.datasets.test))
        # Squeeze out the gradient_accumulation dimension if present
        if test_batch['image'].ndim > 4 and test_batch['image'].shape[0] == 1:
            test_batch = {k: v[0] for k, v in test_batch.items()}
        key, key1 = jax.random.split(key)
        
        # Check if loss_and_accuracy methods exist
        if args.needs_state and hasattr(task, 'loss_and_accuracy_with_state'):
            state = opt.get_state(opt_state)
            step_loss, step_acc = task.loss_and_accuracy_with_state(params, state, key1, test_batch)
            has_accuracy = True
        elif not args.needs_state and hasattr(task, 'loss_and_accuracy'):
            step_loss, step_acc = task.loss_and_accuracy(params, key1, test_batch)
            has_accuracy = True
        else:
            # Fallback to loss-only methods
            if step_idx == 0:
                Warning("test_task does not have loss_and_accuracy method, defaulting to loss")
            if args.needs_state:
                state = opt.get_state(opt_state)
                step_loss, state = task.loss_with_state(params, state, key1, test_batch)
            else:
                step_loss = task.loss(params, key1, test_batch)
            step_acc = 0.0
            has_accuracy = False
        
        accumulated_loss += step_loss
        accumulated_acc += step_acc
    
    # Average over accumulation steps
    test_loss = accumulated_loss / num_accumulate_steps
    test_acc = accumulated_acc / num_accumulate_steps if has_accuracy else 0.0
    
    # All-reduce mean across all processes
    if args.world_size > 1:
        from jax.experimental import multihost_utils
        gathered = multihost_utils.process_allgather(jnp.array([test_loss]))
        test_loss = jnp.mean(gathered)
        if has_accuracy:
            gathered_acc = multihost_utils.process_allgather(jnp.array([test_acc]))
            test_acc = jnp.mean(gathered_acc)
    
    safe_block_until_ready(test_loss)
    if has_accuracy:
        safe_block_until_ready(test_acc)
    
    # Build test_log
    test_log = {"test loss": test_loss}
    if has_accuracy:
        test_log["test accuracy"] = test_acc
    
    return test_loss, test_acc, test_log, key


# ---------------------------------------------------------------------------
# Benchmark checkpoint helpers
# ---------------------------------------------------------------------------

def _bench_ckpt_base():
    return os.path.join(os.environ.get('SCRATCH', ''), 'checkpoints', 'scaling_l2o')


def _find_bench_resume_dir(name):
    """Return the outer checkpoint dir with the largest saved iteration whose name suffix equals `name`."""
    base = _bench_ckpt_base()
    if not os.path.isdir(base):
        return None
    best_dir = None
    best_step = -1
    for d in os.listdir(base):
        if not (os.path.isdir(os.path.join(base, d)) and len(d) > 9 and d[9:] == name):
            continue
        latest_path = os.path.join(base, d, 'latest')
        if not os.path.isfile(latest_path):
            continue
        step_name = open(latest_path).read().strip()
        if not os.path.isfile(os.path.join(base, d, f'{step_name}.pkl')):
            continue
        try:
            step = int(step_name.replace('global_step', ''))
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_dir = os.path.join(base, d)
    return best_dir



def _peek_bench_resume(name):
    """Read checkpoint location to get (wandb_run_id, start_iteration) cheaply, without loading opt_state."""
    outer_dir = _find_bench_resume_dir(name)
    if outer_dir is None:
        return None, 0
    step_name = open(os.path.join(outer_dir, 'latest')).read().strip()
    if not os.path.isfile(os.path.join(outer_dir, f'{step_name}.pkl')):
        return None, 0
    # outer_dir is named "{wandb_run_id}_{name}"; wandb run ids are 8 chars.
    wandb_run_id = os.path.basename(outer_dir)[:8]
    start_iteration = int(step_name.replace('global_step', ''))
    return wandb_run_id, start_iteration


def _load_bench_checkpoint(args, opt_state):
    """
    Load opt_state and metadata from the latest benchmark checkpoint matching args.name.
    All ranks call this independently (shared filesystem).
    Returns (opt_state, start_iteration, key, wandb_run_id, last_test_log),
    or original values if not found.

    """
    outer_dir = _find_bench_resume_dir(args.name)
    if outer_dir is None:
        print_rank_0("[bench resume] No checkpoint found, starting fresh.")

        return opt_state, 0, None, None, {}


    step_name = open(os.path.join(outer_dir, 'latest')).read().strip()
    ckpt_path = os.path.join(outer_dir, f'{step_name}.pkl')

    if not os.path.isfile(ckpt_path):
        print_rank_0(f"[bench resume] Checkpoint incomplete in {outer_dir}, starting fresh.")
        return opt_state, 0, None, None, {}

    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    device = jax.local_devices()[0]
    opt_state = jax.device_put(ckpt['opt_state'], device)
    key = jax.device_put(ckpt['key'], device)
    start_iteration = ckpt['iteration'] + 1

    print_rank_0(f"[bench resume] Resuming from step {start_iteration} ({ckpt_path})")

    return opt_state, start_iteration, key, ckpt['wandb_run_id'], ckpt['last_test_log']


def _save_bench_checkpoint(args, iteration, opt_state, key, wandb_run_id, last_test_log):
    """
    Save inner-loop checkpoint. Only rank 0 writes to disk.
    Keeps the last `args.checkpoints_to_keep` (default 2) checkpoints.
    """
    if args.rank != 0:
        return
    outer_dir = os.path.join(_bench_ckpt_base(), f"{wandb_run_id}_{args.name}")
    os.makedirs(outer_dir, exist_ok=True)
    step_name = f'global_step{iteration + 1}'
    ckpt_path = os.path.join(outer_dir, f'{step_name}.pkl')

    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'opt_state': jax.device_get(opt_state),
            'iteration': iteration,
            'key': np.array(key),
            'wandb_run_id': wandb_run_id,
            'last_test_log': {k: float(v) for k, v in last_test_log.items()},
        }, f, protocol=4)

    with open(os.path.join(outer_dir, 'latest'), 'w') as f:
        f.write(step_name)

    delete_old_checkpoints(
        save_dir=outer_dir,
        n_to_keep=getattr(args, 'checkpoints_to_keep', 2),
        world_size=args.world_size,
    )
    print_rank_0(f"[bench ckpt] Saved step {iteration + 1} → {ckpt_path}")


# ---------------------------------------------------------------------------

def benchmark(args, sweep=False):
    if sweep:
        if args.rank == 0:
            run = wandb.init(entity="<NEED>", project=args.test_project, group=args.name, config=vars(args))
            run.log_code(".")
            args = argparse.Namespace(**run.config)
            override = [x for x in args.__dict__.keys() if '__' in x]
            override_config = {k: args.__dict__[k] for k in override}
        else:
            print("type(args): ", type(args))
            # args = argparse.Namespace(**args)

        # Broadcast override keys from rank 0 to all ranks
        if args.world_size > 1:
            # Use JAX to broadcast the override config from rank 0 to all ranks
            import jax.experimental.multihost_utils as multihost_utils
            import pickle

            if args.rank == 0:
                # Serialize the config dict to bytes
                config_bytes = pickle.dumps(override_config)
                config_size = len(config_bytes)
            else:
                config_bytes = b''
                config_size = 0

            # First broadcast the size
            size_array = jnp.array([config_size], dtype=jnp.int32)
            gathered_sizes = multihost_utils.process_allgather(size_array, tiled=False)
            config_size = int(gathered_sizes[0].item())


            # Pad bytes to fixed size for all ranks
            if args.rank == 0:
                padded_bytes = config_bytes + b'\x00' * (config_size - len(config_bytes))
            else:
                padded_bytes = b'\x00' * config_size

            # Convert to array and broadcast
            byte_array = jnp.array(list(padded_bytes), dtype=jnp.uint8)
            gathered_bytes = multihost_utils.process_allgather(byte_array, tiled=False)

            # Take rank 0's value and deserialize
            rank0_bytes = bytes(gathered_bytes[0].tolist())
            override_config = pickle.loads(rank0_bytes.rstrip(b'\x00'))

            # Synchronize to ensure all ranks have received the override config
            multihost_utils.sync_global_devices('broadcast_sync')

            
        args.num_runs = 1
        
        # Apply overrides
        print_rank_0("Overriding sweep args:")
        for key, value in override_config.items():
            print_rank_0(f"Setting {key} to {value}")
            parts = key.split('__')
            target = args
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    target[part] = value
                else:
                    parent = target
                    target = target.__dict__.get(part) if i == 0 else target.get(part)
        
        # Update wandb config with the overridden args
        if args.rank == 0:
            run.config.update(vars(args), allow_val_change=True)


    
    args = set_non_hashable_args(args)
    # Set up globals used in truncated step for benchmarking
    globals.needs_state = args.needs_state
    globals.num_grads = args.num_grads
    globals.num_local_steps = args.num_local_steps
    globals.local_batch_size = args.local_batch_size
    globals.use_pmap = args.use_pmap
    globals.num_devices = args.num_devices

    key = jax.random.PRNGKey(args.seed)
    task = get_task(args)[0]
    # test_task = get_task(args, is_test=True)

    key, key1 = jax.random.split(key)
    params, state = get_params_and_state(args.needs_state, task, key1)
    
    # Setup CompletedP parameterization for muP scaling
    completed_p_optimizers = ('mucompletedpadafacmlplopt', 'mucompletedpadamlopt')

    uses_completed_p_lopt = args.optimizer.lower() in completed_p_optimizers
    has_parameterization_args = getattr(args, 'parameterization_args', None) is not None

    if uses_completed_p_lopt and not has_parameterization_args:
        print_rank_0(
            "WARNING: optimizer is a CompletedP learned optimizer but no parameterization config "
            "was provided. The optimizer will receive identity (all-ones) scales, which differs "
            "from training conditions. Add a config/parameterization/*.py to --config to fix this."
        )

    task_supports_completed_p = hasattr(task, 'flax_module')
    if has_parameterization_args and not task_supports_completed_p:
        print_rank_0(
            "WARNING: parameterization_args provided but task has no flax_module "
            "(e.g. haiku-based MLP/image tasks). Skipping CompletedP parameterization."
        )

    if has_parameterization_args and task_supports_completed_p:
        params, state, mup_update_dict, parameterization, tensor_types, key = \
            setup_completed_p_parameterization(args, task, params, state, key)
    else:
        mup_update_dict, parameterization, tensor_types, key = {}, {}, {}, key


    print_rank_0("====================================================================================")
    num_params_m = count_parameters(params)/1e6
    print_rank_0("Model parameters (M): ", num_params_m)
    num_tensors = len(jax.tree_util.tree_leaves(params))
    print_rank_0("Number of tensors: ", num_tensors)
    print_rank_0("====================================================================================")

    args.model_num_params = num_params_m
    args.model_num_tensors = num_tensors

    print_rank_0("params:")
    if args.rank == 0:
        pprint.pprint(jax.tree_util.tree_map(lambda x: x.shape, params))



    
    # if state is not None:
    #     try:
    #         lrs = state['mup_lrs_to_use']
    #         set_diff = set(lrs.keys()) - set(params.keys())

    #         assert len(lrs) == len(params), f"Number of learning rates ({len(lrs)}) should be equal to number of parameters ({len(params)}), but differed by: " + str("; ".join(set_diff))
    #         assert set(lrs.keys()) == set(params.keys()), "Learning rates should have the same keys as parameters"
    #         args.runtime_mup_lrs = lrs

    #         # Create CompletedP parameterization with per-tensor multipliers
    #         # The per_tensor_*_multipliers are stored in the parameterization object
            
    #         # exit(0)
    #         print_rank_0("Set rruntime_mup_lrs")
    #     except KeyError as e:
    #         # print(state['mup_lrs_to_use'])
    #         print_rank_0("No mup_lrs_to_use in state, for task "+args.task[0])
    # else:
    #     print_rank_0("State is None for task "+args.task[0])


    opt, update = get_optimizer(args, mup_update_dict, task=task)

    # Build a schedule callable for LR logging when opt doesn't expose get_current_lr
    # (e.g. learned optimizers where opt = lopt.opt_fn(meta_params)).
    if not hasattr(opt, "get_current_lr"):
        _sched_cfg = getattr(args, "schedule", None)
        if _sched_cfg is not None:
            _sched_cfg = config_to_dict(_sched_cfg)
            _lr_schedule = get_optax_schedule(_sched_cfg["class_"], _sched_cfg["kwargs"])
            opt.get_current_lr = lambda iteration: float(_lr_schedule(iteration))

    if args.use_pmap:
        assert args.num_grads % args.num_devices == 0, "The number of devices for pmap should be a multiple of the number of clients (gradients)"


    # import pdb; pdb.set_trace()
    test_acc=0
    test_loss=0
    print_rank_0('\nstarting loop')
    for run_idx in tqdm(range(args.num_runs), ascii=True, desc="Outer Loop", disable=args.rank != 0):

        # Auto-resume: peek at checkpoint metadata before W&B init so we can pass the right run ID.
        start_iteration = 0
        resume_wandb_run_id = None
        if getattr(args, 'auto_resume', False) and run_idx == 0:
            resume_wandb_run_id, start_iteration = _peek_bench_resume(args.name)
            if start_iteration > 0:
                print_rank_0(f"[bench resume] Will resume from step {start_iteration} into W&B run {resume_wandb_run_id}")

        if not sweep and args.rank == 0:
            if resume_wandb_run_id is not None:
                run = wandb.init(
                    entity="<NEED>",
                    id=resume_wandb_run_id, resume="allow",
                    project=args.test_project, group=args.name, config=vars(args),
                )
            else:
                run = wandb.init(entity="<NEED>", project=args.test_project, group=args.name, config=vars(args))
            run.log_code(".")

            wandb_run_id = run.id
        else:
            # Non-rank-0 processes: use the peeked ID (or None for fresh runs; checkpoint saving is rank-0-only anyway)
            wandb_run_id = resume_wandb_run_id

        if run_idx > 0:
            params, state = get_params_and_state(args.needs_state, task, key1)
            key, key1 = jax.random.split(key)
            params = parameterization.init_params_pytree(params, tensor_types, key=key1)


        opt_state = opt.init(params, model_state=state, num_steps=args.num_inner_steps)

        # Load checkpoint opt_state (replaces the fresh init above)

        resumed_test_log = {}
        if start_iteration > 0:
            opt_state, start_iteration, resume_key, _, resumed_test_log = _load_bench_checkpoint(args, opt_state)

            if resume_key is not None:
                key = resume_key

        if args.use_localsgd_batches:
            try:
                local_opt = opt.get_local_optimizer(task.get_mup_state({})['mup_lrs_to_use'])
            except AttributeError:
                local_opt = opt.get_local_optimizer(None)
            local_inner_opt_state = local_opt.init(params, model_state=state)
            local_inner_opt_state = jax.tree_util.tree_map(
                    lambda x: jnp.stack([x] * args.num_grads),
                    local_inner_opt_state)

        prev_params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)

        save_iter = getattr(args, 'save_iter', 500)
        time_limit_secs = (getattr(args, 'time_limit_hours', None) or 0) * 3600
        job_start_time = time.time()
        # jax.debug.print("local_inner_opt_state>>>>>>>> {}, {}", _, local_inner_opt_state.state is not None)
        pbar = tqdm(
            range(start_iteration, args.num_inner_steps),
            initial=start_iteration,
            total=args.num_inner_steps,
            ascii=True,
            desc="Inner Loop",
            disable=args.rank != 0
        )
        train_load_time, grad_time, stepl, test_time = [],[],[],[]
        test_log = dict(resumed_test_log)
        test_loss = test_log.get('test loss', 0)
        test_acc = test_log.get('test accuracy', 0)
        for iteration in pbar:

            # update
            with Timing('get traing batch', train_load_time):
                batch = rename_batch(next(task.datasets.train))


            key, key1 = jax.random.split(key)



            with Timing('fw bw full', grad_time):
                # opt_state, loss, grad, aux = update(opt_state, key1, batch)
                if args.use_localsgd_batches:
                    opt_state, local_inner_opt_state, loss, grad = update(opt_state, local_inner_opt_state, key1, batch)
                else:
                    # print_rank_0("batch.keys(): ", batch.keys())
                    opt_state, loss, grad = update(opt_state, key1, batch)


                to_log = {
                        "train loss": loss,
                    }

            params = opt.get_params(opt_state)
            state = opt.get_state(opt_state)


            if args.rank == 0:
                # L2 norm of gradients
                grad_leaves = jax.tree_util.tree_leaves(grad)
                l2_grads = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grad_leaves))

                # L2 norm of parameters
                param_leaves = jax.tree_util.tree_leaves(params)
                l2_params = jnp.sqrt(sum(jnp.sum(p ** 2) for p in param_leaves))

                # L2 norm of updates (param delta)
                update_leaves = jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(lambda p, pp: p - pp, params, prev_params)
                )
                l2_updates = jnp.sqrt(sum(jnp.sum(u ** 2) for u in update_leaves))

                # Learning rate: explicit schedule for hand-crafted optimizers,
                # read from opt_state.scheduled_lr for learned optimizers
                if hasattr(opt, 'get_current_lr'):
                    lr_value = float(opt.get_current_lr(iteration))
                elif hasattr(opt_state, 'scheduled_lr'):
                    lr_value = float(opt_state.scheduled_lr)
                else:
                    raise RuntimeError("Cannot determine LR: optimizer has no get_current_lr() and opt_state has no scheduled_lr")

                to_log.update({
                    "l2_grads": float(l2_grads),
                    "l2_params": float(l2_params),
                    "l2_updates": float(l2_updates),
                    "lr": lr_value,
                })

            with Timing('test',test_time):
                #test loss and accuracy if implemented
                if not args.skip_test \
                   and (iteration % args.test_interval == 0 \
                        or iteration == 0 \
                        or iteration == args.num_inner_steps-1):
                    test_loss, test_acc, test_log, key = evaluate_test(
                        args, task, opt, opt_state, params, key
                    )
                    to_log.update(test_log)
                    ran_test = True
                else:
                    ran_test = False



            if args.rank == 0:
                current_lr = float(opt.get_current_lr(iteration)) if hasattr(opt, "get_current_lr") else 0.0
                pbar.set_postfix({
                    "data":round(train_load_time[-1],4),
                    "fwbw":round(Timing.run_times_dict["fw bw"][-1],4),
                    "opt":round(Timing.run_times_dict["optimizer step"][-1],4),
                    "AR":round(Timing.run_times_dict["AR"][-1],4),
                    "test":round(test_time[-1],4),
                    "train loss":round(float(loss),2),
                    "test loss":round(float(test_loss),2) if ran_test else 0,
                    "test acc":round(float(test_acc),2) if ran_test else 0,
                    "LR": current_lr,
                })

                # log
                to_log.update({
                    "train_opt_time": Timing.run_times_dict["optimizer step"][-1],
                    "train_step_time": Timing.run_times_dict["fw bw"][-1],
                    "AR time":Timing.run_times_dict["AR"][-1],
                    "train_fwd_time": round(test_time[-1],4),
                    "learning_rate": current_lr,
                    
                })

            # to_log.update(flatten_dict(grad, parent_key='', sep='_'))
            # to_log.update(flatten_dict(jax.tree_util.tree_map(lambda x,y:x-y,prev_params,params), parent_key='delta', sep='_'))
            

            if args.log_activations and args.rank == 0:


                if iteration == 0:
                    idxkey = 'mlp' if 'mlp' in state else 'mu_mlp'
                    # initial_state = state
                    initial_tensors_only = {k:v for k,v in state[idxkey].items() if ('act' in k or 'logit' in k) and 'l1' not in k}

                idxkey = 'mlp' if 'mlp' in state else 'mu_mlp'
                to_log.update({k:v.item() for k,v in state[idxkey].items() if ('act' in k or 'logit' in k) and 'l1' in k})

                tensors_only = {k:v for k,v in state[idxkey].items() if ('act' in k or 'logit' in k) and 'l1' not in k}
                std_delta = jax.tree_util.tree_map(lambda x,y : jnp.std(x - y), tensors_only, initial_tensors_only)
                to_log.update({k+'_std_delta':v.item() for k,v in std_delta.items()})


            if args.rank == 0:
                run.log(to_log, step=iteration)

            prev_params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)

            # Periodic checkpoint save
            if wandb_run_id is not None and (iteration + 1) % save_iter == 0:
                _save_bench_checkpoint(args, iteration, opt_state, key, wandb_run_id, test_log)

            # Time-limit checkpoint: save and exit cleanly so the next SLURM job can resume
            if time_limit_secs > 0 and (time.time() - job_start_time) >= time_limit_secs:
                print_rank_0(
                    f"[bench] Time limit reached ({getattr(args, 'time_limit_hours', '?')}h) "
                    f"after step {iteration + 1}. Saving checkpoint and exiting for next job."
                )
                if wandb_run_id is not None:
                    _save_bench_checkpoint(args, iteration, opt_state, key, wandb_run_id, test_log)
                if args.rank == 0:
                    run.finish()
                sys.exit(0)

        # Final checkpoint at end of run
        if wandb_run_id is not None:
            _save_bench_checkpoint(args, args.num_inner_steps - 1, opt_state, key, wandb_run_id, test_log)


        if args.rank == 0:
            run.finish()


def sweep(args):
    import os
    os.environ['WANDB_LOG_LEVEL'] = 'debug'


    args.SWEEP_CONTINUE = True
    if args.rank == 0:

        for k,v in args.__dict__.items():
            if type(v) == list:
                print(k,type(v))

        print(args.sweep_config)

        if args.sweep_id is None:
            args.sweep_id = wandb.sweep(
                sweep=args.sweep_config, entity="<NEED>", project=args.test_project
            )

        wandb.agent(args.sweep_id, partial(benchmark, args, True), entity="<NEED>", project=args.test_project)
    else:

        while args.SWEEP_CONTINUE:
            benchmark(args, True)
