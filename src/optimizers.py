# Standard library imports
import functools
import pickle

# Third-party imports
import gin
import jax
import jax.numpy as jnp
import mmengine
import numpy as np
import optax
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from optax_shampoo.distributed_shampoo import (GraftingType, PreconditionerType,
                                              distributed_shampoo)

# Learned optimization imports
from learned_optimization.learned_optimizers.adafac_mlp_lopt import AdafacMLPLOpt
from learned_optimization.learned_optimizers.rnn_mlp_lopt import RNNMLPLOpt
from learned_optimization.optimizers import OptaxOptimizer
from learned_optimization.optimizers import base as opt_base
from learned_optimization.optimizers import optax_opts
from learned_optimization.research.general_lopt import prefab

from config_utils import config_to_dict

# Local imports
from tasks import get_task
from helpers import Timing, cast_to_bf16, print_rank_0, safe_block_until_ready

from learned_optimizers import build_learned_optimizer
from tasks import get_task
from opt import AnyOptimizer

import learned_optimizers

from my_compression import cocktail_compression


# Define function to compute the mean across devices
def reduce_mean_across_devices(x):
    return jax.lax.pmean(x, axis_name='i')


@functools.partial(jax.jit, static_argnums=(0,))
def grad_loss_state_accumulate(fun, params, state, key, batch):
    # Batch is a dict with 'image' and 'label' keys
    # The first dimension of each value is the number of gradient accumulation steps
    # jax.debug.print("params[0]:{}, params[1]:{}", params[0], params[1])
    # print(jax.tree_util.tree_map(lambda x: x.shape, batch))
    
    # Calculate the number of batches for gradient accumulation
    gradient_accumulation_steps = batch['image'].shape[0]
    
    # If gradient_accumulation_steps is 1, we can skip the loop and return directly
    if gradient_accumulation_steps == 1:
        first_batch = {k: v[0] for k, v in batch.items()}
        return jax.value_and_grad(fun, has_aux=True)(params, state, key, first_batch)
    
    # Define the scan function
    def scan_body(carry, batch_idx):
        (_, current_state), accumulated_grad = carry
        # Extract the current batch
        current_batch = {k: v[batch_idx] for k, v in batch.items()}
        
        # Get loss and gradient for this batch
        (batch_loss, new_state), batch_grad = jax.value_and_grad(fun, has_aux=True)(
            params, current_state, key, current_batch
        )
        
        # Accumulate gradients
        accumulated_grad = jax.tree_util.tree_map(
            lambda acc, g: acc + g, 
            accumulated_grad, 
            batch_grad
        )

        return ((batch_loss, new_state), accumulated_grad), batch_loss
    
    # Process the first batch separately to get initial values
    first_batch = {k: v[0] for k, v in batch.items()}
    (initial_loss, initial_state), initial_grad = jax.value_and_grad(fun, has_aux=True)(
        params, state, key, first_batch
    )
    
    # Initial carry for scan
    initial_carry = ((initial_loss, initial_state), initial_grad)
    
    # Use lax.scan for the loop over remaining batches
    (((final_loss, final_state), accumulated_grad), losses) = lax.scan(
        scan_body,
        initial_carry,
        jnp.arange(1, gradient_accumulation_steps)
    )
    
    # Calculate total loss (initial + sum of remaining)
    total_loss = initial_loss + jnp.sum(losses)

    
    # Average the loss and gradients
    avg_loss = total_loss / gradient_accumulation_steps
    avg_grad = jax.tree_util.tree_map(lambda x: x / gradient_accumulation_steps, accumulated_grad)
    
    return (avg_loss, final_state), avg_grad

    
def _fedlagg(args, mup_update_dict, task=None):

    HD_LOCO_OPTS = ['diloco', 'fedavg', 'localsgd', 'slowmo']
    if args.optimizer in HD_LOCO_OPTS:
        print_rank_0('_fedlagg using AnyOptimizer',args.outer_optimizer_args)
        outer_opt = AnyOptimizer(
            optimizer=args.outer_optimizer_args['optimizer_args'],
            schedule=args.outer_optimizer_args['schedule'],
            gradient_transform_before_optim=args.outer_optimizer_args['gradient_transform_before_optim'],
            gradient_transform_after_optim=args.outer_optimizer_args['gradient_transform_after_optim'],
            mup_lrs=args.runtime_mup_lrs if args.outer_optimizer_args['use_mup'] else None,
            local_optimizer_args=args.local_optimizer_args,
        )
    elif args.optimizer.lower() in [x.lower() for x in learned_optimizers.__all__]:
        lopt = build_learned_optimizer(args)
        with open(args.test_checkpoint, "rb") as f:
            meta_params = pickle.load(f)
        total_lopt_params = count_parameters(meta_params)
        print_rank_0(f"Total LOpt params: {total_lopt_params}")
        outer_opt = lopt.opt_fn(meta_params)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not found")
    

    
    try:
        local_opt = outer_opt.get_local_optimizer(task.get_mup_state({})['mup_lrs_to_use'])
    except AttributeError:
        local_opt = outer_opt.get_local_optimizer(None)

    @jax.jit
    def local_step(local_opt_state_and_key, local_batch):
        # local_batch = jax.tree_util.tree_map(lambda x: x[0], local_batch)
        local_opt_state, key = local_opt_state_and_key
        params = local_opt.get_params(local_opt_state)
        key, key1 = jax.random.split(key)
        if args.needs_state:
            state = local_opt.get_state(local_opt_state)
            (l, s), grad = grad_loss_state_accumulate(task.loss_with_state, params, state, key1, local_batch)
            # (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, local_batch)
        else:
            # (l, s), grad = grad_loss_state_accumulate(task.loss_with_state, params, state, key1, local_batch)
            raise ValueError("needs_state is False but task.loss_with_state is used")
            # l, grad = jax.value_and_grad(task.loss, has_aux=True)(params, key1, local_batch)
            # s = None

        return (local_opt.update(local_opt_state, grad, loss=l, model_state=s), key), l


    @functools.partial(jax.vmap, in_axes=(0, 0, 0))
    def vmap_local_updates(init_local_opt_state, key, client_batch):
        # print('init_local_opt_state before',jax.tree_util.tree_map(lambda x: x.dtype, init_local_opt_state))
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)
        # print('final_local_opt_state after',jax.tree_util.tree_map(lambda x: x.dtype, final_local_opt_state))
        return (
            jnp.mean(local_losses),
            jax.tree_util.tree_map(
                lambda new_p, old_p: new_p - old_p,
                local_opt.get_params(final_local_opt_state),
                local_opt.get_params(init_local_opt_state),
            ),
            local_opt.get_state(final_local_opt_state) if args.needs_state else None,
            final_local_opt_state
        )

        
    @jax.jit
    def opt_update(opt_state, deltas, avg_delta, loss, model_state):
        return outer_opt.update(opt_state=opt_state, grads=deltas, grad=avg_delta, loss=loss, model_state=model_state)

    def update(opt_state, local_inner_opt_state, key, batch):
        # This split creates num_grads new keys but doesn't preserve the original key
        # If we needed to preserve the original key, we would use: key, *keys = jax.random.split(key, args.num_grads + 1)
        # First split a subkey from the original key to preserve the original
        key, subkey = jax.random.split(key)
        # Then use the subkey to generate the required number of keys
        keys = jax.random.split(subkey, args.num_grads)
        # keys = jax.random.split(key, args.num_grads)
        # assert local_inner_opt_state is not None, "local_inner_opt_state is None"
        # params = local_opt.get_params(local_inner_opt_state)
        # print("type of params>>>>:", type(params))
        # print("params[0] >>>>:", params[0])
        # print("params[1] >>>>:", params[1])
        # print(jax.tree_util.tree_map(lambda x: x.shape, batch))
        with Timing('fw bw', []):
            losses, deltas, new_state, final_local_opt_state = vmap_local_updates(local_inner_opt_state, keys, batch)
        
        STATE_FLAG = args.needs_state and not new_state in [None, {}]
        # print('STATE_FLAG>>>>>>>>>>>', STATE_FLAG)

        with Timing("delta compression",[]):
            if args.compression_args:
                key, subkey = jax.random.split(key)
                if local_opt.use_error_correction:
                    final_local_opt_state = local_opt.update_mom_delta(final_local_opt_state, deltas)
                    deltas = cocktail_compression(final_local_opt_state.mom_delta, key=subkey, **args.compression_args)
                    final_local_opt_state = local_opt.correct_mom_delta(final_local_opt_state, deltas)
                else:
                    deltas = cocktail_compression(deltas, key=subkey, **args.compression_args)

                if STATE_FLAG:
                    key, subkey = jax.random.split(key)
                    new_state = cocktail_compression(new_state,  key=subkey, **args.compression_args)

        

        # with Timing("Error Correction",[]):
        #     if args.error_correction:
        #         deltas = jax.tree_util.tree_map(lambda x: x * -1, deltas)

        with Timing('AR', []):
            #############################################################################
            # First compute local means across the first axis (keeping dimension intact)
            #############################################################################
            # For losses
            losses = jnp.mean(losses, axis=0, keepdims=True)
            # For deltas
            deltas = jax.tree_util.tree_map(
                lambda x: jnp.mean(x, axis=0, keepdims=True), 
                deltas
            )
            # For state
            if STATE_FLAG:
                new_state = jax.tree_util.tree_map(
                    lambda x: jnp.mean(x, axis=0, keepdims=True),
                    new_state
                )
            #############################################################################
            # All-reduce
            #############################################################################
            losses = jax.pmap(reduce_mean_across_devices, axis_name='i')(losses)
            deltas = jax.tree_util.tree_map(lambda x: jax.pmap(reduce_mean_across_devices, axis_name='i')(x), deltas)
            # Remove extra dimensions after reduction
            loss = jnp.squeeze(losses)
            avg_delta = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), deltas)
            if STATE_FLAG:
                new_state = jax.tree_util.tree_map(lambda x: jax.pmap(reduce_mean_across_devices, axis_name='i')(x), new_state)
                avg_state = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), new_state)
            else:
                avg_state = new_state
                # avg_state = None


        with Timing('optimizer step', []):
            if args.optimizer in HD_LOCO_OPTS:
                avg_delta = jax.tree_util.tree_map(lambda x: x * -1, avg_delta)

            out = opt_update(opt_state=opt_state, deltas=deltas, avg_delta=avg_delta, loss=loss, model_state=avg_state)
            global_s = jax.tree_util.tree_map(lambda x: jnp.stack([x] * args.num_grads), avg_state)
            global_p = jax.tree_util.tree_map(lambda x: jnp.stack([x] * args.num_grads), out.params)
            final_local_opt_state = local_opt.resume_init(opt_state=final_local_opt_state, params=global_p, model_state=global_s)


        return out, final_local_opt_state, loss, avg_delta

    return outer_opt, update



def count_parameters(params):
    return sum(jnp.size(param) for param in jax.tree_util.tree_leaves(params))





#############################################################################
# Communicate-every-step Optimizers
#############################################################################
def load_jax_pickle_with_compat(filepath):
    """Load a JAX pickle file with compatibility handling for version mismatches."""
    import jax._src.core as jax_core
    
    # Store original ShapedArray update method
    original_update = jax_core.ShapedArray.update
    
    # Create a wrapper that filters out 'named_shape' if not supported
    def compatible_update(self, **kwargs):
        # Remove 'named_shape' if it's in kwargs (for compatibility with older pickles)
        kwargs.pop('named_shape', None)
        return original_update(self, **kwargs)
    
    # Temporarily patch the update method
    jax_core.ShapedArray.update = compatible_update
    
    try:
        with open(filepath, "rb") as f:
            result = pickle.load(f)
        return result
    finally:
        # Restore original method
        jax_core.ShapedArray.update = original_update

#############################################################################
def _default_lopt(args, mup_update_dict, task=None):
    if args.optimizer == 'velo':
        opt = prefab.LearnedOptimizer(args.num_inner_steps)
    elif args.optimizer.lower() in [x.lower() for x in learned_optimizers.__all__]:
        lopt = build_learned_optimizer(args)
        if hasattr(lopt, 'load_meta_params'):
            meta_params = lopt.load_meta_params(args.test_checkpoint)
        else:
            meta_params = load_jax_pickle_with_compat(args.test_checkpoint)
        total_lopt_params = count_parameters(meta_params)
        print_rank_0(f"Total LOpt params: {total_lopt_params}")

        # Initialize the learned optimizer with a random key
        # key = jax.random.PRNGKey(args.seed)
        # meta_params = lopt.init(key)
        opt = lopt.opt_fn(meta_params)
    else:

        USE_MUP = args.optimizer.lower().startswith('mu') and args.optimizer.lower() not in ('muon', 'muloco')
        if USE_MUP:
            assert args.task[0].startswith('mu'), "optimizer starts with mu but task does not"

            if args.optimizer.lower() == 'mup_adamw':
                args.optimizer_args['kwargs'].update(mup_update_dict)
                args.runtime_mup_lrs = None
            elif args.optimizer.lower() == 'mup_muon':
                args.optimizer_args['kwargs'].update(mup_update_dict)
                args.runtime_mup_lrs = None
                    
        opt = AnyOptimizer(
            optimizer=config_to_dict(args.optimizer_args),
            schedule=config_to_dict(args.schedule),
            gradient_transform_before_optim=config_to_dict(args.gradient_transform_before_optim),
            gradient_transform_after_optim=config_to_dict(args.gradient_transform_after_optim),
            mup_lrs=args.runtime_mup_lrs if USE_MUP else None,
        )

    # Use the provided task or create a new one (for backward compatibility)
    if task is None:
        task = get_task(args)[0]

    @functools.partial(jax.jit, donate_argnums=(0,))  # donate opt_state buffer
    def opt_update(opt_state, grad, loss, model_state, key):
        return opt.update(opt_state=opt_state, grad=grad, loss=loss, model_state=model_state, key=key)

    # Implicit GSPMD: shard batch across devices, replicate params/state.
    # XLA automatically inserts NCCL allreduce for gradients of replicated
    # params computed from sharded data.  This avoids the XLA AutotunerPass
    # DEVICE_TYPE_INVALID crash that explicit lax.psum inside shard_map triggers.
    if args.world_size > 1:
        _all_devices = jax.devices()
        _mesh = Mesh(np.array(_all_devices), ('dp',))
        _replicated = NamedSharding(_mesh, P())
        # Batch arrays: [grad_accum, local_bs, ...] → global [grad_accum, global_bs, ...]
        _batch_sharding = NamedSharding(_mesh, P(None, 'dp'))
        _n_devices = jax.device_count()

        def _to_global_batch(local_batch):
            """Wrap per-process local batch dict into globally-sharded arrays."""
            def _shard(x):
                local_arr = jax.device_put(x, jax.local_devices()[0])
                global_shape = (x.shape[0], x.shape[1] * _n_devices) + x.shape[2:]
                return jax.make_array_from_single_device_arrays(
                    global_shape, _batch_sharding, [local_arr])
            return jax.tree_util.tree_map(_shard, local_batch)

        def _to_global_replicated(pytree):
            """Wrap local arrays into globally-replicated arrays."""
            def _rep(x):
                local_arr = jax.device_put(x, jax.local_devices()[0])
                return jax.make_array_from_single_device_arrays(
                    x.shape, _replicated, [local_arr])
            return jax.tree_util.tree_map(_rep, pytree)

        def _to_local(pytree):
            """Extract local single-device arrays from global arrays."""
            def _loc(x):
                if isinstance(x, jax.Array) and not x.is_fully_addressable:
                    return x.addressable_data(0)
                if isinstance(x, jax.Array) and hasattr(x, 'addressable_shards'):
                    return x.addressable_data(0)
                return x
            return jax.tree_util.tree_map(_loc, pytree)

        @jax.jit
        def _grad_and_allreduce(params, state, key, batch):
            (loss, model_state), grad = grad_loss_state_accumulate(
                task.loss_with_state, params, state, key, batch
            )
            return loss, model_state, grad

    def update(opt_state, key, batch):
        state = opt.get_state(opt_state)
        params = opt.get_params(opt_state)
        key, key1 = jax.random.split(key)

        with Timing('fw bw', []):
            if args.world_size > 1:
                # Implicit GSPMD: place data on mesh, XLA auto-inserts allreduce
                g_batch = _to_global_batch(batch)
                g_params = _to_global_replicated(params)
                g_state = _to_global_replicated(state)
                g_key = _to_global_replicated(key1)
                g_loss, g_ms, g_grad = _grad_and_allreduce(
                    g_params, g_state, g_key, g_batch)
                loss = _to_local(g_loss)
                model_state = _to_local(g_ms)
                grad = _to_local(g_grad)
            else:
                (loss, model_state), grad = grad_loss_state_accumulate(
                    task.loss_with_state, params=params, state=state, key=key1, batch=batch,
                )

        with Timing('AR', []):
            pass  # allreduce is implicit via GSPMD sharding propagation

        with Timing('optimizer step', []):
            temp = opt_update(
                opt_state=opt_state,
                grad=grad,
                loss=loss,
                model_state=model_state,
                key=key
            )


        # ELO_*LOpt.update with meta_train=True returns (state, cosine_loss, magnitude_loss);
        # benchmark only consumes the state.
        if isinstance(temp, tuple):
            temp = temp[0]

        return temp, loss, grad

    return opt, update



AdamWLinearCosine = None
AdamW = None
def get_optimizer(args, mup_update_dict, task=None):
    optimizers_registry = {
        ############################################################
        # Distributed LoCo Optimizers
        ############################################################
        "fedlopt": _fedlagg,
        "fedlopt-adafac": _fedlagg,
        "fedlagg": _fedlagg,
        "fedlagg-wavg": _fedlagg,
        "fedlagg-adafac": _fedlagg,
        "diloco": _fedlagg,
        "fedavg": _fedlagg,
        "localsgd": _fedlagg,
        "slowmo": _fedlagg,
    }
    

    return optimizers_registry.get(args.optimizer.lower(),_default_lopt)(args, mup_update_dict, task=task)



