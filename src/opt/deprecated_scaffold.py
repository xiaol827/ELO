import copy
import pdb
import functools
from typing import Any, Callable, Optional, Sequence, Union
import jax
import jax.numpy as jnp
import optax
import chex
import gin


from learned_optimization.optimizers.optax_opts import OptaxOptimizer, OptaxState


ModelState = Any
Params = Any
Gradient = Params
OptState = Any



@gin.configurable
class LocalScaffold(OptaxOptimizer):
    """Stochastic gradient descent with momentum."""

    def __init__(self, 
                 learning_rate, 
                 num_clients,
                 num_local_steps,
                 num_participants):
        self.learning_rate = learning_rate
        self.num_clients = num_clients
        self.num_local_steps = num_local_steps
        self.num_participants = num_participants
        opt = optax.sgd(learning_rate)
        super().__init__(opt)

    @property
    def name(self):
        return f"Scaffold_lr{self.learning_rate}"

    def init(
        self,
        params: Params,
        control_variate: Optional[Params] = None,
        model_state: Optional[ModelState] = None,
        num_steps: Optional[int] = None,
        key: Optional[chex.PRNGKey] = None,
    ):


        return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            params=params,
            optax_opt_state=[
                self.opt.init(params)
            ],
            state=model_state,
            iteration=0,
        )

    def locally_update_control_variate(self,
                                  global_control_variate,
                                  local_control_variate,
                                  global_params,
                                  opt_state,
                                  ):

        def update_cv(cvl, cvg, lr, pg, pl ):
            return cvl - ( cvg + lr * (pg - pl) )

        lr_like_params = jax.tree_util.tree_map(lambda x: 1 / ( self.num_local_steps * self.learning_rate), global_params)

        return jax.tree_util.tree_map(
            update_cv,
            global_control_variate,
            local_control_variate,
            lr_like_params,
            global_params,
            self.get_params(opt_state),
        )

    # @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        opt_state: OptaxState,
        grad: Gradient,
        global_control_variate,
        local_control_variate,
        # control_variate_local: Optional[jnp.ndarray] = None,
        loss: Optional[jnp.ndarray] = None,
        model_state: Optional[ModelState] = None,
        key: Optional[chex.PRNGKey] = None,
        **kwargs,
    ):
        
        def get_updated_params(p, cvl, cvg, g, lr):
            return p - ( lr * (g - cvl + cvg) )


        # print("local opt LR", jax.tree_util.tree_map(lambda x: self.learning_rate, grad))

        # ALGO1 LINE 10: local update
        updated_params = jax.tree_util.tree_map(get_updated_params,
                                    self.get_params(opt_state),
                                    local_control_variate,
                                    global_control_variate,
                                    grad,
                                    jax.tree_util.tree_map(lambda x: self.learning_rate, grad))

        return OptaxState(
            state=model_state,
            params=updated_params,
            optax_opt_state=[
                opt_state,
            ],
            iteration=opt_state.iteration + 1,
        )






import numpy as np

@gin.configurable
class Scaffold(OptaxOptimizer):
    """Stochastic gradient descent with momentum."""

    def __init__(self,
                 global_learning_rate, 
                 local_learning_rate,
                 num_local_steps,
                 num_clients, 
                 num_participants):
        self.global_learning_rate = global_learning_rate
        self.local_learning_rate = local_learning_rate
        self.num_local_steps = num_local_steps
        self.num_participants = num_participants
        self.num_clients = num_clients
        self.local_optimizer =  LocalScaffold(
                 learning_rate=local_learning_rate, 
                 num_clients=num_clients, 
                 num_local_steps=num_local_steps,
                 num_participants=num_participants)
        opt = optax.sgd(global_learning_rate)
        super().__init__(opt)

    @property
    def name(self):
        return f"Scaffold_lr{self.learning_rate}"

    def __getitem__(self, idx):
        # Handle dynamic indexing
        if isinstance(idx, jax.core.Tracer):
            return jax.lax.dynamic_index_in_dim(self.local_optimizers, idx, keepdims=False)
        return self.local_optimizers[idx]
        # return self.local_optimizers[idx]

    def init(
        self,
        params: Params,
        local_control_variates: Optional[Params] = None,
        global_control_variate: Optional[Params] = None,
        model_state: Optional[ModelState] = None,
        num_steps: Optional[int] = None,
        key: Optional[chex.PRNGKey] = None,
    ):

        if local_control_variates is None:
            local_control_variates = [copy.deepcopy(params) for x in range(self.num_clients)]
            local_control_variates = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *local_control_variates)
            # print(jax.tree_util.tree_map(lambda x: x.shape, local_control_variates))

        if global_control_variate is None:
            global_control_variate = copy.deepcopy(params)

        return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            params=params,
            optax_opt_state=[
                self.opt.init(params),
                {"local_control_variates": local_control_variates,
                 "global_control_variate": global_control_variate}
            ],
            state=model_state,
            iteration=0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        opt_state: OptaxState,
        grad: Gradient,
        loss: Optional[jnp.ndarray] = None,
        model_state: Optional[ModelState] = None,
        key: Optional[chex.PRNGKey] = None,
        **kwargs,
    ):
        del loss
        update, new_opt_state = self.opt.update(
            grad, opt_state.optax_opt_state[0], opt_state.params
        )
        return OptaxState(
            state=model_state,
            params=optax.apply_updates(opt_state.params, update),
            optax_opt_state=[
                new_opt_state,
                opt_state.optax_opt_state[1],
            ],
            iteration=opt_state.iteration + 1,
        )







def _scaffold(args):


    def pytree_mean(pytree):
        # Flatten the pytree into a list of leaves
        leaves, _ = jax.tree.flatten(pytree)
        # Convert the leaves into a single array
        values = jnp.array(leaves)
        # Compute the mean
        return jnp.mean(values)

    def update_stacked_params(stacked_params, indices, updated_params):
        def update_layer(layer, indices, updated_layer):
            # Create a mask to identify the indices to replace
            full_layer = layer.at[indices].set(updated_layer)
            return full_layer
    
        # Apply the update to each layer in the pytree
        return jax.tree_util.tree_map(update_layer, stacked_params, indices, updated_params)

    def split(arr, split_factor):
        """Splits the first axis of `arr` evenly across the number of devices."""
        return arr.reshape(
            split_factor, arr.shape[0] // split_factor, *arr.shape[1:]
        )


    # print("local batch size", args.local_batch_size)

    opt = Scaffold(global_learning_rate=args.learning_rate, 
                   local_learning_rate=args.gradient_estimator_args['kwargs']['local_learning_rate'], 
                   num_local_steps=args.num_local_steps,
                   num_clients=args.number_clients, 
                   num_participants=int(args.number_clients * args.participation_rate),)

    VERBOSE = False

    task = get_task(args)
    @jax.jit
    def update(opt_state, key, batch):
        images = jnp.array(batch["image"])
        labels = jnp.array(batch["label"])
        opt_idx = jnp.array(batch["client_idx"])
        
        @jax.jit
        def local_updates(local_control_variate, im, lab, key, global_control_variate):
            # local_opt = opt_idx
            local_opt = opt.local_optimizer
            local_opt_state = copy.deepcopy(opt_state)
            global_params = local_opt.get_params(opt_state)

            if VERBOSE:
                print("global_params in local_updated BEFORE",jax.tree_util.tree_map(lambda x: x.mean(), global_params),'\n\n')
                print()
                print("local_params in local_updated BEFORE",jax.tree_util.tree_map(lambda x: x.mean(), local_opt_state.params),'\n\n')
                print()
                print("local_control_variate in local_updated BEFORE",jax.tree_util.tree_map(lambda x: x.mean(), local_control_variate),'\n\n')
                print()
                print("global_control_variate in local_updated BEFORE",jax.tree_util.tree_map(lambda x: x.mean(), global_control_variate),'\n\n')

                print("starting local steps")
            losses = []
            for _ in range(args.num_local_steps): # Total number of local epochs
                key, key1 = jax.random.split(key) # Key is same so permutations are the same for each array
                s_c_images = split(jax.random.permutation(key1, im), len(im) // args.local_batch_size)
                s_c_labels = split(jax.random.permutation(key1, lab), len(lab) // args.local_batch_size)

                s_c_batch = []
                for i in range(len(im) // args.local_batch_size):
                    sub_batch_dict = {}
                    sub_batch_dict["image"] = s_c_images[i]
                    sub_batch_dict["label"] = s_c_labels[i]
                    s_c_batch.append(FlatMap(sub_batch_dict))

                for sub_client_batch in s_c_batch:  # One local epoch
                    params = local_opt.get_params(local_opt_state)
                    l, grad = jax.value_and_grad(task.loss)(params, key, sub_client_batch)
                    losses.append(l)
                    local_opt_state = local_opt.update(
                        opt_state=local_opt_state, 
                        grad=grad,
                        local_control_variate=local_control_variate,
                        global_control_variate=global_control_variate, 
                        loss=l)
            
            local_params = local_opt.get_params(local_opt_state)

            if VERBOSE:

                print("local_params in local_updated after",jax.tree_util.tree_map(lambda x: x.mean(), local_params))
                print()

            def update_cv(cvl, cvg, lr, pg, pl ):
                # LINE 12 of scaffold
                return cvl - cvg + lr * (pg - pl) 

            #IMPORTANT, we need to set the num_local_steps because it is 
            # determined at runtime with respect to the number of
            #  samples on each device
            num_local_steps = (len(im) // args.local_batch_size ) * args.num_local_steps

            lr_like_params = jax.tree_util.tree_map(lambda x: 1 / ( num_local_steps * local_opt.learning_rate), 
                                          global_params)

            if VERBOSE:

                print("LR LIKE PARAMS:",1 / ( num_local_steps * local_opt.learning_rate),)
            #Algo1 line 12: update the local control variate
            updated_cv = jax.tree_util.tree_map(
                update_cv,
                local_control_variate,
                global_control_variate,
                lr_like_params,
                global_params,
                local_params,
            )

            print("global-local param diff",jax.tree_util.tree_map(lambda x,y: (x-y).mean(), local_params, global_params))
        
            return jnp.mean(jnp.array(losses)), \
                   local_params, \
                   updated_cv
                   

        key, key1 = jax.random.split(key)


        # retrieve the local and grobal CVs
        # shape [num_clients, param_dims...]
        local_control_variates = opt_state.optax_opt_state[1]['local_control_variates']
        # shape [param_dims...]
        global_control_variate = opt_state.optax_opt_state[1]['global_control_variate']

        # get only the clients that are participating in the round
        # shape [num_participating_clients, param_dims...]
        indexed_local_control_variates = jax.tree_util.tree_map(lambda x: x[opt_idx], local_control_variates)
        losses, new_params, mew_local_cvs = jax.vmap(
            local_updates, 
            in_axes=(0, 0, 0, None, None)
        )(indexed_local_control_variates, 
        images, 
        labels, 
        key1, 
        global_control_variate)



        if VERBOSE:
        
            print("local_control_variates BEFORE",jax.tree_util.tree_map(lambda x: x.mean(), local_control_variates))
        # Updates the control variates within the matrix of all control variates
        local_control_variates = update_stacked_params( stacked_params=local_control_variates, 
                                                        indices=jax.tree_util.tree_map(lambda x: opt_idx, local_control_variates), 
                                                        updated_params=mew_local_cvs)

        if VERBOSE:
            print("local_control_variates AFTER",jax.tree_util.tree_map(lambda x: x.mean(), local_control_variates))
        
        # Line 13 of algorithms 1: getting the average delta for CVs and params
        avg_params_delta = jax.tree_util.tree_map(
            lambda p, new_p: jnp.mean(new_p - p , axis=0),
            opt.get_params(opt_state), 
            new_params
        )
        avg_cvs_delta = jax.tree_util.tree_map(
            lambda p, new_p: jnp.mean(new_p - p , axis=0),
            indexed_local_control_variates,
            mew_local_cvs
        )

        p_delta_mean = jax.tree_util.tree_map(lambda x: x.mean(), avg_params_delta)


        if VERBOSE:
            print("avg_params_delta",p_delta_mean)
        cv_delta_mean = jax.tree_util.tree_map(lambda x: x.mean(), avg_cvs_delta)
        if VERBOSE:
            print("avg_cvs_delta",cv_delta_mean)


        def update_param(param_delta, lr, params):
            return params + param_delta * lr




        global_lr_like_params = jax.tree_util.tree_map(lambda x: opt.global_learning_rate, avg_params_delta)
        if VERBOSE:
            print("global_lr_like_params",global_lr_like_params)


        # Update the parameters
        updated_params = jax.tree_util.tree_map(
            update_param,
            avg_params_delta,
            global_lr_like_params,
            opt_state.params,
            # current_params,
        )

        cv_lr_like_params = jax.tree_util.tree_map(lambda x: int(args.number_clients * args.participation_rate) / args.number_clients, avg_cvs_delta)
        # Update the cv
        updated_cv = jax.tree_util.tree_map(
            update_param,
            avg_cvs_delta,
            cv_lr_like_params,
            global_control_variate,
        )


        logging = {
            'cv_delta_mean': pytree_mean(cv_delta_mean),
            'param_delta_mean': pytree_mean(p_delta_mean),
            'global_params': pytree_mean(jax.tree_util.tree_map(lambda x: x.mean(), updated_params)),
            'global_cv': pytree_mean(jax.tree_util.tree_map(lambda x: x.mean(), updated_cv)),
            'local_cvs': pytree_mean(jax.tree_util.tree_map(lambda x: x.mean(), local_control_variates)),

        }

        return opt.init(updated_params, local_control_variates=local_control_variates, global_control_variate=updated_cv), jnp.mean(jnp.array(losses)), logging

    return opt, update
