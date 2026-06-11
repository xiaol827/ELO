
from learned_optimization.optimizers.optax_opts import OptaxOptimizer
import jax
import optax
import gin
import chex
from typing import Any, Callable, Optional, Sequence, Union
import functools
import jax.numpy as jnp
from flax import struct
import numpy as np


ModelState = Any
Params = Any
Gradient = Params
OptState = Any

@struct.dataclass
class OptaxState:
  params: chex.ArrayTree
  state: chex.ArrayTree
  optax_opt_state: chex.ArrayTree
  iteration: jnp.ndarray
  mom_delta: chex.ArrayTree

def get_optax_optimizer(name, kwargs):
    opts_ = {
        "sm3": optax.sm3,
        "adabelief": optax.adabelief,
        "adadelta": optax.adadelta,
        "adan": optax.adan,
        "adafactor": optax.adafactor,
        "adagrad": optax.adagrad,
        "adam": optax.adam,
        "adamw": optax.adamw,
        "adamax": optax.adamax,
        "adamaxw": optax.adamaxw,
        "amsgrad": optax.amsgrad,
        "fromage": optax.fromage,
        "lamb": optax.lamb,
        "lars": optax.lars,
        "lbfgs": optax.lbfgs,
        "lion": optax.lion,
        "nadam": optax.nadam,
        "nadamw": optax.nadamw,
        "noisy_sgd": optax.noisy_sgd,
        "novograd": optax.novograd,
        "optimistic_gradient_descent": optax.optimistic_gradient_descent,
        "optimistic_adam": optax.optimistic_adam,
        "polyak_sgd": optax.polyak_sgd,
        "radam": optax.radam,
        "rmsprop": optax.rmsprop,
        "sgd": optax.sgd,
        "sign_sgd": optax.sign_sgd,
        "yogi": optax.yogi,
        "muon": optax.contrib.muon,
        "DoubleAdam": DoubleAdam,
        "double_adam": double_adam,
    }
    return opts_[name](**kwargs)


def get_optax_schedule(name, kwargs):
    sched_ = {
        "constant_schedule": optax.constant_schedule,
        "cosine_decay_schedule": optax.cosine_decay_schedule,
        "cosine_onecycle_schedule": optax.cosine_onecycle_schedule,
        "exponential_decay": optax.exponential_decay,
        "join_schedules": optax.join_schedules,
        "linear_onecycle_schedule": optax.linear_onecycle_schedule,
        "linear_schedule": optax.linear_schedule,
        "piecewise_constant_schedule": optax.piecewise_constant_schedule,
        "piecewise_interpolate_schedule": optax.piecewise_interpolate_schedule,
        "polynomial_schedule": optax.polynomial_schedule,
        "sgdr_schedule": optax.sgdr_schedule,
        "warmup_constant_schedule": optax.warmup_constant_schedule,
        "warmup_cosine_decay_schedule": optax.warmup_cosine_decay_schedule,
        "warmup_exponential_decay_schedule": optax.warmup_exponential_decay_schedule,
    }
    return sched_[name](**kwargs)


def get_gradient_transformation(name, kwargs):
    grad_trans_ = {
        "clip_by_global_norm": optax.clip_by_global_norm,
        "clip_by_block_rms": optax.clip_by_block_rms,
        "clip": optax.clip,
        "add_decayed_weights": optax.add_decayed_weights,
    }
    return grad_trans_[name](**kwargs)




# class EnhancedOptaxOptimizer(OptaxOptimizer):
#     """OptaxOptimizer with resume_init capability."""
# @functools.partial(jax.jit, static_argnums=(0,))
# def resume_init(self,
#         opt_state: OptaxState,
#         params: Params,
#         model_state: Optional[ModelState] = None,
#         key: Optional[chex.PRNGKey] = None):
#     # Update Model parameters and state
#     return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
#         state=model_state,
#         params=params,
#         optax_opt_state=opt_state.optax_opt_state,
#         iteration=opt_state.iteration,
#     )


@gin.configurable
class AnyOptimizer(OptaxOptimizer):
    """Optax optimizer wrapper"""
    
    # @functools.partial(jax.jit, static_argnums=(0,))
    # def update(
    #     self,
    #     opt_state: OptaxState,
    #     grad: Params,
    #     loss: Optional[jnp.ndarray] = None,
    #     model_state: Optional[ModelState] = None,
    #     key: Optional[chex.PRNGKey] = None,
    #     **kwargs
    # ) -> OptaxState:
    #     """Update the parameters and state.
        
    #     Overrides the parent method to accept all kwargs.
    #     """
    #     return super().update(opt_state, grad, loss, model_state, key)


    def __init__(
        self, 
        optimizer,
        schedule,
        gradient_transform_before_optim,
        gradient_transform_after_optim,
        mup_lrs=None,
        local_optimizer_args=None,
        use_error_correction=False,
        ec_beta=None):
        self.local_optimizer_args = local_optimizer_args
        self.use_error_correction = use_error_correction
        self.ec_beta = ec_beta

        if use_error_correction:
            assert ec_beta is not None, "ec_beta must be provided if use_error_correction is True"

        optimizer_args = []

        ############################################################
        # Setup gradient transformations before optimizer
        ############################################################    
        for x in gradient_transform_before_optim:
            optimizer_args.append(
                get_gradient_transformation(x['class_'], 
                                            x['kwargs'])
            )

        ############################################################
        # Setup schedule + optimizer
        ############################################################
        schedule = get_optax_schedule(schedule['class_'],
                                      schedule['kwargs'])
        self.schedule_ = schedule
        optimizer['kwargs']['learning_rate'] = schedule
        optimizer_args.append(get_optax_optimizer(optimizer['class_'], 
                                                  optimizer['kwargs']))


        ############################################################
        # Setup MuP LRS
        ############################################################
        if mup_lrs is not None:
            def init_fn(params):
                del params
                return optax.EmptyState()
            
            def update_fn(updates, state, params=None):
                del params
                updates = jax.tree_util.tree_map(
                    lambda update, scale: update * scale,
                    updates,
                    mup_lrs
                )
                return updates, state

            optimizer_args.append(optax.GradientTransformation(init_fn, update_fn))

        ############################################################
        # Setup gradient transformations after optimizer
        ############################################################  
        for x in gradient_transform_after_optim:
            optimizer_args.append(
                get_gradient_transformation(x['class_'], 
                                            x['kwargs'])
            )
        opt = optax.chain(*optimizer_args)
        super().__init__(opt)


    def get_local_optimizer(self, mup_lrs=None):
        if self.local_optimizer_args['use_mup']:
            assert mup_lrs is not None, "mup_lrs must be provided if use_mup is True"

        return AnyOptimizer(
            optimizer=self.local_optimizer_args['optimizer_args'],
            schedule=self.local_optimizer_args['schedule'],
            gradient_transform_before_optim=self.local_optimizer_args['gradient_transform_before_optim'],
            gradient_transform_after_optim=self.local_optimizer_args['gradient_transform_after_optim'],
            mup_lrs=mup_lrs if self.local_optimizer_args['use_mup'] else None,
            local_optimizer_args=self.local_optimizer_args,
            use_error_correction=self.local_optimizer_args['use_error_correction'],
            ec_beta=self.local_optimizer_args['ec_beta']
        )

    def init(self,
           params: Params,
           model_state: Optional[ModelState] = None,
           num_steps: Optional[int] = None,
           key: Optional[chex.PRNGKey] = None,
           beta_delta: Optional[float] = 0.9,
           ):
        if self.use_error_correction:
            mom_delta = jax.tree_util.tree_map(jnp.zeros_like, params)
        else:
            mom_delta = None

        return OptaxState(
            params=params,
            optax_opt_state=self.opt.init(params),
            state=model_state,
            iteration=0,
            mom_delta=mom_delta
        )
  


    @functools.partial(jax.jit, static_argnums=(0,))
    def resume_init(self,
            opt_state: OptaxState,
            params: Params,
            model_state: Optional[ModelState] = None,
            key: Optional[chex.PRNGKey] = None):
        return OptaxState(  
            state=model_state,
            params=params,
            optax_opt_state=opt_state.optax_opt_state,
            iteration=opt_state.iteration,
            mom_delta=opt_state.mom_delta
        )
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def update_mom_delta(self, opt_state, delta):
            new_mom_delta = jax.tree_util.tree_map(lambda x, y: self.ec_beta * x + y, opt_state.mom_delta, delta)
            return OptaxState(
                state=opt_state.state,
                params=opt_state.params,
                optax_opt_state=opt_state.optax_opt_state,
                iteration=opt_state.iteration,
                mom_delta=new_mom_delta
            )
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def correct_mom_delta(self, opt_state, compressed_mom):
        corrected_mom_delta = jax.tree_util.tree_map(lambda x, y: x - y,  opt_state.mom_delta, compressed_mom)
        return OptaxState(
            state=opt_state.state,
            params=opt_state.params,
            optax_opt_state=opt_state.optax_opt_state,
            iteration=opt_state.iteration,
            mom_delta=corrected_mom_delta
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self,
                opt_state: OptaxState,
                grad: Gradient,
                loss: Optional[jnp.ndarray] = None,
                model_state: Optional[ModelState] = None,
                key: Optional[chex.PRNGKey] = None,
                **kwargs):
        del loss
        update, new_opt_state = self.opt.update(
            opt_state.optax_opt_state,
            grad,
            params=opt_state.params,
            # **kwargs
        )
        return OptaxState(
            state=model_state,
            params=optax.apply_updates(opt_state.params, update),
            optax_opt_state=new_opt_state,
            iteration=opt_state.iteration + 1,
            mom_delta=opt_state.mom_delta,
        )






@gin.configurable
class DoubleAdam(OptaxOptimizer):
    """Stochastic gradient descent with momentum."""

    def __init__(self,
                 learning_rate,
                 merging_rate,
                 adam_bc,
                 adam_es,
                 ):
        self.adam_bc = get_optax_optimizer(adam_bc['class_'], 
                                                  adam_bc['kwargs'])
        self.adam_es = get_optax_optimizer(adam_es['class_'], 
                                                  adam_es['kwargs'])
        self.merging_rate = get_optax_schedule(merging_rate['class_'], 
                                                  merging_rate['kwargs'])

        self.learning_rate =  get_optax_schedule(learning_rate['class_'], 
                                                  learning_rate['kwargs'])
        # opt = optax.sgd(learning_rate)
        # super().__init__(opt)


    def init(
        self,
        params: Params,
        model_state: Optional[ModelState] = None,
        num_steps: Optional[int] = None,
        key: Optional[chex.PRNGKey] = None,
    ):
        adam_bc_opt_state = self.adam_bc.init(params)
        adam_es_opt_state = self.adam_es.init(params)
        # import pprint
        # print("adam_bc_opt_state")
        # pprint.pprint(adam_bc_opt_state)
        # print("adam_es_opt_state")
        # pprint.pprint(adam_es_opt_state)
        # exit()

        return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            params=params,
            optax_opt_state=[
                adam_bc_opt_state,
                adam_es_opt_state,
            ],
            state=model_state,
            iteration=0,
            mom_delta=None,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        opt_state: OptaxState,
        grad: Gradient,
        grad_bc: Gradient,
        grad_es: Gradient,
        loss: Optional[jnp.ndarray] = None,
        model_state: Optional[ModelState] = None,
        key: Optional[chex.PRNGKey] = None,
        **kwargs,
    ):
        del loss
        # print("opt_state.optax_opt_state[0]")
        # pprint.pprint(opt_state.optax_opt_state[0])

        # opt_state = opt_state.optax_opt_state[1]
        # print("opt_state.optax_opt_state[1]")
        # pprint.pprint(opt_state.optax_opt_state[1])
        # exit()
        update_bc, new_opt_state_bc = self.adam_bc.update(
            grad_bc, opt_state.optax_opt_state[0], opt_state.params
        )
        update_es, new_opt_state_es = self.adam_es.update(
            grad_es, opt_state.optax_opt_state[1], opt_state.params
        )

        merging_rate = self.merging_rate(opt_state.iteration)
        learning_rate = self.learning_rate(opt_state.iteration)
        
        merged_update = jax.tree_util.tree_map(
            lambda bc, es, orig: 
            learning_rate * ((bc - orig) * merging_rate 
                                + (es - orig) * (1 - merging_rate)), 
                update_bc, update_es, opt_state.params
            )

        return OptaxState(
            state=model_state,
            params=optax.apply_updates(opt_state.params, merged_update),
            optax_opt_state=[
                new_opt_state_bc,
                new_opt_state_es,
            ],
            iteration=opt_state.iteration + 1,
            mom_delta=None
        )

def double_adam(learning_rate, merging_rate_sched, adam_bc, adam_es):
    # Create the two Adam optimizers
    # adam_bc_opt = optax.adam(**adam_bc)
    # adam_es_opt = optax.adam(**adam_es)
    # merging_rate = optax.linear_schedule(**merging_rate_sched)

    adam_bc_opt = get_optax_optimizer(adam_bc['class_'], 
                                                  adam_bc['kwargs'])
    adam_es_opt = get_optax_optimizer(adam_es['class_'], 
                                                  adam_es['kwargs'])
    merging_rate = get_optax_schedule(merging_rate_sched['class_'], 
                                                  merging_rate_sched['kwargs'])
    def init_fn(params):
        return {
            'adam_bc': adam_bc_opt.init(params),
            'adam_es': adam_es_opt.init(params),
            'iteration': jnp.array(0, dtype=jnp.int32),
        }

    def update_fn(updates, state, params=None):
        # You may want to pass in both grad_bc and grad_es, but Optax expects a single update.
        # So, you can pass a tuple: (grad_bc, grad_es)
        grad_bc, grad_es = updates

        # Compute Adam updates
        update_bc, new_state_bc = adam_bc_opt.update(grad_bc, state['adam_bc'], params)
        update_es, new_state_es = adam_es_opt.update(grad_es, state['adam_es'], params)

        # Compute merging rate
        m_rate = merging_rate(state['iteration'])

        # Merge the updates
        merged_update = jax.tree_util.tree_map(
            lambda bc, es: learning_rate * (bc * m_rate + es * (1 - m_rate)),
            update_bc, update_es
        )

        new_state = {
            'adam_bc': new_state_bc,
            'adam_es': new_state_es,
            'iteration': state['iteration'] + 1,
        }
        return merged_update, new_state

    return optax.GradientTransformation(init_fn, update_fn)


if __name__ == "__main__":
    # Set up a toy optimization problem
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    
    # Create a simple quadratic function to optimize
    def loss_fn(params, x, y):
        pred = jnp.sum(params['w'] * x) + params['b']
        return jnp.mean((pred - y) ** 2)
    
    # Initialize parameters
    params = {
        'w': jax.random.normal(key, (10,)),
        'b': jnp.array(0.0)
    }
    
    # Create random data
    key, subkey = jax.random.split(key)
    x = jax.random.normal(key, (10,))
    true_w = jnp.ones((10,))
    true_b = 2.0
    y = jnp.sum(true_w * x) + true_b + 0.1 * jax.random.normal(subkey, ())
    
    # Define gradient function
    @jax.jit
    def compute_grads(params, x, y):
        return jax.grad(loss_fn)(params, x, y)
    
    # Configure DoubleAdam optimizer directly
    learning_rate = dict(
    class_="constant_schedule",
    kwargs=dict(
        value=0.1
        )
    )
    merging_rate_sched = {
        'class_': 'linear_schedule',
        'kwargs': {
            'init_value': 1.0,
            'end_value': 0.1,
            'transition_steps': 100,
            'transition_begin': 500
        }
    }
    adam_bc = {
        'class_': 'adam',
        'kwargs': {
            'learning_rate': 1.0,
            'b1': 0.9, 
            'b2': 0.999, 
            'eps': 1e-8
        }
    }
    adam_es = {
        'class_': 'adam',
        'kwargs': {
            'learning_rate': 1.0,
            'b1': 0.9, 
            'b2': 0.999, 
            'eps': 1e-8
        }
    }
    
    # Initialize DoubleAdam directly using the class
    optimizer = DoubleAdam(
        learning_rate=learning_rate,
        merging_rate=merging_rate_sched,
        adam_bc=adam_bc,
        adam_es=adam_es
    )
    
    # Initialize optimizer state
    opt_state = optimizer.init(params)
    
    # Run optimization for a few steps
    num_steps = 1000
    losses = []
    
    print("Starting optimization...")
    for i in range(num_steps):
        # Compute gradients
        grads = compute_grads(opt_state.params, x, y)
        
        # For DoubleAdam, we need both BC and ES gradients
        # In this toy example, we'll use the same gradients for both
        grad_bc = grads
        grad_es = grads
        
        # Update parameters using the optimizer
        opt_state = optimizer.update(
            opt_state=opt_state,
            grad=None,
            grad_bc=grad_bc,
            grad_es=grad_es,
        )
        
        # Compute loss for tracking
        current_loss = loss_fn(opt_state.params, x, y)
        losses.append(current_loss)
        
        if i % 100 == 0:
            print(f"Step {i}, Loss: {current_loss:.6f}")
    
    print("\nOptimization complete!")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"True parameters: w={true_w}, b={true_b}")
    print(f"Learned parameters: w={opt_state.params['w']}, b={opt_state.params['b']}")