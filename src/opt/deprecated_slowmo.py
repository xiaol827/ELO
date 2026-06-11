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
class SGDClip(OptaxOptimizer):
    """Adam with a piecewise linear learning rate schedule."""

    def __init__(
        self,
        learning_rate=0.01,
        clip=None,
        weight_decay=None,
        args=None,
    ):
        if weight_decay is None:
            if clip is None:
                opt = optax.sgd(learning_rate=learning_rate)
            else:
                opt = optax.chain(
                    optax.clip_by_global_norm(clip),
                    optax.sgd(learning_rate=learning_rate),
                )
        else:
            if clip is None:
                opt = optax.chain(
                    optax.add_decayed_weights(weight_decay),
                    optax.sgd(learning_rate=learning_rate),
                )
            
            else:
                if args.piecewise_schedule == {}:
                    opt = optax.chain(
                        optax.clip_by_global_norm(clip),
                        optax.add_decayed_weights(weight_decay),
                        optax.sgd(learning_rate=learning_rate),
                    )
                else:
                    print(args.piecewise_schedule)
                    schedule = optax.piecewise_constant_schedule(
                        **args.piecewise_schedule
                    )
                    opt = optax.chain(
                        optax.clip_by_global_norm(clip),
                        optax.add_decayed_weights(weight_decay),
                        optax.sgd(learning_rate=schedule),
                    )
                    
        super().__init__(opt)
    
@gin.configurable
class AdamClip(OptaxOptimizer):
    """Adam with a piecewise linear learning rate schedule."""

    def __init__(
        self,
        learning_rate=0.01,
        clip=None,
        weight_decay=None,
        args=None,
    ):
        if weight_decay is None:
            if clip is None:
                opt = optax.adam(learning_rate=learning_rate)
            else:
                opt = optax.chain(
                    optax.clip_by_global_norm(clip),
                    optax.adam(learning_rate=learning_rate, 
                          b1=args.benchmark_b1,
                          b2=args.benchmark_b2),
                )
        else:
            if clip is None:
                opt = optax.chain(
                    optax.add_decayed_weights(weight_decay),
                    optax.adam(learning_rate=learning_rate, 
                          b1=args.benchmark_b1,
                          b2=args.benchmark_b2),
                )
            
            else:
                if args.piecewise_schedule == {}:
                    opt = optax.chain(
                        optax.clip_by_global_norm(clip),
                        optax.add_decayed_weights(weight_decay),
                        optax.adam(learning_rate=learning_rate, 
                          b1=args.benchmark_b1,
                          b2=args.benchmark_b2),
                    )
                else:
                    print(args.piecewise_schedule)
                    schedule = optax.piecewise_constant_schedule(
                        **args.piecewise_schedule
                    )
                    opt = optax.chain(
                        optax.clip_by_global_norm(clip),
                        optax.add_decayed_weights(weight_decay),
                        optax.adam(learning_rate=schedule, 
                          b1=args.benchmark_b1,
                          b2=args.benchmark_b2),
                    )
                    
        super().__init__(opt)

@gin.configurable
class SGDSlowMo(OptaxOptimizer):
    """Stochastic gradient descent with momentum."""

    def __init__(self, learning_rate=0.01, momentum=0.9, clip=None, weight_decay=None):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum_state = None
        # self.opt = SGDClip(
        #     learning_rate=learning_rate,
        #     clip=clip,
        #     weight_decay=weight_decay,
        # )
        if weight_decay is None:
            if clip is None:
                opt = optax.sgd(learning_rate=learning_rate)
            else:
                opt = optax.chain(
                    optax.clip_by_global_norm(clip),
                    optax.sgd(learning_rate=learning_rate),
                )
        else:
            if clip is None:
                opt = optax.chain(
                    optax.add_decayed_weights(weight_decay),
                    optax.sgd(learning_rate=learning_rate),
                )
            
            else:
                opt = optax.chain(
                    optax.clip_by_global_norm(clip),
                    optax.add_decayed_weights(weight_decay),
                    optax.sgd(learning_rate=learning_rate),
                )
        super().__init__(opt)

    @property
    def name(self):
        return f"SGDSlowMo_lr{self.learning_rate}_m{self.momentum}"

    def init(
        self,
        params: Params,
        momentum: Optional[Params] = None,
        model_state: Optional[ModelState] = None,
        num_steps: Optional[int] = None,
        key: Optional[chex.PRNGKey] = None,
    ):
        # opt_state = self.opt.init(params)
        return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            params=params,
            optax_opt_state=[
                self.opt.init(params),
                {"momentum": jax.tree_util.tree_map(lambda x : jax.numpy.zeros(shape=x.shape), params)} # copy.deepcopy(params)
                if momentum is None
                else {"momentum": momentum},
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
        # print("opt state", opt_state)
        # print(opt_state.__dict__.keys())
        # print({k:len(v) for k,v in opt_state.__dict__.items()})
        # print(len(opt_state.optax_opt_state))
        # exit(0)
        #.optax_opt_state[0]
        # update, new_opt_state = self.opt.update(
        #         opt_state=opt_state.optax_opt_state[0],
        #         grad=grad,
        #         # loss=loss,
        #         model_state=model_state,
        #         key=key,
        # )
        update, new_opt_state = self.opt.update(grad, 
                                opt_state.optax_opt_state[0],
                                opt_state.params)
        return OptaxState(
            state=model_state,
            params=optax.apply_updates(opt_state.params, update),
            optax_opt_state=[
                new_opt_state,
                opt_state.optax_opt_state[1],
            ],
            iteration=opt_state.iteration + 1,
        )
