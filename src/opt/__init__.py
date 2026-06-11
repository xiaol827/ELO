from .new_optimizers import AnyOptimizer
from .mup_adamw import (
    mup_adamw,
    # mup_adam,
    # scale_by_mup_adam,
    # scale_by_mup_lr,
    # mup_add_decayed_weights,
    # create_mup_adamw_from_parameterization,
    # create_mup_adamw_with_tensor_types,
    # ScaleByMuPAdamState,
    # MuPAdamWState,
)

# from learned_optimization.optimizers.optax_opts import (
#     SGD, 
#     Adam, 
#     AdamW, 
#     AdaBelief, 
#     RMSProp, 
#     Adafactor, 
#     AdaGrad, 
#     Yogi, 
#     SM3, 
#     Lars, 
#     Lamb, 
#     RAdam
# )
# from learned_optimization.optimizers.shampoo import (
#     Shampoo
# )

__all__ = [
    # "Muon",
    # "SGD",
    # "Adam",
    # "AdamW",
    # "AdaBelief",
    # "RMSProp",
    # "Adafactor",
    # "AdaGrad",
    # "Yogi",
    # "SM3",
    # "Lars",
    # "Lamb",
    # "RAdam",
    "AnyOptimizer",
    # MuP AdamW components
    "mup_adamw",
    # "mup_adam",
    # "scale_by_mup_adam",
    # "scale_by_mup_lr",
    # "mup_add_decayed_weights",
    # "create_mup_adamw_from_parameterization",
    # "create_mup_adamw_with_tensor_types",
    # "ScaleByMuPAdamState",
    # "MuPAdamWState",
]


# def build_optimizer(opt_class, opt_kwargs):
#     opts = {
#         "Muon".lower(): Muon,
#         "SGD".lower(): SGD,
#         "Adam".lower(): Adam,
#         "AdamW".lower(): AdamW,
#         "AdaBelief".lower(): AdaBelief,
#         "RMSProp".lower(): RMSProp,
#         "Adafactor".lower(): Adafactor,
#         "AdaGrad".lower(): AdaGrad,
#         "Yogi".lower(): Yogi,
#         "SM3".lower(): SM3,
#         "Lars".lower(): Lars,
#         "Lamb".lower(): Lamb,
#         "RAdam".lower(): RAdam,
#         "Shampoo".lower(): Shampoo,
#     }

#     # lopt_class = args.optimizer_args['class_']
#     # lopt_args = args.optimizer_args['kwargs']
#     return lopts[opt_class.lower()](**opt_kwargs)

