from .mup_adafac_mlp_lopt import MuAdafacMLPLOpt
from learned_optimization.learned_optimizers.adafac_mlp_lopt import AdafacMLPLOpt

from .elo_adfac_mlp_lopt import ELO_AdafacMLPLOpt
from .chen_adfac_mlp_lopt import ChenAdafacMLPLOpt
from .celo2_lopt import Celo2LOpt
from .elo_celo2 import ELO_Celo2LOpt


__all__ = [
    "MuAdafacMLPLOpt",
    "AdafacMLPLOpt",

    "ELO_AdafacMLPLOpt",
    "ChenAdafacMLPLOpt",
    "Celo2LOpt",
    "ELO_Celo2LOpt",
]


def build_learned_optimizer(args):
    lopts = {
        "MuAdafacMLPLOpt".lower(): MuAdafacMLPLOpt,
        "AdafacMLPLOpt".lower(): AdafacMLPLOpt,

        "ELO_AdafacMLPLOpt".lower(): ELO_AdafacMLPLOpt,
        "ChenAdafacMLPLOpt".lower(): ChenAdafacMLPLOpt,
        "Celo2LOpt".lower(): Celo2LOpt,
        "ELO_Celo2LOpt".lower(): ELO_Celo2LOpt,
    }

    lopt_class = args.learned_optimizer_args['class_']
    lopt_args = args.learned_optimizer_args['kwargs']

    return lopts[lopt_class.lower()](**lopt_args)
