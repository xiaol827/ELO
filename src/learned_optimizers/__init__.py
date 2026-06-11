from .mup_adafac_mlp_lopt import MuAdafacMLPLOpt
from .mup_adafac_mlp_lopt_bc import MuAdafacMLPLOptBC
from .mup_adafac_mlp_lopt_bc_full_opt import MuAdafacMLPLOptBC
from .mup_adafac_mlp_lopt_v2 import MuAdafacMLPLOptv2
from .mup_hyper import MuHyperV2
from .mup_rnn import MuRNNMLPLOpt
from .hyper_v2_new import HyperV2
# from .fed_mlp_lopt import FedMLPLOpt
from .fed_adafac_mlp_lopt import FedAdafacMLPLOpt
from .mup_hyper_bc import MuHyperV2BC
from learned_optimization.learned_optimizers.adafac_mlp_lopt import AdafacMLPLOpt
from learned_optimization.learned_optimizers.rnn_mlp_lopt import RNNMLPLOpt
from .mup_adafac_mlp_lopt_v3 import MuAdafacMLPLOptV3
from .mup_adafac_mlp_lopt_v4 import MuAdafacMLPLOptV4
from .mup_adafac_mlp_lopt_v4_bc import MuAdafacMLPLOptV4BC
from .mup_adafac_mlp_lopt_completed_p import MuCompletedPAdafacMLPLOpt

from .elo_adfac_mlp_lopt import ELO_AdafacMLPLOpt
from .chen_adfac_mlp_lopt import ChenAdafacMLPLOpt
from .celo2_lopt import Celo2LOpt
from .elo_celo2 import ELO_Celo2LOpt

from .mup_adam_lopt_completed_p import MuCompletedPAdamLOpt
from .mup_muon_lopt_completed_p import MuCompletedPMuonLOpt


__all__ = [
    "MuAdafacMLPLOpt",
    "MuAdafacMLPLOptBC",
    "MuAdafacMLPLOptv2",
    "MuHyperV2",
    "MuRNNMLPLOpt",
    "AdafacMLPLOpt",
    "RNNMLPLOpt",
    "HyperV2",
    "MuHyperV2BC",
    "FedAdafacMLPLOpt",
    "MuAdafacMLPLOptV3",
    "MuAdafacMLPLOptV4",
    "MuAdafacMLPLOptV4BC",
    "MuCompletedPAdafacMLPLOpt",

    "ELO_AdafacMLPLOpt",
    "ChenAdafacMLPLOpt",
    "Celo2LOpt",
    "ELO_Celo2LOpt",

    "MuCompletedPAdamLOpt",
    "MuCompletedPMuonLOpt",
]


def build_learned_optimizer(args):
    flopts = {
        "FedAdafacMLPLOpt".lower(): FedAdafacMLPLOpt,
    }

    lopts = {
        "MuAdafacMLPLOpt".lower(): MuAdafacMLPLOpt,
        "MuAdafacMLPLOptBC".lower(): MuAdafacMLPLOptBC,
        "MuAdafacMLPLOptv2".lower(): MuAdafacMLPLOptv2,
        "MuHyperV2".lower(): MuHyperV2,
        "MuHyperV2BC".lower(): MuHyperV2BC,
        "HyperV2".lower(): HyperV2,
        "MuRNNMLPLOpt".lower(): MuRNNMLPLOpt,
        "AdafacMLPLOpt".lower(): AdafacMLPLOpt,
        "MuAdafacMLPLOptV3".lower(): MuAdafacMLPLOptV3,
        "MuAdafacMLPLOptV4".lower(): MuAdafacMLPLOptV4,
        "MuAdafacMLPLOptV4BC".lower(): MuAdafacMLPLOptV4BC,
        "MuCompletedPAdafacMLPLOpt".lower(): MuCompletedPAdafacMLPLOpt,

        "ELO_AdafacMLPLOpt".lower(): ELO_AdafacMLPLOpt,
        "ChenAdafacMLPLOpt".lower(): ChenAdafacMLPLOpt,
        "Celo2LOpt".lower(): Celo2LOpt,
        "ELO_Celo2LOpt".lower(): ELO_Celo2LOpt,

        "MuCompletedPAdamLOpt".lower(): MuCompletedPAdamLOpt,
        "MuCompletedPMuonLOpt".lower(): MuCompletedPMuonLOpt,
        # "FedMLPLOpt".lower(): FedMLPLOpt,
    }
    lopts.update(flopts)

    lopt_class = args.learned_optimizer_args['class_']
    lopt_args = args.learned_optimizer_args['kwargs']

    if lopt_class in flopts.keys():
        lopt_args.update( 
            {"local_optimizer_args", args.local_optimizer_args}
        )


    return lopts[lopt_class.lower()](**lopt_args)



        