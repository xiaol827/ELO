_base_ = [
    "adamw.py"
]

local_optimizer_args = dict(

use_error_correction = True,
ec_beta = 0.9

)
