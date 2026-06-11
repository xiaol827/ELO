"""
WandB sweep config: MuP Muon base HP tuning at w128-d4.

Matches the training setup from the CompletedP Adam baseline:
  Task:   mutransformer-dense-w128-d4-h4_fineweb-s128-gpt2
  Param:  complete_p_bs100k_steps2000.py
  Steps:  2000 (warmup=100, decay=1900)
  Batch:  local_batch_size=64, gradient_accumulation_steps=128

Adam branch defaults from optimal mup_adamw sweep:
  adam_b1=0.95484, adam_b2=0.9908, adam_weight_decay=0.093198

Swept HPs (3 axes):
  - LR (via schedule peak_value): 14 powers of sqrt(2) in [0.005, 0.5]
  - Muon weight_decay: 14 powers of sqrt(2) in [0.005, 0.5]
  - Muon beta (momentum): [0.85, 0.9, 0.95]

Total: 14 * 14 * 3 = 588 configs.
Each config: ~5-10 min at w128-d4 for 2000 steps on 1 GPU.

Usage (job script):
    python src/main.py \\
        --config config/sweeps/sweep_mup_muon_base.py,config/schedule/warmup_cosine_decay.py,config/gradient_transform/before/clip_by_global_norm.py,config/gradient_transform/after/none.py,config/optimizer/mup_muon.py,config/parameterization/complete_p_bs100k_steps2000.py \\
        --cfg_options gradient_transform_before_optim.0.kwargs.max_norm=1.0 \\
        --task mutransformer-dense-w128-d4-h4_fineweb-s128-gpt2 \\
        --optimizer mup_muon \\
        --local_batch_size 64 \\
        --gradient_accumulation_steps 128 \\
        --ovr_test_batch_size 128 \\
        --num_inner_steps 2000 \\
        --num_runs 1 \\
        --needs_state \\
        --test_project mup-muon-sweep \\
        --test_interval 15 \\
        --name_suffix _mup_muon_base_sweep
"""

_base_ = ["./sweeps_base.py"]

import math

# Powers of sqrt(2) in [0.005, 0.5]: 2^(k/2) for k = -15, -14, ..., -2
_sqrt2_values = [2 ** (k / 2) for k in range(-15, -1)]  # 14 values

# Build schedule dicts: warmup=100, decay=1900, end_value = peak / 2
_schedule_values = [
    {
        'init_value': 0.0,
        'peak_value': pv,
        'warmup_steps': 100,
        'decay_steps': 1900,
        'end_value': pv / 2.0,
        'exponent': 1.0,
    }
    for pv in _sqrt2_values
]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(

        # Muon learning rate (via schedule peak_value)
        # AnyOptimizer replaces optimizer learning_rate with the schedule callable,
        # so we sweep the schedule kwargs directly.
        # Powers of sqrt(2) from ~0.005 to 0.5 (14 values)
        schedule__kwargs=dict(
            values=_schedule_values,
        ),

        # Muon weight decay (base value; MuP wd_scales handle per-param 1/width scaling)
        # Same powers of sqrt(2) range as LR (14 values)
        optimizer_args__kwargs__weight_decay=dict(
            values=_sqrt2_values,
        ),

        # Muon momentum (applied before Newton-Schulz orthogonalization)
        # Nesterov=True (fixed). 3 values around the typical Muon sweet spot.
        optimizer_args__kwargs__beta=dict(
            values=[0.85, 0.9, 0.95],
        ),

    ),
)
