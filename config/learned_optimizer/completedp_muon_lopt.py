# Config for the MuP Muon learned optimizer (MuCompletedPMuonLOpt).
#
# This is the meta-learnable counterpart of ``src/opt/mup_muon.py``. To use it,
# pair this file with a parameterization config that sets
# ``parameterization_args['parameterization_class'] = 'muon_completedp'`` so
# that ``mu_task_base._compute_completed_p_scales`` instantiates
# ``MuonCompletedPParameterization`` and emits the additional
# ``mup_muon_lr_scales`` / ``mup_muon_eps_scales`` / ``mup_is_muon_mask`` keys
# that this lopt consumes from ``model_state``.
#
# Defaults match the validated ``mup_muon`` factory in ``src/opt/mup_muon.py``.

learned_optimizer_args = dict(
    class_="MuCompletedPMuonLOpt",
    kwargs=dict(
        # Shared base LR (mup_muon factory default = 0.02)
        initial_lr=0.0221,
        # Muon side
        initial_muon_beta=0.95,
        initial_muon_eps=1e-8,
        initial_muon_wd=0.125,
        # Adam side (mup_muon factory defaults)
        initial_adam_b1=0.95484,
        initial_adam_b2=0.9908,
        initial_adam_eps=1e-8,
        initial_adam_wd=0.093198,
        # Newton-Schulz (Jordan et al. 2024 coefficients)
        ns_coeffs=(3.4445, -4.775, 2.0315),
        ns_steps=5,
        nesterov=True,
        # Gradient clipping (mirrors completedp_adam_lopt.py)
        clip_grad=True,
        clip_norm=1.0,
        # 19 covers all TensorType enum values
        num_tensor_types=19,
        # Built-in warmup + cosine decay schedule (mirrors completedp_adam_lopt.py)
        warmup_steps=100,
        decay_steps=1900,
        end_lr_ratio=0.1,
    )
)
