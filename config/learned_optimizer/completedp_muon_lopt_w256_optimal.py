# Optimal MuP Muon learned optimizer config for w256-d4-h2.
#
# HPs from wandb run <NEED>/mup-muon-optimal-w256/runs/cm5itlja
# (mup_muon sweep on muztransformer-dense-w256-d4-h2_fineweb-s512-gpt2).
#
# Muon HPs from Muon sweep i3mm78cu (best=3.8328).
# Adam HPs from original CompletedP Adam defaults (as used in the optimal run).

learned_optimizer_args = dict(
    class_="MuCompletedPMuonLOpt",
    kwargs=dict(
        # Base LR
        initial_lr=0.011,
        # Muon side
        initial_muon_beta=0.95,
        initial_muon_eps=1e-8,
        initial_muon_wd=0.0156,
        # Adam side (CompletedP Adam defaults from the optimal run)
        initial_adam_b1=0.95484,
        initial_adam_b2=0.9908,
        initial_adam_eps=1e-8,
        initial_adam_wd=0.093198,
        # Newton-Schulz (Jordan et al. 2024 coefficients)
        ns_coeffs=(3.4445, -4.775, 2.0315),
        ns_steps=5,
        nesterov=True,
        # Gradient clipping
        clip_grad=True,
        clip_norm=1.0,
        # 19 covers all TensorType enum values
        num_tensor_types=19,
        # Built-in warmup + cosine decay schedule
        warmup_steps=500,
        decay_steps=4000,
        end_lr_ratio=0.1,
    )
)
