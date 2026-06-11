learned_optimizer_args = dict(
    class_="ELO_Celo2LOpt",
    kwargs=dict(
        # CELO2 backbone
        orthogonalize=True,
        ff_hidden_size=8,
        ff_hidden_layers=2,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        initial_rms_decays=(0.95,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        # LR schedule
        init_lr=0.0,
        peak_lr=1e-3,
        warmup_steps=0,
        warmup_fraction=0.05,
        end_lr=0.0,
        # Weight decay
        weight_decay=0.1,
        # Adam for 1D params (uses momentum[0] and rms[-1] from shared accumulators)
        adam_lr_mult=1.0,
        adam_weight_decay=None,
        use_adamw_for_1d=True,
        # Gradient clipping
        clip_grad=True,
        clip_norm=1.0,
        # Expert (uses momentum[0]/rms[-1] from shared accumulators)
        expert_lr_max=0.01,
        expert_lr_min=1e-4,
        expert_lr_decay_steps=10000,
        expert_weight_decay=0.0,
        expert_optim="adamw",
        muon_expert_adamlr_scaler=0.3,
        # Mode
        meta_train=True,
    )
)
