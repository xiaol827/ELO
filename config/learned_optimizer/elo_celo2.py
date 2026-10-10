learned_optimizer_args = dict(
    class_="ELO_Celo2LOpt",
    kwargs=dict(
        # CELO2 backbone
        orthogonalize=True,
        # bias correction of MLP input features (official celo2 meta-training: False)
        bias_correction=False,
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
        # Adam for 1D params (shares a dedicated 1D accumulator with the expert)
        adam_lr_mult=20.0,
        adam_epsilon=1e-8,
        # celo meta-training uses bias_correction=False for the 1D Adam
        adam_bias_correction=False,
        adam_weight_decay=None,
        use_adamw_for_1d=True,
        # Gradient clipping
        clip_grad=True,
        clip_norm=1.0,
        # elementwise grad clip magnitude (official celo2: 1000.0)
        clip_value=1000.0,
        # Expert (uses momentum[0]/rms[-1] from shared accumulators)
        expert_lr_max=0.01,
        expert_lr_min=1e-4,
        expert_lr_decay_steps=10000,
        expert_weight_decay=0.0,
        expert_optim="adamw",
        muon_expert_adamlr_scaler=0.3,
        # Expert accumulator channel indices (which decay channel each expert reads)
        exp_sgdm_mom_idx=0,
        exp_adam_mom_idx=0,
        exp_muon_mom_idx=1,
        exp_adam_rms_idx=-1,
        # Mode
        meta_train=True,
    )
)
