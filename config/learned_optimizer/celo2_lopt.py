
learned_optimizer_args = dict(
    class_="Celo2LOpt",
    kwargs=dict(
        checkpoint_path="FILL_IN_LOCAL_CHECKPOINT_PATH",
        # LR schedule params (optax.schedules.warmup_cosine_decay_schedule)
        init_lr=0.0,
        peak_lr=1e-3,
        warmup_steps=0,
        warmup_fraction=0.05,
        end_lr=0.0,
        # Weight decay (optax.add_decayed_weights)
        weight_decay=0.0,
        # Adam for 1D params (biases, norms, embeddings)
        adam_lr_mult=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        # celo meta-training uses bias_correction=False for the 1D Adam;
        # the official inference recipe (optax.adamw) uses True
        adam_bias_correction=False,
        adam_weight_decay=None,
        use_adamw_for_1d=True,
        # Celo2 model config
        orthogonalize=True,
        bias_correction=False,
        # Optional gradient clipping
        clip_grad=False,
        clip_norm=1.0,
    )
)

