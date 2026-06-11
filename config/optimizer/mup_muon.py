optimizer_args = dict(
    class_="mup_muon",
    kwargs=dict(
        learning_rate=0.02,
        ns_coeffs=(3.4445, -4.775, 2.0315),
        ns_steps=5,
        beta=0.9,
        eps=1e-8,
        weight_decay=0.01,
        nesterov=True,
        adaptive=False,
        mu_dtype=None,
        # Adam defaults from optimal mup_adamw sweep at w128-d4:
        adam_b1=0.95484,
        adam_b2=0.9908,
        adam_eps=1e-8,
        adam_eps_root=0.0,
        adam_weight_decay=0.093198,
        weight_decay_mask=None,
    )
)
