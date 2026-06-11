gradient_estimator_args = dict(
    class_="FullES",
    kwargs=dict(
        std=0.01,
        steps_per_jit=5,
        stack_antithetic_samples=False,
        sign_delta_loss_scalar=None,
        pmap_across_devices=False,
        loss_type="avg",
        use_baseline_losses=False,
        bc_grad_weight=None,
    )
)