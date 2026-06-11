gradient_estimator_args = dict(
    class_="TruncatedPES",
    kwargs=dict(
        std=0.01,
        steps_per_jit=5,
        stack_antithetic_samples=False,
        sign_delta_loss_scalar=None,
        trunc_length=50,
        pmap_across_devices=False,
        use_bc_grads=False,
        use_baseline_losses=False,
        bc_grad_weight=None,
    )
)