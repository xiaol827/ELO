gradient_estimator_args = dict(
    class_="TruncatedPES_CHEN",
    kwargs=dict(
        std=0.01,
        steps_per_jit=5,
        stack_antithetic_samples=False,
        sign_delta_loss_scalar=None,
        trunc_length=50,
        trunc_schedule=None,
        pmap_across_devices=False,
        use_baseline_losses=False,
        bc_grad_weight=None,
        delta_loss_scalar_afsnm=0.01,
    )
)
