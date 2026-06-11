gradient_estimator_args = dict(
    class_="ESSingle_ELO",
    kwargs=dict(
        # ES-Single base
        std=0.01,
        steps_per_jit=5,
        stack_antithetic_samples=False,
        sign_delta_loss_scalar=None,
        trunc_length=50,
        trunc_schedule=None,
        pmap_across_devices=False,
        use_bc_grads=False,
        std_schedule=None,
        loss_type="mean",
        final_loss_weight=0.0,
        # ELO additions (mirror pes_elo.py)
        expert_wd_sp=1000,
        expert_wd_ep=4500,
        expert_traj_wmin=0.0,
        bc_type="elo",
        use_baseline_losses=False,
        bc_grad_weight=None,
        expert_dirloss_weight=0.7,
        expert_magloss_weight=0.3,
        expert_dirloss_wmin=0.0,
        expert_magloss_wmin=0.0,
        delta_loss_scalar_afsnm=0.01,
    )
)
