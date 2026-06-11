

learned_optimizer_args = dict(
    class_="FedMLPLOpt",
    kwargs=dict(
        exp_mult=0.001,
        step_mult=0.001,
        hidden_size=4,
        hidden_layers=2,
        compute_summary=True,
        num_grads=4,
        with_all_grads=True,
        with_avg=False,
      ))


      