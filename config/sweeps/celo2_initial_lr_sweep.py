_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        learned_optimizer_args__kwargs__initial_lr=dict(
            values=[1.0, 0.1, 0.01, 0.001, 0.0001, 1e-5]
        ),
    ),
)
