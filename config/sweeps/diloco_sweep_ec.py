_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        local_optimizer_args__ec_beta=dict(
            values=[ 0.999, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7 ]
        ),
    ),
)

