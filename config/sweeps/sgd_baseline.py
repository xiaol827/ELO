_base_ = ["./sweeps_base.py"]

optimizer = "sgd"
task = "mlp128x128_fmnist_32"

num_grads = 1
num_local_steps = 1

sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        learning_rate=dict(
            values=[
                1,
                0.5,
                0.1,
                0.05,
                0.01,
                0.005,
                0.001,
                0.0005,
            ]
        ),
        num_inner_steps=dict(
            values=[
                32000,
                64000,
                128000
            ]
        ),
    ),
)