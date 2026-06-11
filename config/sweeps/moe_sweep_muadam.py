_base_ = ["./sweeps_base.py"]


num_inner_steps = 5000

sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        learning_rate=dict(
            values=[0.5       , 0.35984284, 0.25897373, 0.18637969, 0.13413479,
       0.09653489, 0.06947477, 0.05      ]
        ),
        benchmark_b1=dict(
            values=[0.85      , 0.87464278, 0.9       ]
        ),
        benchmark_b2=dict(
            values=[0.95      , 0.97419197, 0.999     ]
        ),
    ),
)

