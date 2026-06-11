_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        schedule__kwargs=dict(
            values=[
                # {'init_value': 0.0, 'peak_value': 3e-2, 'end_value': 3e-3, 'warmup_steps': 100, 'decay_steps': 10000, 'exponent': 1.0},
                # {'init_value': 0.0, 'peak_value': 1e-2, 'end_value': 1e-3, 'warmup_steps': 100, 'decay_steps': 10000, 'exponent': 1.0},
                # {'init_value': 0.0, 'peak_value': 3e-3, 'end_value': 3e-4, 'warmup_steps': 100, 'decay_steps': 10000, 'exponent': 1.0},
                # {'init_value': 0.0, 'peak_value': 1e-3, 'end_value': 1e-4, 'warmup_steps': 100, 'decay_steps': 10000, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 1e-1, 'end_value': 1e-2, 'warmup_steps': 100, 'decay_steps': 10000, 'exponent': 1.0},
                # {'init_value': 0.0, 'peak_value': 3e-4, 'end_value': 3e-5, 'warmup_steps': 100, 'decay_steps': 10000, 'exponent': 1.0},
                # {'init_value': 0.0, 'peak_value': 1e-4, 'end_value': 1e-5, 'warmup_steps': 100, 'decay_steps': 10000, 'exponent': 1.0},
            ]
        ),
    ),
)
