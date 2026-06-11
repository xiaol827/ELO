_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        schedule__kwargs=dict(
            values=[
                {'init_value': 0.0, 'peak_value': 5e-2, 'end_value': 5e-4, 'warmup_steps': 0, 'decay_steps': 10000, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 3e-2, 'end_value': 3e-4, 'warmup_steps': 0, 'decay_steps': 10000, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 2e-2, 'end_value': 2e-4, 'warmup_steps': 0, 'decay_steps': 10000, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 1e-2, 'end_value': 1e-4, 'warmup_steps': 0, 'decay_steps': 10000, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 8e-3, 'end_value': 8e-5, 'warmup_steps': 0, 'decay_steps': 10000, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 6e-3, 'end_value': 6e-5, 'warmup_steps': 0, 'decay_steps': 10000, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 4e-3, 'end_value': 4e-5, 'warmup_steps': 0, 'decay_steps': 10000, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 2e-3, 'end_value': 2e-5, 'warmup_steps': 0, 'decay_steps': 10000, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 1e-3, 'end_value': 1e-5, 'warmup_steps': 0, 'decay_steps': 10000, 'exponent': 1.0},
                # {'init_value': 0.0, 'peak_value': 3e-4, 'end_value': 3e-5, 'warmup_steps': 0, 'decay_steps': 10000, 'exponent': 1.0},
                # {'init_value': 0.0, 'peak_value': 1e-4, 'end_value': 1e-5, 'warmup_steps': 0, 'decay_steps': 10000, 'exponent': 1.0},
            ]
        ),
    ),
)
