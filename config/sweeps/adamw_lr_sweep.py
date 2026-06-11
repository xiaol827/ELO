_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        schedule__kwargs=dict(
            values=[
                {'init_value': 0.0, 'peak_value': 1.00e-2, 'end_value': 1.00e-3, 'warmup_steps': 954, 'decay_steps': 19074, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 3.16e-3, 'end_value': 3.16e-4, 'warmup_steps': 954, 'decay_steps': 19074, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 1.00e-3, 'end_value': 1.00e-4, 'warmup_steps': 954, 'decay_steps': 19074, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 3.16e-4, 'end_value': 3.16e-5, 'warmup_steps': 954, 'decay_steps': 19074, 'exponent': 1.0},
                {'init_value': 0.0, 'peak_value': 1.00e-4, 'end_value': 1.00e-5, 'warmup_steps': 954, 'decay_steps': 19074, 'exponent': 1.0},
            ]
        ),
    ),
)
