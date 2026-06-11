_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        local_optimizer_args__schedule__kwargs=dict(
            values=[{'init_value': 0.0, 'peak_value': 6e-05, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 6e-06, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 6e-05, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 3e-06, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 8.14325285e-05, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 8.143252850000001e-06, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 8.14325285e-05, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 4.0716264250000006e-06, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.000110520945, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 1.10520945e-05, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.000110520945, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 5.52604725e-06, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.00015, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 1.4999999999999999e-05, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.00015, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 7.499999999999999e-06, 'exponent': 1.0} ,
                ]
),
        
    ),
)

