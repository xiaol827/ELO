_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        local_optimizer_args__schedule__kwargs=dict(
            values=[{'init_value': 0.0, 'peak_value': 0.0074, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0007400000000000001, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.01028227, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.001028227, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.01428716, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.001428716, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.01985195, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.001985195, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.02758419, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0027584190000000002, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.03832811, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.003832811, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.05325674, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.005325674, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.074, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0074, 'exponent': 1.0} ,
                ]
),
        
    ),
)





