_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        local_optimizer_args__schedule__kwargs=dict(
            values=[{'init_value': 0.0, 'peak_value': 0.0003, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 2.9999999999999997e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.0003, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 1.4999999999999999e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00041685, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 4.1685000000000005e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00041685, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 2.0842500000000002e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00057921, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 5.7921e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00057921, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 2.89605e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00080481, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 8.0481e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00080481, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 4.02405e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00111828, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.000111828, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00111828, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 5.5914e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00155384, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.000155384, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00155384, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 7.7692e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00215906, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.000215906, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00215906, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.000107953, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.003, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00030000000000000003, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.003, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00015000000000000001, 'exponent': 1.0} ,
                ]
),
        
    ),
)

