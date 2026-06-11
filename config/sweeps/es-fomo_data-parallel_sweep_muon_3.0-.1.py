_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        local_optimizer_args__schedule__kwargs=dict(
            values=[
                
                
{'init_value': 0.0, 'peak_value': 0.001, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0001, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.001, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 5e-05, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.0013895, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00013895, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.0013895, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 6.9475e-05, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.0019307, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00019307000000000002, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.0019307, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 9.653500000000001e-05, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.0026827, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00026827000000000003, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.0026827, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00013413500000000002, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.00372759, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.000372759, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.00372759, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0001863795, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.00517947, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.000517947, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.00517947, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0002589735, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.00719686, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0007196860000000001, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.00719686, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00035984300000000004, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.01, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.001, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.01, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0005, 'exponent': 1.0} ,
                ]
),
        
    ),
)





