_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        schedule__kwargs=dict(
            values=[
{'init_value': 0.0, 'peak_value': 0.01005924, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.001005924, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.01367409, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.001367409, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.01858796, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0018587960000000002, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.02526766, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.002526766, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.03434776, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.003434776, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.04669084, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0046690839999999996, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.0634695, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0063469500000000005, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.08627767, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.008627767, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.1172821, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.011728210000000001, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.15942817, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.015942817, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.2167197, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.02167197, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.29459931, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.029459931, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.40046545, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.040046545, 'exponent': 1.0} ,
# {'init_value': 0.0, 'peak_value': 0.54437527, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.054437527, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.74, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.074, 'exponent': 1.0} ,
                ]
),
        
    ),
)

