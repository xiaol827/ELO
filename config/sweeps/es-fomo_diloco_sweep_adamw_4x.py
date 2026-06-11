_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        local_optimizer_args__schedule__kwargs=dict(
            values=[
                {'init_value': 0.0, 'peak_value': 0.0001, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 1e-05, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 0.00015848931924611142, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 1.584893192461114e-05, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 0.00025118864315095795, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 2.5118864315095798e-05, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 0.00039810717055349735, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 3.9810717055349735e-05, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 0.000630957344480193, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 6.30957344480193e-05, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 0.001, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0001, 'exponent': 1.0} ,
                ]
),
       
        
    ),
)

