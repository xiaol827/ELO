_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        local_optimizer_args__schedule__kwargs=dict(
            values=[
{'init_value': 0.0, 'peak_value': 0.001, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0001, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.001584893192461114, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00015848931924611142, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.0025118864315095794, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00025118864315095795, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.003981071705534973, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00039810717055349735, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.00630957344480193, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0006309573444801931, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.01, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.001, 'exponent': 1.0} ,
                ]
),

        outer_optimizer_args__schedule__kwargs__value=dict(
            values=[
                # 0.2,
                # 0.4,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                # 1.2,
                # 1.4,
            ]
        ),
      
    ),
)

