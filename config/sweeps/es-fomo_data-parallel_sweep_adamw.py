_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        schedule__kwargs=dict(
            values=[
                {'init_value': 0.0, 'peak_value': 0.0074, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0007400000000000001, 'exponent': 1.0} ,
                # {'init_value': 0.0, 'peak_value': 0.0074, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00037000000000000005, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 0.00383281126, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.000383281126, 'exponent': 1.0} ,
                # {'init_value': 0.0, 'peak_value': 0.00383281126, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.000191640563, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 0.00198519489, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.000198519489, 'exponent': 1.0} ,
                # {'init_value': 0.0, 'peak_value': 0.00198519489, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 9.92597445e-05, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 0.00102822667, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.00010282266700000002, 'exponent': 1.0} ,
                # {'init_value': 0.0, 'peak_value': 0.00102822667, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 5.141133350000001e-05, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 0.000532567398, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 5.32567398e-05, 'exponent': 1.0} ,
                # {'init_value': 0.0, 'peak_value': 0.000532567398, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 2.66283699e-05, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 0.000275841935, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 2.7584193499999998e-05, 'exponent': 1.0} ,
                # {'init_value': 0.0, 'peak_value': 0.000275841935, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 1.3792096749999999e-05, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 0.000142871632, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 1.4287163200000001e-05, 'exponent': 1.0} ,
                # {'init_value': 0.0, 'peak_value': 0.000142871632, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 7.143581600000001e-06, 'exponent': 1.0} ,
                {'init_value': 0.0, 'peak_value': 7.4e-05, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 7.4e-06, 'exponent': 1.0} ,
                # {'init_value': 0.0, 'peak_value': 7.4e-05, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 3.7e-06, 'exponent': 1.0} ,
                ]
),
        
    ),
)


{'init_value': 0.0, 'peak_value': 0.0074, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0007400000000000001, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.01028227, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.001028227, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.01428716, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.001428716, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.01985195, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.001985195, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.02758419, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0027584190000000002, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.03832811, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.003832811, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.05325674, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.005325674, 'exponent': 1.0} ,
{'init_value': 0.0, 'peak_value': 0.074, 'warmup_steps': 1000, 'decay_steps': 12732, 'end_value': 0.0074, 'exponent': 1.0} ,


