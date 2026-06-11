_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
    schedule__kwargs=dict(
            values=[ 
 {
  'decay_steps': 1900,
  'end_value': 0.05,
  'exponent': 1.0,
  'init_value': 0.0,
  'peak_value': 0.5,
  'warmup_steps': 100,
},
 {
  'decay_steps': 1900,
  'end_value': 0.025,
  'exponent': 1.0,
  'init_value': 0.0,
  'peak_value': 0.25,
  'warmup_steps': 100,
},
 {
  'decay_steps': 1900,
  'end_value': 0.0125,
  'exponent': 1.0,
  'init_value': 0.0,
  'peak_value': 0.125,
  'warmup_steps': 100,
},
 {
  'decay_steps': 1900,
  'end_value': 0.00625,
  'exponent': 1.0,
  'init_value': 0.0,
  'peak_value': 0.0625,
  'warmup_steps': 100,
},
 {
  'decay_steps': 1900,
  'end_value': 0.003125,
  'exponent': 1.0,
  'init_value': 0.0,
  'peak_value': 0.03125,
  'warmup_steps': 100,
},
 {
  'decay_steps': 1900,
  'end_value': 0.0015625,
  'exponent': 1.0,
  'init_value': 0.0,
  'peak_value': 0.015625,
  'warmup_steps': 100,
},
 {
  'decay_steps': 1900,
  'end_value': 0.00078125,
  'exponent': 1.0,
  'init_value': 0.0,
  'peak_value': 0.0078125,
  'warmup_steps': 100,
},
                ]
        ),
        optimizer_args__kwargs__b1=dict(
                values=[0.7, 0.8, 0.9, 0.95, 0.99]
            
        ),
        optimizer_args__kwargs__b2=dict(
                values=[0.7, 0.8, 0.9, 0.95, 0.99]
            
        ),
        optimizer_args__kwargs__weight_decay=dict(
                values=[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
            
        ),
        
    ),
)
