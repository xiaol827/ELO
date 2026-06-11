_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        local_optimizer_args__schedule__kwargs=dict(
            values=[ {'init_value': 0.0, 'peak_value': 0.000134596032, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 6.7298016e-06, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.000210174801, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 1.0508740050000002e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.000328192787, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 1.640963935e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.000512480588, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 2.56240294e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.000800250228, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 4.001251140000001e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00124960914, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 6.2480457e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00195129342, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 9.7564671e-05, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00304698957, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.00015234947850000001, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00475794431, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.0002378972155, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.00742963951, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.0003714819755, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.011601553, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.00058007765, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.0181160919, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.000905804595, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.0282886943, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.0014144347150000002, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.044173447, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.00220867235, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.0689778538, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.00344889269, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.107710506, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.0053855253, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.168192432, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.0084096216, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.262636353, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.01313181765, 'exponent': 1.0} ,
                    {'init_value': 0.0, 'peak_value': 0.410112707, 'warmup_steps': 1000, 'decay_steps': 29000, 'end_value': 0.020505635350000002, 'exponent': 1.0} ,
                ]
        ),
        outer_optimizer_args__schedule__kwargs__value=dict(
            values=[
                0.2,
                0.4,
                0.6,
                # 0.7,
                0.8,
                # 0.9,
                1.0,
                # 1.2,
                # 1.4,
            ]
        ),
        local_optimizer_args__ec_beta=dict(
            values=[ 
                # 0.999, 
                # 0.99, 
                # 0.95, 
                0.9, 
                # 0.85, 
                # 0.8, 
                # 0.75, 
                0.7,
                # 0.6,
                0.5,
                0.3,
                0.0,
            ]
        ),
        
    ),
)


if __name__ == "__main__":
    import argparse
    from argparse import Namespace
    import pprint

    # Create a default args object with the values from muadam.py
    args = Namespace(
        local_optimizer_args=dict(
            use_mup=True,
            schedule=dict(
                class_="constant_schedule",
                kwargs=dict(
                    value=0.01,
                ),
            ),
            optimizer_args=dict(
                class_="adam",
                kwargs=dict(
                    learning_rate=0.01,
                    b1=0.9,
                    b2=0.99,
                    eps=1e-8,
                    eps_root=0.0,
                    mu_dtype=None,
                    nesterov=False
                )
            ),
            gradient_transform_before_optim=[],
            gradient_transform_after_optim=[],
        )
    )

    # Example of how to override args with values from sweep_config
    print("Original args:")
    pprint.pprint(args)
    
    # Simulate overriding with a specific configuration from the sweep
    override_config = {
        'local_optimizer_args__schedule__kwargs__value': 0.001,
        'local_optimizer_args__optimizer_args__kwargs__b1': 0.95,
        'local_optimizer_args__optimizer_args__kwargs__b2': 0.999
    }
    
    # Apply overrides
    for key, value in override_config.items():
        parts = key.split('__')
        target = args
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                target[part] = value
            else:
                parent = target
                target = target.__dict__.get(part) if i == 0 else target.get(part)
    
    print("\nArgs after override:")
    pprint.pprint(args)
