_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        local_optimizer_args__schedule__kwargs__value=dict(
            values=[ 1.34596032e-04,2.10174801e-04, 3.28192787e-04, 5.12480588e-04, 8.00250228e-04,
                    1.24960914e-03, 1.95129342e-03, 3.04698957e-03, 4.75794431e-03,
                    7.42963951e-03, 1.16015530e-02, 1.81160919e-02, 2.82886943e-02,
                    4.41734470e-02, 6.89778538e-02, 1.07710506e-01, 1.68192432e-01,
                    2.62636353e-01, 4.10112707e-01, ]
        ),
        outer_optimizer_args__schedule__kwargs__value=dict(
            values=[
                0.4,
                0.6,
                0.8,
                1.0,
                1.2,
                1.4,
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
