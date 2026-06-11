_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        schedule__kwargs__value=dict(
            values=[ 1.34596032e-04,2.10174801e-04, 3.28192787e-04, 5.12480588e-04, 8.00250228e-04,
                    1.24960914e-03, 1.95129342e-03, 3.04698957e-03, 4.75794431e-03,
                    7.42963951e-03, 1.16015530e-02, 1.81160919e-02, 2.82886943e-02,
                    4.41734470e-02, 6.89778538e-02, 1.07710506e-01, 1.68192432e-01,
                    2.62636353e-01, 4.10112707e-01, ]
        ),
    ),
)
