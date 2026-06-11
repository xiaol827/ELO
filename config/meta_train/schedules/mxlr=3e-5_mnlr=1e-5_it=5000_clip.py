_base_ = ["../meta_train_base.py"]
schedule = dict(
    init_value=3e-10,
    peak_value=3e-5,
    end_value=1e-5,
    warmup_steps=100,
    decay_steps=4900,
    exponent=1.0,
    clip_before_optim=1.0,
    clip_after_optim=10000000,
)

num_outer_steps = 5000
num_inner_steps = 1000