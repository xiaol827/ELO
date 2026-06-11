_base_ = ["../meta_train_base.py"]
schedule = dict(
    init_value=3e-10,
    peak_value=3e-3,
    end_value=3e-4,
    warmup_steps=5,
    decay_steps=4950,
    exponent=1.0,
    clip_before_optim=5.0,clip_after_optim=1.0,
)



num_outer_steps = 5000
num_inner_steps = 1000