_base_ = ["../meta_train_base.py"]

schedule = dict(
    init_value=0,
    peak_value=1.5e-3,
    end_value=1.5e-4,
    warmup_steps=50,
    decay_steps=19950,
    exponent=1.0,
    clip_before_optim=5.0,clip_after_optim=1.0,
)
num_outer_steps = 20000
num_inner_steps = 1000