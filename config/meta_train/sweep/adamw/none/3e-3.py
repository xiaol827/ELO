_base_ = ["../../../meta_train_base.py","../../sweep_base.py", ]
schedule = dict(        
        init_value=3e-10,
        peak_value=3e-3,
        warmup_steps=100,
        decay_steps=900,
        end_value=3e-3,
        exponent=1.0,
        #adamw
        b1 = 0.9,
        b2 = 0.999,
        eps = 1e-08,
        eps_root = 0.0,
        weight_decay = 0.0001, 
        #clipping
        clip_before_optim=1000.0,
        clip_after_optim=1000.0,
        use_clamp_clip=False,
)

num_outer_steps = 1000
num_inner_steps = 1000