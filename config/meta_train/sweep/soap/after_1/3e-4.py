_base_ = ["../../../meta_train_base.py","../../sweep_base.py", ]
schedule = dict(        
        init_value=3e-10,
        peak_value=3e-4,
        warmup_steps=100,
        decay_steps=900,
        end_value=3e-4,
        exponent=1.0,
        #soap
        b1 = 0.95,
        b2 = 0.99,
        shampoo_beta = -1,
        eps = 1e-8,
        weight_decay = 0.0001, 
        precondition_frequency= 1,
        max_precond_dim = 10000,
        #clipping
        clip_before_optim=5000,
        clip_after_optim=1.0,
        use_clamp_clip=False,
)

num_outer_steps = 1000
num_inner_steps = 1000
