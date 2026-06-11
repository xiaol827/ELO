_base_ = ["../meta_train_base.py"]
schedule = dict(
        init_value=3e-10,
        peak_value=1e-3,
        end_value=1e-4,
        warmup_steps=100,
        decay_steps=9900,
        exponent=1.0,
        #adamw
        b1 = 0.9,
        b2 = 0.999,
        eps = 1e-08,
        eps_root = 0.0,
        weight_decay = 0.0001, 
        #clipping
        clip_before_optim=3.0,
        clip_after_optim=1000.0,
        use_clamp_clip=True,
        #muon
        # muon_ns_coeffs = (3.4445, -4.7750, 2.0315),
        # muon_ns_steps = 5,
        # muon_beta = 0.95,
        # muon_eps = 1e-8,
        # muon_mu_dtype = None,
        # muon_nesterov = True,
        # muon_adaptive = False,
        # muon_adam_b1 = 0.9,
        # muon_adam_b2 = 0.999,
        # muon_adam_eps_root = 0.0,
        # muon_adam_weight_decay = 0.0,
)

num_outer_steps = 10000
num_inner_steps = 5000
