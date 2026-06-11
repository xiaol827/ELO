_base_ = ["../meta_train_base.py"]

schedule = dict(
        init_value=3e-10,
        peak_value=3e-3,
        end_value=1e-5,
        warmup_steps=100,
        decay_steps=4900,
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
)

num_outer_steps = 5000
num_inner_steps = 1000

expert_wd_sp = 1000
expert_wd_ep = 4500
expert_traj_wmin = 0.0
expert_dirloss_wmin = 0.0
expert_magloss_wmin = 0.0

buffer_cfg = {'thred': 0.1, 'min_thred': 0.1, 'update_idx': 0, 'buffer_size' : 2}

steps = [100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500]

curriculum_lengths = [x for x in steps for _ in range(20)]