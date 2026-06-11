

learned_optimizer_args = dict(
    class_="AdafacMLPLOpt",
    kwargs=dict(
      exp_mult=0.001,
      step_mult=0.001,
      init_lr=0.0,
      warmup_fraction=0.05,
      hidden_size=32,
      hidden_layers=2,
      initial_momentum_decays=(0.9, 0.99, 0.999),
      initial_rms_decays=(0.999,),
      initial_adafactor_decays=(0.9, 0.99, 0.999),
      concat_weights=True,
      make_separate_weights=False,
      split_weights=False,
      use_lo_cosine_scheduler=False,
      step_mult_min=1e-4,
      ))


      