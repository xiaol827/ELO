
learned_optimizer_args = dict(
    class_="MuAdafacMLPLOpt",
    kwargs=dict(
      exp_mult=0.001,
      step_mult=0.01,
      hidden_size=32,
      hidden_layers=2,
      initial_momentum_decays=(0.9, 0.99, 0.999),
      initial_rms_decays=(0.999,),
      initial_adafactor_decays=(0.9, 0.99, 0.999),
      concat_weights=True,
      make_separate_weights=False,
      split_weights=False,
      clip_grad=False,
      clip_norm=1.0,
      zero_training_step_feature=False))


      