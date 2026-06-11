
learned_optimizer_args = dict(
    class_="MuAdafacMLPLOptv2",
    kwargs=dict(
      exp_mult=0.001,
      step_mult=0.01,
      hidden_size=32,
      hidden_layers=2,
      expansion_factor=4,
      initial_momentum_decays=(0.55, 0.9, 0.99),
      initial_rms_decays=(0.95,),
      initial_adafactor_decays=(0.35, 0.85, 0.9995),
      concat_weights=True,
      make_separate_weights=False,
      split_weights=False,
      clip_grad=False,
      clip_norm=1.0,
      mup_lrs=None,
      dyn_tanh=False,
      zero_training_step_feature=False,
      num_params_normalizer=128,
      quantized=None))


      