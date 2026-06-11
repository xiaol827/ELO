

learned_optimizer_args = dict(
    class_="MuAdafacMLPLOptBC",  
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
      zero_training_step_feature=False,
      teacher_force=False,
      train=True,
      bc_optimizer_args= dict(

use_mup=False,

schedule = dict(
    class_="constant_schedule",
    kwargs=dict(
        value=3e-4
    )
),

optimizer_args = dict(
    class_="muon",
    kwargs=dict(
        learning_rate=0.02,
        ns_coeffs=(3.4445, -4.775, 2.0315),
        ns_steps=5,
        beta=0.95,
        eps=1e-8,
        weight_decay=0.0001,
        weight_decay_mask=None,
        mu_dtype=None,
        nesterov=True,
        adaptive=False,
        adam_b1=0.9,
        adam_b2=0.99,
        adam_eps_root=0.0,
        adam_weight_decay=0.0001
    )),


gradient_transform_before_optim = [

],

gradient_transform_after_optim = [

],

)
      
      ))

      