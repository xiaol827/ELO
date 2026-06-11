
learned_optimizer_args = dict(
    class_="MuAdafacMLPLOptV4BC",
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
      clip_norm=None,
      zero_training_step_feature=False,
      epsilon=1e-10,
      teacher_force=False,
      train=True,
      teacher_force_schedule = dict(
        class_="linear_schedule",
        kwargs=dict(
            init_value=1.0,
            end_value=0.0,
            transition_steps=500,
            transition_begin=40
        )),
      bc_optimizer_args= dict(

use_mup=True,

schedule = dict(
    class_="constant_schedule",
    kwargs=dict(
        value=0.044173
    )
),

optimizer_args = dict(
    class_="adam",
    kwargs=dict(
       learning_rate=0.044173,
       b1=0.9, 
       b2=0.999, 
       eps=1e-8, 
       eps_root=0.0, 
       mu_dtype=None, 
       nesterov=False
    )),


gradient_transform_before_optim = [

],

gradient_transform_after_optim = [

],

)
    )
)


      