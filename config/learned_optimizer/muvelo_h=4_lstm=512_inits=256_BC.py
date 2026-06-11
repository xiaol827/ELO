


learned_optimizer_args = dict(
    class_="MuHyperV2BC",
    kwargs=dict(
      interp_mult=1.0,
      lstm_hidden_size=512,
      ff_hidden_size=4,
      ff_hidden_layers=2,
      initial_momentum_decays=(0.9, 0.99, 0.999),
      initial_rms_decays=(0.999,),
      initial_adafactor_decays=(0.9, 0.99, 0.999),
      param_inits=256,
      mix_layers=True,
      exp_mult=0.001,
      step_mult=0.01,
      validation_mode=False,
      with_validation_feature_dim=False,

      # ablation flags.
      identity_lstm_controls=False,
      identity_lr_mult=False,

      # ablation flags.
      with_g=True,
      with_m=True,
      with_m_feat=True,
      with_rms=True,
      with_rms_feat=True,
      with_rms_norm_g=True,
      with_rsqrt_rms=True,
      with_p=True,
      with_fac_norm_g=True,
      with_fac_rms=True,
      with_fac_rsqrt=True,
      with_grad_clip_feat=True,
      with_fac_mom_mult=True,
      with_rms_only_norm_g=True,
      adafactor_accumulator=True,

      param_scale_mult=False,

      use_bugged_next_lstm_state=False,
      use_bugged_loss_features=False,
      precondition_output=False,
      reparam_decay=10.,
      rnn_state_decay=0.0,

      train=True,
      teacher_force=False,

      # more summaries
      summarize_each_layer=False,
      summarize_all_control=False,

      # Modify the lopt to probe behavior
      constant_loss=False,
      clip_param_scale_amount=None,

      bc_optimizer_args=[

dict(
use_mup=True,
schedule = dict(class_="constant_schedule",kwargs=dict(value=0.10771)),
optimizer_args = dict(
    class_="adam",
    kwargs=dict(
       learning_rate=0.044173,
       b1=0.85, 
       b2=0.999, 
       eps=1e-8, 
       eps_root=0.0, 
       mu_dtype=None, 
       nesterov=False
    )),
gradient_transform_before_optim = [],
gradient_transform_after_optim = [],),

dict(
use_mup=True,
schedule = dict(class_="constant_schedule",kwargs=dict(value=0.10771)),
optimizer_args = dict(
    class_="adam",
    kwargs=dict(
       learning_rate=0.044173,
       b1=0.85, 
       b2=0.999, 
       eps=1e-8, 
       eps_root=0.0, 
       mu_dtype=None, 
       nesterov=False
    )),
gradient_transform_before_optim = [],
gradient_transform_after_optim = [],),

dict(
use_mup=True,
schedule = dict(class_="constant_schedule",kwargs=dict(value=0.10771)),
optimizer_args = dict(
    class_="adam",
    kwargs=dict(
       learning_rate=0.044173,
       b1=0.85, 
       b2=0.999, 
       eps=1e-8, 
       eps_root=0.0, 
       mu_dtype=None, 
       nesterov=False
    )),
gradient_transform_before_optim = [],
gradient_transform_after_optim = [],),
#mumlp        
dict(
use_mup=True,
schedule = dict(class_="constant_schedule",kwargs=dict(value=0.044173)),
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
gradient_transform_before_optim = [],
gradient_transform_after_optim = [],),
dict(
use_mup=True,
schedule = dict(class_="constant_schedule",kwargs=dict(value=0.044173)),
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
gradient_transform_before_optim = [],
gradient_transform_after_optim = [],),
dict(
use_mup=True,
schedule = dict(class_="constant_schedule",kwargs=dict(value=0.044173)),
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
gradient_transform_before_optim = [],
gradient_transform_after_optim = [],),
dict(
use_mup=True,
schedule = dict(class_="constant_schedule",kwargs=dict(value=0.044173)),
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
gradient_transform_before_optim = [],
gradient_transform_after_optim = [],),
dict(
use_mup=True,
schedule = dict(class_="constant_schedule",kwargs=dict(value=0.044173)),
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
gradient_transform_before_optim = [],
gradient_transform_after_optim = [],),

],

      
      
      
      ))