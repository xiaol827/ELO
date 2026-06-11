
local_optimizer_args = dict(

use_mup=True,

schedule = dict(
    class_="warmup_cosine_decay_schedule",
    kwargs=dict(init_value=0.0, peak_value=3e-4, warmup_steps=100, decay_steps=4900, end_value=3e-5, exponent=1.0)
),

optimizer_args = dict(
    class_="adam",
    kwargs=dict(
       learning_rate=0.01,
       b1=0.9, 
       b2=0.99, 
       eps=1e-8, 
       eps_root=0.0, 
       mu_dtype=None, 
       nesterov=False
    )),


gradient_transform_before_optim = [

],

gradient_transform_after_optim = [

],

use_error_correction = True,
ec_beta = 0.9
)
