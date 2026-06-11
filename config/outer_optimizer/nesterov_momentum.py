
outer_optimizer_args = dict(

use_mup=False,

schedule = dict(
    class_="constant_schedule",
    kwargs=dict(
        value=0.8
    )
),

optimizer_args = dict(
    class_="sgd",
    kwargs=dict(learning_rate=0.1, 
                momentum=0.9, 
                nesterov=True, 
                accumulator_dtype=None
    )),


gradient_transform_before_optim = [
],

gradient_transform_after_optim = [],

)
