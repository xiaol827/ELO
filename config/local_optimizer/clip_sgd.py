
local_optimizer_args = dict(

use_mup=False,

schedule = dict(
    class_="constant_schedule",
    kwargs=dict(
        value=0.1
    )
),

optimizer_args = dict(
    class_="sgd",
    kwargs=dict(learning_rate=0.1, 
                momentum=0.9, 
                nesterov=False, 
                accumulator_dtype= None
    )),


gradient_transform_before_optim = [
    dict(class_="clip_by_global_norm",
        kwargs=dict(max_norm=5.0)),
    dict(class_="add_decayed_weights",
        kwargs=dict(weight_decay=0.0001))
],

gradient_transform_after_optim = [],

)
