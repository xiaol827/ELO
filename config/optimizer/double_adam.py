optimizer_args = dict(
    class_="DoubleAdam",
    kwargs=dict(
        clip_norm=3.0,

        learning_rate=dict(
            class_="warmup_cosine_decay_schedule",
        kwargs=dict(
            init_value=0.0, 
            peak_value=3e-4, 
            warmup_steps=50, 
            decay_steps=4950, 
            end_value=3e-5, 
            exponent=1.0
        )
    ),


    # merging_rate=dict(
    #         class_="constant_schedule",
    #         kwargs=dict(
    #             value=0.5
    #         )
    #     ),
    
    
    merging_rate = dict(
        class_="linear_schedule",
        kwargs=dict(
            init_value=0.01,
            end_value=0.0,
            transition_steps=500,
            transition_begin=1500
        )),
       
    adam_bc=dict(
        class_="adamw",
        kwargs=dict(
                learning_rate=1.0,
                b1=0.9, 
                b2=0.999, 
                eps=1e-8, 
                eps_root=0.0, 
                mu_dtype=None,
                weight_decay=0.0001,
                nesterov=False
    )),
        
    adam_es=dict(
        class_="adamw",
        kwargs=dict(
                learning_rate=1.0,
                b1=0.9, 
                b2=0.999, 
                eps=1e-8, 
                eps_root=0.0, 
                mu_dtype=None,
                weight_decay=0.0001,
                nesterov=False
    )),
)
)