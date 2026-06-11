schedule = dict(
    class_="warmup_cosine_decay_schedule",
    kwargs=dict(
        init_value=0.0, 
        peak_value=3e-4, 
        warmup_steps=100, 
        decay_steps=4900, 
        end_value=3e-5, 
        exponent=1.0
    )
)