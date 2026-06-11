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
    )
)