inner_problem_length_schedule = dict(
    min=dict(
    class_="cosine_decay_schedule",
    kwargs=dict(
        init_value=100, decay_steps=5000, alpha=50
    )),
    max=dict(
        class_="cosine_decay_schedule",
        kwargs=dict(
           init_value=100, decay_steps=5000, alpha=50
        )
    ),
    sample_choice='log_uniform'
)