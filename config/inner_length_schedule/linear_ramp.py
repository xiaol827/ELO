inner_problem_length_schedule = dict(
    min=dict(
        class_="linear_schedule",
        kwargs=dict(
            init_value=100, end_value=5000, transition_steps=5000, transition_begin=0
    )),
    max=dict(
        class_="linear_schedule",
        kwargs=dict(
            init_value=100, end_value=5000, transition_steps=5000, transition_begin=0
        )
    ),
    sample_choice='log_uniform'
)