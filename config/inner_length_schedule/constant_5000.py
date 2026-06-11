inner_problem_length_schedule = dict(
    min=dict(
    class_="constant_schedule",
    kwargs=dict(
        value=5000
    )),
    max=dict(
        class_="constant_schedule",
        kwargs=dict(
            value=5000
        )
    ),
    sample_choice='log_uniform'
)