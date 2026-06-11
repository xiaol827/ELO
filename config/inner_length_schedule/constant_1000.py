inner_problem_length_schedule = dict(
    min=dict(
    class_="constant_schedule",
    kwargs=dict(
        value=1000
    )),
    max=dict(
        class_="constant_schedule",
        kwargs=dict(
            value=1000
        )
    ),
    sample_choice='log_uniform'
)