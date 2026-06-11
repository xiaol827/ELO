inner_problem_length_schedule = dict(
    min=dict(
    class_="constant_schedule",
    kwargs=dict(
        value=2000
    )),
    max=dict(
        class_="constant_schedule",
        kwargs=dict(
            value=2000
        )
    ),
    sample_choice='log_uniform'
)