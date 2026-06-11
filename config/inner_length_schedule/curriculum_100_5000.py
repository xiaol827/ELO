inner_problem_length_schedule = dict(
    min=dict(
    class_="constant_schedule",
    kwargs=dict(
        value=100
    )),
    max=dict(
        class_="constant_schedule",
        kwargs=dict(
            value=1000
        )
    ),
    sample_choice='log_uniform',
    curriculum_lengths = [x for x in [100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000] for _ in range(20)]
)

