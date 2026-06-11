inner_problem_length_schedule = dict(
    min=dict(
        class_="constant_schedule",
        kwargs=dict(
            value=100
        )),
    max=dict(
        class_="constant_schedule",
        kwargs=dict(
            value=5000
        )),
    sample_choice='chen_curriculum_constant',
    init_length=100,
    increment=200,
    N_period=20,
    max_length=5000,
)
