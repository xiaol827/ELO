# CELO2 paper: log-uniform unroll between 100 and 2000 steps
inner_problem_length_schedule = dict(
    min=dict(
        class_="constant_schedule",
        kwargs=dict(
            value=100
        )),
    max=dict(
        class_="constant_schedule",
        kwargs=dict(
            value=2000
        )
    )
)
