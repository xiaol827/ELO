optimizer_args = dict(
    class_="celo2",
    kwargs=dict(
        learning_rate=0.001,
        weight_decay=0.0,
        checkpoint_path="<PATH_TO_CELO2_CHECKPOINT>/theta.state",
        orthogonalize=True,
    )
)
