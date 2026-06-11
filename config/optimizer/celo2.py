optimizer_args = dict(
    class_="celo2",
    kwargs=dict(
        learning_rate=0.001,
        weight_decay=0.0,
        checkpoint_path="/scratch/therien/l2o_install/celo2_checkpoint/theta.state",
        orthogonalize=True,
    )
)
