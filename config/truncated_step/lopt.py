truncated_step_args = dict(
    class_="VectorizedLOptTruncatedStep",
    kwargs=dict(
        meta_loss_split=None,
        outer_data_split="train",
        meta_loss_with_aux_key=None,
        random_initial_iteration_offset=1000,
        random_initial_iteration_offset_linspace=False,
        global_num_particles=8,
        use_bc_grads=False,
    )
)