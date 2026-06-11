truncated_step_args = dict(
    class_="VectorizedFedLOptTruncatedStep",
    kwargs=dict(
        meta_loss_split=None,
        outer_data_split="train",
        meta_loss_with_aux_key=None,
        random_initial_iteration_offset=1000,
        local_optimizer="deprecated",
        local_learning_rate=0.5,
        num_local_steps=4,
        keep_batch_in_gpu_memory=False,
        use_bc_grads=False,
    )
)