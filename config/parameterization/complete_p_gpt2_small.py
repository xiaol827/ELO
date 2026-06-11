from parameterization import TensorType  # Uncomment to use per-tensor multipliers

# Targeted at transformer-dense-w768-d12-h12 (gpt2-small) with use_mup=False.
# per_tensor_init_multipliers are chosen so that init_params_pytree produces
# sigma ~= flax-default sigma / 4 per tensor type. fan_in assumed: 768 for
# attention QKVO / mlp_up / mlp_gate / unembedding, 2112 (=768*2.75) for mlp_down.
parameterization_args = dict(
    # Base model configuration (for hyperparameter transfer)
    base_width=32,
    base_batch_size=64 * 1 * 64,
    base_dataset_size=64 * 1 * 64 * 10000,
    alpha=1.0,

    # Depth-wise multipliers (length must match base_depth)
    # These are linearly interpolated when current_depth != base_depth
    # Applied uniformly to all hyperparameters (lr, eps, wd, init, beta1, beta2)
    base_depth=1.0,
    depth_multipliers=[1.0],

    # Per-tensor-type hyperparameter multipliers
    # These multiply the CompletedP scaling factors for specific tensor types
    # Example: {TensorType.ATTENTION_QUERY: 0.5} would halve LR for Q projections
    # Available TensorTypes:
    #   EMBEDDING, POS_EMBEDDING,
    #   ATTENTION_QUERY, ATTENTION_KEY, ATTENTION_VALUE, ATTENTION_OUTPUT,
    #   ATTENTION_QUERY_NORM, ATTENTION_KEY_NORM,
    #   MLP_UP, MLP_GATE, MLP_DOWN,
    #   POST_ATTENTION_NORM, POST_MLP_NORM, OUTPUT_NORM,
    #   UNEMBEDDING,
    #   MOE_GATE, MOE_UP, MOE_GATE_PROJ, MOE_DOWN
    per_tensor_lr_multipliers = {
        # attention
        "attention_query": 1.0,
        "attention_key": 1.0,
        "attention_value": 1.0,
        "attention_output": 1.0,

        "attention_query_norm": 1.0,
        "attention_key_norm": 1.0,

        # mlp
        "mlp_up": 1.0,
        "mlp_gate": 1.0,
        "mlp_down": 1.0,

        # norm
        "post_attention_norm": 1.0,
        "post_mlp_norm": 1.0,

        # input
        "embedding": 1.0,
        # output
        "output_norm": 1.0,
        "unembedding": 1.0,
    },                                      # Dict[str, float]
    per_tensor_eps_multipliers={
        # attention
        "attention_query": 1.0,
        "attention_key": 1.0,
        "attention_value": 1.0,
        "attention_output": 1.0,

        "attention_query_norm": 1.0,
        "attention_key_norm": 1.0,

        # mlp
        "mlp_up": 1.0,
        "mlp_gate": 1.0,
        "mlp_down": 1.0,

        # norm
        "post_attention_norm": 1.0,
        "post_mlp_norm": 1.0,

        # input
        "embedding": 1.0,
        # output
        "output_norm": 1.0,
        "unembedding": 1.0,
    },
    per_tensor_wd_multipliers={
        # attention
        "attention_query": 1.0,
        "attention_key": 1.0,
        "attention_value": 1.0,
        "attention_output": 1.0,

        "attention_query_norm": 1.0,
        "attention_key_norm": 1.0,

        # mlp
        "mlp_up": 1.0,
        "mlp_gate": 1.0,
        "mlp_down": 1.0,

        # norm
        "post_attention_norm": 1.0,
        "post_mlp_norm": 1.0,

        # input
        "embedding": 1.0,
        # output
        "output_norm": 1.0,
        "unembedding": 1.0,
    },
    per_tensor_init_multipliers={
        # attention
        "attention_query": 1.0,
        "attention_key": 1.0,
        "attention_value": 1.0,
        "attention_output": 1.0,

        "attention_query_norm": 1.0,
        "attention_key_norm": 1.0,

        # mlp
        "mlp_up": 1.0,
        "mlp_gate": 1.0,
        "mlp_down": 1.0,

        # norm
        "post_attention_norm": 1.0,
        "post_mlp_norm": 1.0,

        # input
        "embedding": 1.0,
        # output
        "output_norm": 1.0,
        "unembedding": 1.0,
    },
    per_tensor_beta1_multipliers={
        # attention
        "attention_query": 1.0,
        "attention_key": 1.0,
        "attention_value": 1.0,
        "attention_output": 1.0,

        "attention_query_norm": 1.0,
        "attention_key_norm": 1.0,

        # mlp
        "mlp_up": 1.0,
        "mlp_gate": 1.0,
        "mlp_down": 1.0,

        # norm
        "post_attention_norm": 1.0,
        "post_mlp_norm": 1.0,

        # input
        "embedding": 1.0,
        # output
        "output_norm": 1.0,
        "unembedding": 1.0,
    },
    per_tensor_beta2_multipliers={
        # attention
        "attention_query": 1.0,
        "attention_key": 1.0,
        "attention_value": 1.0,
        "attention_output": 1.0,

        "attention_query_norm": 1.0,
        "attention_key_norm": 1.0,

        # mlp
        "mlp_up": 1.0,
        "mlp_gate": 1.0,
        "mlp_down": 1.0,

        # norm
        "post_attention_norm": 1.0,
        "post_mlp_norm": 1.0,

        # input
        "embedding": 1.0,
        # output
        "output_norm": 1.0,
        "unembedding": 1.0,
    }
    
    # Whether to re-initialize model parameters using CompletedP init scaling
    # reinit_params=False,
)



