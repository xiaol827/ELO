# Muon-mode variant of complete_p_bs100k_steps2000.py.
#
# Identical to the Adam version EXCEPT for the new ``parameterization_class``
# routing key, which causes ``mu_task_base._compute_completed_p_scales`` to
# instantiate ``MuonCompletedPParameterization`` and emit
# ``mup_muon_lr_scales`` / ``mup_muon_eps_scales`` / ``mup_muon_shape_scales``
# / ``mup_is_muon_mask`` alongside the standard Adam scales. Pair this with
# ``config/learned_optimizer/completedp_muon_lopt.py`` to drive the
# ``MuCompletedPMuonLOpt`` learned optimizer.

from parameterization import TensorType  # noqa: F401  (kept for parity with sibling config)

parameterization_args = dict(
    parameterization_class='muon_completedp',
    base_width=128,
    base_batch_size=64 * 16 * 128,
    base_dataset_size=64 * 16 * 128 * 4500,
    alpha=1.0,
    base_depth=4.0,
    depth_multipliers=[4.0, 4.0, 4.0, 4.0],
    per_tensor_lr_multipliers={
        "attention_query": 1.0, "attention_key": 1.0,
        "attention_value": 1.0, "attention_output": 1.0,
        "attention_query_norm": 1.0, "attention_key_norm": 1.0,
        "mlp_up": 1.0, "mlp_gate": 1.0, "mlp_down": 1.0,
        "post_attention_norm": 1.0, "post_mlp_norm": 1.0,
        "embedding": 1.0, "output_norm": 1.0, "unembedding": 1.0,
    },
    per_tensor_eps_multipliers={
        "attention_query": 1.0, "attention_key": 1.0,
        "attention_value": 1.0, "attention_output": 1.0,
        "attention_query_norm": 1.0, "attention_key_norm": 1.0,
        "mlp_up": 1.0, "mlp_gate": 1.0, "mlp_down": 1.0,
        "post_attention_norm": 1.0, "post_mlp_norm": 1.0,
        "embedding": 1.0, "output_norm": 1.0, "unembedding": 1.0,
    },
    per_tensor_wd_multipliers={
        "attention_query": 1.0, "attention_key": 1.0,
        "attention_value": 1.0, "attention_output": 1.0,
        "attention_query_norm": 1.0, "attention_key_norm": 1.0,
        "mlp_up": 1.0, "mlp_gate": 1.0, "mlp_down": 1.0,
        "post_attention_norm": 1.0, "post_mlp_norm": 1.0,
        "embedding": 1.0, "output_norm": 1.0, "unembedding": 1.0,
    },
    per_tensor_init_multipliers={
        "attention_query": 1.0, "attention_key": 1.0,
        "attention_value": 1.0, "attention_output": 1.0,
        "attention_query_norm": 1.0, "attention_key_norm": 1.0,
        "mlp_up": 1.0, "mlp_gate": 1.0, "mlp_down": 1.0,
        "post_attention_norm": 1.0, "post_mlp_norm": 1.0,
        "embedding": 1.0, "output_norm": 1.0, "unembedding": 1.0,
    },
    per_tensor_beta1_multipliers={
        "attention_query": 1.0, "attention_key": 1.0,
        "attention_value": 1.0, "attention_output": 1.0,
        "attention_query_norm": 1.0, "attention_key_norm": 1.0,
        "mlp_up": 1.0, "mlp_gate": 1.0, "mlp_down": 1.0,
        "post_attention_norm": 1.0, "post_mlp_norm": 1.0,
        "embedding": 1.0, "output_norm": 1.0, "unembedding": 1.0,
    },
    per_tensor_beta2_multipliers={
        "attention_query": 1.0, "attention_key": 1.0,
        "attention_value": 1.0, "attention_output": 1.0,
        "attention_query_norm": 1.0, "attention_key_norm": 1.0,
        "mlp_up": 1.0, "mlp_gate": 1.0, "mlp_down": 1.0,
        "post_attention_norm": 1.0, "post_mlp_norm": 1.0,
        "embedding": 1.0, "output_norm": 1.0, "unembedding": 1.0,
    },
)
