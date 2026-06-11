# Optimal MuP Muon config for muztransformer-dense-w256-d4-h2_fineweb-s512-gpt2
# From sweep i3mm78cu (588 configs, best loss = 3.8328)

_base_ = ["./meta_test_base.py"]

run_type = "benchmark"

optimizer_args = dict(
    class_="mup_muon",
    kwargs=dict(
        learning_rate=0.0110,
        ns_coeffs=(3.4445, -4.775, 2.0315),
        ns_steps=5,
        beta=0.95,
        eps=1e-8,
        weight_decay=0.0156,
        nesterov=True,
        adaptive=False,
        mu_dtype=None,
        adam_b1=0.95484,
        adam_b2=0.9908,
        adam_eps=1e-8,
        adam_eps_root=0.0,
        adam_weight_decay=0.093198,
        weight_decay_mask=None,
    )
)

schedule = dict(
    class_="warmup_cosine_decay",
    kwargs=dict(
        peak_value=0.0110,
        end_value=0.00110,
        warmup_steps=500,
        decay_steps=4000,
        init_value=0.0,
        exponent=1.0,
    )
)

gradient_transform_before_optim = [
    dict(class_="clip_by_global_norm", kwargs=dict(max_norm=1.0))
]

gradient_transform_after_optim = []

num_inner_steps = 4500
save_iter = 999999
num_runs = 1
test_interval = 15
