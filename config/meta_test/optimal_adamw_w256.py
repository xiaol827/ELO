# Optimal MuP AdamW config for muztransformer-dense-w256-d4-h2_fineweb-s512-gpt2
# From sweep fyr7px8o (625 configs, best loss = 3.8683)

_base_ = ["./meta_test_base.py"]

run_type = "benchmark"

optimizer_args = dict(
    class_="mup_adamw",
    kwargs=dict(
        learning_rate=0.02236,
        b1=0.9,
        b2=0.975,
        eps=1e-8,
        eps_root=0.0,
        mu_dtype=None,
        nesterov=False,
        mask=None,
        weight_decay=0.0045,
    )
)

schedule = dict(
    class_="warmup_cosine_decay",
    kwargs=dict(
        peak_value=0.02236,
        end_value=0.002236,
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
