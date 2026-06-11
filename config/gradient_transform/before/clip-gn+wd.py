

gradient_transform_before_optim = [
    dict(class_="clip_by_global_norm",
        kwargs=dict(max_norm=1.0)),
    dict(class_="add_decayed_weights",
        kwargs=dict(weight_decay=0.0001))
]