_base_ = ["../config_base.py"]

run_type = "benchmark"

num_inner_steps = 1000
save_iter = 500          # save a checkpoint every N inner steps
checkpoints_to_keep = 2  # keep only the last N checkpoints on disk
time_limit_hours = None   # no wall-clock limit; rely on SLURM --time
