_base_ = ["../config_base.py"]

run_type = "meta-train"
save_iter = 500
checkpoints_to_keep = 2

# CELO2 paper: 8 parallel tasks, K=50 outer update frequency
num_tasks = 8
num_outer_steps = 5000
num_inner_steps = 2000
truncation_length = 50

# Small MLP tasks for efficient meta-training
local_batch_size = 64
hidden_size = 32
