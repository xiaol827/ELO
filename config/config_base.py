# must be overriden
run_type = None

# common
optimizer = "fedavg"
task = "small-image-mlp-fmst"
hidden_size = 32
local_learning_rate = 0.5
local_batch_size = 128
num_grads = 8
num_local_steps = 4
num_inner_steps = 1000
learning_rate = 0.0001
name_suffix = ""
needs_state = False # For models storing inside state such as resnets
num_devices = 1

# meta training only
num_outer_steps = 5000
from_checkpoint = False
use_pmap = False
auto_resume = False
meta_loss_split = None
steps_per_jit = 10
num_tasks = 8
train_project = "learned_aggregation_meta_train"
prefetch_batches = 20

# meta testing only
num_runs = 10
wandb_checkpoint_id = None
gradient_accumulation_steps = 1
test_project = "learned_aggregation_meta_test"

truncation_schedule_min_length = 100
# for slowmo
beta = 0.99
test_interval = 50
adafac_step_mult=0.001
truncation_length=50
seed=0


#MuP
mup_input_mult = 1.0
mup_output_mult = 1.0
mup_hidden_lr_mult = 1.0
mup_depth_mult = 1.0
mup_depth_lr_mult = 1.0


# sweeps only
sweep_config = dict()
sweep_id = None

pmap_across_devices = False

keep_batch_in_gpu_memory = False
test_checkpoint=None


piecewise_schedule={}
#new sweeping variables
benchmark_momentum=0.0001
benchmark_weight_decay=0.0001
benchmark_b1=0.9
benchmark_b2=0.999


zero_lstm_features = False
mup_to_lstm = False
zero_training_step_feature=False

log_activations=False

use_es=False

selected_checkpoint=None
checkpoint_soup_range=None
force_resoup=False

quantized=None


sgd_clip=None
weight_decay=None
es_std=0.01
es_loss_type="mean"
es_final_loss_weight=0.0
meta_optimizer='adamw'

truncation_inner_problem_ratio=50
use_bf16=False

use_benchmark_schedule=False

bc_grad_weight=None
mup_small_fc_v2_args = dict()
local_optimizer=None

use_baseline_losses=False

ovr_test_batch_size=None

random_initial_iteration_offset_linspace=False

test_accumulate_steps=1

step_mult = 0.001


time_limit_hours=None  # if set, save checkpoint and exit cleanly when this many hours have elapsed                              
use_task_augmentation = False                                                                                                    
task_aug_level = "global"          # "global", "tensor", "parameter"                                                             
task_aug_scale_range = [0.1, 10]

test_accumulate_steps=1


