# Python standard library
import argparse
import os
os.environ["KERAS_BACKEND"] = "jax"

# os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # disabled for debugging

# XLA flags must be set BEFORE importing JAX to ensure they take effect
# for all XLA compilation paths (pmap AND GSPMD/shard_map).
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_triton_gemm=true '
    '--xla_gpu_enable_pipelined_all_reduce=true '
    '--xla_gpu_autotune_level=0 '  # disable autotuner to avoid DEVICE_TYPE_INVALID crash in AutotunerPass on multi-process collectives

)
import os.path as osp
import pprint
import sys

# JAX related
import jax
from jax import lax

# from jax.lib import xla_bridge
import jax.numpy as jnp

# ML frameworks
import numpy as np

import wandb
# from mpi4py import MPI  # Not needed - using SLURM environment variables instead
from mmengine.config import Config, DictAction
try:
    from jax.experimental import multihost_utils
except ImportError:
    print("Error: jax.experimental.multihost_utils not found. Make sure you have a recent version of JAX installed.")
    exit(1)
# Local imports - MOVED to after JAX initialization to avoid TensorFlow conflicts
# from benchmark import benchmark, sweep
# from meta_train import meta_train
# from helpers import print_rank_0, test_bf16_support_on_gpu, download_wandb_checkpoint

def comma_separated_strings(string):
    # This function will be used to parse the comma-separated string into a list
    return string.split(',')

def parse_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--config_dir", type=str, default="")
    parser.add_argument("--config", type=comma_separated_strings, required=True, help='space separated list of config files')
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument("--local_batch_size", metavar='N', type=int, nargs='+',help='an integer for the list')
    parser.add_argument("--run_type", type=str, choices=["benchmark", "meta-train","sweep"])
    parser.add_argument("--local_optimizer", type=str, choices=["sgd", "adam", "adamw", "muon"])
    parser.add_argument("--optimizer", type=str, )
    parser.add_argument("--task", type=comma_separated_strings)
    parser.add_argument("--needs_state", action="store_true")
    parser.add_argument("--name", type=str)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--local_learning_rate", type=float)
    parser.add_argument("--num_grads", type=int)
    parser.add_argument("--num_local_steps", type=int)
    parser.add_argument("--steps_per_jit", type=int)
    parser.add_argument("--num_runs", type=int)
    parser.add_argument("--num_inner_steps", type=int)
    parser.add_argument("--num_outer_steps", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--sweep_config", type=str)
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--test_checkpoint", type=str)
    parser.add_argument("--use_pmap", action="store_true")
    parser.add_argument("--num_tasks", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--num_devices", type=int)
    parser.add_argument("--name_suffix", type=str)
    parser.add_argument("--slowmo_learning_rate", type=float)
    parser.add_argument("--wandb_checkpoint_id", type=str)
    parser.add_argument("--meta_loss_split", type=str)
    parser.add_argument("--test_project", type=str)
    parser.add_argument("--train_project", type=str)
    parser.add_argument("--tfds_data_dir", type=str, default="<DATA_DIR>") # os.getenv("SLURM_TMPDIR") "<DATA_DIR>"
    parser.add_argument("--wandb_dir", type=str, default=os.getenv("SCRATCH"))
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--truncation_schedule_min_length", type=int)
    parser.add_argument("--sweep_id", type=str)
    parser.add_argument("--lo_clip_grad", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--image_dtype", type=str, default="float32",
                        choices=["float32", "bfloat16", "float16"],
                        help="Dtype of image batches yielded by vision dataset iterators.")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--test_interval", type=int)
    parser.add_argument("--prefetch_batches", type=int)
    parser.add_argument("--adafac_step_mult", type=float)
    parser.add_argument("--mup_input_mult", type=float)
    parser.add_argument("--mup_output_mult", type=float)
    parser.add_argument("--mup_hidden_lr_mult", type=float)
    parser.add_argument("--mup_depth_mult", type=float)
    parser.add_argument("--mup_depth_lr_mult", type=float)
    parser.add_argument("--keep_batch_in_gpu_memory", action="store_true")
    parser.add_argument("--no_meta_clip", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--truncation_length", type=int)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--log_activations", action="store_true")
    parser.add_argument("--mup_to_lstm", action="store_true")
    parser.add_argument("--zero_lstm_features", action="store_true")
    parser.add_argument("--zero_training_step_feature", action="store_true")
    parser.add_argument("--use_es", action="store_true")
    parser.add_argument("--use_es_single", action="store_true")

    

    parser.add_argument("--use_localsgd_batches", action="store_true")
    parser.add_argument("--quantized", type=str)
    parser.add_argument("--master_node", type=str)
    parser.add_argument("--benchmark_momentum", type=float)
    parser.add_argument("--benchmark_b1", type=float)
    parser.add_argument("--benchmark_b2", type=float)
    parser.add_argument("--benchmark_weight_decay", type=float)
    parser.add_argument("--selected_checkpoint", type=str, default='')
    parser.add_argument("--checkpoint_soup_range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Closed interval [START, END] of global_step values. "
                             "All wandb checkpoints in this range are downloaded and averaged "
                             "(checkpoint soup) before meta-test.")
    parser.add_argument("--force_resoup", action="store_true",
                        help="Force re-averaging even if a cached soup file already exists.")
    parser.add_argument("--sgd_clip", type=float)
    parser.add_argument("--save_iter", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--es_std", type=float)
    parser.add_argument("--es_std_schedule_step", type=int, default=None)
    parser.add_argument("--es_std_final", type=float, default=None)
    parser.add_argument("--master_port", type=int)
    parser.add_argument("--meta_optimizer", type=str)
    parser.add_argument("--truncation_inner_problem_ratio", type=int)

    parser.add_argument("--pmap_across_devices", action="store_true")
    parser.add_argument("--es_loss_type", type=str, default=None,
                        choices=["mean", "sum", "final", "telescoping", "weighted"],
                        help="Loss aggregation for ES-Single (default: mean)")
    parser.add_argument("--es_final_loss_weight", type=float, default=None,
                        help="Blend weight for 'weighted' ES loss type (0=mean, 1=final)")

    parser.add_argument("--bc_grad_weight", type=float)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--ovr_test_batch_size", type=int)


    parser.add_argument("--use_benchmark_schedule", action="store_true")
    parser.add_argument("--test_accumulate_steps", type=int)

    parser.add_argument("--task_aug_level", type=str, 
                        choices=["global", "tensor", "parameter"],
                        help="Task augmentation level")
    parser.add_argument("--task_aug_range", type=float, nargs=2,
                        help="Task augmentation param_scale range (log-uniform)")

    parser.add_argument("--use_task_augmentation", action="store_true")
    parser.add_argument("--time_limit_hours", type=float)
    parser.add_argument("--checkpoints_to_keep", type=int, default=2)

    # fmt: on

    return parser.parse_args()


def assert_args(args):
    # fmt: off
    if args.run_type == "benchmark" and args.optimizer in ["fedlopt", "fedlopt-adafac", "fedlagg", "fedlagg-wavg", "fedlagg-adafac"]:
        assert os.path.exists(args.test_checkpoint), "need to meta-train learned optimizer before benchmarking"
        assert args.test_checkpoint.endswith('.pickle'), "optimizer checkpoints must be saved as .pickle files"
    if args.run_type == "meta-train":
        assert args.optimizer not in ["adam", "fedavg", "fedavg-slowmo"], "can't meta-train a non learned optimizer"
    if getattr(args, "checkpoint_soup_range", None) is not None:
        assert len(args.checkpoint_soup_range) == 2, "--checkpoint_soup_range must be 2 ints: START END"
        start, end = args.checkpoint_soup_range
        assert start <= end, f"--checkpoint_soup_range START ({start}) must be <= END ({end})"
        assert not getattr(args, "selected_checkpoint", None), \
            "--checkpoint_soup_range is mutually exclusive with --selected_checkpoint"
        assert getattr(args, "wandb_checkpoint_id", None) is not None, \
            "--checkpoint_soup_range requires --wandb_checkpoint_id to locate the source run"



if __name__ == "__main__":
    print(f"Started ")
    args = parse_args()
    sys.path.append(os.getcwd())

    #########################################################
    # Set hardcoded environment variables
    #########################################################

    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    # XLA_FLAGS are now set at module top (before `import jax`) to ensure

    # they take effect for all XLA paths including GSPMD/shard_map.
    #recommended for single device comms
    os.environ.update({
        "NCCL_LL128_BUFFSIZE": "-2",
        "NCCL_LL_BUFFSIZE": "-2",
        "NCCL_PROTO": "SIMPLE,LL,LL128",
    })
    

    #########################################################
    # Setup distribute

    #########################################################

    # Use OpenMPI env vars (from mpirun) if available, else SLURM env vars
    if os.getenv('OMPI_COMM_WORLD_SIZE') is not None:
        rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
        size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
        local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
    else:
        rank = int(os.getenv('SLURM_PROCID', '0'))
        size = int(os.getenv('SLURM_NTASKS', '1'))
        local_rank = int(os.getenv('SLURM_LOCALID', '0'))

    # Define print_rank_0 helper function
    def print_rank_0(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    print_rank_0(f"before distributed init: CUDA_VISIBLE_DEVICES = {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"Process {rank} of {size} is running on {os.uname()[1]}")
    if args.master_node is not None:
        coordinator_address = f"{args.master_node}:{args.master_port}"
        print_rank_0(f"Initializing JAX distributed with coordinator_address={coordinator_address}, num_processes={size}, process_id={rank}")

        # Initialize the distributed environment
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=size,
            process_id=rank,
            local_device_ids=[local_rank],

        )


        # print(xla_bridge.get_backend().platform)
        
        print(jax.devices())
        print(jax.local_devices())

        # Commenting out sync - causes NCCL errors with single-node multi-process setup
        multihost_utils.sync_global_devices('sync')
        
        # CRITICAL: Set TensorFlow env vars BEFORE importing TensorFlow
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
        import tensorflow as tf
        tf.config.experimental.set_visible_devices([], "GPU")

    # Import local modules AFTER JAX and TensorFlow initialization
    print_rank_0("Importing local modules...")
    from benchmark import benchmark, sweep
    from meta_train import meta_train
    from helpers import test_bf16_support_on_gpu, download_wandb_checkpoint, build_checkpoint_soup
    import parameterization
    parameterization.VERBOSE = args.verbose

    assert len(args.local_batch_size) == len(args.task), f"local batch size and task length mismatch: {len(args.local_batch_size)} != {len(args.task)} , pass batch size for each tasks"

    
    #########################################################
    # load all configs
    #########################################################
    config_dir = args.config_dir
    config_files = [osp.join(config_dir, f) for f in args.config]
    cfg = Config.fromfile(config_files[0])

    # Manually merge config files, checking for duplicates
    for config_file in config_files[1:]:
        new_cfg = Config.fromfile(config_file)
        for key, value in new_cfg._cfg_dict.items():
            if key in cfg._cfg_dict and cfg._cfg_dict[key] != value:
                raise ValueError(f"Duplicate config key '{key}' with different values: "
                                f"'{cfg._cfg_dict[key]}' vs '{value}'. "
                                f"Found in file: {config_file}")
            cfg._cfg_dict[key] = value

    


    #########################################################
    # override args from the command line
    #########################################################





    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    for k, v in vars(args).items():
        if v is not None:
            if args.verbose:
                print_rank_0("[INFO] Overriding config value: {}={}".format(k, v))
            cfg._cfg_dict[k] = v

    # Sync --pmap_across_devices CLI flag to gradient_estimator config
    if args.pmap_across_devices:
        cfg._cfg_dict['gradient_estimator_args']['kwargs']['pmap_across_devices'] = True

    #########################################################
    # Set wandb logging names
    #########################################################

    if args.use_localsgd_batches:
        cfg.name = "{}{}_{}_K{}_H{}{}".format(
            cfg.optimizer, cfg.hidden_size, cfg.task,
            cfg.num_grads,
            cfg.num_local_steps, cfg.name_suffix
        )
    else:
        cfg.name = "{}{}_{}{}".format(
            cfg.optimizer, cfg.hidden_size, cfg.task, cfg.name_suffix
        )
    cfg.meta_train_name = "{}{}_{}_K{}_H{}_{}_{}_{}".format(
        cfg.optimizer,
        cfg.hidden_size,
        cfg.task[0] if len(cfg.task) == 1 else "multi-task-with"+cfg.task[0],
        cfg.num_grads,
        cfg.num_local_steps,
        cfg.local_optimizer,
        cfg.local_learning_rate,
        cfg.name_suffix,
    )
    if args.num_devices is not None:
        # User explicitly provided --num_devices via CLI, respect it
        pass  # cfg.num_devices already set from CLI override at line 264
    elif args.use_localsgd_batches:
        cfg.num_devices = len(jax.devices())
    else:
        cfg.num_devices = 1

    # dont download the checkpoint if it already exists
    if args.test_checkpoint is None \
       and cfg.wandb_checkpoint_id is not None:
        if getattr(cfg, "checkpoint_soup_range", None) is not None:
            cfg.test_checkpoint = build_checkpoint_soup(cfg)
        else:
            cfg.test_checkpoint = download_wandb_checkpoint(cfg)
    elif args.test_checkpoint is not None \
         and getattr(cfg, "checkpoint_soup_range", None) is not None:
        raise ValueError(
            "--test_checkpoint and --checkpoint_soup_range are mutually exclusive: "
            "soup builds its own averaged checkpoint."
        )

    args = argparse.Namespace(**cfg._cfg_dict)
    assert_args(args)

    # Ensure TFDS_DATA_DIR is set. Use existing env var if present; fall back to
    # args.tfds_data_dir (which may come from --tfds_data_dir CLI flag or config).
    if not os.environ.get('TFDS_DATA_DIR') and args.tfds_data_dir:
        os.environ['TFDS_DATA_DIR'] = args.tfds_data_dir

    # Check the rank of the current process# Check the rank of the current process
    args.rank = jax.process_index()
    print(f"Process rank: {args.rank}")

    # Get the world size
    args.world_size = jax.process_count()
    print(f"World size: {args.world_size}")

    args.global_task_size = len(args.task)

    #########################################################
    # Set distributed tasks
    #########################################################
    if args.world_size > 1 and args.run_type == "meta-train":
        #setup distributed
        if args.pmap_across_devices:
            # All ranks get the same task list when using pmap across devices
            print(
                f"[PARALLEL|RANK {args.rank}] pmap_across_devices=True: "
                f"all {args.world_size} ranks share the same task list={args.task}. "
                f"Parallelism is at the PES-particle level: each rank runs "
                f"num_tasks={args.num_tasks} particles locally, so globally there are "
                f"world_size*num_tasks={args.world_size}*{args.num_tasks}="
                f"{args.world_size * args.num_tasks} particles."
            )
        else:
            assert len(args.task) % args.world_size == 0, "world size must divide the number of tasks"
            args.task = np.array(args.task).reshape(args.world_size, -1)[args.rank].tolist()
            args.local_batch_size = np.array(args.local_batch_size).reshape(args.world_size, -1)[args.rank].tolist()

            print(args.rank, args.task)

    #########################################################
    # Set precision
    #########################################################
    if args.use_bf16 and test_bf16_support_on_gpu():
        print_rank_0('setting bf 16 as default supported')
        jax.config.update('jax_default_matmul_precision', 'bfloat16')
    else:
        jax.config.update("jax_default_matmul_precision", "high")

    # print_rank_0("augmentations", args.augmentations)
    # exit(0)

    run_types = {"benchmark": benchmark,
                 "meta-train": meta_train,
                 "sweep": sweep}
    run_types[args.run_type](args)



