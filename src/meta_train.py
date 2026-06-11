import os
import os.path as osp
import pickle


# from mpi4py import MPI  # Not needed - using SLURM/JAX-based communication instead

from tqdm import tqdm
import numpy as np
import os
import jax
import jax.numpy as jnp
import wandb
import jax.experimental.multihost_utils as multihost_utils

from learned_optimization import checkpoints
from learned_optimization.outer_trainers import (
    gradient_learner,
    truncated_pes,
    truncation_schedule,
    full_es,
    es_single,
)

from meta_trainers import get_meta_trainer
from helpers import get_resume_ckpt, save_checkpoint, set_non_hashable_args, cast_to_bf16, Timing
import globals


def write_wandb_custlog(args, wandb_run_id, outer_step):
    """Write checkpoint info to jobs/wandb_custlog/<meta_train_name>.log.

    Each run_id keeps only one line (the latest step). Entity is read from wandb.run.
    """
    log_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "jobs", "wandb_custlog")
    os.makedirs(log_dir, exist_ok=True)
    log_path = osp.join(log_dir, f"{args.meta_train_name}.log")
    entity = wandb.run.entity if wandb.run else "<NEED>"  # default to my personal account if not running in a wandb run context
    wandb_checkpoint_id = f"{entity}/{args.train_project}/{wandb_run_id}"
    new_line = f"{args.meta_train_name},{wandb_checkpoint_id},{outer_step}"

    # Read existing lines, replace if same run_id exists, else append
    lines = []
    if osp.exists(log_path):
        with open(log_path, "r") as f:
            lines = [l.rstrip("\n") for l in f.readlines()]
    replaced = False
    for idx, line in enumerate(lines):
        if f"/{wandb_run_id}," in line:
            lines[idx] = new_line
            replaced = True
            break
    if not replaced:
        lines.append(new_line)
    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def broadcast_string_across_hosts(string_value, max_length=100):
    """Broadcast a string from rank 0 to all other ranks using JAX."""
    import jax.experimental.multihost_utils as multihost_utils

    rank = jax.process_index()

    # Convert string to array of integers (ASCII values)
    if rank == 0:
        # Pad or truncate to fixed size
        padded_str = string_value.ljust(max_length)[:max_length]
        str_array = jnp.array([ord(c) for c in padded_str], dtype=jnp.int32)
    else:
        # Create empty array on non-zero ranks
        str_array = jnp.zeros(max_length, dtype=jnp.int32)

    # Broadcast from rank 0 to all ranks using process_allgather
    # All ranks will gather all values, then we take rank 0's value
    gathered = multihost_utils.process_allgather(str_array, tiled=False)

    # Take rank 0's value (first in the gathered array)
    rank0_array = gathered[0]

    # Convert back to string
    result_str = ''.join(chr(int(c)) for c in rank0_array if c != 0).strip()


    return result_str


def meta_train(args):
    args = set_non_hashable_args(args)
    meta_trainer, meta_opt = get_meta_trainer(args)

    # Rank-specific key drives particle diversity (perturbations, inner-problem
    # trajectory offsets).  Theta must be IDENTICAL on all ranks so that the
    # PES all-gather mixes particles that were all evolved under the same theta.
    # We therefore init theta from a fixed key (rank 0's key) and broadcast it,
    # while keeping the rank-specific key only for particle initialization.
    key = jax.random.PRNGKey(args.rank * 10000)   # rank-specific: for particles
    theta_init_key = jax.random.PRNGKey(0)         # fixed: same on all ranks
    _, theta_key1 = jax.random.split(theta_init_key)
    key, particle_key1 = jax.random.split(key)

    # Build a temporary init key that uses the fixed theta key for the first
    # split (theta) and the rank-specific key for the second split (particles).
    # SingleMachineGradientLearner.init() does: key1, key = split(key); then
    # uses key1 for theta and key for particles.  We pre-split to match that.
    import jax.experimental.multihost_utils as multihost_utils

    # Use a fixed seed so theta_opt_state is identical on all ranks at init.
    outer_trainer_state = meta_trainer.init(theta_key1)

    # Broadcast the gradient_learner_state (theta) from rank 0 to all ranks.
    # Particle states (gradient_estimator_states) are intentionally left
    # rank-specific so each process simulates different inner trajectories.
    if args.world_size > 1:
        synced_gl_state = multihost_utils.broadcast_one_to_all(
            outer_trainer_state.gradient_learner_state
        )
        from learned_optimization.outer_trainers.gradient_learner import SingleMachineState
        outer_trainer_state = SingleMachineState(
            gradient_learner_state=synced_gl_state,
            gradient_estimator_states=outer_trainer_state.gradient_estimator_states,
        )
        print(
            f"[PARALLEL|RANK {args.rank}] theta broadcast from rank 0 complete. "
            f"All ranks now share the same meta-optimizer parameters."
        )


    # Keep the rank-specific key alive for the training loop (drives per-step
    # perturbation diversity when key is split each outer iteration).

    key = particle_key1

    globals.needs_state = args.needs_state
    globals.num_grads = args.num_grads
    globals.num_local_steps = args.num_local_steps
    globals.local_batch_size = args.local_batch_size[0]
    globals.use_pmap = args.use_pmap
    globals.num_devices = args.num_devices

    if args.use_pmap:
        assert args.num_grads % args.num_devices == 0, "The number of devices for parallelism should be a divisor of the number of clients (gradients)"
    
    if args.finetune:
        with open(args.test_checkpoint, "rb") as f:
            meta_params = pickle.load(f)
        
    run = None
    wandb_run_id = ''
    if args.from_checkpoint:
        dirname = osp.join("checkpoints", args.meta_train_name)
        ckpt = open(osp.join(dirname, "latest"), "r").readline().strip()
        outer_trainer_state = checkpoints.load_state(
            osp.join(dirname, "{}.ckpt".format(ckpt)), outer_trainer_state
        )
        if args.rank == 0:
            run = wandb.init(
                entity="<NEED>",
                project=args.train_project,
                group=args.meta_train_name,
                config=vars(args),
            )
            run.log_code(".")

    elif args.auto_resume:

        ckpt = get_resume_ckpt(osp.join(os.environ["SCRATCH"], "checkpoints"), args.meta_train_name)

        if ckpt is not None:
            outer_trainer_state = checkpoints.load_state(
                osp.join(ckpt,"rank-{}_outer_trainer_state.ckpt".format(args.rank)), outer_trainer_state
            )
            if args.rank == 0:
                run = wandb.init(
                    entity="<NEED>",
                    project=args.train_project,
                    group=args.meta_train_name,
                    config=vars(args),
                    resume='allow',
                    id=ckpt.split('/')[-1][:8]
                )
                run.log_code(".")
                wandb_run_id = run.id
            
    
    if run == None:
        if args.rank == 0:
            run = wandb.init(
                entity="<NEED>",
                project=args.train_project,
                group=args.meta_train_name,
                config=vars(args),
            )
            run.log_code(".")
            wandb_run_id = run.id



    # Broadcast wandb_run_id from rank 0 to all ranks using JAX
    wandb_run_id = broadcast_string_across_hosts(wandb_run_id if wandb_run_id else '')

    # Print the result from each process
    print(f"Rank {args.rank}: Wandb run name is {wandb_run_id}")

    i = None
    iteration = int(
        outer_trainer_state.gradient_learner_state.theta_opt_state.iteration
    )
    pbar = tqdm(
        range(iteration, args.num_outer_steps),
        initial=iteration,
        total=args.num_outer_steps,
        ascii=True,
        desc="Outer Loop",
        mininterval=0,  # update as often as possible
        miniters=1,      # update every iteration
        # dynamic_ncols=True
    )
    logging_task_name = args.task[0] if len(args.task) == 1 else "multi-task-with_" + args.task[0]


    meta_train_update, metric_all_reduce_time = [], []
    for i in range(iteration, args.num_outer_steps):
        
        key, key1 = jax.random.split(key)

        with Timing('meta train update',meta_train_update):
            outer_trainer_state, meta_loss, metrics = meta_trainer.update(
                outer_trainer_state, key1, with_metrics=True
            )
            # synchronize to get correct step time
            jax.experimental.multihost_utils.sync_global_devices('sync')

        # update truncation length
        for x in range(len(meta_trainer.gradient_estimators)):

            if type(meta_trainer.gradient_estimators[x]) in (truncated_pes.TruncatedPES, es_single.ESSingle) or (hasattr(meta_trainer.gradient_estimators[x], 'update_truncation_length')):
                meta_trainer.gradient_estimators[x].update_truncation_length(i)
            if hasattr(meta_trainer.gradient_estimators[x], 'update_std'):
                meta_trainer.gradient_estimators[x].update_std(i)
                try:
                    meta_trainer.gradient_estimators[x].truncated_step.learned_opt.outer_step = i
                except:
                    print(f"Gradient estimator {x} does not have a learned optimizer")
                    pass


        # Calculate local mean and max data time
        local_mean_data_time = np.mean(meta_trainer.gradient_estimators[0].truncated_step.timings[-50 // args.steps_per_jit:])
        local_total_time = np.sum(meta_trainer.gradient_estimators[0].truncated_step.timings[-50 // args.steps_per_jit:])
        local_max_data_time = np.max(meta_trainer.gradient_estimators[0].truncated_step.timings[-50 // args.steps_per_jit:])

        # All-reduce to get max across all processes using JAX

        # Note: This is synchronous, but JAX handles async execution internally via NCCL
        with Timing('AR time',metric_all_reduce_time):
            if args.world_size > 1:
                # Convert to JAX arrays and perform all-reduce max
                local_metrics = jnp.array([local_max_data_time, meta_train_update[-1]])

                # Use process_allgather to get all values, then take max
                import jax.experimental.multihost_utils as multihost_utils
                gathered_metrics = multihost_utils.process_allgather(local_metrics, tiled=False)
                max_metrics = jnp.max(gathered_metrics, axis=0)

                max_data_time = float(max_metrics[0])
                meta_train_time = float(max_metrics[1])
            else:
                # Single process case

                max_data_time = local_max_data_time
                meta_train_time = meta_train_update[-1]





        if args.rank == 0:

            _gather_times = Timing.run_times_dict.get('PES Gather') or Timing.run_times_dict.get('ES-Single Gather') or [0]
            more_to_log = {
                    "iteration": i,
                    "meta loss": meta_loss,

                    # "PES Gather" : _gather_times[-1],
                    # "Global AR": Timing.run_times_dict['meta train all reduce'][-1],
                    # 'Unroll Time': Timing.run_times_dict['meta train unroll'][-1],
                    # "AR metric time" : round(metric_all_reduce_time[-1], 4),
                    # "meta iter time" : round(meta_train_time, 4),
                    # "local data time mean" : round(local_mean_data_time, 7),
                    # "local data time total" : round(local_total_time, 7),
                    # "Data time max" : round(max_data_time, 7),
                    "learning rate" : meta_opt.__dict__.get(
                        "schedule", lambda x: args.learning_rate
                    )(
                        outer_trainer_state.gradient_learner_state.theta_opt_state.iteration
                        - 1
                    ),
                }

            pbar.set_postfix({
                "meta loss" : round(float(meta_loss),2), #this has been all-reduced
                # "Global AR" : more_to_log["Global AR"],
                # "Metric AR" : more_to_log["AR metric time"],
                # "PES Gather" : more_to_log["PES Gather"],
                # "Iter T" : more_to_log["meta iter time"],
                # "max Data T" : more_to_log["Data time max"],
                # "Unroll T" : more_to_log['Unroll Time'],
                # "L-Data T total" : more_to_log["local data time total"],
                # "Local Data T mean" : more_to_log["local data time mean"],
                "LR:" : round(more_to_log["learning rate"],5),
                
            })
            pbar.update(1)
            
            metrics.update(more_to_log)
            run.log(
                metrics,
                step=i,
            )

            if (i + 1) % args.save_iter == 0 or i == 1:

                #TODO: add support for saving meta-training checkpoints in parallel
                savepath = save_checkpoint(
                    prefix=wandb_run_id, i=i, args=args, outer_trainer_state=outer_trainer_state, rank=args.rank,
                )
                wandb.save(savepath)
                write_wandb_custlog(args, wandb_run_id, i)

        else:

            if (i + 1) % args.save_iter == 0 or i == 1:
                save_checkpoint(
                    prefix=wandb_run_id, i=i, args=args, outer_trainer_state=outer_trainer_state, rank=args.rank
                )
                
        jax.experimental.multihost_utils.sync_global_devices('sync')

    


    if args.rank == 0:

        # Todo: check if this is a fix to error when resuming from final checkpoint
        if i is None:
            i = iteration

        savepath = save_checkpoint(
            prefix=wandb_run_id, i=i, args=args, outer_trainer_state=outer_trainer_state, rank=args.rank
        )

        wandb.save(savepath)
        write_wandb_custlog(args, wandb_run_id, i)
        run.finish()

    # all procs wait for wandb to finish
    jax.experimental.multihost_utils.sync_global_devices('sync')

    exit(0)
        

