
import time
import os

from mpi4py import MPI
import jax
from jax import random
import jax.numpy as jnp
import jax.distributed

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Simulate a `wandb.run.name` with a mock string, which will only be present on rank 0
    if rank == 0:
        wandb_run_name = "wandb_test_run"
    else:
        wandb_run_name = None  # Other ranks have nothing initially

    # Broadcast the wandb run name from rank 0 to all other ranks
    wandb_run_name = comm.bcast(wandb_run_name, root=0)

    # Print the result from each process
    print(f"Rank {rank}: Wandb run name is {wandb_run_name}")


def main():


    #recommended for single device comms
    os.environ.update({
        "NCCL_LL128_BUFFSIZE": "-2",
        "NCCL_LL_BUFFSIZE": "-2",
        "NCCL_PROTO": "SIMPLE,LL,LL128",
    })
    # os.environ.update({
    # "NCCL_PROTO": "LL,LL128",  # Prioritize low-latency protocols
    # "NCCL_LL_BUFFSIZE": "16384",  # Set a small buffer for LL
    # "NCCL_LL128_BUFFSIZE": "8192",  # Set a smaller buffer for LL128
    # "NCCL_MIN_NCHANNELS": "4",  # Increase channels to improve parallelism
    # "NCCL_MAX_NCHANNELS": "8",  # Increase further if beneficial
    # })


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Number of trials to run
    num_trials = 10
    total_time = 0

    # Function to compute the mean across devices
    def compute_mean(x):
        return jax.lax.pmean(x, axis_name='i')

    from tqdm import tqdm
    if rank == 0:
        iter_ = tqdm(range(num_trials))
    else:
        iter_ = range(num_trials)

    # @functools.partial(jax.pmap, in_axes=(None, 0), out_axes=(None, 0), axis_name="dev")
    # def gather_sequences(p_yses, n_yses):
    #     p_yses = jax.lax.all_gather(p_yses, axis_name="dev", axis=1)
    #     n_yses = jax.lax.all_gather(n_yses, axis_name="dev", axis=1)
    #     return p_yses, n_yses
    
    for trial in iter_:

        # Define the tensor x with the specified shape
        # Create initial tensors on each device
        x = {
            'adafactor_decays': jnp.ones((1, 3)) * rank,
            'momentum_decays': jnp.ones((1, 3)) * rank,
            'nn': {
                '~': {
                    'b0': jnp.ones((1, 32)) * rank,
                    'b1': jnp.ones((1, 32)) * rank,
                    'b2': jnp.ones((1, 2)) * rank, 
                    'w0': jnp.ones((1, 39, 32)) * rank,
                    'w1': jnp.ones((1, 32, 32)) * rank,
                    'w2': jnp.ones((1, 32, 2)) * rank
                }
            },
            'rms_decays': jnp.ones((1, 1)) * rank
        }

        # Gather tensors across devices using all_gather
        x_gathered = jax.tree_util.tree_map(
            lambda x: jax.lax.all_gather(x, axis_name='dev', axis=0),
            x
        )

        print( jax.tree.map(lambda y: y.shape, x_gathered))

        exit(0)
        # print(jax.tree_util.tree_map(lambda x: x.device, x))
        start_time = time.time()
        # Apply the function using jax.pmap
        # mean = jax.tree.map(lambda x: jax.pmap(compute_mean, axis_name='i')(x), x)

        print( jax.tree.map(lambda y: y.shape, x))

        # both = jax.pmap(jax.lax.all_gather, axis_name="dev")(x, axis_name="dev", axis=0)
        both = jax.tree.map(lambda y: jax.pmap(jax.lax.all_gather, axis_name="dev")(y, axis_name="dev", axis=0, tile=True), x)
        print( jax.tree.map(lambda x: x.shape,both))
        exit(0)

        # Block until computation is done for accurate timing
        jax.tree.map(lambda x: x.block_until_ready(), mean)

        trial_time = time.time() - start_time
        total_time += trial_time

        # if rank == 0:
        #     print(f"\nTrial {trial + 1}:")
        # print(f"Time taken: {trial_time:.4f} seconds")
        # print("Mean values:", mean)
        # print("Shapes:", jax.tree_util.tree_map(lambda x : x.shape, mean))
        # print("Expanded shapes:", jax.tree_util.tree_map(lambda x : jnp.expand_dims(x,axis=0).shape, mean))

    avg_time = total_time / num_trials
    if rank==0:
        print(f"\nAverage time over {num_trials} trials: {avg_time:.4f} seconds")


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    from functools import partial
    def allgather_pytree(pytree, axis=0):
        """
        Perform an all-gather operation on all leaf tensors in the pytree.
        The tensors are stacked along dimension 0.
        """
        return jax.tree_util.tree_map(lambda x: jax.lax.all_gather(x, 'devices', axis=axis), pytree)

    # Initialize JAX distributed process
    jax.distributed.initialize()

    # Get the current device rank
    rank = jax.process_index()
    num_devices = jax.device_count()

    # Create a pytree of tensors
    x = {
        'adafactor_decays': jnp.ones((2, 3)) * rank,
        'momentum_decays': jnp.ones((2, 3)) * rank,
        'nn': {
            '~': {
                'b0': jnp.ones((2, 32)) * rank,
                'b1': jnp.ones((2, 32)) * rank,
                'b2': jnp.ones((2, 2)) * rank, 
                'w0': jnp.ones((2, 39, 32)) * rank,
                'w1': jnp.ones((2, 32, 32)) * rank,
                'w2': jnp.ones((2, 32, 2)) * rank
            }
        },
        'rms_decays': jnp.ones((2, 1)) * rank
    }

    # Unsqueeze all leaves in the pytree by adding a dimension at axis 0
    x = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), x)
    # Perform all-gather across devices
    # allgathered_x = jax.pmap(allgather_pytree, axis_name='devices')(x)

    allgathered_x = jax.pmap(partial(allgather_pytree,axis=0), axis_name='devices')(x)
    # Reshape the pytree to combine the first two dimensions
    def reshape_first_two_dims(x):
        # Get shape and reshape to combine first two dims
        shape = x.shape
        return x.reshape([shape[0] * shape[1] * shape[2]] + list(shape[3:]))
    
    allgathered_x = jax.tree_util.tree_map(reshape_first_two_dims, allgathered_x)


    print( jax.tree_util.tree_map(lambda x: x.shape, allgathered_x))
    # Print result
    print(rank, allgathered_x['rms_decays'])

    exit(0)

    # Initialize the distributed system with JAX
    jax.distributed.initialize()

    main()
    exit(0)






# import jax
# import jax.numpy as jnp


# jax.distributed.initialize() 
# rank = jax.process_index()

# x = jnp.ones((1, 10)) * rank

# x = jax.lax.pmean(x, axis_name='i')

# print(x)
