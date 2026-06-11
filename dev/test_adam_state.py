from learned_optimization.optimizers import optax_opts

local_opt = optax_opts.Adam(
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)

import jax
import jax.numpy as jnp
import haiku as hk

# Define a simple MLP
def mlp_fn(x):
    mlp = hk.Sequential([
        hk.Linear(2), jax.nn.relu,
        hk.Linear(2), jax.nn.relu,
        hk.Linear(2)
    ])
    return mlp(x)

# Transform the MLP function
mlp = hk.transform(mlp_fn)

# Initialize parameters
key = jax.random.PRNGKey(42)
dummy_input = jnp.ones((1, 2))  # Example input shape for MNIST
params = mlp.init(key, dummy_input)

# Initialize Adam optimizer state for the MLP parameters
adam_state = local_opt.init(params)
# Create 8 copies of the adam_state and stack them along the first dimension
num_workers = 8
# Create a batch of identical adam states
batched_adam_state = jax.tree_util.tree_map(
    lambda x: jnp.stack([x] * num_workers),
    adam_state
)
# Get the state from the batched optimizer state


# print(state)
# exit(0)

# print("Adam optimizer state initialized for MLP")
# print(f"Parameter count: {sum(x.size for x in jax.tree_util.tree_leaves(params))}")
# # print(f"Adam state: {adam_state}")
# # print(jax.tree_util.tree_map(lambda x: x.shape, params))
# print(adam_state.__dict__.keys())
print(adam_state)
print(adam_state.params)

adam_state.params = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), adam_state.params)

print(adam_state.params)
exit(0)
print(adam_state.optax_opt_state)
print(len(adam_state.optax_opt_state))
print(adam_state.optax_opt_state[0].count)
print(adam_state.optax_opt_state[0].mu)
print(adam_state.optax_opt_state[0].nu)

print("nu shape",jax.tree_util.tree_map(lambda x: x.shape, adam_state.optax_opt_state[0].nu))
print("nu shape",jax.tree_util.tree_map(lambda x: x.shape, batched_adam_state.optax_opt_state[0].nu))