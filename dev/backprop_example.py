import jax
import jax.numpy as jnp
import optax
import haiku as hk
from typing import Dict, Any, Callable, Tuple, Optional
from functools import partial

# Define a simple neural network that updates a pytree
class UpdateNetwork(hk.Module):
    def __init__(self, hidden_size=32, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        
    def __call__(self, params):
        # Function that processes each leaf in the pytree
        def process_leaf(leaf):
            # Get shape for reshaping back later
            original_shape = leaf.shape
            
            # Flatten the leaf for processing
            flat = leaf.reshape(-1)
            
            # Simple MLP to process the flattened tensor
            x = hk.Linear(self.hidden_size)(flat)
            x = jax.nn.relu(x)
            x = hk.Linear(flat.shape[0])(x)
            
            # Reshape back to original shape
            return x.reshape(original_shape)
        
        # Apply the leaf processor to each leaf in the pytree
        return jax.tree_map(process_leaf, params)

def run_backprop_example(
    initial_params: Dict[str, Any],
    network_params: Dict[str, Any],
    update_fn_t: Callable[[Dict[str, Any]], Dict[str, Any]],
    num_iterations: int = 10,
    learning_rate: float = 0.1,
    scale_factor: float = 0.1,
    use_teacher: bool = False,
    optimizer = None
) -> Tuple[Dict[str, Any], Dict[str, jnp.ndarray]]:
    """
    Run a backpropagation example where a neural network updates a pytree.
    
    Args:
        initial_params: Initial parameters pytree
        target_params: Target parameters pytree to compute MSE against
        num_iterations: Number of sequential updates
        learning_rate: Learning rate for gradient descent
        scale_factor: Scalar multiplier for the original pytree
        use_teacher: Whether to use teacher forcing (use target output) or not
        optimizer: Optax optimizer to use (defaults to SGD if None)
        
    Returns:
        final_params: The final parameters after all iterations
        metrics: Dictionary containing loss values for each iteration
    """
    rng_key = jax.random.PRNGKey(42)
    
    # Use default SGD if no optimizer is provided
    if optimizer is None:
        optimizer = optax.sgd(learning_rate)
    
    # Define the loss function (MSE between output and target)
    def mse_loss(params1, params2):
        squared_diff = jax.tree_util.tree_map(lambda x, y: jnp.square(x - y), params1, params2)
        leaves = jax.tree_util.tree_leaves(squared_diff)
        return jnp.mean(jnp.array([jnp.mean(leaf) for leaf in leaves]))
    
    # Define a function to compute gradients for a single step
    def compute_step_gradients(current_params, net_params):
        # Define a function that computes loss for a single step with given network parameters
        def loss_fn(params):
            # Apply the network to get the update
            updated_params = update_fn_t.apply(params, rng_key, current_params)
            
            # Apply the scale factor to the original params and add to the update
            scaled_original = jax.tree_util.tree_map(lambda x: scale_factor * x, initial_params)
            combined_params = jax.tree_util.tree_map(lambda x, y: x + y, updated_params, scaled_original)
            
            # Compute loss against target
            loss = mse_loss(combined_params, scaled_original)
            
            return loss, (loss, combined_params, scaled_original)
        
        # Compute gradient and auxiliary values with respect to network parameters
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (aux_loss, combined_params, scaled_original)), grads = grad_fn(net_params)
        
        return grads, loss, scaled_original, combined_params
    
    # Initialize accumulated gradients with the same structure as network_params but filled with zeros
    init_accumulated_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), network_params)
    
    # Define the update step for a single iteration - now just accumulating gradients
    def update_step(state, step_idx):
        current_input, net_params, accumulated_grads = state
        
        # Compute gradients and get auxiliary values for this step
        grads, loss, scaled_original, combined_params = compute_step_gradients(current_input, net_params)
        
        # Accumulate gradients
        accumulated_grads = jax.tree_util.tree_map(
            lambda acc_g, g: acc_g + g,
            accumulated_grads,
            grads
        )
        
        # For the next iteration, use either the teacher output (target_params) 
        # or the student output (combined_params)
        next_input = target_params if use_teacher else combined_params
        
        return (next_input, net_params, accumulated_grads), (combined_params, loss, grads)
    
    # Run the sequential updates
    (final_input, final_net_params, accumulated_grads), (final_params, losses, step_gradients) = jax.lax.scan(
        update_step, 
        (initial_params, network_params, init_accumulated_grads), 
        jnp.arange(num_iterations)
    )
    
    # Initialize optimizer state
    opt_state = optimizer.init(network_params)
    
    # Apply the accumulated gradients using the optimizer
    updates, _ = optimizer.update(accumulated_grads, opt_state, network_params)
    updated_net_params = optax.apply_updates(network_params, updates)
    
    metrics = {
        'losses': losses,
        'gradients': step_gradients,
        'accumulated_gradients': accumulated_grads
    }
    
    return updated_net_params, final_params, metrics

# Example usage:
def example():
    # Create some example parameters
    # Set a fixed random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create keys for each parameter
    key, key_w1, key_b1, key_w2, key_b2 = jax.random.split(key, 5)
    
    initial_params = {
        'layer1': {
            'w': jax.random.normal(key_w1, (10, 5)) * 0.1,
            'b': jax.random.normal(key_b1, (5,)) * 0.1
        },
        'layer2': {
            'w': jax.random.normal(key_w2, (5, 1)) * 0.1,
            'b': jax.random.normal(key_b2, (1,)) * 0.1
        }
    }
    # Transform the update network into a pure function
    def update_fn(params):
        network = UpdateNetwork()
        return network(params)
    
    update_fn_t = hk.transform(update_fn)
    
    # Initialize network parameters
    rng_key = jax.random.PRNGKey(42)
    network_params = update_fn_t.init(rng_key, initial_params)
    
    # Create Adam optimizer
    optimizer = optax.adam(learning_rate=0.001)
    
    # Run the example
    for x in range(20):
        # print(f"Iteration {x}")
        network_params, final_params, metrics = run_backprop_example(
            initial_params=initial_params,
            network_params=network_params,
            update_fn_t=update_fn_t,
            num_iterations=5,
            scale_factor=1.1,
            learning_rate=1.0,
            use_teacher=False,
            optimizer=optimizer
        )
        print(f"Iteration {x}", jnp.mean(jnp.array(metrics['losses'])))
        # Print the norm of the gradients
        # grad_norm = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), metrics['gradients'])
        # print(f"Gradient norms: {grad_norm}")
        
        # # Print the norm of the network parameters
        # param_norm = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), network_params)
        # print(f"Network parameter norms: {param_norm}")


    # print("Final loss:", metrics['losses'][-1])
    # print("Gradient w.r.t scale factor:", metrics['gradients'])
    # import pdb; pdb.set_trace()
    
    return final_params, metrics

if __name__ == "__main__":
    example()
