# Bad practice but the learned_optimization code is so nested that this is probably the easiest way to implement changes

needs_state = True
num_grads = 8
num_local_steps = 30
local_batch_size = 128
use_pmap = False
num_devices = 1