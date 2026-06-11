
import jax
import gc
import threading
import pprint
import jax.numpy as jnp
import numpy as np
from functools import reduce
from typing import Optional, Dict, Any, Tuple

from parameterization import CompletedPParameterization, MuonCompletedPParameterization


is_leaf = lambda x : reduce(np.logical_and, [type(x1) != dict for x1 in x.values()])

def add_prefix(prefix,s):
    if prefix != '':
        prefix = prefix + '/'
    return prefix + s

def get_mup_lrs_hk(state,prefix):
    d = {}
    for k,v in state.items():
        if is_leaf(v):
            d[add_prefix(prefix,k)] = v
        else:
            for kk,vv in get_mup_lrs_hk(v,k).items():
                d[add_prefix(prefix,kk)] = vv
    
    d = {k.replace('/mup_lrs',''):v for k,v in d.items()}
    return d

def get_mup_lrs_from_state(state):
    if 'flax_mup_lrs' in state:
        lrs = state['flax_mup_lrs']
    else:
        lrs = get_mup_lrs_hk({k:{'mup_lrs':v['mup_lrs']} \
                              for k,v in state.items() if 'mup_lrs'in v.keys()}, 
                             prefix='')
    

    return lrs




class MuTask(object):
  """Base class for tasks with muP (maximal update parameterization) support.
  
  This class provides methods for computing and storing muP scaling factors
  that are used by learned optimizers for hyperparameter transfer.
  
  For CompletedP parameterization support, subclasses should:
  1. Call `set_parameterization_args()` during __init__ with base config
  2. Call `set_training_config()` before training with batch/steps info
  3. The `init_with_state()` method will automatically compute all scales
  
  Attributes:
    mup_state: Cached muP LR scales
    mup_eps_mult: Cached epsilon multiplier
    parameterization_args: Base CompletedP configuration (base_width, base_depth, etc.)
    training_config: Training-time configuration (batch_size, num_steps, etc.)
    parameterization: Cached CompletedPParameterization object
    completed_p_scales: Cached dictionary of all CompletedP scales
  """

  def set_parameterization_args(self, parameterization_args: Optional[Dict[str, Any]] = None):
    """Set the base CompletedP parameterization arguments.
    
    Args:
      parameterization_args: Dictionary containing:
        - base_width: Base model width for HP transfer
        - base_depth: Base model depth for HP transfer
        - base_batch_size: Base batch size for HP transfer
        - base_dataset_size: Base dataset size for HP transfer
        - depth_multipliers: Per-layer depth multipliers
        - alpha: Depth scaling exponent (0.5 to 1.0)
        - per_tensor_*_multipliers: Optional per-tensor-type multipliers
    """
    self.parameterization_args = parameterization_args
    self.parameterization = None
    self.completed_p_scales = None
  
  def set_training_config(
      self, 
      gradient_accumulation_steps: int = 1,
      local_batch_size: int = 1,
      num_inner_steps: int = 1000,
  ):
    """Set training-time configuration needed for CompletedP scaling.
    
    This should be called before training to configure the batch size and
    dataset size for CompletedP hyperparameter transfer.
    
    Args:
      gradient_accumulation_steps: Number of gradient accumulation steps
      local_batch_size: Local batch size per device
      num_inner_steps: Number of inner training steps (for dataset size calc)
    """
    self.training_config = {
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'local_batch_size': local_batch_size,
        'num_inner_steps': num_inner_steps,
    }
    # Reset cached parameterization when config changes
    self.parameterization = None
    self.completed_p_scales = None

  def _compute_completed_p_scales(self, params: Any) -> Dict[str, Any]:
    """Compute all CompletedP scaling factors for the given parameters.

    This method creates the CompletedP parameterization object and computes
    all the scaling pytrees (LR, epsilon, weight decay, betas).

    The parameterization class is selected via
    ``parameterization_args['parameterization_class']`` (popped before being
    forwarded to the constructor):
      - ``'completedp'`` (default): standard ``CompletedPParameterization`` —
        Adam scaling rules.
      - ``'muon_completedp'``: ``MuonCompletedPParameterization`` —
        Muon-specific LR / epsilon rules from Qiu et al. (2025), still
        producing the standard Adam scales for non-Muon params plus the
        additional ``mup_muon_*`` keys consumed by the Muon learned optimizer.

    Args:
      params: Model parameters to compute scales for

    Returns:
      Dictionary containing all CompletedP scales. Always present:
        - mup_lr_scales: Per-parameter LR scales (Adam scaling rules; for the
          muon path these are the ``adam_lr_scales`` from
          ``MuonCompletedPParameterization``).
        - mup_eps_scales: Per-parameter epsilon scales (Adam-side).
        - mup_wd_scales: Per-parameter weight decay scales (shared).
        - mup_one_minus_beta1_scales: Per-parameter (1-β₁) scales.
        - mup_one_minus_beta2_scales: Per-parameter (1-β₂) scales.
        - mup_tensor_type_indices: int32 per param into the TensorType enum.
      Additionally present when ``parameterization_class='muon_completedp'``:
        - mup_muon_lr_scales: Per-parameter Muon LR scales
          (carries √(d_out/d_in) * sqrt(m_B/m_D) for hidden weights, 1.0
          elsewhere).
        - mup_muon_eps_scales: Per-parameter Muon Newton-Schulz epsilon scales
          (carries √(d_in/d_out)/L * sqrt(m_D/m_B) for hidden weights, 1.0
          elsewhere).
        - mup_is_muon_mask: bool per param, True for Muon-side leaves
          (``ModuleType.HIDDEN_WEIGHT``).
    """
    if not hasattr(self, 'parameterization_args') or self.parameterization_args is None:
      return {}

    if not hasattr(self, 'training_config') or self.training_config is None:
      return {}

    # Get model configuration from flax_module
    if not hasattr(self, 'flax_module'):
      raise ValueError("Task must have flax_module attribute for CompletedP support")

    # Compute current batch size accounting for gradient accumulation and sequence length
    sequence_length = self.datasets.extra_info.get('sequence_length', 1)
    current_bs = (
        self.training_config['gradient_accumulation_steps']
        * self.training_config['local_batch_size']
        * sequence_length
    )

    # Pop the parameterization-class selector before passing kwargs to the
    # constructor (the Parameterization classes don't accept it).
    pargs = dict(self.parameterization_args)
    param_class_name = pargs.pop('parameterization_class', 'completedp').lower()
    if param_class_name == 'muon_completedp':
      ParamClass = MuonCompletedPParameterization
      use_muon = True
    elif param_class_name == 'completedp':
      ParamClass = CompletedPParameterization
      use_muon = False
    else:
      raise ValueError(
          f"Unknown parameterization_class={param_class_name!r}; "
          f"expected 'completedp' or 'muon_completedp'."
      )

    # Create the parameterization object
    self.parameterization = ParamClass(
        current_width=self.flax_module.docfg.width,
        current_depth=self.flax_module.docfg.depth,
        current_batch_size=current_bs,
        current_dataset_size=current_bs * self.training_config['num_inner_steps'],
        **pargs,
    )

    # Get tensor type annotations from the model
    tensor_types = self.flax_module.get_tensor_types(params)
    device = jax.devices()[jax.process_index()]

    # Standard scales — for the muon path we route through the explicit
    # ``get_adam_*_scales_pytree`` methods so the call site documents intent
    # (the inherited methods from the Adam parent return identical values for
    # non-HIDDEN_WEIGHT leaves, but typing the Adam path makes the behaviour
    # robust to any future override of ``get_lr_scales_pytree`` on the muon
    # subclass).
    if use_muon:
      lr_scales = self.parameterization.get_adam_lr_scales_pytree(tensor_types, device=device)
      eps_scales = self.parameterization.get_adam_eps_scales_pytree(tensor_types, device=device)
    else:
      lr_scales = self.parameterization.get_lr_scales_pytree(tensor_types, device=device)
      eps_scales = self.parameterization.get_eps_scales_pytree(tensor_types, device=device)
    wd_scales = self.parameterization.get_wd_scales_pytree(tensor_types, device=device)
    one_minus_beta1_scales = self.parameterization.get_one_minus_beta1_scales_pytree(tensor_types, device=device)
    one_minus_beta2_scales = self.parameterization.get_one_minus_beta2_scales_pytree(tensor_types, device=device)

    # Compute tensor type integer indices for learned optimizers
    from parameterization import TENSOR_TYPE_INDEX, get_module_type
    def _tt_index_scale_fn(module_type, fan_in, fan_out, tensor_type, layer_idx, device):
        return jax.device_put(
            jnp.array(TENSOR_TYPE_INDEX[tensor_type.value], dtype=jnp.int32),
            device
        )
    tensor_type_indices = self.parameterization._transform_tensor_types_tree_with_depth(
        tensor_types, _tt_index_scale_fn, device)

    scales = {
        'mup_lr_scales': lr_scales,
        'mup_eps_scales': eps_scales,
        'mup_wd_scales': wd_scales,
        'mup_one_minus_beta1_scales': one_minus_beta1_scales,
        'mup_one_minus_beta2_scales': one_minus_beta2_scales,
        'mup_tensor_type_indices': tensor_type_indices,
    }

    if use_muon:
      # Muon-specific per-leaf scales (computed by MuonCompletedPParameterization).
      muon_lr_scales = self.parameterization.get_muon_lr_scales_pytree(
          tensor_types, device=device)
      muon_eps_scales = self.parameterization.get_muon_eps_scales_pytree(
          tensor_types, device=device)

      # ``is_muon_mask`` and ``muon_shape_scales`` are derived from the SAME
      # ``make_muon_weight_dimension_numbers`` function that ``src/opt/mup_muon.py``
      # uses internally to partition params into Muon vs Adam. Going through
      # the dim_nums (rather than ``module_type == HIDDEN_WEIGHT``) guarantees
      # the lopt's partition stays in lockstep with the validated reference
      # optimizer under future module additions (e.g., MoE 3-D experts, LoRA
      # adapters), and lets us produce the same per-leaf
      # ``sqrt(d_out/d_in)`` shape factor that the reference applies inside
      # ``_compute_shape_scales`` (mup_muon.py:295-320). The lopt then applies
      # this shape factor to the Newton-Schulz output to match the validated
      # behaviour of mup_muon (which carries the factor *twice* — once via
      # this internal cache, once via the muon_lr_scales pytree above).
      from opt.mup_muon import _compute_shape_scales
      from opt.new_optimizers import make_muon_weight_dimension_numbers
      from optax.contrib._muon import MuonDimensionNumbers
      from optax.transforms._masking import MaskedNode

      dim_nums = make_muon_weight_dimension_numbers(params)

      def _is_leaf(x):
          return x is None or isinstance(x, MaskedNode) or isinstance(x, MuonDimensionNumbers)

      is_muon_mask = jax.tree_util.tree_map(
          lambda dn: jax.device_put(
              jnp.asarray(
                  not (dn is None or isinstance(dn, MaskedNode)),
                  dtype=jnp.bool_,
              ),
              device,
          ),
          dim_nums,
          is_leaf=_is_leaf,
      )

      muon_shape_scales_raw = _compute_shape_scales(
          params, make_muon_weight_dimension_numbers
      )
      muon_shape_scales = jax.tree_util.tree_map(
          lambda s: jax.device_put(jnp.asarray(s, dtype=jnp.float32), device),
          muon_shape_scales_raw,
      )

      scales['mup_muon_lr_scales'] = muon_lr_scales
      scales['mup_muon_eps_scales'] = muon_eps_scales
      scales['mup_muon_shape_scales'] = muon_shape_scales
      scales['mup_is_muon_mask'] = is_muon_mask

    return scales

  def _reinit_params_with_completed_p(self, params: Any, key: jax.random.PRNGKey) -> Any:
    """Re-initialize parameters with proper CompletedP scaling.
    
    Args:
      params: Original model parameters
      key: JAX random key for initialization
      
    Returns:
      Re-initialized parameters with CompletedP scaling
    """
    if getattr(self, 'parameterization', None) is None:
      return params

    tensor_types = self.flax_module.get_tensor_types(params)
    return self.parameterization.init_params_pytree(params, tensor_types, key=key)
  
  def _set_model_multipliers(self):
    """Set forward pass multipliers on the model from CompletedP parameterization."""
    if getattr(self, 'parameterization', None) is None:
      return
    
    device = jax.devices()[jax.process_index()]
    self.flax_module.set_multipliers_from_dict(
        self.parameterization.get_multipliers(device=device), 
        device=device
    )

  def get_mup_state(self, state, eps_mult=None):
    """Add muP state to the given state dictionary.
    
    This method adds the cached mup_lrs to the state dictionary.
    For full CompletedP support, use init_with_state() instead.
    
    Args:
      state: State dictionary to update
      eps_mult: Optional epsilon multiplier
      
    Returns:
      Updated state dictionary with mup_lrs_to_use
    """
    if self.mup_state is None:
      if state == {}:
        raise ValueError("State is empty, cannot get mup state from it")

      device = jax.devices()[jax.process_index()]
      print(device)
      self.mup_state = get_mup_lrs_from_state(state)
      self.mup_state = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), self.mup_state)
      if eps_mult is not None:
        self.mup_eps_mult = {'eps_mult':jax.device_put(jnp.array(eps_mult), device)}

    state['mup_lrs_to_use'] = self.mup_state
    if eps_mult is not None:
      state['eps_mult'] = self.mup_eps_mult
    return state

  def get_completed_p_state(self, params: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Add all CompletedP scales to the state dictionary.
    
    This computes and caches all CompletedP scaling factors, then adds them
    to the state dictionary for use by learned optimizers.
    
    Args:
      params: Model parameters (needed for computing tensor types)
      state: State dictionary to update
      
    Returns:
      Updated state dictionary with all CompletedP scales
    """
    if self.completed_p_scales is None:
      self.completed_p_scales = self._compute_completed_p_scales(params)
    
    # Add all scales to state
    state.update(self.completed_p_scales)
    
    return state

  
  def init_mup_state(self): 
    """Initialize muP state by calling init_with_state and discarding results.
    
    This creates and saves mup state outside of jit for later use.
    Also sets model multipliers for CompletedP parameterization.
    """
    key = jax.random.PRNGKey(0)
    params, state = self.init_with_state(key)
    
    # Set forward pass multipliers for CompletedP (must be done outside JIT)
    # This modifies self.flax_module which would cause tracer leaks inside JIT
    self._set_model_multipliers()
    
    del params
    del state
        
    # Force garbage collection in a separate thread to make it non-blocking
    gc_thread = threading.Thread(target=gc.collect)
    gc_thread.start()
    # gc_thread.join()  # Optionally wait for the GC to complete
