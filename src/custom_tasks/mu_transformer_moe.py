"""Transformer Decoder-only model."""
# pylint: disable=g-importing-member
# pylint: disable=invalid-name
import dataclasses
import functools
import numpy as np
import os
import sys
from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union, Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.random import PRNGKey

from learned_optimization.tasks import base

if __name__ == "__main__":
  from mu_task_base import MuTask
  from mu_moe_mlp import MoeBlock, nd_dense_init
  from rope import apply_rope
  from parameterization import ModuleType, TensorType
else:

  from .mu_task_base import MuTask
  from .mu_moe_mlp import MoeBlock, nd_dense_init
  from .rope import apply_rope
  from parameterization import ModuleType, TensorType


Params = Any
ModelState = Any
Batch = Any
MoeConfig = Any
Mesh = jax.sharding.Mesh

from jax import lax
from flax.linen import initializers
from custom_tasks.blockwise_attention import blockwise_flash_attention
from custom_tasks.fused_rmsnorm_linear import fused_rmsnorm_linear

if __name__ == "__main__":
  from blockwise_attention import blockwise_flash_attention
  from fused_rmsnorm_linear import fused_rmsnorm_linear
else:
  from .blockwise_attention import blockwise_flash_attention
  from .fused_rmsnorm_linear import fused_rmsnorm_linear

Initializer = Callable[[PRNGKey, Sequence[int], Any], Any]


class KernelParam(nn.Module):
  """Holds a bare kernel parameter without computing a matmul.

  Creates a param at <name>/kernel matching nn.Dense/DenseGeneral naming,
  enabling fused matmuls while preserving the parameter tree for muP/CompletedP.
  """
  shape: Sequence[int]
  kernel_init: Initializer
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self):
    return self.param('kernel', self.kernel_init, self.shape, self.dtype)


class RMSNorm(nn.Module):
  """RMS normalization."""

  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  kernel_axes: Tuple[Optional[str], ...] = ()
  scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = self.param(
        "scale",
        self.scale_init,
        (features,),
        self.weight_dtype,
    )

    scale = jnp.asarray(scale, self.dtype)
    return y * scale


def create_document_relative_positions(tokens, pad_token_id=0):
  """
  Create position indices that reset at each document boundary.
  
  For packed sequences where documents are separated by padding,
  each document gets positions starting from 0.
  
  Args:
    tokens: [batch, length] token ids
    pad_token_id: Token id used for padding (default 0)
  
  Returns:
    positions: [batch, length] position indices, resetting at document boundaries
  
  Example:
    tokens = [3, 5, 7, 0, 0, 2, 4, 6, 0, 1, 8]
    positions = [0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 1]
                 |doc 1  |pad  |doc 2    |p |doc3 |
  """
  B, L = tokens.shape
  
  # Create indicator of non-padding positions
  non_padding = (tokens != pad_token_id).astype(jnp.int32)  # [B, L]
  
  # Shift tokens right to check if previous was padding (or is first position)
  padded_tokens = jnp.pad(tokens, ((0, 0), (1, 0)), constant_values=pad_token_id)[:, :-1]
  is_after_padding = (padded_tokens == pad_token_id).astype(jnp.int32)  # [B, L]
  
  # At document boundaries, reset the position counter
  # Create a position counter that increments within documents
  positions = jnp.arange(L)[None, :]  # [1, L] - global positions
  positions = jnp.broadcast_to(positions, (B, L))  # [B, L]
  
  # Find where documents start (after padding or at beginning)
  doc_starts = is_after_padding & non_padding  # [B, L]
  
  # Get the starting position of each document segment
  # For each position, find the most recent document start
  doc_start_positions = jnp.where(doc_starts, positions, 0)  # [B, L]
  
  # Use cumulative maximum to propagate document start positions forward
  doc_start_positions = jnp.maximum.accumulate(doc_start_positions, axis=1)  # [B, L]
  
  # Relative positions = current position - document start position
  relative_positions = positions - doc_start_positions  # [B, L]
  
  # Zero out positions at padding tokens
  relative_positions = relative_positions * non_padding  # [B, L]
  
  return relative_positions




@dataclasses.dataclass
class DoConfig:
  """Hyper-parameters for Transformer decoder-only."""
  D: int  # model/embed dim  = qkv dim
  H: int  # num attention heads
  L: int  # max context/sequence length (move out of config?)
  N: int  # number of transformer block layers
  V: int  # vocab size
  F: int  # FF inner dimension
  use_qk_norm: bool = False
  ffn_type: str = 'regular_ffn'
  moe_config: MoeConfig = None
  dtype: jnp.dtype = jnp.float32
  fsdp_enabled: bool = False

  # Transformer block rematerialization / gradient checkpointing to save memory.
  remat: bool = False


  # Attention implementation:
  #   'cudnn'     = cuDNN flash O(n), 1.5x faster than xla. Requires patched JAX
  #                 (patches/jax_cudnn_vmap_bwd_fix.patch) for vmap(grad(...)) support.
  #   'xla'       = naive O(n²), always works but slower
  #   'blockwise' = pure-JAX flash O(n), fully vmap-safe but slower than xla at seq<=2048
  attention_impl: str = 'cudnn'

  # Fused RMSNorm+Linear: eliminates intermediate HBM write/read of normalized
  # activations by recomputing x_norm in backward from saved (x, rms_inv).

  use_fused_norm: bool = True

  # MuP (Maximal Update Parametrization) configuration
  use_mup: bool = False  # Use MuP initialization and scaling
  completep_alpha: float = 1.0
  mup_base_width: int = 64
  
  # Numerical stability
  epsilon: float = None  # Global epsilon for numerical stability (set to 1e-10/H if use_mup=True, 1e-5 otherwise)
  
  # RoPE (Rotary Position Embeddings) configuration
  use_rope: bool = True  # Use RoPE instead of learned positional embeddings
  rope_max_wavelength: int = 10_000  # Maximum wavelength for RoPE
  rope_scale_factor: float = 1.0  # Scale factor for RoPE
  pad_token_id: int = 0  # Padding token ID for document boundary detection in RoPE
  
  # Centralized initialization configurations
  # Input layer initializations
  embed_init: nn.initializers.Initializer = None  # Token/position embeddings
  
  # Hidden layer initializations
  hidden_kernel_layer_init: nn.initializers.Initializer = None  # Attention Q/K/V/O projections
  hidden_kernel_init_std: nn.initializers.Initializer = None  # MLP/FFN projections
  moe_kernel_init: Callable = None  # MoE expert projections
  
  # Normalization layer initialization
  norm_scale_init: nn.initializers.Initializer = None  # RMSNorm scale parameters
  
  # Output layer initialization
  output_proj_init: nn.initializers.Initializer = None  # Final output projection
  
  def __post_init__(self):
    """Set default initializers and epsilon if not provided."""
    # Set global epsilon for numerical stability

    self.depth = self.N
    self.width = self.D

    device = jax.devices()[jax.process_index()]

    self.multipliers = {
      'mha_residual_mult': jnp.array(1.0, dtype=jnp.float32, device=device),
      'mlp_residual_mult': jnp.array(1.0, dtype=jnp.float32, device=device),
      'output_mult': jnp.array(1.0, dtype=jnp.float32, device=device),  # Unaugmented
    }
    self.completep_depth_lr_scaling = self.N ** (self.completep_alpha - 1)
    self.completep_residual_scaling = (1 / self.N) ** (self.completep_alpha)
    self.mup_width_scaling = self.mup_base_width / self.D
    self.epsilon = 1e-5 * self.mup_base_width / self.D
    
    if self.use_mup:
      init_base = 0.02 * self.mup_base_width
      # MuP initialization scheme: variance scaling with fan_in
      if self.embed_init is None:
        self.embed_init = nn.initializers.variance_scaling(init_base, "fan_in", "truncated_normal")
      if self.hidden_kernel_init_std is None:
        self.hidden_kernel_init_std = nn.initializers.variance_scaling(init_base, "fan_in", "truncated_normal")
      if self.hidden_kernel_layer_init is None:
        self.hidden_kernel_layer_init = nn.initializers.variance_scaling(init_base, "fan_in", "truncated_normal")
      if self.moe_kernel_init is None:
        self.moe_kernel_init = lambda: nd_dense_init(init_base, "fan_in", "truncated_normal")
      if self.norm_scale_init is None:
        self.norm_scale_init = nn.initializers.ones  # Identity initialization for RMSNorm
      if self.output_proj_init is None:
        self.output_proj_init = nn.initializers.zeros
    else:
      # Standard (TorchTitan) initialization scheme: fixed std of 0.02
      self.epsilon = 1e-5

      weight_init_std = 0.02 / (2 * self.N) ** 0.5
      std_init = nn.initializers.truncated_normal(stddev=0.02)
      std_weight_init = nn.initializers.truncated_normal(stddev=weight_init_std)

      if self.embed_init is None:
        self.embed_init = nn.initializers.normal(stddev=1.0)
      if self.hidden_kernel_init_std is None:
        self.hidden_kernel_init_std = std_init
      if self.hidden_kernel_layer_init is None:
        self.hidden_kernel_layer_init = std_weight_init
      if self.moe_kernel_init is None:
        self.moe_kernel_init = lambda: nd_dense_init(0.02, "fan_in", "normal")
      if self.norm_scale_init is None:
        self.norm_scale_init = nn.initializers.ones  # Identity initialization for RMSNorm
      if self.output_proj_init is None:
        final_out_std = self.D**-0.5
        cutoff_factor = 3
        self.output_proj_init = nn.initializers.truncated_normal(
            stddev=final_out_std,
            lower=-cutoff_factor * final_out_std,
            upper=cutoff_factor * final_out_std
        )


class TransformerDo(nn.Module):
  """Transformer decoder-only."""
  docfg: DoConfig



  def update_multipliers(self, 
                         mha_residual_mult: Optional[float] = None,
                         mlp_residual_mult: Optional[float] = None,
                         output_mult: Optional[float] = None,
                         device = None):
    """Updates the residual scaling multipliers used by all sub-modules.
    
    This method allows dynamic updating of the multipliers that control
    residual connection scaling throughout the transformer. All sub-modules
    (TBlock, CausalAttn) share the same docfg reference, so updates here
    affect the entire model.
    
    Args:
        mha_residual_mult: Multiplier for multi-head attention residual connections.
                          Used in TBlock: x_BxLxD += in_BxLxD * mha_residual_mult
        mlp_residual_mult: Multiplier for MLP/FFN residual connections.
                          Used in TBlock: x_BxLxD + z_BxLxD * mlp_residual_mult
        output_mult: Multiplier for attention output projection.
                    Used in CausalAttn: out_BxLxD * output_mult
        device: JAX device for creating arrays. If None, uses first device.
        
    Returns:
        self for method chaining.
        
    Example:
        model.update_multipliers(
            mha_residual_mult=0.5,
            mlp_residual_mult=0.5,
            output_mult=1.0
        )
    """
    if device is None:
      device = jax.devices()[jax.process_index()]
    
    if mha_residual_mult is not None:
      self.docfg.multipliers['mha_residual_mult'] = jnp.array(
          mha_residual_mult, dtype=jnp.float32, device=device)
    
    if mlp_residual_mult is not None:
      self.docfg.multipliers['mlp_residual_mult'] = jnp.array(
          mlp_residual_mult, dtype=jnp.float32, device=device)
    
    if output_mult is not None:
      self.docfg.multipliers['output_mult'] = jnp.array(
          output_mult, dtype=jnp.float32, device=device)
    
    return self

  def get_multipliers(self):
    """Returns the current multiplier values as a dictionary.
    
    Returns:
        Dictionary with keys 'mha_residual_mult', 'mlp_residual_mult', 'output_mult'
        containing the current JAX array values.
    """
    return dict(self.docfg.multipliers)

  def set_multipliers_from_dict(self, multipliers: dict, device=None):
    """Sets all multipliers from a dictionary.
    
    Args:
        multipliers: Dictionary with keys 'mha_residual_mult', 'mlp_residual_mult', 
                    'output_mult'. Missing keys are ignored.
        device: JAX device for creating arrays. If None, uses first device.
        
    Returns:
        self for method chaining.
    """
    return self.update_multipliers(
        mha_residual_mult=multipliers.get('mha_residual_mult'),
        mlp_residual_mult=multipliers.get('mlp_residual_mult'),
        output_mult=multipliers.get('output_mult'),
        device=device
    )

  def get_mup_lrs(self, params, device):
    """Returns the MuP learning rate multipliers that match the parameter structure."""
    def get_dense(v, fan_in):
      assert 'kernel' in v, f"Expected 'kernel' key in v, but got keys: {list(v.keys())}"
      lr = self.docfg.completep_depth_lr_scaling * self.docfg.mup_base_width / fan_in
      return jax.tree_util.tree_map(lambda x: jnp.array(lr, dtype=jnp.float32, device=device), v)

    def get_norm(v, lr=1.0):
      # Norm parameters (RMSNorm) always have lr 1.0
      return jax.tree_util.tree_map(lambda x: jnp.array(lr, dtype=jnp.float32, device=device), v)
    
    def get_attention(v):
      # For attention layers, use 1/fan_in for all kernels
      result = {}
      for k, val in v.items():
        if k == 'attn_out_proj':
          # For output projection, fan_in is H*head_dim
          fan_in = val['kernel'].shape[0] * val['kernel'].shape[1]
          lr = self.docfg.completep_depth_lr_scaling * self.docfg.mup_base_width / fan_in
          result[k] = {'kernel': jnp.array(lr, dtype=jnp.float32, device=device)}
        elif k.startswith('RMSNorm'):
          # RMS norm parameters for key and query normalization
          result[k] = get_norm(val, lr=self.docfg.completep_depth_lr_scaling)
        else:  # query, key, value
          fan_in = val['kernel'].shape[0]  # embedding dimension
          lr = self.docfg.completep_depth_lr_scaling * self.docfg.mup_base_width / fan_in
          result[k] = {'kernel': jnp.array(lr, dtype=jnp.float32, device=device)}
      return result
    
    def get_moe(v):
      # For MoE blocks
      result = {}
      
      # Gate kernel - use 1/fan_in
      if 'gate' in v:
        fan_in = getattr(v['gate']['kernel'], 'value', v['gate']['kernel']).shape[0]
        lr = self.docfg.completep_depth_lr_scaling * self.docfg.mup_base_width / fan_in
        result['gate'] = {'kernel': jax.tree_util.tree_map(
          lambda x: jnp.array(lr, dtype=jnp.float32, device=device), 
          v['gate']['kernel'])}
      
      # Expert weights - use 1/fan_in
      for key in ['wi_0', 'wi_1', 'wo']:
        if key in v:
          if key.startswith('wi'):
            fan_in = getattr(v[key], 'value', v[key]).shape[1]  # embedding dimension
          else:  # wo
            fan_in = getattr(v[key], 'value', v[key]).shape[1]  # mlp dimension
          lr = self.docfg.completep_depth_lr_scaling * self.docfg.mup_base_width / fan_in
          result[key] = jax.tree_util.tree_map(
            lambda x: jnp.array(lr, dtype=jnp.float32, device=device), 
            v[key])
      
      return result

    # Start with all learning rates at 1.0
    lr_tree = jax.tree_util.tree_map(
      lambda x: jnp.array(-1.0, dtype=jnp.float32, device=device), 
      params['params'])
    
    # Process each parameter group
    for k, v in params['params'].items():
      if k == 'embed' or k == 'pos_embed' or k == 'output_proj':
        # Embeddings always have lr 1.0
        lr_tree[k] = jax.tree_util.tree_map(lambda x: jnp.array(1.0, dtype=jnp.float32, device=device), v)
      
      elif k.startswith('blocks_'):
        # Process transformer blocks
        for block_k, block_v in v.items():
          if 'CausalAttn' in block_k:
            lr_tree[k][block_k] = get_attention(block_v)
          elif 'RMSNorm' in block_k:
            lr_tree[k][block_k] = get_norm(block_v, lr=self.docfg.completep_depth_lr_scaling)
          elif 'MoeBlock' in block_k:
            lr_tree[k][block_k] = get_moe(block_v)
          elif 'MlpSwiGLU' in block_k or 'Mlp' in block_k:
            # Handle SwiGLU MLP blocks
            result = {}
            for mlp_k, mlp_v in block_v.items():
              if 'Dense' in mlp_k:
                fan_in = mlp_v['kernel'].shape[0]
                result[mlp_k] = get_dense(mlp_v, fan_in)
            lr_tree[k][block_k] = result
          elif 'Dense' in block_k:
            fan_in = block_v['kernel'].shape[0]
            lr_tree[k][block_k] = get_dense(block_v, fan_in)
      
      elif k == 'out_ln':
        # Output RMS norm
        lr_tree[k] = get_norm(v, lr=1.0)
    
    # Assert that no learning rates remain -1.0 (unprocessed)
    def check_no_negative_lrs(tree):
      leaves = jax.tree_util.tree_leaves(tree)
      for leaf in leaves:
        if jnp.any(leaf < 0):
          return False
      return True
    
    assert check_no_negative_lrs(lr_tree), "Some learning rates were not properly set (remain -1.0)"
    
    return {"params": lr_tree}



  def get_mup_epsilons(self, params, device):
    """Returns the MuP epsilon multipliers that match the parameter structure."""
    # from https://openreview.net/pdf?id=elB9k4nTL1 (Table 1)
    hidden_weights_biases_norms = lambda w :(self.docfg.N ** (-self.docfg.completep_alpha)) * self.docfg.mup_base_width / w
    qk_norms = lambda w : (self.docfg.N ** (-self.docfg.completep_alpha))
    input_emb = lambda w : self.docfg.mup_base_width / w
    output_weights_biases_norms = lambda w : 1.0
    
    make_jnp_array = lambda x : jnp.array(x, dtype=jnp.float32, device=device)

    map_param_to_eps = lambda rescale_fun, width, param: jax.tree_util.tree_map(
      lambda x: make_jnp_array(rescale_fun(width)), param)

    hidden_weights_biases_norms_eps = partial(map_param_to_eps, rescale_fun=hidden_weights_biases_norms)
    qk_norms_eps = partial(map_param_to_eps, rescale_fun=qk_norms)
    input_emb_eps = partial(map_param_to_eps, rescale_fun=input_emb)
    output_weights_biases_norms_eps = partial(map_param_to_eps, rescale_fun=output_weights_biases_norms)


    def get_attention(v):
      # For attention layers, use 1/fan_in for all kernels
      result = {}
      for k, val in v.items():
        if k == 'attn_out_proj':
          # For output projection, fan_in is H*head_dim
          fan_in = val['kernel'].shape[0] * val['kernel'].shape[1]
          result[k] = hidden_weights_biases_norms_eps(width=fan_in, param=val)
        elif k.startswith('RMSNorm'):
          # RMS norm parameters for key and query normalization
          result[k] = qk_norms_eps(width=self.docfg.H, param=val)
        else:  # query, key, value
          fan_in = val['kernel'].shape[0]  
          result[k] = hidden_weights_biases_norms_eps(width=fan_in, param=val)
      return result
    
    def get_moe(v):
      # For MoE blocks
      result = {}
      
      # Gate kernel - use 1/fan_in
      if 'gate' in v:
        fan_in = getattr(v['gate']['kernel'], 'value', v['gate']['kernel']).shape[0]
        result['gate'] = {'kernel': hidden_weights_biases_norms_eps(width=fan_in, param=v['gate']['kernel'])}
      
      # Expert weights - use 1/fan_in
      for key in ['wi_0', 'wi_1', 'wo']:
        if key in v:
          if key.startswith('wi'):
            fan_in = getattr(v[key], 'value', v[key]).shape[1]  # embedding dimension
          else:  # wo
            fan_in = getattr(v[key], 'value', v[key]).shape[1]  # mlp dimension

          result[key] = hidden_weights_biases_norms_eps(width=fan_in, param=v[key])
          
      return result

    # Start with all learning rates at 1.0
    lr_tree = jax.tree_util.tree_map(
      lambda x: jnp.array(-1.0, dtype=jnp.float32, device=device), 
      params['params'])
    
    # Process each parameter group
    for k, v in params['params'].items():
      if k == 'embed' or k == 'pos_embed' or k == 'output_proj':
        # Embeddings always have lr 1.0
        lr_tree[k] = jax.tree_util.tree_map(lambda x: jnp.array(1.0, dtype=jnp.float32, device=device), v)
      
      elif k.startswith('blocks_'):
        # Process transformer blocks
        for block_k, block_v in v.items():
          if 'CausalAttn' in block_k:
            lr_tree[k][block_k] = get_attention(block_v)
          elif 'RMSNorm' in block_k:
            lr_tree[k][block_k] = output_weights_biases_norms_eps(width=self.docfg.H, param=block_v)
          elif 'MoeBlock' in block_k:
            lr_tree[k][block_k] = get_moe(block_v)
          elif 'MlpSwiGLU' in block_k or 'Mlp' in block_k:
            # Handle SwiGLU MLP blocks
            result = {}
            for mlp_k, mlp_v in block_v.items():
              if 'Dense' in mlp_k:
                fan_in = mlp_v['kernel'].shape[0]
                result[mlp_k] = hidden_weights_biases_norms_eps(width=fan_in, param=mlp_v)

            lr_tree[k][block_k] = result
          elif 'Dense' in block_k:
            fan_in = block_v['kernel'].shape[0]
            lr_tree[k][block_k] = hidden_weights_biases_norms_eps(width=fan_in, param=block_v)
      
      elif k == 'out_ln':
        # Output RMS norm
        lr_tree[k] = output_weights_biases_norms_eps(width=self.docfg.H, param=v)
    
    # Assert that no learning rates remain -1.0 (unprocessed)
    def check_no_negative_lrs(tree):
      leaves = jax.tree_util.tree_leaves(tree)
      for leaf in leaves:
        if jnp.any(leaf < 0):
          return False
      return True
    
    assert check_no_negative_lrs(lr_tree), "Some learning rates were not properly set (remain -1.0)"
    
    return {"params": lr_tree}


  def get_muon_weight_dimension_numbers(self, params):
    """Returns a pytree of MuonDimensionNumbers for use with optax.contrib.muon.
    
    This method creates the muon_weight_dimension_numbers argument that specifies
    which parameters should be optimized with Muon and how to reshape them.
    
    Muon is applied to weight matrices (kernels), while embeddings, biases, and 
    normalization parameters are handled by AdamW (marked as None).
    
    For attention layers:
      - Q/K/V kernels: shape (D, H, Dh) -> reshape to (D, H*Dh) for Muon
      - Output proj kernel: shape (H, Dh, D) -> reshape to (H*Dh, D) for Muon
    
    For MLP layers:
      - Dense kernels: 2D matrices use default Muon dimension numbers
    
    Args:
        params: The model parameters dictionary with 'params' key
        
    Returns:
        A pytree matching params structure with MuonDimensionNumbers for Muon-optimized
        parameters and None for AdamW-optimized parameters.
    """
    import optax
    
    # Default 2D dimension numbers (reduction_axis=0, output_axis=1)
    DEFAULT_2D = optax.contrib.MuonDimensionNumbers((0,), (1,))
    
    def get_dense_muon(v):
      """Get Muon dimension numbers for dense layers (2D kernels)."""
      result = {}
      for k, val in v.items():
        if k == 'kernel':
          # 2D kernel: use default Muon
          result[k] = DEFAULT_2D
        else:
          # Bias or other params: use Adam
          result[k] = None
      return result
    
    def get_norm_muon(v):
      """Normalization layers always use Adam (None)."""
      return jax.tree_util.tree_map(lambda x: None, v)
    
    def get_embed_muon(v):
      """Embedding layers use Adam (None)."""
      return jax.tree_util.tree_map(lambda x: None, v)
    
    def get_attention_muon(v):
      """Get Muon dimension numbers for attention layers.
      
      Attention kernels are 3D and need special handling:
        - Q/K/V: (D, H, Dh) -> reduction_axis=(0,), output_axis=(1, 2) treats as (D, H*Dh)
        - Out proj: (H, Dh, D) -> reduction_axis=(0, 1), output_axis=(2,) treats as (H*Dh, D)
      """
      result = {}
      for k, val in v.items():
        if k == 'attn_out_proj':
          # Output projection kernel: (H, Dh, D) -> (H*Dh, D)
          result[k] = {'kernel': optax.contrib.MuonDimensionNumbers(
            reduction_axis=(0, 1), 
            output_axis=(2,)
          )}
        elif k.startswith('RMSNorm'):
          # QK norm parameters: use Adam
          result[k] = get_norm_muon(val)
        else:  # query, key, value
          # Q/K/V kernels: (D, H, Dh) -> (D, H*Dh)
          result[k] = {'kernel': optax.contrib.MuonDimensionNumbers(
            reduction_axis=(0,), 
            output_axis=(1, 2)
          )}
      return result
    
    def get_moe_muon(v):
      """Get Muon dimension numbers for MoE blocks.
      
      MoE has:
        - gate: 2D kernel -> default Muon
        - wi_0, wi_1, wo: 3D expert weights (num_experts, in, out) 
          -> treat experts as batch dim, (in, out) for Muon
      """
      result = {}
      
      # Gate kernel - 2D, use default Muon
      if 'gate' in v:
        result['gate'] = {'kernel': jax.tree_util.tree_map(
          lambda x: DEFAULT_2D, 
          v['gate']['kernel']
        )}
      
      # Expert weights - 3D (num_experts, in_dim, out_dim)
      # Treat num_experts as batch dimension, apply Muon to (in_dim, out_dim)
      for key in ['wi_0', 'wi_1', 'wo']:
        if key in v:
          # Shape is (num_experts, fan_in, fan_out) - axis 0 is batch
          result[key] = jax.tree_util.tree_map(
            lambda x: optax.contrib.MuonDimensionNumbers(
              reduction_axis=(1,), 
              output_axis=(2,)
            ), 
            v[key]
          )
      
      return result
    
    # Initialize with None to track unprocessed parameters
    muon_tree = jax.tree_util.tree_map(lambda x: None, params['params'])
    
    # Process each parameter group
    for k, v in params['params'].items():
      if k == 'embed' or k == 'pos_embed':
        # Embeddings use Adam
        muon_tree[k] = get_embed_muon(v)
      
      elif k == 'output_proj':
        # Output projection is 2D, could use Muon
        muon_tree[k] = get_embed_muon(v)
      
      elif k.startswith('blocks_'):
        # Process transformer blocks
        for block_k, block_v in v.items():
          if 'CausalAttn' in block_k:
            muon_tree[k][block_k] = get_attention_muon(block_v)
          elif 'RMSNorm' in block_k:
            muon_tree[k][block_k] = get_norm_muon(block_v)
          elif 'MoeBlock' in block_k:
            muon_tree[k][block_k] = get_moe_muon(block_v)
          elif 'MlpSwiGLU' in block_k or 'Mlp' in block_k:
            # Handle SwiGLU MLP blocks
            result = {}
            for mlp_k, mlp_v in block_v.items():
              if 'Dense' in mlp_k:
                result[mlp_k] = get_dense_muon(mlp_v)
            muon_tree[k][block_k] = result
          elif 'Dense' in block_k:
            muon_tree[k][block_k] = get_dense_muon(block_v)
      
      elif k == 'out_ln':
        # Output RMS norm uses Adam
        muon_tree[k] = get_norm_muon(v)
    
    return {"params": muon_tree}

  def get_module_types(self, params):
    """Returns a pytree of (ModuleType, (fan_in, fan_out)) tuples matching the parameter structure.
    
    This method classifies each parameter by its module type and extracts fan_in/fan_out
    dimensions for use with the CompletedP parameterization scaling functions.
    
    Args:
        params: The model parameters dictionary with 'params' key
        
    Returns:
        A pytree with the same structure as params, where each leaf is a tuple of
        (ModuleType, (fan_in, fan_out))
    """
    
    def get_dense_info(v, fan_in):
      """Get module info for dense layers."""
      assert 'kernel' in v, f"Expected 'kernel' key in v, but got keys: {list(v.keys())}"
      fan_out = v['kernel'].shape[-1]
      return jax.tree_util.tree_map(
        lambda x: (ModuleType.HIDDEN_WEIGHT, (fan_in, x.shape[-1] if len(x.shape) > 1 else fan_in)), 
        v
      )
    
    def get_norm_info(v, module_type=ModuleType.HIDDEN_NORM):
      """Get module info for normalization layers."""
      def get_leaf_info(x):
        fan = x.shape[0] if len(x.shape) > 0 else 1
        return (module_type, (fan, fan))
      return jax.tree_util.tree_map(get_leaf_info, v)
    
    def get_embed_info(v, module_type=ModuleType.INPUT_EMBED):
      """Get module info for embedding layers."""
      def get_leaf_info(x):
        if len(x.shape) >= 2:
          return (module_type, (x.shape[0], x.shape[1]))
        else:
          fan = x.shape[0] if len(x.shape) > 0 else 1
          return (module_type, (fan, fan))
      return jax.tree_util.tree_map(get_leaf_info, v)
    
    def get_attention_info(v):
      """Get module info for attention layers."""
      result = {}
      for k, val in v.items():
        if k == 'attn_out_proj':
          # For output projection, fan_in is H*head_dim
          fan_in = val['kernel'].shape[0] * val['kernel'].shape[1]
          fan_out = val['kernel'].shape[-1]
          result[k] = {'kernel': (ModuleType.HIDDEN_WEIGHT, (fan_in, fan_out))}
        elif k.startswith('RMSNorm'):
          # QK norms - special type
          result[k] = get_norm_info(val, module_type=ModuleType.QK_NORM)
        else:  # query, key, value
          fan_in = val['kernel'].shape[0]
          fan_out = val['kernel'].shape[-1] if len(val['kernel'].shape) > 1 else fan_in
          result[k] = {'kernel': (ModuleType.HIDDEN_WEIGHT, (fan_in, fan_out))}
      return result
    
    def get_moe_info(v):
      """Get module info for MoE blocks."""
      result = {}
      
      # Gate kernel
      if 'gate' in v:
        fan_in = getattr(v['gate']['kernel'], 'value', v['gate']['kernel']).shape[0]
        fan_out = getattr(v['gate']['kernel'], 'value', v['gate']['kernel']).shape[-1]
        result['gate'] = {'kernel': jax.tree_util.tree_map(
          lambda x: (ModuleType.HIDDEN_WEIGHT, (fan_in, fan_out)), 
          v['gate']['kernel']
        )}
      
      # Expert weights
      for key in ['wi_0', 'wi_1', 'wo']:
        if key in v:
          if key.startswith('wi'):
            fan_in = getattr(v[key], 'value', v[key]).shape[1]  # embedding dimension
            fan_out = getattr(v[key], 'value', v[key]).shape[-1]
          else:  # wo
            fan_in = getattr(v[key], 'value', v[key]).shape[1]  # mlp dimension
            fan_out = getattr(v[key], 'value', v[key]).shape[-1]
          result[key] = jax.tree_util.tree_map(
            lambda x: (ModuleType.HIDDEN_WEIGHT, (fan_in, fan_out)), 
            v[key]
          )
      
      return result
    
    def get_output_proj_info(v):
      """Get module info for output projection."""
      def get_leaf_info(x):
        if len(x.shape) >= 2:
          return (ModuleType.UNEMBED_WEIGHT, (x.shape[0], x.shape[-1]))
        else:
          fan = x.shape[0] if len(x.shape) > 0 else 1
          return (ModuleType.UNEMBED_WEIGHT, (fan, fan))
      return jax.tree_util.tree_map(get_leaf_info, v)
    
    # Initialize with None to track unprocessed parameters
    info_tree = jax.tree_util.tree_map(lambda x: None, params['params'])
    
    # Process each parameter group
    for k, v in params['params'].items():
      if k == 'embed':
        # Token embeddings
        info_tree[k] = get_embed_info(v, module_type=ModuleType.INPUT_EMBED)
      
      elif k == 'pos_embed':
        # Positional embeddings
        info_tree[k] = get_embed_info(v, module_type=ModuleType.POS_EMBED)
      
      elif k == 'output_proj':
        # Output projection (unembedding)
        info_tree[k] = get_output_proj_info(v)
      
      elif k.startswith('blocks_'):
        # Process transformer blocks
        for block_k, block_v in v.items():
          if 'CausalAttn' in block_k:
            info_tree[k][block_k] = get_attention_info(block_v)
          elif 'RMSNorm' in block_k:
            # Hidden layer norms
            info_tree[k][block_k] = get_norm_info(block_v, module_type=ModuleType.HIDDEN_NORM)
          elif 'MoeBlock' in block_k:
            info_tree[k][block_k] = get_moe_info(block_v)
          elif 'MlpSwiGLU' in block_k or 'Mlp' in block_k:
            # Handle SwiGLU MLP blocks
            result = {}
            for mlp_k, mlp_v in block_v.items():
              if 'Dense' in mlp_k:
                fan_in = mlp_v['kernel'].shape[0]
                result[mlp_k] = get_dense_info(mlp_v, fan_in)
            info_tree[k][block_k] = result
          elif 'Dense' in block_k:
            fan_in = block_v['kernel'].shape[0]
            info_tree[k][block_k] = get_dense_info(block_v, fan_in)
      
      elif k == 'out_ln':
        # Output layer normalization (unembedding norm)
        info_tree[k] = get_norm_info(v, module_type=ModuleType.UNEMBED_NORM)
    
    # Assert that no parameters remain None (unprocessed)
    def check_no_none(tree):
      leaves = jax.tree_util.tree_leaves(tree)
      for leaf in leaves:
        if leaf is None:
          return False
      return True
    
    assert check_no_none(info_tree), "Some parameters were not properly classified (remain None)"
    
    return {"params": info_tree}

  def get_tensor_types(self, params):
    """Returns a pytree of (TensorType, (fan_in, fan_out)) tuples matching the parameter structure.
    
    This method provides fine-grained tensor type classification for each parameter,
    enabling per-tensor learning rate control. Each TensorType maps to a ModuleType
    which determines the base scaling rules from CompletedP parameterization.
    
    TensorTypes include:
      - EMBEDDING, POS_EMBEDDING: Token and positional embeddings
      - ATTENTION_QUERY, ATTENTION_KEY, ATTENTION_VALUE: Q/K/V projections
      - ATTENTION_OUTPUT: Attention output projection
      - ATTENTION_QUERY_NORM, ATTENTION_KEY_NORM: QK normalization
      - MLP_UP, MLP_GATE, MLP_DOWN: MLP projections
      - POST_ATTENTION_NORM, POST_MLP_NORM: Layer norms
      - OUTPUT_NORM: Final output normalization
      - UNEMBEDDING: Output projection to vocab
      - MOE_GATE, MOE_UP, MOE_GATE_PROJ, MOE_DOWN: MoE components
    
    Args:
        params: The model parameters dictionary with 'params' key
        
    Returns:
        A pytree with the same structure as params, where each leaf is a tuple of
        (TensorType, (fan_in, fan_out))
    """
    
    def get_dense_info(v, fan_in, tensor_type=TensorType.MLP_UP):
      """Get tensor info for dense layers."""
      assert 'kernel' in v, f"Expected 'kernel' key in v, but got keys: {list(v.keys())}"
      return jax.tree_util.tree_map(
        lambda x: (tensor_type, (fan_in, x.shape[-1] if len(x.shape) > 1 else fan_in)), 
        v
      )
    
    def get_norm_info(v, tensor_type=TensorType.POST_ATTENTION_NORM):
      """Get tensor info for normalization layers."""
      def get_leaf_info(x):
        fan = x.shape[0] if len(x.shape) > 0 else 1
        return (tensor_type, (fan, fan))
      return jax.tree_util.tree_map(get_leaf_info, v)
    
    def get_embed_info(v, tensor_type=TensorType.EMBEDDING):
      """Get tensor info for embedding layers."""
      def get_leaf_info(x):
        if len(x.shape) >= 2:
          return (tensor_type, (x.shape[0], x.shape[1]))
        else:
          fan = x.shape[0] if len(x.shape) > 0 else 1
          return (tensor_type, (fan, fan))
      return jax.tree_util.tree_map(get_leaf_info, v)
    
    def get_attention_info(v):
      """Get tensor info for attention layers with fine-grained tensor types."""
      result = {}
      for k, val in v.items():
        if k == 'attn_out_proj':
          # Output projection
          fan_in = val['kernel'].shape[0] * val['kernel'].shape[1]
          fan_out = val['kernel'].shape[-1]
          result[k] = {'kernel': (TensorType.ATTENTION_OUTPUT, (fan_in, fan_out))}
        elif k == 'query':
          fan_in = val['kernel'].shape[0]
          fan_out = val['kernel'].shape[-1] if len(val['kernel'].shape) > 1 else fan_in
          result[k] = {'kernel': (TensorType.ATTENTION_QUERY, (fan_in, fan_out))}
        elif k == 'key':
          fan_in = val['kernel'].shape[0]
          fan_out = val['kernel'].shape[-1] if len(val['kernel'].shape) > 1 else fan_in
          result[k] = {'kernel': (TensorType.ATTENTION_KEY, (fan_in, fan_out))}
        elif k == 'value':
          fan_in = val['kernel'].shape[0]
          fan_out = val['kernel'].shape[-1] if len(val['kernel'].shape) > 1 else fan_in
          result[k] = {'kernel': (TensorType.ATTENTION_VALUE, (fan_in, fan_out))}
        elif k == 'RMSNorm_0':  # Query norm (first norm in attention)
          result[k] = get_norm_info(val, tensor_type=TensorType.ATTENTION_QUERY_NORM)
        elif k == 'RMSNorm_1':  # Key norm (second norm in attention)
          result[k] = get_norm_info(val, tensor_type=TensorType.ATTENTION_KEY_NORM)
        elif k.startswith('RMSNorm'):
          # Generic QK norm handling - determine based on position
          # RMSNorm_0 is typically query, RMSNorm_1 is typically key
          result[k] = get_norm_info(val, tensor_type=TensorType.ATTENTION_QUERY_NORM)
      return result
    
    def get_mlp_info(v, is_swiglu=True):
      """Get tensor info for MLP layers with fine-grained tensor types.
      
      For SwiGLU MLP:
        - Dense_0: MLP_UP (embed -> ffn)
        - Dense_1: MLP_GATE (embed -> ffn, gate branch)
        - Dense_2: MLP_DOWN (ffn -> embed)
      
      For regular MLP:
        - Dense_0: MLP_UP (embed -> ffn)
        - Dense_1: MLP_DOWN (ffn -> embed)
      """
      result = {}
      for mlp_k, mlp_v in v.items():
        if 'Dense' not in mlp_k:
          continue
        fan_in = mlp_v['kernel'].shape[0]
        
        if is_swiglu:
          if mlp_k == 'Dense_0':
            result[mlp_k] = get_dense_info(mlp_v, fan_in, TensorType.MLP_UP)
          elif mlp_k == 'Dense_1':
            result[mlp_k] = get_dense_info(mlp_v, fan_in, TensorType.MLP_GATE)
          elif mlp_k == 'Dense_2':
            result[mlp_k] = get_dense_info(mlp_v, fan_in, TensorType.MLP_DOWN)
          else:
            # Unknown Dense layer, default to MLP_UP
            result[mlp_k] = get_dense_info(mlp_v, fan_in, TensorType.MLP_UP)
        else:
          # Regular MLP
          if mlp_k == 'Dense_0':
            result[mlp_k] = get_dense_info(mlp_v, fan_in, TensorType.MLP_UP)
          elif mlp_k == 'Dense_1':
            result[mlp_k] = get_dense_info(mlp_v, fan_in, TensorType.MLP_DOWN)
          else:
            result[mlp_k] = get_dense_info(mlp_v, fan_in, TensorType.MLP_UP)
      return result
    
    def get_moe_info(v):
      """Get tensor info for MoE blocks with fine-grained tensor types."""
      result = {}
      
      # Gate kernel
      if 'gate' in v:
        fan_in = getattr(v['gate']['kernel'], 'value', v['gate']['kernel']).shape[0]
        fan_out = getattr(v['gate']['kernel'], 'value', v['gate']['kernel']).shape[-1]
        result['gate'] = {'kernel': jax.tree_util.tree_map(
          lambda x: (TensorType.MOE_GATE, (fan_in, fan_out)), 
          v['gate']['kernel']
        )}
      
      # Expert weights
      for key in ['wi_0', 'wi_1', 'wo']:
        if key in v:
          if key == 'wi_0':
            fan_in = getattr(v[key], 'value', v[key]).shape[1]  # embedding dimension
            fan_out = getattr(v[key], 'value', v[key]).shape[-1]
            tensor_type = TensorType.MOE_UP
          elif key == 'wi_1':
            fan_in = getattr(v[key], 'value', v[key]).shape[1]  # embedding dimension
            fan_out = getattr(v[key], 'value', v[key]).shape[-1]
            tensor_type = TensorType.MOE_GATE_PROJ
          else:  # wo
            fan_in = getattr(v[key], 'value', v[key]).shape[1]  # mlp dimension
            fan_out = getattr(v[key], 'value', v[key]).shape[-1]
            tensor_type = TensorType.MOE_DOWN
          
          result[key] = jax.tree_util.tree_map(
            lambda x: (tensor_type, (fan_in, fan_out)), 
            v[key]
          )
      
      return result
    
    def get_output_proj_info(v):
      """Get tensor info for output projection (unembedding)."""
      def get_leaf_info(x):
        if len(x.shape) >= 2:
          return (TensorType.UNEMBEDDING, (x.shape[0], x.shape[-1]))
        else:
          fan = x.shape[0] if len(x.shape) > 0 else 1
          return (TensorType.UNEMBEDDING, (fan, fan))
      return jax.tree_util.tree_map(get_leaf_info, v)
    
    # Initialize with None to track unprocessed parameters
    info_tree = jax.tree_util.tree_map(lambda x: None, params['params'])
    
    # Determine if we're using SwiGLU based on config
    is_swiglu = self.docfg.ffn_type == 'swiglu'
    
    # Process each parameter group
    for k, v in params['params'].items():
      if k == 'embed':
        # Token embeddings
        info_tree[k] = get_embed_info(v, tensor_type=TensorType.EMBEDDING)
      
      elif k == 'pos_embed':
        # Positional embeddings
        info_tree[k] = get_embed_info(v, tensor_type=TensorType.POS_EMBEDDING)
      
      elif k == 'output_proj':
        # Output projection (unembedding)
        info_tree[k] = get_output_proj_info(v)
      
      elif k.startswith('blocks_'):
        # Process transformer blocks
        # Track norm index within block for post-attention vs post-MLP
        norm_idx = 0
        for block_k, block_v in v.items():
          if 'CausalAttn' in block_k:
            info_tree[k][block_k] = get_attention_info(block_v)
          elif 'RMSNorm' in block_k:
            # Block-level RMSNorms: first is post-attention, second is post-MLP
            if norm_idx == 0:
              info_tree[k][block_k] = get_norm_info(block_v, tensor_type=TensorType.POST_ATTENTION_NORM)
            else:
              info_tree[k][block_k] = get_norm_info(block_v, tensor_type=TensorType.POST_MLP_NORM)
            norm_idx += 1
          elif 'MoeBlock' in block_k:
            info_tree[k][block_k] = get_moe_info(block_v)
          elif 'MlpSwiGLU' in block_k or 'Mlp' in block_k:
            # Handle SwiGLU MLP blocks
            info_tree[k][block_k] = get_mlp_info(block_v, is_swiglu='SwiGLU' in block_k or is_swiglu)
          elif 'Dense' in block_k:
            fan_in = block_v['kernel'].shape[0]
            info_tree[k][block_k] = get_dense_info(block_v, fan_in, TensorType.MLP_UP)
      
      elif k == 'out_ln':
        # Output layer normalization
        info_tree[k] = get_norm_info(v, tensor_type=TensorType.OUTPUT_NORM)
    
    # Assert that no parameters remain None (unprocessed)
    def check_no_none(tree):
      leaves = jax.tree_util.tree_leaves(tree)
      for leaf in leaves:
        if leaf is None:
          return False
      return True
    
    assert check_no_none(info_tree), "Some parameters were not properly classified (remain None)"
    
    return {"params": info_tree}

  def setup(self):
    cfg = self.docfg
    
    # Input layer: Token embeddings
    self.embed = nn.Embed(
        num_embeddings=cfg.V,
        features=cfg.D,
        embedding_init=cfg.embed_init,
    )
    
    # Positional embeddings: learned or RoPE
    if not cfg.use_rope:
      # Use learned positional embeddings
      self.pos_embed = nn.Embed(
          num_embeddings=cfg.L,
          features=cfg.D,
          embedding_init=cfg.embed_init,
      )

    # Optional output projection for untied weights
    self.tie_weights = getattr(cfg, 'tie_weights', False)
    if not self.tie_weights:
      self.output_proj = nn.Dense(
          features=cfg.V,
          use_bias=False,
          kernel_init=cfg.output_proj_init,
          dtype=cfg.dtype,
          name='output_proj'
      )

    # Hidden layers: Transformer blocks
    block = nn.remat(TBlock) if cfg.remat else TBlock
    self.blocks = [block(cfg) for _ in range(cfg.N)]
    
    # Output RMS norm - Use centralized norm initialization and epsilon
    self.out_ln = RMSNorm(dtype=cfg.dtype, scale_init=cfg.norm_scale_init, epsilon=cfg.epsilon)

  # def get_mup_lrs(self, params, device):
  #   return jax.tree_util.tree_map(lambda x: jnp.array([1.0], device=device), params)

  def __call__(self, y_BxL: jax.Array, attention_mask: Optional[jax.Array] = None):
    """
    Args:
      y_BxL: Token ids [batch, length]
      attention_mask: Optional [batch, length, length] mask for document boundaries
                     (used for masking but positions are derived from tokens for RoPE)
    """
    cfg = self.docfg
    
    # Token embeddings
    y_BxLxD = self.embed(y_BxL)
    
    # Add positional information (learned embeddings or pass positions for RoPE)
    if not cfg.use_rope:
      # Use learned positional embeddings (global positions)
      y_BxLxD += self.pos_embed(jnp.arange(0, y_BxL.shape[1])[None, ...])
      positions = None
    else:
      # RoPE: compute document-relative positions that reset at each document boundary
      # This ensures each document starts from position 0, which is important for
      # packed sequences where multiple documents are concatenated
      positions = create_document_relative_positions(y_BxL, pad_token_id=cfg.pad_token_id)  # [B, L]
    
    # Track load balance losses
    load_balance_losses = []
    
    for i, block in enumerate(self.blocks):
      y_BxLxD, lbl_loss = block(y_BxLxD, attention_mask=attention_mask, positions=positions)

      load_balance_losses.append(lbl_loss)
      self.sow("intermediates", f"layer_{i}_load_balance_loss", lbl_loss)
    
    # Calculate total load balance loss if any
    total_load_balance_loss = sum(load_balance_losses)
    self.sow("intermediates", "load_balance_loss", total_load_balance_loss)
    
    y_BxLxD = self.out_ln(y_BxLxD)
    
    # Use either tied weights (embedding.attend) or separate output projection
    if self.tie_weights:
      logits_BxLxV = self.embed.attend(y_BxLxD.astype(jnp.float32))
    else:
      logits_BxLxV = self.output_proj(y_BxLxD)
      
    return logits_BxLxV


class Mlp(nn.Module):
  """Multilayer perceptron with raw kernel matmuls."""
  cfg: DoConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array):
    cfg = self.cfg
    D = x_BxLxD.shape[-1]
    up_kernel = KernelParam(shape=(D, cfg.F), kernel_init=cfg.hidden_kernel_init_std, dtype=cfg.dtype, name='Dense_0')()
    down_kernel = KernelParam(shape=(cfg.F, cfg.D), kernel_init=cfg.hidden_kernel_layer_init, dtype=cfg.dtype, name='Dense_1')()
    x_BxLxF = jax.nn.gelu(x_BxLxD @ up_kernel)
    x_BxLxD = x_BxLxF @ down_kernel
    return x_BxLxD


class MlpSwiGLU(nn.Module):
  """Multilayer perceptron with SwiGLU activation.

  Uses fused up+gate matmul (2 kernel launches → 1) while preserving
  separate Dense_0/kernel, Dense_1/kernel params for muP compatibility.

  When use_fused_norm=True in cfg, accepts pre-normalized input and norm_scale
  to fuse RMSNorm + up+gate projection via custom_vjp (saves HBM bandwidth).
  """
  cfg: DoConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array, pre_norm_input: jax.Array = None,
               norm_scale: jax.Array = None):
    cfg = self.cfg
    D = x_BxLxD.shape[-1]

    # Create params matching original Dense_0/kernel, Dense_1/kernel, Dense_2/kernel paths
    up_kernel = KernelParam(shape=(D, cfg.F), kernel_init=cfg.hidden_kernel_layer_init, dtype=cfg.dtype, name='Dense_0')()
    gate_kernel = KernelParam(shape=(D, cfg.F), kernel_init=cfg.hidden_kernel_init_std, dtype=cfg.dtype, name='Dense_1')()
    down_kernel = KernelParam(shape=(cfg.F, cfg.D), kernel_init=cfg.hidden_kernel_layer_init, dtype=cfg.dtype, name='Dense_2')()

    # Fused up+gate kernel
    combined_kernel = jnp.concatenate([up_kernel, gate_kernel], axis=-1)  # [D, 2F]

    # Fused RMSNorm+Linear or standard matmul
    if pre_norm_input is not None and norm_scale is not None:
      # Fused path: RMSNorm(pre_norm_input) @ combined_kernel via custom_vjp
      combined = fused_rmsnorm_linear(pre_norm_input, norm_scale, combined_kernel, cfg.epsilon)
    else:
      # Standard path: x is already normalized
      combined = x_BxLxD @ combined_kernel  # [B, L, 2F]

    x_BxLxF = combined[..., :cfg.F]
    gate_BxLxF = combined[..., cfg.F:]

    # SwiGLU activation + down projection
    x_BxLxD = (x_BxLxF * nn.silu(gate_BxLxF)) @ down_kernel
    return x_BxLxD


class TBlock(nn.Module):
  """Transformer Block."""
  docfg: DoConfig

  @nn.compact
  def __call__(self, 
               in_BxLxD: jax.Array, 
               attention_mask: Optional[jax.Array] = None, 
               positions: Optional[jax.Array] = None):
    cfg = self.docfg

    # "pre-norm" - Use centralized RMSNorm initialization and epsilon
    x_BxLxD = RMSNorm(
      dtype=cfg.dtype, 
      scale_init=cfg.norm_scale_init, 
      epsilon=cfg.epsilon
    )(in_BxLxD)

    x_BxLxD = CausalAttn(cfg)(
      x_BxLxD, 
      attention_mask=attention_mask, 
      positions=positions
    ) * cfg.multipliers['mha_residual_mult']

    # residual connection
    x_BxLxD += in_BxLxD

    # Pre-FFN norm: fused with first projection if enabled
    use_fused = getattr(cfg, 'use_fused_norm', False) and cfg.ffn_type == 'swiglu'
    if use_fused:
      # Create RMSNorm scale param but don't apply norm yet — pass to MlpSwiGLU for fusion
      norm_scale = RMSNorm(
        dtype=cfg.dtype,
        scale_init=cfg.norm_scale_init,
        epsilon=cfg.epsilon
      )
      # Need to materialize the scale param by calling norm (but we pass pre_norm_input to MlpSwiGLU)
      z_BxLxD = norm_scale(x_BxLxD)
    else:
      z_BxLxD = RMSNorm(
        dtype=cfg.dtype,
        scale_init=cfg.norm_scale_init,
        epsilon=cfg.epsilon
      )(x_BxLxD)


    # Pre-FFN norm: fused with first projection if enabled
    use_fused = getattr(cfg, 'use_fused_norm', False) and cfg.ffn_type == 'swiglu'
    if use_fused:
      # Create RMSNorm scale param but don't apply norm yet — pass to MlpSwiGLU for fusion
      norm_scale = RMSNorm(
        dtype=cfg.dtype,
        scale_init=cfg.norm_scale_init,
        epsilon=cfg.epsilon
      )
      # Need to materialize the scale param by calling norm (but we pass pre_norm_input to MlpSwiGLU)
      z_BxLxD = norm_scale(x_BxLxD)
    else:
      z_BxLxD = RMSNorm(
        dtype=cfg.dtype,
        scale_init=cfg.norm_scale_init,
        epsilon=cfg.epsilon
      )(x_BxLxD)


    # Choose FFN type based on config
    if cfg.ffn_type == 'swiglu':
      if use_fused:
        # Pass un-normalized input + norm scale for fused RMSNorm+Linear
        z_BxLxD = MlpSwiGLU(cfg)(z_BxLxD, pre_norm_input=x_BxLxD,
                                  norm_scale=norm_scale.variables['params']['scale'])
      else:
        z_BxLxD = MlpSwiGLU(cfg)(z_BxLxD)
      loss = 0.0
    elif cfg.ffn_type == 'moe':
      # Create and apply MoeBlock with centralized initialization
      moe_layer = MoeBlock(
        config=cfg.moe_config,
        num_experts=cfg.moe_config.num_experts,
        num_experts_per_tok=cfg.moe_config.num_experts_per_tok,
        mesh=cfg.moe_config.mesh,
        kernel_init=cfg.moe_kernel_init(),
        kernel_axes=("embed", "experts"),
        name="MoeBlock"
      )
      # Apply MoE layer and get output and load balancing loss
      z_BxLxD, loss = moe_layer(z_BxLxD)

    elif cfg.ffn_type == 'regular_ffn':
      # Default to standard MLP
      z_BxLxD = Mlp(cfg)(z_BxLxD)
      loss = 0.0
    else:
      raise ValueError(f"Invalid FFN type: {cfg.ffn_type}") 

    return x_BxLxD + z_BxLxD * cfg.multipliers['mlp_residual_mult'], loss


class CausalAttn(nn.Module):
  """Causal attention layer with optional document boundary masking."""
  cfg: DoConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array, attention_mask: Optional[jax.Array] = None,
               positions: Optional[jax.Array] = None):
    """
    Args:
      x_BxLxD: Input tensor [batch, length, dim]
      attention_mask: Optional mask [batch, length, length] or [batch, 1, length, length]
                     where True = can attend, False = mask out.
                     If None, only causal masking is applied.
      positions: Optional position tensor [batch, length] for RoPE.
                 If None and use_rope=True, will generate positions.
    """
    cfg = self.cfg

    assert cfg.D % cfg.H == 0, f'D {cfg.D} not divisible by H {cfg.H}'
    Dh = cfg.D // cfg.H
    D = cfg.D

    # Fused QKV: create separate params (query/kernel, key/kernel, value/kernel)
    # but do a single matmul (3 kernel launches → 1)
    q_kernel = KernelParam(shape=(D, cfg.H, Dh), kernel_init=cfg.hidden_kernel_init_std, dtype=cfg.dtype, name='query')()
    k_kernel = KernelParam(shape=(D, cfg.H, Dh), kernel_init=cfg.hidden_kernel_init_std, dtype=cfg.dtype, name='key')()
    v_kernel = KernelParam(shape=(D, cfg.H, Dh), kernel_init=cfg.hidden_kernel_init_std, dtype=cfg.dtype, name='value')()


    # Reshape to 2D, concatenate, single matmul, split, reshape back
    qkv_kernel = jnp.concatenate([
        q_kernel.reshape(D, cfg.H * Dh),
        k_kernel.reshape(D, cfg.H * Dh),
        v_kernel.reshape(D, cfg.H * Dh),
    ], axis=-1)  # [D, 3*H*Dh]
    qkv = x_BxLxD @ qkv_kernel  # [B, L, 3*H*Dh]
    q_flat, k_flat, v_flat = jnp.split(qkv, 3, axis=-1)
    q_BxLxHxDh = q_flat.reshape(*x_BxLxD.shape[:-1], cfg.H, Dh)
    k_BxLxHxDh = k_flat.reshape(*x_BxLxD.shape[:-1], cfg.H, Dh)
    v_BxLxHxDh = v_flat.reshape(*x_BxLxD.shape[:-1], cfg.H, Dh)
    
    # Apply RMS normalization to k and q if enabled
    if cfg.use_qk_norm:
      # Use RMSNorm with separate instances for k and q - Use centralized norm initialization and epsilon
      k_rms_norm = RMSNorm(epsilon=cfg.epsilon,
                           dtype=cfg.dtype,
                           scale_init=cfg.norm_scale_init)
      q_rms_norm = RMSNorm(epsilon=cfg.epsilon,
                           dtype=cfg.dtype,
                           scale_init=cfg.norm_scale_init)

      # Apply separate RMSNorm instances along the last dimension
      k_BxLxHxDh = k_rms_norm(k_BxLxHxDh)
      q_BxLxHxDh = q_rms_norm(q_BxLxHxDh)

    # Apply RoPE if enabled (after QK norm, following the example)
    if cfg.use_rope:
      # Generate positions if not provided
      if positions is None:
        L = x_BxLxD.shape[1]
        positions = jnp.arange(L)[None, :]  # [1, L]

      # Apply RoPE to queries and keys
      q_BxLxHxDh = apply_rope(
          q_BxLxHxDh,
          positions,
          max_wavelength=cfg.rope_max_wavelength,
          scale_factor=cfg.rope_scale_factor
      )
      k_BxLxHxDh = apply_rope(
          k_BxLxHxDh,
          positions,
          max_wavelength=cfg.rope_max_wavelength,
          scale_factor=cfg.rope_scale_factor
      )

    # Attention scale: MuP uses 1/d, standard uses 1/sqrt(d)
    scale = 1.0 / Dh if cfg.use_mup else 1.0 / (Dh ** 0.5)

    # Build document boundary mask for flash attention.
    # attention_mask can be:
    #   - [B, L] doc_ids (int32): compute T×T mask on GPU from document IDs
    #   - [B, L, L] or [B, 1, L, L] boolean: legacy T×T mask from data pipeline
    #   - None: causal-only attention
    flash_mask = None
    if attention_mask is not None:
      if attention_mask.ndim == 2:  # [B, L] doc_ids — compute mask on GPU
        doc_ids = attention_mask
        same_doc = (doc_ids[:, :, None] == doc_ids[:, None, :])  # [B, L, L]
        non_padding = (doc_ids > 0)
        can_attend_to = non_padding[:, None, :]  # [B, 1, L]
        flash_mask = (same_doc & can_attend_to)[:, None, :, :]  # [B, 1, L, L]
      elif attention_mask.ndim == 3:  # [B, L, L]
        flash_mask = attention_mask[:, None, :, :]  # [B, 1, L, L]
      else:
        flash_mask = attention_mask

    # Flash attention via cuDNN: fuses QK^T, masking, softmax, and AV into a
    # single kernel with tiled computation (no T×T materialization).
    # is_causal=True handles the causal mask without materializing it;
    # flash_mask handles document boundary masking.
    # Cast to bf16 for cuDNN (fp32 not supported by flash attention kernels).
    q_bf16 = q_BxLxHxDh.astype(jnp.bfloat16)
    k_bf16 = k_BxLxHxDh.astype(jnp.bfloat16)
    v_bf16 = v_BxLxHxDh.astype(jnp.bfloat16)
    # Select attention implementation:
    #   'xla'       = naive O(n²) attention, vmap-safe (required for meta-training)
    #   'cudnn'     = cuDNN flash attention O(n), NOT vmap-safe but 1.5x faster
    #   'blockwise' = pure-JAX flash attention O(n), vmap-safe, best of both worlds
    attn_impl = getattr(cfg, 'attention_impl', 'xla')
    if attn_impl == 'blockwise':
      out_BxLxHxDh = blockwise_flash_attention(
          q_bf16, k_bf16, v_bf16,
          mask=flash_mask,
          scale=scale,
          is_causal=True,
      ).astype(cfg.dtype)
    else:
      out_BxLxHxDh = jax.nn.dot_product_attention(
          q_bf16, k_bf16, v_bf16,
          mask=flash_mask,
          scale=scale,
          is_causal=True,
          implementation=attn_impl,
      ).astype(cfg.dtype)
    # Output projection: contract (H, Dh) -> D using raw matmul
    out_kernel = KernelParam(shape=(cfg.H, Dh, cfg.D), kernel_init=cfg.hidden_kernel_layer_init, dtype=cfg.dtype, name='attn_out_proj')()
    out_BxLxD = jnp.einsum('...hk,hkd->...d', out_BxLxHxDh, out_kernel)
    return out_BxLxD * cfg.multipliers['output_mult']




class MoeConfig:
    def __init__(self):
      self.dtype = jnp.float32
      self.weight_dtype = jnp.float32
      self.emb_dim = 128
      self.mlp_dim = 256
      self.num_experts = 8
      self.num_experts_per_tok = 2
      self.capacity_factor = 1.0
      self.matmul_precision = jax.lax.Precision.DEFAULT
      self.model_call_mode = "train"
      self.load_balance_loss_weight = 0.01
      self.use_moe_linears = True
      self.ici_expert_parallelism = 1
      self.dcn_expert_parallelism = 1
      self.activations_in_float32 = True
      self.megablox = False
      self.activation = nn.swish
      self.mlp_activations = ["swish"]
      devices = jax.devices()
      self.mesh = Mesh(devices, ("data",))

class MuTransformerMoETask(base.Task, MuTask):
  """Transformer-based next token prediction task with MoE layers.
  
  This task supports CompletedP parameterization for hyperparameter transfer
  across model scales (width, depth, batch size, and dataset size).
  
  To enable CompletedP support:
  1. Pass parameterization_args during __init__
  2. Call set_training_config() before training with batch/steps info
  3. The init_with_state() method will automatically compute all scales
  
  Args:
    datasets: Dataset object with train/test iterators
    name: Task name string
    cfg: Model configuration dictionary
    mup_multipliers: Dictionary of muP multipliers (input_mult, output_mult, hidden_lr_mult)
    parameterization_args: Optional CompletedP configuration containing:
      - base_width: Base model width for HP transfer
      - base_depth: Base model depth for HP transfer  
      - base_batch_size: Base batch size for HP transfer
      - base_dataset_size: Base dataset size for HP transfer
      - depth_multipliers: Per-layer depth multipliers
      - alpha: Depth scaling exponent (0.5 to 1.0)
    training_config: Optional training configuration containing:
      - gradient_accumulation_steps: Number of gradient accumulation steps
      - local_batch_size: Local batch size per device
      - num_inner_steps: Number of inner training steps
  """
  
  def __init__(self, datasets, name, cfg,
               mup_multipliers=dict(input_mult=1.0,
                                    output_mult=1.0,
                                    hidden_lr_mult=1.0),
               parameterization_args=None,
               training_config=None):
    cfg['vocab_size'] = datasets.extra_info['vocab_size']
    cfg['input_mult'] = mup_multipliers['input_mult']
    cfg['output_mult'] = mup_multipliers['output_mult']
    cfg['max_seq_len'] = datasets.extra_info['sequence_length']

    # Assert that cfg.get('ffn_dim') has no remainder and cast if needed
    ffn_dim = cfg.get('ffn_dim')
    if isinstance(ffn_dim, float):
        assert ffn_dim.is_integer(), f"cfg['ffn_dim']={ffn_dim} must be integer-valued, got remainder: {ffn_dim % 1}"
        ffn_dim = int(ffn_dim)
        cfg['ffn_dim'] = ffn_dim
        
    self.hidden_lr_mult = mup_multipliers['hidden_lr_mult']
    self.task_name = name
    
    # Create mesh for MoE
    devices = jax.devices()
    self.mesh = Mesh(devices, ("data",))
    
    # Configure MoE parameters
    self.num_experts = cfg.get('num_experts', 8)
    self.num_experts_per_tok = cfg.get('num_experts_per_tok', 2)
    self.capacity_factor = cfg.get('capacity_factor', 1.5)
    self.load_balance_loss_weight = cfg.get('load_balance_loss_weight', 0.01)
    self.zloss_coefficient = cfg.get('zloss_coefficient', 0.0)

    # Create MoE config
    moe_config = MoeConfig()
    moe_config.num_experts = self.num_experts
    moe_config.num_experts_per_tok = self.num_experts_per_tok
    moe_config.capacity_factor = self.capacity_factor
    moe_config.load_balance_loss_weight = self.load_balance_loss_weight
    moe_config.mesh = self.mesh
    moe_config.emb_dim = cfg['model_dim']
    moe_config.mlp_dim = cfg['ffn_dim']

    # Optional MoE forward-pass implementation flags.
    # These can be set EITHER via the task cfg dict OR via environment
    # variables (USE_PALLAS_KERNEL, USE_RAGGED_DOT, MATMUL_DTYPE). The env
    # var fallback is needed because --cfg_options merges into the top-level
    # config in main.py but NOT into the task-level cfg dict that reaches
    # this __init__. Without the fallback, `use_pallas_kernel=True` in
    # --cfg_options silently has no effect.
    import os as _os
    _env_bool = lambda v, d: v.lower() in ("true", "1") if v else d
    moe_config.use_pallas_kernel = cfg.get(
        'use_pallas_kernel',
        _env_bool(_os.environ.get("USE_PALLAS_KERNEL", ""), False))
    moe_config.use_ragged_dot = cfg.get(
        'use_ragged_dot',
        _env_bool(_os.environ.get("USE_RAGGED_DOT", ""), True))
    _md = cfg.get('matmul_dtype', _os.environ.get("MATMUL_DTYPE", None))
    if isinstance(_md, str):
      _md = {'bf16': jnp.bfloat16, 'fp16': jnp.float16, 'fp32': jnp.float32, 'none': None}.get(_md.lower(), None)
    moe_config.matmul_dtype = _md

    import pprint
    pprint.pprint(cfg)
    # Create transformer config
    self.transformer_config = DoConfig(
        D=cfg.get('model_dim'),
        H=cfg.get('num_heads'),
        L=cfg.get('max_seq_len'),
        N=cfg.get('num_layers'),
        V=cfg.get('vocab_size'),
        F=cfg.get('ffn_dim'),
        use_qk_norm=cfg.get('use_qk_norm', True),
        ffn_type=cfg.get('ffn_type', 'moe'),
        moe_config=moe_config,
        dtype=jnp.float32,
        fsdp_enabled=False,
        remat=cfg.get('remat', False),
        use_mup=cfg.get('use_mup', True),
    )
    pprint.pprint(self.transformer_config)
    
    # Create the Flax module
    self.flax_module = TransformerDo(docfg=self.transformer_config)
    
    self.datasets = datasets
    self.mup_lrs = None
    self.mup_eps = None
    self.mup_state = None

    self.eps_mult = 1 / cfg.get('model_dim')
    
    # Initialize CompletedP parameterization support
    self.set_parameterization_args(parameterization_args)
    if training_config is not None:
      self.set_training_config(**training_config)
    else:
      self.training_config = None
    self.completed_p_scales = None
    
    # Initialize mup state (this calls init_with_state internally)
    self.init_mup_state()

    if 'lm1b' in self.datasets.extra_info['name'].lower():
      self.eos_token_id = 0
    else:
      self.eos_token_id = self.datasets.extra_info['eos_token_id']

  def init(self, key: PRNGKey):
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    
    return self.flax_module.init({"params": key}, batch["image"])

  def init_with_state(self, key: PRNGKey) -> Tuple[Params, ModelState]:
    """Initialize model parameters and state with muP/CompletedP scaling.
    
    If CompletedP parameterization is configured (via parameterization_args and
    training_config), this method will:
    1. Re-initialize parameters with proper CompletedP scaling
    2. Set forward pass multipliers on the model
    3. Compute all CompletedP scales (LR, epsilon, WD, betas)
    4. Include all scales in the returned state dictionary
    
    Args:
      key: JAX random key for initialization
      
    Returns:
      Tuple of (params, state) where state contains all muP/CompletedP scales
    """
    params = self.init(key)
    
    # Get basic muP LRs and epsilons from flax module
    if self.mup_lrs is None:
      self.mup_lrs = self.flax_module.get_mup_lrs(params, jax.devices()[jax.process_index()])
    if self.mup_eps is None:
      self.mup_eps = self.flax_module.get_mup_epsilons(params, jax.devices()[jax.process_index()])
    
    state = {'flax_mup_lrs': self.mup_lrs, 'flax_mup_eps': self.mup_eps}
    
    # Check if CompletedP parameterization is configured
    if hasattr(self, 'parameterization_args') and self.parameterization_args is not None:
      if hasattr(self, 'training_config') and self.training_config is not None:
        # Compute CompletedP scales (only if not already computed)
        # This caches the scales for later use
        if self.completed_p_scales is None:
          self.completed_p_scales = self._compute_completed_p_scales(params)
        
        # Re-initialize parameters with CompletedP scaling
        key, key1 = jax.random.split(key)
        params = self._reinit_params_with_completed_p(params, key1)
        
        # NOTE: Do NOT call _set_model_multipliers() here - it modifies self.flax_module
        # which is a side effect that escapes JIT scope. Multipliers are set in 
        # init_mup_state() which is called outside of JIT before training.
        
        # Add CompletedP scales to state
        state = self.get_completed_p_state(params, state)
        
        # Also add legacy mup_lrs_to_use for backwards compatibility with some optimizers
        state = self.get_mup_state(state, eps_mult=self.eps_mult)
        
        return params, state
    
    # Fallback to legacy muP state handling
    return params, self.get_mup_state(state, eps_mult=self.eps_mult)

  def get_loss(self, logits, data):
    """Compute cross entropy loss for next token prediction."""
    targets = data["label"]
    mask = (data["image"] !=  self.eos_token_id)
    loss = base.softmax_cross_entropy(
        logits=logits, labels=jax.nn.one_hot(targets, self.transformer_config.V))
    return jnp.sum(loss * mask) / jnp.sum(mask)

  def compute_zloss(self, logits, data):
    """Z-loss regularization on output logits (arXiv 2309.14322).

    L_z = coeff * mean(logsumexp(logits, axis=-1)^2)
    Masked to non-padding tokens, consistent with get_loss().
    """
    mask = (data["image"] != self.eos_token_id)
    lse = jax.scipy.special.logsumexp(logits, axis=-1)  # [B, L]
    zloss_per_token = lse ** 2  # [B, L]
    return self.zloss_coefficient * jnp.sum(zloss_per_token * mask) / jnp.sum(mask)

  @functools.partial(jax.jit, static_argnums=(0,))
  def loss(self, params: Any, key: PRNGKey, data: Any):
    tokens = data["image"]
    
    # Create attention mask from padding if not provided
    attention_mask = data["attention_mask"]
    
    # Forward pass with attention mask
    logits, intermediates = self.flax_module.apply(
        params,
        tokens,
        attention_mask=attention_mask,
        mutable=['intermediates']
    )
    
    # Calculate next token prediction loss
    task_loss = self.get_loss(logits, data)
    
    # Add load balance loss if present
    total_loss = task_loss
    if 'intermediates' in intermediates and 'load_balance_loss' in intermediates['intermediates']:
        load_balance_loss = intermediates['intermediates']['load_balance_loss'][0]
        total_loss = task_loss + load_balance_loss

    # Add z-loss if configured (gradient only; value cancels via stop_gradient)
    if self.zloss_coefficient > 0:
        zloss = self.compute_zloss(logits, data)
        total_loss = total_loss + zloss - jax.lax.stop_gradient(zloss)

    return total_loss

  @functools.partial(jax.jit, static_argnums=(0,))
  def loss_with_state(self, params: Any, state: Any, key: PRNGKey, data: Any):
    tokens = data["image"]
    
    # Create attention mask from padding if not provided
    attention_mask = data["attention_mask"]
    
    # Forward pass with intermediates and attention mask
    logits, intermediates = self.flax_module.apply(
        params,
        tokens,
        attention_mask=attention_mask,
        mutable=['intermediates']
    )
    
    # Calculate next token prediction loss
    task_loss = self.get_loss(logits, data)
    
    # Add load balance loss if present
    total_loss = task_loss
    if 'intermediates' in intermediates and 'load_balance_loss' in intermediates['intermediates']:
        load_balance_loss = intermediates['intermediates']['load_balance_loss'][0]
        total_loss = task_loss + load_balance_loss

    # Add z-loss if configured (gradient only; value cancels via stop_gradient)
    if self.zloss_coefficient > 0:
        zloss = self.compute_zloss(logits, data)
        total_loss = total_loss + zloss - jax.lax.stop_gradient(zloss)

    return total_loss, self.get_mup_state(state)

  @functools.partial(jax.jit, static_argnums=(0,))
  def loss_with_state_and_aux(
      self, params: Params, state: ModelState, key: PRNGKey,
      data: Batch) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
    
    tokens = data["image"]
    
    # Create attention mask from padding if not provided
    
    attention_mask = data["attention_mask"]
    
    # Forward pass with intermediates and attention mask
    logits, intermediates = self.flax_module.apply(
        params,
        tokens,
        attention_mask=attention_mask,
        mutable=['intermediates']
    )
    
    # Calculate next token prediction loss
    task_loss = self.get_loss(logits, data)
    
    # Prepare aux dict with task loss and load balance losses
    aux = {'train loss': task_loss}

    
    # Add load balance losses if present
    total_loss = task_loss
    if 'intermediates' in intermediates and 'load_balance_loss' in intermediates['intermediates']:
        load_balance_loss = intermediates['intermediates']['load_balance_loss'][0]
        total_loss = task_loss + load_balance_loss
        aux['load_balance_loss'] = load_balance_loss
        
        # Add individual layer load balance losses if available
        for k, v in intermediates['intermediates'].items():
            if k.endswith('_load_balance_loss'):
                aux[k] = v[0]

    # Add z-loss if configured (gradient only; value cancels via stop_gradient)
    if self.zloss_coefficient > 0:
        zloss = self.compute_zloss(logits, data)
        total_loss = total_loss + zloss - jax.lax.stop_gradient(zloss)
        aux['zloss'] = zloss

    return total_loss, {'state': self.get_mup_state(state), 'aux': aux}

  # ----- Eval-only pure cross-entropy loss methods (decouple LBL from test) -----
  # benchmark.evaluate_test prefers loss_and_accuracy(_with_state) when present
  # over the plain loss(_with_state) methods. Providing these methods makes the
  # reported "test loss" pure CE for both dense and MoE so it is directly
  # comparable to the dense baseline (e.g. wandb run <NEED>/complete_p_testing/2l2fscxq,
  # final test loss 4.5219). Training still uses loss_with_state which includes
  # the load_balance_loss in the gradient path so the gate stays balanced.
  @functools.partial(jax.jit, static_argnums=(0,))
  def loss_and_accuracy(self, params: Any, key: PRNGKey, data: Any):
    tokens = data["image"]
    attention_mask = data["attention_mask"]
    logits, _ = self.flax_module.apply(
        params,
        tokens,
        attention_mask=attention_mask,
        mutable=['intermediates'],
    )
    task_loss = self.get_loss(logits, data)
    return task_loss, jnp.array(0.0, dtype=task_loss.dtype)

  @functools.partial(jax.jit, static_argnums=(0,))
  def loss_and_accuracy_with_state(
      self, params: Any, state: Any, key: PRNGKey, data: Any):
    tokens = data["image"]
    attention_mask = data["attention_mask"]
    logits, _ = self.flax_module.apply(
        params,
        tokens,
        attention_mask=attention_mask,
        mutable=['intermediates'],
    )
    task_loss = self.get_loss(logits, data)
    return task_loss, jnp.array(0.0, dtype=task_loss.dtype)


if __name__ == "__main__":

  


  # Create a small transformer configuration
  config = DoConfig(
      D=128,  # model dimension
      H=4,    # number of attention heads
      L=64,   # max sequence length
      N=2,    # number of transformer blocks
      V=1000, # vocabulary size
      F=256,  # feed-forward dimension
      use_qk_norm=True,
      ffn_type='regular_ffn',
      moe_config=MoeConfig(),
      use_mup=True,
      completep_alpha=1.0,
      mup_base_width=64,

  )
  
  # Initialize model
  model = TransformerDo(docfg=config,)
  
  # Create random input data
  batch_size = 2
  seq_length = 32
  rng = jax.random.PRNGKey(0)
  input_tokens = jax.random.randint(
      rng, shape=(batch_size, seq_length), minval=0, maxval=config.V
  )
  
  # Initialize parameters
  params = model.init(rng, input_tokens)
  import pprint
  pprint.pprint(jax.tree_util.tree_map(lambda x: x.shape, params))


  print("--"*100)
  print("MuP LRs")
  print("--"*100)


  pprint.pprint(
    
    jax.tree_util.tree_map(lambda x: x, model.get_mup_lrs(params, jax.devices()[jax.process_index()]))
    
  )


  print("--"*100)
  print("MuP EPS")
  print("--"*100)



  pprint.pprint(
    
    jax.tree_util.tree_map(lambda x: x, model.get_mup_epsilons(params, jax.devices()[jax.process_index()]))
    
  )

  print("--"*100)
  print("Module Types")
  print("--"*100)

  pprint.pprint(
    
    jax.tree_util.tree_map(lambda x: x, model.get_module_types(params))
    
  )
  exit(0)
  
  # Run a forward pass
  logits, intermediates = model.apply(params, 
                                      input_tokens,
                                      mutable=['intermediates'])
  
  # Print shapes and summary
  print(f"Input shape: {input_tokens.shape}")
  print(f"Output logits shape: {logits.shape}")
  print(f"Model parameter count: {sum(x.size for x in jax.tree_util.tree_leaves(params))}")
  
  # Test a prediction
  predicted_tokens = jnp.argmax(logits, axis=-1)
  print(f"Predicted tokens shape: {predicted_tokens.shape}")
  print("Test successful!")

  # Extract and print the load balance losses from intermediates
  if intermediates and 'intermediates' in intermediates:
    if 'load_balance_loss' in intermediates['intermediates']:
      total_load_balance_loss = intermediates['intermediates']['load_balance_loss']
      print(f"Total load balance loss: {total_load_balance_loss}")
