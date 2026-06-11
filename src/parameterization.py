"""
Parameterization module for hyperparameter transfer across model scales.

This module implements parameterization schemes that enable hyperparameter transfer
across width, depth, batch size, and dataset size. The main implementation is
CompletedP, based on the paper's methodology for stable training across scales.

Key scaling ratios:
- m_N: width ratio (current_width / base_width)
- m_L: depth ratio (current_depth / base_depth)  
- m_B: batch ratio (current_batch / base_batch)
- m_D: data ratio (current_dataset_size / base_dataset_size)
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Tuple, Optional, Callable
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

# Set to True (or via main.py: parameterization.VERBOSE = True) to enable debug prints
VERBOSE = False


class ModuleType(Enum):
    """
    Enum for different module types in a transformer architecture.
    Each type has different scaling rules for initialization and optimization.
    
    This is the coarse-grained classification used by the CompletedP parameterization
    to determine scaling factors. See TensorType for fine-grained tensor classification.
    """
    # Input layers
    INPUT_EMBED = "input_embed"           # Token embeddings
    POS_EMBED = "pos_embed"               # Positional embeddings
    
    # Hidden layers - weights
    HIDDEN_WEIGHT = "hidden_weight"       # Q/K/V projections, MLP weights, attention output
    HIDDEN_BIAS = "hidden_bias"           # Hidden layer biases (if any)
    HIDDEN_NORM = "hidden_norm"           # RMSNorm/LayerNorm scale parameters in hidden layers
    
    # Special attention components
    QK_NORM = "qk_norm"                   # Query-Key normalization layers
    
    # Output layers
    UNEMBED_WEIGHT = "unembed_weight"     # Output/unembedding projection weights
    UNEMBED_BIAS = "unembed_bias"         # Output projection biases (if any)
    UNEMBED_NORM = "unembed_norm"         # Output layer normalization


class TensorType(Enum):
    """
    Fine-grained tensor type enum for per-tensor learning rate control.
    
    Each TensorType maps to a ModuleType which determines the base scaling rules
    from CompletedP parameterization. This allows per-tensor LR overrides while
    maintaining proper scaling behavior.
    
    Tensor types cover all possible tensors in a transformer architecture:
    - Embeddings
    - Attention components (Q, K, V, output, norms)
    - MLP/FFN components (up, gate, down projections)
    - Layer normalization (post-attention, post-MLP, output)
    - Output/unembedding
    - MoE components (gate, expert weights)
    """
    # Embedding layers
    EMBEDDING = "embedding"                     # Token embeddings
    POS_EMBEDDING = "pos_embedding"             # Positional embeddings (learned)
    
    # Attention layers
    ATTENTION_QUERY = "attention_query"         # Query projection (W_Q)
    ATTENTION_KEY = "attention_key"             # Key projection (W_K)
    ATTENTION_VALUE = "attention_value"         # Value projection (W_V)
    ATTENTION_OUTPUT = "attention_output"       # Attention output projection (W_O)
    ATTENTION_QUERY_NORM = "attention_query_norm"  # Query RMSNorm (QK norm)
    ATTENTION_KEY_NORM = "attention_key_norm"      # Key RMSNorm (QK norm)
    
    # MLP/FFN layers
    MLP_UP = "mlp_up"                           # MLP up projection (embed -> ffn_dim)
    MLP_GATE = "mlp_gate"                       # MLP gate projection (for SwiGLU)
    MLP_DOWN = "mlp_down"                       # MLP down projection (ffn_dim -> embed)
    
    # Layer normalization
    POST_ATTENTION_NORM = "post_attention_norm"   # RMSNorm/LayerNorm after attention (pre-MLP)
    POST_MLP_NORM = "post_mlp_norm"               # RMSNorm/LayerNorm after MLP
    OUTPUT_NORM = "output_norm"                   # Final output normalization (before unembedding)
    
    # Output/Unembedding
    UNEMBEDDING = "unembedding"                 # Output projection to vocab
    
    # MoE-specific tensors
    MOE_GATE = "moe_gate"                       # MoE router/gating network
    MOE_UP = "moe_up"                           # MoE expert up projection (wi_0)
    MOE_GATE_PROJ = "moe_gate_proj"             # MoE expert gate projection (wi_1, for SwiGLU)
    MOE_DOWN = "moe_down"                       # MoE expert down projection (wo)


# Integer index mapping for TensorType (used by learned optimizers)
TENSOR_TYPE_NAMES = [tt.value for tt in TensorType]
TENSOR_TYPE_INDEX = {tt.value: i for i, tt in enumerate(TensorType)}
NUM_TENSOR_TYPES = len(TensorType)


# Mapping from TensorType to ModuleType for parameterization scaling
TENSOR_TYPE_TO_MODULE_TYPE: Dict[TensorType, ModuleType] = {
    # Embeddings
    TensorType.EMBEDDING: ModuleType.INPUT_EMBED,
    TensorType.POS_EMBEDDING: ModuleType.POS_EMBED,
    
    # Attention weights -> HIDDEN_WEIGHT
    TensorType.ATTENTION_QUERY: ModuleType.HIDDEN_WEIGHT,
    TensorType.ATTENTION_KEY: ModuleType.HIDDEN_WEIGHT,
    TensorType.ATTENTION_VALUE: ModuleType.HIDDEN_WEIGHT,
    TensorType.ATTENTION_OUTPUT: ModuleType.HIDDEN_WEIGHT,
    
    # Attention norms -> QK_NORM (special scaling)
    TensorType.ATTENTION_QUERY_NORM: ModuleType.QK_NORM,
    TensorType.ATTENTION_KEY_NORM: ModuleType.QK_NORM,
    
    # MLP weights -> HIDDEN_WEIGHT
    TensorType.MLP_UP: ModuleType.HIDDEN_WEIGHT,
    TensorType.MLP_GATE: ModuleType.HIDDEN_WEIGHT,
    TensorType.MLP_DOWN: ModuleType.HIDDEN_WEIGHT,
    
    # Layer norms -> HIDDEN_NORM
    TensorType.POST_ATTENTION_NORM: ModuleType.HIDDEN_NORM,
    TensorType.POST_MLP_NORM: ModuleType.HIDDEN_NORM,
    
    # Output norm -> UNEMBED_NORM
    TensorType.OUTPUT_NORM: ModuleType.UNEMBED_NORM,
    
    # Unembedding -> UNEMBED_WEIGHT
    TensorType.UNEMBEDDING: ModuleType.UNEMBED_WEIGHT,
    
    # MoE components -> HIDDEN_WEIGHT (same scaling as dense layers)
    TensorType.MOE_GATE: ModuleType.HIDDEN_WEIGHT,
    TensorType.MOE_UP: ModuleType.HIDDEN_WEIGHT,
    TensorType.MOE_GATE_PROJ: ModuleType.HIDDEN_WEIGHT,
    TensorType.MOE_DOWN: ModuleType.HIDDEN_WEIGHT,
}


def get_module_type(tensor_type: TensorType) -> ModuleType:
    """
    Get the ModuleType for a given TensorType.
    
    Args:
        tensor_type: The fine-grained tensor type
        
    Returns:
        The corresponding ModuleType for parameterization scaling
        
    Raises:
        ValueError: If tensor_type is not in the mapping
    """
    if tensor_type not in TENSOR_TYPE_TO_MODULE_TYPE:
        raise ValueError(f"Unknown tensor type: {tensor_type}")
    return TENSOR_TYPE_TO_MODULE_TYPE[tensor_type]


class Parameterization(ABC):
    """
    Abstract base class for parameterization schemes.
    
    Parameterizations define how to scale hyperparameters (learning rate, epsilon,
    weight decay) and how to initialize parameters when changing model configuration
    (width, depth, batch size, dataset size).
    
    Args:
        base_width: Base model width (e.g., number of heads or embedding dimension)
        base_depth: Base model depth (number of layers)
        base_batch_size: Base batch size
        base_dataset_size: Base dataset size (number of tokens)
        current_width: Current model width
        current_depth: Current model depth
        current_batch_size: Current batch size
        current_dataset_size: Current dataset size
    """
    def __init__(
        self,
        base_width: int,
        base_depth: int,
        base_batch_size: int,
        base_dataset_size: int,
        current_width: int,
        current_depth: int,
        current_batch_size: int,
        current_dataset_size: int,
    ):
        self.base_width = base_width
        self.base_depth = base_depth
        self.base_batch_size = base_batch_size
        self.base_dataset_size = base_dataset_size
        
        self.current_width = current_width
        self.current_depth = current_depth
        self.current_batch_size = current_batch_size
        self.current_dataset_size = current_dataset_size
        
        # Compute scaling ratios
        self.m_N = current_width / base_width      # Width ratio

        # just use a base depth of 2.0 residual blocks 
        # which corresponds to the number of transformer layers
        self.m_L = current_depth     
        
        self.m_B = current_batch_size / base_batch_size  # Batch ratio
        self.m_D = current_dataset_size / base_dataset_size  # Data ratio
        
        # Print all values for debugging
        if VERBOSE:
            print(f"Parameterization initialized with:")
            print(f"  base_width: {self.base_width}")
            print(f"  base_depth: {self.base_depth}")
            print(f"  base_batch_size: {self.base_batch_size}")
            print(f"  base_dataset_size: {self.base_dataset_size}")
            print(f"  current_width: {self.current_width}")
            print(f"  current_depth: {self.current_depth}")
            print(f"  current_batch_size: {self.current_batch_size}")
            print(f"  current_dataset_size: {self.current_dataset_size}")
            print(f"  m_N (width ratio): {self.m_N}")
            print(f"  m_L (depth): {self.m_L}")
            print(f"  m_B (batch ratio): {self.m_B}")
            print(f"  m_D (data ratio): {self.m_D}")

        
    @abstractmethod
    def get_lr_rescaling(
        self, 
        params_info: Dict[str, Tuple[ModuleType, Tuple[int, int]]],
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Get learning rate rescaling factors for each parameter.
        
        Args:
            params_info: Dictionary mapping parameter names to tuples of
                        (ModuleType, [fan_in, fan_out])
            device: JAX device to place the arrays on (e.g., jax.devices()[0])
        
        Returns:
            Dictionary mapping parameter names to LR scaling factors (jnp.ndarray)
        """
        pass
    
    @abstractmethod
    def get_eps_rescaling(
        self,
        params_info: Dict[str, Tuple[ModuleType, Tuple[int, int]]],
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Get AdamW epsilon rescaling factors for each parameter.
        
        Args:
            params_info: Dictionary mapping parameter names to tuples of
                        (ModuleType, [fan_in, fan_out])
            device: JAX device to place the arrays on (e.g., jax.devices()[0])
        
        Returns:
            Dictionary mapping parameter names to epsilon scaling factors (jnp.ndarray)
        """
        pass
    
    @abstractmethod
    def get_wd_rescaling(
        self,
        params_info: Dict[str, Tuple[ModuleType, Tuple[int, int]]],
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Get weight decay rescaling factors for each parameter.
        
        Args:
            params_info: Dictionary mapping parameter names to tuples of
                        (ModuleType, [fan_in, fan_out])
            device: JAX device to place the arrays on (e.g., jax.devices()[0])
        
        Returns:
            Dictionary mapping parameter names to weight decay scaling factors (jnp.ndarray)
        """
        pass
    
    @abstractmethod
    def get_multipliers(self) -> Dict[str, float]:
        """
        Get forward pass multipliers for residual connections and output.
        
        Returns:
            Dictionary with keys:
                - 'mha_residual_mult': MHA residual connection multiplier
                - 'mlp_residual_mult': MLP residual connection multiplier  
                - 'output_mult': Output layer multiplier
        """
        pass
    
    @abstractmethod
    def init_params(
        self,
        params: Dict[str, Any],
        params_info: Dict[str, Tuple[ModuleType, Tuple[int, int]]],
        key: jax.random.PRNGKey,
        base_init_std: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Re-initialize parameters according to the parameterization scheme.
        
        Args:
            params: Dictionary of model parameters to re-initialize
            params_info: Dictionary mapping parameter names to tuples of
                        (ModuleType, [fan_in, fan_out])
            key: JAX random key for initialization
            base_init_std: Base standard deviation for initialization
        
        Returns:
            Re-initialized parameters dictionary
        """
        pass


class CompletedPParameterization(Parameterization):
    """
    CompletedP parameterization for hyperparameter transfer.
    
    This implements the CompletedP parameterization scheme which enables
    hyperparameter transfer across:
    - Width (number of heads / embedding dimension)
    - Depth (number of layers)
    - Batch size
    - Dataset size (number of tokens)
    
    Key features:
    - Residual scaling by m_L^{-alpha} for depth transfer
    - Width-dependent learning rate and initialization scaling
    - SDE-based batch size and token horizon scaling
    - Per-tensor hyperparameter overrides for fine-grained control
    - Depth-wise multipliers for layer-specific scaling
    
    Args:
        alpha: Depth scaling exponent, must be in [0.5, 1.0]. Default is 1.0.
               - alpha=0.5: May give slightly better performance at larger depths
               - alpha=1.0: More conservative, well-tested scaling
        depth_multipliers: List of per-layer multipliers of length base_depth.
                          When current_depth != base_depth, values are linearly interpolated.
                          Applied uniformly to all hyperparameters (lr, eps, wd, init, beta1, beta2).
        per_tensor_lr_multipliers: Dict mapping TensorType to LR multiplier (default 1.0).
        per_tensor_eps_multipliers: Dict mapping TensorType to epsilon multiplier (default 1.0).
        per_tensor_wd_multipliers: Dict mapping TensorType to weight decay multiplier (default 1.0).
        per_tensor_init_multipliers: Dict mapping TensorType to init scale multiplier (default 1.0).
        per_tensor_beta1_multipliers: Dict mapping TensorType to (1-beta1) multiplier (default 1.0).
        per_tensor_beta2_multipliers: Dict mapping TensorType to (1-beta2) multiplier (default 1.0).
        **kwargs: Base Parameterization arguments
    """
    
    def __init__(
        self,
        base_width: int,
        base_depth: int,
        base_batch_size: int,
        base_dataset_size: int,
        current_width: int,
        current_depth: int,
        current_batch_size: int,
        current_dataset_size: int,
        depth_multipliers: list[float],
        alpha: float = 1.0,
        per_tensor_lr_multipliers: Optional[Dict[TensorType, float]] = None,
        per_tensor_eps_multipliers: Optional[Dict[TensorType, float]] = None,
        per_tensor_wd_multipliers: Optional[Dict[TensorType, float]] = None,
        per_tensor_init_multipliers: Optional[Dict[TensorType, float]] = None,
        per_tensor_beta1_multipliers: Optional[Dict[TensorType, float]] = None,
        per_tensor_beta2_multipliers: Optional[Dict[TensorType, float]] = None,
    ):
        super().__init__(
            base_width=base_width,
            base_depth=base_depth,
            base_batch_size=base_batch_size,
            base_dataset_size=base_dataset_size,
            current_width=current_width,
            current_depth=current_depth,
            current_batch_size=current_batch_size,
            current_dataset_size=current_dataset_size,
        )
        
        assert 0.5 <= alpha <= 1.0, f"alpha must be in [0.5, 1.0], got {alpha}"
        self.alpha = alpha
        
        # Precompute common scaling factors for depth
        self._m_L_alpha_minus_1 = self.m_L ** (alpha - 1)  # m_L^{α-1}
        self._m_L_neg_alpha = self.m_L ** (-alpha)         # m_L^{-α}
        
        # Note: Width scaling is now computed per-layer using base_width / fan_in
        # instead of a global m_N^{-1} to handle layers with different widths
        # (e.g., MLP layers with 4x expansion)
        
        # SDE scaling factors for batch/data transfer
        self._sde_lr_wd_scale = math.sqrt(self.m_B / self.m_D)   # √(m_B/m_D)
        self._sde_eps_scale = math.sqrt(self.m_D / self.m_B)     # √(m_D/m_B)
        
        # Store depth-wise multipliers (default to all 1.0 if not provided)
        # Single set of depth multipliers applies uniformly to all hyperparameters
        self.depth_multipliers = depth_multipliers 
        
        # if depth_multipliers is not None else [1.0] * base_depth
        assert len(self.depth_multipliers) == base_depth, \
            f"depth_multipliers must have length {base_depth} (base_depth), got {len(self.depth_multipliers)}"
        
        # Store per-tensor-type multipliers for each hyperparameter
        # These allow fine-grained control by tensor type (e.g., different LR for Q vs K vs V)
        self.per_tensor_lr_multipliers = per_tensor_lr_multipliers or {}
        self.per_tensor_eps_multipliers = per_tensor_eps_multipliers or {}
        self.per_tensor_wd_multipliers = per_tensor_wd_multipliers or {}
        self.per_tensor_init_multipliers = per_tensor_init_multipliers or {}
        self.per_tensor_beta1_multipliers = per_tensor_beta1_multipliers or {}
        self.per_tensor_beta2_multipliers = per_tensor_beta2_multipliers or {}
    
    def _get_width_scale_inv(self, fan_in: int) -> float:
        """
        Get width scaling factor (inverse) for a layer based on its fan_in.
        
        This replaces the global m_N^{-1} with base_width / fan_in,
        which properly handles layers with different widths.
        
        Args:
            fan_in: Input dimension of the layer
            
        Returns:
            base_width / fan_in scaling factor
        """
        return self.base_width / fan_in
    
    def _get_width_scale(self, fan_in: int) -> float:
        """
        Get width scaling factor for a layer based on its fan_in.
        
        This replaces the global m_N with fan_in / base_width,
        which properly handles layers with different widths.
        
        Args:
            fan_in: Input dimension of the layer
            
        Returns:
            fan_in / base_width scaling factor
        """
        return fan_in / self.base_width
    
    def _get_width_scale_inv_sq(self, fan_in: int) -> float:
        """
        Get squared inverse width scaling for a layer.
        
        Used for output layer initialization variance scaling.
        
        Args:
            fan_in: Input dimension of the layer
            
        Returns:
            (base_width / fan_in)^2 scaling factor
        """
        return (self.base_width / fan_in) ** 2
    
    def _interpolate_depth_multiplier(self, layer_idx: int, multipliers: list) -> float:
        """
        Interpolate depth multiplier for a given layer index.
        
        When current_depth != base_depth, the base_depth multiplier values are treated
        as spanning a [0, 1] interval and linearly interpolated to obtain the multiplier
        for the given layer in the current model.
        
        Args:
            layer_idx: The layer index in the current model (0 to current_depth-1)
            multipliers: List of multipliers of length base_depth
            
        Returns:
            Interpolated multiplier value for the given layer
        """
        if VERBOSE:
            print(f"DEBUG _interpolate_depth_multiplier: layer_idx={layer_idx}, current_depth={self.current_depth}, base_depth={self.base_depth}")
            print(f"DEBUG _interpolate_depth_multiplier: multipliers={multipliers}")

        if self.current_depth == self.base_depth:
            # No interpolation needed
            if VERBOSE:
                print(f"DEBUG _interpolate_depth_multiplier: No interpolation needed, returning multipliers[{layer_idx}]={multipliers[layer_idx]}")
            return multipliers[layer_idx]

        # Map layer_idx to [0, 1] interval based on current_depth
        # Layer 0 maps to 0, layer (current_depth-1) maps to 1
        if self.current_depth == 1:
            t = 0.5  # Single layer maps to middle
            if VERBOSE:
                print(f"DEBUG _interpolate_depth_multiplier: Single layer, t=0.5")
        else:
            t = layer_idx / (self.current_depth - 1)
            if VERBOSE:
                print(f"DEBUG _interpolate_depth_multiplier: t = {layer_idx} / ({self.current_depth} - 1) = {t}")

        # Map t to base_depth multiplier positions
        # Position 0 in base corresponds to t=0, position (base_depth-1) to t=1
        if self.base_depth == 1:
            if VERBOSE:
                print(f"DEBUG _interpolate_depth_multiplier: Base depth is 1, returning multipliers[0]={multipliers[0]}")
            return multipliers[0]

        # Find the position in base multipliers
        base_pos = t * (self.base_depth - 1)
        lower_idx = int(base_pos)
        upper_idx = min(lower_idx + 1, self.base_depth - 1)
        if VERBOSE:
            print(f"DEBUG _interpolate_depth_multiplier: base_pos = {t} * ({self.base_depth} - 1) = {base_pos}")
            print(f"DEBUG _interpolate_depth_multiplier: lower_idx={lower_idx}, upper_idx={upper_idx}")

        # Linear interpolation
        frac = base_pos - lower_idx
        result = multipliers[lower_idx] * (1 - frac) + multipliers[int(upper_idx)] * frac
        if VERBOSE:
            print(f"DEBUG _interpolate_depth_multiplier: frac={frac}")
            print(f"DEBUG _interpolate_depth_multiplier: result = {multipliers[lower_idx]} * (1 - {frac}) + {multipliers[int(upper_idx)]} * {frac} = {result}")
        return result


        
    def _get_depth_multiplier_for_layer(self, layer_idx: Optional[int]) -> float:
        """
        Get depth multiplier for a specific layer or global (non-layer) parameter.
        
        Args:
            layer_idx: Layer index (0 to current_depth-1) or None for non-layer params
            
        Returns:
            The appropriate multiplier (1.0 for non-layer params, interpolated for layers)
        """
        if layer_idx is None:
            return 1.0

        interpolated_multiplier = self._interpolate_depth_multiplier(layer_idx, self.depth_multipliers)
        if VERBOSE:
            print(f"interpolated_multiplier: {interpolated_multiplier}")
            print(f"depth_multipliers: {self.depth_multipliers}")
            print(f"base_depth: {self.base_depth}")
        return interpolated_multiplier / self.base_depth
    
    def _get_per_tensor_multiplier(
        self,
        tensor_type: TensorType,
        multipliers_dict: Dict[TensorType, float]
    ) -> float:
        """
        Get per-tensor-type multiplier for a specific hyperparameter.
        
        Args:
            tensor_type: The TensorType to look up
            multipliers_dict: Dict mapping TensorType (or string) to multiplier values.
                             Can use either TensorType enum keys or string keys
                             (e.g., TensorType.ATTENTION_QUERY or "attention_query")
            
        Returns:
            The multiplier for this tensor type (1.0 if not specified)
        """
        # Try TensorType enum key first
        if tensor_type in multipliers_dict:
            return multipliers_dict[tensor_type]
        # Try string key (the .value of the enum)
        if tensor_type.value in multipliers_dict:
            return multipliers_dict[tensor_type.value]
        # Default to 1.0 (identity multiplier)
        return 1.0
    
    def get_lr_rescaling(
        self,
        params_info: Dict[str, Tuple[Any, Tuple[int, int]]],
        device: Optional[Any] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Get learning rate rescaling for CompletedP.
        
        From Table 1 (using base_width/fan_in instead of global m_N^{-1}):
        - Input Emb: base × √(m_B/m_D)
        - Hidden weights: × (base_width/fan_in) × m_L^{α-1} × √(m_B/m_D)
        - Hidden biases/norm: × m_L^{α-1} × √(m_B/m_D)
        - Unemb. LN: base × √(m_B/m_D)
        - Unemb. weights: × (base_width/fan_in) × √(m_B/m_D)
        
        Args:
            params_info: Dict mapping param names to (ModuleType|TensorType, (fan_in, fan_out))
                        fan_in is used to compute per-layer width scaling.
                        Accepts both ModuleType and TensorType (TensorType is converted to ModuleType).
            device: JAX device to place the arrays on (e.g., jax.devices()[0])
        
        Returns:
            Dict mapping param names to jnp.ndarray scaling factors
        """
        lr_scales = {}
        
        for name, (type_enum, (fan_in, fan_out)) in params_info.items():
            # Convert TensorType to ModuleType if needed
            if isinstance(type_enum, TensorType):
                module_type = get_module_type(type_enum)
            else:
                module_type = type_enum
            if module_type == ModuleType.INPUT_EMBED:
                # Input embeddings: base × √(m_B/m_D)
                scale = 1.0 * self._sde_lr_wd_scale
                
            elif module_type == ModuleType.POS_EMBED:
                # Positional embeddings: same as input embeddings
                scale = 1.0 * self._sde_lr_wd_scale
                
            elif module_type == ModuleType.HIDDEN_WEIGHT:
                # Hidden weights: × (base_width/fan_in) × m_L^{α-1} × √(m_B/m_D)
                width_scale = self._get_width_scale_inv(fan_in)
                scale = width_scale * self._m_L_alpha_minus_1 * self._sde_lr_wd_scale
                
            elif module_type == ModuleType.HIDDEN_BIAS:
                # Hidden biases: × m_L^{α-1} × √(m_B/m_D)
                scale = self._m_L_alpha_minus_1 * self._sde_lr_wd_scale
                
            elif module_type == ModuleType.HIDDEN_NORM:
                # Hidden norms: × m_L^{α-1} × √(m_B/m_D)
                scale = self._m_L_alpha_minus_1 * self._sde_lr_wd_scale
                
            elif module_type == ModuleType.QK_NORM:
                # QK norms: × m_L^{α-1} × √(m_B/m_D) (same as hidden norms)
                scale = self._m_L_alpha_minus_1 * self._sde_lr_wd_scale
                
            elif module_type == ModuleType.UNEMBED_WEIGHT:
                # Unembed weights: × (base_width/fan_in) × √(m_B/m_D)
                width_scale = self._get_width_scale_inv(fan_in)
                scale = width_scale * self._sde_lr_wd_scale
                
            elif module_type == ModuleType.UNEMBED_BIAS:
                # Unembed biases: base × √(m_B/m_D)
                scale = 1.0 * self._sde_lr_wd_scale
                
            elif module_type == ModuleType.UNEMBED_NORM:
                # Unembed LN: base × √(m_B/m_D)
                scale = 1.0 * self._sde_lr_wd_scale
                
            else:
                raise ValueError(f"Unknown module type: {module_type}")
            
            lr_scales[name] = jnp.array(scale, dtype=jnp.float32, device=device)
        
        return lr_scales
    
    def get_eps_rescaling(
        self,
        params_info: Dict[str, Tuple[Any, Tuple[int, int]]],
        device: Optional[Any] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Get AdamW epsilon rescaling for CompletedP.
        
        From Table 1 (using base_width/fan_in instead of global m_N^{-1}):
        - Hidden weights/biases/norms: × (base_width/fan_in) × m_L^{-α} × √(m_D/m_B)
        - QK norms: × m_L^{-α} × √(m_D/m_B)
        - Input Emb: × (base_width/fan_in) × √(m_D/m_B)
        - Output weights/biases/norms: base × √(m_D/m_B)
        
        Args:
            params_info: Dict mapping param names to (ModuleType|TensorType, (fan_in, fan_out))
                        fan_in is used to compute per-layer width scaling.
                        Accepts both ModuleType and TensorType (TensorType is converted to ModuleType).
            device: JAX device to place the arrays on (e.g., jax.devices()[0])
        
        Returns:
            Dict mapping param names to jnp.ndarray scaling factors
        """
        eps_scales = {}
        
        for name, (type_enum, (fan_in, fan_out)) in params_info.items():
            # Convert TensorType to ModuleType if needed
            if isinstance(type_enum, TensorType):
                module_type = get_module_type(type_enum)
            else:
                module_type = type_enum
            if module_type == ModuleType.INPUT_EMBED:
                # Input embeddings: × (base_width/fan_in) × √(m_D/m_B)
                # Note: for embeddings, fan_in is typically the embedding dim
                width_scale = self._get_width_scale_inv(fan_in)
                scale = width_scale * self._sde_eps_scale
                
            elif module_type == ModuleType.POS_EMBED:
                # Positional embeddings: same as input embeddings
                width_scale = self._get_width_scale_inv(fan_in)
                scale = width_scale * self._sde_eps_scale
                
            elif module_type == ModuleType.HIDDEN_WEIGHT:
                # Hidden weights: × (base_width/fan_in) × m_L^{-α} × √(m_D/m_B)
                width_scale = self._get_width_scale_inv(fan_in)
                scale = width_scale * self._m_L_neg_alpha * self._sde_eps_scale
                
            elif module_type == ModuleType.HIDDEN_BIAS:
                # Hidden biases: × (base_width/fan_in) × m_L^{-α} × √(m_D/m_B)
                width_scale = self._get_width_scale_inv(fan_in)
                scale = width_scale * self._m_L_neg_alpha * self._sde_eps_scale
                
            elif module_type == ModuleType.HIDDEN_NORM:
                # Hidden norms: × (base_width/fan_in) × m_L^{-α} × √(m_D/m_B)
                width_scale = self._get_width_scale_inv(fan_in)
                scale = width_scale * self._m_L_neg_alpha * self._sde_eps_scale
                
            elif module_type == ModuleType.QK_NORM:
                # QK norms: × m_L^{-α} × √(m_D/m_B)
                # Note: QK norms don't have width scaling per the table
                scale = self._m_L_neg_alpha * self._sde_eps_scale
                
            elif module_type == ModuleType.UNEMBED_WEIGHT:
                # Unembed weights: base × √(m_D/m_B)
                scale = 1.0 * self._sde_eps_scale
                
            elif module_type == ModuleType.UNEMBED_BIAS:
                # Unembed biases: base × √(m_D/m_B)
                scale = 1.0 * self._sde_eps_scale
                
            elif module_type == ModuleType.UNEMBED_NORM:
                # Unembed LN: base × √(m_D/m_B)
                scale = 1.0 * self._sde_eps_scale
                
            else:
                raise ValueError(f"Unknown module type: {module_type}")
            
            eps_scales[name] = jnp.array(scale, dtype=jnp.float32, device=device)
        
        return eps_scales
    
    def get_wd_rescaling(
        self,
        params_info: Dict[str, Tuple[Any, Tuple[int, int]]],
        device: Optional[Any] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Get weight decay rescaling for CompletedP.
        
        From Table 1 (using fan_in/base_width instead of global m_N):
        - Hidden weights: × (fan_in/base_width) × √(m_B/m_D)
        - Unembed weights: × (fan_in/base_width) × √(m_B/m_D)
        - Rest: × 1 × √(m_B/m_D)
        
        Args:
            params_info: Dict mapping param names to (ModuleType|TensorType, (fan_in, fan_out))
                        fan_in is used to compute per-layer width scaling.
                        Accepts both ModuleType and TensorType (TensorType is converted to ModuleType).
            device: JAX device to place the arrays on (e.g., jax.devices()[0])
        
        Returns:
            Dict mapping param names to jnp.ndarray scaling factors
        """
        wd_scales = {}
        
        for name, (type_enum, (fan_in, fan_out)) in params_info.items():
            # Convert TensorType to ModuleType if needed
            if isinstance(type_enum, TensorType):
                module_type = get_module_type(type_enum)
            else:
                module_type = type_enum
            if module_type == ModuleType.HIDDEN_WEIGHT:
                # Hidden weights: × (fan_in/base_width) × √(m_B/m_D)
                width_scale = self._get_width_scale(fan_in)
                scale = width_scale * self._sde_lr_wd_scale
                
            elif module_type == ModuleType.UNEMBED_WEIGHT:
                # Unembed weights: × (fan_in/base_width) × √(m_B/m_D)
                width_scale = self._get_width_scale(fan_in)
                scale = width_scale * self._sde_lr_wd_scale
                
            else:
                # All other parameters: × 1 × √(m_B/m_D)
                scale = 1.0 * self._sde_lr_wd_scale
            
            wd_scales[name] = jnp.array(scale, dtype=jnp.float32, device=device)
        
        return wd_scales
    
    def get_multipliers(self, device: Optional[Any] = None) -> Dict[str, jnp.ndarray]:
        """
        Get forward pass multipliers for CompletedP.
        
        From Table 1:
        - MHA Residual: m_L^{-α}
        - MLP Residual: m_L^{-α}
        - Output (Unembed): Unaugmented (1.0)
        
        Args:
            device: JAX device to place the arrays on (e.g., jax.devices()[0])
        
        Returns:
            Dictionary mapping multiplier names to jnp.ndarray scaling factors
        """
        return {
            'mha_residual_mult': jnp.array(self._m_L_neg_alpha, dtype=jnp.float32, device=device),
            'mlp_residual_mult': jnp.array(self._m_L_neg_alpha, dtype=jnp.float32, device=device),
            'output_mult': jnp.array(1.0, dtype=jnp.float32, device=device),  # Unaugmented
        }
    def get_beta_rescaling(self) -> Dict[str, float]:
        """
        Get AdamW beta rescaling for CompletedP.
        
        From Table 1:
        - (1-β₁): × m_B/m_D
        - (1-β₂): × m_B/m_D
        
        Returns:
            Dictionary with:
                - 'one_minus_beta1_mult': Multiplier for (1-β₁)
                - 'one_minus_beta2_mult': Multiplier for (1-β₂)
        """
        ratio = self.m_B / self.m_D
        return {
            'one_minus_beta1_mult': ratio,
            'one_minus_beta2_mult': ratio,
        }
    
    def _compute_one_minus_beta1_scale(
        self, 
        module_type: ModuleType, 
        fan_in: int, 
        tensor_type: Optional[TensorType],
        layer_idx: Optional[int],
        device: Any
    ) -> jnp.ndarray:
        """Compute (1-beta1) scale for a given parameter.
        
        From Table 1: (1-β₁) × m_B/m_D for all parameters.
        Depth-wise and per-tensor multipliers can further adjust values.
        
        Args:
            module_type: The module type (not used for base scaling but included for consistency)
            fan_in: Input dimension (not used for base scaling but included for consistency)
            tensor_type: The TensorType for per-tensor multiplier lookup, or None
            layer_idx: Layer index for depth-wise scaling, or None for non-layer params
            device: JAX device to place the array on
            
        Returns:
            Scaled (1-beta1) multiplier as jnp.ndarray
        """
        # Base scaling: × m_B/m_D
        scale = self.m_B / self.m_D
        
        # Apply depth-wise multiplier
        depth_mult = self._get_depth_multiplier_for_layer(layer_idx)
        scale *= depth_mult
        
        # Apply per-tensor-type multiplier
        if tensor_type is not None:
            tensor_mult = self._get_per_tensor_multiplier(tensor_type, self.per_tensor_beta1_multipliers)
            scale *= tensor_mult
        
        return jnp.array(scale, dtype=jnp.float32, device=device)
    
    def _compute_one_minus_beta2_scale(
        self, 
        module_type: ModuleType, 
        fan_in: int, 
        tensor_type: Optional[TensorType],
        layer_idx: Optional[int],
        device: Any
    ) -> jnp.ndarray:
        """Compute (1-beta2) scale for a given parameter.
        
        From Table 1: (1-β₂) × m_B/m_D for all parameters.
        Depth-wise and per-tensor multipliers can further adjust values.
        
        Args:
            module_type: The module type (not used for base scaling but included for consistency)
            fan_in: Input dimension (not used for base scaling but included for consistency)
            tensor_type: The TensorType for per-tensor multiplier lookup, or None
            layer_idx: Layer index for depth-wise scaling, or None for non-layer params
            device: JAX device to place the array on
            
        Returns:
            Scaled (1-beta2) multiplier as jnp.ndarray
        """
        # Base scaling: × m_B/m_D
        scale = self.m_B / self.m_D
        
        # Apply depth-wise multiplier
        depth_mult = self._get_depth_multiplier_for_layer(layer_idx)
        scale *= depth_mult
        
        # Apply per-tensor-type multiplier
        if tensor_type is not None:
            tensor_mult = self._get_per_tensor_multiplier(tensor_type, self.per_tensor_beta2_multipliers)
            scale *= tensor_mult
        
        return jnp.array(scale, dtype=jnp.float32, device=device)
    
    def get_one_minus_beta1_scales_pytree(
        self,
        tensor_types_pytree: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """
        Get (1-beta1) rescaling as a pytree matching the tensor_types structure.
        
        Uses the per_tensor_beta1_multipliers stored in the parameterization.
        
        Args:
            tensor_types_pytree: Pytree where each leaf is (TensorType, (fan_in, fan_out))
                                This is the output of model.get_tensor_types(params)
            device: JAX device to place the arrays on
        
        Returns:
            Pytree with same structure containing (1-beta1) scaling factors (jnp.ndarray)
        """
        def compute_scale(
            module_type: ModuleType,
            fan_in: int,
            fan_out: int,
            tensor_type: TensorType,
            layer_idx: Optional[int],
            device: Any
        ) -> jnp.ndarray:
            return self._compute_one_minus_beta1_scale(
                module_type, fan_in, tensor_type, layer_idx, device
            )

        return self._transform_tensor_types_tree_with_depth(
            tensor_types_pytree,
            compute_scale,
            device
        )

    def get_one_minus_beta2_scales_pytree(
        self,
        tensor_types_pytree: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """
        Get (1-beta2) rescaling as a pytree matching the tensor_types structure.
        
        Uses the per_tensor_beta2_multipliers stored in the parameterization.
        
        Args:
            tensor_types_pytree: Pytree where each leaf is (TensorType, (fan_in, fan_out))
                                This is the output of model.get_tensor_types(params)
            device: JAX device to place the arrays on
        
        Returns:
            Pytree with same structure containing (1-beta2) scaling factors (jnp.ndarray)
        """
        def compute_scale(
            module_type: ModuleType,
            fan_in: int,
            fan_out: int,
            tensor_type: TensorType,
            layer_idx: Optional[int],
            device: Any
        ) -> jnp.ndarray:
            return self._compute_one_minus_beta2_scale(
                module_type, fan_in, tensor_type, layer_idx, device
            )
        
        return self._transform_tensor_types_tree_with_depth(
            tensor_types_pytree,
            compute_scale,
            device
        )
    
    def get_training_iterations_scaling(self) -> float:
        """
        Get training iterations scaling.
        
        From Table 1: Training iterations ∝ m_D/m_B
        
        Returns:
            Scaling factor for training iterations
        """
        return self.m_D / self.m_B
    
    def init_params(
        self,
        params: Dict[str, Any],
        params_info: Dict[str, Tuple[ModuleType, Tuple[int, int]]],
        key: jax.random.PRNGKey,
        base_init_std: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Re-initialize parameters according to CompletedP scheme (legacy method).
        
        NOTE: This is the legacy method. For per-tensor initialization control,
        use init_params_pytree() instead which uses TensorType and per_tensor_init_multipliers.
        
        From Table 1 (Init Variances, using base_width/fan_in instead of global m_N^{-1}):
        - Input Emb: base σ²_b
        - Hidden weights: σ²_b × (base_width/fan_in)
        - Hidden biases/norms: base σ²_b
        - Unemb. LN: base σ²_b
        - Unemb. Weights: σ²_b × (base_width/fan_in)^2
        
        Args:
            params: Dictionary of parameters to re-initialize
            params_info: Dictionary mapping param paths to (ModuleType, [fan_in, fan_out])
            key: JAX random key
            base_init_std: Base initialization standard deviation
        
        Returns:
            Re-initialized parameters
        """
        
        def get_init_variance_scale(module_type: ModuleType, fan_in: int) -> float:
            """Get variance scaling factor based on module type and fan_in.
            
            IMPORTANT: Since JAX's variance_scaling gives variance = scale / fan_in,
            we need to multiply by fan_in to get the desired variance.
            
            Target variance = σ²_b × width_scale (from CompletedP table)
            variance_scaling gives: variance = scale / fan_in
            So: scale = σ²_b × width_scale × fan_in
            """
            if module_type == ModuleType.INPUT_EMBED:
                # Target: σ²_b (base variance)
                # scale = σ²_b × fan_in
                return fan_in
            elif module_type == ModuleType.POS_EMBED:
                # Target: σ²_b (base variance)
                return fan_in
            elif module_type == ModuleType.HIDDEN_WEIGHT:
                # Target: σ²_b × (base_width/fan_in)
                # scale = σ²_b × (base_width/fan_in) × fan_in = σ²_b × base_width
                return self.base_width
            elif module_type == ModuleType.HIDDEN_BIAS:
                # Target: σ²_b (base variance)
                return fan_in
            elif module_type == ModuleType.HIDDEN_NORM:
                # Norms: typically initialized to ones (handled separately)
                return fan_in
            elif module_type == ModuleType.QK_NORM:
                # QK norms: typically initialized to ones (handled separately)
                return fan_in
            elif module_type == ModuleType.UNEMBED_WEIGHT:
                # Target: σ²_b × (base_width/fan_in)²
                # scale = σ²_b × (base_width/fan_in)² × fan_in = σ²_b × base_width² / fan_in
                return (self.base_width ** 2) / fan_in
            elif module_type == ModuleType.UNEMBED_BIAS:
                # Target: σ²_b (base variance)
                return fan_in
            elif module_type == ModuleType.UNEMBED_NORM:
                # Norms: typically initialized to ones (handled separately)
                return fan_in
            else:
                raise ValueError(f"Unknown module type: {module_type}")
        
        # Flatten params for easier processing
        flat_params = {}
        
        def flatten_dict(d, prefix=''):
            for k, v in d.items():
                full_key = f"{prefix}/{k}" if prefix else k
                if isinstance(v, dict):
                    flatten_dict(v, full_key)
                else:
                    flat_params[full_key] = v
        
        flatten_dict(params)
        
        # Re-initialize each parameter
        new_flat_params = {}
        keys = jax.random.split(key, len(flat_params))
        
        for i, (param_path, param_value) in enumerate(flat_params.items()):
            # Find matching info entry
            param_info = None
            for info_key, info_value in params_info.items():
                if info_key in param_path or param_path.endswith(info_key):
                    param_info = info_value
                    break
            
            if param_info is None:
                # If no specific info, keep original
                new_flat_params[param_path] = param_value
                continue
            
            module_type, (fan_in, fan_out) = param_info
            variance_scale = get_init_variance_scale(module_type, fan_in)
            
            # Handle different parameter types
            if module_type in [ModuleType.HIDDEN_NORM, ModuleType.QK_NORM, 
                              ModuleType.UNEMBED_NORM]:
                # Normalization parameters: initialize to ones
                new_flat_params[param_path] = jnp.ones_like(param_value)
                
            elif module_type == ModuleType.UNEMBED_WEIGHT:
                # Output projection: scaled variance
                # variance_scaling gives variance = scale / fan_in
                # We want variance = σ²_b × (base_width/fan_in)²
                init_scale = base_init_std ** 2 * variance_scale
                initializer = nn.initializers.variance_scaling(
                    init_scale, "fan_in", "truncated_normal"
                )
                new_flat_params[param_path] = initializer(
                    keys[i], param_value.shape, param_value.dtype
                )
                
            elif 'bias' in param_path.lower():
                # Biases: initialize to zeros
                new_flat_params[param_path] = jnp.zeros_like(param_value)
                
            else:
                # Weights: use variance scaling
                init_scale = base_init_std ** 2 * variance_scale
                initializer = nn.initializers.variance_scaling(
                    init_scale, "fan_in", "truncated_normal"
                )
                new_flat_params[param_path] = initializer(
                    keys[i], param_value.shape, param_value.dtype
                )
        
        # Unflatten back to nested dict
        def unflatten_dict(flat_dict):
            result = {}
            for key, value in flat_dict.items():
                parts = key.split('/')
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            return result
        
        return unflatten_dict(new_flat_params)
    
    def _get_init_scale_for_tensor_type(
        self,
        tensor_type: TensorType,
        module_type: ModuleType,
        fan_in: int,
        layer_idx: Optional[int],
        base_init_std: float,
    ) -> float:
        """
        Get the initialization scale for variance_scaling initializer based on TensorType.
        
        IMPORTANT: JAX's variance_scaling with mode="fan_in" gives:
            variance = scale / fan_in
        
        From CompletedP Table 1 (Init Variances):
        - Input Emb: σ²_b (base variance)
        - Hidden weights: σ²_b × (base_width/fan_in)
        - Hidden biases/norms: σ²_b
        - Unemb. LN: σ²_b
        - Unemb. Weights: σ²_b × (base_width/fan_in)²
        
        To get target variance V with variance_scaling("fan_in"):
            scale / fan_in = V
            scale = V × fan_in
        
        Args:
            tensor_type: The TensorType for per-tensor multiplier lookup
            module_type: The ModuleType for base scaling rules
            fan_in: Input dimension of the layer
            layer_idx: Layer index for depth-wise scaling, or None for non-layer params
            base_init_std: Base initialization standard deviation (σ_b)
        
        Returns:
            The scale parameter to pass to variance_scaling initializer
        """
        base_variance = base_init_std ** 2  # σ²_b
        
        # Compute target variance based on module type
        if module_type in [ModuleType.INPUT_EMBED, ModuleType.POS_EMBED]:
            # Target: σ²_b
            target_variance = base_variance
        elif module_type == ModuleType.HIDDEN_WEIGHT:
            # Target: σ²_b × (base_width/fan_in)
            target_variance = base_variance * self.base_width / fan_in
        elif module_type in [ModuleType.HIDDEN_BIAS, ModuleType.HIDDEN_NORM, ModuleType.QK_NORM]:
            # Target: σ²_b (norms typically initialized to ones, handled separately)
            target_variance = base_variance
        elif module_type == ModuleType.UNEMBED_WEIGHT:
            # Target: σ²_b × (base_width/fan_in)²
            target_variance = base_variance * (self.base_width / fan_in) ** 2
        elif module_type in [ModuleType.UNEMBED_BIAS, ModuleType.UNEMBED_NORM]:
            # Target: σ²_b
            target_variance = base_variance
        else:
            # Default to base variance
            target_variance = base_variance
        
        # Apply depth-wise multiplier if applicable
        depth_mult = self._get_depth_multiplier_for_layer(layer_idx)
        target_variance *= depth_mult
        
        # Apply per-tensor-type multiplier
        tensor_mult = self._get_per_tensor_multiplier(tensor_type, self.per_tensor_init_multipliers)
        target_variance *= tensor_mult
        
        # Convert target variance to scale for variance_scaling initializer
        # variance_scaling gives: variance = scale / fan_in
        # So: scale = target_variance × fan_in
        scale = target_variance * fan_in
        
        return scale
    
    def init_params_pytree(
        self,
        params: Any,
        tensor_types_pytree: Any,
        key: jax.random.PRNGKey,
        base_init_std: float = 0.02,
    ) -> Any:
        """
        Re-initialize parameters in-place according to CompletedP scheme using TensorTypes.
        
        This method walks through the params pytree and tensor_types_pytree in parallel,
        re-initializing each parameter according to its TensorType. It applies:
        - CompletedP base scaling rules based on ModuleType
        - Depth-wise multipliers from depth_multipliers
        - Per-tensor multipliers from per_tensor_init_multipliers
        
        From Table 1 (Init Variances):
        - Input Emb: σ²_b (base variance)
        - Hidden weights: σ²_b × (base_width/fan_in)
        - Hidden biases/norms: σ²_b
        - Unemb. LN: σ²_b
        - Unemb. Weights: σ²_b × (base_width/fan_in)²
        
        Args:
            params: Pytree of model parameters to re-initialize
            tensor_types_pytree: Pytree with same structure as params, where each leaf
                                is (TensorType, (fan_in, fan_out)). This is the output
                                of model.get_tensor_types(params).
            key: JAX random key for initialization
            base_init_std: Base initialization standard deviation (σ_b)
        
        Returns:
            Re-initialized params pytree with same structure
        
        Example:
            >>> tensor_types = model.get_tensor_types(params)
            >>> key = jax.random.PRNGKey(42)
            >>> new_params = parameterization.init_params_pytree(
            ...     params, tensor_types, key, base_init_std=0.02
            ... )
        """
        # Count leaves for key splitting
        num_leaves = len(jax.tree_util.tree_leaves(params))
        keys = jax.random.split(key, num_leaves)
        key_iter = iter(keys)
        
        def init_leaf(
            param: jnp.ndarray,
            tensor_info: Any,
            current_layer_idx: Optional[int],
        ) -> jnp.ndarray:
            """Initialize a single parameter based on its tensor info."""
            # MoE expert kernels (wi_0, wi_1, wo) and the gate DenseGeneral kernel
            # are wrapped in flax.linen.spmd.LogicallyPartitioned via
            # nn.with_logical_partitioning. Unwrap for shape/dtype access and
            # rebox the new value so the params pytree structure is preserved
            # (downstream get_mup_lrs / optimizer state expect the wrapper).
            is_wrapped = hasattr(param, 'value') and hasattr(param, 'replace')
            wrapper = param if is_wrapped else None
            if is_wrapped:
                param = param.value

            # Extract TensorType and fan dimensions
            tensor_type_info = self._extract_tensor_info(tensor_info)
            if tensor_type_info is None:
                # Fallback: try module info
                module_info = self._extract_module_info(tensor_info)
                if module_info is not None:
                    module_type, fan_in, fan_out = module_info
                    # Use a default tensor type based on module type
                    tensor_type = self._get_default_tensor_type(module_type)
                else:
                    # Can't identify this parameter, keep original.
                    # Return unwrapped to keep the result tree uniform.
                    return param
            else:
                tensor_type, fan_in, fan_out = tensor_type_info
                module_type = get_module_type(tensor_type)

            # Get the next random key
            param_key = next(key_iter)

            # Intentionally do NOT re-box: returning unwrapped values strips
            # the LogicallyPartitioned wrappers from MoE leaves so that the
            # downstream optimizer state (state.mu/state.nu in mup_adamw) and
            # the CompletedP scales tree (which has plain jnp leaves) all
            # share a uniform unwrapped pytree structure. tree_map across
            # those would otherwise raise a "Custom dataclass node type
            # mismatch" because mu was created from a wrapped param tree
            # while the scales tree is unwrapped. _compute_completed_p_scales
            # was already called and cached BEFORE _reinit_params_with_completed_p
            # in MuTransformerMoETask.init_with_state, so no downstream
            # consumer needs the wrappers.
            def _rebox(v):
                return v

            # Handle normalization parameters specially (initialize to ones)
            if module_type in [ModuleType.HIDDEN_NORM, ModuleType.QK_NORM, ModuleType.UNEMBED_NORM]:
                return _rebox(jnp.ones_like(param))

            # Handle bias parameters + output layer (initialize to zeros)
            # Check if this is a bias by looking at shape and module type
            if module_type in [ModuleType.HIDDEN_BIAS, ModuleType.UNEMBED_BIAS,  ModuleType.UNEMBED_WEIGHT ]:
                return _rebox(jnp.zeros_like(param))
            
            # For weights: compute per-element target stddev directly. We can NOT
            # use nn.initializers.variance_scaling("fan_in") because for MoE
            # expert-stacked kernels of shape (num_experts, emb_dim, mlp_dim)
            # flax computes fan_in = prod(shape[:-1]) = num_experts*emb_dim
            # instead of the true per-expert fan_in = emb_dim, under-initialising
            # experts by sqrt(num_experts). truncated_normal with explicit stddev
            # is shape-agnostic and correct for both 2D dense and N-D expert kernels.
            scale = self._get_init_scale_for_tensor_type(
                tensor_type=tensor_type,
                module_type=module_type,
                fan_in=fan_in,
                layer_idx=current_layer_idx,
                base_init_std=base_init_std,
            )
            # _get_init_scale_for_tensor_type returns scale = target_variance * fan_in
            # (so variance_scaling(scale, "fan_in") would yield variance = target_variance).
            # Recover target_variance and use truncated_normal with explicit stddev.
            target_variance = float(scale) / float(fan_in)
            stddev = math.sqrt(max(target_variance, 0.0))
            initializer = nn.initializers.truncated_normal(stddev=stddev)
            return _rebox(initializer(param_key, param.shape, param.dtype))
        
        def walk_and_init(
            params_tree: Any,
            tensor_types_tree: Any,
            current_layer_idx: Optional[int] = None,
        ) -> Any:
            """Recursively walk params and tensor_types trees in parallel."""
            if isinstance(params_tree, dict) and isinstance(tensor_types_tree, dict):
                result = {}
                for k in params_tree.keys():
                    if k not in tensor_types_tree:
                        # Keep original if no tensor type info
                        result[k] = params_tree[k]
                        continue
                    
                    # Check if this key indicates a layer (e.g., 'blocks_0')
                    layer_idx = self._extract_layer_idx_from_key(k)
                    if layer_idx is not None:
                        # Update layer index for children
                        result[k] = walk_and_init(
                            params_tree[k], tensor_types_tree[k], layer_idx
                        )
                    else:
                        # Preserve current layer index
                        result[k] = walk_and_init(
                            params_tree[k], tensor_types_tree[k], current_layer_idx
                        )
                return result
            
            # We've reached a leaf in the params tree
            # tensor_types_tree should contain (TensorType, (fan_in, fan_out))
            return init_leaf(params_tree, tensor_types_tree, current_layer_idx)
        
        return walk_and_init(params, tensor_types_pytree)
    
    def _get_default_tensor_type(self, module_type: ModuleType) -> TensorType:
        """Get a default TensorType for a ModuleType when TensorType is not available."""
        # This provides backwards compatibility for code that uses ModuleType
        module_to_tensor = {
            ModuleType.INPUT_EMBED: TensorType.EMBEDDING,
            ModuleType.POS_EMBED: TensorType.POS_EMBEDDING,
            ModuleType.HIDDEN_WEIGHT: TensorType.MLP_UP,  # Default hidden weight
            ModuleType.HIDDEN_BIAS: TensorType.MLP_UP,
            ModuleType.HIDDEN_NORM: TensorType.POST_ATTENTION_NORM,
            ModuleType.QK_NORM: TensorType.ATTENTION_QUERY_NORM,
            ModuleType.UNEMBED_WEIGHT: TensorType.UNEMBEDDING,
            ModuleType.UNEMBED_BIAS: TensorType.UNEMBEDDING,
            ModuleType.UNEMBED_NORM: TensorType.OUTPUT_NORM,
        }
        return module_to_tensor.get(module_type, TensorType.MLP_UP)
    
    def _extract_module_info(self, leaf):
        """Extract (ModuleType, fan_in, fan_out) from a module_types or tensor_types leaf.
        
        This method handles both ModuleType and TensorType leaves:
        - If leaf is (ModuleType, (fan_in, fan_out)), returns as-is
        - If leaf is (TensorType, (fan_in, fan_out)), converts to ModuleType using mapping
        """
        # Handle LogicallyPartitioned values from Flax
        if hasattr(leaf, 'value'):
            inner = leaf.value
        else:
            inner = leaf
        
        # inner should be (ModuleType|TensorType, (fan_in, fan_out))
        if isinstance(inner, tuple) and len(inner) == 2:
            type_enum, fan_dims = inner
            
            # Handle TensorType by converting to ModuleType
            if isinstance(type_enum, TensorType):
                type_enum = get_module_type(type_enum)
            
            if isinstance(type_enum, ModuleType):
                if isinstance(fan_dims, tuple) and len(fan_dims) == 2:
                    fan_in, fan_out = fan_dims
                    return type_enum, fan_in, fan_out
        
        return None  # Not a valid module info leaf
    
    def _extract_tensor_info(self, leaf):
        """Extract (TensorType, fan_in, fan_out) from a tensor_types leaf.
        
        This method extracts the original TensorType without converting to ModuleType,
        useful for per-tensor LR overrides.
        """
        # Handle LogicallyPartitioned values from Flax
        if hasattr(leaf, 'value'):
            inner = leaf.value
        else:
            inner = leaf
        
        # inner should be (TensorType, (fan_in, fan_out))
        if isinstance(inner, tuple) and len(inner) == 2:
            tensor_type, fan_dims = inner
            if isinstance(tensor_type, TensorType):
                if isinstance(fan_dims, tuple) and len(fan_dims) == 2:
                    fan_in, fan_out = fan_dims
                    return tensor_type, fan_in, fan_out
        
        return None  # Not a valid tensor info leaf
    
    def _compute_lr_scale(self, module_type: ModuleType, fan_in: int, device: Any) -> jnp.ndarray:
        """Compute LR scale for a given module type and fan_in."""
        if module_type == ModuleType.INPUT_EMBED:
            scale = 1.0 * self._sde_lr_wd_scale
        elif module_type == ModuleType.POS_EMBED:
            scale = 1.0 * self._sde_lr_wd_scale
        elif module_type == ModuleType.HIDDEN_WEIGHT:
            width_scale = self._get_width_scale_inv(fan_in)
            scale = width_scale * self._m_L_alpha_minus_1 * self._sde_lr_wd_scale
            # print(f"hidden weight scale: {scale}")
            # print(f"width scale: {width_scale}")
            # print(f"m_L_alpha_minus_1: {self._m_L_alpha_minus_1}")
            # print(f"sde_lr_wd_scale: {self._sde_lr_wd_scale}")
            # print(f"fan_in: {fan_in}")
            # exit(0)
        elif module_type == ModuleType.HIDDEN_BIAS:
            scale = self._m_L_alpha_minus_1 * self._sde_lr_wd_scale
        elif module_type == ModuleType.HIDDEN_NORM:
            scale = self._m_L_alpha_minus_1 * self._sde_lr_wd_scale
        elif module_type == ModuleType.QK_NORM:
            scale = self._m_L_alpha_minus_1 * self._sde_lr_wd_scale
        elif module_type == ModuleType.UNEMBED_WEIGHT:
            width_scale = self._get_width_scale_inv(fan_in)
            scale = width_scale * self._sde_lr_wd_scale
        elif module_type == ModuleType.UNEMBED_BIAS:
            scale = 1.0 * self._sde_lr_wd_scale
        elif module_type == ModuleType.UNEMBED_NORM:
            scale = 1.0 * self._sde_lr_wd_scale
        else:
            raise ValueError(f"Unknown module type: {module_type}")
        
        return jnp.array(scale, dtype=jnp.float32, device=device)
    
    def _compute_eps_scale(self, module_type: ModuleType, fan_in: int, device: Any) -> jnp.ndarray:
        """Compute epsilon scale for a given module type and fan_in."""
        if module_type == ModuleType.INPUT_EMBED:
            width_scale = self._get_width_scale_inv(fan_in)
            scale = width_scale * self._sde_eps_scale
        elif module_type == ModuleType.POS_EMBED:
            width_scale = self._get_width_scale_inv(fan_in)
            scale = width_scale * self._sde_eps_scale
        elif module_type == ModuleType.HIDDEN_WEIGHT:
            width_scale = self._get_width_scale_inv(fan_in)
            scale = width_scale * self._m_L_neg_alpha * self._sde_eps_scale
        elif module_type == ModuleType.HIDDEN_BIAS:
            width_scale = self._get_width_scale_inv(fan_in)
            scale = width_scale * self._m_L_neg_alpha * self._sde_eps_scale
        elif module_type == ModuleType.HIDDEN_NORM:
            width_scale = self._get_width_scale_inv(fan_in)
            scale = width_scale * self._m_L_neg_alpha * self._sde_eps_scale
        elif module_type == ModuleType.QK_NORM:
            scale = self._m_L_neg_alpha * self._sde_eps_scale
        elif module_type == ModuleType.UNEMBED_WEIGHT:
            scale = 1.0 * self._sde_eps_scale
        elif module_type == ModuleType.UNEMBED_BIAS:
            scale = 1.0 * self._sde_eps_scale
        elif module_type == ModuleType.UNEMBED_NORM:
            scale = 1.0 * self._sde_eps_scale
        else:
            raise ValueError(f"Unknown module type: {module_type}")
        
        return jnp.array(scale, dtype=jnp.float32, device=device)
    
    def _compute_wd_scale(self, module_type: ModuleType, fan_in: int, device: Any) -> jnp.ndarray:
        """Compute weight decay scale for a given module type and fan_in."""
        if module_type == ModuleType.HIDDEN_WEIGHT:
            width_scale = self._get_width_scale(fan_in)
            scale = width_scale * self._sde_lr_wd_scale
        elif module_type == ModuleType.UNEMBED_WEIGHT:
            width_scale = self._get_width_scale(fan_in)
            scale = width_scale * self._sde_lr_wd_scale
        else:
            # All other parameters: × 1 × √(m_B/m_D)
            scale = 1.0 * self._sde_lr_wd_scale
        
        return jnp.array(scale, dtype=jnp.float32, device=device)
    
    def _transform_module_types_tree(
        self,
        tree: Any,
        scale_fn: Callable[[ModuleType, int, Any], jnp.ndarray],
        device: Any,
    ) -> Any:
        """
        Transform a module_types tree by applying a scale function to each leaf.
        
        This manually walks the tree structure because JAX's tree_map would
        flatten the (ModuleType, (fan_in, fan_out)) tuples.
        """
        if isinstance(tree, dict):
            return {k: self._transform_module_types_tree(v, scale_fn, device) 
                    for k, v in tree.items()}
        
        # Try to extract module info from this leaf
        info = self._extract_module_info(tree)
        if info is not None:
            module_type, fan_in, fan_out = info
            return scale_fn(module_type, fan_in, device)
        
        # Not a recognized leaf - might be a nested structure we missed
        # Check for other container types
        if isinstance(tree, (list, tuple)):
            # Only recurse if it's a container of dicts (not the (ModuleType, (fan_in, fan_out)) tuple)
            if len(tree) > 0 and isinstance(tree[0], dict):
                return type(tree)(self._transform_module_types_tree(item, scale_fn, device) 
                                  for item in tree)
        
        # Fallback: return tree as-is or raise error
        raise ValueError(f"Unexpected tree node: {type(tree)}: {tree}")
    
    def _extract_layer_idx_from_key(self, key: str) -> Optional[int]:
        """
        Extract the layer index from a key like 'blocks_0', 'blocks_12', etc.
        
        Args:
            key: A key from the parameter tree
            
        Returns:
            Layer index (int) if the key matches 'blocks_N' pattern, else None
        """
        import re
        match = re.match(r'^blocks_(\d+)$', key)
        if match:
            return int(match.group(1))
        return None
    
    def _transform_module_types_tree_with_depth(
        self,
        tree: Any,
        scale_fn: Callable[[ModuleType, int, Optional[int], Any], jnp.ndarray],
        device: Any,
        current_layer_idx: Optional[int] = None,
    ) -> Any:
        """
        Transform a module_types tree by applying a scale function that includes layer index.
        
        This version supports depth-wise scaling by tracking the current layer index
        as it walks through the tree structure.
        
        Args:
            tree: The tree to transform
            scale_fn: Function that takes (module_type, fan_in, layer_idx, device)
            device: JAX device to place arrays on
            current_layer_idx: Current layer index (updated when traversing 'blocks_N' keys)
        """
        if isinstance(tree, dict):
            result = {}
            for k, v in tree.items():
                # Check if this key indicates a layer (e.g., 'blocks_0')
                layer_idx = self._extract_layer_idx_from_key(k)
                if layer_idx is not None:
                    # Update layer index for children
                    result[k] = self._transform_module_types_tree_with_depth(
                        v, scale_fn, device, layer_idx
                    )
                else:
                    # Preserve current layer index
                    result[k] = self._transform_module_types_tree_with_depth(
                        v, scale_fn, device, current_layer_idx
                    )
            return result
        
        # Try to extract module info from this leaf
        info = self._extract_module_info(tree)
        if info is not None:
            module_type, fan_in, fan_out = info
            return scale_fn(module_type, fan_in, current_layer_idx, device)
        
        # Not a recognized leaf - might be a nested structure we missed
        if isinstance(tree, (list, tuple)):
            if len(tree) > 0 and isinstance(tree[0], dict):
                return type(tree)(
                    self._transform_module_types_tree_with_depth(item, scale_fn, device, current_layer_idx)
                    for item in tree
                )
        
        raise ValueError(f"Unexpected tree node: {type(tree)}: {tree}")
    
    def _transform_tensor_types_tree_with_depth(
        self,
        tree: Any,
        scale_fn: Callable[[ModuleType, int, int, TensorType, Optional[int], Any], jnp.ndarray],
        device: Any,
        current_layer_idx: Optional[int] = None,
    ) -> Any:
        """
        Transform a tensor_types tree by applying a scale function that includes layer index.

        This version preserves TensorType information and supports depth-wise scaling.

        Args:
            tree: The tree to transform
            scale_fn: Function that takes (module_type, fan_in, fan_out, tensor_type, layer_idx, device)
            device: JAX device to place arrays on
            current_layer_idx: Current layer index
        """
        if isinstance(tree, dict):
            result = {}
            for k, v in tree.items():
                layer_idx = self._extract_layer_idx_from_key(k)
                if layer_idx is not None:
                    result[k] = self._transform_tensor_types_tree_with_depth(
                        v, scale_fn, device, layer_idx
                    )
                else:
                    result[k] = self._transform_tensor_types_tree_with_depth(
                        v, scale_fn, device, current_layer_idx
                    )
            return result

        # Try to extract tensor info from this leaf
        tensor_info = self._extract_tensor_info(tree)
        if tensor_info is not None:
            tensor_type, fan_in, fan_out = tensor_info
            module_type = get_module_type(tensor_type)
            return scale_fn(module_type, fan_in, fan_out, tensor_type, current_layer_idx, device)

        # Also try module info (backward compatibility)
        module_info = self._extract_module_info(tree)
        if module_info is not None:
            module_type, fan_in, fan_out = module_info
            # Create a dummy tensor type call - just use module_type based scaling
            return self._compute_lr_scale(module_type, fan_in, device)

        if isinstance(tree, (list, tuple)):
            if len(tree) > 0 and isinstance(tree[0], dict):
                return type(tree)(
                    self._transform_tensor_types_tree_with_depth(item, scale_fn, device, current_layer_idx)
                    for item in tree
                )

        raise ValueError(f"Unexpected tree node: {type(tree)}: {tree}")
    
    # =========================================================================
    # TensorType-aware methods for per-tensor hyperparameter control
    # =========================================================================
    
    def get_lr_scales_pytree(
        self,
        tensor_types_pytree: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """
        Get LR rescaling as a pytree matching the tensor_types structure.
        
        Uses the per_tensor_lr_multipliers and depth_multipliers stored in the parameterization.
        
        Args:
            tensor_types_pytree: Pytree where each leaf is (TensorType, (fan_in, fan_out))
                                This is the output of model.get_tensor_types(params)
            device: JAX device to place the arrays on
        
        Returns:
            Pytree with same structure containing LR scaling factors (jnp.ndarray)
        
        Example:
            tensor_types = model.get_tensor_types(params)
            lr_scales = parameterization.get_lr_scales_pytree(tensor_types, device=device)
        """
        def compute_scale(
            module_type: ModuleType,
            fan_in: int,
            fan_out: int,
            tensor_type: TensorType,
            layer_idx: Optional[int],
            device: Any
        ) -> jnp.ndarray:
            # Get base scale from ModuleType
            base_scale = self._compute_lr_scale(module_type, fan_in, device)

            # Apply depth-wise multiplier
            depth_mult = self._get_depth_multiplier_for_layer(layer_idx)
            base_scale = base_scale * depth_mult

            # Apply per-tensor-type multiplier
            tensor_mult = self._get_per_tensor_multiplier(tensor_type, self.per_tensor_lr_multipliers)
            return base_scale * tensor_mult

        return self._transform_tensor_types_tree_with_depth(
            tensor_types_pytree,
            compute_scale,
            device
        )

    def get_eps_scales_pytree(
        self,
        tensor_types_pytree: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """
        Get epsilon rescaling as a pytree matching the tensor_types structure.
        
        Uses the per_tensor_eps_multipliers and depth_multipliers stored in the parameterization.
        
        Args:
            tensor_types_pytree: Pytree where each leaf is (TensorType, (fan_in, fan_out))
            device: JAX device to place the arrays on
        
        Returns:
            Pytree with same structure containing epsilon scaling factors (jnp.ndarray)
        """
        def compute_scale(
            module_type: ModuleType,
            fan_in: int,
            fan_out: int,
            tensor_type: TensorType,
            layer_idx: Optional[int],
            device: Any
        ) -> jnp.ndarray:
            base_scale = self._compute_eps_scale(module_type, fan_in, device)

            # Apply depth-wise multiplier
            depth_mult = self._get_depth_multiplier_for_layer(layer_idx)
            base_scale = base_scale * depth_mult

            # Apply per-tensor-type multiplier
            tensor_mult = self._get_per_tensor_multiplier(tensor_type, self.per_tensor_eps_multipliers)
            return base_scale * tensor_mult

        return self._transform_tensor_types_tree_with_depth(
            tensor_types_pytree,
            compute_scale,
            device
        )

    def get_wd_scales_pytree(
        self,
        tensor_types_pytree: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """
        Get weight decay rescaling as a pytree matching the tensor_types structure.
        
        Uses the per_tensor_wd_multipliers and depth_multipliers stored in the parameterization.
        
        Args:
            tensor_types_pytree: Pytree where each leaf is (TensorType, (fan_in, fan_out))
            device: JAX device to place the arrays on
        
        Returns:
            Pytree with same structure containing weight decay scaling factors (jnp.ndarray)
        """
        def compute_scale(
            module_type: ModuleType,
            fan_in: int,
            fan_out: int,
            tensor_type: TensorType,
            layer_idx: Optional[int],
            device: Any
        ) -> jnp.ndarray:
            base_scale = self._compute_wd_scale(module_type, fan_in, device)

            # Apply depth-wise multiplier
            depth_mult = self._get_depth_multiplier_for_layer(layer_idx)
            base_scale = base_scale * depth_mult

            # Apply per-tensor-type multiplier
            tensor_mult = self._get_per_tensor_multiplier(tensor_type, self.per_tensor_wd_multipliers)
            return base_scale * tensor_mult

        return self._transform_tensor_types_tree_with_depth(
            tensor_types_pytree,
            compute_scale,
            device
        )

    def get_init_scales_pytree(
        self,
        tensor_types_pytree: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """
        Get initialization variance rescaling as a pytree matching the tensor_types structure.
        
        Uses the per_tensor_init_multipliers and depth_multipliers stored in the parameterization.
        
        From Table 1 (Init Variances):
        - Input Emb: base σ²_b (scale = 1.0)
        - Hidden weights: σ²_b × (base_width/fan_in)
        - Hidden biases/norms: base σ²_b (scale = 1.0)
        - Unemb. LN: base σ²_b (scale = 1.0)
        - Unemb. Weights: σ²_b × (base_width/fan_in)² 
        
        Args:
            tensor_types_pytree: Pytree where each leaf is (TensorType, (fan_in, fan_out))
            device: JAX device to place the arrays on
        
        Returns:
            Pytree with same structure containing init scaling factors (jnp.ndarray)
        """
        def compute_init_scale(
            module_type: ModuleType,
            fan_in: int,
            fan_out: int,
            tensor_type: TensorType,
            layer_idx: Optional[int],
            device: Any
        ) -> jnp.ndarray:
            # Base init variance scaling
            if module_type in [ModuleType.INPUT_EMBED, ModuleType.POS_EMBED]:
                scale = 1.0
            elif module_type == ModuleType.HIDDEN_WEIGHT:
                scale = self._get_width_scale_inv(fan_in)
            elif module_type in [ModuleType.HIDDEN_BIAS, ModuleType.HIDDEN_NORM, ModuleType.QK_NORM]:
                scale = 1.0
            elif module_type == ModuleType.UNEMBED_WEIGHT:
                scale = self._get_width_scale_inv_sq(fan_in)
            elif module_type in [ModuleType.UNEMBED_BIAS, ModuleType.UNEMBED_NORM]:
                scale = 1.0
            else:
                scale = 1.0
            
            # Apply depth-wise multiplier
            depth_mult = self._get_depth_multiplier_for_layer(layer_idx)
            scale *= depth_mult
            
            # Apply per-tensor-type multiplier
            tensor_mult = self._get_per_tensor_multiplier(tensor_type, self.per_tensor_init_multipliers)
            scale *= tensor_mult
            
            return jnp.array(scale, dtype=jnp.float32, device=device)
        
        return self._transform_tensor_types_tree_with_depth(
            tensor_types_pytree,
            compute_init_scale,
            device
        )
    
    def _transform_tensor_types_tree(
        self,
        tree: Any,
        scale_fn: Callable[[ModuleType, int, TensorType, Any], jnp.ndarray],
        device: Any,
    ) -> Any:
        """
        Transform a tensor_types tree by applying a scale function to each leaf.
        
        Unlike _transform_module_types_tree, this preserves TensorType information
        and passes it to the scale function for per-tensor overrides.
        """
        if isinstance(tree, dict):
            return {k: self._transform_tensor_types_tree(v, scale_fn, device) 
                    for k, v in tree.items()}
        
        # Try to extract tensor info from this leaf
        tensor_info = self._extract_tensor_info(tree)
        if tensor_info is not None:
            tensor_type, fan_in, fan_out = tensor_info
            module_type = get_module_type(tensor_type)
            return scale_fn(module_type, fan_in, tensor_type, device)
        
        # Also try module info (backward compatibility)
        module_info = self._extract_module_info(tree)
        if module_info is not None:
            module_type, fan_in, fan_out = module_info
            # Create a dummy tensor type - use the standard mapping
            return self._compute_lr_scale(module_type, fan_in, device)
        
        # Not a recognized leaf - might be a nested structure we missed
        if isinstance(tree, (list, tuple)):
            if len(tree) > 0 and isinstance(tree[0], dict):
                return type(tree)(self._transform_tensor_types_tree(item, scale_fn, device) 
                                  for item in tree)
        
        raise ValueError(f"Unexpected tree node: {type(tree)}: {tree}")
    
    # Legacy methods using classify_param_fn (kept for backwards compatibility)
    def apply_lr_rescaling_to_pytree(
        self,
        params: Any,
        classify_param_fn: Callable[[str, Any], Tuple[ModuleType, Tuple[int, int]]],
        device: Optional[Any] = None,
    ) -> Any:
        """
        Apply LR rescaling to a pytree of parameters.
        
        This is a convenience method that walks through a parameter pytree,
        classifies each parameter using the provided function, and returns
        a pytree of learning rate multipliers with the same structure.
        
        Args:
            params: Parameter pytree
            classify_param_fn: Function that takes (param_path, param_value) and
                              returns (ModuleType, [fan_in, fan_out])
            device: JAX device to place the arrays on (e.g., jax.devices()[0])
        
        Returns:
            Pytree with same structure containing LR scaling factors (jnp.ndarray)
        """
        def apply_scale(path, value):
            # Convert path tuple to string
            path_str = '/'.join(str(p) for p in path)
            module_type, fan_dims = classify_param_fn(path_str, value)
            
            # Build single-item params_info and get scaling
            params_info = {path_str: (module_type, fan_dims)}
            scales = self.get_lr_rescaling(params_info, device=device)
            
            return scales[path_str]
        
        return jax.tree_util.tree_map_with_path(apply_scale, params)
    
    def apply_eps_rescaling_to_pytree(
        self,
        params: Any,
        classify_param_fn: Callable[[str, Any], Tuple[ModuleType, Tuple[int, int]]],
        device: Optional[Any] = None,
    ) -> Any:
        """
        Apply epsilon rescaling to a pytree of parameters.
        
        Args:
            params: Parameter pytree
            classify_param_fn: Function that takes (param_path, param_value) and
                              returns (ModuleType, [fan_in, fan_out])
            device: JAX device to place the arrays on (e.g., jax.devices()[0])
        
        Returns:
            Pytree with same structure containing epsilon scaling factors (jnp.ndarray)
        """
        def apply_scale(path, value):
            path_str = '/'.join(str(p) for p in path)
            module_type, fan_dims = classify_param_fn(path_str, value)
            
            params_info = {path_str: (module_type, fan_dims)}
            scales = self.get_eps_rescaling(params_info, device=device)
            
            return scales[path_str]
        
        return jax.tree_util.tree_map_with_path(apply_scale, params)
    
    def apply_wd_rescaling_to_pytree(
        self,
        params: Any,
        classify_param_fn: Callable[[str, Any], Tuple[ModuleType, Tuple[int, int]]],
        device: Optional[Any] = None,
    ) -> Any:
        """
        Apply weight decay rescaling to a pytree of parameters.
        
        Args:
            params: Parameter pytree
            classify_param_fn: Function that takes (param_path, param_value) and
                              returns (ModuleType, [fan_in, fan_out])
            device: JAX device to place the arrays on (e.g., jax.devices()[0])
        
        Returns:
            Pytree with same structure containing weight decay scaling factors (jnp.ndarray)
        """
        def apply_scale(path, value):
            path_str = '/'.join(str(p) for p in path)
            module_type, fan_dims = classify_param_fn(path_str, value)
            
            params_info = {path_str: (module_type, fan_dims)}
            scales = self.get_wd_rescaling(params_info, device=device)
            
            return scales[path_str]
        
        return jax.tree_util.tree_map_with_path(apply_scale, params)


def classify_transformer_param(
    param_path: str,
    param_value: Any,
) -> Tuple[ModuleType, Tuple[int, int]]:
    """
    Example classifier function for transformer parameters.
    
    This function classifies transformer parameters based on common naming conventions.
    Customize this for your specific model architecture.
    
    Args:
        param_path: Path string like "blocks_0/CausalAttn_0/query/kernel"
        param_value: The parameter array
    
    Returns:
        Tuple of (ModuleType, [fan_in, fan_out])
    """
    path_lower = param_path.lower()
    shape = param_value.shape
    
    # Determine fan_in and fan_out from shape
    if len(shape) >= 2:
        fan_in = shape[0]
        fan_out = shape[-1]
    else:
        fan_in = fan_out = shape[0] if len(shape) > 0 else 1
    
    # Classify based on path
    if 'embed' in path_lower and 'unembed' not in path_lower and 'output' not in path_lower:
        if 'pos' in path_lower:
            return (ModuleType.POS_EMBED, (fan_in, fan_out))
        return (ModuleType.INPUT_EMBED, (fan_in, fan_out))
    
    if 'output_proj' in path_lower or 'unembed' in path_lower:
        if 'norm' in path_lower or 'ln' in path_lower:
            return (ModuleType.UNEMBED_NORM, (fan_in, fan_out))
        if 'bias' in path_lower:
            return (ModuleType.UNEMBED_BIAS, (fan_in, fan_out))
        return (ModuleType.UNEMBED_WEIGHT, (fan_in, fan_out))
    
    if 'out_ln' in path_lower or ('norm' in path_lower and 'out' in path_lower):
        return (ModuleType.UNEMBED_NORM, (fan_in, fan_out))
    
    # QK norms in attention
    if 'rmsnorm' in path_lower and ('query' in path_lower or 'key' in path_lower or 
                                     'attn' in path_lower or 'causalattn' in path_lower):
        return (ModuleType.QK_NORM, (fan_in, fan_out))
    
    # Other norms in hidden layers
    if 'norm' in path_lower or 'ln' in path_lower or 'rmsnorm' in path_lower:
        return (ModuleType.HIDDEN_NORM, (fan_in, fan_out))
    
    # Biases
    if 'bias' in path_lower:
        return (ModuleType.HIDDEN_BIAS, (fan_in, fan_out))
    
    # Default: hidden weight
    return (ModuleType.HIDDEN_WEIGHT, (fan_in, fan_out))


class MuonCompletedPParameterization(CompletedPParameterization):
    """CompletedP parameterization with Muon-specific scaling rules.

    Implements the scaling rules from Qiu et al. (2025) "Hyperparameter Transfer
    Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales".

    For hidden 2D weights (Muon-optimized via Newton-Schulz):
      - LR:  sqrt(d_out / d_in) * sqrt(m_B / m_D)
             No width transfer scaling (spectral norm handles it).
             No depth scaling (paper: Muon LR has no L dependence).
      - eps: sqrt(d_in / d_out) / L * sqrt(m_D / m_B)
             Per-layer shape correction and 1/L depth scaling.

    For non-Muon params (embeddings, biases, norms, readout — optimized with Adam):
      - Same as standard CompletedPParameterization (Adam mu-P rules).

    Weight decay and initialization are the same as Adam CompletedP.

    EXPERIMENTAL: Batch/data SDE scaling (sqrt(m_B/m_D) for LR, sqrt(m_D/m_B) for eps)
    is borrowed from the Adam theory. The Qiu et al. paper does not cover batch/duration
    transfer for Muon. This is a conservative assumption.

    Args:
        Same as CompletedPParameterization.
    """

    def get_muon_lr_scales_pytree(
        self,
        tensor_types_pytree: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """Get LR scales for Muon-optimized hidden weights.

        Uses sqrt(d_out/d_in) * sqrt(m_B/m_D) for HIDDEN_WEIGHT params.
        For all other ModuleTypes, returns 1.0 (they will use Adam LR scales separately).

        The sqrt(d_out/d_in) factor is the correct shape-dependent scaling from Table 1
        of Qiu et al. This replaces optax.contrib.muon's sqrt(max/min) scaling which
        gives wrong transfer behavior (Section G of the paper).
        """
        def compute_muon_lr_scale(
            module_type: ModuleType,
            fan_in: int,
            fan_out: int,
            tensor_type: TensorType,
            layer_idx: Optional[int],
            device: Any
        ) -> jnp.ndarray:
            if module_type == ModuleType.HIDDEN_WEIGHT:
                # Muon hidden LR: sqrt(d_out/d_in) * sqrt(m_B/m_D)
                # No width transfer scaling (spectral norm handles it)
                # No depth scaling (paper: Muon LR has no L dependence under alpha=1)
                # EXPERIMENTAL: SDE batch/data scaling borrowed from Adam
                scale = math.sqrt(fan_out / fan_in) * self._sde_lr_wd_scale
            else:
                # Non-Muon params: return 1.0 (Adam LR scales are applied separately)
                scale = 1.0

            # Apply depth-wise multiplier
            depth_mult = self._get_depth_multiplier_for_layer(layer_idx)
            scale *= depth_mult

            # Apply per-tensor-type multiplier
            tensor_mult = self._get_per_tensor_multiplier(tensor_type, self.per_tensor_lr_multipliers)
            scale *= tensor_mult

            return jnp.array(scale, dtype=jnp.float32, device=device)

        return self._transform_tensor_types_tree_with_depth(
            tensor_types_pytree,
            compute_muon_lr_scale,
            device
        )

    def get_muon_eps_scales_pytree(
        self,
        tensor_types_pytree: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """Get epsilon scales for Muon-optimized hidden weights.

        Uses sqrt(d_in/d_out) / L * sqrt(m_D/m_B) for HIDDEN_WEIGHT params.
        For all other ModuleTypes, returns 1.0 (they will use Adam eps scales separately).

        The 1/L factor means Muon epsilon shrinks with depth, ensuring the Newton-Schulz
        normalization threshold adapts to the residual scaling (Table 2 of Qiu et al.).
        """
        def compute_muon_eps_scale(
            module_type: ModuleType,
            fan_in: int,
            fan_out: int,
            tensor_type: TensorType,
            layer_idx: Optional[int],
            device: Any
        ) -> jnp.ndarray:
            if module_type == ModuleType.HIDDEN_WEIGHT:
                # Muon hidden eps: sqrt(d_in/d_out) / L * sqrt(m_D/m_B)
                # m_L = current_depth (absolute, not ratio to base)
                # EXPERIMENTAL: SDE scaling borrowed from Adam
                scale = math.sqrt(fan_in / fan_out) * (1.0 / self.m_L) * self._sde_eps_scale
            else:
                # Non-Muon params: return 1.0 (Adam eps scales are applied separately)
                scale = 1.0

            # Apply depth-wise multiplier
            depth_mult = self._get_depth_multiplier_for_layer(layer_idx)
            scale *= depth_mult

            # Apply per-tensor-type multiplier
            tensor_mult = self._get_per_tensor_multiplier(tensor_type, self.per_tensor_eps_multipliers)
            scale *= tensor_mult

            return jnp.array(scale, dtype=jnp.float32, device=device)

        return self._transform_tensor_types_tree_with_depth(
            tensor_types_pytree,
            compute_muon_eps_scale,
            device
        )

    def get_adam_lr_scales_pytree(
        self,
        tensor_types_pytree: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """Get LR scales for Adam-optimized params (embeddings, biases, norms, readout).

        Delegates to the parent CompletedPParameterization's get_lr_scales_pytree,
        which uses the standard Adam mu-P rules.
        """
        return super().get_lr_scales_pytree(tensor_types_pytree, device)

    def get_adam_eps_scales_pytree(
        self,
        tensor_types_pytree: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """Get epsilon scales for Adam-optimized params.

        Delegates to the parent CompletedPParameterization's get_eps_scales_pytree.
        """
        return super().get_eps_scales_pytree(tensor_types_pytree, device)


# Example usage
if __name__ == "__main__":
    # Create a CompletedP parameterization instance
    # base_width = 64 is the reference width (e.g., base embed dim or num_heads * head_dim)
    param = CompletedPParameterization(
        base_width=64,
        base_depth=4,
        base_batch_size=32,
        base_dataset_size=1_000_000,
        current_width=128,  # 2x width (not directly used, fan_in is used per-layer)
        current_depth=8,    # 2x depth
        current_batch_size=64,  # 2x batch
        current_dataset_size=4_000_000,  # 4x data
        alpha=1.0,
    )
    
    print("=" * 60)
    print("CompletedP Parameterization Demo")
    print("=" * 60)
    
    print("\nBase Config:")
    print(f"  base_width: {param.base_width}")
    print(f"  base_depth: {param.base_depth}")
    print(f"  base_batch_size: {param.base_batch_size}")
    print(f"  base_dataset_size: {param.base_dataset_size}")
    
    print("\nScaling Ratios:")
    print(f"  m_L (depth): {param.m_L}")
    print(f"  m_B (batch): {param.m_B}")
    print(f"  m_D (data): {param.m_D}")
    
    print("\nMultipliers (for forward pass):")
    mults = param.get_multipliers()
    for k, v in mults.items():
        print(f"  {k}: {v}")
    
    print("\nBeta Rescaling:")
    betas = param.get_beta_rescaling()
    for k, v in betas.items():
        print(f"  {k}: {v}")
    
    # Example params_info with different fan_in values to show per-layer scaling
    # Note: fan_in varies by layer type!
    embed_dim = 128  # Current model embedding dimension
    mlp_dim = 512    # MLP hidden dimension (4x expansion)
    
    params_info = {
        # Embeddings (fan_in is embed_dim for the output dimension)
        'embed/embedding': (ModuleType.INPUT_EMBED, (50000, embed_dim)),
        
        # Attention layers (fan_in = embed_dim)
        'blocks_0/CausalAttn/query/kernel': (ModuleType.HIDDEN_WEIGHT, (embed_dim, embed_dim)),
        'blocks_0/CausalAttn/key/kernel': (ModuleType.HIDDEN_WEIGHT, (embed_dim, embed_dim)),
        'blocks_0/CausalAttn/value/kernel': (ModuleType.HIDDEN_WEIGHT, (embed_dim, embed_dim)),
        'blocks_0/CausalAttn/attn_out/kernel': (ModuleType.HIDDEN_WEIGHT, (embed_dim, embed_dim)),
        
        # QK norms (special case, no width scaling)
        'blocks_0/CausalAttn/RMSNorm_q/scale': (ModuleType.QK_NORM, (embed_dim, embed_dim)),
        
        # MLP layers - different fan_in values!
        'blocks_0/MLP/up_proj/kernel': (ModuleType.HIDDEN_WEIGHT, (embed_dim, mlp_dim)),    # fan_in = embed_dim
        'blocks_0/MLP/down_proj/kernel': (ModuleType.HIDDEN_WEIGHT, (mlp_dim, embed_dim)),  # fan_in = mlp_dim (larger!)
        
        # Layer norms
        'blocks_0/RMSNorm_0/scale': (ModuleType.HIDDEN_NORM, (embed_dim, embed_dim)),
        
        # Output projection
        'output_proj/kernel': (ModuleType.UNEMBED_WEIGHT, (embed_dim, 50000)),
    }
    
    print(f"\n{'=' * 60}")
    print("Per-Layer Scaling (base_width={}, embed_dim={}, mlp_dim={})".format(
        param.base_width, embed_dim, mlp_dim))
    print("=" * 60)
    
    print("\nLR Rescaling (base_width/fan_in for weights):")
    lr_scales = param.get_lr_rescaling(params_info)
    for k, v in lr_scales.items():
        module_type, (fan_in, _) = params_info[k]
        print(f"  {k}")
        print(f"      fan_in={fan_in}, scale={v:.6f}")
    
    print("\n" + "-" * 60)
    print("\nEpsilon Rescaling:")
    eps_scales = param.get_eps_rescaling(params_info)
    for k, v in eps_scales.items():
        module_type, (fan_in, _) = params_info[k]
        print(f"  {k}")
        print(f"      fan_in={fan_in}, scale={v:.6f}")
    
    print("\n" + "-" * 60)
    print("\nWeight Decay Rescaling (fan_in/base_width for weights):")
    wd_scales = param.get_wd_rescaling(params_info)
    for k, v in wd_scales.items():
        module_type, (fan_in, _) = params_info[k]
        print(f"  {k}")
        print(f"      fan_in={fan_in}, scale={v:.6f}")
    
    # Highlight the difference between layers with different fan_in
    print("\n" + "=" * 60)
    print("Key Insight: MLP down_proj has larger fan_in -> different scaling")
    print("=" * 60)
    print(f"\n  MLP up_proj (fan_in={embed_dim}):")
    print(f"    LR scale: {lr_scales['blocks_0/MLP/up_proj/kernel']:.4f} = {param.base_width}/{embed_dim} * depth * sde")
    print(f"    WD scale: {wd_scales['blocks_0/MLP/up_proj/kernel']:.4f} = {embed_dim}/{param.base_width} * sde")
    print(f"\n  MLP down_proj (fan_in={mlp_dim}):")
    print(f"    LR scale: {lr_scales['blocks_0/MLP/down_proj/kernel']:.4f} = {param.base_width}/{mlp_dim} * depth * sde")
    print(f"    WD scale: {wd_scales['blocks_0/MLP/down_proj/kernel']:.4f} = {mlp_dim}/{param.base_width} * sde")

