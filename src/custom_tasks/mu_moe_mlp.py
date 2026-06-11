#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Linear Layers."""

import functools
import json
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from aqt.jax.v2 import aqt_tensor, calibration, config as aqt_config
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2 import tiled_dot_general
from flax.linen import partitioning
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import shard_map
from jax.tree_util import tree_flatten_with_path, tree_unflatten

# Constants for quantization
DEFAULT = "__default__"  # default config
_W_BITS = "w_bits"  # Number of bits used to represent weights
_A_BITS = "a_bits"  # Number of bits used to represent activations
_W_SCALE = "w_scale"  # Clipping scale for weights
_A_SCALE = "a_scale"  # Clipping scale for activations
_TILE_SIZE = "tile_size"  # Tile size for subchannel
MAX_INT8 = 127.5
MAX_INT4 = 7.5

# Type definitions
Config = Any
Array = jnp.ndarray
PRNGKey = jnp.ndarray
DType = jnp.dtype
Shape = Sequence[int]
Mesh = jax.sharding.Mesh
ScanIn = partitioning.ScanIn
AxisNames = tuple[str, ...]
AxisIdxes = tuple[int, ...]
QTensor = aqt_tensor.QTensor
KVTensor = aqt_tensor.QTensor

# Constants for activation axes
BATCH = "activation_batch"
LENGTH = "activation_length"
EMBED = "activation_embed"
HEAD = "activation_heads"
PREFILL_KV_BATCH = "activation_prefill_kv_batch"
KV_BATCH = "activation_kv_batch"
KV_HEAD = "activation_kv_heads"
KV_HEAD_DIM = "activation_kv_head_dim"
D_KV = "activation_kv"

# Constants for cache axes
CACHE_BATCH_PREFILL = "cache_batch_prefill"
CACHE_BATCH = "cache_batch"
CACHE_SEQUENCE = "cache_sequence"
CACHE_HEADS = "cache_heads"
CACHE_KV = "cache_kv"
CACHE_SCALE_BATCH = "cache_scale_batch"
CACHE_SCALE_SEQUENCE = "cache_scale_sequence"
CACHE_SCALE_HEADS = "cache_scale_heads"
CACHE_SCALE_KV = "cache_scale_kv"

# Model mode constants
MODEL_MODE_AUTOREGRESSIVE = "autoregressive"
MODEL_MODE_PREFILL = "prefill"
MODEL_MODE_TRAIN = "train"

# MoE constants
DISPATCH = "dispatch"
COMBINE = "combine"

# Other constants
DECODING_ACTIVE_SEQUENCE_INDICATOR = 1
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)

class max_logging:
    @classmethod
    def log(cls, message):
        print(message)

def unbox_logicallypartioned(boxed_pytree):
  """Unboxes the flax.LogicallyPartitioned pieces

  Args:
    boxed_pytree: a pytree that includes LogicallyPartitioned
      leaves.
  Returns:
    a pytree where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
      boxed_pytree,
      is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
  )



def in_serve_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.SERVE)


Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_embed_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)

default_bias_init = jax.nn.initializers.constant(0.0)


def nd_dense_init(scale, mode, distribution):
  """Initializer with in_axis, out_axis set at call time."""

  def init_fn(key, shape, dtype, in_axis, out_axis):
    fn = jax.nn.initializers.variance_scaling(scale, mode, distribution, in_axis, out_axis)
    return fn(key, shape, dtype)

  return init_fn



@dataclass
class AqtQuantization:
  """Configures AQT quantization github.com/google/aqt."""

  quant_dg: aqt_config.DotGeneral
  quant_mode: aqt_flax.QuantMode = aqt_flax.QuantMode.TRAIN
  replicate_scale: bool = False

  def _get_mixed_precision_cfg(self):
    quant_dg = None
    is_tiled = False
    tiling_fn = None
    module_path = "/".join(nn.module._context.module_stack[-1].path)
    for layer_name_re, layer_quant_dg in self.quant_dg.items():
      if re.fullmatch(layer_name_re, module_path):
        quant_dg, tile_size = layer_quant_dg
    if quant_dg is None:
      quant_dg, tile_size = self.quant_dg[DEFAULT]
    if tile_size != -1:
      is_tiled = True
      tiling_fn = functools.partial(_tiling_fn, tile_size=tile_size)
    return quant_dg, is_tiled, tiling_fn

  def _get_rhs_axis_metadata_wrapper(
      self, mesh_axes: Tuple[str, ...] = (), is_tiled: bool = False, replicate_scale: bool = False
  ):
    if self.quant_mode == aqt_flax.QuantMode.CONVERT:
      return None
    return functools.partial(
        _rhs_axis_metadata_wrapper, mesh_axes=mesh_axes, is_tiled=is_tiled, replicate_scale=replicate_scale
    )

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    if isinstance(self.quant_dg, dict):
      quant_dg, is_tiled, tiling_fn = self._get_mixed_precision_cfg()
    else:
      quant_dg, is_tiled, tiling_fn = self.quant_dg, False, None
    rhs_axis_metadata_wrapper = self._get_rhs_axis_metadata_wrapper(
        mesh_axes, is_tiled, replicate_scale=self.replicate_scale
    )
    # module_path = "/".join(nn.module._context.module_stack[-1].path)
    # print(f"quant_dg: {quant_dg}, is_tiled: {is_tiled}, module_path: {module_path}")
    aqt_dg_cls = functools.partial(
        aqt_flax.AqtDotGeneral,
        quant_dg,
        rhs_quant_mode=self.quant_mode,
        lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
        rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
        rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
        use_legacy_freezer=False,
        tiling_fn=tiling_fn,
    )
    return aqt_dg_cls

  def einsum(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns einsum configured with aqt params."""
    if isinstance(self.quant_dg, dict):
      quant_dg, is_tiled, tiling_fn = self._get_mixed_precision_cfg()
    else:
      quant_dg, is_tiled, tiling_fn = self.quant_dg, False, None

    rhs_axis_metadata_wrapper = self._get_rhs_axis_metadata_wrapper(
        mesh_axes, is_tiled, replicate_scale=self.replicate_scale
    )
    aqt_einsum = functools.partial(
        aqt_flax.AqtEinsum(
            cfg=quant_dg,
            rhs_quant_mode=self.quant_mode,
            lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
            rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
            rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
            use_legacy_freezer=False,
            tiling_fn=tiling_fn,
        )
    )
    return aqt_einsum

Quant = AqtQuantization


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
        nn.with_logical_partitioning(self.scale_init, self.kernel_axes),
        (features,),
        self.weight_dtype,
    )

    scale = jnp.asarray(scale, self.dtype)
    return y * scale





def _convert_to_activation_function(fn_or_string: Union[str, Callable[..., Any]]) -> Callable[..., Any]:
  """Convert a string to an activation function."""
  if fn_or_string == "linear":
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(
        f"""Don't know how to convert {fn_or_string}
                         to an activation function"""
    )


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


class DenseGeneral(nn.Module):
  """A linear transformation with flexible axes.

  Attributes:
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    use_bias: whether to add bias in linear transformation
    quant: quantization config, defaults to None implying no quantization.
  """

  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  kernel_axes: Tuple[Optional[str], ...] = ()
  quant: Optional[Quant] = None
  use_bias: bool = False
  matmul_precision: str = "default"

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """

    def compute_dot_general(inputs, kernel, axis, contract_ind):
      """Computes a dot_general operation that may be quantized."""
      dot_general = lax.dot_general
      matmul_precision = lax.Precision(self.matmul_precision)
      if self.quant:
        dot_general_cls = self.quant.dot_general_cls(mesh_axes=self.kernel_axes)
        dot_general = dot_general_cls()
        return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)
      return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision)

    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
    if in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      kernel = jnp.zeros(kernel_shape)
    else:
      kernel = self.param(
          "kernel",
          nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
          kernel_shape,
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(axis)))
    output = compute_dot_general(inputs, kernel, axis, contract_ind)

    if self.use_bias:
      bias_axes, bias_shape = (
          self.kernel_axes[-len(features) :],
          kernel_shape[-len(features) :],
      )
      bias = self.param(
          "bias",
          nn.with_logical_partitioning(bias_init, bias_axes),
          bias_shape,
          self.weight_dtype,
      )
      bias = jnp.asarray(bias, self.dtype)
      output += bias
    return output


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: computation data type for the dense layer.
    weight_dtype: weight data type for the dense layer.
    use_bias: whether to add bias in all feedforward layers.
    use_pre_norm: whether to add pre layer norm in mlp layers.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable[..., Any]]] = ("relu",)
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  use_bias: bool = False
  use_pre_norm: bool = False
  quant: Optional[Quant] = None

  def get_norm_layer(self):
    if self.config.decoder_block in ("default", "llama2", "mistral", "gemma"):
      return RMSNorm
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return functools.partial(gpt3.Gpt3LayerNorm, reductions_in_fp32=False, use_bias=self.use_bias)
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    if self.use_pre_norm:
      inputs = self.get_norm_layer()(
          name="mlp_layer_norm",
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("norm",),
          epsilon=cfg.normalization_layer_epsilon,
      )(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    if cfg.fused_mlp:
      x = DenseGeneral(
          (len(self.activations), self.intermediate_dim),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "num_activations", "mlp"),
          name="wi",
          quant=self.quant,
          use_bias=self.use_bias,
          matmul_precision=self.config.matmul_precision,
      )(inputs)
      x = checkpoint_name(x, "mlpwi")
      for idx, act_fn in enumerate(self.activations):
        y = _convert_to_activation_function(act_fn)(x[:, :, idx, ...])
        activations.append(y)
    else:
      for idx, act_fn in enumerate(self.activations):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        x = DenseGeneral(
            self.intermediate_dim,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("embed", "mlp"),
            name=dense_name,
            quant=self.quant,
            use_bias=self.use_bias,
            matmul_precision=self.config.matmul_precision,
        )(inputs)
        x = checkpoint_name(x, "mlp" + dense_name)
        if cfg.activations_in_float32:
          x = x.astype(jnp.float32)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations).astype(self.dtype)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
        x, deterministic=deterministic
    )  # Broadcast along length.
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_mlp"))
    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("mlp", "embed"),
        name="wo",
        quant=self.quant,
        use_bias=self.use_bias,
        matmul_precision=self.config.matmul_precision,
    )(x)

    output = checkpoint_name(output, "mlpwo")
    return output


class MoeBlock(nn.Module):
  """Mixture of Experts (MoE) block.

  Attributes:
    num_experts: Number of experts.
    num_experts_per_tok: Number of experts for each token.
    mesh: Mesh, device mesh.
    kernel_init: Kernel function, passed to the dense layers.
    kernel_axes: Tuple with axes to apply kernel function.
    weight_dtype: Type for the weights.
    dtype: Type for the dense layer.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  num_experts: int
  num_experts_per_tok: int
  mesh: Mesh
  kernel_init: NdInitializer
  kernel_axes: Tuple[Optional[str], ...]
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  quant: Optional[Quant] = None

  # The first axes is expert
  wi_kernel_axes = ("exp", "embed_no_exp", "mlp")
  wo_kernel_axes = ("exp", "mlp", "embed_no_exp")

  def generate_kernels(self, num_experts, emb_dim, mlp_dim):

    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)
    # Honor the muP/CompletedP-scaled initializer supplied to MoeBlock
    # (matches the dense MlpSwiGLU path which uses cfg.hidden_kernel_init_std).
    # Previously hardcoded scale=1.0 silently shadowed self.kernel_init for
    # the expert weights wi_0/wi_1/wo, breaking muP for MoE experts.
    kernel_init = self.kernel_init

    if in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      w0_kernel = jnp.zeros((num_experts, emb_dim, mlp_dim))
    else:
      w0_kernel = self.param(
          "wi_0",
          nn.with_logical_partitioning(kernel_init, self.wi_kernel_axes),
          (num_experts, emb_dim, mlp_dim),
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )

    w0_kernel = jnp.asarray(w0_kernel, self.dtype)

    if in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      w1_kernel = jnp.zeros((num_experts, emb_dim, mlp_dim))
    else:
      w1_kernel = self.param(
          "wi_1",
          nn.with_logical_partitioning(kernel_init, self.wi_kernel_axes),
          (num_experts, emb_dim, mlp_dim),
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
    w1_kernel = jnp.asarray(w1_kernel, self.dtype)

    if in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      wo_kernel = jnp.zeros((num_experts, mlp_dim, emb_dim))
    else:
      wo_kernel = self.param(
          "wo",
          nn.with_logical_partitioning(kernel_init, self.wo_kernel_axes),
          (num_experts, mlp_dim, emb_dim),
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
    wo_kernel = jnp.asarray(wo_kernel, self.dtype)
    return w0_kernel, w1_kernel, wo_kernel

  def permute(self, inputs, gate_logits):
    """Permute tokens to group by expert to fit gmm call."""

    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_shape = inputs.shape
    inputs_2d = jnp.reshape(inputs, (inputs_shape[0] * inputs_shape[1], inputs_shape[2]))
    weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1).astype(self.dtype)
    flatten_selected_experts = jnp.ravel(selected_experts)
    sorted_selected_experts = jnp.argsort(flatten_selected_experts)
    sorted_indices = sorted_selected_experts // self.num_experts_per_tok
    # sort inputs for number of selected experts
    sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
    group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
    return sorted_inputs, sorted_selected_experts, weights, group_size

  def unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length):
    """Unpermute tokens to original order and combine weights."""

    unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
    reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(
        unsort_intermediate,
        (reshaped_weights.shape[0], self.num_experts_per_tok, -1),
    )
    with jax.named_scope("weight_sum"):
      matmul_precision = lax.Precision(self.config.matmul_precision)
      output = jnp.einsum(
          "BKE,BK -> BE",
          reshaped_intermediate.astype(jnp.float32),
          reshaped_weights.astype(jnp.float32),
          precision=matmul_precision,
      )
    return output.reshape(batch_size, sequence_length, -1).astype(self.dtype)

  # def megablox(self, inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel):
  #   tile_size = (512, 1024, 1024)

  #   def gmm(inputs, kernel, group_sizes):
  #     hs_shape = inputs.shape
  #     # pad length is the 1st dimension of tiling size in gmm call
  #     pad_length = 512
  #     if hs_shape[0] % pad_length:
  #       pad_length = pad_length - hs_shape[0] % pad_length
  #       inputs = jax.lax.pad(inputs.astype(jnp.float32), 0.0, [(0, pad_length, 0), (0, 0, 0)])

  #     inputs = inputs.astype(self.dtype)
  #     kernel = kernel.astype(self.dtype)

  #     lhs_quantize_dtype, rhs_quantize_dtype = None, None
  #     if self.quant is not None:
  #       quant_dg = self.quant.quant_dg
  #       lhs_quantize_dtype = quant_dg.fwd.dg_quantizer.lhs.numerics.get_dtype()
  #       rhs_quantize_dtype = quant_dg.fwd.dg_quantizer.rhs.numerics.get_dtype()

  #     output = mblx.gmm(
  #         lhs=inputs,
  #         rhs=kernel,
  #         group_sizes=group_sizes,
  #         preferred_element_type=jnp.bfloat16,
  #         tiling=tile_size,
  #         lhs_quantize_dtype=lhs_quantize_dtype,
  #         rhs_quantize_dtype=rhs_quantize_dtype,
  #     )
  #     if hs_shape[0] % pad_length:
  #       output = output[: hs_shape[0]]
  #     return output

  #   # Currently, we only support data and tensor parallelism with Megablox.
  #   # We all gather the input activations over tensor parallelism to follow strategy
  #   # in https://parsa.epfl.ch/course-info/cs723/papers/Megatron.pdf.
  #   input_partition_spec = nn.logical_to_mesh_axes(("activation_batch", None, None))
  #   gate_logits_pspec = nn.logical_to_mesh_axes(("activation_batch", None, None))
  #   w0_pspec = nn.logical_to_mesh_axes((None, None, "mlp"))
  #   w1_pspec = nn.logical_to_mesh_axes((None, None, "mlp"))
  #   wo_pspec = nn.logical_to_mesh_axes((None, "mlp", None))

  #   if isinstance(w0_kernel, QTensor):
  #     w0_pspec = aqt_tensor.partition_spec(w0_pspec, (1,), w0_kernel.dtype, use_bias=False)
  #   if isinstance(w1_kernel, QTensor):
  #     w1_pspec = aqt_tensor.partition_spec(w1_pspec, (1,), w1_kernel.dtype, use_bias=False)
  #   if isinstance(wo_kernel, QTensor):
  #     wo_pspec = aqt_tensor.partition_spec(wo_pspec, (1,), wo_kernel.dtype, use_bias=False)

  #   @functools.partial(
  #       shard_map.shard_map,
  #       mesh=self.mesh,
  #       in_specs=(input_partition_spec, gate_logits_pspec, w0_pspec, w1_pspec, wo_pspec),
  #       out_specs=(nn.logical_to_mesh_axes(("activation_batch", None, "activation_embed"))),
  #       check_rep=False,
  #   )
  #   def wrapper(x, logits, w0, w1, wo):
  #     batch_size, sequence_length, _ = x.shape
  #     x, sorted_selected_experts, weights, group_sizes = self.permute(x, logits)
  #     layer_w0 = gmm(x, w0, group_sizes)
  #     layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")
  #     layer_w1 = gmm(x, w1, group_sizes)
  #     layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")
  #     layer_act = _convert_to_activation_function(self.config.mlp_activations[0])(layer_w0)
  #     intermediate_layer = jnp.multiply(layer_act, layer_w1)
  #     intermediate_output = gmm(intermediate_layer, wo, group_sizes)
  #     intermediate_output = checkpoint_name(intermediate_output, "mlpwo")
  #     tensor_parallelism = self.config.ici_tensor_parallelism * self.config.dcn_tensor_parallelism
  #     if tensor_parallelism > 1:
  #       intermediate_output = jax.lax.psum_scatter(intermediate_output, "tensor", scatter_dimension=1, tiled=True)
  #     output = self.unpermute(
  #         intermediate_output, sorted_selected_experts, weights, batch_size=batch_size, sequence_length=sequence_length
  #     )
  #     return output, None

  #   return wrapper(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)

  def reshape_and_update_weights(self, weights, indices):
    # input of weights & indices: (batch_size, seq_len, num_experts_per_tok)
    # output of updated weights: (batch_size, seq_len, num_experts)
    update_weights = jnp.zeros((weights.shape[0], weights.shape[1], self.num_experts), dtype=self.dtype)
    index_update = (
        jnp.arange(weights.shape[0])[:, None, None],
        jnp.arange(weights.shape[1])[:, None],
        indices,
    )
    update_weights = update_weights.at[index_update].set(weights)
    return update_weights

  def generate_masks(self, top_k_indices, softmax_probs):
    # calculate expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape
    tokens_per_batch = seq_len * self.num_experts_per_tok
    # this is to avoid expert_capacity_per_batch = 0
    expert_capacity_per_batch = int(
        max(
            math.ceil(tokens_per_batch / self.num_experts) * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log(f"Applying potential token dropping with a batch expert_capacity of {expert_capacity_per_batch}")

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok)
    #
    # A small example:
    # give num_experts=4 & num_experts_per_tok=2, and two tokens are routed to expert [0, 1] & [1, 3],
    # then expert_mask becomes [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 1, 0, 0],[0, 0, 0, 1]]]],
    # after cumsum, expert_token_count becomes [[[[1, 0, 0, 0],[1, 1, 0, 0]], [[1, 2, 0, 0],[1, 2, 0, 1]]]],
    # if we set expert_capacity=1,
    # trunc_expert_mask becomes [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 0, 0],[0, 0, 0, 1]]]],
    # so the 2nd token for expert #1 ([0, 1] & [1, 3]) is dropped, output of updated_expert_mask is [[[1, 1],[0, 1]]].
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(expert_mask, (batch_size, seq_len * self.num_experts_per_tok, self.num_experts))
    expert_mask_fused = nn.with_logical_constraint(expert_mask_fused, ("activation_batch", None, None))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=1)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, seq_len, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = nn.with_logical_constraint(
        expert_token_count, ("activation_batch", "activation_length", None, None)
    )
    trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
    combined_expert_mask = jnp.sum(trunc_expert_mask, axis=2)

    # Mask out non-top-k experts. Do NOT renormalise softmax_probs over the
    # top-k entries: keeping the raw router probability lets gradient flow
    # back through the gate via the combine_mask einsum, which is essential
    # for the router to learn task-relevant routing (especially at k=1, where
    # a renormalised one-hot mask makes d(combine_mask)/d(softmax_probs) = 0
    # everywhere except via the load balance loss). This matches the
    # Switch-Transformer / MaxText convention.
    softmax_probs *= combined_expert_mask

    # calculate token position in expert capacity dimension
    expert_token_position_fused = expert_mask_fused * expert_token_count_fused
    expert_token_position = jnp.reshape(
        expert_token_position_fused,
        (batch_size, seq_len, self.num_experts_per_tok, self.num_experts),
    )
    combined_expert_token_position = jnp.sum(expert_token_position, axis=2) * combined_expert_mask
    expert_token_position_in_capacity = jax.nn.one_hot(
        combined_expert_token_position,
        num_classes=expert_capacity_per_batch + 1,
        dtype=jnp.int32,
    )

    # shape of combine_mask is (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.astype(bool)
    return dispatch_mask, combine_mask

  # See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
  def load_balance_loss(self, top_k_indices, logits):
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    summed_expert_mask = jnp.sum(expert_mask, axis=2)
    # Get fraction of tokens dispatched to each expert
    density = jnp.mean(summed_expert_mask, axis=1)
    # get fraction of probability allocated to each expert
    density_prob = jnp.mean(logits, axis=1)
    loss = jnp.mean(density * density_prob) * (self.num_experts**2) * self.config.load_balance_loss_weight
    return loss

  def get_einsum(self, rhs_mesh_axes: Tuple[Optional[str], ...] = (), einsum_name=None):

    # the check is to prevent aqteinsum as einsum op for dispatch and combine einsums in ase when capacity_factor > 0
    # this is necessary to load pre-quantized weights in case of inference
    if self.config.model_call_mode == "inference" and (einsum_name == DISPATCH or einsum_name == COMBINE):
      return jnp.einsum

    if self.quant:

      def aqt_einsum(*args, **kwargs):
        # simply skip kwargs, since aqt einsum doesn't support any kwargs like precision
        return self.quant.einsum(rhs_mesh_axes)(*args)

      einsum_op = aqt_einsum
    else:
      einsum_op = jnp.einsum
    return einsum_op

  def is_expert_parallelism_enabled(self):
    return self.config.ici_expert_parallelism > 1 or self.config.dcn_expert_parallelism > 1

  def maybe_all_gather_kernel_weight_in_expert_parallelism(self, kernel, kernel_axes):
    if self.is_expert_parallelism_enabled():
      # This will trigger all-gather using weight_dtype
      # relax it unless really necessary in expert parallelism only
      # Otherwise compiler will handle communication automatically
      # esp. with int8 quantization, kernel will be all-gathered in int8 instead of weight_dtype
      kernel = nn.with_logical_constraint(kernel, kernel_axes)
    return kernel

  def ragged_matmul(self, inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel):
    """Optimized MoE forward using jax.lax.ragged_dot (grouped GEMM).

    For the dropless case (capacity_factor >= num_experts) the dispatch/combine
    path in dense_matmul wastes ~num_experts/num_experts_per_tok of its compute
    on the zero-padded capacity dimension. This implementation:

      1. Skips generate_masks (no token dropping = no cumsum/one_hot dance).
      2. Sorts tokens by their assigned expert (argsort + take).
      3. Runs three grouped matmuls via jax.lax.ragged_dot — wi_0, wi_1, wo —
         each of which processes only the actual (token, expert) pairs.
      4. Applies the gate weight, unsorts back to original token order, and
         sums across the K top-k slots (relevant when K > 1).

    Numerically equivalent to dense_matmul under dropless capacity, modulo
    floating-point reordering inside the matmul kernels.

    If `cfg.matmul_dtype` is set (e.g. jnp.bfloat16) the inputs and kernels are
    cast to that dtype just before each ragged_dot, with the accumulator and
    activations remaining in `self.dtype`. On A100/H100 this typically gives
    a ~2x speedup with negligible quality loss for forward inference, since
    bf16 matmul throughput is ~2x fp32 TF32.
    """
    cfg = self.config
    B, S, M = inputs.shape
    K = self.num_experts_per_tok
    E = self.num_experts
    matmul_precision = lax.Precision(cfg.matmul_precision)
    matmul_dtype = getattr(cfg, "matmul_dtype", None)

    # Router probabilities and top-k expert selection.
    softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
    top_k_weights, top_k_indices = jax.lax.top_k(softmax_probs, K)
    # top_k_weights: (B, S, K), top_k_indices: (B, S, K)

    # Load balance loss is computed from the full softmax, before permutation.
    if cfg.model_call_mode != "inference":
      loss = self.load_balance_loss(top_k_indices, softmax_probs)
    else:
      loss = None

    # Flatten (B, S, K) -> (B*S*K,) so we can sort tokens by expert.
    flat_indices = top_k_indices.reshape(-1)        # (BS*K,) expert id per slot
    flat_weights = top_k_weights.reshape(-1).astype(self.dtype)  # (BS*K,) gate prob per slot

    # Argsort by expert id so all tokens for expert 0 come first, then expert 1, etc.
    sorted_perm = jnp.argsort(flat_indices)         # (BS*K,)
    sorted_expert_ids = jnp.take(flat_indices, sorted_perm)  # (BS*K,) monotonic
    # Map sorted slot back to its original token row in the flat (B*S, M) input.
    sorted_token_rows = sorted_perm // K            # (BS*K,)
    inputs_2d = inputs.reshape(B * S, M)            # (BS, M)
    sorted_inputs = jnp.take(inputs_2d, sorted_token_rows, axis=0)  # (BS*K, M)
    sorted_weights = jnp.take(flat_weights, sorted_perm)            # (BS*K,)

    # Group sizes per expert for ragged_dot. We use one_hot + sum instead of
    # jnp.bincount because bincount has dynamic-output-shape detection logic
    # that costs an extra dispatch on GPU; one_hot is fully static and lets
    # XLA fuse with the surrounding ops.
    group_sizes = jnp.sum(
        jax.nn.one_hot(sorted_expert_ids, E, dtype=jnp.int32), axis=0
    )                                                                # (E,)

    # Optionally cast inputs/kernels to bf16 for the matmul. The cast is free
    # if matmul_dtype is None (no-op).
    sorted_inputs_mm = sorted_inputs.astype(matmul_dtype) if matmul_dtype is not None else sorted_inputs
    w0_mm = w0_kernel.astype(matmul_dtype) if matmul_dtype is not None else w0_kernel
    w1_mm = w1_kernel.astype(matmul_dtype) if matmul_dtype is not None else w1_kernel
    wo_mm = wo_kernel.astype(matmul_dtype) if matmul_dtype is not None else wo_kernel

    # Grouped GEMMs. ragged_dot signature:
    #   lhs (m, k) @ rhs (g, k, n) with group_sizes (g,) -> (m, n)
    # Our wi_* kernels are already (E, M, H) and wo is (E, H, M) — exactly
    # the (g, k, n) layout ragged_dot expects.
    with jax.named_scope("wi_0_ragged"):
      layer_w0 = jax.lax.ragged_dot(
          lhs=sorted_inputs_mm,
          rhs=w0_mm,
          group_sizes=group_sizes,
          precision=matmul_precision,
      )  # (BS*K, H)
      if cfg.activations_in_float32:
        layer_w0 = layer_w0.astype(jnp.float32)
      layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")

    with jax.named_scope("wi_1_ragged"):
      layer_w1 = jax.lax.ragged_dot(
          lhs=sorted_inputs_mm,
          rhs=w1_mm,
          group_sizes=group_sizes,
          precision=matmul_precision,
      )  # (BS*K, H)
      if cfg.activations_in_float32:
        layer_w1 = layer_w1.astype(jnp.float32)
      layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")

    layer_w0_act = _convert_to_activation_function(cfg.mlp_activations[0])(layer_w0)
    layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
    # Fold the per-token gate weight INTO the SwiGLU output before the wo
    # matmul. Mathematically:
    #   wo @ (act * sw)  ==  (wo @ act) * sw
    # because sw is a scalar per row of the lhs. This saves one elementwise
    # multiply kernel after the wo matmul, and lets XLA fuse the scale into
    # the SwiGLU activation.
    layer_multiply = layer_multiply * sorted_weights[:, None]

    with jax.named_scope("wo_ragged"):
      layer_multiply_mm = layer_multiply.astype(matmul_dtype) if matmul_dtype is not None else layer_multiply
      sorted_output = jax.lax.ragged_dot(
          lhs=layer_multiply_mm,
          rhs=wo_mm,
          group_sizes=group_sizes,
          precision=matmul_precision,
      )  # (BS*K, M)
      if cfg.activations_in_float32:
        sorted_output = sorted_output.astype(jnp.float32)
      sorted_output = checkpoint_name(sorted_output, "mlpwo")

    # Unpermute back to original token order. Build inv_perm with an O(N)
    # scatter instead of an O(N log N) argsort:
    #   inv_perm[sorted_perm[i]] = i
    n_slots = sorted_perm.shape[0]
    inv_perm = jnp.zeros(n_slots, dtype=jnp.int32).at[sorted_perm].set(
        jnp.arange(n_slots, dtype=jnp.int32)
    )
    unsorted_output = jnp.take(sorted_output, inv_perm, axis=0)     # (BS*K, M)

    if K == 1:
      # Fast path: nothing to sum across, skip the reshape+sum.
      output = unsorted_output.reshape(B, S, M).astype(self.dtype)
    else:
      output = unsorted_output.reshape(B * S, K, M).sum(axis=1)
      output = output.reshape(B, S, M).astype(self.dtype)

    return output, loss

  def pallas_matmul(self, inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel):
    """Optimized MoE forward using a custom Pallas-Triton grouped GEMM kernel.

    Same algorithm as `ragged_matmul`, but the three `jax.lax.ragged_dot`
    calls are replaced by `grouped_matmul` from
    `custom_tasks.moe_pallas_kernel`. The Pallas kernel fuses all
    `(m_tile, n_tile, group)` tile launches into a single Triton-compiled
    program, which avoids the per-group dispatch loop that
    `jax.lax.ragged_dot` does internally on GPU.

    Equivalence: this method passes the same `bench/test_moe_equivalence.py`
    allclose test as `ragged_matmul` (atol=1e-6, rtol=1e-5 in fp32 mode).
    bf16 mode is gated on `cfg.matmul_dtype` exactly like `ragged_matmul`.

    Requires `cfg.use_pallas_kernel = True`. The kernel is opt-in; the
    default `__call__` dispatch falls back to `ragged_matmul` (which itself
    falls back to `dense_matmul` when token dropping is enabled).
    """
    # Local import to avoid pulling Pallas symbols into modules that don't use them.
    from custom_tasks.moe_pallas_kernel import grouped_matmul

    cfg = self.config
    B, S, M = inputs.shape
    K = self.num_experts_per_tok
    E = self.num_experts
    matmul_dtype = getattr(cfg, "matmul_dtype", None)

    softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
    top_k_weights, top_k_indices = jax.lax.top_k(softmax_probs, K)

    if cfg.model_call_mode != "inference":
      loss = self.load_balance_loss(top_k_indices, softmax_probs)
    else:
      loss = None

    flat_indices = top_k_indices.reshape(-1)
    flat_weights = top_k_weights.reshape(-1).astype(self.dtype)
    sorted_perm = jnp.argsort(flat_indices)
    sorted_expert_ids = jnp.take(flat_indices, sorted_perm)
    sorted_token_rows = sorted_perm // K
    inputs_2d = inputs.reshape(B * S, M)
    sorted_inputs = jnp.take(inputs_2d, sorted_token_rows, axis=0)
    sorted_weights = jnp.take(flat_weights, sorted_perm)
    group_sizes = jnp.sum(
        jax.nn.one_hot(sorted_expert_ids, E, dtype=jnp.int32), axis=0
    )

    sorted_inputs_mm = sorted_inputs.astype(matmul_dtype) if matmul_dtype is not None else sorted_inputs
    w0_mm = w0_kernel.astype(matmul_dtype) if matmul_dtype is not None else w0_kernel
    w1_mm = w1_kernel.astype(matmul_dtype) if matmul_dtype is not None else w1_kernel
    wo_mm = wo_kernel.astype(matmul_dtype) if matmul_dtype is not None else wo_kernel

    with jax.named_scope("wi_0_pallas"):
      layer_w0 = grouped_matmul(sorted_inputs_mm, w0_mm, group_sizes)
      if cfg.activations_in_float32:
        layer_w0 = layer_w0.astype(jnp.float32)
      layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")

    with jax.named_scope("wi_1_pallas"):
      layer_w1 = grouped_matmul(sorted_inputs_mm, w1_mm, group_sizes)
      if cfg.activations_in_float32:
        layer_w1 = layer_w1.astype(jnp.float32)
      layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")

    layer_w0_act = _convert_to_activation_function(cfg.mlp_activations[0])(layer_w0)
    layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
    layer_multiply = layer_multiply * sorted_weights[:, None]

    with jax.named_scope("wo_pallas"):
      layer_multiply_mm = layer_multiply.astype(matmul_dtype) if matmul_dtype is not None else layer_multiply
      sorted_output = grouped_matmul(layer_multiply_mm, wo_mm, group_sizes)
      if cfg.activations_in_float32:
        sorted_output = sorted_output.astype(jnp.float32)
      sorted_output = checkpoint_name(sorted_output, "mlpwo")

    n_slots = sorted_perm.shape[0]
    inv_perm = jnp.zeros(n_slots, dtype=jnp.int32).at[sorted_perm].set(
        jnp.arange(n_slots, dtype=jnp.int32)
    )
    unsorted_output = jnp.take(sorted_output, inv_perm, axis=0)

    if K == 1:
      output = unsorted_output.reshape(B, S, M).astype(self.dtype)
    else:
      output = unsorted_output.reshape(B * S, K, M).sum(axis=1)
      output = output.reshape(B, S, M).astype(self.dtype)

    return output, loss

  def dense_matmul(self, inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel):
    # gate_logits: batch, length, expert
    gate_logits = nn.with_logical_constraint(gate_logits, ("activation_batch", "activation_length", None))
    softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
    # shape of top_k_weights & top_k_indices: (batch, sequence, num_experts_per_tok)
    top_k_weights, top_k_indices = jax.lax.top_k(softmax_probs, self.num_experts_per_tok)
    matmul_precision = lax.Precision(self.config.matmul_precision)

    if self.config.capacity_factor > 0:
      # token dropping if needed
      dispatch_mask, combine_mask = self.generate_masks(top_k_indices, softmax_probs)
      mask_axes = ("activation_batch", "activation_length", None, None)
      dispatch_mask = nn.with_logical_constraint(dispatch_mask, mask_axes)
      combine_mask = nn.with_logical_constraint(combine_mask, mask_axes)
      if self.config.model_call_mode != "inference":
        loss = self.load_balance_loss(top_k_indices, softmax_probs)
      else:
        loss = None
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
      with jax.named_scope("dispatch"):
        dispatch = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=DISPATCH)(
            "BSM,BSEC -> EBCM", inputs, dispatch_mask, precision=matmul_precision
        )
        dispatch = nn.with_logical_constraint(
            dispatch,
            ("activation_exp", "activation_batch_no_exp", None, "activation_embed"),
        )
      with jax.named_scope("wi_0"):
        w0_kernel_axes = ("exp", None, "mlp")
        w0_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w0_kernel, w0_kernel_axes)
        layer_w0 = self.get_einsum(rhs_mesh_axes=w0_kernel_axes)(
            "EBCM,EMH -> EBCH", dispatch, w0_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = nn.with_logical_constraint(
            layer_w0,
            ("activation_exp", "activation_batch_no_exp", None, "activation_mlp"),
        )
        layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")
      with jax.named_scope("wi_1"):
        w1_kernel_axes = ("exp", None, "mlp")
        w1_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w1_kernel, w1_kernel_axes)
        layer_w1 = self.get_einsum(rhs_mesh_axes=w1_kernel_axes)(
            "EBCM,EMH -> EBCH", dispatch, w1_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = nn.with_logical_constraint(
            layer_w1,
            ("activation_exp", "activation_batch_no_exp", None, "activation_mlp"),
        )
        layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")
      layer_w0_act = _convert_to_activation_function(self.config.mlp_activations[0])(layer_w0)
      layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
      with jax.named_scope("wo"):
        wo_kernel_axes = ("exp", "mlp", None)
        wo_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(wo_kernel, wo_kernel_axes)
        intermediate_layer = self.get_einsum(rhs_mesh_axes=wo_kernel_axes)(
            "EBCH,EHM -> EBCM", layer_multiply, wo_kernel, precision=matmul_precision
        )
        intermediate_layer = nn.with_logical_constraint(
            intermediate_layer,
            ("activation_exp", "activation_batch_no_exp", None, "activation_embed"),
        )
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        intermediate_layer = checkpoint_name(intermediate_layer, "mlpwo")
      with jax.named_scope("combine"):
        # Matmul & element wise operation
        output = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=COMBINE)(
            "EBCM,BSEC -> BSM",
            intermediate_layer,
            combine_mask,
            precision=matmul_precision,
        ).astype(self.dtype)
      return output, loss
    else:
      top_k_weights /= top_k_weights.sum(-1, keepdims=True)
      weights = self.reshape_and_update_weights(top_k_weights, top_k_indices)
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
      with jax.named_scope("wi_0"):
        layer_w0 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w0_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")
      with jax.named_scope("wi_1"):
        layer_w1 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w1_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")
      layer_w0_act = _convert_to_activation_function(self.config.mlp_activations[0])(layer_w0)
      layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
      with jax.named_scope("wo"):
        intermediate_layer = self.get_einsum(rhs_mesh_axes=self.wo_kernel_axes)(
            "BSEH,EHM -> BSEM", layer_multiply, wo_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        intermediate_layer = checkpoint_name(intermediate_layer, "mlpwo")
      with jax.named_scope("w_sum"):
        output = jnp.einsum(
            "BSEM,BSE -> BSM",
            intermediate_layer,
            weights,
        ).astype(self.dtype)
      return output, None

  def retrieve_quantized_weight(
      self, inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel
  ) -> tuple[QTensor, QTensor, QTensor]:
    # This is called only during tracing. This is to invoke creation of quantized tensor inside AqtEinsum.
    # After jit, this will become no-op and will not affect performance.
    _ = self.dense_matmul(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)

    w0_kernel = self.variables["aqt"]["AqtEinsum_0"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    w1_kernel = self.variables["aqt"]["AqtEinsum_1"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    wo_kernel = self.variables["aqt"]["AqtEinsum_2"]["AqtDotGeneral_0"]["qrhs"]["frozen"]

    w0_kernel = max_utils.unbox_logicallypartioned(w0_kernel)
    w1_kernel = max_utils.unbox_logicallypartioned(w1_kernel)
    wo_kernel = max_utils.unbox_logicallypartioned(wo_kernel)
    return w0_kernel, w1_kernel, wo_kernel

  @nn.compact
  def __call__(self, inputs):
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    gate_logits = DenseGeneral(
        self.num_experts,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        name="gate",
        matmul_precision=self.config.matmul_precision,
    )(inputs)
    w0_kernel, w1_kernel, wo_kernel = self.generate_kernels(cfg.num_experts, cfg.emb_dim, cfg.mlp_dim)
    if cfg.megablox:
      raise NotImplementedError("MegaBlox is not implemented.")
      # max_logging.log("Running MoE megablox implementation.")
      # if in_serve_mode(self.quant):
      #   w0_kernel, w1_kernel, wo_kernel = self.retrieve_quantized_weight(
      #       inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel
      #   )
      # return self.megablox(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)
    elif getattr(cfg, "use_pallas_kernel", False) and cfg.capacity_factor >= self.num_experts:
      # Dropless path: use the custom Pallas-Triton grouped GEMM kernel.
      # Opt-in via cfg.use_pallas_kernel; default is False so the safe
      # ragged_matmul path is used unless the user explicitly enables this.
      max_logging.log("Running MoE pallas grouped-GEMM implementation.")
      return self.pallas_matmul(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)
    elif getattr(cfg, "use_ragged_dot", True) and cfg.capacity_factor >= self.num_experts:
      # Dropless path: use jax.lax.ragged_dot grouped GEMM. This skips the
      # generate_masks cumsum/one_hot dance and the dispatch/combine einsums,
      # which together waste ~num_experts/k of their compute on zero padding
      # in the original dense_matmul path.
      max_logging.log("Running MoE ragged_dot implementation.")
      return self.ragged_matmul(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)
    else:
      max_logging.log("Running MoE matmul implementation.")
      return self.dense_matmul(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)





# # from .mu_task_base import MuTask
# from learned_optimization.tasks import base
# from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask

# import haiku as hk
# import jax
# import jax.numpy as jnp
# from learned_optimization.tasks import base

# from collections.abc import Iterable
# from typing import Any, Mapping, Tuple, Callable, Optional
# from learned_optimization.tasks import base
# from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask
# from haiku._src.typing import Initializer

# import functools




# State = Any
# Params = Any
# ModelState = Any
# PRNGKey = jnp.ndarray
# Batch = Any





# # MoE MLP implementation in Flax
# import flax.linen as nn
# from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
# from jax.sharding import Mesh
# import functools
# import jax.numpy as jnp
# import optax
# import argparse
# from jax.sharding import Mesh
# import jax
# import time

# class MuMoeMLP(nn.Module):
#   """A multi-layer perceptron module with Mixture of Experts layers."""
  
#   output_sizes: Sequence[int]
#   num_experts: int
#   num_experts_per_tok: int
#   moe_layers: Union[Sequence[int], str] = "all"  # "all" or indices of layers to use MoE
#   w_init: Optional[Callable] = None
#   b_init: Optional[Callable] = None
#   input_mult: float = 1.0
#   output_mult: float = 1.0
#   hidden_lr_mult: float = 1.0
#   with_bias: bool = True
#   activation: Callable = nn.relu
#   activate_final: bool = False
#   log_activations: bool = False
#   mesh: Optional[Mesh] = None
#   capacity_factor: float = 1.5
#   load_balance_loss_weight: float = 0.01
#   dtype: Any = jnp.float32
#   weight_dtype: Any = jnp.float32
#   matmul_precision: str = "default"
#   flip_batch_and_sl_dim: bool = True
  
#   def setup(self):
#     # Calculate output multiplier based on the second-to-last layer size
#     self.used_output_mult = self.output_mult * 1.0 / self.output_sizes[-2]

    
#   @nn.compact
#   def __call__(
#       self,
#       inputs: jnp.ndarray,
#       dropout_rate: Optional[float] = None,
#       deterministic: bool = False,
#       training: bool = True,
#   ) -> jnp.ndarray:
#     """Connects the module to some inputs.
    
#     Args:
#       inputs: A Tensor of shape ``[batch_size, input_size]``.
#       dropout_rate: Optional dropout rate.
#       deterministic: If True, dropout is not applied.
#       training: Whether the model is in training mode.
      
#     Returns:
#       The output of the model of size ``[batch_size, output_size]``.
#     """

#     # Create a config object for MoeBlock
#     class MoeConfig:
#       def __init__(self, parent):
#         self.dtype = parent.dtype
#         self.weight_dtype = parent.weight_dtype
#         self.num_experts = parent.num_experts
#         self.num_experts_per_tok = parent.num_experts_per_tok
#         self.capacity_factor = parent.capacity_factor
#         self.matmul_precision = parent.matmul_precision
#         self.model_call_mode = "train" if training else "eval"
#         self.load_balance_loss_weight = parent.load_balance_loss_weight
#         self.use_moe_linears = True
#         self.ici_expert_parallelism = 1
#         self.dcn_expert_parallelism = 1
#         self.activations_in_float32 = True
#         self.megablox = False
#         self.activation = parent.activation
#         self.mlp_activations = [parent.activation.__name__]

    
#     moe_config = MoeConfig(self)
        
#     x = inputs
#     num_layers = len(self.output_sizes)
    
#     # Track load balance losses
#     load_balance_losses = []
    
#     for i, output_size in enumerate(self.output_sizes):
#       # Determine if this layer should use MoE
#       if self.moe_layers[i] and i < (num_layers - 1):  # Don't use MoE for the final layer

#         # Set MLP dimensions for this layer
#         moe_config.emb_dim = x.shape[-1]
#         moe_config.mlp_dim = output_size
        
#         # Create MoeBlock
#         moe_layer = MoeBlock(
#           config=moe_config,
#           num_experts=self.num_experts,
#           num_experts_per_tok=self.num_experts_per_tok,
#           mesh=self.mesh,
#           # kernel_init=MupVarianceScaling(1.0, "fan_in",  "truncated_normal")
#           kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"), #nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
#           kernel_axes=("embed", "experts"),
#           name=f"{i}_MoeBlock"
#         )
        
#         # Reshape input to (batch_size, sequence_length=1, hidden_size) for MoE layer
#         if self.flip_batch_and_sl_dim:
#           x_reshaped = x.reshape(1,x.shape[0], x.shape[-1])
#         else:
#           x_reshaped = x.reshape(x.shape[0], 1, x.shape[-1])

#         # Apply MoE layer
#         moe_output, loss = moe_layer(x_reshaped)
#         # Reshape back to original shape (batch_size, hidden_size)
#         if self.flip_batch_and_sl_dim:
#           moe_output = moe_output.reshape(moe_output.shape[1], moe_output.shape[2])
#         else:
#           moe_output = moe_output.reshape(moe_output.shape[0], moe_output.shape[-1])
#         load_balance_losses.append(loss)
          
#         x = moe_output
#       else:

          
#         # Regular dense layer
#         if i == 0:
#           # Input layer
#           w_init = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
#           b_init = nn.initializers.normal(stddev=1.0)
#         elif i == num_layers - 1:
#           # Output layer
#           w_init = nn.initializers.zeros
#           b_init = nn.initializers.normal(stddev=1.0)
#         else:
#           # Hidden layer
#           w_init = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
#           b_init = nn.initializers.normal(stddev=1.0)
        
#         x = nn.Dense(
#           features=output_size,
#           use_bias=self.with_bias,
#           kernel_init=w_init,
#           bias_init=b_init,
#           dtype=self.dtype,
#           param_dtype=self.weight_dtype,
#           precision=self.matmul_precision,
#           name=f"{i}_Dense"
#         )(x)
      
#       # Apply scaling based on layer position
#       if i == 0:
#         x = x * self.input_mult
#       elif i < (num_layers - 1):
#         x = x * 1.0  # hidden_mult
      
#       # Log activations if requested
#       if self.log_activations:
#         self.sow("intermediates", f"layer_{i}_pre-act_l1", jnp.mean(jnp.abs(x)))
#         self.sow("intermediates", f"layer_{i}_pre-act", x)
      
#       # Apply activation and dropout
#       if i < (num_layers - 1) or self.activate_final:
#         if dropout_rate is not None and not deterministic:
#           x = nn.Dropout(rate=dropout_rate, deterministic=deterministic)(x)
#         x = self.activation(x)
        
#         if self.log_activations:
#           self.sow("intermediates", f"layer_{i}_act_l1", jnp.mean(jnp.abs(x)))
#           self.sow("intermediates", f"layer_{i}_act", x)
#       else:
#         if self.log_activations:
#           self.sow("intermediates", f"layer_{i}_logits_l1", jnp.mean(jnp.abs(x * self.output_mult)))
#           self.sow("intermediates", f"layer_{i}_logits", x * self.output_mult)
    
#     # Apply final output scaling
#     x = x * self.used_output_mult
    
#     # Compute total load balance loss and store it in intermediates
#     if load_balance_losses:
#       total_load_balance_loss = sum(load_balance_losses)
#       self.sow("intermediates", "load_balance_loss", total_load_balance_loss)
#       # Store individual layer losses
#       for i, loss in enumerate(load_balance_losses):
#         self.sow("intermediates", f"layer_{i}_load_balance_loss", loss)
    
#     return x
  
#   def get_mup_lrs(self, params, device):
#     """Returns the MuP learning rate multipliers that match the parameter structure."""
#     def get_dense(k,v,fan_in, is_last):
#         if "0" in k or is_last: # first and last
#             lrs = {'bias': jnp.array(1.0, dtype=jnp.float32, device=device), 'kernel': jnp.array(1.0, dtype=jnp.float32, device=device)}
#         else: #hidden
#             lrs = {'bias': jnp.array(1.0, dtype=jnp.float32, device=device), 'kernel': jnp.array(1 / fan_i, dtype=jnp.float32, device=device)}
#         return lrs

#     def get_moe(k,v,fan_in):
#         # the following line assumes that the MoE's num experts will scale with width
#         # router is treated as not growing in width 
#         # print(jax.tree_util.tree_map(lambda x: x.shape, v))
#         # print(v['gate'].keys())

#         #use 1/fan_in for the gate kernel to stabilize logits?
#         return {'gate': {'kernel': jax.tree_util.tree_map(lambda x : jnp.array(1/fan_in, dtype=jnp.float32, device=device), v['gate']['kernel']),},
#                 # 'gate': {'kernel': jax.tree_util.tree_map(lambda x : jnp.array(1.0, dtype=jnp.float32, device=device), v['gate']['kernel']),},
#                'wi_0': jax.tree_util.tree_map(lambda x : jnp.array(1/fan_in, dtype=jnp.float32, device=device), v['wi_0']),
#                'wi_1': jax.tree_util.tree_map(lambda x : jnp.array(1/fan_in, dtype=jnp.float32, device=device), v['wi_1']),
#                'wo': jax.tree_util.tree_map(lambda x : jnp.array(1/fan_in, dtype=jnp.float32, device=device), v['wo'])}
        
        
        
#     lr_tree = jax.tree_util.tree_map(lambda x: 1.0, params['params'])
#     num_params = len(lr_tree)
#     # print("num_params",num_params)
#     for i, (k,v) in enumerate(params['params'].items()):
        
#         if k.endswith("Dense"):
#             fan_in = v['kernel'].shape[0]
#             lr_tree[k] = get_dense(k,v,fan_in, i == num_params-1)
            
#         else:
#             fan_in = v['gate']['kernel'].value.shape[0]
#             print(fan_in)
#             lr_tree[k] = get_moe(k,v,fan_in)

#     return {"params" : lr_tree}


# class MuMoeMlpImageTask(base.Task, MuTask):
#   """MLP based image task with MoE layers."""
  
#   def __init__(self, datasets, cfg,
#                mup_multipliers=dict(input_mult=1.0,
#                                     output_mult=1.0,
#                                     hidden_lr_mult=1.0)):
#     cfg['input_mult'] = mup_multipliers['input_mult']
#     cfg['output_mult'] = mup_multipliers['output_mult']
#     self.hidden_lr_mult = mup_multipliers['hidden_lr_mult']
    
#     num_classes = datasets.extra_info["num_classes"]
#     hidden_sizes = cfg.get('hidden_sizes', [128, 128])
#     output_sizes = list(hidden_sizes) + [num_classes]
    
#     # Create mesh for MoE
#     devices = jax.devices()
#     self.mesh = Mesh(devices, ("data",))
    
#     # Configure MoE parameters
#     self.num_experts = cfg.get('num_experts', 8)
#     self.num_experts_per_tok = cfg.get('num_experts_per_tok', 2)
#     self.moe_layers = cfg.get('moe_layers', "all")
#     self.capacity_factor = cfg.get('capacity_factor', 1.5)
#     self.load_balance_loss_weight = cfg.get('load_balance_loss_weight', 0.01)
    
#     # Convert moe_layers from "all" to a list of booleans if needed
#     if self.moe_layers == "all":
#       self.moe_layers = [True] * (len(output_sizes) - 1) + [False]  # No MoE for final layer
    
#     # Create the Flax module
#     self.flax_module = MuMoeMLP(
#         output_sizes=output_sizes,
#         num_experts=self.num_experts,
#         num_experts_per_tok=self.num_experts_per_tok,
#         moe_layers=self.moe_layers,
#         activation=jax.nn.relu,
#         log_activations=cfg.get('log_activations', False),
#         mesh=self.mesh,
#         capacity_factor=self.capacity_factor,
#         load_balance_loss_weight=self.load_balance_loss_weight,
#         **mup_multipliers
#     )
    
#     self.datasets = datasets
#     self.dropout_rate = cfg.get('dropout_rate', 0.0)
#     self.mup_lrs = None
#     self.mup_state = None
#     self.init_mup_state()

#   def init(self, key: PRNGKey):
#     batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
#                                    self.datasets.abstract_batch)
#     # Reshape image data for MLP
#     image_data = jnp.reshape(batch["image"], [batch["image"].shape[0], -1])
    
#     return self.flax_module.init({"params": key, "dropout": key},
#                                 image_data,
#                                 dropout_rate=self.dropout_rate,
#                                 training=True)

#   def init_with_state(self, key: PRNGKey) -> Tuple[Params, ModelState]:
#     params = self.init(key)
#     if self.mup_lrs is None:
#       self.mup_lrs = self.flax_module.get_mup_lrs(params, jax.devices()[jax.process_index()])
#     state = {'flax_mup_lrs': self.mup_lrs}
#     return params, self.get_mup_state(state)

#   @functools.partial(jax.jit, static_argnums=(0,))
#   def loss(self, params: Any, key: PRNGKey, data: Any):
#     # Reshape image data for MLP
#     image_data = jnp.reshape(data["image"], [data["image"].shape[0], -1])
    
#     # Forward pass
#     logits, intermediates = self.flax_module.apply(
#         params,
#         image_data, 
#         dropout_rate=self.dropout_rate,
#         training=True, 
#         # rngs={"dropout": key}
#         mutable=['intermediates']
#     )
    
#     # Calculate loss
#     num_classes = self.datasets.extra_info["num_classes"]
#     labels_onehot = jax.nn.one_hot(data["label"], num_classes)
#     loss_vec = base.softmax_cross_entropy(logits=logits, labels=labels_onehot)
#     task_loss = jnp.mean(loss_vec)
    
#     # Add load balance loss if present
#     total_loss = task_loss
#     if 'intermediates' in intermediates and 'load_balance_loss' in intermediates['intermediates']:
#         load_balance_loss = intermediates['intermediates']['load_balance_loss'][0]
#         total_loss = task_loss + load_balance_loss
    
#     return total_loss

#   @functools.partial(jax.jit, static_argnums=(0,))
#   def loss_with_state(self, params: Any, state: Any, key: PRNGKey, data: Any):
#     # Reshape image data for MLP
#     image_data = jnp.reshape(data["image"], [data["image"].shape[0], -1])
    
#     # Forward pass with intermediates to capture load balance losses
#     logits, intermediates = self.flax_module.apply(
#         params,
#         image_data, 
#         dropout_rate=self.dropout_rate,
#         training=True, 
#         # rngs={"dropout": key},
#         mutable=['intermediates']
#     )

    
#     # Calculate loss
#     num_classes = self.datasets.extra_info["num_classes"]
#     labels_onehot = jax.nn.one_hot(data["label"], num_classes)
#     loss_vec = base.softmax_cross_entropy(logits=logits, labels=labels_onehot)
#     task_loss = jnp.mean(loss_vec)
    
#     # Add load balance loss if present
#     total_loss = task_loss
#     if 'intermediates' in intermediates and 'load_balance_loss' in intermediates['intermediates']:
#         load_balance_loss = intermediates['intermediates']['load_balance_loss'][0]
#         total_loss = task_loss + load_balance_loss
    
#     return total_loss, self.get_mup_state(state)

#   @functools.partial(jax.jit, static_argnums=(0,))
#   def loss_with_state_and_aux(
#       self, params: Params, state: ModelState, key: PRNGKey,
#       data: Batch) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
#     # Reshape image data for MLP
#     image_data = jnp.reshape(data["image"], [data["image"].shape[0], -1])
    
#     # Forward pass with intermediates to capture load balance losses
#     logits, intermediates = self.flax_module.apply(
#         params,
#         image_data, 
#         dropout_rate=self.dropout_rate,
#         training=True, 
#         # rngs={"dropout": key},
#         mutable=['intermediates']
#     )
    
#     # Calculate loss
#     num_classes = self.datasets.extra_info["num_classes"]
#     labels_onehot = jax.nn.one_hot(data["label"], num_classes)
#     loss_vec = base.softmax_cross_entropy(logits=logits, labels=labels_onehot)
#     task_loss = jnp.mean(loss_vec)
    
#     # Prepare aux dict with task loss and load balance losses
#     aux = {'task_loss': task_loss}
    
#     # Add load balance losses if present
#     total_loss = task_loss
#     if 'intermediates' in intermediates and 'load_balance_loss' in intermediates['intermediates']:
#         load_balance_loss = intermediates['intermediates']['load_balance_loss'][0]
#         total_loss = task_loss + load_balance_loss
#         aux['load_balance_loss'] = load_balance_loss
        
#         # Add individual layer load balance losses if available
#         for k, v in intermediates['intermediates'].items():
#             if k.endswith('_load_balance_loss'):
#                 aux[k] = v
    
#     return total_loss, self.get_mup_state(state), aux

#   @functools.partial(jax.jit, static_argnums=(0,))
#   def loss_and_accuracy(self, params: Params, key: PRNGKey, data: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     # Reshape image data for MLP
#     image_data = jnp.reshape(data["image"], [data["image"].shape[0], -1])
    
#     # Forward pass with intermediates to capture load balance losses
#     logits, intermediates = self.flax_module.apply(
#         params,
#         image_data, 
#         dropout_rate=0.0,  # No dropout during evaluation
#         training=False, 
#         # rngs={"dropout": key},
#         mutable=['intermediates']
#     )
    
#     # Calculate loss
#     num_classes = self.datasets.extra_info["num_classes"]
#     labels_onehot = jax.nn.one_hot(data["label"], num_classes)
#     loss_vec = base.softmax_cross_entropy(logits=logits, labels=labels_onehot)
#     task_loss = jnp.mean(loss_vec)
    
#     # Add load balance loss if present
#     total_loss = task_loss
#     if 'intermediates' in intermediates and 'load_balance_loss' in intermediates['intermediates']:
#         load_balance_loss = intermediates['intermediates']['load_balance_loss'][0]
#         total_loss = task_loss + load_balance_loss
    
#     # Calculate accuracy
#     predictions = jnp.argmax(logits, axis=-1)
#     actual = data["label"]
#     correct_predictions = predictions == actual
#     accuracy = jnp.mean(correct_predictions.astype(jnp.float32))
    
#     return total_loss, accuracy

#   @functools.partial(jax.jit, static_argnums=(0,))
#   def loss_and_accuracy_with_state(self, params: Params, state: State, key: PRNGKey, data: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     loss, accuracy = self.loss_and_accuracy(params, key, data)
#     return loss, accuracy
    

# if __name__ == "__main__":

#   # from layers.initializers import nd_dense_init
  
#   parser = argparse.ArgumentParser(description="Test MoeMLP functionality")
#   parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
#   parser.add_argument("--num_experts_per_tok", type=int, default=2, help="Number of experts per token")
#   parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
#   parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size")
#   parser.add_argument("--input_size", type=int, default=784, help="Input dimension")
#   parser.add_argument("--output_size", type=int, default=10, help="Output dimension")
#   parser.add_argument("--moe_layers", type=str, default="all", help="Which layers use MoE ('all' or comma-separated indices)")
#   args = parser.parse_args()
  
#   # Parse moe_layers argument
#   if args.moe_layers == "all":
#     moe_layers = "all"
#   else:
#     moe_layers = [int(i) for i in args.moe_layers.split(",")]
  
#   # Create a simple mesh for testing
#   devices = jax.devices()
#   mesh = Mesh(devices, ("data",))

#   args.batch_size = 4096
#   args.num_experts = 8
#   args.num_experts_per_tok = 1
#   args.hidden_size = 1024
#   args.input_size = 32*32*3
#   args.output_size = 1000
  
#   # Create random input
#   key = jax.random.PRNGKey(0)
#   inputs = jax.random.normal(key, (args.batch_size, args.input_size))
  
#   # Initialize the MoeMLP
#   moe_mlp = MuMoeMLP(
#       output_sizes=[args.hidden_size, args.hidden_size, args.output_size],
#       num_experts=args.num_experts,
#       num_experts_per_tok=args.num_experts_per_tok,
#       moe_layers=[False, True, False],
#       activation=nn.relu,
#       mesh=mesh,
#       capacity_factor=1.5,
#       load_balance_loss_weight=0.01,
#       dtype=jnp.float32,
#       weight_dtype=jnp.float32,
#       matmul_precision="default",
#       flip_batch_and_sl_dim=False,
#   )
  
#   # Initialize parameters
#   params = moe_mlp.init(key, inputs)
#   import pprint
#   pprint.pprint(jax.tree_util.tree_map(lambda x: x if type(x) in [float, int] else x.shape, params))

#   print("mup_lrs:\n")
#   device = jax.devices()[0]
#   pprint.pprint(moe_mlp.get_mup_lrs(params, device))  
#   # exit(0)
  
















