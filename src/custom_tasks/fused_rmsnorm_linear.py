"""Fused RMSNorm + Linear via custom_vjp.

Eliminates intermediate HBM write/read of normalized activations by saving
only (x, rms_inv, scale, weight) as residuals and recomputing x_norm in
the backward pass with 2 cheap element-wise ops.

Mixed precision: `compute_dtype` is the matmul operand dtype. Normalization
statistics are computed in float32, the normalized activations are cast to
compute_dtype and multiplied by the (cast) scale exactly like the RMSNorm
module, and the weight is cast to compute_dtype for the matmul. Gradients are
returned in the dtypes of the primal inputs (x, scale, weight), i.e. the
weight/scale gradients accumulate into the float32 master params.

Memory saved per fusion site: B * L * D * sizeof(compute_dtype) bytes.
FLOP overhead: negligible (2 element-wise ops in backward vs storing full tensor).
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax


def _norm_stats(x, epsilon):
    x_f32 = x.astype(jnp.float32)
    rms_inv = lax.rsqrt(jnp.mean(lax.square(x_f32), axis=-1, keepdims=True) + epsilon)
    return x_f32, rms_inv


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def fused_rmsnorm_linear(x, scale, weight, epsilon=1e-6, compute_dtype=jnp.float32):
    """Compute RMSNorm(x, scale, eps) @ weight with fused backward.

    Args:
        x: (*batch, D) input tensor (residual stream, typically float32)
        scale: (D,) RMSNorm learnable scale
        weight: (D, F) linear projection weight
        epsilon: RMSNorm epsilon
        compute_dtype: matmul operand dtype (float32 or bfloat16)

    Returns:
        (*batch, F) output tensor in compute_dtype
    """
    x_f32, rms_inv = _norm_stats(x, epsilon)
    x_norm = jnp.asarray(x_f32 * rms_inv, compute_dtype) * scale.astype(compute_dtype)
    return x_norm @ weight.astype(compute_dtype)


def _fused_fwd(x, scale, weight, epsilon, compute_dtype):
    x_f32, rms_inv = _norm_stats(x, epsilon)
    x_norm = jnp.asarray(x_f32 * rms_inv, compute_dtype) * scale.astype(compute_dtype)
    out = x_norm @ weight.astype(compute_dtype)
    # Save compact residuals — x_norm is NOT saved (recomputed in backward)
    return out, (x, rms_inv, scale, weight)


def _fused_bwd(epsilon, compute_dtype, residuals, dout):
    x, rms_inv, scale, weight = residuals
    D = x.shape[-1]

    # Recompute x_norm from saved residuals (the key memory saving)
    x_f32 = x.astype(jnp.float32)
    y = jnp.asarray(x_f32 * rms_inv, compute_dtype)          # normalized, before scale
    scale_c = scale.astype(compute_dtype)
    x_norm = y * scale_c
    weight_c = weight.astype(compute_dtype)

    # Gradient through matmul: dout @ weight.T
    dx_norm = dout @ weight_c.T  # (*batch, D), compute_dtype

    # Gradient for weight: x_norm.T @ dout, accumulated into the weight's dtype
    x_norm_2d = x_norm.reshape(-1, x_norm.shape[-1])
    dout_2d = dout.reshape(-1, dout.shape[-1])
    dweight = jnp.dot(x_norm_2d.T, dout_2d, preferred_element_type=weight.dtype)  # (D, F)

    # Gradient for scale: sum over batch dims of (dx_norm * y), reduced in float32
    dscale = jnp.sum(
        (dx_norm * y).astype(jnp.float32),
        axis=tuple(range(x.ndim - 1))
    ).astype(scale.dtype)  # (D,)

    # Gradient for x through RMSNorm (float32):
    # y = x * rms_inv ;  dy = dx_norm * scale
    # dx = dy * rms_inv - (rms_inv^2 / D) * x * dot(dy, x * rms_inv)
    dy = (dx_norm * scale_c).astype(jnp.float32)  # (*batch, D)
    inner = jnp.sum(dy * x_f32 * rms_inv, axis=-1, keepdims=True)  # (*batch, 1)
    dx = jnp.asarray(
        dy * rms_inv - (rms_inv * rms_inv / D) * x_f32 * inner,
        x.dtype
    )

    return dx, dscale, dweight


fused_rmsnorm_linear.defvjp(_fused_fwd, _fused_bwd)
