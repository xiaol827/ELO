"""Fused RMSNorm + Linear via custom_vjp.

Eliminates intermediate HBM write/read of normalized activations by saving
only (x, rms_inv, scale, weight) as residuals and recomputing x_norm in
the backward pass with 2 cheap element-wise ops.

Memory saved per fusion site: B * L * D * sizeof(dtype) bytes.
FLOP overhead: negligible (2 element-wise ops in backward vs storing full tensor).
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def fused_rmsnorm_linear(x, scale, weight, epsilon=1e-6):
    """Compute RMSNorm(x, scale, eps) @ weight with fused backward.

    Args:
        x: (*batch, D) input tensor
        scale: (D,) RMSNorm learnable scale
        weight: (D, F) linear projection weight
        epsilon: RMSNorm epsilon

    Returns:
        (*batch, F) output tensor
    """
    x_f32 = x.astype(jnp.float32)
    rms_inv = lax.rsqrt(jnp.mean(lax.square(x_f32), axis=-1, keepdims=True) + epsilon)
    x_norm = jnp.asarray(x_f32 * rms_inv, x.dtype) * scale
    return x_norm @ weight


def _fused_fwd(x, scale, weight, epsilon):
    x_f32 = x.astype(jnp.float32)
    rms_inv = lax.rsqrt(jnp.mean(lax.square(x_f32), axis=-1, keepdims=True) + epsilon)
    x_norm = jnp.asarray(x_f32 * rms_inv, x.dtype) * scale
    out = x_norm @ weight
    # Save compact residuals — x_norm is NOT saved (recomputed in backward)
    return out, (x, rms_inv, scale, weight)


def _fused_bwd(epsilon, residuals, dout):
    x, rms_inv, scale, weight = residuals
    D = x.shape[-1]

    # Recompute x_norm from saved residuals (the key memory saving)
    x_f32 = x.astype(jnp.float32)
    x_norm = jnp.asarray(x_f32 * rms_inv, x.dtype) * scale

    # Gradient through matmul: dout @ weight.T
    dx_norm = dout @ weight.T  # (*batch, D)

    # Gradient for weight: x_norm.T @ dout
    # Reshape to 2D for matmul: (*batch, D) -> (M, D) and (*batch, F) -> (M, F)
    x_norm_2d = x_norm.reshape(-1, x_norm.shape[-1])
    dout_2d = dout.reshape(-1, dout.shape[-1])
    dweight = x_norm_2d.T @ dout_2d  # (D, F)

    # Gradient for scale: sum over batch dims of (dx_norm * x * rms_inv)
    dscale = jnp.sum(
        dx_norm * jnp.asarray(x_f32 * rms_inv, x.dtype),
        axis=tuple(range(x.ndim - 1))
    )  # (D,)

    # Gradient for x through RMSNorm:
    # x_norm = cast(x * rms_inv) * scale
    # dx = dx_norm * scale * rms_inv - (rms_inv^2 / D) * x * dot(dx_norm * scale, x * rms_inv)
    dx_norm_scaled = dx_norm * scale  # (*batch, D)
    # Inner product per sample: sum_d(dx_norm_scaled * x_f32 * rms_inv)
    inner = jnp.sum(
        dx_norm_scaled.astype(jnp.float32) * x_f32 * rms_inv,
        axis=-1, keepdims=True
    )  # (*batch, 1)
    dx = jnp.asarray(
        dx_norm_scaled.astype(jnp.float32) * rms_inv
        - (rms_inv * rms_inv / D) * x_f32 * inner,
        x.dtype
    )

    return dx, dscale, dweight


fused_rmsnorm_linear.defvjp(_fused_fwd, _fused_bwd)
