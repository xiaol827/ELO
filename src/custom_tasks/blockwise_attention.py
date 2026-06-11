"""Blockwise flash attention: O(n) memory, vmap-compatible.

Pure JAX implementation using lax.scan with online softmax.
Drop-in replacement for jax.nn.dot_product_attention(implementation='xla').

Based on EasyLM's blockwise_attn (https://github.com/young-geng/EasyLM).
"""
import functools
from typing import Optional, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class _Carry(NamedTuple):
    numerator: jax.Array
    denominator: jax.Array
    max_so_far: jax.Array


def blockwise_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: Optional[jax.Array] = None,
    scale: Optional[float] = None,
    is_causal: bool = False,
    query_chunk_size: int = 512,
    key_chunk_size: int = 512,
) -> jax.Array:
    """Memory-efficient attention via blockwise computation with online softmax.

    Args:
        query: [B, T, H, Dh] (bf16 or fp32)
        key:   [B, S, H, Dh]
        value: [B, S, H, Dh]
        mask:  Optional [B, 1, T, S] boolean mask (True = attend)
        scale: Attention scale (default: 1/sqrt(Dh))
        is_causal: Whether to apply causal mask
        query_chunk_size: Block size for queries
        key_chunk_size: Block size for keys

    Returns:
        [B, T, H, Dh] attention output
    """
    B, T, H, Dh = query.shape
    S = key.shape[1]

    if scale is None:
        scale = 1.0 / (Dh ** 0.5)

    # Apply scale to query upfront
    query = query * scale

    # Upcast to float32 for numerically stable softmax
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

    # Pad T and S to be divisible by chunk sizes
    pad_t = (query_chunk_size - T % query_chunk_size) % query_chunk_size
    pad_s = (key_chunk_size - S % key_chunk_size) % key_chunk_size
    if pad_t > 0:
        query = jnp.pad(query, ((0, 0), (0, pad_t), (0, 0), (0, 0)))
        if mask is not None:
            mask = jnp.pad(mask, ((0, 0), (0, 0), (0, pad_t), (0, 0)))
    if pad_s > 0:
        key = jnp.pad(key, ((0, 0), (0, pad_s), (0, 0), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, pad_s), (0, 0), (0, 0)))
        if mask is not None:
            mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, pad_s)))

    T_pad = T + pad_t
    S_pad = S + pad_s
    num_q = T_pad // query_chunk_size
    num_kv = S_pad // key_chunk_size

    # Reshape into chunks: (num_chunks, B, chunk_size, H, Dh)
    query = query.reshape(B, num_q, query_chunk_size, H, Dh).transpose(1, 0, 2, 3, 4)
    key = key.reshape(B, num_kv, key_chunk_size, H, Dh).transpose(1, 0, 2, 3, 4)
    value = value.reshape(B, num_kv, key_chunk_size, H, Dh).transpose(1, 0, 2, 3, 4)

    if mask is not None:
        # mask: [B, 1, T_pad, S_pad] -> [num_q, num_kv, B, 1, q_cs, k_cs]
        mask = mask.reshape(B, 1, num_q, query_chunk_size, num_kv, key_chunk_size)
        mask = mask.transpose(2, 4, 0, 1, 3, 5)  # [num_q, num_kv, B, 1, q_cs, k_cs]

    def _compute_chunk_bias(q_idx, kv_idx):
        """Compute causal + optional mask bias for a (q_chunk, kv_chunk) pair.

        Returns bias broadcastable to attn_weights shape (B, q_cs, H, k_cs).
        """
        bias = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)

        if is_causal:
            q_offset = q_idx * query_chunk_size
            kv_offset = kv_idx * key_chunk_size
            q_pos = jnp.arange(query_chunk_size).reshape(query_chunk_size, 1) + q_offset
            kv_pos = jnp.arange(key_chunk_size).reshape(1, key_chunk_size) + kv_offset
            causal_bias = jnp.where(q_pos < kv_pos, jnp.finfo(jnp.float32).min, 0.0)
            # Shape: (1, q_cs, 1, k_cs) to broadcast with (B, q_cs, H, k_cs)
            bias = bias + causal_bias.reshape(1, query_chunk_size, 1, key_chunk_size)

        if mask is not None:
            # mask_chunk: [B, 1, q_cs, k_cs] -> reshape to (B, q_cs, 1, k_cs)
            mask_chunk = mask[q_idx, kv_idx]  # [B, 1, q_cs, k_cs]
            mask_chunk = mask_chunk.transpose(0, 2, 1, 3)  # [B, q_cs, 1, k_cs]
            mask_bias = jnp.where(mask_chunk, 0.0, jnp.finfo(jnp.float32).min)
            bias = bias + mask_bias

        return bias

    def scan_attention(args):
        query_chunk, q_idx = args

        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable())
        def scan_kv_block(carry, args):
            key_chunk, value_chunk, kv_idx = args
            numerator, denominator, prev_max = carry

            # Attention scores: [B, q_cs, H, k_cs]
            attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk)

            # Add causal + mask bias
            bias = _compute_chunk_bias(q_idx, kv_idx)
            attn_weights = attn_weights + bias

            # Online softmax update
            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
            max_score = jnp.maximum(prev_max, max_score)
            max_score = jax.lax.stop_gradient(max_score)

            exp_weights = jnp.exp(attn_weights - max_score)
            exp_values = jnp.einsum('bqhk,bkhd->bqhd', exp_weights, value_chunk)

            correction = jnp.exp(prev_max - max_score)
            numerator = numerator * correction + exp_values
            denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)

            return _Carry(numerator, denominator, max_score), None

        init_carry = _Carry(
            jnp.zeros((B, query_chunk_size, H, Dh), dtype=jnp.float32),
            jnp.zeros((B, query_chunk_size, H, 1), dtype=jnp.float32),
            jnp.full((B, query_chunk_size, H, 1), -jnp.inf, dtype=jnp.float32),
        )

        (numerator, denominator, _), _ = lax.scan(
            scan_kv_block, init_carry, (key, value, jnp.arange(num_kv))
        )
        return numerator / denominator

    _, outputs = lax.scan(
        lambda _, x: ((), scan_attention(x)),
        (), (query, jnp.arange(num_q))
    )

    # outputs: [num_q, B, q_cs, H, Dh] -> [B, T_pad, H, Dh]
    outputs = outputs.transpose(1, 0, 2, 3, 4).reshape(B, T_pad, H, Dh)

    # Remove padding
    if pad_t > 0:
        outputs = outputs[:, :T, :, :]

    return outputs
