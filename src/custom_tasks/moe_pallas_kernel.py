"""Minimal Pallas-Triton grouped GEMM kernel for the MoE forward path.

This is a from-scratch implementation of the same operation that
`jax.lax.ragged_dot` performs:

    out[m, n] = sum_k lhs[m, k] * rhs[g, k, n]

where `g` is the group index for row `m`, defined by `group_sizes` (a 1D
int32 array of length `num_groups`). Specifically, rows `0..group_sizes[0]`
are mapped to group 0, the next `group_sizes[1]` rows to group 1, etc. The
`lhs` rows MUST already be sorted by group (the caller is responsible).

Why this exists: `jax.lax.ragged_dot` on H100 dispatches per-group internally,
which costs kernel-launch overhead. A single fused Pallas-Triton kernel can
process all (m_tile, n_tile, group_id) combinations as one launch, using
program_id along the group axis to look up the per-group `lhs` row range
via cumulative-sum of `group_sizes`.

Structure inspired by `tokamax/_src/ops/ragged_dot/pallas_triton.py` (Apache
2.0, openxla/tokamax). We do NOT pip-depend on tokamax because (a) it requires
jax >= 0.9.2 and we have 0.9.0/0.8.1, and (b) it brings qwix/pydantic/etc.
deps we don't want. The kernel below uses ONLY `jax.experimental.pallas` and
its triton backend, both of which ship with stock jax.

Equivalence requirement: when called in fp32 mode, the output must be
allclose(atol=1e-6, rtol=1e-5) to `jax.lax.ragged_dot` on the same inputs.
This is checked by `bench/test_moe_equivalence.py`.

License: Apache 2.0 (matches scaling_l2o; the algorithm is from the megablocks
paper, https://arxiv.org/abs/2211.15841).
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


@dataclasses.dataclass(frozen=True)
class GroupedMatmulConfig:
  """Tile / scheduling parameters for the Pallas grouped GEMM kernel."""
  block_m: int = 32
  block_n: int = 256
  block_k: int = 32
  num_warps: int = 4
  num_stages: int = 3


# Forward-pass defaults — winning unified config from `bench/tile_sweep.py`
# on H100 (rorqual) across both small (w128-d4) and large (w1024-d12) shapes:
#   small/fwd:  83.8us (vs old (32,128,64,4,3) 85.2us, 1.02x)
#   large/fwd: 751.8us (vs old 1058.5us, **1.41x**)
# Rank 1/344 on H100, rank 7/330 on A100 — transfers well across GPUs.
DEFAULT_CONFIG = GroupedMatmulConfig()


# drhs-pass defaults — winning unified config from `bench/tile_sweep.py` for
# the dedicated drhs kernel (tiles (K,N), contracts along M):
#   small/drhs: 106.0us (vs old (32,128,64,4,3) 131.9us, 1.24x)
#   large/drhs: 847.6us (vs old 2397.2us, **2.83x**)
# Rank 1/365 on H100. The drhs kernel benefits from very wide block_n (256)
# and small block_m (16) — wider n-tiles amortize the M loop.
DEFAULT_DRHS_CONFIG = GroupedMatmulConfig(
    block_m=16, block_n=256, block_k=32, num_warps=2, num_stages=4,
)


def _grouped_matmul_kernel(
    lhs_ref,
    rhs_ref,
    group_lo_ref,
    group_hi_ref,
    out_ref,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
):
  """Per-tile kernel: produces a (block_m, block_n) chunk of the output.

  Grid: (num_m_tiles_per_group, num_n_tiles, num_groups).

  Each program is responsible for the output rows
  `[group_lo + pid_m * block_m, group_lo + (pid_m + 1) * block_m)` of group
  `pid_g`, columns `[pid_n * block_n, (pid_n + 1) * block_n)`. If
  `start_m >= group_hi` (this tile is past the end of the group), the
  program returns immediately and stores nothing — those tiles are wasted
  but cheap (one if-check + early return).
  """
  pid_m = pl.program_id(0)
  pid_n = pl.program_id(1)
  pid_g = pl.program_id(2)

  lo = group_lo_ref[pid_g]
  hi = group_hi_ref[pid_g]
  start_m = lo + pid_m * block_m
  start_n = pid_n * block_n

  @pl.when(start_m < hi)
  def _do_tile():
    offs_m = start_m + jnp.arange(block_m)
    offs_n = start_n + jnp.arange(block_n)
    mask_m = offs_m < hi
    n_total = out_ref.shape[1]
    mask_n = offs_n < n_total

    acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
    k_total = lhs_ref.shape[1]
    n_k_blocks = pl.cdiv(k_total, block_k)

    def body(i, acc):
      offs_k = i * block_k + jnp.arange(block_k)
      mask_k = offs_k < k_total
      a = plgpu.load(
          lhs_ref.at[offs_m[:, None], offs_k[None, :]],
          mask=mask_m[:, None] & mask_k[None, :],
          other=0.0,
      )
      b = plgpu.load(
          rhs_ref.at[pid_g, offs_k[:, None], offs_n[None, :]],
          mask=mask_k[:, None] & mask_n[None, :],
          other=0.0,
      )
      a = a.astype(jnp.float32)
      b = b.astype(jnp.float32)
      return acc + pl.dot(a, b)

    acc = jax.lax.fori_loop(0, n_k_blocks, body, acc)
    plgpu.store(
        out_ref.at[offs_m[:, None], offs_n[None, :]],
        acc.astype(out_ref.dtype),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _grouped_matmul_drhs_kernel(
    lhs_ref,         # (M, K)
    dout_ref,        # (M, N)
    group_lo_ref,    # (G,)
    group_hi_ref,    # (G,)
    out_ref,         # (G, K, N)
    *,
    block_m: int,
    block_k: int,
    block_n: int,
):
  """Per-tile kernel for the drhs gradient (ragged contracting dim).

  Computes one (BLOCK_K, BLOCK_N) tile of `out[pid_g, :, :]`:

      out[g, k, n] = sum_{m: g(m)=g} lhs[m, k] * dout[m, n]

  Grid: (cdiv(K, block_k), cdiv(N, block_n), num_groups).

  Each program loops over the M dim in chunks of `block_m`, accumulating
  `lhs[m_chunk, k_block].T @ dout[m_chunk, n_block]` into a (block_k, block_n)
  fp32 accumulator. The number of M chunks is `ceil((hi - lo) / block_m)`,
  variable per group — `jax.lax.fori_loop` handles the dynamic loop bound.
  Empty groups (lo == hi) run zero iterations and write zeros.
  """
  pid_k = pl.program_id(0)
  pid_n = pl.program_id(1)
  pid_g = pl.program_id(2)

  lo = group_lo_ref[pid_g]
  hi = group_hi_ref[pid_g]

  start_k = pid_k * block_k
  start_n = pid_n * block_n
  offs_k = start_k + jnp.arange(block_k)
  offs_n = start_n + jnp.arange(block_n)
  k_total = lhs_ref.shape[1]
  n_total = dout_ref.shape[1]
  mask_k = offs_k < k_total
  mask_n = offs_n < n_total

  acc = jnp.zeros((block_k, block_n), dtype=jnp.float32)

  # Number of M-chunks for this group. ceil((hi - lo) / block_m).
  group_size = hi - lo
  n_iters = pl.cdiv(group_size, block_m)

  def body(i, acc):
    m_start = lo + i * block_m
    offs_m = m_start + jnp.arange(block_m)
    mask_m = offs_m < hi
    a = plgpu.load(  # (block_m, block_k)
        lhs_ref.at[offs_m[:, None], offs_k[None, :]],
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    )
    b = plgpu.load(  # (block_m, block_n)
        dout_ref.at[offs_m[:, None], offs_n[None, :]],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    return acc + pl.dot(a.T, b)  # (block_k, block_n)

  acc = jax.lax.fori_loop(0, n_iters, body, acc)

  plgpu.store(
      out_ref.at[pid_g, offs_k[:, None], offs_n[None, :]],
      acc.astype(out_ref.dtype),
      mask=mask_k[:, None] & mask_n[None, :],
  )


def _grouped_matmul_pallas(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    config: GroupedMatmulConfig,
    out_dtype: jnp.dtype,
) -> jax.Array:
  """Raw pallas-call wrapper. Forward-only — see `grouped_matmul` for the
  custom_vjp version that supports `jax.grad`."""
  m, k = lhs.shape
  num_groups, _, n = rhs.shape

  block_m = config.block_m
  block_n = config.block_n
  block_k = config.block_k

  cum = jnp.cumulative_sum(group_sizes.astype(jnp.int32), include_initial=True)
  group_lo = cum[:-1]   # (G,)
  group_hi = cum[1:]    # (G,)

  num_m_tiles = pl.cdiv(m, block_m)
  num_n_tiles = pl.cdiv(n, block_n)

  kernel = functools.partial(
      _grouped_matmul_kernel,
      block_m=block_m,
      block_n=block_n,
      block_k=block_k,
  )

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
      grid=(num_m_tiles, num_n_tiles, num_groups),
      compiler_params=plgpu.CompilerParams(
          num_warps=config.num_warps,
          num_stages=config.num_stages,
      ),
  )(lhs, rhs, group_lo, group_hi)


def _grouped_matmul_drhs_pallas(
    lhs: jax.Array,
    dout: jax.Array,
    group_sizes: jax.Array,
    *,
    config: GroupedMatmulConfig,
    out_dtype: jnp.dtype,
) -> jax.Array:
  """Pallas-Triton drhs kernel: computes the rhs gradient for grouped_matmul.

  Given lhs (M, K), dout (M, N), and group_sizes (G,), returns drhs (G, K, N)
  where drhs[g, k, n] = sum_{m: g(m)=g} lhs[m, k] * dout[m, n].

  Used by `_grouped_matmul_bwd` to compute the rhs gradient via a single
  Pallas kernel call instead of falling back to `jax.lax.ragged_dot`'s vjp.
  """
  M, K = lhs.shape
  M_dout, N = dout.shape
  if M != M_dout:
    raise ValueError(f"lhs.shape[0] ({M}) must equal dout.shape[0] ({M_dout})")
  G = group_sizes.shape[0]

  block_m = config.block_m
  block_k = config.block_k
  block_n = config.block_n

  cum = jnp.cumulative_sum(group_sizes.astype(jnp.int32), include_initial=True)
  group_lo = cum[:-1]   # (G,)
  group_hi = cum[1:]    # (G,)

  num_k_tiles = pl.cdiv(K, block_k)
  num_n_tiles = pl.cdiv(N, block_n)

  kernel = functools.partial(
      _grouped_matmul_drhs_kernel,
      block_m=block_m,
      block_k=block_k,
      block_n=block_n,
  )

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((G, K, N), out_dtype),
      grid=(num_k_tiles, num_n_tiles, G),
      compiler_params=plgpu.CompilerParams(
          num_warps=config.num_warps,
          num_stages=config.num_stages,
      ),
  )(lhs, dout, group_lo, group_hi)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def grouped_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    config: GroupedMatmulConfig = DEFAULT_CONFIG,
    out_dtype: jnp.dtype | None = None,
) -> jax.Array:
  """Pallas-Triton grouped GEMM, drop-in replacement for `jax.lax.ragged_dot`.

  Forward uses our custom Pallas kernel. Backward delegates to
  `jax.lax.ragged_dot` via `jax.vjp` so the gradients are bit-equivalent to
  the ragged path (which we know is allclose to dense_matmul). This means:

    - inference / forward-only workloads benefit from the fast Pallas kernel
    - training workloads get the same backward as the ragged path
      (a dedicated Pallas backward kernel is future work)

  Args:
    lhs: (M, K) input. Rows MUST be sorted by group.
    rhs: (G, K, N) per-group weight tensor.
    group_sizes: (G,) int32, number of rows of `lhs` per group.
                 sum(group_sizes) <= M; trailing rows are ignored.
    config: tile sizes and Triton stage/warp counts. Static, not differentiated.
    out_dtype: output dtype. Static, not differentiated. Defaults to
               `jnp.promote_types(lhs.dtype, rhs.dtype)`.

  Returns:
    out: (M, N) array of dtype `out_dtype`.
  """
  # Validation runs in the wrapper, before custom_vjp dispatches.
  if lhs.ndim != 2:
    raise ValueError(f"lhs must be 2D (M, K); got shape {lhs.shape}")
  if rhs.ndim != 3:
    raise ValueError(f"rhs must be 3D (G, K, N); got shape {rhs.shape}")
  if group_sizes.ndim != 1:
    raise ValueError(f"group_sizes must be 1D; got shape {group_sizes.shape}")
  if rhs.shape[0] != group_sizes.shape[0]:
    raise ValueError(
        f"rhs.shape[0] ({rhs.shape[0]}) must equal group_sizes.shape[0] "
        f"({group_sizes.shape[0]})"
    )
  if lhs.shape[1] != rhs.shape[1]:
    raise ValueError(
        f"lhs.shape[1] ({lhs.shape[1]}) must equal rhs.shape[1] "
        f"({rhs.shape[1]}) (the contracting axis)"
    )
  if out_dtype is None:
    out_dtype = jnp.promote_types(lhs.dtype, rhs.dtype)
  return _grouped_matmul_pallas(
      lhs, rhs, group_sizes, config=config, out_dtype=out_dtype
  )


def _grouped_matmul_fwd(lhs, rhs, group_sizes, config, out_dtype):
  if out_dtype is None:
    out_dtype = jnp.promote_types(lhs.dtype, rhs.dtype)
  out = _grouped_matmul_pallas(
      lhs, rhs, group_sizes, config=config, out_dtype=out_dtype
  )
  return out, (lhs, rhs, group_sizes)


def _grouped_matmul_bwd(config, out_dtype, residuals, dout):
  """Backward for grouped_matmul — both gradients via Pallas kernels.

  Two gradient computations:

    dlhs[m, k]    = sum_n dout[m, n] * rhs[g(m), k, n]
    drhs[g, k, n] = sum_{m: g(m)=g} lhs[m, k] * dout[m, n]

  dlhs has the same structure as the forward kernel — pass `dout` as the new
  lhs and `rhs.swapaxes(-1, -2)` (shape (G, N, K)) as the new rhs to reuse
  the existing forward pallas kernel directly:

    fwd_kernel(dout, rhs_T, group_sizes) computes
       out[m, k] = sum_n dout[m, n] * rhs_T[g(m), n, k]
                 = sum_n dout[m, n] * rhs[g(m), k, n]
                 = dlhs[m, k]

  drhs is computed via the dedicated `_grouped_matmul_drhs_pallas` kernel
  (ragged contracting dim — for each group g, do a normal matmul on the rows
  of group g, with the M dim as the contraction).
  """
  del out_dtype  # output dtype is determined by lhs dtype on the backward
  lhs, rhs, group_sizes = residuals

  # dlhs via the forward pallas kernel applied to (dout, rhs.T).
  rhs_T = jnp.swapaxes(rhs, -1, -2)  # (G, N, K)
  dlhs = _grouped_matmul_pallas(
      dout.astype(rhs.dtype),
      rhs_T,
      group_sizes,
      config=config,
      out_dtype=lhs.dtype,
  )

  # drhs via the dedicated pallas drhs kernel (ragged contracting dim).
  # The drhs kernel has a different shape from the forward kernel (it tiles
  # (K, N) instead of (M, N) and contracts over M instead of K), so it has
  # its own tile defaults — DEFAULT_DRHS_CONFIG, derived from the same A100
  # tile sweep that picked DEFAULT_CONFIG above.
  drhs = _grouped_matmul_drhs_pallas(
      lhs,
      dout.astype(lhs.dtype),
      group_sizes,
      config=DEFAULT_DRHS_CONFIG,
      out_dtype=rhs.dtype,
  )

  return dlhs, drhs, None  # group_sizes has no gradient


grouped_matmul.defvjp(_grouped_matmul_fwd, _grouped_matmul_bwd)
