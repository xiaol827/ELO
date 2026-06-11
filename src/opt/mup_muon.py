"""
MuP Muon optimizer with per-parameter scaling for CompletedP hyperparameter transfer.

Implements the Muon optimizer (Newton-Schulz orthogonalized momentum) with correct
mu-P scaling rules from Qiu et al. (2025) "Hyperparameter Transfer Enables Consistent
Gains of Matrix-Preconditioned Optimizers Across Scales".

Key differences from optax.contrib.muon:
- Uses sqrt(d_out/d_in) shape scaling (not sqrt(max/min) which is Moonlight/Kimi style)
- Supports per-parameter epsilon scaling (needed for depth transfer: eps ~ 1/L)
- Supports per-parameter learning rate and weight decay scaling

The optimizer splits parameters into two groups:
- Muon params (hidden 2D/3D weights): Newton-Schulz + momentum
- Adam params (embeddings, biases, norms, readout): Standard MuP Adam

References:
- Qiu et al. (2025): arxiv.org/abs/2512.05620
- Jordan et al. (2024): Muon optimizer
"""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union
import math

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base
from optax._src import numerics

from opt.mup_adamw import (
    scale_by_mup_adam,
    scale_by_mup_lr,
    mup_add_decayed_weights,
)


def _get_make_muon_weight_dimension_numbers():
    """Lazy import to avoid circular dependency with new_optimizers.py."""
    from opt.new_optimizers import make_muon_weight_dimension_numbers
    return make_muon_weight_dimension_numbers


# ============================================================================
# Newton-Schulz Orthogonalization (self-contained, avoids haiku dependency)
# Adapted from celo2_optax.py
# ============================================================================

def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> jax.Array:
    """Newton-Schulz orthogonalization for Muon optimizer."""
    if x.ndim < 2:
        raise ValueError(f'Input must have >= 2 dims, got {x.shape}')
    if ns_coeffs.shape != (3,):
        raise ValueError(f'ns_coeffs must have shape (3,), got {ns_coeffs}')

    def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
        x_mT = jnp.swapaxes(x, -2, -1)
        a = x @ x_mT
        b = coeffs[1] * a + coeffs[2] * a @ a
        return coeffs[0] * x + b @ x

    transposed = False
    if x.shape[-2] > x.shape[-1]:
        x = jnp.swapaxes(x, -2, -1)
        transposed = True
    x /= (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(
        0, ns_steps,
        lambda _, x: newton_schulz_iterator(x, ns_coeffs), x,
    )
    if transposed:
        x = jnp.swapaxes(x, -2, -1)
    return x


# ============================================================================
# State Classes
# ============================================================================

class ScaleByMuonNSState(NamedTuple):
    """State for Muon (momentum + Newton-Schulz) transform."""
    count: chex.Array   # shape=(), dtype=jnp.int32
    mu: base.Updates    # Momentum buffer


# ============================================================================
# Core Muon Transform: Momentum + Newton-Schulz
# ============================================================================

def scale_by_muon_ns(
    ns_coeffs: Tuple[float, float, float] = (3.4445, -4.775, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.9,
    eps: float = 1e-8,
    nesterov: bool = True,
    mu_dtype: Optional[chex.ArrayDType] = None,
    eps_scales: Optional[base.Params] = None,
) -> base.GradientTransformation:
    """Core Muon transform: momentum followed by Newton-Schulz orthogonalization.

    This does NOT include shape scaling or learning rate — those are applied
    as separate transforms in the chain.

    Args:
        ns_coeffs: Newton-Schulz polynomial coefficients.
        ns_steps: Number of Newton-Schulz iterations.
        beta: Momentum coefficient.
        eps: Base epsilon for Newton-Schulz normalization.
        nesterov: Whether to use Nesterov momentum.
        mu_dtype: Optional dtype for momentum buffer (e.g., bfloat16).
        eps_scales: Optional per-parameter epsilon scales. If provided,
            effective eps for each param is eps * eps_scales[param].
    """
    ns_coeffs_arr = jnp.array(ns_coeffs, dtype=jnp.float32)

    def init_fn(params: base.Params) -> ScaleByMuonNSState:
        mu = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p, dtype=mu_dtype or p.dtype),
            params
        )
        return ScaleByMuonNSState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
        )

    def update_fn(
        updates: base.Updates,
        state: ScaleByMuonNSState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, ScaleByMuonNSState]:
        count_inc = numerics.safe_increment(state.count)

        # Momentum update
        new_mu = jax.tree_util.tree_map(
            lambda m, g: beta * m + (1 - beta) * g.astype(m.dtype),
            state.mu, updates
        )

        if nesterov:
            # Nesterov: use beta * new_mu + (1 - beta) * grad
            updates_for_ns = jax.tree_util.tree_map(
                lambda m, g: (beta * m + (1 - beta) * g).astype(jnp.float32),
                new_mu, updates
            )
        else:
            updates_for_ns = jax.tree_util.tree_map(
                lambda m: m.astype(jnp.float32),
                new_mu
            )

        # Newton-Schulz orthogonalization with per-param epsilon
        def _apply_ns(grad, eps_scale=None):
            effective_eps = eps * eps_scale if eps_scale is not None else eps
            original_shape = grad.shape
            # Reshape to 2D for Newton-Schulz (handle 3D attention kernels)
            if grad.ndim == 3:
                # (H, Dh, D) or (D, H, Dh) -> reshape to 2D
                grad_2d = grad.reshape(grad.shape[0], -1) if grad.shape[0] >= grad.shape[-1] else grad.reshape(-1, grad.shape[-1])
            elif grad.ndim == 2:
                grad_2d = grad
            else:
                return grad  # Shouldn't happen for Muon params

            orth = orthogonalize_via_newton_schulz(
                grad_2d, ns_coeffs_arr, ns_steps, effective_eps
            )
            return orth.reshape(original_shape)

        if eps_scales is not None:
            new_updates = jax.tree_util.tree_map(
                _apply_ns, updates_for_ns, eps_scales
            )
        else:
            new_updates = jax.tree_util.tree_map(
                lambda g: _apply_ns(g), updates_for_ns
            )

        return new_updates, ScaleByMuonNSState(count=count_inc, mu=new_mu)

    return base.GradientTransformation(init_fn, update_fn)


# ============================================================================
# Correct Shape Scaling: sqrt(d_out / d_in)
# ============================================================================

def scale_by_muon_shape(
    muon_weight_dimension_numbers_fn: Optional[Callable] = None,
) -> base.GradientTransformation:
    """Apply correct sqrt(d_out/d_in) shape scaling for Muon mu-P.

    From Qiu et al. Table 1: the per-layer Muon LR is eta ~ sqrt(d_out/d_in).
    This replaces optax.contrib.muon's sqrt(max(m,n)/min(m,n)) scaling which
    gives wrong transfer for layers where d_out < d_in (e.g., MLP down).

    The shape scaling is computed from the MuonDimensionNumbers which specify
    reduction and output axes for each parameter.

    Args:
        muon_weight_dimension_numbers_fn: Function that takes params and returns
            a pytree of MuonDimensionNumbers (or None for non-Muon params).
            If None, uses the default make_muon_weight_dimension_numbers.
    """
    if muon_weight_dimension_numbers_fn is None:
        muon_weight_dimension_numbers_fn = _get_make_muon_weight_dimension_numbers()

    def init_fn(params: base.Params) -> base.EmptyState:
        return base.EmptyState()

    def update_fn(
        updates: base.Updates,
        state: base.EmptyState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, base.EmptyState]:
        if params is None:
            raise ValueError("params required for shape scaling")

        dim_nums = muon_weight_dimension_numbers_fn(params)

        def _apply_shape_scale(update, param, dim_num):
            from optax.transforms._masking import MaskedNode
            if dim_num is None or isinstance(dim_num, MaskedNode):
                return update  # Non-Muon param or masked, no shape scaling
            if isinstance(param, MaskedNode) or isinstance(update, MaskedNode):
                return update
            shape = param.shape

            # Compute d_in (reduction dims) and d_out (output dims)
            if hasattr(dim_num, 'reduction_axis'):
                red_axes = dim_num.reduction_axis
                out_axes = dim_num.output_axis
                if isinstance(red_axes, int):
                    red_axes = (red_axes,)
                if isinstance(out_axes, int):
                    out_axes = (out_axes,)
                d_in = 1
                for ax in red_axes:
                    d_in *= shape[ax]
                d_out = 1
                for ax in out_axes:
                    d_out *= shape[ax]
            else:
                # Simple 2D case: (reduction_axis, output_axis) as ints
                d_in = shape[dim_num[0]] if isinstance(dim_num[0], int) else math.prod(shape[a] for a in dim_num[0])
                d_out = shape[dim_num[1]] if isinstance(dim_num[1], int) else math.prod(shape[a] for a in dim_num[1])

            # sqrt(d_out / d_in) — the correct mu-P shape factor
            shape_scale = math.sqrt(d_out / d_in)
            return update * shape_scale

        from optax.transforms._masking import MaskedNode
        is_leaf = lambda x: x is None or isinstance(x, MaskedNode) or (hasattr(x, 'reduction_axis'))

        new_updates = jax.tree_util.tree_map(
            _apply_shape_scale, updates, params, dim_nums,
            is_leaf=is_leaf
        )
        return new_updates, state

    return base.GradientTransformation(init_fn, update_fn)


# ============================================================================
# Full MuP Muon Optimizer (manual partition to avoid multi_transform masking)
# ============================================================================

class MuPMuonState(NamedTuple):
    """State for the combined MuP Muon + Adam optimizer."""
    count: chex.Array           # step counter
    muon_mu: base.Updates       # Muon momentum buffer
    adam_mu: base.Updates       # Adam first moment
    adam_nu: base.Updates       # Adam second moment


def _compute_muon_mask(params, muon_weight_dimension_numbers_fn):
    """Compute boolean pytree: True for Muon params, False for Adam."""
    from optax.transforms._masking import MaskedNode
    from optax.contrib._muon import MuonDimensionNumbers
    dim_nums = muon_weight_dimension_numbers_fn(params)
    is_leaf = lambda x: (x is None or isinstance(x, MaskedNode)
                         or isinstance(x, MuonDimensionNumbers))
    return jax.tree_util.tree_map(
        lambda dn: not (dn is None or isinstance(dn, MaskedNode)),
        dim_nums, is_leaf=is_leaf,
    )


def _compute_shape_scales(params, muon_weight_dimension_numbers_fn):
    """Pre-compute sqrt(d_out/d_in) shape scales for Muon params."""
    from optax.transforms._masking import MaskedNode
    from optax.contrib._muon import MuonDimensionNumbers
    dim_nums = muon_weight_dimension_numbers_fn(params)
    is_leaf = lambda x: (x is None or isinstance(x, MaskedNode)
                         or isinstance(x, MuonDimensionNumbers))

    def _get_scale(param, dim_num):
        if dim_num is None or isinstance(dim_num, MaskedNode):
            return 1.0
        shape = param.shape
        if hasattr(dim_num, 'reduction_axis'):
            red_axes = dim_num.reduction_axis
            out_axes = dim_num.output_axis
            if isinstance(red_axes, int):
                red_axes = (red_axes,)
            if isinstance(out_axes, int):
                out_axes = (out_axes,)
            d_in = math.prod(shape[a] for a in red_axes)
            d_out = math.prod(shape[a] for a in out_axes)
        else:
            d_in = shape[0]
            d_out = shape[1] if len(shape) > 1 else shape[0]
        return math.sqrt(d_out / d_in)

    return jax.tree_util.tree_map(_get_scale, params, dim_nums, is_leaf=is_leaf)


def mup_muon(
    learning_rate: base.ScalarOrSchedule = 0.02,
    ns_coeffs: Tuple[float, float, float] = (3.4445, -4.775, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.9,
    eps: float = 1e-8,
    weight_decay: float = 0.0001,
    nesterov: bool = True,
    adaptive: bool = False,
    mu_dtype: Optional[chex.ArrayDType] = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.99,
    adam_eps: float = 1e-8,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0001,
    # MuP scale pytrees (all optional, default to no scaling):
    muon_lr_scales: Optional[base.Params] = None,
    muon_eps_scales: Optional[base.Params] = None,
    muon_wd_scales: Optional[base.Params] = None,
    adam_lr_scales: Optional[base.Params] = None,
    adam_eps_scales: Optional[base.Params] = None,
    adam_wd_scales: Optional[base.Params] = None,
    one_minus_beta1_scales: Optional[base.Params] = None,
    one_minus_beta2_scales: Optional[base.Params] = None,
    muon_weight_dimension_numbers: Any = None,
    weight_decay_mask: Optional[Any] = None,
) -> base.GradientTransformation:
    """MuP Muon optimizer with per-parameter CompletedP scaling.

    Manually partitions parameters into Muon (hidden 2D/3D weights) and Adam
    (everything else), applies the correct mu-P scaling rules from Qiu et al.
    (2025) to each group, then merges the updates.

    Args:
        learning_rate: Base learning rate or schedule.
        ns_coeffs: Newton-Schulz polynomial coefficients.
        ns_steps: Newton-Schulz iterations.
        beta: Muon momentum coefficient.
        eps: Base epsilon for Newton-Schulz normalization.
        weight_decay: Muon weight decay.
        nesterov: Use Nesterov momentum for Muon.
        adaptive: Reserved (unused).
        mu_dtype: Momentum buffer dtype.
        adam_b1: Adam beta1.
        adam_b2: Adam beta2.
        adam_eps: Adam epsilon.
        adam_eps_root: Adam epsilon root.
        adam_weight_decay: Adam weight decay.
        muon_lr_scales: Per-param LR scales for Muon params.
        muon_eps_scales: Per-param epsilon scales for Muon Newton-Schulz.
        muon_wd_scales: Per-param weight decay scales for Muon params.
        adam_lr_scales: Per-param LR scales for Adam params.
        adam_eps_scales: Per-param epsilon scales for Adam.
        adam_wd_scales: Per-param weight decay scales for Adam params.
        one_minus_beta1_scales: Per-param (1-beta1) scales for Adam.
        one_minus_beta2_scales: Per-param (1-beta2) scales for Adam.
        muon_weight_dimension_numbers: Function or pytree for Muon/Adam split.
        weight_decay_mask: Optional mask for weight decay.
    """
    if muon_weight_dimension_numbers is None:
        muon_weight_dimension_numbers = _get_make_muon_weight_dimension_numbers()

    ns_coeffs_arr = jnp.array(ns_coeffs, dtype=jnp.float32)
    dim_nums_fn = (
        muon_weight_dimension_numbers if callable(muon_weight_dimension_numbers)
        else lambda p: muon_weight_dimension_numbers
    )

    # is_muon and shape_scales are computed at init and captured in closure
    # They're Python-level constants, NOT jax arrays, so they work with Python if
    _is_muon_cache = {}
    _shape_scales_cache = {}

    def init_fn(params: base.Params) -> MuPMuonState:
        is_muon = _compute_muon_mask(params, dim_nums_fn)
        _is_muon_cache['mask'] = is_muon
        _shape_scales_cache['scales'] = _compute_shape_scales(params, dim_nums_fn)
        muon_mu = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p, dtype=mu_dtype or p.dtype), params
        )
        adam_mu = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        adam_nu = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        return MuPMuonState(
            count=jnp.zeros([], jnp.int32),
            muon_mu=muon_mu,
            adam_mu=adam_mu,
            adam_nu=adam_nu,
        )

    def update_fn(
        updates: base.Updates,
        state: MuPMuonState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, MuPMuonState]:
        if params is None:
            raise ValueError("params must be provided for mup_muon")

        count_inc = numerics.safe_increment(state.count)
        # Static Python-level masks (not traced by JAX)
        is_muon = _is_muon_cache['mask']
        shape_scales = _shape_scales_cache['scales']

        # Get current learning rate from schedule
        lr = learning_rate(count_inc - 1) if callable(learning_rate) else learning_rate

        # --- Muon update for Muon params ---
        # Momentum (is_m is a static Python bool, safe for if)
        new_muon_mu = jax.tree_util.tree_map(
            lambda m, g, is_m: (beta * m + (1 - beta) * g.astype(m.dtype)) if is_m else m,
            state.muon_mu, updates, is_muon,
        )
        if nesterov:
            muon_updates = jax.tree_util.tree_map(
                lambda m, g, is_m: (beta * m + (1 - beta) * g).astype(jnp.float32) if is_m else jnp.zeros_like(g),
                new_muon_mu, updates, is_muon,
            )
        else:
            muon_updates = jax.tree_util.tree_map(
                lambda m, is_m: m.astype(jnp.float32) if is_m else jnp.zeros_like(m),
                new_muon_mu, is_muon,
            )

        # Newton-Schulz + shape scaling for Muon params
        def _muon_ns_and_shape(grad, is_m, shape_s, eps_s):
            if not is_m:
                return jnp.zeros_like(grad)
            effective_eps = eps * eps_s if eps_s is not None else eps
            original_shape = grad.shape
            if grad.ndim == 3:
                grad_2d = (grad.reshape(grad.shape[0], -1) if grad.shape[0] >= grad.shape[-1]
                           else grad.reshape(-1, grad.shape[-1]))
            elif grad.ndim == 2:
                grad_2d = grad
            else:
                return grad
            orth = orthogonalize_via_newton_schulz(grad_2d, ns_coeffs_arr, ns_steps, effective_eps)
            result = orth.reshape(original_shape)
            return result * shape_s

        if muon_eps_scales is not None:
            muon_updates = jax.tree_util.tree_map(
                _muon_ns_and_shape, muon_updates, is_muon, shape_scales, muon_eps_scales,
            )
        else:
            muon_updates = jax.tree_util.tree_map(
                lambda g, is_m, ss: _muon_ns_and_shape(g, is_m, ss, None),
                muon_updates, is_muon, shape_scales,
            )

        # Muon weight decay
        if weight_decay > 0:
            if muon_wd_scales is not None:
                muon_updates = jax.tree_util.tree_map(
                    lambda u, p, is_m, ws: u + weight_decay * ws * p if is_m else u,
                    muon_updates, params, is_muon, muon_wd_scales,
                )
            else:
                muon_updates = jax.tree_util.tree_map(
                    lambda u, p, is_m: u + weight_decay * p if is_m else u,
                    muon_updates, params, is_muon,
                )

        # Muon LR + per-param scaling
        if muon_lr_scales is not None:
            muon_updates = jax.tree_util.tree_map(
                lambda u, s, is_m: -lr * s * u if is_m else u,
                muon_updates, muon_lr_scales, is_muon,
            )
        else:
            muon_updates = jax.tree_util.tree_map(
                lambda u, is_m: -lr * u if is_m else u,
                muon_updates, is_muon,
            )

        # --- Adam update for non-Muon params ---
        def _adam_moment_update(old, new_val, b, scale, is_m):
            if is_m:
                return old
            if scale is not None:
                effective_one_minus_b = (1 - b) * scale
            else:
                effective_one_minus_b = (1 - b)
            return (1 - effective_one_minus_b) * old + effective_one_minus_b * new_val

        # First moment
        if one_minus_beta1_scales is not None:
            new_adam_mu = jax.tree_util.tree_map(
                lambda m, g, s, is_m: _adam_moment_update(m, g, adam_b1, s, is_m),
                state.adam_mu, updates, one_minus_beta1_scales, is_muon,
            )
        else:
            new_adam_mu = jax.tree_util.tree_map(
                lambda m, g, is_m: _adam_moment_update(m, g, adam_b1, None, is_m),
                state.adam_mu, updates, is_muon,
            )

        # Second moment
        if one_minus_beta2_scales is not None:
            new_adam_nu = jax.tree_util.tree_map(
                lambda v, g, s, is_m: _adam_moment_update(v, g**2, adam_b2, s, is_m),
                state.adam_nu, updates, one_minus_beta2_scales, is_muon,
            )
        else:
            new_adam_nu = jax.tree_util.tree_map(
                lambda v, g, is_m: _adam_moment_update(v, g**2, adam_b2, None, is_m),
                state.adam_nu, updates, is_muon,
            )

        # Bias correction + Adam update
        bc_mu = jax.tree_util.tree_map(
            lambda m, is_m: m / (1 - adam_b1 ** count_inc) if not is_m else m,
            new_adam_mu, is_muon,
        )
        bc_nu = jax.tree_util.tree_map(
            lambda v, is_m: v / (1 - adam_b2 ** count_inc) if not is_m else v,
            new_adam_nu, is_muon,
        )

        def _adam_update(mu_hat, nu_hat, is_m, eps_s):
            if is_m:
                return jnp.zeros_like(mu_hat)
            effective_eps = adam_eps * eps_s if eps_s is not None else adam_eps
            return mu_hat / (jnp.sqrt(nu_hat) + effective_eps)

        if adam_eps_scales is not None:
            adam_updates = jax.tree_util.tree_map(
                _adam_update, bc_mu, bc_nu, is_muon, adam_eps_scales,
            )
        else:
            adam_updates = jax.tree_util.tree_map(
                lambda m, v, is_m: _adam_update(m, v, is_m, None),
                bc_mu, bc_nu, is_muon,
            )

        # Adam weight decay
        if adam_weight_decay > 0:
            if adam_wd_scales is not None:
                adam_updates = jax.tree_util.tree_map(
                    lambda u, p, is_m, ws: u + adam_weight_decay * ws * p if not is_m else u,
                    adam_updates, params, is_muon, adam_wd_scales,
                )
            else:
                adam_updates = jax.tree_util.tree_map(
                    lambda u, p, is_m: u + adam_weight_decay * p if not is_m else u,
                    adam_updates, params, is_muon,
                )

        # Adam LR + per-param scaling
        if adam_lr_scales is not None:
            adam_updates = jax.tree_util.tree_map(
                lambda u, s, is_m: -lr * s * u if not is_m else u,
                adam_updates, adam_lr_scales, is_muon,
            )
        else:
            adam_updates = jax.tree_util.tree_map(
                lambda u, is_m: -lr * u if not is_m else u,
                adam_updates, is_muon,
            )

        # --- Merge ---
        final_updates = jax.tree_util.tree_map(
            lambda mu_u, ad_u, is_m: mu_u if is_m else ad_u,
            muon_updates, adam_updates, is_muon,
        )

        new_state = MuPMuonState(
            count=count_inc,
            muon_mu=new_muon_mu,
            adam_mu=new_adam_mu,
            adam_nu=new_adam_nu,
        )
        return final_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)
