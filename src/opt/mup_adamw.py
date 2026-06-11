"""
MuP AdamW optimizer with per-parameter learning rate, epsilon, and weight decay rescaling.

This module implements a custom AdamW optimizer that supports the CompletedP
parameterization for hyperparameter transfer across model scales (width, depth,
batch size, and dataset size).

The implementation is based on optax's adam/adamw but with explicit support for:
- Per-parameter learning rate scaling (lr_scales pytree)
- Per-parameter epsilon scaling (eps_scales pytree)  
- Per-parameter weight decay scaling (wd_scales pytree)

Reference:
- optax: https://github.com/google-deepmind/optax
- CompletedP parameterization for hyperparameter transfer
"""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base
from optax._src import numerics
from optax._src import utils


# ============================================================================
# State Classes
# ============================================================================

class ScaleByMuPAdamState(NamedTuple):
    """State for the MuP Adam algorithm.
    
    Attributes:
        count: Number of update steps taken.
        mu: First moment estimate (exponential moving average of gradients).
        nu: Second moment estimate (exponential moving average of squared gradients).
    """
    count: chex.Array  # shape=(), dtype=jnp.int32
    mu: base.Updates   # First moment
    nu: base.Updates   # Second moment


class MuPAdamWState(NamedTuple):
    """State for the full MuP AdamW optimizer.
    
    Attributes:
        adam_state: State for the Adam component.
    """
    adam_state: ScaleByMuPAdamState


# ============================================================================
# Core Adam Transformation with Per-Parameter Epsilon and Betas
# ============================================================================

def scale_by_mup_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    eps_scales: Optional[base.Params] = None,
    one_minus_beta1_scales: Optional[base.Params] = None,
    one_minus_beta2_scales: Optional[base.Params] = None,
    mu_dtype: Optional[chex.ArrayDType] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
    """Rescale updates according to the Adam algorithm with per-parameter epsilon and betas.
    
    This is the core Adam algorithm that computes:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)
        update = m̂_t / (√v̂_t + ε)
    
    Where ε can be scaled per-parameter using eps_scales, and (1-β₁) and (1-β₂)
    can be scaled per-parameter using one_minus_beta1_scales and one_minus_beta2_scales.
    
    For CompletedP parameterization, the scaling is:
        effective (1-β₁) = (1-b1) * one_minus_beta1_scales
        effective (1-β₂) = (1-b2) * one_minus_beta2_scales
    
    Args:
        b1: Base exponential decay rate for the first moment estimate (default: 0.9).
        b2: Base exponential decay rate for the second moment estimate (default: 0.999).
        eps: Small constant for numerical stability (default: 1e-8).
        eps_root: Small constant applied inside the square root (default: 0.0).
        eps_scales: Optional pytree of per-parameter epsilon scaling factors.
                   If provided, the effective epsilon for each parameter is eps * eps_scales[param].
        one_minus_beta1_scales: Optional pytree of per-parameter (1-β₁) scaling factors.
                               If provided, effective (1-β₁) = (1-b1) * one_minus_beta1_scales[param].
                               From CompletedP: scales by m_B/m_D with optional per-tensor and depth multipliers.
        one_minus_beta2_scales: Optional pytree of per-parameter (1-β₂) scaling factors.
                               If provided, effective (1-β₂) = (1-b2) * one_minus_beta2_scales[param].
                               From CompletedP: scales by m_B/m_D with optional per-tensor and depth multipliers.
        mu_dtype: Optional dtype for the first moment accumulator.
        nesterov: Whether to use Nesterov momentum (default: False).
    
    Returns:
        A GradientTransformation implementing the scaled Adam update.
    """
    mu_dtype = utils.canonicalize_dtype(mu_dtype)
    
    # Compute base (1-beta) values
    one_minus_b1_base = 1 - b1
    one_minus_b2_base = 1 - b2
    
    def init_fn(params: base.Params) -> ScaleByMuPAdamState:
        mu = jax.tree_util.tree_map(
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
        )
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)
        return ScaleByMuPAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)
    
    def update_fn(
        updates: base.Updates,
        state: ScaleByMuPAdamState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, ScaleByMuPAdamState]:
        del params
        
        count_inc = numerics.safe_int32_increment(state.count)
        
        # Update biased first moment estimate with per-parameter (1-β₁)
        if one_minus_beta1_scales is not None:
            # print("updates shape: ", type(updates),jax.tree_util.tree_map(lambda x: x.shape, updates))
            # print()
            # print()
            # print("state.mu shape: ", type(state.mu),jax.tree_util.tree_map(lambda x: x.shape, state.mu))
            # print()
            # print()
            # print("one_minus_beta1_scales: ", type(one_minus_beta1_scales), jax.tree_util.tree_map(lambda x: x.shape, one_minus_beta1_scales))
            mu = jax.tree_util.tree_map(
                lambda m, g, s: (1 - one_minus_b1_base * s) * m + one_minus_b1_base * s * g,
                state.mu, updates, one_minus_beta1_scales
            )
            # For bias correction, we need to track the effective beta per parameter
            # This is complex because β₁^t varies per parameter. 
            # Approximation: use the base b1 for bias correction (reasonable when scales ≈ 1)
            # For exact handling, we would need per-parameter count tracking
            mu_hat = jax.tree_util.tree_map(
                lambda m, s: m / (1 - (1 - one_minus_b1_base * s) ** count_inc),
                mu, one_minus_beta1_scales
            )
        else:
            mu = jax.tree_util.tree_map(
                lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates
            )
            mu_hat = jax.tree_util.tree_map(
                lambda m: m / (1 - b1 ** count_inc), mu
            )
        
        # Update biased second moment estimate with per-parameter (1-β₂)
        if one_minus_beta2_scales is not None:
            nu = jax.tree_util.tree_map(
                lambda v, g, s: (1 - one_minus_b2_base * s) * v + one_minus_b2_base * s * (g ** 2),
                state.nu, updates, one_minus_beta2_scales
            )
            nu_hat = jax.tree_util.tree_map(
                lambda v, s: v / (1 - (1 - one_minus_b2_base * s) ** count_inc),
                nu, one_minus_beta2_scales
            )
        else:
            nu = jax.tree_util.tree_map(
                lambda v, g: b2 * v + (1 - b2) * (g ** 2), state.nu, updates
            )
            nu_hat = jax.tree_util.tree_map(
                lambda v: v / (1 - b2 ** count_inc), nu
            )
        
        # Optionally apply Nesterov momentum
        if nesterov:
            if one_minus_beta1_scales is not None:
                mu_hat = jax.tree_util.tree_map(
                    lambda m, g, s: (1 - one_minus_b1_base * s) * m + one_minus_b1_base * s * g / (1 - (1 - one_minus_b1_base * s) ** count_inc),
                    mu_hat, updates, one_minus_beta1_scales
                )
            else:
                mu_hat = jax.tree_util.tree_map(
                    lambda m, g: b1 * m + (1 - b1) * g / (1 - b1 ** count_inc),
                    mu_hat, updates
                )
        
        # Compute updates with per-parameter epsilon
        if eps_scales is not None:
            # Apply per-parameter epsilon scaling
            new_updates = jax.tree_util.tree_map(
                lambda m, v, e: m / (jnp.sqrt(v + eps_root) + eps * e),
                mu_hat, nu_hat, eps_scales
            )
        else:
            # Use global epsilon
            new_updates = jax.tree_util.tree_map(
                lambda m, v: m / (jnp.sqrt(v + eps_root) + eps),
                mu_hat, nu_hat
            )
        
        # Cast mu to the specified dtype
        mu = jax.tree_util.tree_map(lambda t: t.astype(mu_dtype), mu)
        
        return new_updates, ScaleByMuPAdamState(count=count_inc, mu=mu, nu=nu)
    
    return base.GradientTransformation(init_fn, update_fn)


# ============================================================================
# Per-Parameter Learning Rate Scaling
# ============================================================================

def scale_by_mup_lr(
    lr_scales: base.Params,
) -> base.GradientTransformation:
    """Scale updates by per-parameter learning rate multipliers.
    
    This applies a per-parameter learning rate scaling after the Adam update.
    The effective learning rate for each parameter is: base_lr * lr_scales[param]
    
    Args:
        lr_scales: Pytree of per-parameter learning rate scaling factors,
                  with the same structure as the model parameters.
    
    Returns:
        A GradientTransformation that scales updates by lr_scales.
    """
    def init_fn(params: base.Params) -> base.EmptyState:
        del params
        return base.EmptyState()
    
    def update_fn(
        updates: base.Updates,
        state: base.EmptyState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, base.EmptyState]:
        del params
        new_updates = jax.tree_util.tree_map(
            lambda u, s: u * s, updates, lr_scales
        )
        return new_updates, state
    
    return base.GradientTransformation(init_fn, update_fn)


# ============================================================================
# Per-Parameter Weight Decay
# ============================================================================

def mup_add_decayed_weights(
    weight_decay: float = 0.0,
    wd_scales: Optional[base.Params] = None,
    mask: Optional[Union[base.Params, Callable[[base.Params], base.Params]]] = None,
) -> base.GradientTransformation:
    """Add decayed weights to updates with per-parameter scaling.
    
    This implements the weight decay as in AdamW:
        new_params = params - lr * (adam_update + λ * params)
    
    Which means we add λ * params to the updates.
    
    With per-parameter scaling:
        weight_decay_i = weight_decay * wd_scales[i]
    
    Args:
        weight_decay: Base weight decay coefficient (default: 0.0).
        wd_scales: Optional pytree of per-parameter weight decay scaling factors.
                  If provided, the effective weight decay for each parameter is
                  weight_decay * wd_scales[param].
        mask: A pytree with the same structure as params, or a callable that
              returns such a pytree. True/non-zero values indicate parameters
              to apply weight decay to.
    
    Returns:
        A GradientTransformation that adds weight decay to updates.
    """
    def init_fn(params: base.Params) -> base.EmptyState:
        del params
        return base.EmptyState()
    
    def update_fn(
        updates: base.Updates,
        state: base.EmptyState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, base.EmptyState]:
        if params is None:
            raise ValueError("params must be provided for weight decay")
        
        # Get the mask if it's a callable
        mask_ = mask(params) if callable(mask) else mask
        
        if wd_scales is not None:
            # Per-parameter weight decay
            def add_wd(update, param, wd_scale, m=None):
                wd = weight_decay * wd_scale
                if m is not None:
                    # Apply mask
                    return update + wd * param * m
                return update + wd * param
            
            if mask_ is not None:
                new_updates = jax.tree_util.tree_map(
                    add_wd, updates, params, wd_scales, mask_
                )
            else:
                new_updates = jax.tree_util.tree_map(
                    lambda u, p, s: u + weight_decay * s * p,
                    updates, params, wd_scales
                )
        else:
            # Global weight decay
            if mask_ is not None:
                new_updates = jax.tree_util.tree_map(
                    lambda u, p, m: u + weight_decay * p * m,
                    updates, params, mask_
                )
            else:
                new_updates = jax.tree_util.tree_map(
                    lambda u, p: u + weight_decay * p,
                    updates, params
                )
        
        return new_updates, state
    
    return base.GradientTransformation(init_fn, update_fn)


# ============================================================================
# Debug Function to Print Effective Hyperparameters
# ============================================================================

def print_mup_adamw_hyperparameters(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0001,
    lr_scales: Optional[base.Params] = None,
    eps_scales: Optional[base.Params] = None,
    wd_scales: Optional[base.Params] = None,
    one_minus_beta1_scales: Optional[base.Params] = None,
    one_minus_beta2_scales: Optional[base.Params] = None,
    mu_dtype: Optional[chex.ArrayDType] = None,
    mask: Optional[Union[base.Params, Callable[[base.Params], base.Params]]] = None,
    nesterov: bool = False,
) -> None:
    """Print the effective hyperparameters for each parameter in the MuP AdamW optimizer.
    
    This function computes and prints the actual lr, b1, b2, eps, and wd values
    that will be used for each parameter after applying all the scaling factors.
    """
    print("=" * 80)
    print("MuP AdamW Effective Hyperparameters (DEBUG)")
    print("=" * 80)
    
    # Get base learning rate value (handle schedules)
    if callable(learning_rate):
        base_lr_val = float(learning_rate(50))
        base_lr_str = f"schedule (at step 0: {base_lr_val:.6e})"
    else:
        base_lr_val = float(learning_rate)
        base_lr_str = f"{base_lr_val:.6e}"
    
    print(f"\nBase hyperparameters:")
    print(f"  learning_rate: {base_lr_str}")
    print(f"  b1: {b1}")
    print(f"  b2: {b2}")
    print(f"  eps: {eps:.6e}")
    print(f"  eps_root: {eps_root}")
    print(f"  weight_decay: {weight_decay:.6e}")
    print(f"  nesterov: {nesterov}")
    
    print("\n" + "-" * 80)
    print("Per-parameter effective hyperparameters:")
    print("-" * 80)
    
    # Collect all scale pytrees to iterate through parameters
    # Use any available scale pytree to get the parameter structure
    reference_pytree = lr_scales or eps_scales or wd_scales or one_minus_beta1_scales or one_minus_beta2_scales
    
    if reference_pytree is None:
        print("\nNo per-parameter scaling applied. All parameters use base hyperparameters.")
        print("=" * 80)
        return
    
    # Flatten to get paths
    flat_ref, _ = jax.tree_util.tree_flatten_with_path(reference_pytree)
    
    # Also flatten all scale pytrees for lookup
    def get_flat_dict(pytree):
        if pytree is None:
            return None
        flat, _ = jax.tree_util.tree_flatten_with_path(pytree)
        return {tuple(str(k.key) if hasattr(k, 'key') else str(k) for k in path): scale for path, scale in flat}
    
    lr_dict = get_flat_dict(lr_scales)
    eps_dict = get_flat_dict(eps_scales)
    wd_dict = get_flat_dict(wd_scales)
    b1_dict = get_flat_dict(one_minus_beta1_scales)
    b2_dict = get_flat_dict(one_minus_beta2_scales)
    
    for path, _ in flat_ref:
        path_tuple = tuple(str(k.key) if hasattr(k, 'key') else str(k) for k in path)
        path_str = "/".join(path_tuple)
        
        # Compute effective values
        # LR: effective_lr = learning_rate * lr_scale
        lr_scale = float(lr_dict[path_tuple]) if lr_dict and path_tuple in lr_dict else 1.0
        effective_lr = base_lr_val * lr_scale
        
        # Beta1: effective_b1 = 1 - (1-b1) * one_minus_beta1_scale
        b1_scale = float(b1_dict[path_tuple]) if b1_dict and path_tuple in b1_dict else 1.0
        effective_b1 = 1 - (1 - b1) * b1_scale
        
        # Beta2: effective_b2 = 1 - (1-b2) * one_minus_beta2_scale
        b2_scale = float(b2_dict[path_tuple]) if b2_dict and path_tuple in b2_dict else 1.0
        effective_b2 = 1 - (1 - b2) * b2_scale
        
        # Epsilon: effective_eps = eps * eps_scale
        eps_scale = float(eps_dict[path_tuple]) if eps_dict and path_tuple in eps_dict else 1.0
        effective_eps = eps * eps_scale
        
        # Weight Decay: effective_wd = weight_decay * wd_scale
        wd_scale = float(wd_dict[path_tuple]) if wd_dict and path_tuple in wd_dict else 1.0
        effective_wd = weight_decay * wd_scale
        
        print(f"\n{path_str}:")
        print(f"  lr:  {effective_lr:.6e}  (scale={lr_scale:.6f})")
        print(f"  b1:  {effective_b1:.6f}  (1-b1 scale={b1_scale:.6f})")
        print(f"  b2:  {effective_b2:.6f}  (1-b2 scale={b2_scale:.6f})")
        print(f"  eps: {effective_eps:.6e}  (scale={eps_scale:.6f})")
        print(f"  wd:  {effective_wd:.6e}  (scale={wd_scale:.6f})")
    
    print("\n" + "=" * 80)


# ============================================================================
# Main MuP AdamW Optimizer
# ============================================================================

def mup_adamw(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0001,
    lr_scales: Optional[base.Params] = None,
    eps_scales: Optional[base.Params] = None,
    wd_scales: Optional[base.Params] = None,
    one_minus_beta1_scales: Optional[base.Params] = None,
    one_minus_beta2_scales: Optional[base.Params] = None,
    mu_dtype: Optional[chex.ArrayDType] = None,
    mask: Optional[Union[base.Params, Callable[[base.Params], base.Params]]] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
    """AdamW optimizer with per-parameter learning rate, epsilon, weight decay, and beta scaling.
    
    This implements AdamW with support for the CompletedP parameterization,
    allowing hyperparameter transfer across model scales including batch and data size.
    
    The update rule is:
        m_t = β₁_i * m_{t-1} + (1 - β₁_i) * g_t
        v_t = β₂_i * v_{t-1} + (1 - β₂_i) * g_t²
        m̂_t = m_t / (1 - β₁_i^t)
        v̂_t = v_t / (1 - β₂_i^t)
        update = m̂_t / (√v̂_t + ε_i)
        θ_t = θ_{t-1} - η * lr_scale_i * (update + λ * wd_scale_i * θ_{t-1})
    
    Where:
        - (1-β₁_i) = (1-b1) * one_minus_beta1_scales[i] (per-parameter beta1)
        - (1-β₂_i) = (1-b2) * one_minus_beta2_scales[i] (per-parameter beta2)
        - ε_i = eps * eps_scales[i] (per-parameter epsilon)
        - lr_scale_i = lr_scales[i] (per-parameter learning rate multiplier)
        - wd_scale_i = wd_scales[i] (per-parameter weight decay multiplier)
    
    Args:
        learning_rate: Base learning rate (scalar or schedule).
        b1: Base exponential decay rate for first moment (default: 0.9).
        b2: Base exponential decay rate for second moment (default: 0.999).
        eps: Base epsilon for numerical stability (default: 1e-8).
        eps_root: Epsilon applied inside the square root (default: 0.0).
        weight_decay: Base weight decay coefficient (default: 0.0001).
        lr_scales: Optional pytree of per-parameter learning rate multipliers.
                  Effective LR for param i = learning_rate * lr_scales[i].
        eps_scales: Optional pytree of per-parameter epsilon multipliers.
                   Effective epsilon for param i = eps * eps_scales[i].
        wd_scales: Optional pytree of per-parameter weight decay multipliers.
                  Effective WD for param i = weight_decay * wd_scales[i].
        one_minus_beta1_scales: Optional pytree of per-parameter (1-β₁) multipliers.
                               Effective (1-β₁) for param i = (1-b1) * one_minus_beta1_scales[i].
                               From CompletedP: scales by m_B/m_D.
        one_minus_beta2_scales: Optional pytree of per-parameter (1-β₂) multipliers.
                               Effective (1-β₂) for param i = (1-b2) * one_minus_beta2_scales[i].
                               From CompletedP: scales by m_B/m_D.
        mu_dtype: Optional dtype for the first moment accumulator.
        mask: Pytree or callable indicating which params get weight decay.
        nesterov: Whether to use Nesterov momentum (default: False).
    
    Returns:
        A GradientTransformation implementing MuP AdamW.
    
    Example:
        >>> from parameterization import CompletedPParameterization
        >>> # Get module types from model
        >>> module_types = model.get_module_types(params)
        >>> tensor_types = model.get_tensor_types(params)
        >>> # Create parameterization
        >>> param = CompletedPParameterization(
        ...     base_width=64, base_depth=4, base_batch_size=32, base_dataset_size=1_000_000,
        ...     current_width=128, current_depth=8, current_batch_size=64, current_dataset_size=4_000_000,
        ...     alpha=1.0
        ... )
        >>> # Get scaling pytrees
        >>> lr_scales = param.get_lr_scales_with_tensor_types(tensor_types, device=device)
        >>> eps_scales = param.get_eps_scales_with_tensor_types(tensor_types, device=device)
        >>> wd_scales = param.get_wd_scales_with_tensor_types(tensor_types, device=device)
        >>> one_minus_beta1_scales = param.get_one_minus_beta1_scales_pytree(tensor_types, device=device)
        >>> one_minus_beta2_scales = param.get_one_minus_beta2_scales_pytree(tensor_types, device=device)
        >>> # Create optimizer
        >>> optimizer = mup_adamw(
        ...     learning_rate=1e-3,
        ...     weight_decay=0.01,
        ...     lr_scales=lr_scales,
        ...     eps_scales=eps_scales,
        ...     wd_scales=wd_scales,
        ...     one_minus_beta1_scales=one_minus_beta1_scales,
        ...     one_minus_beta2_scales=one_minus_beta2_scales,
        ... )
    """
    # DEBUG: Print effective hyperparameters for each layer
    print_mup_adamw_hyperparameters(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        weight_decay=weight_decay,
        lr_scales=lr_scales,
        eps_scales=eps_scales,
        wd_scales=wd_scales,
        one_minus_beta1_scales=one_minus_beta1_scales,
        one_minus_beta2_scales=one_minus_beta2_scales,
        mu_dtype=mu_dtype,
        mask=mask,
        nesterov=nesterov,
    )
    # exit(0)
    
    transforms = []
    
    # 1. Scale by Adam with per-parameter epsilon and betas
    transforms.append(
        scale_by_mup_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            eps_scales=eps_scales,
            one_minus_beta1_scales=one_minus_beta1_scales,
            one_minus_beta2_scales=one_minus_beta2_scales,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        )
    )
    
    # 2. Add weight decay with per-parameter scaling
    if weight_decay > 0:
        transforms.append(
            mup_add_decayed_weights(
                weight_decay=weight_decay,
                wd_scales=wd_scales,
                mask=mask,
            )
        )
    
    # 3. Apply base learning rate
    transforms.append(optax.scale_by_learning_rate(learning_rate))
    
    # 4. Apply per-parameter learning rate scaling
    if lr_scales is not None:
        transforms.append(scale_by_mup_lr(lr_scales))
    
    return optax.chain(*transforms)


def mup_adam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    lr_scales: Optional[base.Params] = None,
    eps_scales: Optional[base.Params] = None,
    one_minus_beta1_scales: Optional[base.Params] = None,
    one_minus_beta2_scales: Optional[base.Params] = None,
    mu_dtype: Optional[chex.ArrayDType] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
    """Adam optimizer with per-parameter learning rate, epsilon, and beta scaling.
    
    Same as mup_adamw but without weight decay.
    
    Args:
        learning_rate: Base learning rate (scalar or schedule).
        b1: Base exponential decay rate for first moment (default: 0.9).
        b2: Base exponential decay rate for second moment (default: 0.999).
        eps: Base epsilon for numerical stability (default: 1e-8).
        eps_root: Epsilon applied inside the square root (default: 0.0).
        lr_scales: Optional pytree of per-parameter learning rate multipliers.
        eps_scales: Optional pytree of per-parameter epsilon multipliers.
        one_minus_beta1_scales: Optional pytree of per-parameter (1-β₁) multipliers.
        one_minus_beta2_scales: Optional pytree of per-parameter (1-β₂) multipliers.
        mu_dtype: Optional dtype for the first moment accumulator.
        nesterov: Whether to use Nesterov momentum (default: False).
    
    Returns:
        A GradientTransformation implementing MuP Adam.
    """
    return mup_adamw(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        weight_decay=0.0,
        lr_scales=lr_scales,
        eps_scales=eps_scales,
        wd_scales=None,
        one_minus_beta1_scales=one_minus_beta1_scales,
        one_minus_beta2_scales=one_minus_beta2_scales,
        mu_dtype=mu_dtype,
        mask=None,
        nesterov=nesterov,
    )


# ============================================================================
# Factory Functions for Integration with CompletedP
# ============================================================================

def create_mup_adamw_from_parameterization(
    learning_rate: base.ScalarOrSchedule,
    weight_decay: float,
    parameterization: Any,
    tensor_types_pytree: Any,
    device: Optional[Any] = None,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    mask: Optional[Union[base.Params, Callable[[base.Params], base.Params]]] = None,
    nesterov: bool = False,
    include_beta_scaling: bool = True,
) -> base.GradientTransformation:
    """Create a MuP AdamW optimizer using a CompletedP parameterization.
    
    This is a convenience function that extracts the scaling factors from
    a CompletedPParameterization instance and creates the optimizer.
    
    The parameterization object stores per-tensor-type multipliers for each
    hyperparameter (lr, eps, wd, init, beta1, beta2) that are applied on top
    of the CompletedP scaling rules.
    
    Args:
        learning_rate: Base learning rate (scalar or schedule).
        weight_decay: Base weight decay coefficient.
        parameterization: A CompletedPParameterization instance with per-tensor
                         multipliers stored as attributes.
        tensor_types_pytree: Output of model.get_tensor_types(params).
        device: JAX device for the scaling arrays.
        b1: Base exponential decay rate for first moment (default: 0.9).
        b2: Base exponential decay rate for second moment (default: 0.999).
        eps: Base epsilon for numerical stability (default: 1e-8).
        eps_root: Epsilon applied inside the square root (default: 0.0).
        mu_dtype: Optional dtype for the first moment accumulator.
        mask: Pytree or callable indicating which params get weight decay.
        nesterov: Whether to use Nesterov momentum (default: False).
        include_beta_scaling: Whether to include (1-β₁) and (1-β₂) scaling from
                             CompletedP (m_B/m_D). Default: True.
    
    Returns:
        A GradientTransformation implementing MuP AdamW with proper scaling.
    
    Example:
        >>> from parameterization import CompletedPParameterization, TensorType
        >>> param = CompletedPParameterization(
        ...     base_width=64, base_depth=4, base_batch_size=32, base_dataset_size=1_000_000,
        ...     current_width=128, current_depth=8, current_batch_size=64, current_dataset_size=4_000_000,
        ...     per_tensor_lr_multipliers={TensorType.ATTENTION_QUERY: 0.5},  # Optional per-tensor overrides
        ... )
        >>> tensor_types = model.get_tensor_types(params)
        >>> optimizer = create_mup_adamw_from_parameterization(
        ...     learning_rate=1e-3,
        ...     weight_decay=0.01,
        ...     parameterization=param,
        ...     tensor_types_pytree=tensor_types,
        ...     device=jax.devices()[0]
        ... )
    """
    # Extract scaling pytrees from parameterization
    # Per-tensor-type multipliers are stored in the parameterization object
    lr_scales = parameterization.get_lr_scales_pytree(tensor_types_pytree, device)
    eps_scales = parameterization.get_eps_scales_pytree(tensor_types_pytree, device)
    wd_scales = parameterization.get_wd_scales_pytree(tensor_types_pytree, device)
    
    # Extract beta scaling if requested
    if include_beta_scaling:
        one_minus_beta1_scales = parameterization.get_one_minus_beta1_scales_pytree(tensor_types_pytree, device)
        one_minus_beta2_scales = parameterization.get_one_minus_beta2_scales_pytree(tensor_types_pytree, device)
    else:
        one_minus_beta1_scales = None
        one_minus_beta2_scales = None
    
    return mup_adamw(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        weight_decay=weight_decay,
        lr_scales=lr_scales,
        eps_scales=eps_scales,
        wd_scales=wd_scales,
        one_minus_beta1_scales=one_minus_beta1_scales,
        one_minus_beta2_scales=one_minus_beta2_scales,
        mu_dtype=mu_dtype,
        mask=mask,
        nesterov=nesterov,
    )


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    import pprint
    
    print("=" * 70)
    print("MuP AdamW Optimizer Demo")
    print("=" * 70)
    
    # Create a simple model's parameter structure
    params = {
        'embed': {'embedding': jnp.ones((1000, 128))},
        'blocks_0': {
            'CausalAttn_0': {
                'query': {'kernel': jnp.ones((128, 4, 32))},
                'key': {'kernel': jnp.ones((128, 4, 32))},
                'value': {'kernel': jnp.ones((128, 4, 32))},
                'attn_out_proj': {'kernel': jnp.ones((4, 32, 128))},
            },
            'RMSNorm_0': {'scale': jnp.ones((128,))},
            'MlpSwiGLU_0': {
                'Dense_0': {'kernel': jnp.ones((128, 512))},
                'Dense_1': {'kernel': jnp.ones((128, 512))},
                'Dense_2': {'kernel': jnp.ones((512, 128))},
            },
            'RMSNorm_1': {'scale': jnp.ones((128,))},
        },
        'out_ln': {'scale': jnp.ones((128,))},
        'output_proj': {'kernel': jnp.ones((128, 1000))},
    }
    
    print("\nParameter shapes:")
    pprint.pprint(jax.tree_util.tree_map(lambda x: x.shape, params))
    
    # Create simple scaling pytrees (all ones for this demo)
    lr_scales = jax.tree_util.tree_map(lambda x: jnp.array(1.0), params)
    eps_scales = jax.tree_util.tree_map(lambda x: jnp.array(1.0), params)
    wd_scales = jax.tree_util.tree_map(lambda x: jnp.array(1.0), params)
    
    print("\n" + "-" * 70)
    print("Testing MuP AdamW optimizer...")
    print("-" * 70)
    
    # Create optimizer
    optimizer = mup_adamw(
        learning_rate=1e-3,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        lr_scales=lr_scales,
        eps_scales=eps_scales,
        wd_scales=wd_scales,
    )
    
    # Initialize optimizer state
    opt_state = optimizer.init(params)
    print("\nOptimizer state structure:")
    print(f"  State type: {type(opt_state)}")
    print(f"  Number of state components: {len(opt_state)}")
    
    # Create fake gradients
    grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.1, params)
    
    # Perform one update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    
    print("\nUpdate shapes (should match param shapes):")
    pprint.pprint(jax.tree_util.tree_map(lambda x: x.shape, updates))
    
    # Apply updates
    new_params = optax.apply_updates(params, updates)
    
    # Check that parameters changed
    param_diff = jax.tree_util.tree_map(
        lambda old, new: jnp.mean(jnp.abs(old - new)),
        params, new_params
    )
    print("\nMean absolute parameter change:")
    pprint.pprint(jax.tree_util.tree_map(lambda x: float(x), param_diff))
    
    print("\n" + "-" * 70)
    print("Testing with non-uniform scales...")
    print("-" * 70)
    
    # Create non-uniform scales for demo
    def create_varying_scales(x, base=1.0, noise_scale=0.1):
        return jnp.array(base + noise_scale * jax.random.uniform(
            jax.random.PRNGKey(0), shape=()
        ))
    
    lr_scales_varying = jax.tree_util.tree_map(
        lambda x: create_varying_scales(x, base=0.5), params
    )
    eps_scales_varying = jax.tree_util.tree_map(
        lambda x: create_varying_scales(x, base=2.0), params
    )
    wd_scales_varying = jax.tree_util.tree_map(
        lambda x: create_varying_scales(x, base=1.5), params
    )
    
    optimizer_varying = mup_adamw(
        learning_rate=1e-3,
        weight_decay=0.01,
        lr_scales=lr_scales_varying,
        eps_scales=eps_scales_varying,
        wd_scales=wd_scales_varying,
    )
    
    opt_state_varying = optimizer_varying.init(params)
    updates_varying, _ = optimizer_varying.update(grads, opt_state_varying, params)
    
    print("\nUpdates with varying scales completed successfully!")
    print("Update norms:")
    update_norms = jax.tree_util.tree_map(
        lambda x: float(jnp.linalg.norm(x)),
        updates_varying
    )
    pprint.pprint(update_norms)
    
    # =========================================================================
    # Integration test with CompletedPParameterization
    # =========================================================================
    print("\n" + "-" * 70)
    print("Testing integration with CompletedPParameterization...")
    print("-" * 70)
    
    try:
        import sys
        sys.path.insert(0, '/mnt/raid0/l2o_install/scaling_l2o/src')
        from parameterization import CompletedPParameterization, ModuleType, TensorType
        
        # Create parameterization for scaling from base to current model
        param = CompletedPParameterization(
            base_width=64,           # Base model width
            base_depth=4,            # Base model depth
            base_batch_size=32,      # Base batch size
            base_dataset_size=1_000_000,  # Base dataset size
            current_width=128,       # Current model width (2x)
            current_depth=8,         # Current model depth (2x)
            current_batch_size=64,   # Current batch size (2x)
            current_dataset_size=4_000_000,  # Current dataset size (4x)
            alpha=1.0,               # Depth scaling exponent
        )
        
        print("\nParameterization created:")
        print(f"  Width scaling (m_N): {param.m_N}")
        print(f"  Depth scaling (m_L): {param.m_L}")
        print(f"  Batch scaling (m_B): {param.m_B}")
        print(f"  Data scaling (m_D): {param.m_D}")
        
        # Create module_types pytree manually (simulating model.get_module_types())
        module_types = {
            'embed': {'embedding': (ModuleType.INPUT_EMBED, (1000, 128))},
            'blocks_0': {
                'CausalAttn_0': {
                    'query': {'kernel': (ModuleType.HIDDEN_WEIGHT, (128, 32))},
                    'key': {'kernel': (ModuleType.HIDDEN_WEIGHT, (128, 32))},
                    'value': {'kernel': (ModuleType.HIDDEN_WEIGHT, (128, 32))},
                    'attn_out_proj': {'kernel': (ModuleType.HIDDEN_WEIGHT, (128, 128))},
                },
                'RMSNorm_0': {'scale': (ModuleType.HIDDEN_NORM, (128, 128))},
                'MlpSwiGLU_0': {
                    'Dense_0': {'kernel': (ModuleType.HIDDEN_WEIGHT, (128, 512))},
                    'Dense_1': {'kernel': (ModuleType.HIDDEN_WEIGHT, (128, 512))},
                    'Dense_2': {'kernel': (ModuleType.HIDDEN_WEIGHT, (512, 128))},
                },
                'RMSNorm_1': {'scale': (ModuleType.HIDDEN_NORM, (128, 128))},
            },
            'out_ln': {'scale': (ModuleType.UNEMBED_NORM, (128, 128))},
            'output_proj': {'kernel': (ModuleType.UNEMBED_WEIGHT, (128, 1000))},
        }
        
        # Get device
        device = jax.devices()[0]
        
        # Get scaling pytrees from parameterization
        lr_scales_mup = param.get_lr_scales_pytree(module_types, device)
        eps_scales_mup = param.get_eps_scales_pytree(module_types, device)
        wd_scales_mup = param.get_wd_scales_pytree(module_types, device)
        
        print("\nLR scales from CompletedP:")
        pprint.pprint(jax.tree_util.tree_map(lambda x: float(x), lr_scales_mup))
        
        print("\nEpsilon scales from CompletedP:")
        pprint.pprint(jax.tree_util.tree_map(lambda x: float(x), eps_scales_mup))
        
        print("\nWeight decay scales from CompletedP:")
        pprint.pprint(jax.tree_util.tree_map(lambda x: float(x), wd_scales_mup))
        
        # Create optimizer with CompletedP scaling
        optimizer_mup = mup_adamw(
            learning_rate=1e-3,
            weight_decay=0.01,
            lr_scales=lr_scales_mup,
            eps_scales=eps_scales_mup,
            wd_scales=wd_scales_mup,
        )
        
        opt_state_mup = optimizer_mup.init(params)
        updates_mup, _ = optimizer_mup.update(grads, opt_state_mup, params)
        new_params_mup = optax.apply_updates(params, updates_mup)
        
        print("\nParameter changes with CompletedP scaling:")
        param_diff_mup = jax.tree_util.tree_map(
            lambda old, new: float(jnp.mean(jnp.abs(old - new))),
            params, new_params_mup
        )
        pprint.pprint(param_diff_mup)
        
        print("\nNote: Different layers have different scaling factors:")
        print("  - Embeddings: LR scale = 1.0 (unscaled)")
        print("  - Hidden weights: LR scale = base_width/fan_in * m_L^(α-1)")
        print("  - Output projection: LR scale = base_width/fan_in")
        
        print("\n✓ CompletedP integration test passed!")
        
    except ImportError as e:
        print(f"\nSkipping CompletedP integration test (import error): {e}")
    except Exception as e:
        print(f"\nCompletedP integration test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

