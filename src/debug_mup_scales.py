#!/usr/bin/env python3
"""
Debug script to trace MuP scaling factors computation.

This script creates the same parameterization as the training run and traces
through the exact computation of each scaling factor to identify why scales
are not close to 1.0.

Usage:
    python src/debug_mup_scales.py
"""

import math
import sys
import pprint

# Add src to path
sys.path.insert(0, '/mnt/raid0/l2o_install/scaling_l2o/src')

import jax
import jax.numpy as jnp

from parameterization import (
    CompletedPParameterization, 
    ModuleType, 
    TensorType,
    TENSOR_TYPE_TO_MODULE_TYPE,
    get_module_type,
)


def debug_parameterization_creation():
    """
    Trace through the exact creation of the parameterization with the config
    values from complete_p.py and the command line arguments.
    """
    print("=" * 80)
    print("DEBUGGING PARAMETERIZATION CREATION")
    print("=" * 80)
    
    # From config/parameterization/complete_p.py
    base_width = 128
    base_batch_size = 32 * 4 * 128  # = 16384
    base_dataset_size = 128 * 128 * 1000  # = 16384000
    base_depth = 4.0
    alpha = 1.0
    depth_multipliers = [4.0, 4.0, 4.0, 4.0]
    
    # From command line: --local_batch_size 32 --gradient_accumulation_steps 4 --num_inner_steps 1000
    # From task: mutransformer-dense-w128-d4-h4_fineweb-s128-gpt2 (width=128, depth=4)
    gradient_accumulation_steps = 4
    local_batch_size = 32
    num_inner_steps = 1000
    
    current_width = 128
    current_depth = 4
    current_batch_size = gradient_accumulation_steps * local_batch_size  # = 128
    current_dataset_size = gradient_accumulation_steps * local_batch_size * num_inner_steps  # = 128000
    
    print("\n--- CONFIG VALUES ---")
    print(f"base_width: {base_width}")
    print(f"base_depth: {base_depth}")
    print(f"base_batch_size: {base_batch_size}")
    print(f"base_dataset_size: {base_dataset_size}")
    print(f"alpha: {alpha}")
    print(f"depth_multipliers: {depth_multipliers}")
    
    print("\n--- CURRENT VALUES (from command line) ---")
    print(f"current_width: {current_width}")
    print(f"current_depth: {current_depth}")
    print(f"current_batch_size: {current_batch_size} (= {gradient_accumulation_steps} * {local_batch_size})")
    print(f"current_dataset_size: {current_dataset_size} (= {gradient_accumulation_steps} * {local_batch_size} * {num_inner_steps})")
    
    print("\n--- SCALING RATIOS ---")
    m_N = current_width / base_width
    m_L = current_depth  # NOTE: This is NOT current_depth / base_depth!
    m_B = current_batch_size / base_batch_size
    m_D = current_dataset_size / base_dataset_size
    
    print(f"m_N = current_width / base_width = {current_width} / {base_width} = {m_N}")
    print(f"m_L = current_depth = {current_depth}  (NOTE: NOT current_depth / base_depth!)")
    print(f"m_B = current_batch_size / base_batch_size = {current_batch_size} / {base_batch_size} = {m_B}")
    print(f"m_D = current_dataset_size / base_dataset_size = {current_dataset_size} / {base_dataset_size} = {m_D}")
    
    print("\n--- DERIVED SCALING FACTORS ---")
    m_L_alpha_minus_1 = m_L ** (alpha - 1)
    m_L_neg_alpha = m_L ** (-alpha)
    sde_lr_wd_scale = math.sqrt(m_B / m_D)
    sde_eps_scale = math.sqrt(m_D / m_B)
    
    print(f"m_L^(alpha-1) = {m_L}^({alpha}-1) = {m_L}^{alpha-1} = {m_L_alpha_minus_1}")
    print(f"m_L^(-alpha) = {m_L}^(-{alpha}) = {m_L_neg_alpha}")
    print(f"sqrt(m_B / m_D) = sqrt({m_B} / {m_D}) = sqrt({m_B / m_D}) = {sde_lr_wd_scale}")
    print(f"sqrt(m_D / m_B) = sqrt({m_D} / {m_B}) = sqrt({m_D / m_B}) = {sde_eps_scale}")
    print(f"m_B / m_D = {m_B / m_D}  (for beta scaling)")
    
    return {
        'base_width': base_width,
        'base_depth': base_depth,
        'base_batch_size': base_batch_size,
        'base_dataset_size': base_dataset_size,
        'current_width': current_width,
        'current_depth': current_depth,
        'current_batch_size': current_batch_size,
        'current_dataset_size': current_dataset_size,
        'alpha': alpha,
        'depth_multipliers': depth_multipliers,
        'm_N': m_N,
        'm_L': m_L,
        'm_B': m_B,
        'm_D': m_D,
        'm_L_alpha_minus_1': m_L_alpha_minus_1,
        'm_L_neg_alpha': m_L_neg_alpha,
        'sde_lr_wd_scale': sde_lr_wd_scale,
        'sde_eps_scale': sde_eps_scale,
    }


def debug_depth_multiplier(config):
    """Debug the depth multiplier computation."""
    print("\n" + "=" * 80)
    print("DEBUGGING DEPTH MULTIPLIER COMPUTATION")
    print("=" * 80)
    
    depth_multipliers = config['depth_multipliers']
    base_depth = config['base_depth']
    current_depth = config['current_depth']
    
    print(f"\ndepth_multipliers: {depth_multipliers}")
    print(f"base_depth: {base_depth}")
    print(f"current_depth: {current_depth}")
    
    print("\n--- DEPTH MULTIPLIER FOR EACH LAYER ---")
    for layer_idx in range(current_depth):
        # Interpolate depth multiplier
        if current_depth == base_depth:
            interpolated = depth_multipliers[layer_idx]
        else:
            # Interpolation would happen here, but for matching depths:
            interpolated = depth_multipliers[layer_idx]
        
        depth_mult = interpolated / base_depth
        print(f"Layer {layer_idx}: interpolated = {interpolated}, depth_mult = {interpolated} / {base_depth} = {depth_mult}")
    
    # Non-layer parameters (embed, output_proj, out_ln)
    print(f"\nNon-layer params (layer_idx=None): depth_mult = 1.0")


def debug_lr_scale_computation(config):
    """Debug LR scale computation for different module types."""
    print("\n" + "=" * 80)
    print("DEBUGGING LR SCALE COMPUTATION")
    print("=" * 80)
    
    base_width = config['base_width']
    m_L_alpha_minus_1 = config['m_L_alpha_minus_1']
    sde_lr_wd_scale = config['sde_lr_wd_scale']
    depth_multipliers = config['depth_multipliers']
    base_depth = config['base_depth']
    
    # Example tensors with their fan_in values
    test_cases = [
        ("embed/embedding", ModuleType.INPUT_EMBED, 50257, None),  # vocab size
        ("output_proj/kernel", ModuleType.UNEMBED_WEIGHT, 128, None),  # embed dim
        ("out_ln/scale", ModuleType.UNEMBED_NORM, 128, None),
        ("blocks_0/CausalAttn_0/query/kernel", ModuleType.HIDDEN_WEIGHT, 128, 0),
        ("blocks_0/CausalAttn_0/attn_out_proj/kernel", ModuleType.HIDDEN_WEIGHT, 512, 0),  # H * Dh
        ("blocks_0/MlpSwiGLU_0/Dense_0/kernel", ModuleType.HIDDEN_WEIGHT, 128, 0),  # up proj
        ("blocks_0/MlpSwiGLU_0/Dense_1/kernel", ModuleType.HIDDEN_WEIGHT, 128, 0),  # gate proj
        ("blocks_0/MlpSwiGLU_0/Dense_2/kernel", ModuleType.HIDDEN_WEIGHT, 512, 0),  # down proj (fan_in = ffn_dim)
        ("blocks_0/RMSNorm_0/scale", ModuleType.HIDDEN_NORM, 128, 0),
        ("blocks_3/MlpSwiGLU_0/Dense_1/kernel", ModuleType.HIDDEN_WEIGHT, 128, 3),  # gate at layer 3
    ]
    
    print("\nLR Scale Formula by ModuleType:")
    print("  INPUT_EMBED:   1.0 * sqrt(m_B/m_D)")
    print("  HIDDEN_WEIGHT: (base_width/fan_in) * m_L^(alpha-1) * sqrt(m_B/m_D)")
    print("  HIDDEN_NORM:   m_L^(alpha-1) * sqrt(m_B/m_D)")
    print("  UNEMBED_WEIGHT: (base_width/fan_in) * sqrt(m_B/m_D)")
    print("  UNEMBED_NORM:  1.0 * sqrt(m_B/m_D)")
    print("\nThen depth_mult is applied: scale *= (interpolated_depth_mult / base_depth)")
    
    print(f"\n{'Param Path':<50} {'ModuleType':<20} {'fan_in':<8} {'layer':<6} {'base_scale':<12} {'depth_mult':<12} {'final_scale':<12}")
    print("-" * 130)
    
    for path, module_type, fan_in, layer_idx in test_cases:
        # Compute base scale
        if module_type == ModuleType.INPUT_EMBED:
            base_scale = 1.0 * sde_lr_wd_scale
        elif module_type == ModuleType.HIDDEN_WEIGHT:
            width_scale = base_width / fan_in
            base_scale = width_scale * m_L_alpha_minus_1 * sde_lr_wd_scale
        elif module_type == ModuleType.HIDDEN_NORM:
            base_scale = m_L_alpha_minus_1 * sde_lr_wd_scale
        elif module_type == ModuleType.UNEMBED_WEIGHT:
            width_scale = base_width / fan_in
            base_scale = width_scale * sde_lr_wd_scale
        elif module_type == ModuleType.UNEMBED_NORM:
            base_scale = 1.0 * sde_lr_wd_scale
        else:
            base_scale = 1.0
        
        # Compute depth multiplier
        if layer_idx is None:
            depth_mult = 1.0
        else:
            interpolated = depth_multipliers[layer_idx]
            depth_mult = interpolated / base_depth
        
        final_scale = base_scale * depth_mult
        
        print(f"{path:<50} {module_type.value:<20} {fan_in:<8} {str(layer_idx):<6} {base_scale:<12.6f} {depth_mult:<12.6f} {final_scale:<12.6f}")


def debug_beta_scale_computation(config):
    """Debug beta scale computation."""
    print("\n" + "=" * 80)
    print("DEBUGGING BETA SCALE COMPUTATION")
    print("=" * 80)
    
    m_B = config['m_B']
    m_D = config['m_D']
    depth_multipliers = config['depth_multipliers']
    base_depth = config['base_depth']
    
    base_scale = m_B / m_D
    
    print(f"\nBase (1-beta1) scale = m_B / m_D = {m_B} / {m_D} = {base_scale}")
    print(f"Base (1-beta2) scale = m_B / m_D = {m_B} / {m_D} = {base_scale}")
    
    print("\n--- BETA SCALE WITH DEPTH MULTIPLIER ---")
    for layer_idx in range(4):
        interpolated = depth_multipliers[layer_idx]
        depth_mult = interpolated / base_depth
        final_scale = base_scale * depth_mult
        print(f"Layer {layer_idx}: base={base_scale:.6f} * depth_mult={depth_mult:.6f} = {final_scale:.6f}")
    
    print(f"\nNon-layer (layer_idx=None): scale = {base_scale:.6f}")
    
    # Calculate effective beta values
    b1 = 0.9
    b2 = 0.99
    
    print("\n--- EFFECTIVE BETA VALUES (with base scale = 1.0) ---")
    print(f"Effective beta1 = 1 - (1-{b1}) * {base_scale} = 1 - {(1-b1) * base_scale} = {1 - (1-b1) * base_scale}")
    print(f"Effective beta2 = 1 - (1-{b2}) * {base_scale} = 1 - {(1-b2) * base_scale} = {1 - (1-b2) * base_scale}")


def debug_eps_scale_computation(config):
    """Debug epsilon scale computation."""
    print("\n" + "=" * 80)
    print("DEBUGGING EPSILON SCALE COMPUTATION")
    print("=" * 80)
    
    base_width = config['base_width']
    m_L_neg_alpha = config['m_L_neg_alpha']
    sde_eps_scale = config['sde_eps_scale']
    depth_multipliers = config['depth_multipliers']
    base_depth = config['base_depth']
    
    print("\nEpsilon Scale Formula by ModuleType:")
    print("  INPUT_EMBED:   (base_width/fan_in) * sqrt(m_D/m_B)")
    print("  HIDDEN_WEIGHT: (base_width/fan_in) * m_L^(-alpha) * sqrt(m_D/m_B)")
    print("  HIDDEN_NORM:   (base_width/fan_in) * m_L^(-alpha) * sqrt(m_D/m_B)")
    print("  QK_NORM:       m_L^(-alpha) * sqrt(m_D/m_B)  (no width scaling)")
    print("  UNEMBED_*:     1.0 * sqrt(m_D/m_B)")
    
    # Test cases
    test_cases = [
        ("embed/embedding", ModuleType.INPUT_EMBED, 128, None),  # note: fan_in should be embed_dim for lookup
        ("blocks_0/MlpSwiGLU_0/Dense_1/kernel", ModuleType.HIDDEN_WEIGHT, 128, 0),
        ("blocks_0/RMSNorm_0/scale", ModuleType.HIDDEN_NORM, 128, 0),
        ("output_proj/kernel", ModuleType.UNEMBED_WEIGHT, 128, None),
    ]
    
    print(f"\n{'Param Path':<50} {'ModuleType':<20} {'fan_in':<8} {'layer':<6} {'base_scale':<12} {'depth_mult':<12} {'final_scale':<12}")
    print("-" * 130)
    
    for path, module_type, fan_in, layer_idx in test_cases:
        # Compute base scale
        if module_type == ModuleType.INPUT_EMBED:
            width_scale = base_width / fan_in
            base_scale = width_scale * sde_eps_scale
        elif module_type == ModuleType.HIDDEN_WEIGHT:
            width_scale = base_width / fan_in
            base_scale = width_scale * m_L_neg_alpha * sde_eps_scale
        elif module_type == ModuleType.HIDDEN_NORM:
            width_scale = base_width / fan_in
            base_scale = width_scale * m_L_neg_alpha * sde_eps_scale
        elif module_type in [ModuleType.UNEMBED_WEIGHT, ModuleType.UNEMBED_NORM]:
            base_scale = 1.0 * sde_eps_scale
        else:
            base_scale = 1.0
        
        # Compute depth multiplier
        if layer_idx is None:
            depth_mult = 1.0
        else:
            interpolated = depth_multipliers[layer_idx]
            depth_mult = interpolated / base_depth
        
        final_scale = base_scale * depth_mult
        
        print(f"{path:<50} {module_type.value:<20} {fan_in:<8} {str(layer_idx):<6} {base_scale:<12.6f} {depth_mult:<12.6f} {final_scale:<12.6f}")


def debug_wd_scale_computation(config):
    """Debug weight decay scale computation."""
    print("\n" + "=" * 80)
    print("DEBUGGING WEIGHT DECAY SCALE COMPUTATION")
    print("=" * 80)
    
    base_width = config['base_width']
    sde_lr_wd_scale = config['sde_lr_wd_scale']
    depth_multipliers = config['depth_multipliers']
    base_depth = config['base_depth']
    
    print("\nWeight Decay Scale Formula by ModuleType:")
    print("  HIDDEN_WEIGHT: (fan_in/base_width) * sqrt(m_B/m_D)")
    print("  UNEMBED_WEIGHT: (fan_in/base_width) * sqrt(m_B/m_D)")
    print("  All others:    1.0 * sqrt(m_B/m_D)")
    
    test_cases = [
        ("embed/embedding", ModuleType.INPUT_EMBED, 50257, None),
        ("blocks_0/MlpSwiGLU_0/Dense_1/kernel", ModuleType.HIDDEN_WEIGHT, 128, 0),
        ("blocks_0/MlpSwiGLU_0/Dense_2/kernel", ModuleType.HIDDEN_WEIGHT, 512, 0),  # down proj (fan_in = ffn_dim)
        ("output_proj/kernel", ModuleType.UNEMBED_WEIGHT, 128, None),
    ]
    
    print(f"\n{'Param Path':<50} {'ModuleType':<20} {'fan_in':<8} {'layer':<6} {'base_scale':<12} {'depth_mult':<12} {'final_scale':<12}")
    print("-" * 130)
    
    for path, module_type, fan_in, layer_idx in test_cases:
        # Compute base scale
        if module_type == ModuleType.HIDDEN_WEIGHT:
            width_scale = fan_in / base_width
            base_scale = width_scale * sde_lr_wd_scale
        elif module_type == ModuleType.UNEMBED_WEIGHT:
            width_scale = fan_in / base_width
            base_scale = width_scale * sde_lr_wd_scale
        else:
            base_scale = 1.0 * sde_lr_wd_scale
        
        # Compute depth multiplier
        if layer_idx is None:
            depth_mult = 1.0
        else:
            interpolated = depth_multipliers[layer_idx]
            depth_mult = interpolated / base_depth
        
        final_scale = base_scale * depth_mult
        
        print(f"{path:<50} {module_type.value:<20} {fan_in:<8} {str(layer_idx):<6} {base_scale:<12.6f} {depth_mult:<12.6f} {final_scale:<12.6f}")


def create_and_test_parameterization(config):
    """Create the actual parameterization object and test its outputs."""
    print("\n" + "=" * 80)
    print("CREATING ACTUAL PARAMETERIZATION AND TESTING")
    print("=" * 80)
    
    # Create per-tensor multipliers (all 1.0)
    per_tensor_lr_multipliers = {
        "attention_query": 1.0, "attention_key": 1.0, "attention_value": 1.0,
        "attention_output": 1.0, "attention_query_norm": 1.0, "attention_key_norm": 1.0,
        "mlp_up": 1.0, "mlp_gate": 1.0, "mlp_down": 1.0,
        "post_attention_norm": 1.0, "post_mlp_norm": 1.0,
        "embedding": 1.0, "output_norm": 1.0, "unembedding": 1.0,
    }
    
    param = CompletedPParameterization(
        base_width=config['base_width'],
        base_depth=int(config['base_depth']),
        base_batch_size=config['base_batch_size'],
        base_dataset_size=config['base_dataset_size'],
        current_width=config['current_width'],
        current_depth=config['current_depth'],
        current_batch_size=config['current_batch_size'],
        current_dataset_size=config['current_dataset_size'],
        alpha=config['alpha'],
        depth_multipliers=config['depth_multipliers'],
        per_tensor_lr_multipliers=per_tensor_lr_multipliers,
        per_tensor_eps_multipliers=per_tensor_lr_multipliers.copy(),
        per_tensor_wd_multipliers=per_tensor_lr_multipliers.copy(),
        per_tensor_init_multipliers=per_tensor_lr_multipliers.copy(),
        per_tensor_beta1_multipliers=per_tensor_lr_multipliers.copy(),
        per_tensor_beta2_multipliers=per_tensor_lr_multipliers.copy(),
    )
    
    print("\nParameterization internal values:")
    print(f"  m_N: {param.m_N}")
    print(f"  m_L: {param.m_L}")
    print(f"  m_B: {param.m_B}")
    print(f"  m_D: {param.m_D}")
    print(f"  _m_L_alpha_minus_1: {param._m_L_alpha_minus_1}")
    print(f"  _m_L_neg_alpha: {param._m_L_neg_alpha}")
    print(f"  _sde_lr_wd_scale: {param._sde_lr_wd_scale}")
    print(f"  _sde_eps_scale: {param._sde_eps_scale}")
    
    # Get multipliers
    multipliers = param.get_multipliers()
    print("\nForward pass multipliers:")
    for k, v in multipliers.items():
        print(f"  {k}: {float(v)}")
    
    # Get beta rescaling
    beta_rescaling = param.get_beta_rescaling()
    print("\nBeta rescaling:")
    for k, v in beta_rescaling.items():
        print(f"  {k}: {v}")
    
    return param


def identify_issues(config):
    """Identify potential issues with the configuration."""
    print("\n" + "=" * 80)
    print("POTENTIAL ISSUES IDENTIFIED")
    print("=" * 80)
    
    issues = []
    
    # Check batch size ratio
    if config['current_batch_size'] != config['base_batch_size']:
        ratio = config['base_batch_size'] / config['current_batch_size']
        issues.append(f"Batch size mismatch: base={config['base_batch_size']}, current={config['current_batch_size']}, ratio={ratio:.2f}x")
    
    # Check dataset size ratio
    if config['current_dataset_size'] != config['base_dataset_size']:
        ratio = config['base_dataset_size'] / config['current_dataset_size']
        issues.append(f"Dataset size mismatch: base={config['base_dataset_size']}, current={config['current_dataset_size']}, ratio={ratio:.2f}x")
    
    # Check if m_B/m_D != 1.0
    ratio = config['m_B'] / config['m_D']
    if abs(ratio - 1.0) > 0.001:
        issues.append(f"m_B/m_D = {ratio:.6f} != 1.0, this affects all SDE scaling factors")
    
    # Check depth multipliers
    for i, dm in enumerate(config['depth_multipliers']):
        expected = config['base_depth']  # For scale=1.0, depth_multipliers should equal base_depth
        if abs(dm - expected) > 0.001:
            issues.append(f"depth_multipliers[{i}] = {dm}, expected {expected} for scale=1.0")
    
    if not issues:
        print("\n✓ No major issues found with the configuration.")
        print("  All scales should be close to 1.0 for matching base and current configs.")
    else:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    print("\n--- RECOMMENDATIONS ---")
    print("\nTo get scales ≈ 1.0, set:")
    print(f"  base_batch_size = {config['current_batch_size']}  (currently {config['base_batch_size']})")
    print(f"  base_dataset_size = {config['current_dataset_size']}  (currently {config['base_dataset_size']})")
    print(f"  Or adjust current values to match base values")


def check_actual_output_values():
    """
    Compare computed values against the actual output from the training run.
    """
    print("\n" + "=" * 80)
    print("COMPARING AGAINST ACTUAL OUTPUT")
    print("=" * 80)
    
    # From the user's output:
    actual_output = {
        "params/blocks_3/MlpSwiGLU_0/Dense_1/kernel": {
            "lr_scale": 11.313708,
            "b1_scale": 128.0,
            "b2_scale": 128.0,
            "eps_scale": 0.022097,
            "wd_scale": 11.313708,
        },
        "params/embed/embedding": {
            "lr_scale": 11.313708,
            "b1_scale": 128.0,
            "b2_scale": 128.0,
            "eps_scale": 0.000225,
            "wd_scale": 11.313708,
        },
    }
    
    print("\nActual output from training run:")
    for path, scales in actual_output.items():
        print(f"\n{path}:")
        for k, v in scales.items():
            print(f"  {k}: {v}")
    
    print("\n--- ANALYSIS ---")
    print("\n11.313708 ≈ sqrt(128) = 11.3137...")
    print("128 = base_batch_size / current_batch_size = 16384 / 128 = 128")
    print("0.022097 ≈ 1/sqrt(128)/sqrt(128) * (1/m_L) = 1/128/4 = 0.00195 (close)")
    
    print("\nThis suggests the scales are NOT using m_B/m_D but rather")
    print("using base_batch_size/current_batch_size directly somewhere.")
    print("\nLet me check the actual parameterization code...")


def detailed_scale_trace():
    """
    Trace through exactly what the parameterization computes.
    """
    print("\n" + "=" * 80)
    print("DETAILED SCALE TRACE")
    print("=" * 80)
    
    # Values from config
    base_width = 128
    base_depth = 4
    base_batch_size = 32 * 4 * 128  # 16384
    base_dataset_size = 128 * 128 * 1000  # 16384000
    
    current_width = 128
    current_depth = 4
    current_batch_size = 32 * 4  # 128
    current_dataset_size = 32 * 4 * 1000  # 128000
    
    depth_multipliers = [4.0, 4.0, 4.0, 4.0]
    alpha = 1.0
    
    # Scaling ratios (as computed in Parameterization.__init__)
    m_N = current_width / base_width  # 1.0
    m_L = current_depth  # 4 (NOT current_depth / base_depth!)
    m_B = current_batch_size / base_batch_size  # 128 / 16384 = 0.0078125
    m_D = current_dataset_size / base_dataset_size  # 128000 / 16384000 = 0.0078125
    
    print("\nScaling ratios (CORRECT - what should be computed):")
    print(f"  m_N = {m_N}")
    print(f"  m_L = {m_L}")
    print(f"  m_B = {m_B}")
    print(f"  m_D = {m_D}")
    
    # Derived values (as computed in CompletedPParameterization.__init__)
    _m_L_alpha_minus_1 = m_L ** (alpha - 1)  # 4^0 = 1.0
    _m_L_neg_alpha = m_L ** (-alpha)  # 4^-1 = 0.25
    _sde_lr_wd_scale = math.sqrt(m_B / m_D)  # sqrt(1) = 1.0
    _sde_eps_scale = math.sqrt(m_D / m_B)  # sqrt(1) = 1.0
    
    print("\nDerived values (CORRECT):")
    print(f"  _m_L_alpha_minus_1 = {m_L}^({alpha}-1) = {_m_L_alpha_minus_1}")
    print(f"  _m_L_neg_alpha = {m_L}^(-{alpha}) = {_m_L_neg_alpha}")
    print(f"  _sde_lr_wd_scale = sqrt({m_B}/{m_D}) = {_sde_lr_wd_scale}")
    print(f"  _sde_eps_scale = sqrt({m_D}/{m_B}) = {_sde_eps_scale}")
    
    print("\n" + "=" * 80)
    print("BUG IDENTIFIED IN benchmark.py")
    print("=" * 80)
    
    print("""
BUG FOUND on line 276 of benchmark.py:

    current_batch_size=args.gradient_accumulation_steps * args.local_batch_size * 128,
                                                                                  ^^^
                                                              EXTRA * 128 MULTIPLIER!

This causes:
    current_batch_size = 4 * 32 * 128 = 16384 (equals base_batch_size!)
    current_dataset_size = 4 * 32 * 1000 = 128000 (no extra multiplier)

Which gives:
    m_B = 16384 / 16384 = 1.0
    m_D = 128000 / 16384000 = 0.0078125
    m_B / m_D = 1.0 / 0.0078125 = 128  ← This is the 128x scale!
    sqrt(m_B / m_D) = sqrt(128) = 11.3137  ← This is the 11.31x scale!

FIX: Remove the extra * 128 from line 276:
    current_batch_size=args.gradient_accumulation_steps * args.local_batch_size,
""")


def compute_correct_config_for_scale_1():
    """
    Compute what config values would give scales ≈ 1.0.
    """
    print("\n" + "=" * 80)
    print("CORRECTED CONFIG FOR SCALES ≈ 1.0")
    print("=" * 80)
    
    # Current training setup
    gradient_accumulation_steps = 4
    local_batch_size = 32
    num_inner_steps = 1000
    
    current_width = 128
    current_depth = 4
    current_batch_size = gradient_accumulation_steps * local_batch_size  # 128
    current_dataset_size = gradient_accumulation_steps * local_batch_size * num_inner_steps  # 128000
    
    print("\nFor scales ≈ 1.0, update config/parameterization/complete_p.py:")
    print()
    print("parameterization_args = dict(")
    print(f"    base_width={current_width},")
    print(f"    base_batch_size={current_batch_size},  # was 32 * 4 * 128 = 16384")
    print(f"    base_dataset_size={current_dataset_size},  # was 128 * 128 * 1000 = 16384000")
    print(f"    alpha=1.0,")
    print(f"    base_depth={current_depth},  # was 4.0")
    print(f"    depth_multipliers=[{current_depth}.0] * {current_depth},  # [4.0, 4.0, 4.0, 4.0]")
    print("    # ... rest of config unchanged")
    print(")")
    
    print("\nAlternatively, if you want to transfer HPs from a larger model,")
    print("keep the current config and understand that the scales represent")
    print("the transfer factors from base -> current configuration.")


def show_expected_scales_after_fix():
    """
    Show what scales should look like after fixing the bug.
    """
    print("\n" + "=" * 80)
    print("EXPECTED SCALES AFTER FIX")
    print("=" * 80)
    
    print("""
After removing the extra * 128 from line 276 of benchmark.py:

For your training setup:
  - current_batch_size = 4 * 32 = 128
  - current_dataset_size = 4 * 32 * 1000 = 128000
  - base_batch_size = 16384
  - base_dataset_size = 16384000

Scaling ratios:
  m_B = 128 / 16384 = 0.0078125
  m_D = 128000 / 16384000 = 0.0078125
  m_B / m_D = 1.0
  sqrt(m_B / m_D) = 1.0

Expected scales for most parameters:
  - lr_scale: ~1.0 (for layers with fan_in = base_width)
  - eps_scale: ~0.25 (for hidden layers, due to m_L^(-alpha) = 4^(-1))
  - wd_scale: ~1.0 (for layers with fan_in = base_width)
  - beta1_scale: ~1.0
  - beta2_scale: ~1.0

For layers with different fan_in (e.g., MLP down proj with fan_in=512):
  - lr_scale: base_width/fan_in = 128/512 = 0.25
  - wd_scale: fan_in/base_width = 512/128 = 4.0

These are the CORRECT CompletedP scaling factors for hyperparameter transfer!
""")


if __name__ == "__main__":
    # Run all debug functions
    config = debug_parameterization_creation()
    debug_depth_multiplier(config)
    debug_lr_scale_computation(config)
    debug_beta_scale_computation(config)
    debug_eps_scale_computation(config)
    debug_wd_scale_computation(config)
    
    param = create_and_test_parameterization(config)
    
    check_actual_output_values()
    detailed_scale_trace()
    
    identify_issues(config)
    compute_correct_config_for_scale_1()
    show_expected_scales_after_fix()
