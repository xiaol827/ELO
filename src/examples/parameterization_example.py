"""
Example: Using CompletedP Parameterization with a Real Transformer

This script demonstrates the full workflow of:
1. Creating a transformer model using TransformerDo
2. Using get_module_types() to classify all parameters
3. Using CompletedPParameterization to compute LR, epsilon, and weight decay rescalings

Run with the l2o venv from the scaling_l2o directory:
    cd /mnt/raid0/l2o_install/scaling_l2o
    /mnt/raid0/l2o_install/.venv/bin/python src/examples/parameterization_example.py
"""

import sys
import os

# Add the src directory to path so we can import directly
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# Also add the project root for src.* imports
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import pprint

# Direct imports from the existing files
from src.custom_tasks.mu_transformer_moe import TransformerDo, DoConfig, MoeConfig
from src.parameterization import (
    CompletedPParameterization, 
    ModuleType, 
    TensorType,
    TENSOR_TYPE_TO_MODULE_TYPE,
    get_module_type,
)


def flatten_module_types(module_types_pytree):
    """
    Flatten the nested module types pytree into a flat dictionary.
    
    Args:
        module_types_pytree: Nested dict with (ModuleType, (fan_in, fan_out)) leaves
        
    Returns:
        Flat dict mapping path strings to (ModuleType, (fan_in, fan_out))
    """
    flat_info = {}
    
    def flatten(d, prefix=''):
        if isinstance(d, dict):
            for k, v in d.items():
                new_prefix = f"{prefix}/{k}" if prefix else k
                flatten(v, new_prefix)
        elif isinstance(d, tuple) and len(d) == 2 and isinstance(d[0], ModuleType):
            # This is a leaf: (ModuleType, (fan_in, fan_out))
            flat_info[prefix] = d
        else:
            # Recurse if it's some other structure
            try:
                for i, item in enumerate(d):
                    flatten(item, f"{prefix}[{i}]")
            except TypeError:
                pass
    
    flatten(module_types_pytree)
    return flat_info


def flatten_tensor_types(tensor_types_pytree):
    """
    Flatten the nested tensor types pytree into a flat dictionary.
    
    Args:
        tensor_types_pytree: Nested dict with (TensorType, (fan_in, fan_out)) leaves
        
    Returns:
        Flat dict mapping path strings to (TensorType, (fan_in, fan_out))
    """
    flat_info = {}
    
    def flatten(d, prefix=''):
        if isinstance(d, dict):
            for k, v in d.items():
                new_prefix = f"{prefix}/{k}" if prefix else k
                flatten(v, new_prefix)
        elif isinstance(d, tuple) and len(d) == 2 and isinstance(d[0], TensorType):
            # This is a leaf: (TensorType, (fan_in, fan_out))
            flat_info[prefix] = d
        else:
            # Recurse if it's some other structure
            try:
                for i, item in enumerate(d):
                    flatten(item, f"{prefix}[{i}]")
            except TypeError:
                pass
    
    flatten(tensor_types_pytree)
    return flat_info


def main():
    print("=" * 80)
    print("CompletedP Parameterization Example with Real Transformer")
    print("=" * 80)
    
    # =========================================================================
    # Step 1: Create a transformer model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Creating Transformer Model")
    print("=" * 80)
    
    # Model configuration
    model_dim = 128      # Embedding dimension
    num_heads = 4        # Number of attention heads
    num_layers = 2       # Number of transformer blocks
    ffn_dim = 512        # MLP hidden dimension (4x expansion)
    vocab_size = 1000    # Vocabulary size
    seq_length = 64      # Sequence length
    
    # Base configuration (for HP transfer reference)
    base_width = 64      # Base width for scaling
    base_depth = 2       # Base depth
    base_batch_size = 32
    base_dataset_size = 1_000_000
    
    # Current configuration (what we're training)
    current_batch_size = 64      # 2x batch
    current_dataset_size = 4_000_000  # 4x data
    
    config = DoConfig(
        D=model_dim,
        H=num_heads,
        L=seq_length,
        N=num_layers,
        V=vocab_size,
        F=ffn_dim,
        use_qk_norm=True,
        ffn_type='swiglu',
        moe_config=MoeConfig(),
        use_mup=True,
        completep_alpha=1.0,
        mup_base_width=base_width,
        use_rope=True,
    )
    
    print(f"\nModel Configuration:")
    print(f"  Embedding dim (D): {model_dim}")
    print(f"  Num heads (H): {num_heads}")
    print(f"  Num layers (N): {num_layers}")
    print(f"  FFN dim (F): {ffn_dim}")
    print(f"  Vocab size (V): {vocab_size}")
    print(f"  MuP base width: {base_width}")
    
    # Create model and initialize parameters
    model = TransformerDo(docfg=config)
    rng = jax.random.PRNGKey(42)
    input_tokens = jax.random.randint(rng, shape=(2, seq_length), minval=0, maxval=vocab_size)
    params = model.init(rng, input_tokens)
    
    print(f"\nModel initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    
    # =========================================================================
    # Step 2: Get module types using the new method
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Classifying Parameters with get_module_types()")
    print("=" * 80)
    
    module_types = model.get_module_types(params)
    
    print("\nModule Types (nested structure):")
    pprint.pprint(module_types, width=100)
    
    # Flatten for use with parameterization
    flat_module_types = flatten_module_types(module_types['params'])
    
    print(f"\nFlattened module types ({len(flat_module_types)} parameters):")
    for path, (mod_type, (fan_in, fan_out)) in flat_module_types.items():
        print(f"  {path}")
        print(f"      Type: {mod_type.value}, fan_in: {fan_in}, fan_out: {fan_out}")
    
    # =========================================================================
    # Step 3: Create CompletedP parameterization
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Creating CompletedP Parameterization")
    print("=" * 80)
    
    parameterization = CompletedPParameterization(
        base_width=base_width,
        base_depth=base_depth,
        base_batch_size=base_batch_size,
        base_dataset_size=base_dataset_size,
        current_width=model_dim,  # Using model_dim as current width
        current_depth=num_layers,
        current_batch_size=current_batch_size,
        current_dataset_size=current_dataset_size,
        alpha=1.0,
    )
    
    print(f"\nParameterization Config:")
    print(f"  Base: width={base_width}, depth={base_depth}, batch={base_batch_size}, data={base_dataset_size}")
    print(f"  Current: width={model_dim}, depth={num_layers}, batch={current_batch_size}, data={current_dataset_size}")
    print(f"  Alpha: {parameterization.alpha}")
    
    print(f"\nScaling Ratios:")
    print(f"  m_L (depth ratio): {parameterization.m_L}")
    print(f"  m_B (batch ratio): {parameterization.m_B}")
    print(f"  m_D (data ratio): {parameterization.m_D}")
    
    print(f"\nForward Pass Multipliers:")
    mults = parameterization.get_multipliers()
    for k, v in mults.items():
        print(f"  {k}: {v}")
    
    # =========================================================================
    # Step 4: Get LR, Epsilon, and Weight Decay rescalings
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Computing LR, Epsilon, and Weight Decay Rescalings")
    print("=" * 80)
    
    device = jax.devices()[0]
    
    # Get rescalings
    lr_scales = parameterization.get_lr_rescaling(flat_module_types, device=device)
    eps_scales = parameterization.get_eps_rescaling(flat_module_types, device=device)
    wd_scales = parameterization.get_wd_rescaling(flat_module_types, device=device)


    pprint.pprint(lr_scales, width=100)
    # exit(0)
    
    print("\n" + "-" * 80)
    print("Learning Rate Rescaling (smaller for wider layers):")
    print("-" * 80)
    for path, scale in lr_scales.items():
        mod_type, (fan_in, fan_out) = flat_module_types[path]
        print(f"  {path}")
        print(f"      {mod_type.value}: fan_in={fan_in} -> LR scale={float(scale):.6f}")
    
    print("\n" + "-" * 80)
    print("Epsilon Rescaling:")
    print("-" * 80)
    for path, scale in eps_scales.items():
        mod_type, (fan_in, fan_out) = flat_module_types[path]
        print(f"  {path}")
        print(f"      {mod_type.value}: fan_in={fan_in} -> Eps scale={float(scale):.6f}")
    
    print("\n" + "-" * 80)
    print("Weight Decay Rescaling (larger for wider layers):")
    print("-" * 80)
    for path, scale in wd_scales.items():
        mod_type, (fan_in, fan_out) = flat_module_types[path]
        print(f"  {path}")
        print(f"      {mod_type.value}: fan_in={fan_in} -> WD scale={float(scale):.6f}")
    
    # =========================================================================
    # Step 5: Highlight key insights
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Key Insights - Per-Layer Scaling")
    print("=" * 80)
    
    # Find MLP layers to highlight different fan_in scaling
    mlp_up = None
    mlp_down = None
    for path, (mod_type, (fan_in, fan_out)) in flat_module_types.items():
        if 'Mlp' in path and 'Dense_0' in path:
            mlp_up = (path, fan_in, lr_scales[path], wd_scales[path])
        elif 'Mlp' in path and 'Dense_1' in path:
            mlp_down = (path, fan_in, lr_scales[path], wd_scales[path])
    
    if mlp_up and mlp_down:
        print(f"\nMLP Up Projection (embed -> ffn):")
        print(f"  Path: {mlp_up[0]}")
        print(f"  fan_in: {mlp_up[1]} (embed_dim)")
        print(f"  LR scale: {float(mlp_up[2]):.6f}")
        print(f"  WD scale: {float(mlp_up[3]):.6f}")
        
        print(f"\nMLP Down Projection (ffn -> embed):")
        print(f"  Path: {mlp_down[0]}")
        print(f"  fan_in: {mlp_down[1]} (ffn_dim, 4x wider!)")
        print(f"  LR scale: {float(mlp_down[2]):.6f}")
        print(f"  WD scale: {float(mlp_down[3]):.6f}")
        
        print(f"\n  Ratio (down/up):")
        print(f"    LR ratio: {float(mlp_down[2])/float(mlp_up[2]):.4f} (should be ~0.25 = 128/512)")
        print(f"    WD ratio: {float(mlp_down[3])/float(mlp_up[3]):.4f} (should be ~4.0 = 512/128)")
    
    # =========================================================================
    # Step 6: Get LR/Eps/WD scales as pytrees from CompletedPParameterization
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: LR/Eps/WD Scales as Pytrees (from CompletedPParameterization)")
    print("=" * 80)
    
    # Use the NEW methods that take module_types pytree and return scales pytree
    lr_scales_pytree = parameterization.get_lr_scales_pytree(module_types, device)
    eps_scales_pytree = parameterization.get_eps_scales_pytree(module_types, device)
    wd_scales_pytree = parameterization.get_wd_scales_pytree(module_types, device)
    
    print("\nLR scales pytree (from parameterization.get_lr_scales_pytree()):")
    pprint.pprint(lr_scales_pytree, width=100)
    
    print("\n" + "-" * 80)
    print("\nEpsilon scales pytree (from parameterization.get_eps_scales_pytree()):")
    pprint.pprint(eps_scales_pytree, width=100)
    
    print("\n" + "-" * 80)
    print("\nWeight decay scales pytree (from parameterization.get_wd_scales_pytree()):")
    pprint.pprint(wd_scales_pytree, width=100)
    
    # Also show comparison with existing model methods
    print("\n" + "-" * 80)
    print("\nFor comparison - existing model.get_mup_lrs() (no SDE batch/data scaling):")
    existing_mup_lrs = model.get_mup_lrs(params, device)
    pprint.pprint(existing_mup_lrs, width=100)
    
    # =========================================================================
    # Step 7: Test rescaling parameters by learning rate (simulating gradient rescaling)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Testing Parameter Rescaling (Simulating Gradient Rescaling)")
    print("=" * 80)
    
    print("\nThis demonstrates how to apply per-parameter LR scaling to gradients/updates.")
    print("Using LR scales from CompletedPParameterization.get_lr_scales_pytree()")
    print("In practice, this is done during the optimizer step: scaled_grad = grad * lr_scale")
    
    # Create mock gradients (same structure as params, filled with ones for demonstration)
    mock_gradients = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
    
    def get_path_str(path):
        """Convert a JAX path tuple to a readable string."""
        parts = []
        for p in path:
            if hasattr(p, 'key'):
                parts.append(str(p.key))
            elif hasattr(p, 'idx'):
                parts.append(f"[{p.idx}]")
            else:
                parts.append(str(p))
        return '/'.join(parts)
    
    print("\n1. Original mock gradients (all ones for demonstration):")
    print("   Gradient stats per parameter:")
    for path, grad in jax.tree_util.tree_leaves_with_path(mock_gradients['params']):
        path_str = get_path_str(path)
        print(f"   {path_str}: shape={grad.shape}, mean={float(jnp.mean(grad)):.4f}")
    
    # Use the LR pytree from CompletedPParameterization
    lr_pytree = lr_scales_pytree
    
    print("\n2. LR scales from CompletedPParameterization per parameter:")
    for path, lr in jax.tree_util.tree_leaves_with_path(lr_pytree['params']):
        path_str = get_path_str(path)
        print(f"   {path_str}: lr_scale={float(lr):.6f}")
    
    # Rescale gradients by LR (element-wise multiplication)
    # This mimics what happens in an optimizer with per-parameter learning rates
    # Note: We need to handle the structure mismatch where gradients may have 
    # LogicallyPartitioned wrappers but our LR scales are plain arrays
    
    def rescale_tree(grads_tree, lr_tree):
        """Recursively rescale gradients by LR scales, handling structure mismatches."""
        if isinstance(grads_tree, dict) and isinstance(lr_tree, dict):
            result = {}
            for k in grads_tree:
                if k in lr_tree:
                    result[k] = rescale_tree(grads_tree[k], lr_tree[k])
                else:
                    result[k] = grads_tree[k]  # Keep as-is if no LR scale
            return result
        elif hasattr(grads_tree, 'value'):
            # Handle LogicallyPartitioned - extract inner value, scale it, wrap back
            from flax.linen.spmd import LogicallyPartitioned
            inner_scaled = grads_tree.value * lr_tree
            return LogicallyPartitioned(
                value=inner_scaled,
                names=grads_tree.names,
                mesh=grads_tree.mesh,
                rules=grads_tree.rules
            )
        elif hasattr(grads_tree, 'shape'):
            # Plain JAX array
            return grads_tree * lr_tree
        else:
            # Unknown structure, return as-is
            return grads_tree
    
    scaled_gradients = rescale_tree(mock_gradients, lr_pytree)
    
    print("\n3. Scaled gradients (grad * lr_scale):")
    print("   After applying per-parameter LR scaling:")
    for path, grad in jax.tree_util.tree_leaves_with_path(scaled_gradients['params']):
        path_str = get_path_str(path)
        print(f"   {path_str}: shape={grad.shape}, mean={float(jnp.mean(grad)):.6f}")
    
    # Verify the scaling for specific layers
    print("\n4. Verification - comparing MLP layers:")
    
    # Find MLP layer gradients before and after scaling
    mlp_up_orig = None
    mlp_up_scaled = None
    mlp_down_orig = None
    mlp_down_scaled = None
    
    for path, grad in jax.tree_util.tree_leaves_with_path(mock_gradients['params']):
        path_str = get_path_str(path)
        if 'Mlp_0' in path_str and 'Dense_0' in path_str and 'blocks_0' in path_str:
            mlp_up_orig = (path_str, grad)
        elif 'Mlp_0' in path_str and 'Dense_1' in path_str and 'blocks_0' in path_str:
            mlp_down_orig = (path_str, grad)
    
    for path, grad in jax.tree_util.tree_leaves_with_path(scaled_gradients['params']):
        path_str = get_path_str(path)
        if 'Mlp_0' in path_str and 'Dense_0' in path_str and 'blocks_0' in path_str:
            mlp_up_scaled = (path_str, grad)
        elif 'Mlp_0' in path_str and 'Dense_1' in path_str and 'blocks_0' in path_str:
            mlp_down_scaled = (path_str, grad)
    
    if mlp_up_orig and mlp_down_orig and mlp_up_scaled and mlp_down_scaled:
        print(f"\n   MLP Up (Dense_0, fan_in=128):")
        print(f"     Original grad mean: {float(jnp.mean(mlp_up_orig[1])):.4f}")
        print(f"     Scaled grad mean:   {float(jnp.mean(mlp_up_scaled[1])):.6f}")
        print(f"     Scale factor:       {float(jnp.mean(mlp_up_scaled[1])) / float(jnp.mean(mlp_up_orig[1])):.6f}")
        
        print(f"\n   MLP Down (Dense_1, fan_in=512):")
        print(f"     Original grad mean: {float(jnp.mean(mlp_down_orig[1])):.4f}")
        print(f"     Scaled grad mean:   {float(jnp.mean(mlp_down_scaled[1])):.6f}")
        print(f"     Scale factor:       {float(jnp.mean(mlp_down_scaled[1])) / float(jnp.mean(mlp_down_orig[1])):.6f}")
        
        print(f"\n   Ratio of scaled gradients (down/up):")
        ratio = float(jnp.mean(mlp_down_scaled[1])) / float(jnp.mean(mlp_up_scaled[1]))
        print(f"     {ratio:.4f} (should be 0.25 = 128/512 for base_width=64)")
    
    # =========================================================================
    # Step 8: Demonstrate full update simulation
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 8: Full Update Simulation (params = params - lr * scaled_grad)")
    print("=" * 80)
    
    base_lr = 0.001  # Base learning rate
    
    print(f"\nSimulating update with base_lr = {base_lr}")
    print("Formula: new_params = params - base_lr * (grad * lr_scale)")
    
    # Compute parameter norms before update
    param_norms_before = {}
    for path, p in jax.tree_util.tree_leaves_with_path(params['params']):
        path_str = get_path_str(path)
        param_norms_before[path_str] = float(jnp.linalg.norm(p))
    
    # Apply update: params = params - base_lr * scaled_gradients
    updated_params = jax.tree_util.tree_map(
        lambda p, sg: p - base_lr * sg,
        params,
        scaled_gradients
    )
    
    # Compute parameter norms after update
    param_norms_after = {}
    for path, p in jax.tree_util.tree_leaves_with_path(updated_params['params']):
        path_str = get_path_str(path)
        param_norms_after[path_str] = float(jnp.linalg.norm(p))
    
    print("\nParameter norm changes:")
    print("-" * 80)
    for path_str in sorted(param_norms_before.keys()):
        before = param_norms_before[path_str]
        after = param_norms_after[path_str]
        change = after - before
        pct_change = (change / before * 100) if before > 0 else 0
        print(f"  {path_str}")
        print(f"      Before: {before:.6f}, After: {after:.6f}, Change: {change:.6f} ({pct_change:.4f}%)")
    
    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)
    print("""
Summary:
- Created a real transformer model with TransformerDo
- Used get_module_types() to classify all parameters by ModuleType
- Created CompletedPParameterization with base and current configs
- Computed per-layer LR, epsilon, and weight decay rescalings
- Printed LR/epsilon pytrees using pprint for visualization
- Demonstrated gradient rescaling: scaled_grad = grad * lr_scale
- Verified that wider layers (fan_in=512) get 4x smaller effective updates

This enables hyperparameter transfer across:
- Width (different fan_in per layer)
- Depth (m_L^{α-1} for LR, m_L^{-α} for residuals)
- Batch size (√(m_B/m_D) scaling)
- Dataset size (SDE iso-horizon scaling)
""")
    
    # =========================================================================
    # Step 9: Get Muon Weight Dimension Numbers for optax.contrib.muon
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 9: Muon Weight Dimension Numbers (for optax.contrib.muon)")
    print("=" * 80)
    
    print("\nThis pytree specifies which parameters to optimize with Muon vs AdamW.")
    print("  - MuonDimensionNumbers(...): Parameter optimized with Muon")
    print("  - None: Parameter optimized with AdamW")
    print("\nFor attention layers:")
    print("  - Q/K/V kernels (D, H, Dh): reduction_axis=(0,), output_axis=(1, 2)")
    print("  - Out proj (H, Dh, D): reduction_axis=(0, 1), output_axis=(2,)")
    print("\nFor MLP/Dense kernels (2D): Default reduction_axis=(0,), output_axis=(1,)")
    
    muon_dim_nums = model.get_muon_weight_dimension_numbers(params)
    
    print("\nMuon Weight Dimension Numbers pytree:")
    print("-" * 80)
    pprint.pprint(muon_dim_nums, width=120)
    
    # Also show a summary of which params use Muon vs Adam
    print("\n" + "-" * 80)
    print("Summary - Parameters using Muon vs Adam:")
    print("-" * 80)
    
    def summarize_muon_tree(tree, prefix=''):
        """Recursively summarize the Muon tree."""
        if isinstance(tree, dict):
            for k, v in tree.items():
                new_prefix = f"{prefix}/{k}" if prefix else k
                summarize_muon_tree(v, new_prefix)
        elif tree is None:
            print(f"  [Adam] {prefix}")
        elif hasattr(tree, 'reduction_axis'):
            # MuonDimensionNumbers
            print(f"  [Muon] {prefix}: reduction={tree.reduction_axis}, output={tree.output_axis}")
        else:
            # Some other leaf
            print(f"  [????] {prefix}: {type(tree)}")
    
    summarize_muon_tree(muon_dim_nums['params'])
    
    # =========================================================================
    # Step 10: Get Tensor Types (Fine-Grained Classification)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 10: Tensor Types (Fine-Grained Classification)")
    print("=" * 80)
    
    print("\nTensorType provides fine-grained classification for per-tensor LR control.")
    print("Each TensorType maps to a ModuleType for base scaling rules.")
    print("\nTensorType -> ModuleType mapping:")
    for tt, mt in TENSOR_TYPE_TO_MODULE_TYPE.items():
        print(f"  {tt.value:25s} -> {mt.value}")
    
    # Get tensor types from model
    tensor_types = model.get_tensor_types(params)
    
    print("\n" + "-" * 80)
    print("Tensor Types pytree (from model.get_tensor_types()):")
    print("-" * 80)
    pprint.pprint(tensor_types, width=120)
    
    # Flatten for display
    flat_tensor_types = flatten_tensor_types(tensor_types['params'])
    
    print(f"\nFlattened tensor types ({len(flat_tensor_types)} parameters):")
    for path, (tensor_type, (fan_in, fan_out)) in flat_tensor_types.items():
        module_type = get_module_type(tensor_type)
        print(f"  {path}")
        print(f"      TensorType: {tensor_type.value}, ModuleType: {module_type.value}")
        print(f"      fan_in: {fan_in}, fan_out: {fan_out}")
    
    # =========================================================================
    # Step 11: Per-Tensor Learning Rate Control
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 11: Per-Tensor Learning Rate Control")
    print("=" * 80)
    
    print("\nDemonstrating per-tensor LR multipliers with TensorType.")
    print("This allows different LR scaling for specific tensor types while")
    print("maintaining proper CompletedP parameterization scaling.")
    
    # Define per-tensor LR multipliers
    # Example: reduce LR for query projections, increase for value projections
    per_tensor_lr_multipliers = {
        TensorType.ATTENTION_QUERY: 0.5,   # Half LR for query
        TensorType.ATTENTION_VALUE: 2.0,   # Double LR for value
        TensorType.MLP_GATE: 0.75,         # 75% LR for MLP gate (SwiGLU)
    }
    
    print("\nPer-tensor LR multipliers:")
    for tt, mult in per_tensor_lr_multipliers.items():
        print(f"  {tt.value}: {mult}x")
    
    # Get LR scales with per-tensor overrides
    lr_scales_with_overrides = parameterization.get_lr_scales_with_tensor_types(
        tensor_types,
        per_tensor_lr_multipliers=per_tensor_lr_multipliers,
        device=device
    )
    
    # Get standard LR scales (without overrides) for comparison
    lr_scales_standard = parameterization.get_lr_scales_pytree(tensor_types, device)
    
    print("\n" + "-" * 80)
    print("Comparison: Standard LR scales vs. With Per-Tensor Overrides")
    print("-" * 80)
    
    # Compare specific layers affected by overrides
    def compare_lr_scales(standard_tree, override_tree, prefix=''):
        """Recursively compare LR scales."""
        if isinstance(standard_tree, dict) and isinstance(override_tree, dict):
            for k in standard_tree:
                if k in override_tree:
                    new_prefix = f"{prefix}/{k}" if prefix else k
                    compare_lr_scales(standard_tree[k], override_tree[k], new_prefix)
        else:
            # Leaf node - compare values
            std_val = float(standard_tree)
            ovr_val = float(override_tree)
            if abs(std_val - ovr_val) > 1e-6:
                ratio = ovr_val / std_val if std_val > 0 else float('inf')
                print(f"  {prefix}")
                print(f"      Standard: {std_val:.6f}, With Override: {ovr_val:.6f}, Ratio: {ratio:.2f}x")
    
    compare_lr_scales(lr_scales_standard['params'], lr_scales_with_overrides['params'])
    
    # =========================================================================
    # Step 12: Verify TensorType to ModuleType Mapping
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 12: Verify TensorType to ModuleType Consistency")
    print("=" * 80)
    
    print("\nVerifying that LR scales from TensorType (without overrides)")
    print("match LR scales from ModuleType for all parameters.")
    
    # Get module types for comparison
    module_types = model.get_module_types(params)
    lr_from_module_types = parameterization.get_lr_scales_pytree(module_types, device)
    lr_from_tensor_types = parameterization.get_lr_scales_pytree(tensor_types, device)
    
    # Compare all leaf values
    module_leaves = jax.tree_util.tree_leaves(lr_from_module_types)
    tensor_leaves = jax.tree_util.tree_leaves(lr_from_tensor_types)
    
    all_match = True
    for m_leaf, t_leaf in zip(module_leaves, tensor_leaves):
        if abs(float(m_leaf) - float(t_leaf)) > 1e-6:
            all_match = False
            print(f"  MISMATCH: ModuleType={float(m_leaf):.6f}, TensorType={float(t_leaf):.6f}")
    
    if all_match:
        print("\n  ✓ All LR scales match between ModuleType and TensorType (without overrides)")
    else:
        print("\n  ✗ Some LR scales differ - check TensorType to ModuleType mapping")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)
    print("""
Summary:
- Created a real transformer model with TransformerDo
- Used get_module_types() to classify all parameters by ModuleType
- Used get_tensor_types() to classify all parameters by TensorType (fine-grained)
- Created CompletedPParameterization with base and current configs
- Computed per-layer LR, epsilon, and weight decay rescalings
- Demonstrated per-tensor LR control using TensorType and per_tensor_lr_multipliers
- Verified that TensorType maps correctly to ModuleType for base scaling

TensorType enables:
- Fine-grained control over learning rates for specific tensor types
- Example: Different LR for query vs key vs value projections
- Example: Different LR for MLP up vs gate vs down projections
- All while maintaining proper CompletedP parameterization scaling

This enables hyperparameter transfer across:
- Width (different fan_in per layer)
- Depth (m_L^{α-1} for LR, m_L^{-α} for residuals)
- Batch size (√(m_B/m_D) scaling)
- Dataset size (SDE iso-horizon scaling)
""")


if __name__ == "__main__":
    main()

