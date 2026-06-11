#!/usr/bin/env python
"""
Test script for config_to_dict helper function.

This script parses arguments similar to main.py and tests the recursive
conversion of mmengine Config objects to native Python dictionaries.

Usage:
    python src/test_config_conversion.py \
        --config config/meta_test/meta_test_base.py,\
config/schedule/warmup_cosine_decay.py,\
config/gradient_transform/before/clip_by_global_norm.py,\
config/gradient_transform/after/none.py,\
config/optimizer/mup_adamw.py,\
config/parameterization/complete_p.py \
        --cfg_options \
        gradient_transform_before_optim.0.kwargs.max_norm=1.0 \
        schedule.kwargs.peak_value=0.001 \
        schedule.kwargs.end_value=0.0001 \
        schedule.kwargs.decay_steps=950 \
        schedule.kwargs.warmup_steps=50 \
        optimizer_args.kwargs.lr=0.003 \
        optimizer_args.kwargs.b1=0.9 \
        optimizer_args.kwargs.b2=0.99 \
        optimizer_args.kwargs.weight_decay=0.01
"""

import argparse
import os
import os.path as osp
import pprint
import sys

from mmengine.config import Config, DictAction

# Add src to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_utils import config_to_dict


def comma_separated_strings(string):
    """Parse comma-separated string into a list."""
    return string.split(',')


def parse_args():
    """Parse command line arguments similar to main.py."""
    parser = argparse.ArgumentParser(
        description="Test config_to_dict conversion function"
    )
    
    parser.add_argument("--config_dir", type=str, default="")
    parser.add_argument(
        "--config", 
        type=comma_separated_strings, 
        required=True, 
        help='comma-separated list of config files'
    )
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.'
    )
    
    return parser.parse_args()


def load_and_merge_configs(args):
    """Load and merge config files similar to main.py."""
    config_dir = args.config_dir
    config_files = [osp.join(config_dir, f) for f in args.config]
    
    # Load first config
    cfg = Config.fromfile(config_files[0])
    
    # Manually merge remaining config files, checking for duplicates
    for config_file in config_files[1:]:
        new_cfg = Config.fromfile(config_file)
        for key, value in new_cfg._cfg_dict.items():
            if key in cfg._cfg_dict and cfg._cfg_dict[key] != value:
                raise ValueError(
                    f"Duplicate config key '{key}' with different values: "
                    f"'{cfg._cfg_dict[key]}' vs '{value}'. "
                    f"Found in file: {config_file}"
                )
            cfg._cfg_dict[key] = value
    
    # Apply command line overrides
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    return cfg


def test_is_native_dict(obj, path="root"):
    """
    Recursively verify that an object contains only native Python types
    (no mmengine Config objects).
    
    Returns:
        tuple: (is_valid, error_message or None)
    """
    if isinstance(obj, Config):
        return False, f"Found Config object at {path}"
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            is_valid, error = test_is_native_dict(value, f"{path}.{key}")
            if not is_valid:
                return False, error
    
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            is_valid, error = test_is_native_dict(item, f"{path}[{i}]")
            if not is_valid:
                return False, error
    
    return True, None


def print_type_tree(obj, indent=0, name="root"):
    """Print the type tree of a nested structure for debugging."""
    prefix = "  " * indent
    type_name = type(obj).__name__
    
    if isinstance(obj, dict):
        print(f"{prefix}{name}: {type_name}")
        for key, value in obj.items():
            print_type_tree(value, indent + 1, str(key))
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name}: {type_name}[{len(obj)}]")
        for i, item in enumerate(obj):
            print_type_tree(item, indent + 1, f"[{i}]")
    else:
        print(f"{prefix}{name}: {type_name} = {repr(obj)[:50]}")


def main():
    print("=" * 70)
    print("Testing config_to_dict conversion")
    print("=" * 70)
    
    # Parse arguments
    args = parse_args()
    
    # Load and merge configs
    print("\n[1] Loading and merging config files...")
    cfg = load_and_merge_configs(args)
    print(f"    Loaded {len(args.config)} config files")
    
    # Show the keys we care about for AnyOptimizer
    keys_of_interest = [
        'optimizer_args',
        'schedule',
        'gradient_transform_before_optim',
        'gradient_transform_after_optim',
    ]
    
    print("\n[2] Checking types BEFORE conversion:")
    print("-" * 50)
    for key in keys_of_interest:
        if key in cfg:
            value = cfg[key]
            print(f"\n  {key}:")
            print(f"    Type: {type(value).__name__}")
            if isinstance(value, (dict, Config)):
                for k, v in (value.items() if hasattr(value, 'items') else []):
                    print(f"      {k}: {type(v).__name__}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    print(f"      [{i}]: {type(item).__name__}")
    
    # Test that cfg contains Config objects (expected before conversion)
    print("\n[3] Verifying that raw config contains Config objects...")
    has_config_objects = False
    for key in keys_of_interest:
        if key in cfg:
            value = cfg[key]
            if isinstance(value, Config):
                has_config_objects = True
                print(f"    ✓ {key} is a Config object")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Config):
                        has_config_objects = True
                        print(f"    ✓ {key}[{i}] is a Config object")
    
    if not has_config_objects:
        print("    Note: No Config objects found in keys of interest")
        print("    (mmengine may return native dicts for simple configs)")
    
    # Convert the entire config to native dicts
    print("\n[4] Converting config to native dictionaries...")
    converted_cfg = config_to_dict(cfg)
    print(f"    Converted type: {type(converted_cfg).__name__}")
    
    # Verify the conversion worked
    print("\n[5] Verifying conversion (checking for remaining Config objects)...")
    is_valid, error = test_is_native_dict(converted_cfg)
    if is_valid:
        print("    ✓ SUCCESS: All Config objects converted to native dicts!")
    else:
        print(f"    ✗ FAILED: {error}")
        return 1
    
    # Show the converted types
    print("\n[6] Checking types AFTER conversion:")
    print("-" * 50)
    for key in keys_of_interest:
        if key in converted_cfg:
            value = converted_cfg[key]
            print(f"\n  {key}:")
            print(f"    Type: {type(value).__name__}")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"      {k}: {type(v).__name__}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    print(f"      [{i}]: {type(item).__name__}")
    
    # Show the actual converted values
    print("\n[7] Converted values for AnyOptimizer:")
    print("-" * 50)
    for key in keys_of_interest:
        if key in converted_cfg:
            print(f"\n{key}:")
            pprint.pprint(converted_cfg[key], indent=2, width=80)
    
    # Demonstrate how to use with AnyOptimizer
    print("\n[8] Example usage with AnyOptimizer:")
    print("-" * 50)
    print("""
    from helpers import config_to_dict
    from opt import AnyOptimizer
    
    # Convert config values before passing to AnyOptimizer
    opt = AnyOptimizer(
        optimizer=config_to_dict(args.optimizer_args),
        schedule=config_to_dict(args.schedule),
        gradient_transform_before_optim=config_to_dict(args.gradient_transform_before_optim),
        gradient_transform_after_optim=config_to_dict(args.gradient_transform_after_optim),
        mup_lrs=args.runtime_mup_lrs if USE_MUP else None,
    )
    """)
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

