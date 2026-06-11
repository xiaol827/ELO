#!/usr/bin/env python3
"""
Generate AdamW sweep config with tied beta hyperparameters based on halflife.

The halflife h relates to beta by: beta = 2^(-1/h)
This means after h steps, the EMA retains half its original weight.

Usage:
    python generate_adamw_tied_sweep.py                 # prints to stdout
    python generate_adamw_tied_sweep.py -o my_sweep.py  # writes to file
"""

import argparse
import numpy as np

# =============================================================================
# TOP 20 BEST RESULTS FROM FINE-GRAINED SWEEP (for reference)
# =============================================================================
TOP_RESULTS = """
| B1   | B2   | Weight Decay | Peak LR | Train Loss |
|------|------|--------------|---------|------------|
| 0.9  | 0.9  | 0.25         | 0.03125 | 4.83722    |
| 0.9  | 0.95 | 0.03125      | 0.125   | 4.85828    |
| 0.95 | 0.9  | 0.0625       | 0.0625  | 4.87318    |
| 0.95 | 0.99 | 0.0625       | 0.0625  | 4.88939    |
| 0.95 | 0.95 | 0.125        | 0.0625  | 4.89631    |
| 0.95 | 0.95 | 0.0625       | 0.0625  | 4.90174    |
| 0.95 | 0.99 | 0.03125      | 0.25    | 4.90352    |
| 0.8  | 0.99 | 0.03125      | 0.0625  | 4.90680    |
| 0.95 | 0.9  | 0.125        | 0.0625  | 4.90957    |
| 0.95 | 0.95 | 0.03125      | 0.125   | 4.91465    |

Best performing region:
- B1: 0.8-0.95 (halflife ~3-14)
- B2: 0.9-0.99 (halflife ~7-69)
- Weight Decay: 0.015625-0.25
- Peak LR: 0.03125-0.25
"""

# =============================================================================
# CONFIGURABLE HALFLIFE RANGES
# =============================================================================

# Halflife values for beta1 (momentum)
H1_VALUES = [3, 6, 9, 12, 15]

# Halflife multipliers for beta2 relative to h1
# h2 = multiplier * h1
H2_MULTIPLIERS = [1, 2, 5, 10, 100]

# Learning rate range (log-spaced, 6 values)
LR_MIN = 0.015
LR_MAX = 0.5
NUM_LR_VALUES = 7

# Weight decay range (log-spaced, 6 values)
WD_MIN = 0.0075
WD_MAX = 0.5
NUM_WD_VALUES = 7

# Schedule fixed parameters
DECAY_STEPS = 1900
WARMUP_STEPS = 100
EXPONENT = 1.0
INIT_VALUE = 0.0
END_VALUE_RATIO = 0.1  # end_value = peak_value * END_VALUE_RATIO

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def halflife_to_beta(h):
    """Convert halflife h to beta value: beta = 2^(-1/h)"""
    return round(2 ** (-1 / h), 5)


def beta_to_halflife(beta):
    """Convert beta to halflife: h = -1/log2(beta)"""
    return -1 / np.log2(beta)


def generate_log_spaced(min_val, max_val, num_values):
    """Generate log-spaced values between min and max."""
    return [float(x) for x in np.geomspace(min_val, max_val, num_values)]


# =============================================================================
# GENERATION LOGIC
# =============================================================================

def generate_schedule_kwargs(lr_values):
    """Generate schedule kwargs dicts for each peak learning rate."""
    schedules = []
    for peak_value in lr_values:
        schedules.append({
            'decay_steps': DECAY_STEPS,
            'end_value': peak_value * END_VALUE_RATIO,
            'exponent': EXPONENT,
            'init_value': INIT_VALUE,
            'peak_value': peak_value,
            'warmup_steps': WARMUP_STEPS,
        })
    return schedules


def generate_optimizer_kwargs_combinations():
    """
    Generate all combinations of optimizer kwargs with tied beta values.
    
    Returns list of optimizer kwargs dicts (without learning_rate, since that's in schedule).
    """
    wd_values = generate_log_spaced(WD_MIN, WD_MAX, NUM_WD_VALUES)
    
    combinations = []
    
    for h1 in H1_VALUES:
        b1 = halflife_to_beta(h1)
        for h2_mult in H2_MULTIPLIERS:
            h2 = h1 * h2_mult
            b2 = halflife_to_beta(h2)
            for wd in wd_values:
                kwargs = dict(
                    b1=b1,
                    b2=b2,
                    eps=1e-8,
                    eps_root=0.0,
                    mu_dtype=None,
                    nesterov=False,
                    mask=None,
                    weight_decay=wd,
                )
                combinations.append({
                    'h1': h1,
                    'h2_mult': h2_mult,
                    'h2': h2,
                    'b1': b1,
                    'b2': b2,
                    'wd': wd,
                    'kwargs': kwargs,
                })
    
    return combinations, wd_values


def format_schedule_dict(d):
    """Format a single schedule dict with proper indentation."""
    lines = ["{"]
    lines.append(f"  'decay_steps': {d['decay_steps']},")
    lines.append(f"  'end_value': {d['end_value']},")
    lines.append(f"  'exponent': {d['exponent']},")
    lines.append(f"  'init_value': {d['init_value']},")
    lines.append(f"  'peak_value': {d['peak_value']},")
    lines.append(f"  'warmup_steps': {d['warmup_steps']},")
    lines.append("}")
    return '\n'.join(lines)


def format_optimizer_kwargs(kwargs):
    """Format optimizer kwargs dict for the config file."""
    lines = ["{"]
    lines.append(f"    'b1': {kwargs['b1']},")
    lines.append(f"    'b2': {kwargs['b2']},")
    lines.append(f"    'eps': {kwargs['eps']},")
    lines.append(f"    'eps_root': {kwargs['eps_root']},")
    lines.append(f"    'mu_dtype': {kwargs['mu_dtype']},")
    lines.append(f"    'nesterov': {kwargs['nesterov']},")
    lines.append(f"    'mask': {kwargs['mask']},")
    lines.append(f"    'weight_decay': {kwargs['weight_decay']},")
    lines.append("}")
    return '\n'.join(lines)


def generate_sweep_config():
    """Generate the complete sweep config file content."""
    lr_values = generate_log_spaced(LR_MIN, LR_MAX, NUM_LR_VALUES)
    schedule_kwargs_list = generate_schedule_kwargs(lr_values)
    optimizer_combinations, wd_values = generate_optimizer_kwargs_combinations()
    
    # Build schedule kwargs values string
    schedule_entries = []
    for sched in schedule_kwargs_list:
        schedule_entries.append(format_schedule_dict(sched))
    schedule_values_str = ',\n '.join(schedule_entries)
    
    # Build optimizer kwargs values string
    optimizer_entries = []
    for combo in optimizer_combinations:
        optimizer_entries.append(format_optimizer_kwargs(combo['kwargs']))
    optimizer_values_str = ',\n '.join(optimizer_entries)
    
    # Calculate grid size
    grid_size = len(schedule_kwargs_list) * len(optimizer_combinations)
    
    # Print summary info
    summary_lines = []
    summary_lines.append(f"# Grid size: {grid_size} configurations")
    summary_lines.append(f"# - H1 values: {H1_VALUES}")
    summary_lines.append(f"# - H2 multipliers: {H2_MULTIPLIERS}")
    summary_lines.append(f"# - LR values ({NUM_LR_VALUES}): {[round(x, 6) for x in lr_values]}")
    summary_lines.append(f"# - WD values ({NUM_WD_VALUES}): {[round(x, 6) for x in wd_values]}")
    summary_lines.append(f"# - Optimizer combos: {len(optimizer_combinations)} (h1 x h2_mult x wd)")
    summary_lines.append(f"# - Schedule combos: {len(schedule_kwargs_list)} (lr)")
    summary_lines.append("#")
    summary_lines.append("# Beta values derived from halflife (beta = 2^(-1/h)), rounded to 5 decimals:")
    for h1 in H1_VALUES:
        b1 = halflife_to_beta(h1)
        summary_lines.append(f"#   h1={h1:2d} -> b1={b1}")
        for h2_mult in H2_MULTIPLIERS:
            h2 = h1 * h2_mult
            b2 = halflife_to_beta(h2)
            summary_lines.append(f"#       h2={h2:4d} ({h2_mult:3d}x h1) -> b2={b2}")
    summary = '\n'.join(summary_lines)
    
    config = f'''_base_ = ["./sweeps_base.py"]


sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        schedule__kwargs=dict(
            values=[
 {schedule_values_str},
            ]
        ),
        optimizer_args__kwargs=dict(
            values=[
 {optimizer_values_str},
            ]
        ),
    ),
)
'''
    return config, summary, grid_size


def print_beta_table():
    """Print a table showing halflife to beta conversions."""
    print("\nHalflife to Beta conversion table (rounded to 5 decimals):")
    print("-" * 50)
    print(f"{'Halflife':>10} | {'Beta':>12} | {'Description':<20}")
    print("-" * 50)
    
    for h1 in H1_VALUES:
        b1 = halflife_to_beta(h1)
        print(f"{h1:>10} | {b1:>12} | b1 base")
        for h2_mult in H2_MULTIPLIERS:
            h2 = h1 * h2_mult
            b2 = halflife_to_beta(h2)
            print(f"{h2:>10} | {b2:>12} | b2 = {h2_mult}x h1")
        print("-" * 50)
    
    print("\nCommon beta values for reference:")
    for beta in [0.8, 0.9, 0.95, 0.99, 0.999]:
        h = beta_to_halflife(beta)
        print(f"  beta={beta} -> halflife={h:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate AdamW sweep config with tied beta hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=TOP_RESULTS
    )
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (default: print to stdout)")
    parser.add_argument("--show-top", action="store_true",
                        help="Show top results from previous sweep")
    parser.add_argument("--show-betas", action="store_true",
                        help="Show halflife to beta conversion table")
    args = parser.parse_args()
    
    if args.show_top:
        print(TOP_RESULTS)
        return
    
    if args.show_betas:
        print_beta_table()
        return
    
    config, summary, grid_size = generate_sweep_config()
    
    print(summary)
    print()
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(config)
        print(f"Wrote sweep config to: {args.output}")
    else:
        print(config)


if __name__ == "__main__":
    main()
