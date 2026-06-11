#!/usr/bin/env python3
"""
Generate AdamW sweep config files with configurable hyperparameter ranges.

Usage:
    python generate_adamw_sweep.py                      # prints to stdout
    python generate_adamw_sweep.py -o my_sweep.py       # writes to file
"""

import argparse
from pprint import pformat

# =============================================================================
# TOP 10 BEST RESULTS FROM PREVIOUS SWEEP (for reference)
# =============================================================================
TOP_10_RESULTS = """
| Name                 | Train Loss | B1   | B2   | Peak LR   | Weight Decay |
|----------------------|------------|------|------|-----------|--------------|
| mild-sweep-765       | 4.90767    | 0.9  | 0.9  | 0.03125   | 0.125        |
| clean-sweep-807      | 4.97262    | 0.9  | 0.99 | 0.03125   | 0.125        |
| splendid-sweep-817   | 4.97363    | 0.9  | 0.99 | 0.5       | 0.0078125    |
| hardy-sweep-770      | 4.98971    | 0.9  | 0.9  | 0.125     | 0.03125      |
| vital-sweep-602      | 5.02590    | 0.7  | 0.99 | 0.125     | 0.03125      |
| worldly-sweep-723    | 5.04637    | 0.9  | 0.7  | 0.03125   | 0.125        |
| swept-sweep-1016     | 5.08538    | 0.99 | 0.99 | 0.125     | 0.125        |
| solar-sweep-759      | 5.08791    | 0.9  | 0.9  | 0.03125   | 0.5          |
| polished-sweep-802   | 5.08861    | 0.9  | 0.99 | 0.0078125 | 0.5          |
| northern-sweep-760   | 5.09065    | 0.9  | 0.9  | 0.0078125 | 0.5          |

Key observations:
- B1: Best values around 0.9 (range 0.7-0.99)
- B2: Best values 0.7-0.99, with 0.9 and 0.99 most common
- Peak LR: Best values 0.03125-0.5 (geometric range)
- Weight Decay: Best values 0.0078125-0.5 (geometric range)
"""

# =============================================================================
# CONFIGURABLE HYPERPARAMETER RANGES
# =============================================================================

# Beta1 values to sweep (momentum term)
B1_VALUES = [0.7, 0.8, 0.9, 0.95, 0.99]

# Beta2 values to sweep (second moment term)
B2_VALUES = [0.7, 0.8, 0.9, 0.95, 0.99]

# Peak learning rate values to sweep
# Note: end_value is computed as peak_value / 10
PEAK_LR_VALUES = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]

# Weight decay values to sweep
WEIGHT_DECAY_VALUES = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]

# Schedule fixed parameters
DECAY_STEPS = 1900
WARMUP_STEPS = 100
EXPONENT = 1.0
INIT_VALUE = 0.0
END_VALUE_RATIO = 0.1  # end_value = peak_value * END_VALUE_RATIO

# =============================================================================
# GENERATION LOGIC
# =============================================================================

def generate_schedule_kwargs(peak_lr_values):
    """Generate schedule kwargs dicts for each peak learning rate."""
    schedules = []
    for peak_value in peak_lr_values:
        schedules.append({
            'decay_steps': DECAY_STEPS,
            'end_value': peak_value * END_VALUE_RATIO,
            'exponent': EXPONENT,
            'init_value': INIT_VALUE,
            'peak_value': peak_value,
            'warmup_steps': WARMUP_STEPS,
        })
    return schedules


def format_schedule_dict(d):
    """Format a single schedule dict with proper indentation."""
    lines = ["{"]
    for key in ['decay_steps', 'end_value', 'exponent', 'init_value', 'peak_value', 'warmup_steps']:
        value = d[key]
        lines.append(f"  '{key}': {value},")
    lines.append("}")
    return '\n'.join(lines)


def generate_sweep_config():
    """Generate the complete sweep config file content."""
    schedules = generate_schedule_kwargs(PEAK_LR_VALUES)
    
    # Build the schedule values string
    schedule_entries = []
    for s in schedules:
        schedule_entries.append(format_schedule_dict(s))
    schedule_values_str = ',\n '.join(schedule_entries)
    
    # Calculate grid size
    grid_size = len(PEAK_LR_VALUES) * len(B1_VALUES) * len(B2_VALUES) * len(WEIGHT_DECAY_VALUES)
    
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
        optimizer_args__kwargs__b1=dict(
                values={B1_VALUES}
            
        ),
        optimizer_args__kwargs__b2=dict(
                values={B2_VALUES}
            
        ),
        optimizer_args__kwargs__weight_decay=dict(
                values={WEIGHT_DECAY_VALUES}
            
        ),
        
    ),
)
'''
    return config, grid_size


def main():
    parser = argparse.ArgumentParser(
        description="Generate AdamW sweep config files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=TOP_10_RESULTS
    )
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (default: print to stdout)")
    parser.add_argument("--show-top10", action="store_true",
                        help="Show top 10 results from previous sweep")
    args = parser.parse_args()
    
    if args.show_top10:
        print(TOP_10_RESULTS)
        return
    
    config, grid_size = generate_sweep_config()
    
    print(f"# Grid size: {grid_size} configurations")
    print(f"# - Peak LR values: {len(PEAK_LR_VALUES)}")
    print(f"# - B1 values: {len(B1_VALUES)}")
    print(f"# - B2 values: {len(B2_VALUES)}")
    print(f"# - Weight decay values: {len(WEIGHT_DECAY_VALUES)}")
    print()
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(config)
        print(f"Wrote sweep config to: {args.output}")
    else:
        print(config)


if __name__ == "__main__":
    main()
