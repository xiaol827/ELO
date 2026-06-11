#!/bin/bash
# Aggregate per-ckpt 4-task meta-test runs into 9 'mean over tasks' summary runs
# under group='summary' in <NEED>/xiao-meta-testing-lossweight-sweep.
#
# Usage:
#   bash tools/aggregate_meta_test_mean.sh                # normal run
#   bash tools/aggregate_meta_test_mean.sh --dry_run      # preview grouping only
#   bash tools/aggregate_meta_test_mean.sh --overwrite    # delete old summaries first

set -e

cd "$(cd "$(dirname "$0")/.." && pwd)"
source .venv/bin/activate

export WANDB_API_KEY=<WANDB_API_KEY>
export WANDB_ENTITY=<NEED>

python tools/aggregate_meta_test_mean.py --group summary_coslr "$@"
