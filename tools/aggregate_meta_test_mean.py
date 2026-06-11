"""Aggregate per-ckpt meta-test runs into a single 'mean over tasks' summary run.

Source: 36 runs in <NEED>/xiao-meta-testing-lossweight-sweep (9 ckpts x 4 tasks).
Group key: config.wandb_checkpoint_id (each ckpt has 4 task runs).
Output: 9 new runs under group='summary' in the same project, name = "dirX.X_magY.Y",
with per-step mean of all numeric metric columns across the 4 tasks.

Usage:
    python tools/aggregate_meta_test_mean.py
    python tools/aggregate_meta_test_mean.py --dry_run
"""

import argparse
import re
from collections import defaultdict

import pandas as pd
import wandb
from tqdm import tqdm


TIMING_BLACKLIST = {
    "train_opt_time", "train_step_time", "train_fwd_time", "AR time",
}


def is_metric_column(col: str) -> bool:
    if col == "_step":
        return True
    if col.startswith("_"):
        return False
    if col in TIMING_BLACKLIST:
        return False
    if col.endswith("_time"):
        return False
    return True


def parse_dir_mag(name_suffix: str):
    m = re.search(r"dir([0-9.]+)_mag([0-9.]+)", name_suffix or "")
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def fetch_history_df(run) -> pd.DataFrame:
    rows = list(run.scan_history())
    df = pd.DataFrame(rows)
    if "_step" not in df.columns:
        return pd.DataFrame()
    keep = [c for c in df.columns if is_metric_column(c)]
    df = df[keep].set_index("_step")
    df = df.select_dtypes(include="number")
    df = df.dropna(axis=1, how="all")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_project", default="<NEED>/xiao-meta-testing-lossweight-sweep")
    parser.add_argument("--target_entity", default="<NEED>")
    parser.add_argument("--target_project", default="xiao-meta-testing-lossweight-sweep")
    parser.add_argument("--group", default="summary")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing summary runs in target before writing new ones.")
    args = parser.parse_args()

    api = wandb.Api()

    # Pull source runs, exclude any prior summary runs.
    runs = list(api.runs(args.source_project))
    runs = [r for r in runs if r.group != args.group]
    print(f"Source: {len(runs)} runs in {args.source_project} (excluding group='{args.group}')")

    # Group by train ckpt id.
    groups = defaultdict(list)
    for r in runs:
        ckpt_id = r.config.get("wandb_checkpoint_id")
        if not ckpt_id:
            print(f"  skip {r.id}: no wandb_checkpoint_id")
            continue
        groups[ckpt_id].append(r)

    print(f"Grouped into {len(groups)} ckpts:")
    for ckpt_id, rs in groups.items():
        print(f"  {ckpt_id}: {len(rs)} runs")

    if args.overwrite and not args.dry_run:
        target_path = f"{args.target_entity}/{args.target_project}"
        old = list(api.runs(target_path, filters={"group": args.group}))
        print(f"--overwrite: deleting {len(old)} existing summary runs in {target_path} (group='{args.group}')")
        for o in old:
            print(f"  deleting {o.id} ({o.name})")
            o.delete()

    for ckpt_id, rs in tqdm(groups.items(), desc="ckpts"):
        # Resolve train suffix -> dir/mag weights.
        try:
            train_run = api.run(ckpt_id)
            train_suffix = train_run.config.get("name_suffix", "") or ""
        except Exception as e:
            print(f"  WARN: cannot load train run {ckpt_id}: {e}")
            train_suffix = ""

        dir_w, mag_w = parse_dir_mag(train_suffix)
        if dir_w is not None and mag_w is not None:
            run_name = f"dir{dir_w}_mag{mag_w}"
        else:
            run_name = ckpt_id.split("/")[-1]

        # Pull histories.
        dfs = []
        tasks = []
        for r in rs:
            df = fetch_history_df(r)
            if df.empty:
                print(f"  WARN: empty history for {r.id}")
                continue
            task_list = r.config.get("task", [])
            task = task_list[0] if isinstance(task_list, list) and task_list else str(task_list)
            dfs.append(df)
            tasks.append(task)

        if len(dfs) < 2:
            print(f"  ckpt {run_name}: only {len(dfs)} usable runs, skipping")
            continue

        # Per-step mean across the N task runs.
        all_df = pd.concat(dfs)
        mean_df = all_df.groupby(level=0).mean(numeric_only=True).sort_index()

        if args.dry_run:
            print(f"  [dry] {run_name}: would log {len(mean_df)} steps, "
                  f"{len(mean_df.columns)} cols, tasks={tasks}")
            continue

        summary_run = wandb.init(
            entity=args.target_entity,
            project=args.target_project,
            group=args.group,
            name=run_name,
            config={
                "summary_of": ckpt_id,
                "train_name_suffix": train_suffix,
                "dir_weight": dir_w,
                "mag_weight": mag_w,
                "num_tasks_aggregated": len(dfs),
                "tasks": tasks,
            },
            reinit=True,
        )
        for step, row in mean_df.iterrows():
            log = {k: float(v) for k, v in row.items() if pd.notna(v)}
            summary_run.log(log, step=int(step))
        summary_run.finish()
        print(f"  done: {run_name} ({len(mean_df)} steps)")


if __name__ == "__main__":
    main()
