#!/usr/bin/env python3
"""Generate fb2.5B versions of all LM scripts."""
import os
import re
import sys

BASE = "<PATH_TO_REPO>"
os.chdir(BASE)

# --- Cluster-specific paths for trailing comments ---
CLUSTER_PATHS = {
    "mila": "<PATH_TO_REPO>",
    "fir": "<PATH_TO_REPO>",
    "tamia": "<PATH_TO_REPO>",
}

def read_file(path):
    with open(os.path.join(BASE, path)) as f:
        return f.read()

def write_file(path, content):
    full = os.path.join(BASE, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write(content)
    print(f"  WROTE: {path}")

def detect_cluster(path):
    if "mila" in path: return "mila"
    if "fir" in path: return "fir"
    if "tamia" in path: return "tamia"
    raise ValueError(f"Unknown cluster: {path}")

def split_at_run_cmd(text):
    """Split into preamble and command at srun/mpirun line."""
    lines = text.split('\n')
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('srun ') or s.startswith('mpirun '):
            return '\n'.join(lines[:i]), '\n'.join(lines[i:])
    raise ValueError("No srun/mpirun found")

def get_omp(cluster):
    return "16" if cluster == "mila" else "12"

def get_master_var(text):
    """Detect if file uses MASTER_NODE or MASTER_ADDR in the srun command."""
    if '--master_node $MASTER_ADDR' in text:
        return '$MASTER_ADDR'
    return '$MASTER_NODE'

def fix_mpirun_preamble(preamble):
    """Fix SBATCH headers when converting mpirun→srun."""
    preamble = re.sub(r'(#SBATCH\s+--ntasks-per-node=)1\b', r'\g<1>4', preamble)
    preamble = re.sub(r'(#SBATCH\s+--ntasks=)1\b', '#SBATCH --ntasks-per-node=4', preamble)
    m = re.search(r'#SBATCH\s+--cpus-per-task=(\d+)', preamble)
    if m:
        old_cpus = int(m.group(1))
        if old_cpus > 12:
            new_cpus = old_cpus // 4
            preamble = preamble.replace(f'--cpus-per-task={old_cpus}', f'--cpus-per-task={new_cpus}')
    return preamble

def make_suffix(old_suffix, is_8btokens=False):
    """Convert old name_suffix to fb2.5B version."""
    if is_8btokens:
        return old_suffix  # keep as-is for 8btokens
    if '_lm' in old_suffix:
        return old_suffix.replace('_lm', '_fb2.5B')
    return old_suffix

def extract_suffix(text):
    m = re.search(r'--name_suffix\s+(\S+)', text)
    return m.group(1) if m else ''

def make_trailing_comments(target_path, cluster):
    """Generate trailing source/sbatch comments."""
    cpath = CLUSTER_PATHS[cluster]
    rel = target_path  # relative from BASE
    full = f"{cpath}/{rel}"
    return f"\n# source {full}\n# sbatch {full}\n"


# ============================================================
# TRANSFORM: learned_opt (only change data, test_project, suffix)
# ============================================================
def transform_learned_opt(src_path, tgt_path):
    text = read_file(src_path)
    cluster = detect_cluster(src_path)
    src_base = os.path.basename(src_path)
    tgt_base = os.path.basename(tgt_path)
    is_8bt = '8btokens' in tgt_base

    # 1. Data config
    text = text.replace('config/data/light_aug.py', 'config/data/no_aug.py')

    # 2. test_project
    text = re.sub(r'--test_project\s+\S+', '--test_project xiao-meta-testing-fb2.5B', text)

    # 3. name_suffix
    old_suf = extract_suffix(text)
    if old_suf:
        new_suf = make_suffix(old_suf, is_8bt)
        text = text.replace(f'--name_suffix {old_suf}', f'--name_suffix {new_suf}')

    # 4. Update comment paths (old basename → new basename)
    if src_base != tgt_base:
        text = text.replace(src_base, tgt_base)

    # 5. mpirun → srun
    if 'mpirun' in text:
        preamble, cmd = split_at_run_cmd(text)
        preamble = fix_mpirun_preamble(preamble)
        omp = get_omp(cluster)
        cmd = re.sub(
            r"mpirun\s+-np\s+\d+(?:\s+--\S+)*\s+bash\s+-c\s+'(?:OMP_NUM_THREADS=\d+\s+)?(?:SLURM_PROCID=\$OMPI_COMM_WORLD_RANK\s+)?(?:SLURM_NTASKS=\d+\s+)?(?:OMP_NUM_THREADS=\d+\s+)?",
            f"srun bash -c 'OMP_NUM_THREADS={omp} ",
            cmd
        )
        text = preamble + '\n' + cmd

    write_file(tgt_path, text)


# ============================================================
# TRANSFORM: adamw/muon sweep (rebuild command from template)
# ============================================================
def build_sweep_command(opt_type, cluster, master_var, name_suffix):
    omp = get_omp(cluster)
    if opt_type == 'adamw':
        cfg_options = """\
gradient_transform_before_optim.0.kwargs.max_norm=1 \\
optimizer_args.kwargs.b1=0.9 \\
optimizer_args.kwargs.b2=0.95 \\
optimizer_args.kwargs.weight_decay=0.1 \\
schedule.kwargs.warmup_steps=100 \\
schedule.kwargs.decay_steps=9537 \\"""
        sweep_cfg = "config/sweeps/adamw_lr_sweep.py"
        opt_name = "adamw"
        opt_cfg = "config/optimizer/adamw.py"
    else:  # muon
        cfg_options = """\
gradient_transform_before_optim.0.kwargs.max_norm=1 \\
optimizer_args.kwargs.beta=0.95 \\
optimizer_args.kwargs.adam_b1=0.9 \\
optimizer_args.kwargs.adam_b2=0.95 \\
optimizer_args.kwargs.weight_decay=0.01 \\
optimizer_args.kwargs.adam_weight_decay=0.01 \\
schedule.kwargs.warmup_steps=100 \\
schedule.kwargs.decay_steps=9537 \\"""
        sweep_cfg = "config/sweeps/muon_lr_sweep.py"
        opt_name = "muon"
        opt_cfg = "config/optimizer/muon.py"

    return f"""srun bash -c 'OMP_NUM_THREADS={omp} python src/main.py \\
--config config/data/no_aug.py,\\
{opt_cfg},\\
config/schedule/warmup_cosine_decay.py,\\
config/gradient_transform/before/clip_by_global_norm.py,\\
config/gradient_transform/after/none.py,\\
{sweep_cfg} \\
--cfg_options \\
{cfg_options}
--test_project xiao-meta-testing-fb2.5B \\
--master_port $MASTER_PORT \\
--master_node {master_var} \\
--num_runs 1 \\
--local_batch_size 32 \\
--ovr_test_batch_size 64 \\
--test_accumulate_steps 4 \\
--optimizer {opt_name} \\
--name_suffix {name_suffix} \\
--num_inner_steps 9537 \\
--gradient_accumulation_steps 4 \\
--test_interval 20 \\
--needs_state \\
--task "transformer-dense-w768-d12-h12_fineweb-s512-gpt2"'"""


def transform_adamw_muon_sweep(src_path, tgt_path, opt_type):
    text = read_file(src_path)
    cluster = detect_cluster(src_path)
    master_var = get_master_var(text)

    old_suf = extract_suffix(text)
    new_suf = old_suf.replace('_lm', '_fb2.5B') if '_lm' in old_suf else old_suf.replace('lm', 'fb2.5B')
    if not new_suf:
        new_suf = f"sweep_{opt_type}_fb2.5B"

    preamble, _ = split_at_run_cmd(text)
    if 'mpirun' in text:
        preamble = fix_mpirun_preamble(preamble)

    cmd = build_sweep_command(opt_type, cluster, master_var, new_suf)
    comments = make_trailing_comments(tgt_path, cluster)
    write_file(tgt_path, preamble + '\n' + cmd + '\n' + comments)


# ============================================================
# TRANSFORM: adamw/muon non-sweep fb2.5B (w768, 9537 steps)
# ============================================================
def build_nonsweep_fb_command(opt_type, cluster, master_var, name_suffix, peak_lr, end_lr):
    omp = get_omp(cluster)
    if opt_type == 'adamw':
        cfg_options = f"""\
gradient_transform_before_optim.0.kwargs.max_norm=1 \\
optimizer_args.kwargs.b1=0.9 \\
optimizer_args.kwargs.b2=0.95 \\
optimizer_args.kwargs.weight_decay=0.1 \\
schedule.kwargs.warmup_steps=100 \\
schedule.kwargs.decay_steps=9537 \\
schedule.kwargs.peak_value={peak_lr} \\
schedule.kwargs.end_value={end_lr} \\"""
        opt_name = "adamw"
        opt_cfg = "config/optimizer/adamw.py"
    else:  # muon
        cfg_options = f"""\
gradient_transform_before_optim.0.kwargs.max_norm=1 \\
optimizer_args.kwargs.beta=0.95 \\
optimizer_args.kwargs.adam_b1=0.9 \\
optimizer_args.kwargs.adam_b2=0.95 \\
optimizer_args.kwargs.weight_decay=0.01 \\
optimizer_args.kwargs.adam_weight_decay=0.01 \\
schedule.kwargs.warmup_steps=100 \\
schedule.kwargs.decay_steps=9537 \\
schedule.kwargs.peak_value={peak_lr} \\
schedule.kwargs.end_value={end_lr} \\"""
        opt_name = "muon"
        opt_cfg = "config/optimizer/muon.py"

    return f"""srun bash -c 'OMP_NUM_THREADS={omp} python src/main.py \\
--config config/data/no_aug.py,\\
{opt_cfg},\\
config/schedule/warmup_cosine_decay.py,\\
config/gradient_transform/before/clip_by_global_norm.py,\\
config/gradient_transform/after/none.py \\
--cfg_options \\
{cfg_options}
--test_project xiao-meta-testing-fb2.5B \\
--master_port $MASTER_PORT \\
--master_node {master_var} \\
--num_runs 1 \\
--local_batch_size 32 \\
--ovr_test_batch_size 64 \\
--test_accumulate_steps 4 \\
--optimizer {opt_name} \\
--name_suffix {name_suffix} \\
--num_inner_steps 9537 \\
--gradient_accumulation_steps 4 \\
--test_interval 20 \\
--needs_state \\
--task "transformer-dense-w768-d12-h12_fineweb-s512-gpt2"'"""


def extract_lr(text):
    """Extract peak_value and end_value from existing non-sweep script."""
    peak = re.search(r'(?:peak_value|peak_lr)=([0-9.e-]+)', text)
    end = re.search(r'(?:end_value|end_lr)=([0-9.e-]+)', text)
    return (peak.group(1) if peak else "0.001"), (end.group(1) if end else "0.0001")


def transform_adamw_muon_nonsweep_fb(src_path, tgt_path, opt_type):
    text = read_file(src_path)
    cluster = detect_cluster(src_path)
    master_var = get_master_var(text)
    peak_lr, end_lr = extract_lr(text)

    old_suf = extract_suffix(text)
    new_suf = old_suf.replace('_lm', '_fb2.5B').replace('_xiao', '_fb2.5B') if old_suf else f"{opt_type}_fb2.5B"
    # Handle special cases
    if '_1gpu' in src_path or '_1p4g' in old_suf:
        new_suf = f"{opt_type}_fb2.5B_1gpu"
    if '_4gpus' in src_path:
        new_suf = f"{opt_type}_fb2.5B_4gpus"

    preamble, _ = split_at_run_cmd(text)
    if 'mpirun' in text:
        preamble = fix_mpirun_preamble(preamble)

    cmd = build_nonsweep_fb_command(opt_type, cluster, master_var, new_suf, peak_lr, end_lr)
    comments = make_trailing_comments(tgt_path, cluster)
    write_file(tgt_path, preamble + '\n' + cmd + '\n' + comments)


# ============================================================
# TRANSFORM: adamw/muon non-sweep 8btokens (w1024, keep steps)
# ============================================================
def transform_adamw_muon_nonsweep_8b(src_path, tgt_path, opt_type):
    text = read_file(src_path)
    cluster = detect_cluster(src_path)
    src_base = os.path.basename(src_path)
    tgt_base = os.path.basename(tgt_path)

    # 1. Data config
    text = text.replace('config/data/light_aug.py', 'config/data/no_aug.py')

    # 2. test_project
    text = re.sub(r'--test_project\s+\S+', '--test_project xiao-meta-testing-fb2.5B', text)

    # 3. name_suffix
    old_suf = extract_suffix(text)
    new_suf = f"{opt_type}_8btokens"
    if old_suf:
        text = text.replace(f'--name_suffix {old_suf}', f'--name_suffix {new_suf}')

    # 4. Update optimizer_args for muon (adamw already correct)
    if opt_type == 'muon':
        text = re.sub(r'optimizer_args\.kwargs\.adam_b2=\S+', 'optimizer_args.kwargs.adam_b2=0.95', text)
        text = re.sub(r'optimizer_args\.kwargs\.weight_decay=\S+', 'optimizer_args.kwargs.weight_decay=0.01', text)
        # Add adam_weight_decay if not present
        if 'adam_weight_decay' not in text:
            text = text.replace(
                'optimizer_args.kwargs.weight_decay=0.01',
                'optimizer_args.kwargs.weight_decay=0.01 \\\noptimizer_args.kwargs.adam_weight_decay=0.01'
            )

    # 5. Update comment paths
    if src_base != tgt_base:
        text = text.replace(src_base, tgt_base)

    # 6. mpirun → srun
    if 'mpirun' in text:
        preamble, cmd = split_at_run_cmd(text)
        preamble = fix_mpirun_preamble(preamble)
        cmd = re.sub(
            r"mpirun\s+-np\s+\d+\s+bash\s+-c\s+'SLURM_PROCID=\$OMPI_COMM_WORLD_RANK\s+",
            "srun bash -c '",
            cmd
        )
        text = preamble + '\n' + cmd

    write_file(tgt_path, text)


# ============================================================
# FILE MANIFEST
# ============================================================
MANIFEST = []

def add(src, tgt, kind, opt_type=None):
    MANIFEST.append((src, tgt, kind, opt_type))

# --- meta-test-sweep-fir (learned opt only, adamw/muon already done) ---
for opt in ['celo2', 'chen', 'naive', 'elo', 'elo_celo2']:
    add(f"jobs/meta-test-sweep-fir/{opt}/{opt}_lm.sh",
        f"jobs/meta-test-sweep-fir/{opt}/{opt}_fb2.5B.sh", "learned_opt")

# --- meta-test-sweep-mila ---
add("jobs/meta-test-sweep-mila/adamw/adamw_lm.sh",
    "jobs/meta-test-sweep-mila/adamw/adamw_fb2.5B.sh", "sweep", "adamw")
add("jobs/meta-test-sweep-mila/muon/muon_lm.sh",
    "jobs/meta-test-sweep-mila/muon/muon_fb2.5B.sh", "sweep", "muon")
add("jobs/meta-test-sweep-mila/muon/muon_lm_1.sh",
    "jobs/meta-test-sweep-mila/muon/muon_fb2.5B_1.sh", "sweep", "muon")
for opt in ['celo2', 'chen', 'naive', 'elo', 'elo_celo2']:
    add(f"jobs/meta-test-sweep-mila/{opt}/{opt}_lm.sh",
        f"jobs/meta-test-sweep-mila/{opt}/{opt}_fb2.5B.sh", "learned_opt")

# --- meta-test-sweep-tamia ---
add("jobs/meta-test-sweep-tamia/adamw/adamw_lm.sh",
    "jobs/meta-test-sweep-tamia/adamw/adamw_fb2.5B.sh", "sweep", "adamw")
add("jobs/meta-test-sweep-tamia/muon/muon_lm.sh",
    "jobs/meta-test-sweep-tamia/muon/muon_fb2.5B.sh", "sweep", "muon")
add("jobs/meta-test-sweep-tamia/muon/muon_lm_1.sh",
    "jobs/meta-test-sweep-tamia/muon/muon_fb2.5B_1.sh", "sweep", "muon")
for opt in ['celo2', 'chen', 'naive', 'elo', 'elo_celo2']:
    add(f"jobs/meta-test-sweep-tamia/{opt}/{opt}_lm.sh",
        f"jobs/meta-test-sweep-tamia/{opt}/{opt}_fb2.5B.sh", "learned_opt")

# --- meta-test-fir ---
# adamw/muon: both fb2.5B and 8btokens
add("jobs/meta-test-fir/adamw/adamw_lm.sh",
    "jobs/meta-test-fir/adamw/adamw_fb2.5B.sh", "nonsweep_fb", "adamw")
add("jobs/meta-test-fir/adamw/adamw_lm.sh",
    "jobs/meta-test-fir/adamw/adamw_8btokens.sh", "nonsweep_8b", "adamw")
add("jobs/meta-test-fir/muon/muon_lm.sh",
    "jobs/meta-test-fir/muon/muon_fb2.5B.sh", "nonsweep_fb", "muon")
add("jobs/meta-test-fir/muon/muon_lm.sh",
    "jobs/meta-test-fir/muon/muon_8btokens.sh", "nonsweep_8b", "muon")
add("jobs/meta-test-fir/muon/muon_lm_1gpu.sh",
    "jobs/meta-test-fir/muon/muon_fb2.5B_1gpu.sh", "nonsweep_fb", "muon")
# learned opt
for opt in ['celo2', 'chen', 'elo', 'naive', 'elo_celo2']:
    add(f"jobs/meta-test-fir/{opt}/{opt}_lm.sh",
        f"jobs/meta-test-fir/{opt}/{opt}_fb2.5B.sh", "learned_opt")
# learned opt variants
add("jobs/meta-test-fir/chen/chen_lm_lr1e-3.sh",
    "jobs/meta-test-fir/chen/chen_fb2.5B_lr1e-3.sh", "learned_opt")
add("jobs/meta-test-fir/chen/chen_lm_lr1e-5.sh",
    "jobs/meta-test-fir/chen/chen_fb2.5B_lr1e-5.sh", "learned_opt")
add("jobs/meta-test-fir/elo/elo_lm_1.sh",
    "jobs/meta-test-fir/elo/elo_fb2.5B_1.sh", "learned_opt")
add("jobs/meta-test-fir/naive/naive_lm_1.sh",
    "jobs/meta-test-fir/naive/naive_fb2.5B_1.sh", "learned_opt")
add("jobs/meta-test-fir/naive/naive_lm_lr1e-3.sh",
    "jobs/meta-test-fir/naive/naive_fb2.5B_lr1e-3.sh", "learned_opt")
add("jobs/meta-test-fir/naive/naive_lm_lr1e-5.sh",
    "jobs/meta-test-fir/naive/naive_fb2.5B_lr1e-5.sh", "learned_opt")
# 8btokens in-place
add("jobs/meta-test-fir/celo2/celo2_lm_8btokens.sh",
    "jobs/meta-test-fir/celo2/celo2_lm_8btokens.sh", "learned_opt")
add("jobs/meta-test-fir/elo/elo_lm_8btokens.sh",
    "jobs/meta-test-fir/elo/elo_lm_8btokens.sh", "learned_opt")

# --- meta-test-mila ---
# adamw/muon: both fb2.5B and 8btokens
add("jobs/meta-test-mila/adamw/adamw_lm.sh",
    "jobs/meta-test-mila/adamw/adamw_fb2.5B.sh", "nonsweep_fb", "adamw")
add("jobs/meta-test-mila/adamw/adamw_lm.sh",
    "jobs/meta-test-mila/adamw/adamw_8btokens.sh", "nonsweep_8b", "adamw")
add("jobs/meta-test-mila/muon/muon_lm.sh",
    "jobs/meta-test-mila/muon/muon_fb2.5B.sh", "nonsweep_fb", "muon")
add("jobs/meta-test-mila/muon/muon_lm.sh",
    "jobs/meta-test-mila/muon/muon_8btokens.sh", "nonsweep_8b", "muon")
add("jobs/meta-test-mila/muon/muon_lm_1gpu.sh",
    "jobs/meta-test-mila/muon/muon_fb2.5B_1gpu.sh", "nonsweep_fb", "muon")
add("jobs/meta-test-mila/muon_lm_4gpus.sh",
    "jobs/meta-test-mila/muon_fb2.5B_4gpus.sh", "nonsweep_fb", "muon")
# learned opt
for opt in ['celo2', 'chen', 'elo', 'naive', 'elo_celo2']:
    add(f"jobs/meta-test-mila/{opt}/{opt}_lm.sh",
        f"jobs/meta-test-mila/{opt}/{opt}_fb2.5B.sh", "learned_opt")
# learned opt variants
add("jobs/meta-test-mila/celo2/celo2_lm_pretrained.sh",
    "jobs/meta-test-mila/celo2/celo2_fb2.5B_pretrained.sh", "learned_opt")
add("jobs/meta-test-mila/chen/chen_lm_lr1e-3.sh",
    "jobs/meta-test-mila/chen/chen_fb2.5B_lr1e-3.sh", "learned_opt")
add("jobs/meta-test-mila/chen/chen_lm_lr1e-5.sh",
    "jobs/meta-test-mila/chen/chen_fb2.5B_lr1e-5.sh", "learned_opt")
add("jobs/meta-test-mila/elo/elo_lm_1.sh",
    "jobs/meta-test-mila/elo/elo_fb2.5B_1.sh", "learned_opt")
add("jobs/meta-test-mila/elo/elo_lm_2.sh",
    "jobs/meta-test-mila/elo/elo_fb2.5B_2.sh", "learned_opt")
add("jobs/meta-test-mila/naive/naive_lm_1.sh",
    "jobs/meta-test-mila/naive/naive_fb2.5B_1.sh", "learned_opt")
add("jobs/meta-test-mila/naive/naive_lm_lr1e-3.sh",
    "jobs/meta-test-mila/naive/naive_fb2.5B_lr1e-3.sh", "learned_opt")
add("jobs/meta-test-mila/naive/naive_lm_lr1e-5.sh",
    "jobs/meta-test-mila/naive/naive_fb2.5B_lr1e-5.sh", "learned_opt")
# 8btokens in-place
add("jobs/meta-test-mila/celo2/celo2_lm_8btokens.sh",
    "jobs/meta-test-mila/celo2/celo2_lm_8btokens.sh", "learned_opt")
add("jobs/meta-test-mila/celo2/celo2_lm_8btokens_pretrained.sh",
    "jobs/meta-test-mila/celo2/celo2_lm_8btokens_pretrained.sh", "learned_opt")
add("jobs/meta-test-mila/elo/elo_lm_8btokens.sh",
    "jobs/meta-test-mila/elo/elo_lm_8btokens.sh", "learned_opt")

# --- meta-test-tamia ---
# adamw/muon: both fb2.5B and 8btokens
add("jobs/meta-test-tamia/adamw/adamw_lm.sh",
    "jobs/meta-test-tamia/adamw/adamw_fb2.5B.sh", "nonsweep_fb", "adamw")
add("jobs/meta-test-tamia/adamw/adamw_lm.sh",
    "jobs/meta-test-tamia/adamw/adamw_8btokens.sh", "nonsweep_8b", "adamw")
add("jobs/meta-test-tamia/muon/muon_lm.sh",
    "jobs/meta-test-tamia/muon/muon_fb2.5B.sh", "nonsweep_fb", "muon")
add("jobs/meta-test-tamia/muon/muon_lm.sh",
    "jobs/meta-test-tamia/muon/muon_8btokens.sh", "nonsweep_8b", "muon")
# learned opt
for opt in ['celo2', 'chen', 'elo', 'naive', 'elo_celo2']:
    add(f"jobs/meta-test-tamia/{opt}/{opt}_lm.sh",
        f"jobs/meta-test-tamia/{opt}/{opt}_fb2.5B.sh", "learned_opt")
# learned opt variants
add("jobs/meta-test-tamia/celo2/celo2_lm_1.sh",
    "jobs/meta-test-tamia/celo2/celo2_fb2.5B_1.sh", "learned_opt")
add("jobs/meta-test-tamia/elo/elo_lm_2.sh",
    "jobs/meta-test-tamia/elo/elo_fb2.5B_2.sh", "learned_opt")
# 8btokens in-place
add("jobs/meta-test-tamia/celo2/celo2_lm_8btokens.sh",
    "jobs/meta-test-tamia/celo2/celo2_lm_8btokens.sh", "learned_opt")
add("jobs/meta-test-tamia/elo/elo_lm_8btokens.sh",
    "jobs/meta-test-tamia/elo/elo_lm_8btokens.sh", "learned_opt")


# ============================================================
# MAIN
# ============================================================
def main():
    dry_run = '--dry-run' in sys.argv
    errors = []
    count = 0

    for src, tgt, kind, opt_type in MANIFEST:
        src_full = os.path.join(BASE, src)
        if not os.path.exists(src_full):
            errors.append(f"MISSING: {src}")
            continue

        label = f"[{kind}] {src} -> {tgt}"
        if dry_run:
            print(f"  DRY-RUN: {label}")
            count += 1
            continue

        print(label)
        try:
            if kind == "learned_opt":
                transform_learned_opt(src, tgt)
            elif kind == "sweep":
                transform_adamw_muon_sweep(src, tgt, opt_type)
            elif kind == "nonsweep_fb":
                transform_adamw_muon_nonsweep_fb(src, tgt, opt_type)
            elif kind == "nonsweep_8b":
                transform_adamw_muon_nonsweep_8b(src, tgt, opt_type)
            count += 1
        except Exception as e:
            errors.append(f"ERROR: {src} -> {tgt}: {e}")

    print(f"\nProcessed: {count} files")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
