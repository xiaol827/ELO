#!/bin/bash

# Script to generate individual SLURM jobs for learning rate sweep across different model sizes

OUTPUT_DIR="jobs/sweep/generated_lr_sweep"
mkdir -p ${OUTPUT_DIR}

# Model size parameters
widths=(256 512)
depths=(8 12 16)


widths=(128)
depths=(8)

# Learning rates to sweep
learning_rates=(0.060988 0.12298 0.030246 0.015 0.24797 0.5)

echo "Generating SLURM job files..."
echo "Output directory: ${OUTPUT_DIR}"

job_count=0

for w in "${widths[@]}"; do
    for d in "${depths[@]}"; do
        for lr in "${learning_rates[@]}"; do
            # Calculate end learning rate (half of starting lr)
            end_lr=$(echo "scale=20; ${lr}/2" | bc)
            
            # Create job name (replace . with _ for filename compatibility)
            lr_name=$(echo ${lr} | sed 's/\./_/g')
            job_name="lr_sweep_w${w}_d${d}_lr${lr_name}"
            job_file="${OUTPUT_DIR}/${job_name}.sh"
            
            # Generate SLURM job file
            cat > ${job_file} << EOF
#!/bin/bash
#SBATCH --partition=long
#SBATCH -J ${job_name}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --mail-user <EMAIL>
#SBATCH --mail-type=END
#SBATCH --gres=gpu:l40s:1
#SBATCH -o <HOME>/log/sweeps/%j.out
#SBATCH -e <HOME>/log/sweeps/%j.err

# Source setup
source <HOME>/l2o_install/setup.sh

# Change to working directory
cd <HOME>/l2o_install/scaling_l2o

echo "=============================================="
echo "Running with:"
echo "  Model width: ${w}"
echo "  Model depth: ${d}"
echo "  Learning rate: ${lr}"
echo "  End learning rate: ${end_lr}"
echo "=============================================="

mpirun -np 1 \\
--allow-run-as-root \\
--map-by ppr:1:node \\
--bind-to none \\
--oversubscribe \\
-x CUDA_VISIBLE_DEVICES='0' \\
-x OMP_NUM_THREADS=8 \\
python src/main.py \\
--config config/meta_test/meta_test_base.py,\\
config/schedule/warmup_cosine_decay.py,\\
config/gradient_transform/before/clip_by_global_norm.py,\\
config/gradient_transform/after/none.py,\\
config/optimizer/mup_adamw.py,\\
config/parameterization/complete_p_bs100k_steps2000.py \\
--cfg_options \\
gradient_transform_before_optim.0.kwargs.max_norm=1.0 \\
schedule.kwargs.peak_value=${lr} \\
schedule.kwargs.end_value=${end_lr} \\
schedule.kwargs.decay_steps=1900 \\
schedule.kwargs.warmup_steps=100 \\
optimizer_args.kwargs.b1=0.95484 \\
optimizer_args.kwargs.b2=0.9908 \\
optimizer_args.kwargs.weight_decay=0.093198 \\
test_project=complete_p_testing \\
num_inner_steps=2000 \\
name_suffix='_w${w}_d${d}' \\
optimizer=mup_adamw \\
gradient_accumulation_steps=16 \\
ovr_test_batch_size=128 \\
master_node=\$MASTER_NODE \\
master_port=\$MASTER_PORT \\
num_runs=1 \\
test_interval=15 \\
--needs_state \\
--local_batch_size 64 \\
--task mutransformer-dense-w${w}-d${d}-h4_fineweb-s128-gpt2

echo ""
echo "Finished run with lr=${lr}, w=${w}, d=${d}"
echo ""
EOF
            
            chmod +x ${job_file}
            job_count=$((job_count + 1))
            echo "Generated: ${job_file}"
        done
    done
done

echo ""
echo "=============================================="
echo "Job generation complete!"
echo "Total jobs generated: ${job_count}"
echo "=============================================="
echo ""
echo "To submit all jobs, run:"
echo "  for job in ${OUTPUT_DIR}/*.sh; do sbatch \$job; done"
echo ""
echo "Or submit individual jobs with:"
echo "  sbatch ${OUTPUT_DIR}/<job_name>.sh"

