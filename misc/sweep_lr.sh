#!/bin/bash

# Learning rate sweep: 2^-4 to 2^-30

# for power in $(seq -6 -1 -14); do
for power in $(seq -5 -1 -11); do
# for power in $(seq -3 -1 -6); do
    # Calculate 2^power using bc for arbitrary precision
    lr=$(echo "scale=20; 2^($power)" | bc)
    end_lr=$(echo "scale=20; ${lr}/2" | bc)
    
    echo "=============================================="
    echo "Running with learning rate: 2^${power} = ${lr}"
    echo "End learning rate: ${end_lr}"
    echo "=============================================="
    
    mpirun -np 1 \
--allow-run-as-root \
--map-by ppr:1:node \
--bind-to none \
--oversubscribe \
-x CUDA_VISIBLE_DEVICES='0' \
-x OMP_NUM_THREADS=24 \
python src/main.py \
--config config/meta_test/meta_test_base.py,\
config/schedule/warmup_cosine_decay.py,\
config/gradient_transform/before/clip_by_global_norm.py,\
config/gradient_transform/after/none.py,\
config/optimizer/mup_adamw.py,\
config/parameterization/complete_p.py \
--cfg_options \
gradient_transform_before_optim.0.kwargs.max_norm=1.0 \
schedule.kwargs.peak_value=${lr} \
schedule.kwargs.end_value=${end_lr} \
schedule.kwargs.decay_steps=1900 \
schedule.kwargs.warmup_steps=100 \
optimizer_args.kwargs.b1=0.9 \
optimizer_args.kwargs.b2=0.99 \
optimizer_args.kwargs.weight_decay=0.01 \
test_project=complete_p_testing \
num_inner_steps=2000 \
name_suffix='' \
optimizer=mup_adamw \
gradient_accumulation_steps=4 \
ovr_test_batch_size=128 \
master_node=$MASTER_NODE \
master_port=$MASTER_PORT \
num_runs=1 \
test_interval=15 \
--needs_state \
--local_batch_size 64 \
--task mutransformer-dense-w512-d4-h4_fineweb-s128-gpt2
    
    echo ""
    echo "Finished run with lr=2^${power}"
    echo ""
done

echo "=============================================="
echo "Learning rate sweep complete!"
echo "=============================================="
