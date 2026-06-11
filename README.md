# Welcome to (Efficient Long-hOrizon Learning)ELO!

## Installation 
Copy and run this single script to install everything:

```bash
#!/usr/bin/env bash
set -e

export INSTALL_DIR=$PWD
mkdir -p $INSTALL_DIR/l2o_install && cd $INSTALL_DIR/l2o_install

# Install UV and setup Python environment
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -p 3.11 .venv && source .venv/bin/activate
export UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

git clone https://github.com/xiaol827/ELO.git ELO && cd ELO

pushd learned_optimization && uv pip install -e . && popd

# Clone and install vision_transformer
git clone https://github.com/google-research/vision_transformer
pushd vision_transformer && git checkout ac6e056 && uv pip install -e . && popd

# Install all dependencies in one command
uv pip install \
    mmengine seqio wandb aiofiles gin-config optax_shampoo transformers \
    torch torchvision dm-haiku chex flax mpi4py huggingface_hub \
    "jax[cuda12]==0.9.0" orbax-checkpoint==0.3.2 \
    git+https://github.com/haydn-jones/SOAP_JAX \
    git+https://github.com/google-deepmind/optax.git

# big_vision extras: tensorflow_addons (EOL) is required by
# big_vision/pp/archive/autoaugment.py and only works against the Keras 2 API
# exposed by `tf-keras`. Pin tensorflow / tensorflow-text / tf-keras to the
# 2.20.x triplet (the latest line for which all three publish matching wheels).
uv pip install \
    tensorflow==2.20.0 \
    tensorflow-text==2.20.1 \
    tf-keras==2.20.1 \
    tensorflow_addons==0.23.0

# Route `import keras` to tf-keras: tensorflow_addons references private
# Keras 2 internals (e.g. keras.src.engine) that do not exist in Keras 3,
# which TF >= 2.16 installs as the default `keras` package.
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
rm -rf "$SITE_PACKAGES/keras"
ln -snf tf_keras "$SITE_PACKAGES/keras"

# Setup data directories (adjust paths as needed)
mkdir -p ~/scratch/tensorflow_datasets ~/scratch/wandb
ln -sf ~/scratch/tensorflow_datasets data
ln -sf ~/scratch/wandb wandb

# Create setup.sh for future sessions
cat > $INSTALL_DIR/l2o_install/setup.sh << EOF
# Set this to your installation directory (where you ran the install script)
export INSTALL_DIR=$INSTALL_DIR

source \$INSTALL_DIR/l2o_install/.venv/bin/activate
cd \$INSTALL_DIR/l2o_install/ELO
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/include
export WANDB_API_KEY=<WANDB_API_KEY>
export MASTER_NODE=\$HOSTNAME
export MASTER_PORT=12345
export TFDS_DATA_DIR=\$PWD/data
export WANDB_DIR=\$PWD/wandb
EOF

echo "Installation complete! Run 'source $INSTALL_DIR/l2o_install/setup.sh' to activate."
```

After installation, edit `setup.sh` to:
1. Verify `INSTALL_DIR` is set to the correct path (should be auto-populated during installation)
2. Replace `YOUR_WANDB_API_KEY_HERE` with your actual WANDB API key

Then activate the environment with:
```bash
source /path/to/l2o_install/setup.sh
```

## Download data

Use the `download_all_datasets.sh` script to download datasets:

```bash
# Download all datasets
./tools/download_all_datasets.sh

# Download specific datasets (can combine multiple options)
./tools/download_all_datasets.sh --imagenet32
./tools/download_all_datasets.sh --imagenet32 --imagenet64 --lm1b

# Specify custom data directory
./tools/download_all_datasets.sh --data-dir /path/to/data --all
```

### Available options

| Option | Description |
|--------|-------------|
| `--all` | Download all datasets (default if no options specified) |
| `--imagenet32` | Download ImageNet-32x32x3 |
| `--imagenet64` | Download ImageNet-64x64x3 |
| `--lm1b` | Download LM1B |
| `--fineweb10b` | Download Fineweb-edu 10B |
| `--fineweb100b` | Download Fineweb-edu 100B |
| `--data-dir DIR` | Specify data directory (default: `./data`) |
| `-h, --help` | Show help message |

After downloading, set the environment variables:
```bash
export TFDS_DATA_DIR=$PWD/data
export WANDB_DIR=$PWD/wandb
```

### Manual download (individual datasets)

#### ImageNet-1K (32)
```
pip install huggingface_hub
python tools/download_dataset.py --output_dir data --repo_id btherien/imagenet-32x32x3
```

#### Fineweb-edu 10B
```
pip install huggingface_hub
mkdir -p data/fineweb_edu_10B
python tools/download_dataset.py --output_dir data/fineweb_edu_10B --repo_id btherien/edufineweb-tokenized
```
#### ImageNet-1K (224) for Big_vision
See `big_vision/README.md`.

## Usage (ELO-Celo2 as an example)

### Meta-training
```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python src/main.py \
--config config/meta_train/meta_train_base.py,\
config/learned_optimizer/elo_celo2.py,\
config/inner_length_schedule/constant_100_2000.py,\
config/gradient_estimators/pes_elo.py,\
config/truncated_step/lopt_elo.py,\
config/data/no_aug.py,\
config/schedule/warmup_cosine_decay.py,\
config/optimizer/adamw.py,\
config/gradient_transform/before/clip_by_global_norm.py,\
config/gradient_transform/after/none.py \
--cfg_options \
task_aug_scale_range=[0.001,1000] \
gradient_transform_before_optim.0.kwargs.max_norm=1.0 \
learned_optimizer_args.kwargs.peak_lr=0.001 \
learned_optimizer_args.kwargs.init_lr=0.001 \
learned_optimizer_args.kwargs.end_lr=0.001 \
learned_optimizer_args.kwargs.warmup_fraction=0.0 \
learned_optimizer_args.kwargs.adam_lr_mult=20 \
learned_optimizer_args.kwargs.weight_decay=0.0 \
learned_optimizer_args.kwargs.expert_lr_max=0.01 \
learned_optimizer_args.kwargs.expert_lr_min=0.0001 \
learned_optimizer_args.kwargs.muon_expert_adamlr_scaler=0.3 \
learned_optimizer_args.kwargs.expert_optim=muon \
learned_optimizer_args.kwargs.clip_grad=True \
learned_optimizer_args.kwargs.clip_norm=1.0 \
schedule.kwargs.peak_value=1e-4 \
schedule.kwargs.end_value=1e-5 \
schedule.kwargs.decay_steps=99900 \
schedule.kwargs.warmup_steps=100 \
inner_problem_length_schedule.sample_choice=uniform \
gradient_estimator_args.kwargs.steps_per_jit=10 \
gradient_estimator_args.kwargs.truncation_length=50 \
gradient_estimator_args.kwargs.expert_wd_sp=5000 \
gradient_estimator_args.kwargs.expert_wd_ep=50000 \
gradient_estimator_args.kwargs.expert_traj_wmin=0.0 \
gradient_estimator_args.kwargs.expert_dirloss_weight=0.7 \
gradient_estimator_args.kwargs.expert_magloss_weight=0.3 \
gradient_estimator_args.kwargs.expert_dirloss_wmin=0.3 \
gradient_estimator_args.kwargs.expert_magloss_wmin=0.0 \
truncated_step_args.kwargs.buffer_cfg.thred=0.2 \
truncated_step_args.kwargs.buffer_cfg.min_thred=0.2 \
truncated_step_args.kwargs.buffer_cfg.buffer_size=1 \
--num_tasks 8 \
--num_outer_steps 100000 \
--local_batch_size 64 64 64 64 \
--train_project <NEED> \
--optimizer ELO_Celo2LOpt \
--needs_state \
--name_suffix elo_celo2 \
--prefetch_batches 20 \
--task "mlp-w32-d1_mnist-8x8x1,mlp-w32-d1_fashionmnist-8x8x1,mlp-w32-d1_cifar10-8x8x3,mlp-w32-d1_svhn-8x8x3" \
--use_task_augmentation \
--auto_resume \
--image_dtype float32
```
This will automatically log training metrics and the learned optimizer weights to a W&B run. Be sure to record the **wandb_checkpoint_id** and use it to fill in **<NEED>** below.

### Image classification (ViT-Base/16, ImageNet-1K 224 resolution)

```
python3 -m big_vision.train_lo \
--config big_vision/configs/vit_i1k_elo_celo2.py:variant=B/16,aug=strong8 \
--config.wandb.name elo_celo2_in1k_vitb16_lr3.16e_5_strong8 \
--config.wandb_checkpoint_id <NEED> \
--config.total_steps 50000 \
--config.model.dtype_mm bfloat16 \
--config.input.batch_size 2048 \
--config.lr 3.16e-5 \
--config.wd 0.001 \
--config.schedule.min_lr_factor 0.01 \
--config.lo_kwargs.init_lr=0.0 \
--config.lo_kwargs.adam_lr_mult=20 \
--config.lo_kwargs.clip_grad=True \
--config.lo_kwargs.clip_norm=1.0 \
--config.lo_kwargs.meta_train=False \
--resume
```

### Language Modeling (GPT2 (350M), 7B Tokens)

```
srun bash -c 'OMP_NUM_THREADS=16 python src/main.py \
--config config/meta_test/meta_test_base.py,\
config/data/no_aug.py,\
config/learned_optimizer/elo_celo2.py,\
config/parameterization/complete_p_w64_bs64_sl16_steps1000.py,\
config/gradient_transform/before/clip_by_global_norm.py,\
config/gradient_transform/after/none.py \
--cfg_options \
learned_optimizer_args.kwargs.meta_train=False \
learned_optimizer_args.kwargs.init_lr=0.0 \
learned_optimizer_args.kwargs.peak_lr=0.0002 \
learned_optimizer_args.kwargs.end_lr=2e-6 \
learned_optimizer_args.kwargs.weight_decay=0.1 \
learned_optimizer_args.kwargs.adam_lr_mult=20 \
learned_optimizer_args.kwargs.clip_grad=True \
learned_optimizer_args.kwargs.clip_norm=1.0 \
--test_project <NEED> \
--master_port $MASTER_PORT \
--master_node $MASTER_ADDR \
--num_runs 1 \
--local_batch_size 32 \
--ovr_test_batch_size 64 \
--test_accumulate_steps 4 \
--optimizer ELO_Celo2LOpt \
--name_suffix elo_celo2_fb7B \
--num_inner_steps 13351 \
--gradient_accumulation_steps 8 \
--test_interval 20 \
--needs_state \
--task "transformer-dense-w1024-d24-h16_fineweb-s512-gpt2" \
--wandb_checkpoint_id <NEED> \
--auto_resume \
--save_iter 500'
```
