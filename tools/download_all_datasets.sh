#!/usr/bin/env bash
# Script to download all datasets for Scaling L2O
# Usage: ./download_all_datasets.sh [options]
#
# Options:
#   --all           Download all datasets (default if no options specified)
#   --imagenet32    Download ImageNet-32x32x3
#   --imagenet64    Download ImageNet-64x64x3
#   --lm1b          Download LM1B
#   --fineweb10b    Download Fineweb-edu 10B
#   --fineweb100b   Download Fineweb-edu 100B
#   --data-dir DIR  Specify data directory (default: ./data)

set -e

# Default data directory
DATA_DIR="${DATA_DIR:-./data}"

# Flags for which datasets to download
DOWNLOAD_IMAGENET32=false
DOWNLOAD_IMAGENET64=false
DOWNLOAD_LM1B=false
DOWNLOAD_FINEWEB10B=false
DOWNLOAD_FINEWEB100B=false
DOWNLOAD_ALL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        --imagenet32)
            DOWNLOAD_IMAGENET32=true
            shift
            ;;
        --imagenet64)
            DOWNLOAD_IMAGENET64=true
            shift
            ;;
        --lm1b)
            DOWNLOAD_LM1B=true
            shift
            ;;
        --fineweb10b)
            DOWNLOAD_FINEWEB10B=true
            shift
            ;;
        --fineweb100b)
            DOWNLOAD_FINEWEB100B=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --all           Download all datasets (default if no options specified)"
            echo "  --imagenet32    Download ImageNet-32x32x3"
            echo "  --imagenet64    Download ImageNet-64x64x3"
            echo "  --lm1b          Download LM1B"
            echo "  --fineweb10b    Download Fineweb-edu 10B"
            echo "  --fineweb100b   Download Fineweb-edu 100B"
            echo "  --data-dir DIR  Specify data directory (default: ./data)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If no specific dataset selected, download all
if ! $DOWNLOAD_IMAGENET32 && ! $DOWNLOAD_IMAGENET64 && ! $DOWNLOAD_LM1B && ! $DOWNLOAD_FINEWEB10B && ! $DOWNLOAD_FINEWEB100B; then
    DOWNLOAD_ALL=true
fi

if $DOWNLOAD_ALL; then
    DOWNLOAD_IMAGENET32=true
    DOWNLOAD_IMAGENET64=true
    DOWNLOAD_LM1B=true
    DOWNLOAD_FINEWEB10B=true
    DOWNLOAD_FINEWEB100B=true
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure huggingface_hub is installed
uv pip install huggingface_hub --quiet

# Create base data directory
mkdir -p "$DATA_DIR"

echo "============================================"
echo "Downloading datasets to: $DATA_DIR"
echo "============================================"

# Download ImageNet-32
if $DOWNLOAD_IMAGENET32; then
    echo ""
    echo "[1/5] Downloading ImageNet-32x32x3..."
    python "$SCRIPT_DIR/download_dataset.py" --output_dir "$DATA_DIR" --repo_id btherien/imagenet-32x32x3
    echo "ImageNet-32x32x3 download complete."
fi

# Download ImageNet-64
if $DOWNLOAD_IMAGENET64; then
    echo ""
    echo "[2/5] Downloading ImageNet-64x64x3..."
    python "$SCRIPT_DIR/download_dataset.py" --output_dir "$DATA_DIR" --repo_id btherien/imagenet-64x64x3
    echo "ImageNet-64x64x3 download complete."
fi

# Download LM1B
if $DOWNLOAD_LM1B; then
    echo ""
    echo "[3/5] Downloading LM1B..."
    mkdir -p "$DATA_DIR/lm1b/1.1.0"
    python "$SCRIPT_DIR/download_dataset.py" --output_dir "$DATA_DIR/lm1b/1.1.0" --repo_id btherien/lm1b
    echo "LM1B download complete."
fi

# Download Fineweb-edu 10B
if $DOWNLOAD_FINEWEB10B; then
    echo ""
    echo "[4/5] Downloading Fineweb-edu 10B..."
    mkdir -p "$DATA_DIR/fineweb_edu_10B"
    python "$SCRIPT_DIR/download_dataset.py" --output_dir "$DATA_DIR/fineweb_edu_10B" --repo_id btherien/edufineweb-tokenized
    echo "Fineweb-edu 10B download complete."
fi

# Download Fineweb-edu 100B
if $DOWNLOAD_FINEWEB100B; then
    echo ""
    echo "[5/5] Downloading Fineweb-edu 100B..."
    mkdir -p "$DATA_DIR/fineweb_edu_100B"
    python "$SCRIPT_DIR/download_dataset.py" --output_dir "$DATA_DIR/fineweb_edu_100B" --repo_id btherien/edufineweb100BT-tokenized
    echo "Fineweb-edu 100B download complete."
fi

echo ""
echo "============================================"
echo "All requested datasets downloaded!"
echo "============================================"
echo ""
echo "Set the following environment variables:"
echo "  export TFDS_DATA_DIR=$DATA_DIR"
echo "  export WANDB_DIR=\$PWD/wandb"
