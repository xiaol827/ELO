from datasets import Dataset
import numpy as np
import glob
import os
from huggingface_hub import HfApi, login, create_repo
from tqdm import tqdm
import argparse

# Add command line arguments
parser = argparse.ArgumentParser(description='Upload dataset to Hugging Face Hub')
parser.add_argument('--test', action='store_true', help='Test with only one file per split')
args = parser.parse_args()

data_root = "<DATA_DIR>/lm1b/1.1.0"

# Login to Hugging Face
login()

# Initialize the Hugging Face API
api = HfApi()

# Get all .npy files
train_files = [ 
    "<DATA_DIR>/imagenet_64x64x3_JPEG.h5"
]
# train_files = sorted(glob.glob(os.path.join(data_root, "lm1b-*"))) + [
#     os.path.join(data_root, "dataset_info.json")
# ]

val_files = [] #sorted(glob.glob(os.path.join(data_root, "edufineweb_val_*.npy")))

# If in test mode, only use one file per split
if args.test:
    train_files = train_files[:1]
    val_files = val_files[:1]
    print("Test mode: Using only one file per split")

# Create a new dataset repository (if it doesn't exist)
repo_id = "btherien/imagenet-64x64x3"

# Create the repository first if it doesn't exist
try:
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"Repository {repo_id} is ready")
except Exception as e:
    print(f"Error creating repository: {e}")
    exit(1)

# Upload all files with tqdm progress bar
all_files = train_files + val_files
for file in tqdm(all_files, desc="Uploading files", unit="file"):
    tqdm.write(f"Uploading {os.path.basename(file)}...")
    try:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=os.path.basename(file),  # Keep only the filename in the repo
            repo_id=repo_id,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Error uploading {os.path.basename(file)}: {e}")