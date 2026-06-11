import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download, login
from tqdm import tqdm

def download_dataset(output_dir, repo_id=None):
    """
    Download the edufineweb-tokenized dataset from Hugging Face Hub
    
    Args:
        output_dir: Directory to save the dataset files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Login to Hugging Face (will prompt for token if not already logged in)
    login()
    

    print(f"Downloading the entire dataset from {repo_id}")
    try:
      snapshot_download(
          repo_id=repo_id,
          repo_type="dataset",
          local_dir=output_dir,
          local_dir_use_symlinks=False
      )
      print(f"Successfully downloaded the entire dataset to {output_dir}")
    except Exception as e:
      print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download edufineweb-tokenized dataset from Hugging Face Hub')
    parser.add_argument('--output_dir', type=str, default="<OUTPUT_DIR>",
                        help='Directory to save the dataset files')
    parser.add_argument('--repo_id', type=str, default="btherien/edufineweb-tokenized",
                        help='dataset to download')
    args = parser.parse_args()
    
    download_dataset(args.output_dir, args.repo_id)
