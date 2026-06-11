import wandb
import argparse
import os



def upload_checkpoint_to_wandb(project, run_id, checkpoint_path):
    # Check if the checkpoint file exists
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Initialize the existing run
    run = wandb.init(project=project, id=run_id, resume="allow")

    # Create an artifact for the checkpoint
    artifact = wandb.Artifact('checkpoint', type='model')

    # Determine the base path for the file
    base_path = os.path.dirname(checkpoint_path)

    # Add the checkpoint file to the artifact, preserving directory structure
    artifact.add_file(checkpoint_path, name=os.path.relpath(checkpoint_path, base_path))

    # Log the artifact to W&B
    run.log_artifact(artifact)

    wandb.save(checkpoint_path,)

    # Finish the run
    run.finish()

    print(f"Checkpoint {checkpoint_path} uploaded to run {run_id} in project {project}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a checkpoint file to an existing Weights & Biases run.")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--run_id", type=str, required=True, help="W&B run ID")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")

    args = parser.parse_args()

    upload_checkpoint_to_wandb(args.project, args.run_id, args.checkpoint_path)



"""
python tools/upload_checkpoint.py \
--project mup-meta-training \
--run_id l01dw6n0 \
--checkpoint_path /btherien/github/new_install/learned_aggregation/checkpoints/l01dw6n0MuRNNMLPLOpt32_multi-task-withmumlp-w1024-d3_imagenet-32x32x3_K8_H4_0.5_mup_RNN_distributed/global_step5000.pickle

python tools/upload_checkpoint.py \
--project mup-meta-training \
--run_id hso8fj12 \
--checkpoint_path /btherien/github/new_install/learned_aggregation/checkpoints/hso8fj12RNNMLPLOpt32_multi-task-withmlp-w1024-d3_imagenet-32x32x3_K8_H4_0.5_sp_RNN_distributed/global_step5000.pickle

"""