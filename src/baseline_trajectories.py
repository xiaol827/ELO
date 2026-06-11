import numpy as np
import pickle
import os


def load_trajectories(path=None):
    if path is None:
        # Get the path to the baselines.pickle file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)  # scaling_l2o directory
        path = os.path.join(root_dir, 'baselines.pickle')
    
    with open(path, 'rb') as f:
        return pickle.load(f)




def main():
    """Test loading baseline trajectories and print basic information."""
    try:
        trajectories = load_trajectories()
        print(f"Successfully loaded baseline trajectories")
        
        # Print basic information about the loaded data
        if isinstance(trajectories, dict):
            print(f"Number of trajectory entries: {len(trajectories)}")
            print("Keys in the trajectories dictionary:")
            for key in trajectories.keys():
                print(f"  - {key}")
                
            # Sample a key to show its structure
            if trajectories:
                sample_key = next(iter(trajectories))
                print(f"\nSample entry for key '{sample_key}':")
                if isinstance(trajectories[sample_key], dict):
                    for subkey, value in trajectories[sample_key].items():
                        if isinstance(value, np.ndarray):
                            print(f"  {subkey}: numpy array with shape {value.shape}")
                        else:
                            print(f"  {subkey}: {type(value)}")
                else:
                    print(f"  Type: {type(trajectories[sample_key])}")
        else:
            print(f"Trajectories type: {type(trajectories)}")
            
    except Exception as e:
        print(f"Error loading trajectories: {e}")


if __name__ == "__main__":
    main()
