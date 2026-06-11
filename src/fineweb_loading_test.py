import os
import numpy as np
import torch
from learned_optimization.tasks.datasets.base import Datasets, ThreadSafeIterator, LazyIterator
# Minimal config for DataLoaderLite
class Config:
    batch_size = 4  # Small batch size for testing
    class MODEL:
        class GPT2:
            block_size = 8  # Small sequence length for testing

# Function to load tokens (copied from train.py)
def load_tokens(filename):
    npt = np.load(filename, allow_pickle=True)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# DataLoaderLite (copied and slightly modified for standalone use)
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = "data/fineweb_edu_10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.shards = shards
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

if __name__ == "__main__":
    # Minimal config
    config = Config()
    B = config.batch_size
    T = config.MODEL.GPT2.block_size
    process_rank = 0
    num_processes = 1
    split = "train"

    print("Testing DataLoaderLite...")
    loader = DataLoaderLite(B, T, process_rank, num_processes, split)
    it = iter(loader)
    for i in range(3):
        x, y = next(it)
        print(f"Batch {i+1}:")
        print("x:", x)
        print("y:", y)
        print("x shape:", x.shape, "y shape:", y.shape)
        print("-")
    print("DataLoaderLite test complete.") 