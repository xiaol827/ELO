import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Any
from transformers import AutoTokenizer
# from learned_optimization.tasks.datasets.base import ThreadSafeIterator, LazyIterator
from learned_optimization.tasks.datasets.base import Datasets, ThreadSafeIterator, LazyIterator
# DataLoaderLite copied from fineweb_loading_test.py (with JAX output)

def create_doc_ids_from_padding(tokens, pad_token_id=0):
  """
  Create per-token document IDs from packed sequences using padding as boundaries.

  Returns [B, L] int32 document IDs (starting from 1; padding positions get 0).
  The T×T attention mask is computed on GPU inside the model from these IDs,
  avoiding the 500× data transfer overhead of passing [B, L, L] through the
  data pipeline.

  Args:
    tokens: [batch, length] token ids
    pad_token_id: Token id used for padding (default 0)

  Returns:
    doc_ids: [batch, length] int32, where each non-padding token gets the ID
             of its document (1-indexed), and padding tokens get 0.
  """
  non_padding = (tokens != pad_token_id)  # [B, L]
  padded_tokens = jnp.pad(tokens, ((0, 0), (1, 0)), constant_values=pad_token_id)[:, :-1]
  is_after_padding = (padded_tokens == pad_token_id)  # [B, L]
  doc_boundary_markers = is_after_padding & non_padding
  doc_ids = jnp.cumsum(doc_boundary_markers, axis=1)  # [B, L]
  return doc_ids


def create_document_mask_from_padding(tokens, pad_token_id=0):
  """Legacy: create full T×T attention mask. Prefer create_doc_ids_from_padding."""
  non_padding = (tokens != pad_token_id)
  padded_tokens = jnp.pad(tokens, ((0, 0), (1, 0)), constant_values=pad_token_id)[:, :-1]
  is_after_padding = (padded_tokens == pad_token_id)
  doc_boundary_markers = is_after_padding & non_padding
  doc_ids = jnp.cumsum(doc_boundary_markers, axis=1)
  doc_ids_i = doc_ids[:, :, None]
  doc_ids_j = doc_ids[:, None, :]
  same_doc = (doc_ids_i == doc_ids_j)
  can_attend_to = non_padding[:, None, :]
  mask = same_doc & can_attend_to
  return mask

def create_simple_padding_mask(tokens, pad_token_id=0):
  """
  Simple mask that only prevents attending TO padding tokens.
  Does not enforce document boundaries - tokens can attend across documents.
  
  Args:
    tokens: [batch, length] token ids
    pad_token_id: Token id for padding
  
  Returns:
    mask: [batch, 1, length, length] where mask[b, 0, i, j] = True means
          position i can attend to position j
  """
  non_padding = (tokens != pad_token_id)  # [B, L]
  mask = non_padding[:, None, :]  # [B, 1, L] - can attend to non-padding positions
  mask = jnp.broadcast_to(mask, (tokens.shape[0], 1, tokens.shape[1], tokens.shape[1]))
  return mask


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, name="fineweb", data_root=os.path.join(os.environ["TFDS_DATA_DIR"], "fineweb_edu_10B"), eos_token_id=0):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        data_root = data_root
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split}"
        self.eos_token_id = eos_token_id
        self.shards = shards
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = np.load(self.shards[self.current_shard], allow_pickle=True).astype(np.int32)
        self.current_position = self.B * self.T * self.process_rank

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = np.reshape(buf[:-1], (B, T))
        y = np.reshape(buf[1:], (B, T))
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = np.load(self.shards[self.current_shard], allow_pickle=True).astype(np.int32)
            self.current_position = B * T * self.process_rank
        # Convert to jax arrays and put on device
        x = jax.device_put(jnp.array(x), jax.devices('gpu')[jax.process_index()])
        y = jax.device_put(jnp.array(y), jax.devices('gpu')[jax.process_index()])
        doc_ids = create_doc_ids_from_padding(x, pad_token_id=self.eos_token_id)
        return {'image': x, 'label': y, 'attention_mask': doc_ids}

def make_fineweb_datasets(
    batch_size=[2, 1, 1, 1],
    sequence_length=64,
    prefetch_batches=[1, 1, 1, 1],
    process_rank=0,
    num_processes=1,
    batch_shape=None,
    name="fineweb",
    data_root=os.path.join(os.environ["TFDS_DATA_DIR"], "fineweb_edu_10B"),
    hf_tokenizer="openai-community/gpt2",
    **kwargs
):
    """Create a Datasets object for FineWeb, with JAX arrays and GPT2 vocab size."""

    print("Making fineweb datasets with args:")
    print(f"batch_size: {batch_size}")
    print(f"sequence_length: {sequence_length}")
    print(f"prefetch_batches: {prefetch_batches}")
    print(f"process_rank: {process_rank}")
    print(f"num_processes: {num_processes}")
    print(f"batch_shape: {batch_shape}")

    splits = ['train', 'train', 'train', 'val']  # train, inner_valid, outer_valid, test
    split_names = ['train', 'inner_valid', 'outer_valid', 'test']
    split_map = {
        'train':splits[0],
        'inner_valid':splits[1],
        'outer_valid':splits[2],
        'test':splits[3],
    }
    batch_shape_map = {
        'train': batch_shape if batch_shape is not None else (batch_size[0],),
        'inner_valid': (batch_size[1],),
        'outer_valid': (batch_size[2],),
        'test': (batch_size[3],),
    }
    assert len(splits) == len(batch_size) == len(prefetch_batches)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    eos_token_id = hf_tokenizer.eos_token_id
    vocab_size = len(hf_tokenizer.vocab)
    # eos_token_id = 50256
    # vocab_size = 50257

    def make(split, B, T, batch_shape):
        def iterator_fn():
            loader = DataLoaderLite(B, T, process_rank, num_processes, split_map[split], data_root=data_root, eos_token_id=eos_token_id)
            for batch in loader:
                # Reshape to batch_shape if provided
                if batch_shape is not None:
                    shape = batch_shape_map[split] + (T,)
                    batch = {
                        'image': jnp.reshape(batch['image'], shape),
                        'label': jnp.reshape(batch['label'], shape),
                        'attention_mask': jnp.reshape(batch['attention_mask'], shape),  # doc_ids: same shape as image

                    }
                yield batch
        return ThreadSafeIterator(LazyIterator(iterator_fn))

    # Determine batch_shape for each split
    def get_batch_shape(bs):
        if batch_shape is not None:
            return batch_shape
        else:
            return (bs,)

    iters = [make(split_names[i], batch_size[i], sequence_length, get_batch_shape(batch_size[i])) for i in range(4)]
    abstract_batch = {
        'image': jax.core.ShapedArray((1, sequence_length), jnp.int32),
        'label': jax.core.ShapedArray((1, sequence_length), jnp.int32),
         'sequence_length':  jax.core.ShapedArray((1, sequence_length,sequence_length), jnp.int32),
    }
    extra_info = {
        'vocab_size': vocab_size,
        'vocab': None,
        'name': f'{name}-s{sequence_length}-gpt2',
        'eos_token_id': eos_token_id,
        'hf_tokenizer': hf_tokenizer,
        'sequence_length': sequence_length,
    }
    return Datasets(
        train=iters[0],
        inner_valid=iters[1],
        outer_valid=iters[2],
        test=iters[3],
        extra_info=extra_info,
        abstract_batch=abstract_batch,
    )

if __name__ == "__main__":
    # Example settings for hierarchical batch shape
    batch_shape = (2, 3, 4, 5)  # (perturbations, workers, local_steps, local_batch_size)
    sequence_length = 8
    batch_size = [2*3*4*5, 3*4*5, 4*5, 5]  # train, inner_valid, outer_valid, test
    prefetch_batches = [1, 1, 1, 1]
    process_rank = 0
    num_processes = 1

    ds = make_fineweb_datasets(
        batch_size=batch_size,
        sequence_length=sequence_length,
        prefetch_batches=prefetch_batches,
        process_rank=process_rank,
        num_processes=num_processes,
        batch_shape=batch_shape,
    )

    split_names = ["train", "inner_valid", "outer_valid", "test"]
    splits = [ds.train, ds.inner_valid, ds.outer_valid, ds.test]

    for name, split in zip(split_names, splits):
        it = iter(split)
        batch = next(it)
        print(f"Split: {name}")
        for k, v in batch.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, type={type(v)})")
            print(v)
        print("-")
        break
    print("FineWeb DataLoader test complete.") 