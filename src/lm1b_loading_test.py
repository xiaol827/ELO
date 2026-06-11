import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import tensorflow as tf
import tensorflow_datasets as tfds
import seqio
from flax.training import prefetch_iterator
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple
# from learned_optimization.tasks.datasets.base import ThreadSafeIterator, LazyIterator
from learned_optimization.tasks.datasets.base import Datasets, ThreadSafeIterator, LazyIterator
# --- Copied from tasks.py ---
def _crop_or_pad(value, size, pad_token):
    """Either crop or pad value to be of size size."""
    val_size = tf.size(value)
    pad = lambda: tf.pad(
        value, [[0, size - val_size]],
        'CONSTANT',
        constant_values=pad_token)
    return tf.cond(val_size < size, pad, lambda: value[:size])

def _load(name, 
          tokenizer, 
          batch_size: int, 
          sequence_length: int,
          split,
          label_sharding=None,
          image_sharding=None,
          prefetch_batches=None,
          device=None,
          batch_shape=(1,),
          seed=42) -> Tuple[tf.data.Dataset, int]:
    """Load tfds tf.data.Dataset in a streaming fashion."""
    ds = tfds.load(name, split=split, shuffle_files=True)

    crop_size = sequence_length + 1
    ds = ds.repeat()
    ds = ds.map(lambda x: tokenizer.encode_tf(x['text']))
    ds = ds.map(lambda t: _crop_or_pad(t, crop_size, pad_token=0))
    # Set seed for reproducible shuffling
    ds = ds.shuffle(batch_size * 10, seed=seed*jax.process_index())

    # Create the language modeling observation/target pairs and batch them up.
    def create_lm_obs_target(t):
        return hk.data_structures.to_immutable_dict(dict(image=t[:-1], label=t[1:]))

    ds = ds.map(create_lm_obs_target)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(lambda x: jax.tree_util.tree_map(lambda xx: tf.reshape(xx,batch_shape + (sequence_length,)),x))
    
    # Convert to numpy and create prefetch buffer on GPU
    ds = tfds.as_numpy(ds)
    
    def put_on_device(batch):
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, device), batch)
    
    # Create iterator that prefetches to GPU
    it = map(put_on_device, ds)
    it = prefetch_iterator.PrefetchIterator(it, buffer_size=prefetch_batches) # Adjust buffer size as needed
    
    return it

# class Datasets:
#     def __init__(self, train, inner_valid, outer_valid, test, extra_info, abstract_batch):
#         self.train = train
#         self.inner_valid = inner_valid
#         self.outer_valid = outer_valid
#         self.test = test
#         self.extra_info = extra_info
#         self.abstract_batch = abstract_batch

def _make_datasets(tfds_datasetname: str,
                   use_localsgd_batches: bool,
                   vocab: seqio.vocabularies.Vocabulary,
                   batch_size: int,
                   sequence_length: int,
                   has_test: bool = True,
                   prefetch_batches = [1,1,1,1],
                   label_sharding=None,
                   image_sharding=None,
                   batch_shape=None,):  
    """Make Datasets object from tokenized tfds dataset."""
    # For this test, only support sentencepiece vocab
    from learned_optimization.tasks.datasets.language import get_32k_sentence_piece_vocab
    if vocab == 'bytes':
        raise NotImplementedError('bytes vocab not supported in this test')
    elif vocab == 'sentencepiece':
        vocab = get_32k_sentence_piece_vocab()
    else:
        raise ValueError(f'Unknown vocab type {vocab}')
    
    if has_test:
        splits = ['train[2%:100%]', 'train[0%:1%]', 'train[1%:2%]', 'test']
    else:
        splits = ['train[3%:100%]', 'train[0%:1%]', 'train[1%:2%]', 'train[2%:3%]']

    assert len(splits) == len(prefetch_batches), 'number of splits and prefetch_batches should be the same'
    assert len(splits) == len(batch_size), 'number of splits and batch_size should be the same'
    prefetch_batches = {splits[i]:prefetch_batches[i] for i in range(len(splits))}
    batch_size = {splits[i]:batch_size[i] for i in range(len(splits))}

    def make(split):
        def iterator_fn():
            it = _load(name=tfds_datasetname, 
                        tokenizer=vocab, 
                        batch_size=batch_size[split], 
                        sequence_length=sequence_length, 
                        split=split, 
                        label_sharding=label_sharding, 
                        image_sharding=image_sharding,
                        prefetch_batches=prefetch_batches[split],
                        device=jax.devices('gpu')[jax.process_index()],
                        batch_shape=batch_shape if len(batch_shape) > 1 else (batch_size[split],)  )
            return iter(it)
        return ThreadSafeIterator(LazyIterator(iterator_fn))

    train, inner_valid, outer_valid, test = [make(split) for split in splits]
    abstract_batch = {
        'image': jax.core.ShapedArray((1, sequence_length), jnp.int32),
        'label': jax.core.ShapedArray((1, sequence_length), jnp.int32),
    }
    return Datasets(
        train=train,
        inner_valid=inner_valid,
        outer_valid=outer_valid,
        test=test,
        extra_info={
            'vocab_size': vocab.vocab_size,
            'vocab': vocab,
            'name':f'lm1b-s{sequence_length}-{vocab}'
        },
        abstract_batch=abstract_batch)
# --- End copy ---

# Minimal test for loading lm1b dataset and sampling a batch

def test_lm1b_sample():
    # Settings for the dataset
    batch_size = [2, 1, 1, 1]  # train, inner_valid, outer_valid, test
    sequence_length = 64
    prefetch_batches = [1, 1, 1, 1]
    vocab = 'sentencepiece'
    use_localsgd_batches = False
    batch_shape = (2,)  # Set to match the train batch size

    # Create the dataset object
    ds = _make_datasets(
        tfds_datasetname='lm1b',
        use_localsgd_batches=use_localsgd_batches,
        vocab=vocab,
        batch_size=batch_size,
        sequence_length=sequence_length,
        prefetch_batches=prefetch_batches,
        batch_shape=batch_shape,
    )

    # Get an iterator for the training split
    train_iter = iter(ds.train)
    batch = next(train_iter)

    print('Sampled batch:')
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, type={type(v)})")

    exit(0)

if __name__ == "__main__":
    test_lm1b_sample() 