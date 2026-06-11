"""
The following file defines tasks.

Tasks follow the following naming conventions:
MODEL_DATASET

for language modelling:
    DATASET = {dataset name}-s{sequence length}-v{vocab size}

for image datasets:
    DATASET = {dataset name}-{HxWxC}

for MLPs:
    MODEL = {model name}-w{width}-d{depth}

"""

import functools
import gin
import h5py
import haiku as hk
import io
import jax
import jax.numpy as jnp
import ml_collections
import multiprocessing
import numpy as onp
import os
import seqio
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import copy
import warnings
from flax.training import prefetch_iterator
from jax import core
from learned_optimization import profile
from learned_optimization.tasks import base as base_tasks
from learned_optimization.tasks import resnet
from learned_optimization.tasks import transformer
from learned_optimization.tasks.datasets import base
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.datasets import language
from learned_optimization.tasks.datasets.base import Datasets, ThreadSafeIterator, LazyIterator
from learned_optimization.tasks.datasets.language import get_32k_sentence_piece_vocab
from learned_optimization.tasks.fixed.conv import _ConvTask, _cross_entropy_pool_loss
from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask
from learned_optimization.tasks.fixed.transformer_lm import _TransformerTask
from learned_optimization.tasks.resnet import ResNet
from PIL import Image



from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple
from typing import Tuple
from vit_jax import models_vit

import chex
import learned_optimization
from helpers import cast_to_bf16, print_rank_0

from custom_tasks import (
  _ResMLPImageTask,
  _MuDepthMLPImageTask,
  _MuMLPImageTask,
  _MuResMLPImageTask,
  _MuResnetTaskDataset,
  _MuTransformerTask,
  MuTransformerMoETask,
  MuVisionTransformerTask,
  # MuMoeMlpImageTask,
)
from fineweb_datasets import make_fineweb_datasets
from batched_image_augmentations import aug_transform
import random
Batch = Any
Params = Any

def _image_map_fn(cfg: Mapping[str, Any], batch: Batch) -> Batch:
  """Apply transformations + data aug to batch of data."""
  # batch is the entire tensor, with shape:
  # [batchsize, img width, img height, channels]
  batch = {k: v for k, v in batch.items()}
  if tuple(batch["image"].shape[1:3]) != cfg["image_size"]:
    batch["image"] = tf.image.resize(batch["image"], cfg["image_size"])

  if cfg["stack_channels"] != 1:
    assert batch["image"].shape[3] == 1, batch["image"].shape
    batch["image"] = tf.tile(batch["image"], (1, 1, 1, cfg["stack_channels"]))

  if cfg["aug_flip_left_right"]:
    batch["image"] = tf.image.random_flip_left_right(batch["image"])

  if cfg["aug_flip_up_down"]:
    batch["image"] = tf.image.random_flip_up_down(batch["image"])

  if cfg["normalize_mean"] is None:
    batch["image"] = tf.cast(batch["image"], tf.float32) / 255.
  else:
    assert cfg["normalize_std"] is not None
    image = tf.cast(batch["image"], tf.float32)
    image -= tf.constant(
        cfg["normalize_mean"], shape=[1, 1, 1, 3], dtype=image.dtype)
    batch["image"] = image / tf.constant(
        cfg["normalize_std"], shape=[1, 1, 1, 3], dtype=image.dtype)

  if cfg["convert_to_black_and_white"]:
    batch["image"] = tf.reduce_mean(batch["image"], axis=3, keepdims=True)

  batch["label"] = tf.cast(batch["label"], tf.int32)
  return hk.data_structures.to_immutable_dict({
      "image": batch["image"],
      "label": batch["label"]
  })


class _ConvTask(base_tasks.Task):
  """Helper class to construct tasks with different configs."""

  def __init__(self, datasets, base_model_fn, with_state=False):
    super().__init__()
    self._mod = hk.transform_with_state(base_model_fn)
    self.datasets = datasets
    self._with_state = with_state

  def init(self, key) -> Params:
    params, unused_state = self.init_with_state(key)
    return params

  def init_with_state(self, key):
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._mod.init(key, batch)

  def loss(self, params, key, data):
    loss, _ = self.loss_with_state(params, None, key, data)
    return loss

  def loss_with_state(self, params, state, key, data):
    loss, state, _ = self.loss_with_state_and_aux(params, state, key, data)
    return loss, state

  def loss_with_state_and_aux(self, params, state, key, data):
    loss, state = self._mod.apply(params, state, key, data)
    return loss, state, {}

  def normalizer(self, loss):
    return jnp.clip(loss, 0,
                    1.5 * jnp.log(self.datasets.extra_info["num_classes"]))



class VisionTransformerTask(base_tasks.Task):
  """Vision Transformer task."""

  def __init__(self, datasets, cfg):
    num_c = datasets.extra_info["num_classes"]
    self.flax_module = models_vit.VisionTransformer(num_classes=num_c, **cfg)
    self.datasets = datasets

  def init(self, key: chex.PRNGKey):
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    # print(jax.tree_util.tree_map(lambda x: x.shape, batch))
    # exit(0)
    return self.flax_module.init({
        "params": key,
        "dropout": key
    },
        batch["image"],
        train=True)


  @functools.partial(jax.jit, static_argnums=(0,))
  def loss(self, params, key, data):
    logits = self.flax_module.apply(
        params, data["image"], train=True, rngs={"dropout": key})
    labels_onehot = jax.nn.one_hot(data["label"], logits.shape[1])
    loss_vec = base_tasks.softmax_cross_entropy(logits=logits, labels=labels_onehot)
    return jnp.mean(loss_vec)

  
  @functools.partial(jax.jit, static_argnums=(0,))
  def loss_with_state(self, params, state, key, data):
    logits = self.flax_module.apply(
        params, data["image"], train=True, rngs={"dropout": key})
    labels_onehot = jax.nn.one_hot(data["label"], logits.shape[1])
    loss_vec = base_tasks.softmax_cross_entropy(logits=logits, labels=labels_onehot)
    return jnp.mean(loss_vec), state

  @functools.partial(jax.jit, static_argnums=(0,))
  def loss_and_accuracy(self, params, key, data):  # pytype: disable=signature-mismatch  # jax-ndarray
    num_classes = self.datasets.extra_info["num_classes"]

    logits = self.flax_module.apply(params, data["image"], train=False, rngs={"dropout": key})
    
    # Calculate the loss as before
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base_tasks.softmax_cross_entropy(logits=logits, labels=labels)
    loss = jnp.mean(vec_loss)
    
    # Calculate the accuracy
    predictions = jnp.argmax(logits, axis=-1)
    actual = data["label"]
    correct_predictions = predictions == actual
    accuracy = jnp.mean(correct_predictions.astype(jnp.float32))
    
    return loss, accuracy

  @functools.partial(jax.jit, static_argnums=(0,))
  def loss_and_accuracy_with_state(self, params, state, key, data):
    loss, acc = self.loss_and_accuracy(params, key, data)
    return loss, acc

  def normalizer(self, loss):
    max_class = onp.log(2 * self.datasets.extra_info["num_classes"])
    loss = jnp.nan_to_num(
        loss, nan=max_class, neginf=max_class, posinf=max_class)
    # shift to [0, 10] then clip.
    loss = 10 * (loss / max_class)
    return jnp.clip(loss, 0, 10)



class _TransformerTask(base_tasks.Task):
  """Tranformer from a dictionary configuration."""

  def __init__(self, datasets, cfg: Mapping[str, Any], name: str = '__TransformerTask'):
    self.datasets = datasets
    self._cfg = cfg
    self._net = hk.transform(self._hk_forward)
    self._name = name

  @property
  def name(self):
    return self._name

  def _hk_forward(self, batch):
    vocab_size = self.datasets.extra_info['vocab_size']
    mod = transformer.Transformer(
        num_heads=self._cfg['num_heads'],
        num_layers=self._cfg['num_layers'],
        d_model=self._cfg['d_model'],
        dropout_rate=self._cfg['dropout_rate'],
        vocab_size=vocab_size)
    mask = (batch['image'] != 0)
    logits = mod(batch['image'], mask=mask, is_training=True)
    loss = base_tasks.softmax_cross_entropy(
        logits=logits, labels=jax.nn.one_hot(batch['label'], vocab_size))
    return jnp.sum(loss * mask) / jnp.sum(mask)

  def init(self, key: chex.PRNGKey) -> base_tasks.Params:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._net.init(key, batch)
  

  def loss(self, params, key, data):
    return self._net.apply(params, key, data)
  



def _crop_or_pad(value, size, pad_token):
  """Either crop or pad value to be of size size."""
  val_size = tf.size(value)
  pad = lambda: tf.pad(
      value, [[0, size - val_size]],
      'CONSTANT',
      constant_values=pad_token)
  return tf.cond(val_size < size, pad, lambda: value[:size])



# #TODO fix hacky label check
# sharding = jax.devices('gpu')[jax.process_index()]
# temp_batch_shape = batch_shape + x.shape[1:] if len(batch_shape) > 1 \
#                         else (batch_size[split],) + x.shape[1:]

# return jnp.reshape(jax.device_put(x[idxs], device=sharding), temp_batch_shape)

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

  # Shard the dataset by process index
  n_processes = jax.process_count()
  ds = ds.shard(num_shards=n_processes, index=jax.process_index())
  print(f"Sharding dataset by process index {jax.process_index()} of {n_processes}")

  crop_size = sequence_length + 1
  ds = ds.repeat()
  ds = ds.map(lambda x: tokenizer.encode_tf(x['text']))
  ds = ds.map(lambda t: _crop_or_pad(t, crop_size, pad_token=0))
  # Set seed for reproducible shuffling
  ds = ds.shuffle(batch_size * 10, seed=seed*jax.process_index())

  # Create the language modeling observation/target pairs and batch them up.
  def create_lm_obs_target(t):
    # return dict(obs=t[:-1], target=t[1:])
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

def _make_datasets(tfds_datasetname: str,
                   use_localsgd_batches: bool,
                   vocab: seqio.vocabularies.Vocabulary,
                   batch_size: int,
                   sequence_length: int,
                   has_test: bool = True,
                   prefetch_batches = [1,1,1,1],
                   label_sharding=None,
                   image_sharding=None,
                   batch_shape=None,
                   seed=42,
                   **unused_kwargs):
    """Make Datasets object from tokenized tfds dataset."""
    onp.random.seed(seed * jax.process_index())
    if vocab == 'bytes':
        vocab = get_bytes_vocab()
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
                        batch_shape=batch_shape if len(batch_shape) > 1 and split.startswith('train') else (batch_size[split],)  )
            return iter(it)

        return base.ThreadSafeIterator(base.LazyIterator(iterator_fn))

    train, inner_valid, outer_valid, test = [make(split) for split in splits]
    abstract_batch = {
        'image': jax.core.ShapedArray((1, sequence_length), jnp.int32),
        'label': jax.core.ShapedArray((1, sequence_length), jnp.int32),
    }
    return base.Datasets(
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


class Timer:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        print_rank_0(f"Executing {self.func.__name__} took {end_time - start_time:.4f} seconds.")
        return result


def process_batch(encoded_images):
    """Process a batch of encoded images into numpy arrays."""
    return [onp.array(Image.open(io.BytesIO(img_data)).convert('RGB')) for img_data in encoded_images]

class H5Data:
    _instance = None

    @Timer
    def __new__(cls, h5_path, num_workers=24):
        if cls._instance is None:
            print_rank_0("Creating the dataset instance")
            cls._instance = super(H5Data, cls).__new__(cls)
            
            # Read the encoded images and labels from the H5 file
            with h5py.File(h5_path, 'r') as file:
                encoded_images = file['encoded_images'][:]
                targets = file['targets'][:]
            
            # Determine the number of workers
            if num_workers is None:
                num_workers = multiprocessing.cpu_count()
            
            # Create batches of encoded images
            batch_size = len(encoded_images) // num_workers
            image_batches = [encoded_images[i:i + batch_size] for i in range(0, len(encoded_images), batch_size)]

            # Use multiprocessing to process the batches
            with multiprocessing.Pool(num_workers) as pool:
                image_arrays = pool.map(process_batch, image_batches)
            
            # Flatten the list of lists to a single list
            cls._instance.data = onp.array([img for sublist in image_arrays for img in sublist])
            cls._instance.labels = onp.squeeze(targets)

        return cls._instance


    
def parse_split(split_string, data_array, index_array):
    # Extract the range from the string after removing 'train[' and ']'
    range_part = split_string[len('train['):-1]

    # Split the range on ':'
    parts = range_part.split(':')
    num_samples = len(data_array)
    
    # Determine start index
    start = parts[0].strip()
    if start.endswith('%'):
        start_index = int(float(start.rstrip('%')) / 100 * num_samples)
    else:
        start_index = int(start) if start else 0

    # Determine end index
    end = parts[1].strip() if len(parts) > 1 else ''
    if end.endswith('%'):
        end_index = int(float(end.rstrip('%')) / 100 * num_samples)
    else:
        end_index = int(end) if end else num_samples

    # Return the appropriate slice of the data array
    return data_array[start_index:end_index], index_array[start_index:end_index]

    
class PreloadImageNetDatasetH5():
    
    def __init__(self, split, h5_path, num_workers):
        self.split = split
        self.h5_path = h5_path
        self.n_train = 1281167
        self.n_val = 50000
        self.n_test = 100000
        self.data = H5Data(h5_path=h5_path, num_workers=num_workers)
        
    def preload(self):
        im, lab = self.preload_helper()
        return {'image':im, 'label':lab}
    
    def preload_helper(self):
        if self.split.startswith('train'):
            s = self.split.split('train')[-1]
            if s == '':
                return H5Data._instance.data[:self.n_train], H5Data._instance.labels[:self.n_train]
            else:
                return parse_split(split_string=self.split, 
                                   data_array=H5Data._instance.data[:self.n_train], 
                                   index_array=H5Data._instance.labels[:self.n_train])
        elif self.split.lower() == 'validation':
            return H5Data._instance.data[self.n_train:self.n_train + self.n_val], H5Data._instance.labels[self.n_train:self.n_train + self.n_val]
        elif self.split.lower() == 'test':
            return H5Data._instance.data[self.n_train + self.n_val:], H5Data._instance.labels[self.n_train + self.n_val:]
        else:
            raise NotImplemented('not implemented for split'+str(self.split))
        

def _ensure_tfds_downloaded(datasetname):
  """Ensure TFDS dataset is downloaded. In multi-process mode, only rank 0
  downloads to avoid concurrent download races that cause JAX 'DEADLINE
  EXCEEDED' errors when ranks desynchronize."""
  from jax.experimental import multihost_utils
  rank = jax.process_index()
  world_size = jax.process_count()
  data_dir = os.environ.get('TFDS_DATA_DIR', None)
  builder = tfds.builder(datasetname, data_dir=data_dir)
  if not builder.data_path.exists():
    if world_size > 1:
      if rank == 0:
        print_rank_0(f"[DATA] Rank 0 downloading {datasetname}...")
        builder.download_and_prepare()
        print_rank_0(f"[DATA] Rank 0 finished downloading {datasetname}")
      multihost_utils.sync_global_devices(f'tfds_download_{datasetname}')
    else:
      builder.download_and_prepare()


@functools.lru_cache(None)
def _cached_tfds_load(datasetname, split, batch_size):
  assert batch_size == -1
  _ensure_tfds_downloaded(datasetname)
  return tfds.load(datasetname, split=split, batch_size=-1)

def preload_tfds_image_classification_datasets_2(
    datasetname: str,
    splits: Tuple[str, str, str, str],
    batch_size: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    stack_channels: int = 1,
    prefetch_batches: Tuple[int, int, int, int] = (20,1,1,1),
    normalize_mean: Optional[Tuple[int, int, int]] = None,
    normalize_std: Optional[Tuple[int, int, int]] = None,
    convert_to_black_and_white: Optional[bool] = False,
    batch_shape=None,
    label_sharding=None,
    image_sharding=None,
    use_localsgd_batches=False,
    augmentations=None,
    seed=42,
    image_dtype=jnp.float32,
) -> Datasets:
  """Load an image dataset with tfds by first loading into host ram.

  Args:
    datasetname: name of the dataset to be loaded with tfds.
    splits: tfds style splits for different subsets of data. (train,
      inner-valid, outer-valid, and test set)
    batch_size: batch size of iterators
    image_size: target size to resize images to.
    stack_channels: stack the channels in case of 1d outputs (e.g. mnist)
    prefetch_batches: number of batches to prefetch
    normalize_mean: mean RGB value to subtract off of images to normalize imgs
    normalize_std: std RGB of dataset to normalize imgs
    convert_to_black_and_white: conver a color image to black and white.

  Returns:
    A Datasets object containing data iterators.
  """
  onp.random.seed(seed * jax.process_index())
  assert len(splits) == len(prefetch_batches), 'number of splits and prefetch_batches should be the same'
  assert len(splits) == len(batch_size), 'number of splits and batch_size should be the same'
  prefetch_batches = {splits[i]:prefetch_batches[i] for i in range(len(splits))}
  batch_size = {splits[i]:batch_size[i] for i in range(len(splits))}

  cfg = {
      "batch_size": batch_size,
      "image_size": image_size,
      "stack_channels": stack_channels,
      "prefetch_batches": prefetch_batches,
      "aug_flip_left_right": False,
      "aug_flip_up_down": False,
      "normalize_mean": normalize_mean,
      "normalize_std": normalize_std,
      "convert_to_black_and_white": convert_to_black_and_white,
  }

  def make_python_iter(split: str) -> Iterator[Batch]:
    # load the entire dataset into memory
    with profile.Profile(f"tfds.load({datasetname})"):
      dataset = _cached_tfds_load(datasetname, split=split, batch_size=-1)
    data = tfds.as_numpy(_image_map_fn(cfg, dataset))

    print_rank_0(jax.tree_util.tree_map(lambda x:x.shape if type(x) != int else x, data))
    def generator_fn():

      def iter_fn():
        
        if batch_size[split] > data["image"].shape[0]:
          warnings.warn('For {} split {}, batch size ({}) is larger than dataset size ({}). Possible'
                  ' duplicate samples in batch/'.format(datasetname,split,batch_size[split],data["image"].shape[0]), Warning)
          batches = 1
          idx = onp.arange(batch_size[split]) % data["image"].shape[0]
        else:
          batches = data["image"].shape[0] // batch_size[split]
          idx = onp.arange(data["image"].shape[0])


        if 'train' in split:
            print_rank_0('using infinite iterator for training')
            aug_key = jax.random.PRNGKey(random.randint(0, 2**32 - 1))
            #infinite iterator
            while True:
                idxs = onp.random.permutation(idx)[:batch_size[split]]
                aug_key, subkey = jax.random.split(aug_key)

                if True : #not use_localsgd_batches: #jax.process_count() >= 1:

                    def index_into(idxs, x):
                        #TODO fix hacky label check
                        sharding = jax.devices('gpu')[jax.process_index()]
                        temp_batch_shape = batch_shape + x.shape[1:] if len(batch_shape) > 1 \
                                                else (batch_size[split],) + x.shape[1:]

                        return jnp.reshape(jax.device_put(x[idxs], device=sharding), temp_batch_shape)
                else:


                    def index_into(idxs, x):
                        #TODO fix hacky label check
                        sharding = image_sharding if len(x.shape) > 1 else label_sharding
                        # print(sharding)
                        temp_batch_shape = batch_shape + x.shape[1:] if len(batch_shape) > 1 \
                                                else (batch_size[split],) + x.shape[1:]

                        return jnp.reshape(jax.device_put(x[idxs], device=sharding), temp_batch_shape)


                batch = jax.tree_util.tree_map(functools.partial(index_into, idxs), data)
                if augmentations is not None:
                    aug_image, aug_label = aug_transform(
                        batch["image"], batch["label"], augmentations, subkey)
                    batch = {"image": aug_image, "label": aug_label}
                batch = {**batch, "image": batch["image"].astype(image_dtype)}
                yield batch
        else:
            while True:
                # every epoch shuffle indicies
                onp.random.shuffle(idx)

                # if split != 'validation':
                #   print('\nshuffled idx: ', idx[:10] ,split)
                    
                for bi in range(0, batches):
                    idxs = idx[bi * batch_size[split]:(bi + 1) * batch_size[split]]
                    if not use_localsgd_batches:
                        # print(jax.process_count())
                        def index_into(idxs, x):
                            device = jax.devices('gpu')[jax.process_index()]
                            # return x[idxs]
                            #TODO fix hacky label check
                            if len(batch_shape) > 1:
                                return jnp.reshape(jax.device_put(x[idxs], device),
                                                batch_shape + x.shape[1:] )
                            else:
                                return jnp.reshape(jax.device_put(x[idxs],  device),
                                                (batch_size[split],) + x.shape[1:] )
                    else:
                        # print(jax.process_count())
                        # exit(0)
                        def index_into(idxs, x):
                            # return x[idxs]
                            #TODO fix hacky label check
                            if len(batch_shape) > 1:
                                return jnp.reshape(jax.device_put(x[idxs], image_sharding if len(x.shape) > 1 else label_sharding),
                                                batch_shape + x.shape[1:] )
                            else:
                                return jnp.reshape(jax.device_put(x[idxs], image_sharding if len(x.shape) > 1 else label_sharding),
                                                (batch_size[split],) + x.shape[1:] )


                    batch = jax.tree_util.tree_map(
                        functools.partial(index_into, idxs), data)
                    batch = {**batch, "image": batch["image"].astype(image_dtype)}
                    yield batch
      
      return prefetch_iterator.PrefetchIterator(iter_fn(), prefetch_batches[split])
      
    return ThreadSafeIterator(LazyIterator(generator_fn))

  builder = tfds.builder(datasetname)
  num_classes = builder.info.features["label"].num_classes

  if stack_channels == 1:
    output_channel = builder.info.features["image"].shape[-1:]
  else:
    output_channel = (stack_channels,)

  if convert_to_black_and_white:
    output_channel = (1,)

  # abstract_batch = {
  #     "image":
  #         jax.core.ShapedArray(
  #             (batch_size[splits[0]],) + image_size + output_channel, dtype=jnp.float32),
  #     "label":
  #         jax.core.ShapedArray((batch_size[splits[0]],), dtype=jnp.int32)
  # }
  abstract_batch = {
      "image":
          jax.core.ShapedArray(
              (batch_size[splits[0]],) + image_size + output_channel, dtype=image_dtype),
      "label":
          jax.core.ShapedArray((batch_size[splits[0]],), dtype=jnp.int32)
  }

  return Datasets(
      *[make_python_iter(split) for split in splits],
      extra_info={"num_classes": num_classes, 'name':datasetname},
      abstract_batch=abstract_batch)

            
@base.dataset_lru_cache
@gin.configurable
def imagenet_orig_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (64, 64),
    prefetch_batches=50,
    data_fraction=1.0,
    **kwargs,
) -> base.Datasets:
    perc = max(1, int(80 * data_fraction))
    splits = (f"train[0:{perc}%]", "train[80%:90%]", "train[90%:]", "validation")
    # return base.tfds_image_classification_datasets(
    return preload_tfds_image_classification_datasets_2(
        datasetname="imagenet_resized",
        splits=splits,
        batch_size=batch_size,
        image_size=image_size,
        stack_channels=1,
        prefetch_batches=prefetch_batches,
        # shuffle_buffer_size=10000,
        normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
        convert_to_black_and_white=False,
        # cache=True,
        **kwargs,
    )

def normalize_images(images: jnp.ndarray, mean: tuple[float, float, float], std: tuple[float, float, float]) -> jnp.ndarray:
    """
    Normalize images using the given per-channel mean and standard deviation.
    
    Args:
        images: jnp.ndarray of shape (batch_size, height, width, channels), dtype=jnp.float32
        mean: Tuple of means for each channel (R, G, B).
        std: Tuple of standard deviations for each channel (R, G, B).

    Returns:
        Normalized images as a JAX array.
    """
    mean = jnp.array(mean).reshape(1, 1, 1, 3)  # Reshape for broadcasting
    std = jnp.array(std).reshape(1, 1, 1, 3)  # Reshape for broadcasting

    return (images - mean) / std



    

def generate_random_data(num_samples=10000, num_classes=1000, img_shape=(64, 64, 3), seed=42):
    onp.random.seed(seed)
    
    # Generate class labels
    labels = onp.random.randint(0, num_classes, size=num_samples).astype(onp.int64)
    labels = jnp.array(labels)
    
    # Generate class-specific means (each class has a unique mean)
    class_means = onp.linspace(0, 255, num_classes).reshape(-1, 1, 1, 1)
    
    # Generate images with the corresponding mean for each class
    images = onp.zeros((num_samples, *img_shape), dtype=onp.uint8)
    for i in range(num_samples):
        class_mean = class_means[labels[i]]
        images[i] = onp.clip(onp.random.normal(loc=class_mean, scale=30, size=img_shape), 0, 255).astype(onp.float32)
    
    images = jnp.array(images)
    
    return labels, images


def custom_preload_tfds_image_classification_datasets(
    datasetname: str,
    h5_path: str,
    splits: Tuple[str, str, str, str],
    batch_size: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    stack_channels: int = 1,
    prefetch_batches: Tuple[int, int, int, int] = (20,1,1,1),
    normalize_mean: Optional[Tuple[int, int, int]] = None,
    normalize_std: Optional[Tuple[int, int, int]] = None,
    convert_to_black_and_white: Optional[bool] = False,
    batch_shape=None,
    label_sharding=None,
    image_sharding=None,
    use_localsgd_batches=False,
    augmentations=None,
    seed=42,
    image_dtype=jnp.float32,
) -> Datasets:
  """Load an image dataset with tfds by first loading into host ram.

  Args:
    datasetname: name of the dataset to be loaded with tfds.
    splits: tfds style splits for different subsets of data. (train,
      inner-valid, outer-valid, and test set)
    batch_size: batch size of iterators
    image_size: target size to resize images to.
    stack_channels: stack the channels in case of 1d outputs (e.g. mnist)
    prefetch_batches: number of batches to prefetch
    normalize_mean: mean RGB value to subtract off of images to normalize imgs
    normalize_std: std RGB of dataset to normalize imgs
    convert_to_black_and_white: conver a color image to black and white.

  Returns:
    A Datasets object containing data iterators.
  """
  onp.random.seed(seed * jax.process_index())
  assert len(splits) == len(prefetch_batches), 'number of splits and prefetch_batches should be the same'
  assert len(splits) == len(batch_size), 'number of splits and batch_size should be the same'
  prefetch_batches = {splits[i]:prefetch_batches[i] for i in range(len(splits))}
  batch_size = {splits[i]:batch_size[i] for i in range(len(splits))}

  def make_python_iter(split: str) -> Iterator[Batch]:
    if datasetname == 'imagenet_resized':
        # load the entire dataset into memory
        with profile.Profile(f"tfds.load({datasetname})"):
            dataset = PreloadImageNetDatasetH5(split, h5_path=h5_path, num_workers=24)
            dataset = dataset.preload()
        data = dataset
        # data = jax.tree_util.tree_map(lambda x: jnp.array(x), data)
        print_rank_0(jax.tree_util.tree_map(lambda x:x.shape if type(x) != int else x, data),
            "prefetch_batches:", prefetch_batches[split])
    elif datasetname == "random":
        labels, images =  generate_random_data(num_samples=10000, num_classes=1000, img_shape=image_size + (3,), seed=42)
        data = {"image":images,
                "label":labels}
    else:
      raise ValueError(f"Unknown dataset: {datasetname}")


    # Batched augmentation function using @batched_image_augmentations.py.
    # augmentations: list of string names, determining order; e.g. ['random_flip', 'random_crop', ...]


    def generator_fn():
        key = jax.random.PRNGKey(random.randint(0, 2**32 - 1))
        onp.random.seed(key * jax.process_index())
        def iter_fn(key):
            if batch_size[split] > data["image"].shape[0]:
                warnings.warn(f"For {datasetname} split {split}, batch size ({batch_size[split]}) is larger than dataset size ({data['image'].shape[0]}). Possible duplicate samples in batch.", Warning)
                batches = 1
                idx = onp.arange(batch_size[split]) % data["image"].shape[0]
            else:
                batches = data["image"].shape[0] // batch_size[split]
                idx = onp.arange(data["image"].shape[0])

            def process_batch(idxs, subkey, train=True, use_aug=False):
                def index_into(x, name):
                    if use_localsgd_batches:
                        sharding = image_sharding if name == "image" else label_sharding
                    else:
                        sharding = jax.devices('gpu')[jax.process_index()]
                    temp_batch_shape = batch_shape + x.shape[1:] if (len(batch_shape) > 1 and train) else (batch_size[split],) + x.shape[1:]
                    return jnp.reshape(jax.device_put(x[idxs], device=sharding), temp_batch_shape)

                image_batch = index_into(data["image"], "image")
                label_batch = index_into(data["label"], "label")
                if use_aug:
                    image_batch, label_batch = aug_transform(image_batch, label_batch, augmentations, subkey)
                
                if normalize_mean is not None and normalize_std is not None:
                    image_batch = normalize_images(image_batch, normalize_mean, normalize_std)

                image_batch = image_batch.astype(image_dtype)
                return {"image": image_batch, "label": label_batch}

            if 'train' in split:
                print(f'Using infinite iterator for training split {split}')
                # while True:
                #     onp.random.shuffle(idx)
                #     for bi in range(0, batches):
                #         key, subkey = jax.random.split(key)
                #         idxs = idx[bi * batch_size[split]:(bi + 1) * batch_size[split]]
                #         yield process_batch(idxs, subkey, use_aug=use_aug)

                while True:
                    key, subkey = jax.random.split(key)
                    idxs = onp.random.permutation(idx)[:batch_size[split]]
                    yield process_batch(idxs, subkey, use_aug=augmentations is not None)
            else:
                # print(f'Using epoch-based iterator for testing/validation split {split}')
                while True:
                    onp.random.shuffle(idx)
                    for bi in range(0, batches):
                        key, subkey = jax.random.split(key)
                        idxs = idx[bi * batch_size[split]:(bi + 1) * batch_size[split]]
                        yield process_batch(idxs, subkey, train=False, use_aug=False)

        return prefetch_iterator.PrefetchIterator(iter_fn(key), prefetch_batches[split])

    return ThreadSafeIterator(LazyIterator(generator_fn))
      

  builder = tfds.builder("imagenet_resized")
  num_classes = builder.info.features["label"].num_classes

  if stack_channels == 1:
    output_channel = builder.info.features["image"].shape[-1:]
  else:
    output_channel = (stack_channels,)

  if convert_to_black_and_white:
    output_channel = (1,)

  abstract_batch = {
      "image":
          jax.core.ShapedArray(
              (batch_size[splits[0]],) + image_size + output_channel, dtype=image_dtype),
      "label":
          jax.core.ShapedArray((batch_size[splits[0]],), dtype=jnp.int32)
  }
  return Datasets(
      *[make_python_iter(split) for split in splits],
      extra_info={"num_classes": num_classes, 'name':datasetname},
      abstract_batch=abstract_batch)


@base.dataset_lru_cache
@gin.configurable
def imagenet_dataset(
    batch_size: int,
    image_size: Tuple[int, int] = (64, 64),
    prefetch_batches=[20,1,1,1],
    data_fraction=1.0,
    augmentations=None,
    **kwargs,
) -> base.Datasets:

    assert image_size in [(32,32),(64,64),(128,128),(224,224)]
    h5_path = os.path.join(os.environ["TFDS_DATA_DIR"],'imagenet_{}x{}x3_JPEG.h5'.format(image_size[0],image_size[1]))
    perc = max(1, int(80 * data_fraction))
    perc=100
    splits = (f"train[0:{perc}%]", "train[97%:98%]", "train[98:99%]", "validation")
    # splits = (f"train[0:50%]", "train[50%:99%]", "train[50%:99%]", "validation")
    return custom_preload_tfds_image_classification_datasets(
        datasetname="imagenet_resized",
        h5_path=h5_path,
        splits=splits,
        batch_size=batch_size,
        image_size=image_size,
        stack_channels=1,
        prefetch_batches=prefetch_batches,
        normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
        convert_to_black_and_white=False,
        augmentations=augmentations,
        **kwargs,
    )


def imagenet_50_50(
    batch_size: int,
    image_size: Tuple[int, int] = (64, 64),
    prefetch_batches=[20,1,1,1],
    data_fraction=1.0,
    **kwargs,
) -> base.Datasets:

    assert image_size in [(32,32),(64,64),(128,128),(224,224)]
    h5_path = os.path.join(os.environ["TFDS_DATA_DIR"],'imagenet_{}x{}x3_JPEG.h5'.format(image_size[0],image_size[1]))
    perc = max(1, int(80 * data_fraction))
    # splits = (f"train[0:{perc}%]", "train[97%:98%]", "train[98:99%]", "validation")
    splits = (f"train[0:50%]", "train[50%:99%]", "train[50%:99%]", "validation")
    return custom_preload_tfds_image_classification_datasets(
        datasetname="imagenet_resized",
        h5_path=h5_path,
        splits=splits,
        batch_size=batch_size,
        image_size=image_size,
        stack_channels=1,
        prefetch_batches=prefetch_batches,
        # shuffle_buffer_size=10000,
        normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
        convert_to_black_and_white=False,
        # cache=True,
        **kwargs,
    )




@base.dataset_lru_cache
@gin.configurable
def random_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (64, 64),
    prefetch_batches=[20,1,1,1],
    data_fraction=1.0,
    **kwargs,
) -> base.Datasets:

    assert image_size in [(32,32),(64,64),(128,128),(224,224)]
    h5_path = os.path.join(os.environ["TFDS_DATA_DIR"],'imagenet_{}x{}x3_JPEG.h5'.format(image_size[0],image_size[1]))
    perc = max(1, int(80 * data_fraction))
    splits = (f"train[0:{perc}%]", "train[97%:98%]", "train[98:99%]", "validation")
    return custom_preload_tfds_image_classification_datasets(
        datasetname="random",
        h5_path=h5_path,
        splits=splits,
        batch_size=batch_size,
        image_size=image_size,
        stack_channels=1,
        prefetch_batches=prefetch_batches,
        # shuffle_buffer_size=10000,
        normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
        convert_to_black_and_white=False,
        # cache=True,
        **kwargs,
    )


# @gin.configurable
# def mlp128x128_fastinet_32(batch_size):
#     """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
#     h5_path = "/mnt/raid0/imagenet_hdf5/ilsvrc2012.hdf5"
#     datasets = fast_imagenet_datasets(h5_path, 
#         batch_size, 
#         workers=48, 
#         distributed=False,
#         image_size=(32,32,),
#         output_channel=(3,)
#     )
#     return _MLPImageTask(datasets, [128, 128])



# @base.dataset_lru_cache
# @gin.configurable
# def imagenet_dataset(
#     batch_size: int,
#     image_size: Tuple[int, int] = (64, 64),
#     # prefetch_batches=[20,1,1,1],
#     data_fraction=1.0,
#     **kwargs,
# ) -> base.Datasets:
#     perc = max(1, int(80 * data_fraction))
#     splits = (f"train[0:{perc}%]", "train[80%:90%]", "train[90%:]", "validation")
#     return preload_tfds_image_classification_datasets_2(
#         datasetname="imagenet_resized",
#         splits=splits,
#         batch_size=batch_size,
#         image_size=image_size,
#         stack_channels=1,
#         # prefetch_batches=prefetch_batches,
#         # shuffle_buffer_size=10000,
#         normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
#         normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
#         convert_to_black_and_white=False,
#         # cache=True,
#         **kwargs,
#     )
import haiku as hk
class _ResnetTaskDataset(learned_optimization.tasks.base.Task):
  """Tranformer from a dictionary configuration."""

  def __init__(self, datasets, cfg: Mapping[str, Any], name: str = '__Resnet'):
    self.datasets = datasets
    self._cfg = cfg
    self._net = hk.transform_with_state(self._hk_forward)
    self._name = name

  @property
  def name(self):
    return self._name

  def _hk_forward(self, batch):
    args = [
        'blocks_per_group', 'use_projection', 'channels_per_group',
        'initial_conv_kernel_size', 'initial_conv_stride', 'max_pool',
        'resnet_v2'
    ]
    num_classes = self.datasets.extra_info['num_classes']
    mod = resnet.ResNet(
        num_classes=num_classes, **{k: self._cfg[k] for k in args})
    logits = mod(batch['image'], is_training=True)
    loss = learned_optimization.tasks.base.softmax_cross_entropy(
        logits=logits, labels=jax.nn.one_hot(batch['label'], num_classes))
    # return jnp.mean(loss)
    return jnp.mean(loss), logits

  def init_with_state(self, key: chex.PRNGKey) -> learned_optimization.tasks.base.Params:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._net.init(key, batch)

  def init(self, key: chex.PRNGKey) -> learned_optimization.tasks.base.Params:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    params, state = self._net.init(key, batch)
    return params


    #   def loss_with_state(self, params, state, key, data):
    #     # Extract only the scalar loss for gradient computation
    #     loss, _ = self._net.apply(params, state, key, data)
        # return loss, state


  @functools.partial(jax.jit, static_argnums=(0,))
  def loss_with_state(self, params, state, key, data):
    loss, state, _ = self.loss_with_state_and_aux(params, state, key, data)
    return loss, state


  @functools.partial(jax.jit, static_argnums=(0,))
  def loss(self, params, state, key, data):
    loss, state, _ = self.loss_with_state_and_aux(params, state, key, data)
    return loss


  @functools.partial(jax.jit, static_argnums=(0,))
  def loss_with_state_and_aux(self, params, state, key, data):
    (loss, logits), state = self._net.apply(params, state, key, data)
    return loss, state, {}



#   @functools.partial(jax.jit, static_argnums=(0,))
#   def loss_with_state(self, params, state, key, data):
#     loss, state, _ = self.loss_with_state_and_aux(params, state, key, data)
#     return loss, state

# #   @functools.partial(jax.jit, static_argnums=(0,))
#   def loss(self, params, state, key, data):
#     loss, state, _ = self.loss_with_state_and_aux(params, state, key, data)
#     return loss

# #   @functools.partial(jax.jit, static_argnums=(0,))
#   def loss_with_state_and_aux(self, params, state, key, data):
#     logits, loss = self._net.apply(params, state, key, data)
#     return loss, state, {}

  @functools.partial(jax.jit, static_argnums=(0,))
  def loss_and_accuracy_with_state(self, params, state, key, data):
    (loss, logits), state = self._net.apply(params, state, key, data)
    predictions = jnp.argmax(logits, axis=-1)
    labels = data['label']
    accuracy = jnp.mean(predictions == labels)
    return loss, accuracy

def func_create_func(task_fun, ds_args, model_args):
    ds_fun = ds_args['fun']
    # model_fun = model_args['fun']
    dataset = ds_fun(*ds_args['args'], **ds_args['kwargs'])
    print_rank_0('[func_create_func model_args]',model_args)
    print_rank_0('[func_create_func ds_args]',ds_args)
    print_rank_0('[func_create_func dataset]',dataset)
    return task_fun(dataset, **model_args)


def add_MLP_tasks(tasks, image_datasets, widths, depths, mup_muls, log_activations, depth_mup_multipliers):
    for k,ds in image_datasets.items():
        for mlp_width in widths:
            for mlp_depth in depths:
                tasks['mlp-w{}-d{}_{}'.format(mlp_width,mlp_depth,k)] = functools.partial(func_create_func, _MLPImageTask, ds, 
                                                                                            dict(hidden_sizes=[mlp_width] * mlp_depth, log_activations=log_activations,))

                tasks['depthmlp-w{}-d{}_{}'.format(mlp_width,mlp_depth,k)] = functools.partial(func_create_func, _ResMLPImageTask, ds, 
                                                                                            dict(hidden_sizes=[mlp_width] * mlp_depth, log_activations=log_activations,))                                                                            

                                                                                            
                tasks['muresmlp-w{}-d{}_{}'.format(mlp_width,mlp_depth,k)] = functools.partial(func_create_func, 
                                                                                                _MuResMLPImageTask, 
                                                                                                ds,
                                                                                                dict(hidden_sizes=[mlp_width] * mlp_depth,
                                                                                                    log_activations=log_activations,
                                                                                                    mup_multipliers=mup_muls))



                tasks['mumlp-w{}-d{}_{}'.format(mlp_width,mlp_depth,k)] = functools.partial(func_create_func, 
                                                                                            _MuMLPImageTask, 
                                                                                            ds,
                                                                                            dict(hidden_sizes=[mlp_width] * mlp_depth,
                                                                                                 log_activations=log_activations,
                                                                                                 mup_multipliers=mup_muls))

                temp = {}
                temp.update(mup_muls)
                temp.update(depth_mup_multipliers)
                tasks['mudepthmlp-w{}-d{}_{}'.format(mlp_width,mlp_depth,k)] = functools.partial(func_create_func, 
                                                                                                _MuDepthMLPImageTask, 
                                                                                                ds,
                                                                                                dict(hidden_sizes=[mlp_width] * mlp_depth,
                                                                                                     log_activations=log_activations,
                                                                                                      mup_multipliers=temp))




def add_sweepable_MLP_tasks(tasks, image_datasets, widths, depths, mup_muls):
    for k,ds in image_datasets.items():
        for mlp_width in widths:
            for mlp_depth in depths:
                for iname,input_mult in [('2**2',2**2)]:
                    for oname,output_mult in [('2**5',2**5)]:
                        for hname,hidden_mult in [('2**1',2**1),
                                                  ('2**-2',2**-2),
                                                  ('2**-4',2**-4),
                                                  ('2**-3',2**-3),
                                                  ('2**5',2**-5),
                                                  ('2**6',2**-6),
                                                  ('2**8',2**-8)]:
                        
                            tasks['mumlp-w{}-d{}-i{}-o{}-h{}_{}'.format(mlp_width,mlp_depth,iname,oname,hname,k)] = functools.partial(func_create_func, 
                                                                                                                                        _MuMLPImageTask, 
                                                                                                                                        ds,
                                                                                                                                        dict(hidden_sizes=[mlp_width] * mlp_depth,
                                                                                                                                        output_mult=output_mult,
                                                                                                                                        input_mult=input_mult,
                                                                                                                                        hidden_mult=hidden_mult))
                
def add_transformer_lm_tasks(tasks, lm_datasets, widths, depths, mup_muls):
    for k,ds in lm_datasets.items():
        for w,heads in widths:
            for d in depths:
                cfg = {
                    "num_heads": heads,
                    "d_model": w,
                    "num_layers": d,
                    "dropout_rate": 0.1,
                }
                name = 'transformer-w{}-d{}_{}'.format(w,d,k)
                tasks[name] = functools.partial(func_create_func, 
                                                _TransformerTask, 
                                                ds,
                                                dict(cfg=cfg,name=name))
                
                name = 'mutransformer-w{}-d{}_{}'.format(w,d,k)
                tasks[name] = functools.partial(func_create_func, 
                                                _MuTransformerTask, 
                                                ds,
                                                dict(cfg=cfg,
                                                     name=name,
                                                     mup_multipliers=mup_muls))

                # name = 'mutransformer-moe-w{}-d{}_{}'.format(w,d,k)
                # tasks[name] = functools.partial(func_create_func, 
                #                                 _MuTransformerMoETask, 
                #                                 ds,
                #                                 dict(cfg=cfg,
                #                                      name=name,
                #                                      mup_multipliers=mup_muls))

                
                name = 'mudtransformer-w{}-d{}_{}'.format(w,d,k)
                tasks[name] = functools.partial(func_create_func, 
                                                _MuTransformerTask, 
                                                ds,
                                                dict(cfg=cfg,
                                                     name=name,
                                                     use_mu_depth=True,
                                                     mup_multipliers=mup_muls))


def add_transformer_lm_tasks_with_head(tasks, lm_datasets, widths, depths, mup_muls):
    for k,ds in lm_datasets.items():
        for w,heads in widths:
            for d in depths:
                cfg = {
                    "num_heads": heads,
                    "d_model": w,
                    "num_layers": d,
                    "dropout_rate": 0.1,
                }
                name = 'transformer-w{}-d{}-h{}_{}'.format(w,d,heads,k)
                tasks[name] = functools.partial(func_create_func, 
                                                _TransformerTask, 
                                                ds,
                                                dict(cfg=cfg,name=name))
                
                name = 'mutransformer-w{}-d{}-h{}_{}'.format(w,d,heads,k)
                tasks[name] = functools.partial(func_create_func, 
                                                _MuTransformerTask, 
                                                ds,
                                                dict(cfg=cfg,
                                                     name=name,
                                                     mup_multipliers=mup_muls))

                # name = 'mutransformer-moe-w{}-d{}-h{}_{}'.format(w,d,heads,k)
                # tasks[name] = functools.partial(func_create_func, 
                #                                 _MuTransformerMoETask, 
                #                                 ds,
                #                                 dict(cfg=cfg,
                #                                      name=name,
                #                                      mup_multipliers=mup_muls))

                
                name = 'mudtransformer-w{}-d{}-h{}_{}'.format(w,d,heads,k)
                tasks[name] = functools.partial(func_create_func, 
                                                _MuTransformerTask, 
                                                ds,
                                                dict(cfg=cfg,
                                                     name=name,
                                                     use_mu_depth=True,
                                                     mup_multipliers=mup_muls))


                name = 'mucptransformer-w{}-d{}-h{}_{}'.format(w,d,heads,k)
                tasks[name] = functools.partial(func_create_func, 
                                                _MuTransformerTask, 
                                                ds,
                                                dict(cfg=cfg,
                                                     name=name,
                                                     use_mu_depth=True,
                                                     mup_multipliers=mup_muls))

def add_transformer_lm_moe_tasks_with_head(tasks, lm_datasets, widths, depths, num_experts, active_experts, mup_muls,
                                           parameterization_args=None, training_config=None):
    """Add transformer MoE tasks with configurable head counts and CompletedP support.
    
    Args:
      tasks: Dictionary to add tasks to
      lm_datasets: Dictionary of datasets keyed by name
      widths: List of (width, num_heads) tuples
      depths: List of depths
      num_experts: List of number of experts
      active_experts: List of active experts per token
      mup_muls: muP multipliers dict
      parameterization_args: Optional CompletedP parameterization config containing:
        - base_width: Base model width for HP transfer
        - base_depth: Base model depth for HP transfer
        - base_batch_size: Base batch size for HP transfer
        - base_dataset_size: Base dataset size for HP transfer
        - depth_multipliers: Per-layer depth multipliers
        - alpha: Depth scaling exponent (0.5 to 1.0)
      training_config: Optional training config containing:
        - gradient_accumulation_steps: Number of gradient accumulation steps
        - local_batch_size: Local batch size per device
        - num_inner_steps: Number of inner training steps
    """
    for dname,ds in lm_datasets.items():
        for w,heads in widths:
            for d in depths:
              for e in num_experts:
                for k in active_experts:
                  cfg = {
                        'model_dim': w,
                        'num_heads': heads,
                        'num_layers': d,
                        'ffn_dim': w * 2.75,
                        'remat': False,
                        'dropout_rate': 0.0,
                        'num_experts': e,
                        'num_experts_per_tok': k,
                        'capacity_factor': e, # set to e for dropless
                        'load_balance_loss_weight': 0.01,
                        'use_qk_norm': True,
                        'ffn_type': 'moe',
                        'max_seq_len': 64,
                        'tie_weights': False,
                        'pad_token_id': 2,
                        'use_mup': True,
                    }

                  

                  name = 'mutransformer-moe-w{}-d{}-h{}-e{}-k{}_{}'.format(w,d,heads,e,k,dname)
                  tasks[name] = functools.partial(func_create_func,
                                                  MuTransformerMoETask,
                                                  ds,
                                                  dict(cfg=copy.deepcopy(cfg),
                                                      name=name,
                                                      mup_multipliers=mup_muls,
                                                      parameterization_args=parameterization_args,
                                                      training_config=training_config))
                  # Z-loss variant
                  cfg_z = copy.deepcopy(cfg)
                  cfg_z['zloss_coefficient'] = 1e-4
                  name = 'muztransformer-moe-w{}-d{}-h{}-e{}-k{}_{}'.format(w,d,heads,e,k,dname)
                  tasks[name] = functools.partial(func_create_func,
                                                  MuTransformerMoETask,
                                                  ds,
                                                  dict(cfg=cfg_z,
                                                      name=name,
                                                      mup_multipliers=mup_muls,
                                                      parameterization_args=parameterization_args,
                                                      training_config=training_config))
                  cfg['use_mup'] = False
                  name = 'transformer-moe-w{}-d{}-h{}-e{}-k{}_{}'.format(w,d,heads,e,k,dname)
                  tasks[name] = functools.partial(func_create_func,
                                                  MuTransformerMoETask,
                                                  ds,
                                                  dict(cfg=copy.deepcopy(cfg),
                                                      name=name,
                                                      mup_multipliers=mup_muls,
                                                      parameterization_args=parameterization_args,
                                                      training_config=training_config))
                  # Z-loss variant
                  cfg_z = copy.deepcopy(cfg)
                  cfg_z['zloss_coefficient'] = 1e-4
                  name = 'ztransformer-moe-w{}-d{}-h{}-e{}-k{}_{}'.format(w,d,heads,e,k,dname)
                  tasks[name] = functools.partial(func_create_func,
                                                  MuTransformerMoETask,
                                                  ds,
                                                  dict(cfg=cfg_z,
                                                      name=name,
                                                      mup_multipliers=mup_muls,
                                                      parameterization_args=parameterization_args,
                                                      training_config=training_config))
              cfg['use_mup'] = True
              cfg['ffn_type'] = 'swiglu'
              name = 'mutransformer-dense-w{}-d{}-h{}_{}'.format(w,d,heads,dname)
              tasks[name] = functools.partial(func_create_func,
                                              MuTransformerMoETask,
                                              ds,
                                              dict(cfg=copy.deepcopy(cfg),
                                                  name=name,
                                                  mup_multipliers=mup_muls,
                                                  parameterization_args=parameterization_args,
                                                  training_config=training_config))
              # Z-loss variant
              cfg_z = copy.deepcopy(cfg)
              cfg_z['zloss_coefficient'] = 1e-4
              name = 'muztransformer-dense-w{}-d{}-h{}_{}'.format(w,d,heads,dname)
              tasks[name] = functools.partial(func_create_func,
                                              MuTransformerMoETask,
                                              ds,
                                              dict(cfg=cfg_z,
                                                  name=name,
                                                  mup_multipliers=mup_muls,
                                                  parameterization_args=parameterization_args,
                                                  training_config=training_config))
              cfg['use_mup'] = False
              name = 'transformer-dense-w{}-d{}-h{}_{}'.format(w,d,heads,dname)
              tasks[name] = functools.partial(func_create_func,
                                              MuTransformerMoETask,
                                              ds,
                                              dict(cfg=copy.deepcopy(cfg),
                                                  name=name,
                                                  mup_multipliers=mup_muls,
                                                  parameterization_args=parameterization_args,
                                                  training_config=training_config))
              # Z-loss variant
              cfg_z = copy.deepcopy(cfg)
              cfg_z['zloss_coefficient'] = 1e-4
              name = 'ztransformer-dense-w{}-d{}-h{}_{}'.format(w,d,heads,dname)
              tasks[name] = functools.partial(func_create_func,
                                              MuTransformerMoETask,
                                              ds,
                                              dict(cfg=cfg_z,
                                                  name=name,
                                                  mup_multipliers=mup_muls,
                                                  parameterization_args=parameterization_args,
                                                  training_config=training_config))


              cfg['ffn_type'] = 'regular_ffn'
              name = 'mutransformer-reg-dense-w{}-d{}-h{}_{}'.format(w,d,heads,dname)
              tasks[name] = functools.partial(func_create_func,
                                              MuTransformerMoETask,
                                              ds,
                                              dict(cfg=copy.deepcopy(cfg),
                                                  name=name,
                                                  mup_multipliers=mup_muls,
                                                  parameterization_args=parameterization_args,
                                                  training_config=training_config))
              # Z-loss variant
              cfg_z = copy.deepcopy(cfg)
              cfg_z['zloss_coefficient'] = 1e-4
              name = 'muztransformer-reg-dense-w{}-d{}-h{}_{}'.format(w,d,heads,dname)
              tasks[name] = functools.partial(func_create_func,
                                              MuTransformerMoETask,
                                              ds,
                                              dict(cfg=cfg_z,
                                                  name=name,
                                                  mup_multipliers=mup_muls,
                                                  parameterization_args=parameterization_args,
                                                  training_config=training_config))


def create_resnet(width, depth, use_residual=True):
  """A config based on the ViT-S_16 config but narrower."""
  assert depth % 4 == 0, "ResNets have 4 blocks"
  blocks = int(depth / 4)
  return {
    'blocks_per_group': (blocks,blocks,blocks,blocks), 
    'use_projection': (True, True, True, True), 
    'channels_per_group': (width, width, width, width), 
    'initial_conv_kernel_size': 3, 
    'initial_conv_stride': 2,
    'max_pool': True, 
    'resnet_v2': False,
    "use_residual": use_residual
}


def _conv_cross_entropy_pool_loss(
    hidden_units: Sequence[int],
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initializers: Optional[hk.initializers.Initializer] = None,
    norm_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    pool: str = "avg",
    num_classes: int = 10):
  """Haiku function for a conv net with pooling and cross entropy loss."""
  if not initializers:
    initializers = {}

  def _fn(batch):
    net = batch["image"]
    strides = [2] + [1] * (len(hidden_units) - 1)
    for hs, ks, stride in zip(hidden_units, [3] * len(hidden_units), strides):
      net = hk.Conv2D(hs, ks, stride=stride)(net)
      net = activation_fn(net)
      net = norm_fn(net)

    if pool == "avg":
      net = jnp.mean(net, [1, 2])
    elif pool == "max":
      net = jnp.max(net, [1, 2])
    else:
      raise ValueError("pool type not supported")

    logits = hk.Linear(num_classes)(net)

    labels = jax.nn.one_hot(batch["label"], num_classes)
    loss_vec = learned_optimization.tasks.base.softmax_cross_entropy(labels=labels, logits=logits)
    return jnp.mean(loss_vec)

  return _fn


def add_conv_tasks(tasks, image_datasets):
    for k,ds in image_datasets.items():
        base_model_fn = _conv_cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=1000, pool='max')
        name = 'conv-16-32-1000-maxpool_{}'.format(k)
        tasks[name] = functools.partial(func_create_func, 
                                        _ConvTask, 
                                        ds,
                                        dict(base_model_fn=base_model_fn))

        base_model_fn = _conv_cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=10, pool='max')
        name = 'conv-16-32-10-maxpool_{}'.format(k)
        tasks[name] = functools.partial(func_create_func, 
                                        _ConvTask, 
                                        ds,
                                        dict(base_model_fn=base_model_fn))

        base_model_fn = _conv_cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=10)
        name = 'small-conv-10_{}'.format(k)
        tasks[name] = functools.partial(func_create_func, 
                                        _ConvTask, 
                                        ds,
                                        dict(base_model_fn=base_model_fn))

        base_model_fn = _conv_cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=1000)
        name =  'small-conv-1000_{}'.format(k)
        tasks[name] = functools.partial(func_create_func, 
                                        _ConvTask, 
                                        ds,
                                        dict(base_model_fn=base_model_fn))

        base_model_fn = _conv_cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=10)
        name =  'large-conv-10_{}'.format(k)
        tasks[name] = functools.partial(func_create_func, 
                                        _ConvTask, 
                                        ds,
                                        dict(base_model_fn=base_model_fn))

        base_model_fn = _conv_cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=1000)
        name = 'large-conv-1000_{}'.format(k)
        tasks[name] = functools.partial(func_create_func, 
                                        _ConvTask, 
                                        ds,
                                        dict(base_model_fn=base_model_fn))





def add_resnet_tasks(tasks, image_datasets, widths, depths, mup_muls):
    for k,ds in image_datasets.items():
        for prefix, task_class in [('',_ResnetTaskDataset),('mu',_MuResnetTaskDataset)]:

            cfg = dict(initial_conv_kernel_size=7,
                        initial_conv_stride=2,
                        resnet_v2=False, 
                        max_pool=True,
                        use_residual=True,
                        **ResNet.CONFIGS[200])
            w=2048
            d=200
            name = prefix + 'resnet101-w{}-d{}_{}'.format(w,d,k)
            tasks[name] = functools.partial(func_create_func, 
                                            task_class, 
                                            ds,
                                            dict(cfg=cfg))


            cfg = dict(initial_conv_kernel_size=7,
                        initial_conv_stride=2,
                        resnet_v2=False, 
                        max_pool=True,
                        use_residual=True,
                        **ResNet.CONFIGS[152])
            w=2048
            d=152
            name = prefix + 'resnet101-w{}-d{}_{}'.format(w,d,k)
            tasks[name] = functools.partial(func_create_func, 
                                            task_class, 
                                            ds,
                                            dict(cfg=cfg))


            cfg = dict(initial_conv_kernel_size=7,
                        initial_conv_stride=2,
                        resnet_v2=False, 
                        max_pool=True,
                        use_residual=True,
                        **ResNet.CONFIGS[101])
            w=2048
            d=101
            name = prefix + 'resnet101-w{}-d{}_{}'.format(w,d,k)
            tasks[name] = functools.partial(func_create_func, 
                                            task_class, 
                                            ds,
                                            dict(cfg=cfg))


            cfg = dict(initial_conv_kernel_size=7,
                        initial_conv_stride=2,
                        resnet_v2=False, 
                        max_pool=True,
                        use_residual=True,
                        **ResNet.CONFIGS[50])
            w=2048
            d=50
            name = prefix + 'resnet50-w{}-d{}_{}'.format(w,d,k)
            tasks[name] = functools.partial(func_create_func, 
                                            task_class, 
                                            ds,
                                            dict(cfg=cfg))

            cfg = dict(initial_conv_kernel_size=7,
                        initial_conv_stride=2,
                        resnet_v2=False, 
                        max_pool=True,
                        use_residual=False,
                        **ResNet.CONFIGS[50])
            w=2048
            d=50
            name = prefix + 'noresresnet50-w{}-d{}_{}'.format(w,d,k)
            tasks[name] = functools.partial(func_create_func, 
                                            task_class, 
                                            ds,
                                            dict(cfg=cfg))
            
            

            cfg = dict(initial_conv_kernel_size=7,
                        initial_conv_stride=2,
                        resnet_v2=False, 
                        max_pool=True,
                        use_residual=True,
                        **ResNet.CONFIGS[18])
            w=2048
            d=18
            name = prefix + 'resnet18-w{}-d{}_{}'.format(w,d,k)
            tasks[name] = functools.partial(func_create_func, 
                                            task_class, 
                                            ds,
                                            dict(cfg=cfg))
            
            

            cfg = dict(initial_conv_kernel_size=7,
                        initial_conv_stride=2,
                        resnet_v2=False, 
                        max_pool=True,
                        use_residual=False,
                        **ResNet.CONFIGS[18])
            w=2048
            d=18
            name = prefix + 'noresresnet18-w{}-d{}_{}'.format(w,d,k)
            tasks[name] = functools.partial(func_create_func, 
                                            task_class, 
                                            ds,
                                            dict(cfg=cfg))

            for w in widths:
                for d in depths:

                    name = prefix + 'resnet-w{}-d{}_{}'.format(w,d,k)
                    tasks[name] = functools.partial(func_create_func, 
                                                    task_class, 
                                                    ds,
                                                    dict(cfg=create_resnet(width=w, depth=d)))

                    if prefix == 'mu':
                        name = prefix + 'resnet-kernelmult-w{}-d{}_{}'.format(w,d,k)
                        tasks[name] = functools.partial(func_create_func, 
                                                        task_class, 
                                                    ds,
                                                    dict(cfg=create_resnet(width=w, depth=d,),use_kernel_mult=True))



                    name = prefix + 'noresresnet-w{}-d{}_{}'.format(w,d,k)
                    tasks[name] = functools.partial(func_create_func, 
                                                    task_class, 
                                                    ds,
                                                    dict(cfg=create_resnet(width=w, depth=d, use_residual=False)))




# def add_moe_mlp_tasks(tasks, image_datasets, experts, active_experts, widths, depths, mup_muls):
#     for k,ds in image_datasets.items():
#         for e in experts:
#             for ae in active_experts:
#                 for w in widths:
#                     for d in depths:
#                       # name = 'vit-w{}-d{}_{}'.format(w,d,k)
#                       # tasks[name] = functools.partial(func_create_func, 
#                       #                                 VisionTransformerTask, 
#                       #                                 ds,
#                       #                                 dict(cfg=create_vit(hidden_size=w,heads=heads,depth=d)))
#                     #   tasks[name] = lambda ds=ds, w=w, heads=heads, d=d, mup_muls=mup_muls: VisionTransformerTask( ds, create_vit(hidden_size=w, heads=heads, depth=d), mup_multipliers=mup_muls)

#                       cfg = {
#                             'hidden_sizes': [w] * d,
#                             'num_experts': e,
#                             'num_experts_per_tok': ae,
#                             'moe_layers': [False] + (d-1) * [True] + [False],
#                             'capacity_factor':8,
#                             'load_balance_loss_weight': 0.01,
#                             'dropout_rate': 0.0,
#                             'log_activations': False
#                         }
#                       name = 'mumoemlp-w{}-d{}-e{}-k{}_{}'.format(w,d,e,ae,k)
#                       tasks[name] = functools.partial(func_create_func, 
#                                                       MuMoeMlpImageTask, 
#                                                       ds,
#                                                       dict(cfg=cfg,
#                                                           mup_multipliers=mup_muls))


#             #   tasks[name] = lambda ds=ds, w=w, heads=heads, d=d, mup_muls=mup_muls: MuVisionTransformerTask( ds, create_vit(hidden_size=w, heads=heads, depth=d), mup_multipliers=mup_muls)



def create_vit(hidden_size, heads, depth):
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = hidden_size
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = hidden_size * 4
  config.transformer.num_heads = heads
  config.transformer.num_layers = depth
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config


        
#this is kept for backwards compatibility
def add_vision_transformer_tasks(tasks, image_datasets, widths, depths, mup_muls):
    for k,ds in image_datasets.items():
        for w,heads in widths:
            for d in depths:
              name = 'vit-w{}-d{}_{}'.format(w,d,k)
              tasks[name] = functools.partial(func_create_func, 
                                              VisionTransformerTask, 
                                              ds,
                                              dict(cfg=create_vit(hidden_size=w,heads=heads,depth=d)))
            #   tasks[name] = lambda ds=ds, w=w, heads=heads, d=d, mup_muls=mup_muls: VisionTransformerTask( ds, create_vit(hidden_size=w, heads=heads, depth=d), mup_multipliers=mup_muls)


              name = 'muvit-w{}-d{}_{}'.format(w,d,k)
              tasks[name] = functools.partial(func_create_func, 
                                              MuVisionTransformerTask, 
                                              ds,
                                              dict(cfg=create_vit(hidden_size=w,heads=heads,depth=d),
                                                   mup_multipliers=mup_muls))
            #   tasks[name] = lambda ds=ds, w=w, heads=heads, d=d, mup_muls=mup_muls: MuVisionTransformerTask( ds, create_vit(hidden_size=w, heads=heads, depth=d), mup_multipliers=mup_muls)


#this is kept for backwards compatibility
def add_vision_transformer_tasks_with_head(tasks, image_datasets, widths, depths, mup_muls):
    for k,ds in image_datasets.items():
        for w,heads in widths:
            for d in depths:
              
              name = 'vit-w{}-d{}-h{}_{}'.format(w,d,heads,k)
              tasks[name] = functools.partial(func_create_func, 
                                              VisionTransformerTask, 
                                              ds,
                                              dict(cfg=create_vit(hidden_size=w,heads=heads,depth=d)))
              


              name = 'muvit-w{}-d{}-h{}_{}'.format(w,d,heads,k)
              tasks[name] = functools.partial(func_create_func, 
                                              MuVisionTransformerTask, 
                                              ds,
                                              dict(cfg=create_vit(hidden_size=w,heads=heads,depth=d),
                                                   mup_multipliers=mup_muls))
        
        

from typing import Tuple
import gin

from learned_optimization.tasks.datasets import base


@base.dataset_lru_cache
@gin.configurable
def mnist_datasets(batch_size: int,
                   image_size: Tuple[int, int] = (28, 28),
                   stack_channels: int = 1,
                           **kwargs) -> base.Datasets:
  splits = ("train[0:80%]", "train[80%:90%]", "train[90%:]", "test")
  return preload_tfds_image_classification_datasets_2(
      "mnist",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      stack_channels=stack_channels,**kwargs)


@base.dataset_lru_cache
@gin.configurable
def fashion_mnist_datasets(batch_size: int,
                           image_size: Tuple[int, int] = (28, 28),
                           stack_channels: int = 1,
                           prefetch_batches: int = 300,
                           **kwargs) -> base.Datasets:
  splits = ("train[0:80%]", "train[80%:90%]", "train[90%:]", "test")
  return preload_tfds_image_classification_datasets_2(
      "fashion_mnist",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      stack_channels=stack_channels,
      prefetch_batches=prefetch_batches,**kwargs)


@base.dataset_lru_cache
@gin.configurable
def cifar10_datasets(batch_size: int,
                     image_size: Tuple[int, int] = (32, 32),
                     data_fraction: float = 1.,
                    **kwargs) -> base.Datasets:
  perc = max(1, int(80 * data_fraction))
  perc=100
  splits = (f"train[0:{perc}%]", "train[80%:90%]", "train[90%:]", "test")
  return preload_tfds_image_classification_datasets_2(

      "cifar10", splits, 
      batch_size=batch_size, 
      image_size=image_size, 
      normalize_mean=(0.4914 * 255, 0.4822 * 255, 0.4465 * 255),
      normalize_std=(0.2470 * 255, 0.2435 * 255, 0.2616* 255),
      **kwargs)


@base.dataset_lru_cache
@gin.configurable
def cifar100_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (32, 32),
    data_fraction: float = 1.,
    **kwargs) -> base.Datasets:
  perc = min(1, int(80 * data_fraction))
  splits = (f"train[0:{perc}%]", "train[80%:90%]", "train[90%:]", "test")
  return preload_tfds_image_classification_datasets_2(
      "cifar100", splits, batch_size=batch_size, image_size=image_size,**kwargs)


@base.dataset_lru_cache
@gin.configurable
def svhn_cropped_datasets(batch_size: int,
                          image_size: Tuple[int, int] = (32, 32),
                          data_fraction: float = 1.,
                          **kwargs) -> base.Datasets:
  perc = min(1, int(80 * data_fraction))
  splits = (f"train[0:{perc}%]", "train[80%:90%]", "train[90%:]", "test")
  return preload_tfds_image_classification_datasets_2(
      "svhn_cropped",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      **kwargs)


@base.dataset_lru_cache
@gin.configurable
def food101_datasets(batch_size: int,
                     image_size: Tuple[int, int] = (32, 32),
                     data_fraction: float = 1.0,
                     **kwargs) -> base.Datasets:
  perc = min(1, int(80 * data_fraction))
  splits = (f"train[0:{perc}%]", "train[80%:90%]", "train[90%:]", "validation")
  return preload_tfds_image_classification_datasets_2(
      "food101", splits, batch_size=batch_size, image_size=image_size, **kwargs)


def get_test_batch_size(task, ovr_test_batch_size=None):
    if ovr_test_batch_size is not None:
        return ovr_test_batch_size
        
    if 'cifar' in task:
        return 10000
    elif 'food101' in task:
        return 10000
    elif 'fashionmnist' in task:
        return 10000
    elif 'svhn' in task:
        return 10000
    elif 'imagenet' in task:
          return 4096
    elif 'lm1b' in task:
          return 128
    elif 'fineweb' in task or 'fineweb100b' in task:
        return 32
    elif 'random' in task:
        return 128
    else:
        raise ValueError(f"Unknown task: {task}")
                
def get_task(args, is_test=False):

    depth_mup_multipliers = dict(
        depth_mult=args.mup_depth_mult,
        depth_lr_mult=args.mup_depth_lr_mult
    )

    mup_multipliers = dict(
        input_mult=args.mup_input_mult,
        output_mult=args.mup_output_mult,
        hidden_lr_mult=args.mup_hidden_lr_mult
    )


    _IMAGE_DTYPE_MAP = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }
    image_dtype = _IMAGE_DTYPE_MAP[getattr(args, "image_dtype", "float32")]

    # Get CompletedP parameterization config from args (loaded from config file)
    # e.g., config/parameterization/complete_p_bs100k_steps2000.py
    completedp_parameterization_args = getattr(args, 'parameterization_args', None)

    # Build training config from args attributes
    # These are used to compute current_batch_size and current_dataset_size for CompletedP scaling
    if hasattr(args, 'gradient_accumulation_steps') and hasattr(args, 'local_batch_size') and hasattr(args, 'num_inner_steps'):
        # Handle local_batch_size being a list (multi-task training)
        local_batch_size = args.local_batch_size[0] if isinstance(args.local_batch_size, list) else args.local_batch_size
        completedp_training_config = {
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'local_batch_size': local_batch_size,
            'num_inner_steps': args.num_inner_steps,
        }
    else:
        completedp_training_config = None


    created_tasks = []
    for i,chosen_task in enumerate(args.task):

        if args.run_type in ['benchmark','sweep']:
            batch_size = (args.meta_testing_batch_size,1,1,get_test_batch_size(chosen_task, args.ovr_test_batch_size),)
            TRAIN_PFB = 2
            TEST_PFB = 2
            prefetch_batches = (TRAIN_PFB,1,1,TEST_PFB)

            ds_kwargs = dict(   use_localsgd_batches=args.use_localsgd_batches,
                                prefetch_batches=prefetch_batches,
                                batch_shape=args.batch_shape,
                                label_sharding=args.label_sharding,
                                image_sharding=args.image_sharding,
                                image_dtype=image_dtype,)
        else:
            temp_bsz_args = args.meta_training_batch_args[i]
            if args.truncated_step_args['kwargs']['meta_loss_split'] is not None:
                batch_size = (temp_bsz_args["meta_training_batch_size"],1,temp_bsz_args["meta_training_batch_size"],1,)
                prefetch_batches = (args.prefetch_batches,1,args.prefetch_batches,1)
            else:
                batch_size = (temp_bsz_args["meta_training_batch_size"],1,1,1,)
                prefetch_batches = (args.prefetch_batches,1,1,1)


            ds_kwargs = dict(   use_localsgd_batches=args.use_localsgd_batches,
                                prefetch_batches=prefetch_batches,
                                seed=args.seed,
                                batch_shape=temp_bsz_args["batch_shape"],
                                label_sharding=temp_bsz_args["label_sharding"],
                                image_sharding=temp_bsz_args["image_sharding"],
                                image_dtype=image_dtype,)


        IMAGE_DATASET_REGISTY = {

            'imagenet5050-32x32x3': dict(fun=imagenet_50_50,args=[],kwargs=dict(batch_size=batch_size,image_size=(32, 32), **ds_kwargs)),
            'imagenet5050-64x64x3':  dict(fun=imagenet_50_50,args=[],kwargs=dict(batch_size=batch_size,image_size=(64, 64), **ds_kwargs)),
            'imagenet5050-128x128x3':  dict(fun=imagenet_50_50,args=[],kwargs=dict(batch_size=batch_size,image_size=(128, 128), **ds_kwargs)),
            'imagenet5050-224x224x3':  dict(fun=imagenet_50_50,args=[],kwargs=dict(batch_size=batch_size,image_size=(224, 224), **ds_kwargs)),

            'imagenet-aug-32x32x3': dict(fun=imagenet_dataset,args=[],kwargs=dict(batch_size=batch_size,image_size=(32, 32), augmentations=tuple(args.augmentations or ()), **ds_kwargs)),
            'imagenet-aug-64x64x3':  dict(fun=imagenet_dataset,args=[],kwargs=dict(batch_size=batch_size,image_size=(64, 64), augmentations=tuple(args.augmentations or ()), **ds_kwargs)),
            'imagenet-aug-128x128x3':  dict(fun=imagenet_dataset,args=[],kwargs=dict(batch_size=batch_size,image_size=(128, 128), augmentations=tuple(args.augmentations or ()), **ds_kwargs)),
            'imagenet-aug-224x224x3':  dict(fun=imagenet_dataset,args=[],kwargs=dict(batch_size=batch_size,image_size=(224, 224), augmentations=tuple(args.augmentations or ()), **ds_kwargs)),

            'imagenet-32x32x3': dict(fun=imagenet_dataset,args=[],kwargs=dict(batch_size=batch_size,image_size=(32, 32), **ds_kwargs)),
            'imagenet-64x64x3':  dict(fun=imagenet_dataset,args=[],kwargs=dict(batch_size=batch_size,image_size=(64, 64), **ds_kwargs)),
            'imagenet-128x128x3':  dict(fun=imagenet_dataset,args=[],kwargs=dict(batch_size=batch_size,image_size=(128, 128), **ds_kwargs)),
            'imagenet-224x224x3':  dict(fun=imagenet_dataset,args=[],kwargs=dict(batch_size=batch_size,image_size=(224, 224), **ds_kwargs)),


            'imagenet-orig-32x32x3': dict(fun=imagenet_orig_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(32, 32), **ds_kwargs)),
            'imagenet-orig-64x64x3':  dict(fun=imagenet_orig_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(64, 64), **ds_kwargs)),
            'imagenet-orig-128x128x3':  dict(fun=imagenet_orig_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(128, 128), **ds_kwargs)),
            'imagenet-orig-224x224x3':  dict(fun=imagenet_orig_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(224, 224), **ds_kwargs)),

            'random-64x64x3':  dict(fun=random_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(64, 64), **ds_kwargs)),

            'random-224x224x3':  dict(fun=random_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(224, 224), **ds_kwargs)),

            'cifar10-32x32x3': dict(fun=cifar10_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(32, 32), **ds_kwargs)),
            'food101-32x32x3': dict(fun=food101_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(32, 32), **ds_kwargs)),
            'fashionmnist-28x28x1': dict(fun=fashion_mnist_datasets,args=[],kwargs=dict(batch_size=batch_size, **ds_kwargs)),
            'fashionmnist-8x8x1': dict(fun=fashion_mnist_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(8, 8), **ds_kwargs)),


            # CELO2 meta-training tasks (8x8 resolution)
            'mnist-28x28x1': dict(fun=mnist_datasets,args=[],kwargs=dict(batch_size=batch_size, **ds_kwargs)),
            'mnist-8x8x1': dict(fun=mnist_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(8, 8), **ds_kwargs)),
            'cifar10-8x8x3': dict(fun=cifar10_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(8, 8), **ds_kwargs)),
            'svhn-32x32x3': dict(fun=svhn_cropped_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(32, 32), **ds_kwargs)),
            'svhn-8x8x3': dict(fun=svhn_cropped_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(8, 8), **ds_kwargs)),


            # aug versions
            'mnist-aug-28x28x1': dict(fun=mnist_datasets,args=[],kwargs=dict(batch_size=batch_size, augmentations=tuple(args.augmentations or ()), **ds_kwargs)),
            'mnist-aug-8x8x1': dict(fun=mnist_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(8, 8), augmentations=tuple(args.augmentations or ()), **ds_kwargs)),
            'cifar10-aug-32x32x3': dict(fun=cifar10_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(32, 32), augmentations=tuple(args.augmentations or ()), **ds_kwargs)),
            'cifar10-aug-8x8x3': dict(fun=cifar10_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(8, 8), augmentations=tuple(args.augmentations or ()), **ds_kwargs)),
            'fashionmnist-aug-28x28x1': dict(fun=fashion_mnist_datasets,args=[],kwargs=dict(batch_size=batch_size, augmentations=tuple(args.augmentations or ()), **ds_kwargs)),
            'fashionmnist-aug-8x8x1': dict(fun=fashion_mnist_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(8, 8), augmentations=tuple(args.augmentations or ()), **ds_kwargs)),
            'svhn-aug-32x32x3': dict(fun=svhn_cropped_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(32, 32), augmentations=tuple(args.augmentations or ()), **ds_kwargs)),
            'svhn-aug-8x8x3': dict(fun=svhn_cropped_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(8, 8), augmentations=tuple(args.augmentations or ()), **ds_kwargs)),

        }
        

        fineweb_kwargs = dict(
          data_root=os.path.join(os.environ["TFDS_DATA_DIR"], "fineweb_edu_10B"), 
          process_rank=args.rank, 
          num_processes=jax.process_count(), 
          batch_size=batch_size,
          **ds_kwargs
        )
        fineweb_100b_kwargs = dict(
          data_root=os.path.join(os.environ["TFDS_DATA_DIR"], "fineweb_edu_100B"), 
          process_rank=args.rank, 
          num_processes=jax.process_count(), 
          batch_size=batch_size,
          **ds_kwargs
        )
        dclm_kwargs = dict(
          data_root="data/dclm_tokenized",
          name='dclm',
          hf_tokenizer=os.path.join(os.environ["TFDS_DATA_DIR"], "meta-llama/Llama-2-7b-hf"),
          process_rank=args.rank, 
          num_processes=jax.process_count(), 
          batch_size=batch_size,
          **ds_kwargs
        )
        lm1b_kwargs = dict(vocab='sentencepiece', batch_size=batch_size, **ds_kwargs)
        LANGUAGE_DATASET_REGISTY = {
            'lm1b-s2048-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict( sequence_length=2048, **ds_kwargs)),
            'lm1b-s1024-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict( sequence_length=1024, **lm1b_kwargs)),
            'lm1b-s512-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict( sequence_length=512, **lm1b_kwargs)),
            'lm1b-s256-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict( sequence_length=256, **lm1b_kwargs)),
            'lm1b-s128-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict( sequence_length=128, **lm1b_kwargs)),
            'lm1b-s64-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict( sequence_length=64, **lm1b_kwargs)),
            'lm1b-s32-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict( sequence_length=32, **lm1b_kwargs)),
            'lm1b-s16-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict( sequence_length=16, **lm1b_kwargs)),
            'lm1b-s8-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict( sequence_length=8, **lm1b_kwargs)),
            'lm1b-s4-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict( sequence_length=4, **lm1b_kwargs)),

            'fineweb-s2048-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=2048, **fineweb_kwargs)),
            'fineweb-s1024-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=1024, **fineweb_kwargs)),
            'fineweb-s512-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=512, **fineweb_kwargs)),
            'fineweb-s256-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=256, **fineweb_kwargs)),
            'fineweb-s128-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=128, **fineweb_kwargs)),
            'fineweb-s64-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=64, **fineweb_kwargs)),
            'fineweb-s32-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=32, **fineweb_kwargs)),
            'fineweb-s16-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=16, **fineweb_kwargs)),
            'fineweb-s8-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=8, **fineweb_kwargs)),
            'fineweb-s4-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=4, **fineweb_kwargs)),

            'fineweb100b-s2048-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict( sequence_length=2048, **fineweb_100b_kwargs)),
            'fineweb100b-s1024-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=1024, **fineweb_100b_kwargs)),
            'fineweb100b-s512-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=512, **fineweb_100b_kwargs)),
            'fineweb100b-s256-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=256, **fineweb_100b_kwargs)),
            'fineweb100b-s128-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=128, **fineweb_100b_kwargs)),
            'fineweb100b-s64-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=64, **fineweb_100b_kwargs)),
            'fineweb100b-s32-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=32, **fineweb_100b_kwargs)),
            'fineweb100b-s16-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=16, **fineweb_100b_kwargs)),
            'fineweb100b-s8-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=8, **fineweb_100b_kwargs)),
            'fineweb100b-s4-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=4, **fineweb_100b_kwargs)),

            'dclm-s2048-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict( sequence_length=2048, **dclm_kwargs)),
            'dclm-s1024-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=1024, **dclm_kwargs)),
            'dclm-s512-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=512, **dclm_kwargs)),
            'dclm-s256-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=256, **dclm_kwargs)),
            'dclm-s128-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=128, **dclm_kwargs)),
            'dclm-s64-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=64, **dclm_kwargs)),
            'dclm-s32-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=32, **dclm_kwargs)),
            'dclm-s16-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=16, **dclm_kwargs)),
            'dclm-s8-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=8, **dclm_kwargs)),
            'dclm-s4-gpt2': dict(fun=make_fineweb_datasets,args=[],kwargs=dict(sequence_length=4, **dclm_kwargs)),

        }

        # Optimization: filter registries to only the single dataset this task needs.
        # Task names follow the format {model_spec}_{dataset_key} where dataset keys
        # contain no underscores. This avoids building millions of unused task entries.
        _dataset_key = None
        for _k in sorted(IMAGE_DATASET_REGISTY.keys(), key=len, reverse=True):
            if chosen_task.endswith('_' + _k):
                _dataset_key = _k
                break
        if _dataset_key is None:
            for _k in sorted(LANGUAGE_DATASET_REGISTY.keys(), key=len, reverse=True):
                if chosen_task.endswith('_' + _k):
                    _dataset_key = _k
                    break
        if _dataset_key is not None:
            if _dataset_key in IMAGE_DATASET_REGISTY:
                IMAGE_DATASET_REGISTY = {_dataset_key: IMAGE_DATASET_REGISTY[_dataset_key]}
                LANGUAGE_DATASET_REGISTY = {}
            else:
                IMAGE_DATASET_REGISTY = {}
                LANGUAGE_DATASET_REGISTY = {_dataset_key: LANGUAGE_DATASET_REGISTY[_dataset_key]}

        _model_spec = chosen_task[:len(chosen_task) - len(_dataset_key) - 1] if _dataset_key else chosen_task
        _is_moe_or_dense = ('moe' in _model_spec or 'dense' in _model_spec)

        tasks = {}

        add_MLP_tasks(tasks,
                      image_datasets=IMAGE_DATASET_REGISTY,
                      widths=[2**i for i in range(16)] + [192 * x for x in range(1,10)], 
                      depths=[1,2,3,6,8,12,16,24,32,64,128,5],
                      log_activations=args.log_activations,
                      mup_muls=mup_multipliers,
                      depth_mup_multipliers=depth_mup_multipliers)
        
        add_sweepable_MLP_tasks(tasks, 
                                image_datasets=IMAGE_DATASET_REGISTY, 
                                widths=[128,512], 
                                depths=[3],
                                # log_activations=args.log_activations,
                                mup_muls=mup_multipliers)
        
        # add_moe_mlp_tasks(tasks, image_datasets=IMAGE_DATASET_REGISTY, 
        #                   experts=[2**i for i in range(16)], 
        #                   active_experts=[x for x in range(1,11)], 
        #                   widths=[2**i for i in range(16)] + [192 * x for x in range(1,10)], 
        #                   depths=[1,2,3,6,8,12,16,24,32,64,128],
        #                   mup_muls=mup_multipliers) 

        add_transformer_lm_tasks(tasks, 
                                 lm_datasets=LANGUAGE_DATASET_REGISTY,
                                 widths=[(64,1),(128,2),(192,3),(256,4),(384,6),(512,4),(768,8),(1024,8),(1280,8),(2048,16),(1024*3,16),(4096,16),(1024*5,16),], 
                                 depths=[1,2,3,6,8,12,16,18,24,32,64,128],
                                 mup_muls=mup_multipliers)


        add_transformer_lm_tasks_with_head(tasks, 
                                            lm_datasets=LANGUAGE_DATASET_REGISTY,
                                            widths=[
                                                    (16,1),(16,2),(16,4),(16,8),
                                                    (32,1),(32,2),(32,4),(32,8),(32,16), 
                                                    (64,1),(64,2),(64,4),(64,8),(64,16),
                                                    (128,1),(128,2),(128,4),(128,6),(128,8),(128,16),
                                                    (192,1),(192,2),(192,3),(192,4),(192,6),(192,8),(192,12),(192,16),
                                                    (256,1),(256,2),(256,4),(256,8),(256,16),
                                                    (384,1),(384,2),(384,4),(384,6),(384,8),(384,12),(384,16),
                                                    (512,1),(512,2),(512,4),(512,6),(512,8),(512,12),(512,16),
                                                    (768,1),(768,2),(768,4),(768,6),(768,8),(768,12),(768,16),
                                                    (1024,1),(1024,2),(1024,4),(1024,8),(1024,16),(1024,32),
                                                    (1280,1),(1280,2),(1280,4),(1280,8),(1280,16),
                                                    (2048,1),(2048,2),(2048,4),(2048,8),(2048,16),(2048,32),
                                                    (3072,1),(3072,2),(3072,4),(3072,8),(3072,16), (3072,24), (3072,64), (3072,128),
                                                    (4096,1),(4096,2),(4096,4),(4096,8),(4096,16), (4096,32), (4096,64), (4096,128),
                                                    (5120,1),(5120,2),(5120,4),(5120,8),(5120,16), (5120,32), (5120,48), (5120,128),
                                                    
                                                    (8192,1),(8192,2),(8192,4),(8192,8),(8192,16),(8192,32), (8192,64), (8192,128),
                                                    (16384,1),(16384,2),(16384,4),(16384,8),(16384,16),(16384,32), (16384,64), (16384,128),
                                                    (32768,1),(32768,2),(32768,4),(32768,8),(32768,16),(32768,32), (32768,64), (32768,128),
                                                    (65536,1),(65536,2),(65536,4),(65536,8),(65536,16),(65536,32), (65536,64), (65536,128),
                                                    # (131072,1),(131072,2),(131072,4),(131072,8),(131072,16),
                                                    # (262144,1),(262144,2),(262144,4),(262144,8),(262144,16),
                                                    ],
                                            depths=[1,2,3,4,6,8,12,16,18,24,28,32,64,128],
                                            mup_muls=mup_multipliers)

        if _is_moe_or_dense:
          add_transformer_lm_moe_tasks_with_head(tasks,
                                                lm_datasets=LANGUAGE_DATASET_REGISTY,
                            widths=[
                              (8,1),(8,2),(8,4),(8,8),(8,16),
                              (16,1),(16,2),(16,4),(16,8),(16,16),
                              (32,1),(32,2),(32,4),(32,8),(32,16),
                              (64,1),(64,2),(64,4),(64,8),(64,16),
                              (128,1),(128,2),(128,4),(128,6),(128,8),(128,16),
                              (192,1),(192,2),(192,3),(192,4),(192,6),(192,8),(192,12),(192,16),
                              (256,1),(256,2),(256,4),(256,8),(256,16),
                              (384,1),(384,2),(384,4),(384,6),(384,8),(384,12),(384,16),
                              (512,1),(512,2),(512,4),(512,6),(512,8),(512,12),(512,16),
                              (768,1),(768,2),(768,4),(768,6),(768,8),(768,12),(768,16),
                              (1024,1),(1024,2),(1024,4),(1024,8),(1024,16),(1024,32),
                              (1280,1),(1280,2),(1280,4),(1280,8),(1280,16),
                              (2048,1),(2048,2),(2048,4),(2048,8),(2048,16),
                              (3072,1),(3072,2),(3072,4),(3072,8),(3072,16),
                              (4096,1),(4096,2),(4096,4),(4096,8),(4096,16),
                              (5120,1),(5120,2),(5120,4),(5120,8),(5120,16)],
                                                depths=[1,2,3,4,6,8,12,16,18,24,32,64,128],
                                                num_experts=[2**i for i in range(16)],
                                                active_experts=[x for x in range(1,11)], 
                                                mup_muls=mup_multipliers,
                                                parameterization_args=completedp_parameterization_args,
                                                training_config=completedp_training_config)
        #this is kept for backwards compatibility
        add_vision_transformer_tasks(tasks,
                                     image_datasets=IMAGE_DATASET_REGISTY,
                                     widths=[(64,2),(128,2),(192,3),(256,4),(384,6),(512,8),(768,8),(1024,8),(2048,16),(1024*3,16),(4096,16),(1024*5,16),], 
                                     depths=[1,2,3,6,8,12,16,24,32,64],
                                     mup_muls=mup_multipliers)
        
        add_vision_transformer_tasks_with_head(tasks,
                                     image_datasets=IMAGE_DATASET_REGISTY,
                                            widths=[
                                                    (16,1),(16,2),(16,4),(16,8),
                                                    (32,1),(32,2),(32,4),(32,8),(32,16), 
                                                    (64,1),(64,2),(64,4),(64,8),(64,16),
                                                    (128,1),(128,2),(128,4),(128,6),(128,8),(128,16),
                                                    (192,1),(192,2),(192,3),(192,4),(192,6),(192,8),(192,12),(192,16),
                                                    (256,1),(256,2),(256,4),(256,8),(256,16),
                                                    (384,1),(384,2),(384,4),(384,6),(384,8),(384,12),(384,16),
                                                    (768,1),(768,2),(768,4),(768,6),(768,8),(768,12),(768,16),
                                                    (1024,1),(1024,2),(1024,4),(1024,8),(1024,16),(1024,32),
                                                    (1280,1),(1280,2),(1280,4),(1280,8),(1280,16),
                                                    (2048,1),(2048,2),(2048,4),(2048,8),(2048,16),
                                                    (3072,1),(3072,2),(3072,4),(3072,8),(3072,16),
                                                    (4096,1),(4096,2),(4096,4),(4096,8),(4096,16),(4096,32),
                                                    (5120,1),(5120,2),(5120,4),(5120,8),(5120,16)],
                                     depths=[1,2,3,6,8,12,16,24,32,64],
                                     mup_muls=mup_multipliers)


        add_resnet_tasks(tasks, 
                         image_datasets=IMAGE_DATASET_REGISTY, 
                         widths=[2**i for i in range(16)] + [192,1024*3], 
                         depths=[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100],
                         mup_muls=mup_multipliers)

        add_conv_tasks(tasks, image_datasets=IMAGE_DATASET_REGISTY)
        created_tasks.append(tasks[chosen_task]())
    return created_tasks



