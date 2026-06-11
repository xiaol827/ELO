# Python standard library
import asyncio
import csv
import os
import os.path as osp
import pickle
import re
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from functools import reduce
from glob import glob
from typing import Any

# Re-export config_to_dict from config_utils for convenience
from config_utils import config_to_dict

# Third-party libraries
import aiofiles
import haiku as hk
from haiku._src.typing import Initializer
from haiku.initializers import *
import jax
from jax import lax
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec
from jax.sharding import Mesh as JaxMesh
import jax.numpy as jnp

# Compatibility wrapper for PositionalSharding (removed in JAX 0.8.0+)
class PositionalSharding:
    """Compatibility wrapper for the old PositionalSharding API."""
    def __init__(self, devices):
        # devices is a numpy array from mesh_utils.create_device_mesh
        # Convert to the new Mesh + NamedSharding API
        if hasattr(devices, 'shape'):
            # Create axis names based on the number of dimensions
            axis_names = tuple(f'axis_{i}' for i in range(len(devices.shape)))
            self._mesh = JaxMesh(devices, axis_names)
            # Use replicated sharding (no partitioning across axes)
            self._sharding = NamedSharding(self._mesh, PartitionSpec())
        else:
            # Single device case
            self._sharding = NamedSharding(JaxMesh(devices, ('x',)), PartitionSpec())
    
    def __getattr__(self, name):
        # Delegate all other attributes to the underlying sharding
        return getattr(self._sharding, name)
    
    def __repr__(self):
        return f"PositionalSharding({self._sharding})"
from learned_optimization import checkpoints
import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm
import wandb

# Type definitions
State = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray



def safe_block_until_ready(x):
    if hasattr(x, 'block_until_ready'):
        # print("blocking until ready")
        return x.block_until_ready()
    return x




def convert_config_to_dict(config):
    """
    Recursively convert ConfigDict objects to regular dictionaries for serialization.
    
    Args:
        config: A ConfigDict object or any other object that might contain ConfigDict objects
        
    Returns:
        A serializable dictionary or the original object if not a ConfigDict
    """
    if hasattr(config, '__dict__') and hasattr(config, 'keys') and callable(config.keys):
        # This is likely a ConfigDict or similar object
        return dict(config)
    elif isinstance(config, dict):
        # Handle regular dictionaries
        return {k: convert_config_to_dict(v) for k, v in config.items()}
    elif isinstance(config, list):
        # Handle lists
        return [convert_config_to_dict(item) for item in config]
    elif isinstance(config, tuple):
        # Handle tuples
        return tuple(convert_config_to_dict(item) for item in config)
    else:
        # Return other types as is
        return config
                

class Timing:

    # Static dictionaries to store run times and historical stats
    run_times_dict = defaultdict(list)  # Stores the elapsed times for each named timer
    historical_stats = defaultdict(lambda: {"mean": [], "std": []})  # Stores the historical mean and std for each named timer

    def __init__(self,name,list):
        self.name = name
        self.list = list

    def __enter__(self):
        # print("entering timing", self.name)
        self.start = time.time()
        return self  # This allows us to use "as x" in the with statement

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        duration = self.end - self.start
        Timing.run_times_dict[self.name].append(duration)
        self.list.append(duration)
        # print("exiti ng timing", self.name, duration)


# CHECKPOINT SAVING HELPERS

def find_smallest_divisor(x,b):
  # Start from the smallest possible divisor greater than 1
  for y in range(2, x + 1):  # We start from 2 as 1 will always divide x and result in a itself
      if x % y == 0:  # Check if y is a divisor of x
          a = x // y  # Calculate a as the quotient of x divided by y
          if a < b:  # Check if a meets the condition
              return y  # Return the smallest y that meets the condition
  print("Warning: No smaller divisor found. Returning the original value.")
  return x  # Return x if no smaller divisor is found satisfying the condition


def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    # print(*message, flush=True)
    if jax.distributed.is_initialized():
        if jax.process_index() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def delete_old_checkpoints(save_dir, n_to_keep, world_size):
    """Prune old flat checkpoint files, keeping only the newest ``n_to_keep``.

    Both meta-train (``global_step{N}.pickle``) and benchmark
    (``global_step{N}.pkl``) save one flat file per step directly in
    ``save_dir``. ``latest`` and ``rank-*_outer_trainer_state.ckpt`` never match
    the regex, so resume state is always preserved. Keeping the directory bounded
    avoids the O(n^2) metadata scan that previously hammered $SCRATCH.
    """
    if save_dir.endswith("/"):
        save_dir = save_dir.rstrip("/")
    if not os.path.isdir(save_dir):
        return

    ckpt_re = re.compile(r"^global_step\d+.*\.(pkl|pickle)$")
    ckpt_files = natural_sort([x for x in os.listdir(save_dir) if ckpt_re.match(x)])

    n_to_delete = len(ckpt_files) - n_to_keep
    if n_to_delete <= 0:
        return
    for name in ckpt_files[:n_to_delete]:
        try:
            os.remove(os.path.join(save_dir, name))
        except OSError as e:
            print(f"Error: {e.strerror} - {e.filename}")



    # all_ckpts = natural_sort(   
    #     [
    #         i
    #         for i in glob(f"{save_dir}/*")
    #         if i.endswith(".ckpt") and re.search(ckpt_dir_regex, i)
    #     ]
    # )
    # all_pkl = natural_sort(
    #     [
    #         i
    #         for i in glob(f"{save_dir}/*")
    #         if i.endswith(".pickle") and re.search(ckpt_dir_regex, i)
    #     ]
    # )

    # n_to_delete = int(len(all_ckpts)) - 1
    # n_to_delete_pkl = len(all_pkl) - n_to_keep
    # if n_to_delete > 0:
    #     to_delete_ckpt = all_ckpts[:n_to_delete]
    #     to_delete_pkl = all_pkl[:n_to_delete_pkl]
    #     print(
    #         f"WARNING: Deleting old checkpoints: \n\t{', '.join(to_delete_ckpt + to_delete_pkl)}"
    #     )
    #     for ckpt in to_delete_ckpt + to_delete_pkl:
    #         try:
    #             os.removedirs(ckpt)
    #         except FileNotFoundError:
    #             pass


# def save_checkpoint(
#     prefix, i, args, outer_trainer_state, rank
# ):  # Checkpoint every 1000th iteration
#     print("inside save_checkpoint, rank:",rank)
#     outer_dir = osp.join("checkpoints", f"{prefix}_{args.meta_train_name}")
#     save_dir = osp.join(outer_dir, 'global_step{}'.format(i+1))
#     os.makedirs(save_dir, exist_ok=True)

#     checkpoints.save_state(
#         osp.join(
#             save_dir,
#             "rank-{}_outer_trainer_state.ckpt".format(rank,i + 1),
#         ),
#         outer_trainer_state,
#     )

#     if rank == 0:
        
#         async def save_files():
#             pickle_filename = osp.join(
#                 save_dir,
#                 "optimizer_ckpt.pickle".format(i + 1),
#             )
            
#             # Save pickle file
#             async with aiofiles.open(pickle_filename, "wb") as f:
#                 await f.write(pickle.dumps(
#                     outer_trainer_state.gradient_learner_state.theta_opt_state.params
#                 ))

#             # Save latest file
#             async with aiofiles.open(osp.join(outer_dir, "latest"), "w") as f:
#                 await f.write("global_step{}".format(i + 1))

#             # Delete old checkpoints
#             delete_old_checkpoints(
#                 save_dir=outer_dir,
#                 n_to_keep=args.checkpoints_to_keep,
#                 world_size=args.world_size
#             )
            
#             return pickle_filename

#         # Run the async function
#         loop = asyncio.get_event_loop()
#         return loop.run_until_complete(save_files())
#     else:
#         return None


def save_checkpoint(
    prefix, i, args, outer_trainer_state, rank, unroll_length=None
):  # Checkpoint every 1000th iteration
    print("inside save_checkpoint, rank:",rank)

    save_dir = osp.join(os.environ["SCRATCH"], f"checkpoints/{prefix}_{args.meta_train_name}")
    os.makedirs(save_dir, exist_ok=True)

    checkpoints.save_state(
        osp.join(
            save_dir,
            "rank-{}_outer_trainer_state.ckpt".format(rank,i + 1),
        ),
        outer_trainer_state,
    )

    if rank == 0:
        pickle_filename = osp.join(
            save_dir,
            "global_step{}.pickle".format(i + 1),
        ) if unroll_length is None else osp.join(
            save_dir,
            "global_step{}.ul{}.pickle".format(i + 1, unroll_length),
        )
        with open(
            pickle_filename,
            "wb",
        ) as f:
            pickle.dump(
                outer_trainer_state.gradient_learner_state.theta_opt_state.params, f
            )

        with open(osp.join(save_dir, "latest"), "w") as f:
            f.write("global_step{}".format(i + 1))

        # Local theta pickles are redundant (every one is also uploaded to W&B via
        # wandb.save), so keep only the newest one on disk. The deleted file is
        # always the *previous* step's pickle, which finished uploading a full
        # save-interval ago, so there is no upload race.
        delete_old_checkpoints(
            save_dir=save_dir,
            n_to_keep=1,
            world_size=args.world_size
        )
        return pickle_filename
    else:
        return None


def get_ckpt_dirs(ckpt_dir, meta_train_name):
    a = os.listdir(ckpt_dir)
    keep = []
    for x in a:
                                            # 8 for wandb id +1 for underscore
        if osp.isdir(osp.join(ckpt_dir, x)) and x[9:] == meta_train_name:
            keep.append(osp.join(ckpt_dir, x))
    return keep


def get_ckpt_to_load(ckpt_dir, dirs):
    def nat_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key[1])]
        return sorted(l, key=alphanum_key)

    sortable = []
    for x in dirs:
        if osp.isfile(osp.join(ckpt_dir, x, "latest")):
            ckpt = open(osp.join(ckpt_dir, x, "latest"), "r").readline().strip()
            sortable.append(
                (
                    osp.join(ckpt_dir, x, ckpt),
                    ckpt,
                )
            )

    if len(sortable) == 0:
        return []
    
    sortable = nat_sort(sortable)

    keep = []
    for x in sortable:
        if x[1] == sortable[-1][1]:
            keep.append(x)



    if len(keep) > 1:
        print(
            "[Warning] multiple directories contain a checkpoint at the same latest iteration. Selecting one arbitrarily."
        )

    return keep[0]



def download_wandb_checkpoint(cfg):
    api = wandb.Api()
    run = api.run(cfg.wandb_checkpoint_id)

    print("selected_checkpoint:",cfg.selected_checkpoint)
    ckpts = [x for x in run.files() if 'global_step' in x.name]
    parent_dir = os.path.join(os.environ["SCRATCH"], "checkpoints/test_cpkts")
    os.makedirs(parent_dir, exist_ok=True)
    def _download_and_resolve(ckpt, parent_dir):
        print('Downloading checkpoint:', ckpt.name)
        ckpt.download(parent_dir, replace=True)
        local_path = osp.join(parent_dir, ckpt.name)
        if osp.exists(local_path):
            return local_path
        # 部分 wandb 版本下载时会丢弃子目录，只保留文件名
        alt_path = osp.join(parent_dir, osp.basename(ckpt.name))
        if osp.exists(alt_path):
            print(f'Warning: checkpoint found at alt path {alt_path}')
            return alt_path
        raise FileNotFoundError(
            f"wandb download silently failed. File not found at:\n"
            f"  {local_path}\n"
            f"  {alt_path}"
        )

    if "velo_ckpt.pickle" == cfg.selected_checkpoint:
        ckpt = run.file("velo_ckpt.pickle")
        return _download_and_resolve(ckpt, parent_dir)

    elif cfg.selected_checkpoint and 'global_step' not in cfg.selected_checkpoint:
        # Direct file name download (e.g., theta.state for celo2)
        ckpt = run.file(cfg.selected_checkpoint)
        return _download_and_resolve(ckpt, parent_dir)

    elif 'global_step' in cfg.selected_checkpoint:
        assert len(ckpts) >= 1, f"selected_checkpoint {cfg.selected_checkpoint} not found"
        ckpt = [x for x in ckpts if cfg.selected_checkpoint in x.name][0]
        return _download_and_resolve(ckpt, parent_dir)
    else:
        assert len(ckpts) >= 1, f"No global_step checkpoints found in run {cfg.wandb_checkpoint_id}"
        ckpt = sorted(ckpts, key=lambda x: int(x.name.split("global_step")[-1].split('.')[0]))[-1]
        return _download_and_resolve(ckpt, parent_dir)



def _extract_global_step(name):
    """Parse the integer global_step from a checkpoint file name like
    'global_step100000.pickle' or 'global_step100000.ul20.pickle'."""
    return int(name.split("global_step")[-1].split('.')[0])


def _average_pickled_pytrees(paths):
    """Element-wise average across a list of pickled JAX pytrees.

    Accumulates in float32 for numerical stability, casts each leaf back to
    its original dtype taken from the first checkpoint.
    """
    n = len(paths)
    assert n >= 1, "need at least one checkpoint to average"

    with open(paths[0], 'rb') as f:
        first = pickle.load(f)
    dtypes = jax.tree_util.tree_map(lambda x: np.asarray(x).dtype, first)
    acc = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float32) / n, first)
    del first

    for p in paths[1:]:
        with open(p, 'rb') as f:
            params = pickle.load(f)
        acc = jax.tree_util.tree_map(
            lambda a, x: a + np.asarray(x, dtype=np.float32) / n,
            acc, params,
        )
        del params

    return jax.tree_util.tree_map(lambda a, dt: np.asarray(a, dtype=dt), acc, dtypes)


def build_checkpoint_soup(cfg):
    """Download every wandb checkpoint whose global_step lies in the closed
    interval [START, END] = cfg.checkpoint_soup_range, average their meta-params
    element-wise, persist the result as a single pickle, and return its path.

    Cache: deterministic filename
    `soup_<run_id>_<start>_<end>_n<count>.pickle` under
    `$SCRATCH/checkpoints/test_cpkts/`. The source meta-train run id is
    included in the filename so concurrent meta-tests targeting different
    meta-trains do not collide on the same cached soup. Reused on
    subsequent runs unless cfg.force_resoup is set.
    """
    import wandb

    start, end = cfg.checkpoint_soup_range
    parent_dir = os.path.join(os.environ["SCRATCH"], "checkpoints/test_cpkts")
    os.makedirs(parent_dir, exist_ok=True)

    api = wandb.Api()
    run = api.run(cfg.wandb_checkpoint_id)
    all_ckpts = [x for x in run.files() if 'global_step' in x.name]

    in_range = []
    for x in all_ckpts:
        try:
            step = _extract_global_step(x.name)
        except (ValueError, IndexError):
            continue
        if start <= step <= end:
            in_range.append((step, x))

    assert len(in_range) >= 1, (
        f"No checkpoints found in range [{start}, {end}] for run "
        f"{cfg.wandb_checkpoint_id}. Available steps: "
        f"{sorted({_extract_global_step(x.name) for x in all_ckpts if 'global_step' in x.name})}"
    )

    in_range.sort(key=lambda t: t[0])
    steps_used = [s for s, _ in in_range]
    print(f"[ckpt soup] range=[{start},{end}] selecting {len(in_range)} checkpoint(s): {steps_used}")

    run_id = cfg.wandb_checkpoint_id.split('/')[-1]
    soup_path = os.path.join(
        parent_dir, f"soup_{run_id}_{start}_{end}_n{len(in_range)}.pickle"
    )
    if os.path.exists(soup_path) and not getattr(cfg, 'force_resoup', False):
        print(f"[ckpt soup] reusing cached soup: {soup_path}")
        return soup_path

    def _download_one(ckpt):
        print(f"[ckpt soup] downloading {ckpt.name}")
        ckpt.download(parent_dir, replace=True)
        local_path = osp.join(parent_dir, ckpt.name)
        if osp.exists(local_path):
            return local_path
        alt_path = osp.join(parent_dir, osp.basename(ckpt.name))
        if osp.exists(alt_path):
            return alt_path
        raise FileNotFoundError(
            f"wandb download silently failed for {ckpt.name}. Tried:\n  {local_path}\n  {alt_path}"
        )

    local_paths = [_download_one(c) for _, c in in_range]

    print(f"[ckpt soup] averaging {len(local_paths)} checkpoint(s)...")
    avg_params = _average_pickled_pytrees(local_paths)

    with open(soup_path, 'wb') as f:
        pickle.dump(avg_params, f, protocol=4)
    print(f"[ckpt soup] wrote averaged checkpoint to {soup_path}")
    return soup_path


def test_bf16_support_on_gpu():
    # Check if there is any GPU available
    gpus = jax.devices()#[device for device in jax.devices() if 'gpu' in device.device_kind.lower()]
    if not gpus:
        print("No GPU devices found.")
        return
    
    # Select the first GPU device
    gpu = gpus[0]
    jax.devices().append(gpu)
    print(f"Testing on GPU: {gpu}")

    try:
        # Create test data in BF16
        a = lax.convert_element_type(np.array([1.0, 2.0, 3.0]), jnp.bfloat16)
        b = lax.convert_element_type(np.array([1.0, 2.0, 3.0]), jnp.bfloat16)
        
        # Perform an addition operation on GPU
        result = lax.add(a, b)

        # Print the results to verify
        print("BF16 operation successful on GPU. Result:", result)
    except Exception as e:
        print(f"Failed to perform BF16 operations on GPU: {e}")

def get_resume_ckpt(ckpt_dir, meta_train_name):

    if not osp.exists(ckpt_dir):
        print("[Info] No existing checkpoint found. Starting from scratch.")
        return None

    ckpt_dirs = get_ckpt_dirs(ckpt_dir, meta_train_name)

    if len(ckpt_dirs) == 0:
        print("[Info] No existing checkpoint found. Starting from scratch.")
        return None

    if len(ckpt_dirs) == 1:
        chosen = ckpt_dirs[0]
    else:
        # Multiple run-id dirs share this meta_train_name. This happens when a
        # prior resume missed (e.g. the first run had no ckpt yet) and a fresh
        # wandb run-id dir got created alongside the old one. os.listdir order is
        # arbitrary, so returning ckpt_dirs[0] could silently rewind to stale
        # progress. Pick the dir whose `latest` points at the largest step.
        def latest_step(d):
            lp = osp.join(d, "latest")
            if not osp.isfile(lp):
                return -1
            name = open(lp).readline().strip()  # e.g. "global_step113250"
            m = re.search(r"(\d+)", name)
            return int(m.group(1)) if m else -1

        chosen = max(ckpt_dirs, key=latest_step)
        print(
            "[Warning] {} checkpoint dirs share name '{}'; resuming the newest "
            "(largest latest-step) at {}".format(
                len(ckpt_dirs), meta_train_name, chosen
            )
        )

    print("[Info] Loading checkpoint from {}".format(chosen))
    return chosen

def cast_to_fp32(pytree):
    """
    Recursively cast all JAX arrays within a PyTree to BF16.
    """
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating) else x, pytree)

def cast_to_bf16(pytree):
    """
    Recursively cast all JAX arrays within a PyTree to BF16.
    """
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16) if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating) else x, pytree)

def cast_to_fp8(pytree):
    """
    Recursively cast all JAX arrays within a PyTree to BF16.
    """
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float8) if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating) else x, pytree)


def save_timings_to_csv(timings, filename, column_name):
    """
    Saves the timings to a CSV file.

    :param timings: List of execution times.
    :param filename: Name of the file to save the data.
    :param column_name: Name of the column under which timings are saved.
    """
    # Calculate and print the average timing
    average_timing = sum(timings) / len(timings)
    print(f"Average timing: {average_timing} seconds")

    # Save the timings to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([column_name])  # Write the header
        for timing in timings:
            writer.writerow([timing])









def _compute_fans(shape, fan_in_axes=None):
  """Computes the number of input and output units for a weight shape."""
  if len(shape) < 1:
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in, fan_out = shape
  else:
    if fan_in_axes is not None:
      # Compute fan-in using user-specified fan-in axes.
      fan_in = np.prod([shape[i] for i in fan_in_axes])
      fan_out = np.prod([s for i, s in enumerate(shape)
                         if i not in fan_in_axes])
    else:
      # If no axes specified, assume convolution kernels (2D, 3D, or more.)
      # kernel_shape: (..., input_depth, depth)
      receptive_field_size = np.prod(shape[:-2])
      fan_in = shape[-2] * receptive_field_size
      fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out

class MupVarianceScaling(hk.initializers.Initializer):
  """Initializer which adapts its scale to the shape of the initialized array.

  The initializer first computes the scaling factor ``s = scale / n``, where n
  is:

    - Number of input units in the weight tensor, if ``mode = fan_in``.
    - Number of output units, if ``mode = fan_out``.
    - Average of the numbers of input and output units, if ``mode = fan_avg``.

  Then, with ``distribution="truncated_normal"`` or ``"normal"``,
  samples are drawn from a distribution with a mean of zero and a standard
  deviation (after truncation, if used) ``stddev = sqrt(s)``.

  With ``distribution=uniform``, samples are drawn from a uniform distribution
  within ``[-limit, limit]``, with ``limit = sqrt(3 * s)``.

  The variance scaling initializer can be configured to generate other standard
  initializers using the scale, mode and distribution arguments. Here are some
  example configurations:

  ==============  ==============================================================
  Name            Parameters
  ==============  ==============================================================
  glorot_uniform  VarianceScaling(1.0, "fan_avg", "uniform")
  glorot_normal   VarianceScaling(1.0, "fan_avg", "truncated_normal")
  lecun_uniform   VarianceScaling(1.0, "fan_in",  "uniform")
  lecun_normal    VarianceScaling(1.0, "fan_in",  "truncated_normal")
  he_uniform      VarianceScaling(2.0, "fan_in",  "uniform")
  he_normal       VarianceScaling(2.0, "fan_in",  "truncated_normal")
  ==============  ==============================================================
  """

  def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal',
               fan_in_axes=None):
    """Constructs the :class:`VarianceScaling` initializer.

    Args:
      scale: Scale to multiply the variance by.
      mode: One of ``fan_in``, ``fan_out``, ``fan_avg``
      distribution: Random distribution to use. One of ``truncated_normal``,
        ``normal`` or ``uniform``.
      fan_in_axes: Optional sequence of int specifying which axes of the shape
        are part of the fan-in. If none provided, then the weight is assumed
        to be like a convolution kernel, where all leading dimensions are part
        of the fan-in, and only the trailing dimension is part of the fan-out.
        Useful if instantiating multi-headed attention weights.
    """
    if scale < 0.0:
      raise ValueError('`scale` must be a positive float.')
    if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
      raise ValueError('Invalid `mode` argument:', mode)
    distribution = distribution.lower()
    if distribution not in {'normal', 'truncated_normal', 'uniform'}:
      raise ValueError('Invalid `distribution` argument:', distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.fan_in_axes = fan_in_axes

  def __call__(self, shape: Sequence[int], dtype: Any) -> jax.Array:
    scale = self.scale
    fan_in, fan_out = _compute_fans(shape, self.fan_in_axes)
    if self.mode == 'fan_in':
      scale /= max(1.0, fan_in)
    elif self.mode == 'fan_out':
      scale /= max(1.0, fan_out)
    else:
      scale /= max(1.0, (fan_in + fan_out) / 2.0)

    if self.distribution == 'truncated_normal':
      stddev = np.sqrt(scale)
      # Adjust stddev for truncation.
      # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      # distribution_stddev = np.asarray(.87962566103423978, dtype=dtype)
      # stddev = stddev / distribution_stddev
      return TruncatedNormal(stddev=stddev)(shape, dtype)
    elif self.distribution == 'normal':
      stddev = np.sqrt(scale)
      return RandomNormal(stddev=stddev)(shape, dtype)
    else:
      limit = np.sqrt(3.0 * scale)
      return RandomUniform(minval=-limit, maxval=limit)(shape, dtype)





def set_non_hashable_args(args):
    if args.run_type in ["benchmark", "sweep"]:

        if type(args.local_batch_size) == list:
            args.local_batch_size = args.local_batch_size[0]
        # Meta-testing
        # if args.optimizer.lower() in ['small_fc_mlp', 'mup_small_fc_mlp', 'adamw', 'velo', 'hyperv2','muadam','muhyperv2','murnnmlplopt','RNNMLPLOpt'.lower()]:

        if args.use_localsgd_batches:
            args.batch_shape = (args.num_grads, args.num_local_steps, args.gradient_accumulation_steps, args.local_batch_size,)
            args.label_sharding = PositionalSharding(mesh_utils.create_device_mesh((args.num_devices,)))
            args.image_sharding = PositionalSharding(mesh_utils.create_device_mesh((args.num_devices,1,1,1))) 

            args.meta_testing_batch_size = args.num_grads \
                                            * args.num_local_steps \
                                            * args.local_batch_size \
                                            * args.gradient_accumulation_steps
        else:
            args.meta_testing_batch_size = args.local_batch_size * args.gradient_accumulation_steps
            args.batch_shape = (args.gradient_accumulation_steps,args.local_batch_size,)
            args.label_sharding = None
            # PositionalSharding(mesh_utils.create_device_mesh((args.num_devices,)))
            args.image_sharding = None
            # PositionalSharding(mesh_utils.create_device_mesh((args.num_devices,1,1,1)))


    else:
        # Meta-training
        if not args.use_localsgd_batches:
            
            args.meta_training_batch_args = []
            K = getattr(args, 'gradient_accumulation_steps', 1)
            for bsz in args.local_batch_size:
                temp = {}

                if K > 1:
                    temp["batch_shape"] = (args.steps_per_jit, args.num_tasks, K, bsz)
                    temp["label_sharding"] = (1, 1, 1, args.num_devices)
                    temp["image_sharding"] = (1, 1, 1, args.num_devices, 1, 1, 1)
                else:
                    temp["batch_shape"] = (args.steps_per_jit, args.num_tasks, bsz)
                    temp["label_sharding"] = (1, 1, args.num_devices)
                    temp["image_sharding"] = (1, 1, args.num_devices, 1, 1, 1)
                temp["meta_training_batch_size"] = bsz \
                                                    * args.num_tasks \
                                                    * args.steps_per_jit \
                                                    * K

                
                args.meta_training_batch_args.append(temp)
            


        else:
            args.meta_training_batch_args = []
            for bsz in args.local_batch_size:
                temp = {}

                if args.num_devices > 1:
                    temp["batch_shape"] = (args.gradient_estimator_args['kwargs']['steps_per_jit'], args.num_tasks, args.num_grads * args.num_local_steps * bsz)
                    temp["label_sharding"] = PositionalSharding(mesh_utils.create_device_mesh((1,1,args.num_devices)))
                    temp["image_sharding"] = PositionalSharding(mesh_utils.create_device_mesh((1,1,args.num_devices,1,1,1)))
                    temp["meta_training_batch_size"] = args.num_grads \
                                                        * args.num_local_steps \
                                                        * bsz \
                                                        * args.num_tasks \
                                                        * args.gradient_estimator_args['kwargs']['steps_per_jit']
                else:
                    temp["batch_shape"] = (args.gradient_estimator_args['kwargs']['steps_per_jit'], args.num_tasks, args.num_grads * args.num_local_steps * bsz)
                    temp["label_sharding"] = (1,1,args.num_devices)
                    temp["image_sharding"] = (1,1,args.num_devices,1,1,1)
                    temp["meta_training_batch_size"] = args.num_grads \
                                                        * args.num_local_steps \
                                                        * bsz \
                                                        * args.num_tasks \
                                                        * args.gradient_estimator_args['kwargs']['steps_per_jit']
                
                args.meta_training_batch_args.append(temp)
                print(temp)


    return args



