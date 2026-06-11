# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resnet haiku module.

Fork from
https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py
to have more configuration options.
"""
import copy
import functools
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import chex
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks import resnet
from learned_optimization.tasks.datasets import image

from .mu_task_base import MuTask

FloatStrOrBool = Union[str, float, bool]

import numpy as onp


class MuBlockV1(hk.Module):
  """ResNet V1 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      w_init: hk.initializers.Initializer,
      b_init: hk.initializers.Initializer,
      input_channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      bn_config: Mapping[str, FloatStrOrBool],
      bottleneck: bool,
      name: Optional[str] = None,
      use_residual: bool = True,
      use_kernel_mult=False,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection
    self.use_residual = use_residual

    mup_lrs = {}

    bn_config = dict(bn_config)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)
    bn_config.setdefault("decay_rate", 0.999)

    if self.use_projection:
      kernel_shape = 1
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=kernel_shape,
          stride=stride,
          w_init=w_init,
          b_init=b_init,
          with_bias=False,
          padding="SAME",
          name="shortcut_conv")
      if not use_kernel_mult:
        mup_lrs['~/shortcut_conv'] = {'w': 1. / ( copy.deepcopy(input_channels) * kernel_shape * kernel_shape )} # MUPLR
      else:
        mup_lrs['~/shortcut_conv'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
      input_channels = channels

      self.proj_batchnorm = hk.BatchNorm(name="shortcut_batchnorm", **bn_config)
      mup_lrs['~/shortcut_batchnorm'] = {'scale': 1., 'offset': 1.} # MUPLR

    channel_div = 4 if bottleneck else 1
    kernel_shape = 1 if bottleneck else 3
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=kernel_shape,
        stride=1 if bottleneck else stride,
        w_init=w_init,
        b_init=b_init,
        with_bias=False,
        padding="SAME",
        name="conv_0")
    if not use_kernel_mult:
      mup_lrs['~/conv_0'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
    else:
      mup_lrs['~/conv_0'] = {'w': 1. / ( copy.deepcopy(input_channels) * kernel_shape * kernel_shape )} # MUPLR
    

    input_channels = channels // channel_div

    bn_0 = hk.BatchNorm(name="batchnorm_0", **bn_config)
    mup_lrs['~/batchnorm_0'] = {'scale': 1., 'offset': 1.} # MUPLR

    kernel_shape = 3
    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=kernel_shape,
        stride=stride if bottleneck else 1,
        with_bias=False,
        w_init=w_init,
        b_init=b_init,
        padding="SAME",
        name="conv_1")
    if not use_kernel_mult:
      mup_lrs['~/conv_1'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
    else:
      mup_lrs['~/conv_1'] = {'w': 1. / ( copy.deepcopy(input_channels) * kernel_shape * kernel_shape )} # MUPLR
    input_channels = channels // channel_div

    bn_1 = hk.BatchNorm(name="batchnorm_1", **bn_config)
    mup_lrs['~/batchnorm_1'] = {'scale': 1., 'offset': 1.} # MUPLR

    layers = ((conv_0, bn_0), (conv_1, bn_1))

    if bottleneck:
      kernel_shape = 1
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=kernel_shape,
          stride=1,
          w_init=w_init,
          b_init=b_init,
          with_bias=False,
          padding="SAME",
          name="conv_2")
          
      if not use_kernel_mult:
        mup_lrs['~/conv_2'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
      else:
        mup_lrs['~/conv_2'] = {'w': 1. / ( copy.deepcopy(input_channels) * kernel_shape * kernel_shape )} # MUPLR


      # mup_lrs['~/conv_2'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
      input_channels = channels

      bn_2 = hk.BatchNorm(name="batchnorm_2", **bn_config)
      mup_lrs['~/batchnorm_2'] = {'scale': 1., 'offset': 1.} # MUPLR

      layers = layers + ((conv_2, bn_2),)

    self.layers = layers


    hk.set_state("mup_lrs",mup_lrs)

  def __call__(self, inputs, is_training, test_local_stats):
    out = shortcut = inputs

    if self.use_projection:
      shortcut = self.proj_conv(shortcut)
      shortcut = self.proj_batchnorm(shortcut, is_training, test_local_stats)

    for i, (conv_i, bn_i) in enumerate(self.layers):
      out = conv_i(out)
      out = bn_i(out, is_training, test_local_stats)
      if i < len(self.layers) - 1:  # Don't apply relu on last layer
        out = jax.nn.relu(out)

    if self.use_residual:
      return jax.nn.relu(out + shortcut)
    else:
      return jax.nn.relu(out)

class MuBlockV2(hk.Module):
  """ResNet V2 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      w_init: hk.initializers.Initializer,
      b_init: hk.initializers.Initializer,
      input_channels: int,
      use_projection: bool,
      bn_config: Mapping[str, FloatStrOrBool],
      bottleneck: bool,
      name: Optional[str] = None,
      use_residual: bool = True,
      use_kernel_mult = False,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection
    self.use_residual = use_residual

    bn_config = dict(bn_config)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)

    if self.use_projection:
      kernel_shape = 1
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=kernel_shape,
          stride=stride,
          with_bias=False,
          w_init=w_init,
          b_init=b_init,
          padding="SAME",
          name="shortcut_conv")
          
      if use_kernel_mult:
        mup_lrs['~/shortcut_conv'] = {'w': 1. / ( copy.deepcopy(input_channels) * kernel_shape * kernel_shape )} # MUPLR
      else:
        mup_lrs['~/shortcut_conv'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR


      # mup_lrs['~/shortcut_conv'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
      input_channels = channels

    channel_div = 4 if bottleneck else 1
    kernel_shape = 1 if bottleneck else 3
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=kernel_shape,
        stride=1 if bottleneck else stride,
        with_bias=False,
          w_init=w_init,
          b_init=b_init,
        padding="SAME",
        name="conv_0")
          
    if not use_kernel_mult:
      mup_lrs['~/conv_0'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
    else:
      mup_lrs['~/conv_0'] = {'w': 1. / ( copy.deepcopy(input_channels) * kernel_shape * kernel_shape )} # MUPLR

    # mup_lrs['~/conv_0'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
    input_channels = channels // channel_div

    bn_0 = hk.BatchNorm(name="batchnorm_0", **bn_config)
    mup_lrs['~/batchnorm_0'] = {'scale': 1., 'offset': 1.} # MUPLR
    kernel_shape=3
    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=kernel_shape,
        stride=stride if bottleneck else 1,
        with_bias=False,
          w_init=w_init,
          b_init=b_init,
        padding="SAME",
        name="conv_1")
          
    if not use_kernel_mult:
      mup_lrs['~/conv_1'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
    else:
      mup_lrs['~/conv_1'] = {'w': 1. / ( copy.deepcopy(input_channels) * kernel_shape * kernel_shape )} # MUPLR

    # mup_lrs['~/conv_1'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
    input_channels = channels // channel_div

    bn_1 = hk.BatchNorm(name="batchnorm_1", **bn_config)
    mup_lrs['~/batchnorm_1'] = {'scale': 1., 'offset': 1.} # MUPLR
    layers = ((conv_0, bn_0), (conv_1, bn_1))

    if bottleneck:
      kernel_shape = 1
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=kernel_shape,
          stride=1,
          with_bias=False,
          w_init=w_init,
          b_init=b_init,
          padding="SAME",
          name="conv_2")
          
      if not use_kernel_mult:
        mup_lrs['~/conv_2'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
      else:
        mup_lrs['~/conv_2'] = {'w': 1. / ( copy.deepcopy(input_channels) * kernel_shape * kernel_shape )} # MUPLR

      # mup_lrs['~/conv_2'] = {'w': 1. / copy.deepcopy(input_channels)} # MUPLR
      input_channels = channels

      # NOTE: Some implementations of ResNet50 v2 suggest initializing
      # gamma/scale here to zeros.
      bn_2 = hk.BatchNorm(name="batchnorm_2", **bn_config)
      mup_lrs['~/batchnorm_2'] = {'scale': 1., 'offset': 1.} # MUPLR
      layers = layers + ((conv_2, bn_2),)

    self.layers = layers

  def __call__(self, inputs, is_training, test_local_stats):
    x = shortcut = inputs

    for i, (conv_i, bn_i) in enumerate(self.layers):
      x = bn_i(x, is_training, test_local_stats)
      x = jax.nn.relu(x)
      if i == 0 and self.use_projection:
        shortcut = self.proj_conv(x)
      x = conv_i(x)

    if self.use_residual:
      x = x + shortcut

    return x


class MuBlockGroup(hk.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      channels: int,
      num_blocks: int,
      w_init: hk.initializers.Initializer,
        b_init: hk.initializers.Initializer,
        input_channels: int,
      stride: Union[int, Sequence[int]],
      bn_config: Mapping[str, FloatStrOrBool],
      resnet_v2: bool,
      bottleneck: bool,
      use_projection: bool,
      name: Optional[str] = None,
      use_residual: bool = True,
      use_kernel_mult = False,
  ):
    super().__init__(name=name)

    block_cls = MuBlockV2 if resnet_v2 else MuBlockV1

    self.blocks = []
    for i in range(num_blocks):
      self.blocks.append(
          block_cls(
              w_init = w_init,
              b_init = b_init,
              input_channels=input_channels,
              channels=channels,
              stride=(1 if i else stride),
              use_projection=(i == 0 and use_projection),
              bottleneck=bottleneck,
              bn_config=bn_config,
              use_residual=use_residual,
              use_kernel_mult=use_kernel_mult,
              name="block_%d" % (i)),)

  def __call__(self, inputs, is_training, test_local_stats):
    out = inputs
    for block in self.blocks:
      out = block(out, is_training, test_local_stats)
    return out


def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class MuResNet(hk.Module):
  """ResNet model."""

  CONFIGS = {
      18: {
          "blocks_per_group": (2, 2, 2, 2),
          "bottleneck": False,
          "channels_per_group": (64, 128, 256, 512),
          "use_projection": (False, True, True, True),
      },
      34: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": False,
          "channels_per_group": (64, 128, 256, 512),
          "use_projection": (False, True, True, True),
      },
      50: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      101: {
          "blocks_per_group": (3, 4, 23, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      152: {
          "blocks_per_group": (3, 8, 36, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      200: {
          "blocks_per_group": (3, 24, 36, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
  }

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      num_classes: int,
      bn_config: Optional[Mapping[str, float]] = None,
      resnet_v2: bool = False,
      bottleneck: bool = True,
      channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
      use_projection: Sequence[bool] = (True, True, True, True),
      logits_config: Optional[Mapping[str, Any]] = None,
      input_channels: int = 3,
      initial_conv_channels: int = 64,
      initial_conv_kernel_size: int = 7,
      initial_conv_stride: int = 2,
      input_mult: float = 1.0,
      output_mult: float = 1.0,
      hidden_lr_mult: float = 1.0,
      max_pool: bool = True,
      act_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      name: Optional[str] = None,
      use_residual: bool = True,
      use_kernel_mult = False,
  ):
    """Constructs a ResNet model.

    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
          ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number of
        channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      logits_config: A dictionary of keyword arguments for the logits layer.
      initial_conv_channels: channels in initial conv layer.
      initial_conv_kernel_size: size of initial conv kernel.
      initial_conv_stride: initial conv stride.
      max_pool: To perform an initial max pool or not.
      act_fn: Activation function to use.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.resnet_v2 = resnet_v2
    self.max_pool = max_pool
    self.act_fn = act_fn
    self.use_residual = use_residual


    self._imput_w_init = hk.initializers.VarianceScaling(1.0, "fan_in",  "normal")
    self._hidden_w_init = hk.initializers.VarianceScaling(1.0, "fan_in",  "normal")
    self._output_w_init = jnp.zeros

    # the bias is an input weight whose input dimension is always 1
    self._b_init = hk.initializers.RandomNormal(stddev=1., mean=0.)

    bn_config = dict(bn_config or {})
    bn_config.setdefault("decay_rate", 0.9)
    bn_config.setdefault("eps", 1e-5 / channels_per_group[0])
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)
    bn_config.setdefault("scale_init", jnp.ones) #weight
    bn_config.setdefault("offset_init", jnp.zeros) #bias

    logits_config = dict(logits_config or {})
    logits_config.setdefault("w_init", jnp.zeros)
    logits_config.setdefault("b_init", self._b_init)
    logits_config.setdefault("name", "logits")

    # Number of blocks in each group for ResNet.
    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")
    mup_lrs = {}

    self.initial_conv = hk.Conv2D(
        output_channels=initial_conv_channels,
        kernel_shape=initial_conv_kernel_size,
        stride=initial_conv_stride,
        w_init=self._imput_w_init,
        b_init=self._b_init,
        with_bias=False,
        padding="SAME",
        name="initial_conv")
    mup_lrs['~/initial_conv'] = {'w': 1.} # MUPLR

    if not self.resnet_v2:
      self.initial_batchnorm = hk.BatchNorm(
          name="initial_batchnorm", **bn_config)
      mup_lrs['~/initial_batchnorm'] = {'scale': 1., 'offset': 1.} # MUPLR

    self.block_groups = []
    strides = (1, 2, 2, 2)
    for i in range(4):
      self.block_groups.append(
          MuBlockGroup(
              input_channels = channels_per_group[i-1] if i > 0 else initial_conv_channels,
              w_init = self._hidden_w_init,
              b_init = self._b_init,
              channels=channels_per_group[i],
              num_blocks=blocks_per_group[i],
              stride=strides[i],
              bn_config=bn_config,
              resnet_v2=resnet_v2,
              bottleneck=bottleneck,
              use_residual=use_residual,
              use_projection=use_projection[i],
              use_kernel_mult=use_kernel_mult,
              name="block_group_%d" % (i)))

    if self.resnet_v2:
      self.final_batchnorm = hk.BatchNorm(name="final_batchnorm", **bn_config)
      mup_lrs['~/final_batchnorm'] = {'scale': 1., 'offset': 1.} # MUPLR

    print("logits_config",logits_config)
    self.logits = hk.Linear(num_classes, **logits_config)
    mup_lrs['~/logits'] = {'w': 1., 'b': 1.} # MUPLR

    #MuP Init below
    self._output_mult = jnp.array([output_mult /  channels_per_group[-1]], dtype=jnp.bfloat16) #num channels in the last layer
    self._input_mult =  jnp.array([input_mult], dtype=jnp.bfloat16) 
    self._hidden_lr_mult =  hidden_lr_mult

    hk.set_state("mup_lrs",mup_lrs)








  def __call__(self, inputs, is_training, test_local_stats=False):
    out = inputs
    out = self.initial_conv(out) #* self._input_mult

    if not self.resnet_v2:
      out = self.initial_batchnorm(out, is_training, test_local_stats)
      out = self.act_fn(out)

    if self.max_pool:
      out = hk.max_pool(
          out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME")

    for block_group in self.block_groups:
      out = block_group(out, is_training, test_local_stats)

    if self.resnet_v2:
      out = self.final_batchnorm(out, is_training, test_local_stats)
      out = self.act_fn(out)
    out = jnp.mean(out, axis=[1, 2])

    return self.logits(out) * self._output_mult



class _MuResnetTaskDataset(base.Task, MuTask):
  """Tranformer from a dictionary configuration."""

  def __init__(self, 
                datasets, 
                cfg: Mapping[str, Any], 
                name: str = '__Resnet',
                use_kernel_mult=False,
                mup_multipliers=dict(input_mult=1.0,
                                        output_mult=1.0,
                                        hidden_mult=1.0),):

    self.datasets = datasets
    self.input_channels = datasets.abstract_batch['image'].shape[-1]
    self._cfg = cfg
    self._net = hk.transform_with_state(self._hk_forward)
    self._name = name
    self.use_kernel_mult = use_kernel_mult
    

    self.mup_state = None
    self.eps_mult = 1 / self._cfg['channels_per_group'][0]
    self.init_mup_state()

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
    mod = MuResNet(
      use_kernel_mult=self.use_kernel_mult,
        num_classes=num_classes, input_channels=self.input_channels,**{k: self._cfg[k] for k in args})
    logits = mod(batch['image'], is_training=True)
    loss = base.softmax_cross_entropy(
        logits=logits, labels=jax.nn.one_hot(batch['label'], num_classes))
    return jnp.mean(loss)

  def _hk_forward_eval(self, batch):
    args = [
        'blocks_per_group', 'use_projection', 'channels_per_group',
        'initial_conv_kernel_size', 'initial_conv_stride', 'max_pool',
        'resnet_v2'
    ]
    num_classes = self.datasets.extra_info['num_classes']
    mod = MuResNet(
      use_kernel_mult=self.use_kernel_mult,
        num_classes=num_classes, input_channels=self.input_channels,**{k: self._cfg[k] for k in args})
    logits = mod(batch['image'], is_training=False)
    return logits

  def init_with_state(self, key: chex.PRNGKey) -> base.Params:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    params, state = self._net.init(key, batch)
    return params, self.get_mup_state(state, eps_mult=self.eps_mult)

  def init(self, key: chex.PRNGKey) -> base.Params:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    params, state = self._net.init(key, batch)
    return params


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
    loss, state = self._net.apply(params, state, key, data)
    return loss, state, {}
  
  def normalizer(self, loss):
    num_classes = self.datasets.extra_info["num_classes"]
    maxval = 1.5 * onp.log(num_classes)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)

  # @functools.partial(jax.jit, static_argnums=(0,))
  # def loss_and_accuracy_with_state(self, params, state, key, data):
  #   loss, acc = self.loss_and_accuracy(params, key, data)
  #   return loss, acc

  # @functools.partial(jax.jit, static_argnums=(0,))
  # def loss_and_accuracy(self, params, key, data):  # pytype: disable=signature-mismatch  # jax-ndarray
  #   num_classes = self.datasets.extra_info["num_classes"]
    
  #   # Create a separate transform for evaluation
  #   eval_net = hk.transform_with_state(self._hk_forward_eval)
  #   logits, _ = eval_net.apply(params, {}, key, data)
    
  #   # Calculate the loss as before
  #   labels = jax.nn.one_hot(data["label"], num_classes)
  #   vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
  #   loss = jnp.mean(vec_loss)
    
  #   # Calculate the accuracy
  #   predictions = jnp.argmax(logits, axis=-1)
  #   actual = data["label"]
  #   correct_predictions = predictions == actual
  #   accuracy = jnp.mean(correct_predictions.astype(jnp.float32))
    
  #   return loss, accuracy
