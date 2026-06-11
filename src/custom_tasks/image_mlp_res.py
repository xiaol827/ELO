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

"""Tasks based on MLP."""
# pylint: disable=invalid-name

from collections.abc import Iterable
from typing import Any, Mapping, Tuple

from typing import Callable, Optional
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import image
import numpy as onp

State = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray




class ResMLP(hk.Module):
  """A multi-layer perceptron module."""

  def __init__(
      self,
      output_sizes: Iterable[int],
      w_init: hk.initializers.Initializer = None,
      b_init: hk.initializers.Initializer = None,
      with_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      activate_final: bool = False,
      name: str = None,
  ):
    """Constructs an MLP.

    Args:
      output_sizes: Sequence of layer sizes.
      w_init: Initializer for :class:`~haiku.Linear` weights.
      b_init: Initializer for :class:`~haiku.Linear` bias. Must be ``None`` if
        ``with_bias=False``.
      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between :class:`~haiku.Linear`
        layers. Defaults to ReLU.
      activate_final: Whether or not to activate the final layer of the MLP.
      name: Optional name for this module.

    Raises:
      ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.
    """
    if not with_bias and b_init is not None:
      raise ValueError("When with_bias=False b_init must not be set.")

    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init
    self.activation = activation
    self.activate_final = activate_final
    layers = []
    output_sizes = tuple(output_sizes)
    for index, output_size in enumerate(output_sizes):
      layers.append(hk.Linear(output_size=output_size,
                              w_init=w_init,
                              b_init=b_init,
                              with_bias=with_bias,
                              name="linear_%d" % index))
    self.layers = tuple(layers)
    self.output_size = output_sizes[-1] if output_sizes else None

  def __call__(
      self,
      inputs: jax.Array,
      dropout_rate: float = None,
      rng=None,
  ) -> jax.Array:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of shape ``[batch_size, input_size]``.
      dropout_rate: Optional dropout rate.
      rng: Optional RNG key. Require when using dropout.

    Returns:
      The output of the model of size ``[batch_size, output_size]``.
    """
    if dropout_rate is not None and rng is None:
      raise ValueError("When using dropout an rng key must be passed.")
    elif dropout_rate is None and rng is not None:
      raise ValueError("RNG should only be passed when using dropout.")

    rng = hk.PRNGSequence(rng) if rng is not None else None
    num_layers = len(self.layers)

    out = inputs
    for i, layer in enumerate(self.layers):
      res = out
      if i == 0:
        out = layer(out) 
      elif i < (num_layers - 1):
        out = layer(out) + res
      else:
        out = layer(out)

      if i < (num_layers - 1) or self.activate_final:
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          out = hk.dropout(next(rng), dropout_rate, out)
        out = self.activation(out)

    return out



class _ResMLPImageTask(base.Task):
  """MLP based image task."""

  def __init__(self,
               datasets,
               hidden_sizes,
               act_fn=jax.nn.relu,
                log_activations=False,
               dropout_rate=0.0):
    super().__init__()
    num_classes = datasets.extra_info["num_classes"]
    sizes = list(hidden_sizes) + [num_classes]
    self.datasets = datasets

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      return ResMLP(
          sizes, activation=act_fn)(
              inp, dropout_rate=dropout_rate, rng=hk.next_rng_key())

    self._mod = hk.transform(_forward)

  def init(self, key: PRNGKey) -> Any:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._mod.init(key, batch["image"])

  def loss(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:  # pytype: disable=signature-mismatch  # jax-ndarray
    num_classes = self.datasets.extra_info["num_classes"]
    logits = self._mod.apply(params, key, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss)

  def loss_and_accuracy(self, params: Params, key: PRNGKey, data: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:  # pytype: disable=signature-mismatch  # jax-ndarray
    num_classes = self.datasets.extra_info["num_classes"]

    threshold = 10000
    if data["image"].shape[0] > threshold:
      # If the batch is too large, we split it into smaller chunks.
      # This is to avoid running out of memory.
      # This is not necessary for the task to work, but it is useful for
      # large batch sizes.
      data["image"] = jnp.array_split(data["image"], 
                                      find_smallest_divisor(data["image"].shape[0],threshold), 
                                      axis=0)
      # print(jax.tree_util.tree_map(lambda x: x.shape, data["image"]))
      # exit(0)
      logits = jnp.concatenate( # pylint: disable=g-complex-comprehension
          [self._mod.apply(params, key, chunk) for chunk in data["image"]])
    else:
      logits = self._mod.apply(params, key, data["image"])
    
    # Calculate the loss as before
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    loss = jnp.mean(vec_loss)
    
    # Calculate the accuracy
    predictions = jnp.argmax(logits, axis=-1)
    actual = data["label"]
    correct_predictions = predictions == actual
    accuracy = jnp.mean(correct_predictions.astype(jnp.float32))
    
    return loss, accuracy

  def normalizer(self, loss):
    num_classes = self.datasets.extra_info["num_classes"]
    maxval = 1.5 * onp.log(num_classes)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)
  