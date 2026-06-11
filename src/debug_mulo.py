from mup_adafac_mlp_lopt import MuAdafacMLPLOpt
import pickle
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from helpers import MupVarianceScaling
from typing import Any, Mapping, Tuple, Callable, Optional
from collections.abc import Iterable
import pprint
class MuMLP(hk.Module):
  """A multi-layer perceptron module."""

  def __init__(
      self,
      output_sizes: Iterable[int],
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      input_mult=1.0,
      output_mult=1.0,
      hidden_lr_mult=1.0,
      with_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      activate_final: bool = False,
      log_activations: bool = False,
      name: Optional[str] = None,
  ):
    if not with_bias and b_init is not None:
      raise ValueError("When with_bias=False b_init must not be set.")

    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init
    self.activation = activation
    self.activate_final = activate_final
    self.get_adam_mup_lr_mul = {}
    self.log_activations = log_activations
    layers = []
    output_sizes = tuple(output_sizes)
    np_w1 = np.random.randn(4, 8).astype(np.float32)
    np_b1 = np.random.randn(4).astype(np.float32)
    np_wm = np.random.randn(4, 4).astype(np.float32)
    np_bm = np.random.randn(4).astype(np.float32)
    np_w3 = np.random.randn(10, 4).astype(np.float32)
    np_b3 = np.random.randn(10).astype(np.float32)
    for index, output_size in enumerate(output_sizes):
      if index ==0:
        #input layer
        layers.append(hk.Linear(output_size=output_size,
                                # w_init=MupVarianceScaling(1.0, "fan_in",  "truncated_normal"),
                                # b_init=hk.initializers.RandomNormal(stddev=1., mean=0.),
                                w_init=lambda *_: jnp.array(np_w1.T), b_init=lambda *_: jnp.array(np_b1),
                                with_bias=with_bias,
                                name="linear_%d" % index))
        self.get_adam_mup_lr_mul["mu_mlp/~/linear_%d"  % index] = {'w':1.0,'b':1.0}
        
      elif index == len(output_sizes) - 1:
        #output layer
        layers.append(hk.Linear(output_size=output_size,
                                # w_init=jnp.zeros,# RandomNormal(stddev=1., mean=0.),
                                # b_init=hk.initializers.RandomNormal(stddev=1., mean=0.),
                                w_init=lambda *_: jnp.array(np_w3.T), b_init=lambda *_: jnp.array(np_b3),
                                with_bias=with_bias,
                                name="linear_%d" % index))
        self.get_adam_mup_lr_mul["mu_mlp/~/linear_%d"  % index] = {'w':1.0,'b':1.0}
      else:
        #hidden layer
        layers.append(hk.Linear(output_size=output_size,
                                # w_init=MupVarianceScaling(1.0, "fan_in",  "truncated_normal"),
                                # b_init=hk.initializers.RandomNormal(stddev=1., mean=0.),
                                w_init=lambda *_: jnp.array(np_wm.T), b_init=lambda *_: jnp.array(np_bm),
                                with_bias=with_bias,
                                name="linear_%d" % index))
        self.get_adam_mup_lr_mul["mu_mlp/~/linear_%d"  % index] = {'w': hidden_lr_mult / output_sizes[index-1] ,'b':1.0}
        
    self.layers = tuple(layers)
    self.output_size = output_sizes[-1] if output_sizes else None
    
    assert len(output_sizes) >= 2, "need more than one layer for MuMLP"

    self.input_mult = input_mult
    self.hidden_mult = 1.0
    self.output_mul =  output_mult * 1 / output_sizes[-2]
    hk.set_state("mup_lrs", self.get_adam_mup_lr_mul)

  @property
  def mup_lrs(self):
    return hk.get_state("mup_lrs")
  
  def __call__(
      self,
      inputs: jax.Array,
      dropout_rate: Optional[float] = None,
      rng=None,
  ) -> jax.Array:
    if dropout_rate is not None and rng is None:
      raise ValueError("When using dropout an rng key must be passed.")
    elif dropout_rate is None and rng is not None:
      raise ValueError("RNG should only be passed when using dropout.")

    rng = hk.PRNGSequence(rng) if rng is not None else None
    num_layers = len(self.layers)
    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i == 0:
        out = out * self.input_mult
      elif i < (num_layers - 1):
        out = out * self.hidden_mult

      if self.log_activations:
        hk.set_state("layer_%d_pre-act_l1" % i, jnp.mean(jnp.abs(out)))
        hk.set_state("layer_%d_pre-act" % i, out)

      # hk.set_state("layer_%d_act_l1" % i, jnp.mean(jnp.abs(out)))
      if i < (num_layers - 1) or self.activate_final:
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          out = hk.dropout(next(rng), dropout_rate, out)
        out = self.activation(out)

        if self.log_activations:
          hk.set_state("layer_%d_act_l1" % i, jnp.mean(jnp.abs(out)))
          hk.set_state("layer_%d_act" % i, out)
      else:
        if self.log_activations:
          hk.set_state("layer_%d_logits_l1" % i, jnp.mean(jnp.abs(out * self.output_mul)))
          hk.set_state("layer_%d_logits" % i, out * self.output_mul)

    return out * self.output_mul

def my_linear_module(np_w, np_b):
    def forward(x):
        linear = hk.Linear(output_size=4, w_init=lambda *_: jnp.array(np_w.T), b_init=lambda *_: jnp.array(np_b))
        return linear(x)
    return hk.transform(forward)

def my_mlp_module():
    mup_multipliers=dict(input_mult=1.0,
                                    output_mult=1.0,
                                    hidden_lr_mult=1.0)
    def forward(x):
        x = jnp.reshape(x, [x.shape[0], -1])
        return MuMLP(
          [4, 4, 4, 10], activation=jax.nn.relu,log_activations=False,
              **mup_multipliers)(
              x, dropout_rate=0.0, 
              rng=hk.next_rng_key())
    return hk.transform_with_state(forward)

def main():
    np.random.seed(42)
    lopt = MuAdafacMLPLOpt(exp_mult=0.001,
                                step_mult=0.01,
                                hidden_size=32,
                                hidden_layers=2,
                                initial_momentum_decays=(0.9, 0.99, 0.999),
                                initial_rms_decays=(0.999,),
                                initial_adafactor_decays=(0.9, 0.99, 0.999),
                                concat_weights=True,
                                make_separate_weights=False,
                                split_weights=False,
                                clip_grad=False,
                                zero_training_step_feature=False)

    print("\n\nnewnew\n\n")
    jax.config.update("jax_default_matmul_precision", "highest")
    jnp.set_printoptions(precision=8, floatmode='fixed')
    np.set_printoptions(precision=8, suppress=True)

    with open('/home/btherien/github/scaling_l2o/global_step5000.pickle', "rb") as f:
            meta_params = pickle.load(f)
            # print('momentum_decays', meta_params['momentum_decays'].shape, meta_params['momentum_decays'])
            # print('rms_decays', meta_params['rms_decays'].shape, meta_params['rms_decays'])
            # print('adafactor_decays', meta_params['adafactor_decays'].shape, meta_params['adafactor_decays'])
            # for k, v in meta_params['nn']['~'].items():
            #      print(k, v.shape, v)
    opt = lopt.opt_fn(meta_params)

    rng = jax.random.PRNGKey(42)
    x = jnp.array(np.random.rand(1, 8))
    # print('input:', x)
    model = my_mlp_module()
    params, state = model.init(rng, x)
    print('params:') 
    pprint.pprint(params)

    model_state = {}
    model_state['mup_lrs_to_use'] = jax.tree.map(lambda x: jnp.array(x), state['mu_mlp']['mup_lrs'])
    print('mup_lrs_to_use:',)
    pprint.pprint( model_state['mup_lrs_to_use'])
    def loss_fn(params, rng, x):
        output, _ = model.apply(params, state, rng, x)
        # print('output:', output)
        return ((1-output)**2).sum()
    opt_state = opt.init(params, model_state=model_state, num_steps=jnp.asarray(0))

    test_steps = 5
    for i in range(test_steps):
        np.random.seed(i)
        input_t = np.random.rand(1, 8)
        x = jnp.array(input_t)
        print('iter:', i)
        grads =  jax.grad(loss_fn)(opt_state.params, rng, x)
        opt_state = opt.update(opt_state, grads, loss=jnp.array(1.0), model_state=model_state)
        print('params:') 
        pprint.pprint(opt_state.params)
        # print('grads:', grads)

if __name__ == '__main__':
    main()



# python debug_mulo.py > /home/btherien/jax_log.txt 2>&1