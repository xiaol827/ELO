import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, Optional, Tuple, Union
import functools
from functools import partial
from einops import rearrange
import math

@partial(jax.jit, static_argnums=(1,))
def dct_1d(x: jnp.ndarray, norm: Optional[str] = None) -> jnp.ndarray:
    """
    Discrete Cosine Transform, Type II (DCT-II)
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = jnp.reshape(x, (-1, N))
    even = x[:, ::2]
    odd = x[:, 1::2]
    odd = jnp.flip(odd, axis=-1)
    v = jnp.concatenate([even, odd], axis=-1)

    Vc = jnp.fft.fft(v, axis=1)
    Vc_real = jnp.real(Vc)
    Vc_imag = jnp.imag(Vc)

    k = -jnp.arange(N, dtype=x.dtype) * jnp.pi / (2 * N)
    W_r = jnp.cos(k)[jnp.newaxis, :]
    W_i = jnp.sin(k)[jnp.newaxis, :]

    V = Vc_real[..., :N] * W_r - Vc_imag[..., :N] * W_i

    if norm == "ortho":
        V = V.at[:, 0].set(V[:, 0] / (math.sqrt(N) * 2))
        V = V.at[:, 1:].set(V[:, 1:] / (math.sqrt(N / 2) * 2))

    V = 2 * jnp.reshape(V, x_shape)

    return V

@partial(jax.jit, static_argnums=(1,))
def idct_1d(X: jnp.ndarray, norm: Optional[str] = None) -> jnp.ndarray:
    x_shape = X.shape
    N = x_shape[-1]
    X = jnp.reshape(X, (-1, N))

    X_v = X / 2.0

    if norm == "ortho":
        X_v = X_v.at[:, 0].multiply(math.sqrt(N) * 2)
        X_v = X_v.at[:, 1:].multiply(math.sqrt(N / 2) * 2)

    k = jnp.arange(N, dtype=X.dtype) * math.pi / (2 * N)
    W_r = jnp.cos(k)[jnp.newaxis, :]
    W_i = jnp.sin(k)[jnp.newaxis, :]

    V_t_r = X_v
    V_t_i = jnp.concatenate([
        jnp.zeros_like(X_v[:, :1]),
        -jnp.flip(X_v[:, 1:], axis=-1)
    ], axis=-1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = jnp.stack([V_r, V_i], axis=-1)
    v = jnp.fft.irfft(jax.lax.complex(V[...,0], V[...,1]), n=N, axis=1)
    x = jnp.zeros_like(v)

    flipped_v = jnp.flip(v, axis=-1)
    
    x = x.at[:, ::2].set(v[:, :N - (N//2)])
    x = x.at[:, 1::2].set(flipped_v[:, :N//2])

    return jnp.reshape(x, x_shape)

def smaller_split(s: int, target: int) -> int:
    divs = [d for d in range(1, s + 1) if s % d == 0]
    for i, d in enumerate(divs):
        if d == target:
            return d
        if d > target:
            return divs[i - 1] if i > 0 else d
    return s   

class TransformDCT:
    """Chunk-wise DCT encoder/decoder (supports 1-D or 2-D weight)."""
    def __init__(self, chunk: int = 64, norm: str = "ortho"):
        self.chunk = chunk
        self.norm = norm
        self.cache = {}

    # def _basis(self, n: int, device):
    #     key = (n, id(device))
    #     if key not in self.cache:
    #         I = jnp.eye(n, dtype=jnp.float32)
    #         F = dct_1d(I, self.norm)
    #         B = idct_1d(I, self.norm)
    #         self.cache[key] = (jax.device_put(F, device), jax.device_put(B, device))
    #     return self.cache[key]

    def _basis(self, n: int):
        if n not in self.cache:
            I = jnp.eye(n, dtype=jnp.float32)
            F = dct_1d(I, self.norm)
            B = idct_1d(I, self.norm)
            self.cache[n] = (F, B)
        return self.cache[n]

    def einsum_2d(self, x, b, d=None):
        if d is None:                       # 1-D 
            return jnp.einsum("...ij, jb -> ...ib", x, b)
        else:                               # 2-D
            return jnp.einsum("...ijkl, jb, ld -> ...ikbd", x, b, d)


    def einsum_2d_t(self, x, b, d=None):
        if d is None:
            return jnp.einsum("...ij, jb -> ...ib", x, b)
        else:
            return jnp.einsum("...ijkl, kb, ld -> ...ibjd", x, b, d)

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        # print('x.shape>>>>>>>>>>:', x.shape)
        if len(x.shape) == 1:  # 1D
            n1 = smaller_split(x.shape[0], self.chunk)
            F, _ = self._basis(n1)
            x = rearrange(x, "(x w) -> x w", w=n1)
            x = self.einsum_2d(x, F)
        elif len(x.shape) == 2:  # 2D
            m, n = x.shape
            n1 = smaller_split(m, self.chunk)
            n2 = smaller_split(n, self.chunk)
            F1, _ = self._basis(n1)
            F2, _ = self._basis(n2)
            # print('F1:', F1, 'F2:', F2)
            x = rearrange(x, "(y h) (x w) -> y h x w", h=n1, w=n2)
            x = self.einsum_2d(x, F1, F2)
        else: # 0D
            pass

        return x

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        if len(x.shape) == 2:  # 1D
            n1 = x.shape[-1]
            _, B1 = self._basis(n1)
            x = self.einsum_2d_t(x, B1)
            x = rearrange(x, "x w -> (x w)")
        elif len(x.shape) > 2:  # 2D
            n1, n2 = x.shape[2:]
            _, B1 = self._basis(n1)
            _, B2 = self._basis(n2)
            # print('B1:', B1, 'B2:', B2)
            x = self.einsum_2d_t(x, B1, B2)
            x = rearrange(x, "y h x w -> (y h) (x w)")
        else: # 0D
            pass
        
        return x

def demo_compression(params: Any, k_percent: float = 0.5, chunk_percent: float = 0.8):
    def _demo(x):
        if x.shape == ():
            return x
        # print('x.shape:', x.shape)
        chunk = max(1, int(min(x.shape) * chunk_percent))
        tf = TransformDCT(chunk=chunk)
        enc_x = tf.encode(x)
        enc_shape = enc_x.shape
        # print('enc_x.shape:', enc_x.shape, 'enc_x:', enc_x)   
        if len(enc_shape) > 2: # 2D
            s1, s3 = enc_x.shape[1], enc_x.shape[3]
            enc_x = rearrange(enc_x, "y x h w -> (y x) (h w)")
        
        k = max(1, int(k_percent * enc_x.shape[-1]))
        def _topk_array(enc_x_row):
            x_abs = jnp.abs(enc_x_row)
            threshold = jnp.sort(x_abs)[-k]
            mask = x_abs >= threshold
            return enc_x_row * mask
        comp_enc_x = jax.vmap(_topk_array)(enc_x)

        if len(enc_shape) > 2: # 2D
            comp_enc_x = rearrange(comp_enc_x, "(y x) (h w) -> y x h w", x=s1, w=s3)
        
        # print('comp_enc_x.shape:', comp_enc_x.shape, comp_enc_x)
        dec_x = tf.decode(comp_enc_x)
        return dec_x
    return jax.tree_util.tree_map(_demo, params)


def topk_sparsification(params: Any, k_percent: float = 0.1) -> Any:
    """Apply top-k sparsification to a pytree of parameters.
    
    Args:
        params: A pytree of parameters.
        k_percent: The percentage of top values to keep (between 0 and 1).
    
    Returns:
        A pytree with the same structure as params, but with only the top k% values kept.
    """
    def _topk_array(x):
        # Flatten the array
        x_flat = x.reshape(-1)
        # Calculate the number of elements to keep
        k = max(1, int(k_percent * x_flat.size))
        # Get the threshold value for top-k
        threshold = jnp.sort(jnp.abs(x_flat))[-k]
        # Create a mask for values above the threshold
        mask = jnp.abs(x) >= threshold
        # Apply the mask
        return x * mask
    
    return jax.tree_util.tree_map(_topk_array, params)

def quantization(params: Any, bits: int = 8, scale_per_tensor: bool = True) -> Any:
    """Quantize a pytree of parameters to a lower bit representation.
    
    Args:
        params: A pytree of parameters.
        bits: Number of bits to quantize to (1-32).
        scale_per_tensor: Whether to use a single scale for the entire tensor (True)
                          or per-channel scales (False).
    
    Returns:
        A pytree with the same structure as params, but with quantized values.
    """
    def _quantize_array(x):
        # Determine the range of values
        if scale_per_tensor:
            x_min = jnp.min(x)
            x_max = jnp.max(x)
        else:
            # Per-channel quantization (assuming last dim is channels)
            x_min = jnp.min(x, axis=tuple(range(x.ndim - 1)), keepdims=True)
            x_max = jnp.max(x, axis=tuple(range(x.ndim - 1)), keepdims=True)
        
        # Calculate the scale and zero point
        scale = (x_max - x_min) / (2**bits - 1)
        # Avoid division by zero
        scale = jnp.where(scale == 0, 1.0, scale)
        
        # Quantize
        x_quant = jnp.round((x - x_min) / scale)
        # Clip to ensure values are within range
        x_quant = jnp.clip(x_quant, 0, 2**bits - 1)
        # Dequantize (to simulate the effect of quantization)
        x_dequant = x_quant * scale + x_min
        
        return x_dequant
    
    return jax.tree_util.tree_map(_quantize_array, params)

def random_sparsification(params: Any, keep_percent: float = 0.1, key: Optional[jax.random.PRNGKey] = None) -> Any:
    """Apply random sparsification to a pytree of parameters.
    
    Args:
        params: A pytree of parameters.
        keep_percent: The percentage of values to randomly keep (between 0 and 1).
        key: A PRNG key for random number generation. If None, a new one will be created.
    
    Returns:
        A pytree with the same structure as params, but with randomly selected values kept.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    def _random_sparsify_array(x, subkey):
        # Create a random mask
        mask = jax.random.uniform(subkey, shape=x.shape) < keep_percent
        # Apply scaling to preserve the expected sum
        scaling = 1.0 / keep_percent
        return jnp.where(mask, x * scaling, 0.0)
    
    # Create a different key for each leaf in the pytree
    keys = jax.random.split(key, jax.tree_util.tree_leaves(params).__len__())
    keys_iter = iter(keys)
    
    # Apply random sparsification to each leaf
    return jax.tree_util.tree_map(lambda x: _random_sparsify_array(x, next(keys_iter)), params)

def _single_cocktail_compression(
    params: Any, 
    topk_percent: float = 0.3, 
    quantize_bits: int = 8, 
    random_percent: float = 0.0,
    chunk_percent: float = 0.6,
    compression_order: str = "random,topk,quantize",
    key: Optional[jax.random.PRNGKey] = None
) -> Any:
    """Apply a combination of compression techniques as in CocktailSGD to a single instance.
    
    Args:
        params: A pytree of parameters.
        topk_percent: The percentage of top values to keep (between 0 and 1).
        quantize_bits: Number of bits to quantize to (1-32).
        random_percent: The percentage of values to randomly keep after topk (between 0 and 1).
        compression_order: The order in which to apply the compression techniques.
        key: A PRNG key for random number generation. If None, a new one will be created.
    
    Returns:
        A pytree with the same structure as params, but with compression applied.
    """
    # Initialize random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)
    
    compression_order = compression_order.split(",")
    
    compressed_params = params
    
    for technique in compression_order:
        if technique == "topk" and topk_percent < 1.0:
            compressed_params = topk_sparsification(compressed_params, topk_percent)
        elif technique == "quantize" and quantize_bits < 32:
            compressed_params = quantization(compressed_params, quantize_bits)
        elif technique == "random" and random_percent > 0.0:
            key, subkey = jax.random.split(key)
            compressed_params = random_sparsification(compressed_params, random_percent, subkey)
        elif technique == "demo":
            compressed_params = demo_compression(compressed_params, topk_percent, chunk_percent)
    # print('compress complete>>>>><<<<<<')
    return compressed_params

def cocktail_compression(
    params: Any, 
    topk_percent: float = 0.3, 
    quantize_bits: int = 8, 
    random_percent: float = 0.0,
    chunk_percent: float = 0.6,
    compression_order: str = "random,topk,quantize",
    key: Optional[jax.random.PRNGKey] = None
) -> Any:
    """Apply a combination of compression techniques as in CocktailSGD.
    This function is vmapped over the first dimension of params.
    
    Args:
        params: A pytree of parameters with a batch dimension.
        topk_percent: The percentage of top values to keep (between 0 and 1).
        quantize_bits: Number of bits to quantize to (1-32).
        random_percent: The percentage of values to randomly keep after topk (between 0 and 1).
        compression_order: The order in which to apply the compression techniques.
        key: A PRNG key for random number generation. If None, a new one will be created.
    
    Returns:
        A pytree with the same structure as params, but with compression applied.
    """
    # Initialize random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Create batch keys for each element in the batch
    batch_size = jax.tree_util.tree_leaves(params)[0].shape[0]
    batch_keys = jax.random.split(key, batch_size)
    
    # Create a vmapped version of the single compression function
    vmapped_compression = jax.vmap(
        _single_cocktail_compression,
        in_axes=(0, None, None, None, None, None, 0),
        out_axes=0
    )
    
    return vmapped_compression(
        params, 
        topk_percent, 
        quantize_bits, 
        random_percent, 
        chunk_percent,
        compression_order,
        batch_keys
    )

if __name__ == "__main__":
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create a sample pytree with the specified structure
    sample_params = {
        'mlp/~/linear_0': {
            'b': jax.random.normal(key, (2, 128)),
            'w': jax.random.normal(jax.random.split(key)[0], (2, 3072, 128))
        },
        'mlp/~/linear_1': {
            'b': jax.random.normal(jax.random.split(key)[1], (2, 128)),
            'w': jax.random.normal(jax.random.split(key)[0], (2, 128, 128))
        },
        'mlp/~/linear_2': {
            'b': jax.random.normal(jax.random.split(key)[1], (2, 128)),
            'w': jax.random.normal(jax.random.split(key)[0], (2, 128, 128))
        },
        'mlp/~/linear_3': {
            'b': jax.random.normal(jax.random.split(key)[1], (2, 10)),
            'w': jax.random.normal(jax.random.split(key)[0], (2, 128, 10))
        }
    }
    
    # Test different compression techniques
    print("Testing cocktail compression with different settings:")
    
    # Test 1: Top-k only
    compressed_params = cocktail_compression(
        sample_params,
        topk_percent=0.3,
        compression_order="topk",
        key=jax.random.PRNGKey(0)
    )
    
    # Test 2: Quantization only
    compressed_params = cocktail_compression(
        sample_params,
        quantize_bits=8,
        compression_order="quantize",
        key=jax.random.PRNGKey(1)
    )
    
    # Test 3: Random sparsification only
    compressed_params = cocktail_compression(
        sample_params,
        random_percent=0.2,
        compression_order="random",
        key=jax.random.PRNGKey(2)
    )
    
    # Test 4: Full cocktail (all techniques)
    compressed_params = cocktail_compression(
        sample_params,
        topk_percent=0.3,
        quantize_bits=8,
        random_percent=0.2,
        compression_order="topk,random,quantize",
        key=jax.random.PRNGKey(3)
    )

    # Test 5: DeMo compression
    compressed_params = cocktail_compression(
        sample_params,
        topk_percent=0.3,
        quantize_bits=8,
        random_percent=0.2,
        chunk_percent=0.6,
        compression_order="demo",
        key=jax.random.PRNGKey(3)
    )
    
    # Print structure of the compressed parameters
    print("\nCompressed parameters structure:")
    jax.tree_util.tree_map(lambda x: print(f"{x.shape}, {x.dtype}"), compressed_params)
    
    # Calculate compression statistics
    original_size = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(sample_params))
    print(f"\nOriginal parameters size: {original_size / 1024:.2f} KB")
    
    # Count non-zero elements in compressed params (for sparsity measurement)
    non_zeros = sum(jnp.count_nonzero(x) for x in jax.tree_util.tree_leaves(compressed_params))
    total_elements = sum(x.size for x in jax.tree_util.tree_leaves(compressed_params))
    sparsity = 1.0 - (non_zeros / total_elements)
    print(f"Sparsity achieved: {sparsity:.2%}")







if __name__ == "__main__":
    # Create sample parameters with a range of values
    sample_params = {
        'weights': jnp.array([[-1.5, -0.5, 0.0, 0.3, 1.2, 2.5]]),
        'bias': jnp.array([0.1, -0.2, 0.15])
    }

    print("\nTesting quantization with different bit widths:")
    print("Original parameters:")
    for name, param in sample_params.items():
        print(f"{name}:\n{param}")
        print(f"Value range: [{param.min():.3f}, {param.max():.3f}]")

    # Test different bit widths
    for bits in [2, 4, 8]:
        print(f"\nQuantizing to {bits} bits:")
        quantized = quantization(sample_params, bits=bits)
        
        print(f"\nQuantized parameters (should have {2**bits} unique values):")
        for name, param in quantized.items():
            unique_values = jnp.unique(param)
            print(f"\n{name}:")
            print(f"Values:\n{param}")
            print(f"Unique values ({len(unique_values)} found): {unique_values}")
            print(f"Value range: [{param.min():.3f}, {param.max():.3f}]")
            
            # Verify number of unique values matches bit width
            assert len(unique_values) <= 2**bits, \
                f"Found {len(unique_values)} unique values, expected <= {2**bits}"

    print("\nQuantization test passed successfully!")
