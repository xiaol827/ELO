
import jax
import jax.numpy as jnp
import math
import jax.lax
from functools import partial
import jax.image as jimg
import numpy as np

def random_flip(images, labels, key):
    """Randomly flips images horizontally."""
    flip_mask = jax.random.bernoulli(key, p=0.5, shape=(images.shape[0],))
    flipped_images = jnp.where(flip_mask[:, None, None, None], images[:, :, ::-1, :], images)
    return flipped_images, labels


def random_crop(images, labels, key, pad=4):
    """Randomly crops a batch of images with padding."""
    batch_size, h, w, c = images.shape
    padded = jnp.pad(images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')

    key_x, key_y = jax.random.split(key)
    crop_x = jax.random.randint(key_x, (batch_size,), 0, 2 * pad)
    crop_y = jax.random.randint(key_y, (batch_size,), 0, 2 * pad)

    def crop(img, x, y):
        return jax.lax.dynamic_slice(img, (y, x, 0), (h, w, c))

    cropped_images = jax.vmap(crop)(padded, crop_x, crop_y)
    return cropped_images, labels


def mixup(images, labels, key, alpha=0.2):
    """Applies MixUp augmentation."""
    batch_size = images.shape[0]
    lam = jax.random.beta(key, alpha, alpha, (batch_size, 1, 1, 1))
    
    indices = jax.random.permutation(key, batch_size)
    mixed_images = lam * images + (1 - lam) * images[indices]
    mixed_labels = lam[:, 0, 0, 0] * labels + (1 - lam[:, 0, 0, 0]) * labels[indices]
    
    return mixed_images, mixed_labels


def cutmix(images, labels, key, alpha=1.0):
    """Applies CutMix augmentation to a batch of images and labels."""
    batch_size, h, w, c = images.shape
    key_lam, key_x, key_y, key_perm = jax.random.split(key, 4)

    lam = jax.random.beta(key_lam, alpha, alpha, (batch_size,))  # Per-sample lambda

    cut_rat = jnp.sqrt(1.0 - lam)
    cut_w = jnp.minimum((w * cut_rat).astype(int), w)  # Ensure within bounds
    cut_h = jnp.minimum((h * cut_rat).astype(int), h)  # Ensure within bounds

    cx = jax.random.randint(key_x, (batch_size,), 0, w)
    cy = jax.random.randint(key_y, (batch_size,), 0, h)

    x1 = jnp.clip(cx - cut_w // 2, 0, w - 1)
    y1 = jnp.clip(cy - cut_h // 2, 0, h - 1)

    indices = jax.random.permutation(key_perm, batch_size)

    def apply_cutmix(img1, img2, x1, y1, cut_w, cut_h):
        """Replaces a patch in `img1` with a patch from `img2`."""
        slice_w = jnp.minimum(cut_w, w - x1)  # Ensure slice is within bounds
        slice_h = jnp.minimum(cut_h, h - y1)  # Ensure slice is within bounds

        patch = jax.lax.dynamic_slice(img2, (y1, x1, 0), (slice_h, slice_w, c))
        return jax.lax.dynamic_update_slice(img1, patch, (y1, x1, 0))

    # Apply CutMix using vmap
    mixed_images = jax.vmap(apply_cutmix)(images, images[indices], x1, y1, cut_w, cut_h)
    mixed_labels = lam[:, None] * labels + (1 - lam[:, None]) * labels[indices]

    return mixed_images, mixed_labels



def random_flip_horizontal(rng, images):
    """Randomly flip images horizontally."""
    flip_rng, rng = jax.random.split(rng)
    mask = jax.random.bernoulli(flip_rng, p=0.5, shape=(images.shape[0], 1, 1, 1))
    return jnp.where(mask, jnp.flip(images, axis=2), images), rng

def random_translate(rng, images, max_shift=4):
    """Randomly translate images using JAX's lax.dynamic_slice."""
    translate_rng, rng = jax.random.split(rng)
    batch_size, height, width, channels = images.shape
    
    # Sample dx and dy shifts
    dx = jax.random.randint(translate_rng, (batch_size,), -max_shift, max_shift + 1)
    dy = jax.random.randint(translate_rng, (batch_size,), -max_shift, max_shift + 1)

    # Pad the image to allow shifts
    pad_size = max_shift
    padded_images = jnp.pad(images, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

    def translate(img, dx, dy):
        """Apply translation using JAX's dynamic slicing."""
        return jax.lax.dynamic_slice(
            img, (dx + pad_size, dy + pad_size, 0), (height, width, channels)
        )

    # Vectorized translation using vmap
    translated_images = jax.vmap(translate)(padded_images, dx, dy)
    
    return translated_images, rng

def random_brightness(rng, images, max_delta=0.2):
    """Randomly adjust brightness."""
    brightness_rng, rng = jax.random.split(rng)
    delta = jax.random.uniform(brightness_rng, (images.shape[0], 1, 1, 1), minval=-max_delta, maxval=max_delta)
    return jnp.clip(images + delta, 0.0, 1.0), rng

def random_contrast(rng, images, lower=0.8, upper=1.2):
    """Randomly adjust contrast."""
    contrast_rng, rng = jax.random.split(rng)
    factors = jax.random.uniform(contrast_rng, (images.shape[0], 1, 1, 1), minval=lower, maxval=upper)
    mean = jnp.mean(images, axis=(1, 2), keepdims=True)
    return jnp.clip((images - mean) * factors + mean, 0.0, 1.0), rng


def random_cutout(rng, images, size=8):
    """Apply cutout by setting a random square region to zero using JAX's lax.dynamic_update_slice."""
    cutout_rng, rng = jax.random.split(rng)
    batch_size, height, width, channels = images.shape

    # Sample random center points for the cutout region
    cx = jax.random.randint(cutout_rng, (batch_size,), 0, height)
    cy = jax.random.randint(cutout_rng, (batch_size,), 0, width)

    # Cutout mask of the same shape as the image
    def apply_cutout(img, cx, cy):
        """Apply cutout by setting a square region to zero."""
        mask = jnp.ones_like(img)

        # Ensure valid slice locations using clipping
        x1 = jnp.clip(cx - size // 2, 0, height - size)
        y1 = jnp.clip(cy - size // 2, 0, width - size)

        cutout_patch = jnp.zeros((size, size, channels), dtype=img.dtype)
        mask =jax.lax.dynamic_update_slice(mask, cutout_patch, (x1, y1, 0))

        return img * mask

    # Apply cutout across the batch using vmap
    images = jax.vmap(apply_cutout)(images, cx, cy)
    
    return images, rng

def random_resize_crop(images, labels, key, scale=(0.7, 1.0)):

    B, H, W, C = images.shape
    key_s, key_y, key_x = jax.random.split(key, 3)

    s = jax.random.uniform(key_s, (B,), minval=scale[0], maxval=scale[1])
    crop_h = (s * H).astype(jnp.float32)  # 用于计算步长/网格
    crop_w = (s * W).astype(jnp.float32)

    max_y = jnp.maximum(H - jnp.floor(crop_h).astype(jnp.int32), 1)
    max_x = jnp.maximum(W - jnp.floor(crop_w).astype(jnp.int32), 1)
    y0 = jax.random.randint(key_y, (B,), 0, max_y).astype(jnp.float32)
    x0 = jax.random.randint(key_x, (B,), 0, max_x).astype(jnp.float32)

    tgt_y = jnp.arange(H, dtype=jnp.float32) + 0.5  # [0.5 .. H-0.5]
    tgt_x = jnp.arange(W, dtype=jnp.float32) + 0.5
    step_y = crop_h / H  # [B]
    step_x = crop_w / W  # [B]

    ys = y0[:, None, None] + (tgt_y[None, :, None]) * step_y[:, None, None] - 0.5
    xs = x0[:, None, None] + (tgt_x[None, None, :]) * step_x[:, None, None] - 0.5

    y0f = jnp.floor(ys)
    x0f = jnp.floor(xs)
    y1f = y0f + 1.0
    x1f = x0f + 1.0

    y0i = jnp.clip(y0f, 0.0, H - 1.0).astype(jnp.int32)
    x0i = jnp.clip(x0f, 0.0, W - 1.0).astype(jnp.int32)
    y1i = jnp.clip(y1f, 0.0, H - 1.0).astype(jnp.int32)
    x1i = jnp.clip(x1f, 0.0, W - 1.0).astype(jnp.int32)

    wy = ys - y0f  # [B,H,W]
    wx = xs - x0f
    wy0 = 1.0 - wy
    wx0 = 1.0 - wx

    def gather(imgs, yi, xi):
        # imgs: [B,H,W,C], yi/xi: [B,H,W]
        b_idx = jnp.arange(B, dtype=jnp.int32)[:, None, None]
        return imgs[b_idx, yi, xi, :]  # [B,H,W,C]

    I00 = gather(images, y0i, x0i)
    I01 = gather(images, y0i, x1i)
    I10 = gather(images, y1i, x0i)
    I11 = gather(images, y1i, x1i)

    wy0 = wy0[..., None]
    wy1 = wy[..., None]
    wx0 = wx0[..., None]
    wx1 = wx[..., None]

    top    = I00 * wy0 * wx0 + I01 * wy0 * wx1
    bottom = I10 * wy1 * wx0 + I11 * wy1 * wx1
    out = top + bottom  # [B,H,W,C]

    return out, labels

def autoaugment_cifar10(images, labels, rng):
    """
    Applies a set of JAX-native augmentations to a batch of CIFAR-10 images.
    
    Args:
        rng: JAX PRNG key.
        images: jnp.ndarray of shape (batch_size, height, width, channels).
        labels: jnp.ndarray of shape (batch_size,).
    
    Returns:
        Augmented images and unchanged labels.
    """
    images, rng = random_flip_horizontal(rng, images)
    images, rng = random_translate(rng, images, max_shift=4)
    images, rng = random_brightness(rng, images, max_delta=0.2)
    images, rng = random_contrast(rng, images, lower=0.8, upper=1.2)
    images, rng = random_cutout(rng, images, size=8)
    return images, labels

def aug_batch(images, labels, key, prob=0.5):
    key1, key2, key3, key4, key_crop, key_trans = jax.random.split(key, 6)
    mask2 = jax.random.bernoulli(key2, p=prob, shape=(images.shape[0], 1, 1, 1))
    mask3 = jax.random.bernoulli(key3, p=prob, shape=(images.shape[0], 1, 1, 1))

    images, labels = random_flip(images, labels, key1)
    images_aug, labels = random_crop(images, labels, key_crop)
    images = jnp.where(mask2, images_aug, images)

    images_aug, _ = random_translate(key_trans, images, max_shift=4)
    images = jnp.where(mask3, images_aug, images)
    # # images_aug, labels_aug = mixup(images, labels, key_mixup)
    # # images = jnp.where(mask4, images_aug, images)
    # # labels = jnp.where(mask4, labels_aug, labels)
    return images, labels
    # images, key = random_flip_horizontal(key, images)
    # images, key = random_translate(key, images, max_shift=4)
    # images, key = random_brightness(key, images, max_delta=0.2)
    # images, key = random_cutout(key, images, size=4)
    # return images, labels

def aug_transform(images, labels, augmentations, key, prob=0.5):
    *prefix, N, H, W, C = images.shape
    P = math.prod(prefix)

    images_flat = images.reshape((P, N, H, W, C))
    labels_flat = labels.reshape((P, N))

    keys = jax.random.split(key, P)

    images_aug_flat, labels_aug_flat = jax.vmap(
        aug_batch, in_axes=(0,0,0,None), out_axes=(0,0)
    )(images_flat, labels_flat, keys, prob)

    images_out = images_aug_flat.reshape((*prefix, N, H, W, C))
    labels_out = labels_aug_flat.reshape((*prefix, N))
    return images_out, labels_out
    

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

def _maybe_apply(images, labels, key, prob, aug_fn):
    """Apply aug_fn to (images, labels) with prob per-example."""
    mask = jax.random.bernoulli(key, p=prob, shape=(images.shape[0], 1, 1, 1))
    images_aug, labels_aug = aug_fn(images, labels, key)
    images = jnp.where(mask, images_aug, images)
    return images, labels_aug


def _aug_flip(images, labels, key):
    return random_flip(images, labels, key)


def _aug_crop(images, labels, key):
    return random_crop(images, labels, key)


def _aug_translate(images, labels, key, max_shift=4):
    images_aug, _ = random_translate(key, images, max_shift=max_shift)
    return images_aug, labels


def _aug_prob_batch(images, labels, key, augmentations, prob=0.5):
    # split a key per augmentation step (order matters)
    keys = jax.random.split(key, len(augmentations) + 1)[1:]

    for k, aug in zip(keys, augmentations):
        if aug == "flip":
            images, labels = _maybe_apply(images, labels, k, prob, _aug_flip)
        elif aug == "crop":
            images, labels = _maybe_apply(images, labels, k, prob, _aug_crop)
        elif aug == "translate":
            images, labels = _maybe_apply(images, labels, k, prob, _aug_translate)

    return images, labels


def aug_prob(images, labels, augmentations, key, prob=0.5):
    *prefix, N, H, W, C = images.shape
    P = math.prod(prefix)

    images_flat = images.reshape((P, N, H, W, C))
    labels_flat = labels.reshape((P, N))
    keys = jax.random.split(key, P)

    images_aug_flat, labels_aug_flat = jax.vmap(
        _aug_prob_batch, in_axes=(0, 0, 0, None, None), out_axes=(0, 0)
    )(images_flat, labels_flat, keys, augmentations, prob)

    images_out = images_aug_flat.reshape((*prefix, N, H, W, C))
    labels_out = labels_aug_flat.reshape((*prefix, N))
    return images_out, labels_out