# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""Pre-training ViT on ILSVRC-2012 with Celo2LOpt learned optimizer."""

import big_vision.configs.common as bvcc
from big_vision.configs.common_fewshot import get_fewshot_lsr
import os
import ml_collections as mlc

MIXUP_DEF = {
    'none': dict(p=0.0, fold_in=None),
    'light1': dict(p=0.0, fold_in=None),
    'light2': dict(p=0.2, fold_in=None),
    'medium1': dict(p=0.2, fold_in=None),
    'medium2': dict(p=0.5, fold_in=None),
    'strong1': dict(p=0.5, fold_in=None),
    'strong2': dict(p=0.8, fold_in=None),
    'strong3': dict(p=0.5, fold_in=None),
    'strong4': dict(p=0.5, fold_in=None),
    'strong5': dict(p=0.5, fold_in=None),
    'strong6': dict(p=0.5, fold_in=None),
    'strong7': dict(p=0.5, fold_in=None),
    'strong8': dict(p=0.6, fold_in=None),
}

RANDAUG_DEF = {
    'none': '',
    'light1': 'randaug(2,0)',  # Actually not nothing!
    'light2': 'randaug(2,10)',
    'medium1': 'randaug(2,15)',
    'medium2': 'randaug(2,15)',
    'strong1': 'randaug(2,20)',
    'strong2': 'randaug(2,20)',
    'strong3': 'randaug(2,25)',
    'strong4': 'randaug(2,30)',
    'strong5': 'randaug(4,15)',
    'strong6': 'randaug(4,20)',
    'strong7': 'randaug(4,10)',
    'strong8': 'randaug(3,10)',
}


def get_config(arg=None):
  """Config for training."""
  arg = bvcc.parse_arg(arg, variant='B/16', runlocal=False, aug='')
  config = mlc.ConfigDict()

  config.lo_name = 'Celo2LOpt'
  config.selected_checkpoint = ''
  config.wandb_checkpoint_id = 'FILL_CELO2_WANDB_ID'
  config.lo_kwargs = dict(
      checkpoint_path="FILL_IN_LOCAL_CHECKPOINT_PATH",
      init_lr=0.0,
      peak_lr=1e-3,
      warmup_steps=0,
      warmup_fraction=0.05,
      end_lr=1e-5,
      weight_decay=0.0,
      adam_lr_mult=1.0,
      adam_beta1=0.9,
      adam_beta2=0.95,
      adam_weight_decay=None,
      use_adamw_for_1d=True,
      orthogonalize=True,
      clip_grad=False,
      clip_norm=1.0,
  )

  config.wandb = dict(
      enabled=True,
      project='xiao_big_vision',
      entity='eb-lab',
      name='celo2_vitb16_lr1e-3',
      tags=[],
      notes='',
      group='celo2',
      resume_wandb=True
  )

  # Auto-derive workdir from $SCRATCH and config.wandb.name (overrides via --config.wandb.name flow through).
  config.workdir = os.environ.get('SCRATCH', '') + '/bv_output/' + config.wandb.get_ref('name')

  config.seed = 0
  # config.total_epochs = 300
  config.total_steps = 0  # placeholder, override via --config.total_steps
  config.num_classes = 1000
  config.loss = 'sigmoid_xent'
  config.init_head_bias = -6.9

  aug_setting = arg.aug or 'none'

  config.input = dict()
  config.input.data = dict(
      name='imagenet2012',
      split='train[:99%]',
  )
  config.input.batch_size = 4096
  config.input.cache_raw = True  # ~140GB RAM, skips disk I/O after epoch 1
  config.input.shuffle_buffer_size = 250_000

  pp_common = (
      '|value_range(-1, 1)'
      '|onehot(1000, key="{lbl}", key_result="labels")'
      '|keep("image", "labels")'
  )
  config.input.pp = (
      'decode_jpeg_and_inception_crop(224)|flip_lr|' +
      RANDAUG_DEF[aug_setting] +
      pp_common.format(lbl='label')
  )
  pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common

  config.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'archive.randaug']

  config.input.prefetch = 8
  config.prefetch_to_device = 4

  config.log_training_steps = 100
  config.ckpt_steps = 1000

  config.model_name = 'vit'
  config.model = dict(
      variant=arg.variant,
      rep_size=True,
      pool_type='tok',
      dtype_mm='bfloat16',
  )

  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16')

  config.lr = 0.001
  config.wd = 0.0001
  config.schedule = dict(warmup_steps=5000, decay_type='cosine', min_lr_factor=0.0)

  # Alias config.lr / .wd / .schedule.min_lr_factor onto lo_kwargs (uniform CLI naming).
  config.lo_kwargs.peak_lr = config.get_ref('lr')
  config.lo_kwargs.end_lr = config.get_ref('lr') * config.schedule.get_ref('min_lr_factor')
  config.lo_kwargs.weight_decay = config.get_ref('wd')

  config.mixup = MIXUP_DEF[aug_setting]

  def get_eval(split, dataset='imagenet2012'):
    return dict(
        type='classification',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval.format(lbl='label'),
        loss_name=config.loss,
        log_steps=1000,
        cache='final_data' if arg.runlocal else 'none',
    )
  config.evals = {}
  config.evals.train = get_eval('train[:2%]')
  config.evals.minival = get_eval('train[99%:]')
  config.evals.val = get_eval('validation')
  config.evals.v2 = get_eval('test', dataset='imagenet_v2')
  config.evals.real = get_eval('validation', dataset='imagenet2012_real')
  config.evals.real.pp_fn = pp_eval.format(lbl='real_label')

  # config.fewshot = get_fewshot_lsr(runlocal=arg.runlocal)
  # config.fewshot.log_steps = 10_000

  if arg.runlocal:
    config.input.shuffle_buffer_size = 10
    config.input.batch_size = 8
    config.input.cache_raw = False
    config.evals.train.data.split = 'train[:16]'
    config.evals.minival.data.split = 'train[:16]'
    config.evals.val.data.split = 'validation[:16]'
    config.evals.v2.data.split = 'test[:16]'
    config.evals.real.data.split = 'validation[:16]'

  return config
