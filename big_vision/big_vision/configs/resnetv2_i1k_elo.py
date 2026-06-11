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
r"""Pre-training BiT on ILSVRC-2012 as in https://arxiv.org/abs/1912.11370

Run training of a BiT-ResNet-50x1 variant, which takes ~32min on v3-128:

big_vision.train \
    --config big_vision/configs/bit_i1k.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
    --config.model.depth 50 --config.model.width 1
"""

# from big_vision.configs.common_fewshot import get_fewshot_lsr
import big_vision.configs.common as bvcc
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
  """Config for training on ImageNet-1k."""
  arg = bvcc.parse_arg(arg, runlocal=False, aug='')
  config = mlc.ConfigDict()
  aug_setting = arg.aug or 'none'

  config.lo_name = 'ELO_AdafacMLPLOpt'
  config.selected_checkpoint = ''
  config.wandb_checkpoint_id = 'hirox827-soon/meta-training-comparison/2kh3rb3k'
  config.lo_kwargs = dict(
      exp_mult=0.001,
      step_mult=0.001,
      init_lr=0.0,
      warmup_fraction=0.05,
      expert_lr_max=0.01,
      expert_lr_min=1e-4,
      expert_lr_decay_steps=10000,
      expert_weight_decay=0.0,
      weight_decay=0.0,
      hidden_size=32,
      hidden_layers=2,
      initial_momentum_decays=(0.9, 0.99, 0.999),
      initial_rms_decays=(0.95,),
      initial_adafactor_decays=(0.9, 0.99, 0.999),
      concat_weights=True,
      make_separate_weights=False,
      split_weights=False,
      meta_train=False,
      clip_grad=False,
      clip_norm=1.0,
      use_lo_cosine_scheduler=False,
      step_mult_min=1e-4,
      expert_optim="adamw",
      )

  config.wandb = dict(
      enabled=True,
      project='xiao_big_vision',
      entity='eb-lab',  # Set to your wandb username/team, or None for default
      name='elo_v2resnet50_lr1e-3',  # Run name, auto-generated if None
      tags=[],  # List of tags for the run
      notes='',  # Notes for the run
      group='elo',  # Group name for organizing runs
      resume_wandb=True
  )

  # Auto-derive workdir from $SCRATCH and config.wandb.name (overrides via --config.wandb.name flow through).
  config.workdir = os.environ.get('SCRATCH', '') + '/bv_output/' + config.wandb.get_ref('name')

  config.seed = 0
  # config.total_epochs = 90
  config.total_steps = 0  # placeholder, override via --config.total_steps
  config.num_classes = 1000
  config.loss = 'softmax_xent'

  config.input = dict()
  config.input.data = dict(
      name='imagenet2012',
      split='train[:99%]',
  )
  config.input.batch_size = 4096
  config.input.cache_raw = True  # Needs up to 120GB of RAM!
  config.input.shuffle_buffer_size = 250_000  # Per host.

  pp_common = '|onehot(1000, key="{lbl}", key_result="labels")'
  pp_common += '|value_range(-1, 1)|keep("image", "labels")'
  config.input.pp = (
      'decode_jpeg_and_inception_crop(224)|flip_lr|' +
      RANDAUG_DEF[aug_setting] +
      pp_common.format(lbl='label')
  )
  pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common

  config.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'archive.randaug']

  config.log_training_steps = 100
  config.ckpt_steps = 1000

  # Model section
  config.model_name = 'bit_paper'
  config.model = dict(
      depth=50,  # You can also pass e.g. [3, 5, 10, 2]
      width=1.0,
  )

  # Optimizer section
  config.optax_name = 'big_vision.momentum_hp'
  config.grad_clip_norm = 1.0

  config.lr = 0.03
  config.wd = 3e-5
  config.schedule = dict(decay_type='cosine', warmup_steps=5000, min_lr_factor=0.0)

  # Alias config.lr / .wd / .schedule.min_lr_factor onto lo_kwargs (uniform CLI naming).
  config.lo_kwargs.step_mult = config.get_ref('lr')
  config.lo_kwargs.step_mult_min = config.get_ref('lr') * config.schedule.get_ref('min_lr_factor')
  config.lo_kwargs.weight_decay = config.get_ref('wd')

  config.mixup = MIXUP_DEF[aug_setting]

  # Eval section
  def get_eval(split, dataset='imagenet2012'):
    return dict(
        type='classification',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval.format(lbl='label'),
        loss_name=config.loss,
        log_steps=1000,  # Very fast O(seconds) so it's fine to run it often.
        cache='final_data',
    )
  config.evals = {}
  config.evals.train = get_eval('train[:2%]')
  config.evals.minival = get_eval('train[99%:]')
  config.evals.val = get_eval('validation')
  config.evals.v2 = get_eval('test', dataset='imagenet_v2')
  config.evals.real = get_eval('validation', dataset='imagenet2012_real')
  config.evals.real.pp_fn = pp_eval.format(lbl='real_label')

  # config.evals.fewshot = get_fewshot_lsr(runlocal=runlocal)
  # config.evals.fewshot.log_steps = 1000

  if arg.runlocal:
    config.input.batch_size = 32
    config.input.cache_raw = False
    config.input.shuffle_buffer_size = 100

    local_eval = config.evals.val
    config.evals = {'val': local_eval}
    config.evals.val.cache = 'none'

  return config