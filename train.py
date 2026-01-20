# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import os
import click
import re
import json
import tempfile

# CUDA 환경 설정 (매번 자동 적용)
from setup_env import setup_cuda_environment
setup_cuda_environment(verbose=False)  # train.py는 출력이 많으므로 verbose=False

import torch
import dnnlib

from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def macs_test(channel_dict):
    """
    채널 딕셔너리를 기반으로 MACs(Multiply-Accumulate Operations)를 계산하는 테스트 함수로 보입니다.
    """
    keys = [4*(2**i) for i in range(7)]
    #channel_dict_base = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64}
    channel_dict_base = {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128}
    s_list = []
    s_list_ = []
    for i,k in enumerate(keys):
        chn = channel_dict[k]
        chn_ = channel_dict_base[k]
        if i>0:
            s_list.append((chn*chn+chn*chn)*k*k)
            s_list_.append((chn_*chn_+chn_*chn_)*k*k)
        else:
            chn_previous = channel_dict[keys[i-1]] # if i>==1
            chn_previous_ = channel_dict_base[keys[i-1]] # if i>==1
            s_list.append((chn_previous*chn+chn*chn)*k*k)
            s_list_.append((chn_previous_*chn_+chn_*chn_)*k*k)
    radio_all = sum(s_list)*1.0/sum(s_list_)
    radio_each = [e*1.0/sum(s_list) for e in s_list]
    return radio_all, radio_each

def setup_training_loop_kwargs(
    # 일반 옵션 (설명에 포함되지 않음).
    gpus       = None, # GPU 개수: <int>, 기본값 = 1 gpu
    snap       = None, # 스냅샷 간격: <int>, 기본값 = 50 ticks
    metrics    = None, # 메트릭 이름 목록: [], ['fid50k_full'] (기본값), ...
    seed       = None, # 랜덤 시드: <int>, 기본값 = 0

    # 데이터셋.
    data       = None, # 학습 데이터셋 (필수): <경로>
    cond       = None, # 데이터셋 레이블에 기반한 조건부 모델 학습: <bool>, 기본값 = False
    subset     = None, # N개의 이미지만 사용하여 학습: <int>, 기본값 = all
    mirror     = None, # x-flips로 데이터셋 증강: <bool>, 기본값 = False

    # 기본 설정.
    cfg        = None, # 기본 설정: 'auto' (기본값), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    gamma      = None, # R1 gamma 오버라이드: <float>
    kimg       = None, # 학습 기간 오버라이드: <int>
    batch      = None, # 배치 사이즈 오버라이드: <int>

    # 판별자 증강 (Discriminator augmentation).
    aug        = None, # 증강 모드: 'ada' (기본값), 'noaug', 'fixed'
    p          = None, # 'fixed' 모드일 때 p 지정 (필수): <float>
    target     = None, # 'ada' 모드일 때 ADA target 오버라이드: <float>, 기본값 = aug에 따름
    augpipe    = None, # 증강 파이프라인: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (기본값), ..., 'bgcfnc'

    # 전이 학습 (Transfer learning).
    resume     = None, # 이전 네트워크 로드: 'noresume' (기본값), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed    = None, # Freeze-D: <int>, 기본값 = 0 판별자 레이어

    # 성능 옵션 (설명에 포함되지 않음).
    fp32       = None, # 혼합 정밀도(mixed-precision) 학습 비활성화: <bool>, 기본값 = False
    nhwc       = None, # FP16과 함께 NHWC 메모리 포맷 사용: <bool>, 기본값 = False
    allow_tf32 = None, # PyTorch가 matmul과 convolutions에 TF32를 사용하도록 허용: <bool>, 기본값 = False
    nobench    = None, # cuDNN 벤치마킹 비활성화: <bool>, 기본값 = False
    workers    = None, # DataLoader 워커 수 오버라이드: <int>, 기본값 = 3
):
    """
    입력된 인자들을 바탕으로 학습 루프에 필요한 설정을 구성하는 함수입니다.
    """
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # 일반 옵션: gpus, snap, metrics, seed
    # ------------------------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        # snap = 100#100
        # snap = 10#100
        snap = 100
        # snap = 500
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # 데이터셋: data, cond, subset, mirror
    # -----------------------------------

    assert data is not None
    assert isinstance(data, str)
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # training.dataset.Dataset의 서브클래스
        args.training_set_kwargs.resolution = training_set.resolution # 해상도 명시
        args.training_set_kwargs.use_labels = training_set.has_labels # 레이블 사용 여부 명시
        args.training_set_kwargs.max_size = len(training_set) # 데이터셋 크기 명시
        desc = training_set.name
        del training_set # 메모리 절약
    except IOError as err:
        raise UserError(f'--data: {err}')

    if cond is None:
        cond = False
    assert isinstance(cond, bool)
    if cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True

    # ------------------------------------
    # 기본 설정: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # 해상도와 GPU 수에 따라 동적으로 채워짐.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # 기존 StyleGAN2와 달리 혼합 정밀도 사용.
        #'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        #'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=16, mbstd=2,  fmaps=1, lrate=0.002, gamma=10,    ema=10,  ramp=None, map=8),
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1, lrate=0.002, gamma=10,    ema=10,  ramp=None, map=8),
        'image128':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        # 'image128':  dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1, lrate=0.002, gamma=10,    ema=10,  ramp=None, map=8),
        # 'image128':  dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1, lrate=0.002, gamma=10,    ema=10,  ramp=None, map=8),
        'paper256_only_gpu':  dict(ref_gpus=2,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper256_1':  dict(ref_gpus=1,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper256_2':  dict(ref_gpus=2,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        # org: 'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        # 'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        # 'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        # 'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=16, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # GPU 메모리 소비를 억제.
        spec.mbstd = min(spec.mb // gpus, 4) # mbstd 그룹 크기가 고정되면 다른 하이퍼파라미터들이 더 예측 가능하게 동작함.
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # 휴리스틱 공식.
        spec.ema = spec.mb * 10 / 32


    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512

    args.task_name = 'StyleLight-training'
    args.G_kwargs.synthesis_kwargs.channels_dict = {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128, 512:64}#, 256: 32, 512: 32}#, 256: 48, 512: 48}   ###
    args.D_kwargs.channels_dict = {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128, 512: 64}#, 256: 64, 512: 64}#, 256: 64, 512: 64} 


    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # 혼합 정밀도(mixed-precision) 학습 활성화
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # float16 오버플로우를 방지하기 위해 활성화를 클램프
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp
    print('batch size:',args.batch_size)
    print('batch gpu:',args.batch_gpu)
    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0 # path length regularization 비활성화
        args.loss_kwargs.style_mixing_prob = 0 # style mixing 비활성화
        args.D_kwargs.architecture = 'orig' # residual skip connections 비활성화

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # ---------------------------------------------------
    # 판별자 증강: aug, p, target, augpipe
    # ---------------------------------------------------

    if aug is None:
        aug = 'ada'
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.6

    elif aug == 'noaug':
        pass

    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')
    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        assert isinstance(target, float)
        if aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
        'cyclic': dict(cyclic=1),
    }

    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])

    # ----------------------------------
    # 전이 학습: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    elif resume in resume_specs:
        desc += f'-resume{resume}'
        args.resume_pkl = resume_specs[resume] # 미리 정의된 URL
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # 사용자 정의 경로 또는 URL

    if resume != 'noresume':
        args.ada_kimg = 100 # 시작할 때 ADA가 더 빠르게 반응하도록 함
        args.ema_rampup = None # EMA 램프업 비활성화

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # 성능 옵션: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    # -------------------------------------------------
    # HDR 물리 모드: Softplus 활성화 및 Full FP32 강제
    # -------------------------------------------------

    # hdr_mode가 전역으로 정의되어 있다면 사용, 아니면 False
    hdr_mode = getattr(args, 'hdr_mode', False) if hasattr(args, 'hdr_mode') else False

    if hdr_mode:
        # HDR 모드: FP32 강제화
        args.G_kwargs.synthesis_kwargs.num_fp16_res = 0
        args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = None
        args.D_kwargs.conv_clamp = None
        # HDR 활성화 설정 (Softplus)
        args.G_kwargs.synthesis_kwargs.hdr_mode = True
        args.G_kwargs.synthesis_kwargs.hdr_activation = 'softplus'
        desc += '-hdr_physical'

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    """
    각 GPU 프로세스에서 실행되는 함수입니다. 분산 학습을 초기화하고 training_loop를 실행합니다.
    """
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # torch.distributed 초기화.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # torch_utils 초기화.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # 학습 루프 실행.
    training_loop.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# 일반 옵션.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
@click.option('--metrics', help='Comma-separated list or "none" [default: fid50k_full]', type=CommaSeparatedList())
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

# 데이터셋.
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
@click.option('--cond', help='Train conditional model based on dataset labels [default: false]', type=bool, metavar='BOOL')
@click.option('--subset', help='Train with only N images [default: all]', type=int, metavar='INT')
@click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')

# 기본 설정.
@click.option('--cfg', help='Base config [default: auto]', type=click.Choice(['auto', 'image128', 'stylegan2', 'paper256', 'paper256_only_gpu','paper256_1','paper256_2','paper512', 'paper1024', 'cifar']))
@click.option('--gamma', help='Override R1 gamma', type=float)
@click.option('--kimg', help='Override training duration', type=int, metavar='INT')
@click.option('--batch', help='Override batch size', type=int, metavar='INT')

# 판별자 증강.
@click.option('--aug', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--p', help='Augmentation probability for --aug=fixed', type=float)
@click.option('--target', help='ADA target value for --aug=ada', type=float)
@click.option('--augpipe', help='Augmentation pipeline [default: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc', 'cyclic']))

# 전이 학습.
@click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')
@click.option('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')

# 성능 옵션.
@click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

def main(ctx, outdir, dry_run,  **config_kwargs):
    """
    메인 진입점 함수입니다. CLI 인자를 파싱하고 학습 프로세스를 시작합니다.

    Train a GAN using the techniques described in the paper
    "Training Generative Adversarial Networks with Limited Data".

    Examples:

    \b
    # Train with custom dataset using 1 GPU.
    python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1

    \b
    # Train class-conditional CIFAR-10 using 2 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/cifar10.zip \\
        --gpus=2 --cfg=cifar --cond=1

    \b
    # Transfer learn MetFaces from FFHQ using 4 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/metfaces.zip \\
        --gpus=4 --cfg=paper1024 --mirror=1 --resume=ffhq1024 --snap=10

    \b
    # Reproduce original StyleGAN2 config F.
    python train.py --outdir=~/training-runs --data=~/datasets/ffhq.zip \\
        --gpus=8 --cfg=stylegan2 --mirror=1 --aug=noaug

    \b
    Base configs (--cfg):
      auto       Automatically select reasonable defaults based on resolution
                 and GPU count. Good starting point for new datasets.
      stylegan2  Reproduce results for StyleGAN2 config F at 1024x1024.
      paper256   Reproduce results for FFHQ and LSUN Cat at 256x256.
      paper512   Reproduce results for BreCaHAD and AFHQ at 512x512.
      paper1024  Reproduce results for MetFaces at 1024x1024.
      cifar      Reproduce results for CIFAR-10 at 32x32.

    \b
    Transfer learning source networks (--resume):
      ffhq256        FFHQ trained at 256x256 resolution.
      ffhq512        FFHQ trained at 512x512 resolution.
      ffhq1024       FFHQ trained at 1024x1024 resolution.
      celebahq256    CelebA-HQ trained at 256x256 resolution.
      lsundog256     LSUN Dog trained at 256x256 resolution.
      <PATH or URL>  Custom network pickle.
    """
    dnnlib.util.Logger(should_flush=True)
    # 학습 옵션 설정.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # 출력 디렉토리 선택.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # 옵션 출력.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of images:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
    print()

    # Dry run? (시뮬레이션)
    if dry_run:
        print('Dry run; exiting.')
        return

    # 출력 디렉토리 생성.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # 프로세스 시작.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
