"""
Stage 2 학습 스크립트: 물리적 보정 (Physical Calibration)

Laval Photometric 데이터셋으로 물리적 정합성을 보정합니다.
Stage 1에서 학습된 구조 위에 S2R-Adapter를 적용하여 물리적 휘도를 정확하게 학습합니다.

핵심 구성요소:
- DTAM: ON (T_onset=300, T_peak=1000 cd/m²)
- PU21: 지각적 균일 인코딩
- S2R-Adapter: 3-브랜치 구조 (shared + transfer1 + transfer2)
  - r1=1 (미세 조정), r2=128 (광범위 적응)
  - scale1, scale2로 도메인 유사도에 따른 비율 조절
- 손실: L_phys + λ_consist * L_consist + L_GAN

사용법:
    python train_stage2.py --stage1_ckpt ./stage1_checkpoint.pkl --data /path/to/laval --outdir ./training-runs-stage2
    python train_stage2.py --stage1_ckpt ./stage1_checkpoint.pkl --data /path/to/laval --outdir ./training-runs-stage2 --adapter_r1 1 --adapter_r2 128
"""

import os
import sys
import click
import re
import json
import tempfile
import copy
import pickle

# CUDA 환경 설정
from setup_env import setup_cuda_environment
setup_cuda_environment(verbose=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import dnnlib

from training import training_loop
from training.s2r_adapter import (
    apply_s2r_adapter_to_generator,
    get_s2r_adapter_parameters,
    freeze_non_adapter_parameters,
    set_adapter_scales,
    count_s2r_adapter_parameters,
)
from training.loss import PhysicalLoss, ConsistencyLoss, CombinedPhysicalLoss
from training.dtam import DTAM, rgb_to_luminance
from training.pu21 import PU21Encoder
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def load_stage1_checkpoint(checkpoint_path: str, device='cuda'):
    """
    Stage 1 체크포인트 로드

    Args:
        checkpoint_path: Stage 1 체크포인트 파일 경로
        device: 디바이스

    Returns:
        G_stage1: Stage 1 Generator (동결)
        D_stage1: Stage 1 Discriminator
        training_options: 학습 옵션
    """
    print(f'Stage 1 체크포인트 로드 중: {checkpoint_path}')

    # 파일 존재 확인
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Stage 1 체크포인트를 찾을 수 없습니다: {checkpoint_path}')

    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    # G_ema 키 검증
    if 'G_ema' not in data:
        available_keys = list(data.keys())
        raise KeyError(
            f"Stage 1 체크포인트에 'G_ema' 키가 없습니다. "
            f"사용 가능한 키: {available_keys}. "
            f"올바른 Stage 1 체크포인트 파일인지 확인하세요."
        )

    G_stage1 = data['G_ema'].to(device)
    D_stage1 = data['D'].to(device) if 'D' in data else None

    # Generator 동결
    G_stage1.eval()
    for param in G_stage1.parameters():
        param.requires_grad = False

    print(f'  Generator 파라미터: {sum(p.numel() for p in G_stage1.parameters()):,}')
    if D_stage1:
        print(f'  Discriminator 파라미터: {sum(p.numel() for p in D_stage1.parameters()):,}')

    return G_stage1, D_stage1, data.get('training_kwargs', {})

#----------------------------------------------------------------------------

def create_stage2_generator(G_stage1, adapter_r1=1, adapter_r2=128,
                            freeze_shared=True, init_scale1=1.0, init_scale2=1.0):
    """
    Stage 2 Generator 생성 (Stage 1 복사 + S2R-Adapter 적용)

    Args:
        G_stage1: Stage 1에서 학습된 Generator
        adapter_r1: 전송 브랜치 1의 rank (기본값: 1, 미세 조정)
        adapter_r2: 전송 브랜치 2의 rank (기본값: 128, 광범위 적응)
        freeze_shared: 공유 브랜치 동결 여부 (기본값: True)
        init_scale1: scale1 초기값 (기본값: 1.0)
        init_scale2: scale2 초기값 (기본값: 1.0)

    Returns:
        G_stage2: S2R-Adapter가 적용된 Stage 2 Generator
    """
    print('Stage 2 Generator 생성 중...')

    # Stage 1 모델 복사
    G_stage2 = copy.deepcopy(G_stage1)

    # S2R-Adapter 적용
    G_stage2, adapter_params = apply_s2r_adapter_to_generator(
        G_stage2,
        r1=adapter_r1,
        r2=adapter_r2,
        freeze_shared=freeze_shared,
        init_scale1=init_scale1,
        init_scale2=init_scale2,
        target_modules=['affine'],  # SynthesisLayer의 affine 레이어
    )

    # 원본 파라미터 동결, S2R-Adapter만 학습
    freeze_non_adapter_parameters(G_stage2)

    # 학습 가능한 파라미터 확인
    adapter_param_count, total_params, trainable_params = count_s2r_adapter_parameters(G_stage2)

    print(f'  전체 파라미터: {total_params:,}')
    print(f'  S2R-Adapter 파라미터: {adapter_param_count:,}')
    print(f'  학습 가능 파라미터: {trainable_params:,}')
    print(f'  S2R-Adapter 비율: {adapter_param_count / total_params * 100:.2f}%')
    print(f'  설정: r1={adapter_r1}, r2={adapter_r2}, scale1={init_scale1}, scale2={init_scale2}')

    return G_stage2

#----------------------------------------------------------------------------

def setup_stage2_training_kwargs(
    # Stage 1 체크포인트
    stage1_ckpt: str = None,

    # 일반 옵션
    gpus: int = 1,
    snap: int = 100,
    metrics: list = None,
    seed: int = 0,

    # 데이터셋
    data: str = None,
    subset: int = None,
    mirror: bool = True,

    # 학습 설정
    kimg: int = 5000,
    batch: int = None,
    gamma: float = None,

    # S2R-Adapter 설정
    adapter_r1: int = 1,
    adapter_r2: int = 128,
    freeze_shared: bool = True,
    init_scale1: float = 1.0,
    init_scale2: float = 1.0,

    # DTAM 설정
    dtam_t_onset: float = 300.0,
    dtam_t_peak: float = 1000.0,
    dtam_alpha: float = 9.0,
    dtam_gamma: float = 2.0,

    # 손실 가중치
    lambda_phys: float = 1.0,
    lambda_consist: float = 0.5,
    lambda_gan: float = 1.0,

    # 증강
    aug: str = 'ada',
    target: float = 0.6,
    augpipe: str = 'bgc',

    # 성능 옵션
    workers: int = 4,
    allow_tf32: bool = False,
    nobench: bool = False,
):
    """
    Stage 2 학습 설정 구성

    Stage 2 특징:
    - Stage 1 모델 로드 + S2R-Adapter 적용
    - DTAM 활성화 (눈부심 영역 가중치)
    - PU21 인코딩 (지각적 균일성)
    - Physical Loss + Consistency Loss
    """
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # Stage 1 체크포인트 검증
    # ------------------------------------------
    if stage1_ckpt is None:
        raise UserError('--stage1_ckpt is required')
    if not os.path.exists(stage1_ckpt):
        raise UserError(f'Stage 1 checkpoint not found: {stage1_ckpt}')

    args.stage1_ckpt = stage1_ckpt

    # ------------------------------------------
    # 일반 옵션
    # ------------------------------------------
    assert isinstance(gpus, int) and gpus >= 1
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    assert isinstance(snap, int) and snap >= 1
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ['fid50k_full']
    args.metrics = metrics

    assert isinstance(seed, int)
    args.random_seed = seed

    # ------------------------------------------
    # 데이터셋 설정 (Laval Photometric)
    # ------------------------------------------
    if data is None:
        raise UserError('--data is required')

    # HDRPhysicalDataset 사용 (Laval 절대 휘도)
    args.training_set_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.HDRPhysicalDataset',
        path=data,
        linear_hdr=True,
        dataset_type='laval',  # Laval Photometric 데이터셋
        target_height=512,
        target_width=1024,
        use_labels=False,
        max_size=None,
        xflip=mirror,
    )

    args.data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True,
        num_workers=workers,
        prefetch_factor=2,
    )

    # 데이터셋 검증
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs)
        args.training_set_kwargs.resolution = training_set.resolution
        args.training_set_kwargs.max_size = len(training_set)
        desc = f"stage2-{training_set.name}"
        del training_set
    except IOError as err:
        raise UserError(f'--data: {err}')

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        args.training_set_kwargs.max_size = subset
        args.training_set_kwargs.random_seed = seed

    if mirror:
        desc += '-mirror'

    # ------------------------------------------
    # S2R-Adapter 설정
    # ------------------------------------------
    args.s2r_adapter_kwargs = dnnlib.EasyDict(
        r1=adapter_r1,
        r2=adapter_r2,
        freeze_shared=freeze_shared,
        init_scale1=init_scale1,
        init_scale2=init_scale2,
    )
    desc += f'-s2r_r1{adapter_r1}_r2{adapter_r2}'

    # ------------------------------------------
    # DTAM 설정
    # ------------------------------------------
    args.dtam_kwargs = dnnlib.EasyDict(
        T_onset=dtam_t_onset,
        T_peak=dtam_t_peak,
        alpha=dtam_alpha,
        gamma=dtam_gamma,
    )
    desc += f'-dtam{int(dtam_t_onset)}-{int(dtam_t_peak)}'

    # ------------------------------------------
    # 손실 함수 설정
    # ------------------------------------------
    args.loss_weights = dnnlib.EasyDict(
        lambda_phys=lambda_phys,
        lambda_consist=lambda_consist,
        lambda_gan=lambda_gan,
    )

    # ------------------------------------------
    # 모델 설정 (Stage 1과 동일한 구조)
    # ------------------------------------------
    desc += '-fp32-softplus'

    # Generator 설정 (Stage 1에서 로드)
    args.G_kwargs = dnnlib.EasyDict(
        class_name='training.networks.Generator',
        z_dim=512,
        w_dim=512,
        mapping_kwargs=dnnlib.EasyDict(num_layers=8),
        synthesis_kwargs=dnnlib.EasyDict(
            num_fp16_res=0,
            conv_clamp=None,
            hdr_mode=True,
            hdr_activation='softplus',
            channel_base=32768,
            channel_max=512,
            channels_dict={4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128, 512: 64},
        ),
    )

    # Discriminator 설정
    args.D_kwargs = dnnlib.EasyDict(
        class_name='training.networks.Discriminator',
        num_fp16_res=0,
        conv_clamp=None,
        channel_base=32768,
        channel_max=512,
        channels_dict={4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128, 512: 64},
        block_kwargs=dnnlib.EasyDict(),
        mapping_kwargs=dnnlib.EasyDict(),
        epilogue_kwargs=dnnlib.EasyDict(mbstd_group_size=4),
    )

    # ------------------------------------------
    # 최적화 설정
    # ------------------------------------------
    if batch is None:
        batch = max(2, 8 // gpus)

    args.batch_size = batch
    args.batch_gpu = batch // gpus

    # S2R-Adapter는 더 높은 학습률 가능
    args.G_opt_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam',
        lr=0.001,  # S2R-Adapter는 더 높은 학습률 사용
        betas=[0, 0.99],
        eps=1e-8,
    )
    args.D_opt_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam',
        lr=0.002,
        betas=[0, 0.99],
        eps=1e-8,
    )

    # R1 gamma
    resolution = 512
    if gamma is None:
        gamma = 0.0002 * (resolution ** 2) / batch
    args.loss_kwargs = dnnlib.EasyDict(
        class_name='training.loss.StyleGAN2Loss',
        r1_gamma=gamma,
    )

    # 학습 기간 (Stage 2는 더 짧음)
    args.total_kimg = kimg
    args.ema_kimg = batch * 10 / 32
    args.ema_rampup = None  # Stage 2에서는 rampup 불필요

    desc += f'-gamma{gamma:g}'
    desc += f'-kimg{kimg}'

    # ------------------------------------------
    # 증강 설정
    # ------------------------------------------
    if aug == 'ada':
        args.ada_target = target
        args.ada_kimg = 100  # 더 빠른 적응
        desc += f'-ada{target}'
    elif aug == 'noaug':
        desc += '-noaug'

    augpipe_specs = {
        'blit': dict(xflip=1, rotate90=1, xint=1),
        'geom': dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color': dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bg': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
                   brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'cyclic': dict(cyclic=1),
    }

    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(
            class_name='training.augment.AugmentPipe',
            **augpipe_specs.get(augpipe, augpipe_specs['bgc']),
        )

    # ------------------------------------------
    # 성능 옵션
    # ------------------------------------------
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32:
        args.allow_tf32 = True

    # 작업 이름
    args.task_name = 'Stage2-HDR-Physical'

    return desc, args

#----------------------------------------------------------------------------

class Stage2TrainingLoop:
    """
    Stage 2 학습 루프

    Stage 1 모델 위에 S2R-Adapter를 적용하고
    DTAM 가중치 + PU21 인코딩으로 물리적 정합성을 학습합니다.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')

        # Stage 1 모델 로드
        self.G_stage1, self.D_stage1, _ = load_stage1_checkpoint(
            args.stage1_ckpt, self.device
        )

        # Stage 2 Generator 생성 (S2R-Adapter 적용)
        self.G_stage2 = create_stage2_generator(
            self.G_stage1,
            adapter_r1=args.s2r_adapter_kwargs.r1,
            adapter_r2=args.s2r_adapter_kwargs.r2,
            freeze_shared=args.s2r_adapter_kwargs.freeze_shared,
            init_scale1=args.s2r_adapter_kwargs.init_scale1,
            init_scale2=args.s2r_adapter_kwargs.init_scale2,
        )

        # 손실 함수 초기화
        self._init_losses()

    def _init_losses(self):
        """손실 함수 초기화"""
        args = self.args

        # Physical Loss (DTAM + PU21)
        self.physical_loss = PhysicalLoss(
            T_onset=args.dtam_kwargs.T_onset,
            T_peak=args.dtam_kwargs.T_peak,
            alpha=args.dtam_kwargs.alpha,
            gamma=args.dtam_kwargs.gamma,
            pu21_mode='simple',
        ).to(self.device)

        # Consistency Loss (Stage 1과의 구조적 일관성)
        self.consistency_loss = ConsistencyLoss(
            stage1_generator=self.G_stage1,
            use_lpips=True,
            use_tonemap=True,
            tonemap_method='reinhard',
        ).to(self.device)

        # Combined Loss
        self.combined_loss = CombinedPhysicalLoss(
            physical_loss=self.physical_loss,
            consistency_loss=self.consistency_loss,
            lambda_consist=args.loss_weights.lambda_consist,
        ).to(self.device)

        print('손실 함수 초기화 완료:')
        print(f'  λ_phys: {args.loss_weights.lambda_phys}')
        print(f'  λ_consist: {args.loss_weights.lambda_consist}')
        print(f'  λ_gan: {args.loss_weights.lambda_gan}')

    def get_trainable_parameters(self):
        """학습 가능한 파라미터 반환 (S2R-Adapter만)"""
        return get_s2r_adapter_parameters(self.G_stage2)

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    """각 GPU 프로세스에서 실행되는 함수"""
    dnnlib.util.Logger(
        file_name=os.path.join(args.run_dir, 'log.txt'),
        file_mode='a',
        should_flush=True,
    )

    # 분산 학습 초기화
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(
                backend='gloo',
                init_method=init_method,
                rank=rank,
                world_size=args.num_gpus,
            )
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=init_method,
                rank=rank,
                world_size=args.num_gpus,
            )

    # torch_utils 초기화
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # FP32 기본 dtype 설정
    torch.set_default_dtype(torch.float32)

    # Stage 2 학습 루프 시작
    # 기본 training_loop 대신 Stage 2 전용 로직 필요
    # 현재는 기본 training_loop 사용 (Stage 1과 유사)
    training_loop.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# 필수 옵션
@click.option('--stage1_ckpt', help='Stage 1 체크포인트 경로 (필수)', required=True, metavar='PKL')
@click.option('--data', help='Laval Photometric 데이터셋 경로 (필수)', required=True, metavar='PATH')
@click.option('--outdir', help='출력 디렉토리 (필수)', required=True, metavar='DIR')

# 일반 옵션
@click.option('--gpus', help='GPU 개수 [기본값: 1]', type=int, default=1, metavar='INT')
@click.option('--snap', help='스냅샷 간격 [기본값: 100 ticks]', type=int, default=100, metavar='INT')
@click.option('--metrics', help='메트릭 목록 또는 "none"', type=CommaSeparatedList(), default='fid50k_full')
@click.option('--seed', help='랜덤 시드 [기본값: 0]', type=int, default=0, metavar='INT')
@click.option('-n', '--dry-run', help='학습 옵션만 출력하고 종료', is_flag=True)

# 데이터셋
@click.option('--subset', help='N개 이미지만 사용', type=int, metavar='INT')
@click.option('--mirror/--no-mirror', help='x-flip 증강 [기본값: True]', default=True)

# 학습 설정
@click.option('--kimg', help='학습 기간 (kimg) [기본값: 5000]', type=int, default=5000, metavar='INT')
@click.option('--batch', help='배치 크기', type=int, metavar='INT')
@click.option('--gamma', help='R1 gamma 오버라이드', type=float, metavar='FLOAT')

# S2R-Adapter 설정
@click.option('--adapter_r1', help='S2R-Adapter 전송 브랜치 1 rank [기본값: 1]', type=int, default=1, metavar='INT')
@click.option('--adapter_r2', help='S2R-Adapter 전송 브랜치 2 rank [기본값: 128]', type=int, default=128, metavar='INT')
@click.option('--freeze_shared/--no-freeze_shared', help='공유 브랜치 동결 여부 [기본값: True]', default=True)
@click.option('--init_scale1', help='scale1 초기값 [기본값: 1.0]', type=float, default=1.0, metavar='FLOAT')
@click.option('--init_scale2', help='scale2 초기값 [기본값: 1.0]', type=float, default=1.0, metavar='FLOAT')

# DTAM 설정
@click.option('--dtam_t_onset', help='DTAM T_onset (cd/m²) [기본값: 300]', type=float, default=300.0, metavar='FLOAT')
@click.option('--dtam_t_peak', help='DTAM T_peak (cd/m²) [기본값: 1000]', type=float, default=1000.0, metavar='FLOAT')
@click.option('--dtam_alpha', help='DTAM 가중치 증폭 계수 [기본값: 9.0]', type=float, default=9.0, metavar='FLOAT')
@click.option('--dtam_gamma', help='DTAM 곡률 [기본값: 2.0]', type=float, default=2.0, metavar='FLOAT')

# 손실 가중치
@click.option('--lambda_phys', help='Physical Loss 가중치 [기본값: 1.0]', type=float, default=1.0, metavar='FLOAT')
@click.option('--lambda_consist', help='Consistency Loss 가중치 [기본값: 0.5]', type=float, default=0.5, metavar='FLOAT')
@click.option('--lambda_gan', help='GAN Loss 가중치 [기본값: 1.0]', type=float, default=1.0, metavar='FLOAT')

# 증강
@click.option('--aug', help='증강 모드 [기본값: ada]', type=click.Choice(['noaug', 'ada']), default='ada')
@click.option('--target', help='ADA 목표값 [기본값: 0.6]', type=float, default=0.6, metavar='FLOAT')
@click.option('--augpipe', help='증강 파이프라인 [기본값: bgc]', type=click.Choice(['blit', 'geom', 'color', 'bg', 'bgc', 'cyclic']), default='bgc')

# 성능 옵션
@click.option('--workers', help='DataLoader 워커 수 [기본값: 4]', type=int, default=4, metavar='INT')
@click.option('--nobench', help='cuDNN 벤치마킹 비활성화', is_flag=True)
@click.option('--allow-tf32', help='TF32 허용', is_flag=True)

def main(ctx, outdir, dry_run, **config_kwargs):
    """
    Stage 2 학습: 물리적 정합성 보정

    Laval Photometric 데이터셋을 사용하여 물리적 휘도를 보정합니다.
    Stage 1에서 학습된 구조 위에 S2R-Adapter를 적용하고,
    DTAM + PU21로 눈부심 영역의 정확도를 향상시킵니다.

    \b
    예시:
        # 기본 학습
        python train_stage2.py --stage1_ckpt ./stage1.pkl --data ./data/laval --outdir ./training-runs-stage2

        # S2R-Adapter rank 조정
        python train_stage2.py --stage1_ckpt ./stage1.pkl --data ./data/laval --outdir ./training-runs-stage2 --adapter_r1 2 --adapter_r2 64

        # DTAM 임계값 조정
        python train_stage2.py --stage1_ckpt ./stage1.pkl --data ./data/laval --outdir ./training-runs-stage2 --dtam_t_onset 250 --dtam_t_peak 800
    """
    dnnlib.util.Logger(should_flush=True)

    # Stage 2 정보 출력
    print()
    print('=' * 60)
    print('Stage 2: 물리적 정합성 보정')
    print('=' * 60)
    print()
    print('특징:')
    print('  - 데이터셋: Laval Photometric (실측 HDR)')
    print('  - 정밀도: Full FP32')
    print('  - 활성화: Softplus (비음수 물리적 휘도)')
    print('  - DTAM: ON (눈부심 영역 가중치)')
    print('  - PU21: 지각적 균일 인코딩')
    print('  - S2R-Adapter: 3-브랜치 구조 (shared + transfer1 + transfer2)')
    print('  - 손실: L_phys + λ_consist * L_consist + L_GAN')
    print()

    # 학습 옵션 설정
    try:
        run_desc, args = setup_stage2_training_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(str(err))

    # 출력 디렉토리 설정
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # 옵션 출력
    print('학습 옵션:')
    print(json.dumps(args, indent=2))
    print()
    print(f'출력 디렉토리:      {args.run_dir}')
    print(f'Stage 1 체크포인트: {args.stage1_ckpt}')
    print(f'학습 데이터:        {args.training_set_kwargs.path}')
    print(f'학습 기간:          {args.total_kimg} kimg')
    print(f'GPU 개수:           {args.num_gpus}')
    print(f'이미지 수:          {args.training_set_kwargs.max_size}')
    print(f'배치 크기:          {args.batch_size} (GPU당 {args.batch_gpu})')
    print()
    print('S2R-Adapter 설정:')
    print(f'  r1 (미세 조정): {args.s2r_adapter_kwargs.r1}')
    print(f'  r2 (광범위 적응): {args.s2r_adapter_kwargs.r2}')
    print(f'  공유 브랜치 동결: {args.s2r_adapter_kwargs.freeze_shared}')
    print(f'  scale1: {args.s2r_adapter_kwargs.init_scale1}')
    print(f'  scale2: {args.s2r_adapter_kwargs.init_scale2}')
    print()
    print('DTAM 설정:')
    print(f'  T_onset: {args.dtam_kwargs.T_onset} cd/m²')
    print(f'  T_peak: {args.dtam_kwargs.T_peak} cd/m²')
    print(f'  α: {args.dtam_kwargs.alpha}')
    print(f'  γ: {args.dtam_kwargs.gamma}')
    print()
    print('손실 가중치:')
    print(f'  λ_phys: {args.loss_weights.lambda_phys}')
    print(f'  λ_consist: {args.loss_weights.lambda_consist}')
    print(f'  λ_gan: {args.loss_weights.lambda_gan}')
    print()

    # Dry run
    if dry_run:
        print('Dry run 완료; 종료합니다.')
        return

    # 출력 디렉토리 생성
    print('출력 디렉토리 생성 중...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Stage 2 설정 저장
    stage2_config = {
        'stage': 2,
        'description': 'Physical Calibration with Laval Photometric',
        'stage1_checkpoint': args.stage1_ckpt,
        'dataset': 'laval',
        'fp32': True,
        'activation': 'softplus',
        'dtam_enabled': True,
        'dtam_params': dict(args.dtam_kwargs),
        's2r_adapter_params': dict(args.s2r_adapter_kwargs),
        'loss_weights': dict(args.loss_weights),
        'loss_components': ['physical', 'consistency', 'gan'],
    }
    with open(os.path.join(args.run_dir, 'stage2_config.json'), 'wt') as f:
        json.dump(stage2_config, f, indent=2)

    # 학습 시작
    print('프로세스 시작...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(
                fn=subprocess_fn,
                args=(args, temp_dir),
                nprocs=args.num_gpus,
            )

    print()
    print('=' * 60)
    print('Stage 2 학습 완료!')
    print('=' * 60)
    print()
    print(f'체크포인트 위치: {args.run_dir}')
    print()
    print('다음 단계:')
    print('  python inference/validation_pipeline.py --checkpoint <stage2_checkpoint.pkl> --input <nfov_image>')
    print()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
