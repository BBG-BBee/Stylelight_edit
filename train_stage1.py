"""
Stage 1 학습 스크립트: 구조 학습 (Structure Learning)

S2R-HDR 데이터셋으로 HDR 파노라마 구조를 학습합니다.
- DTAM: OFF (비활성화)
- 손실: GAN + LPIPS + L2
- 정밀도: Full FP32
- 활성화: Softplus (비음수 물리적 휘도)

사용법:
    python train_stage1.py --data /path/to/s2r_hdr --outdir ./training-runs-stage1
    python train_stage1.py --data /path/to/s2r_hdr --outdir ./training-runs-stage1 --gpus 1 --batch 2
"""

import os
import sys
import click
import re
import json
import tempfile
import copy

# CUDA 환경 설정
from setup_env import setup_cuda_environment
setup_cuda_environment(verbose=False)

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

def setup_stage1_training_kwargs(
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
    kimg: int = 10000,
    batch: int = None,
    gamma: float = None,

    # 증강
    aug: str = 'ada',
    target: float = 0.6,
    augpipe: str = 'bgc',

    # 전이 학습
    resume: str = None,

    # 성능 옵션
    workers: int = 4,
    allow_tf32: bool = False,
    nobench: bool = False,
):
    """
    Stage 1 학습 설정 구성

    Stage 1 특징:
    - Full FP32 (HDR 정밀도 보존)
    - Softplus 활성화 (비음수 물리적 휘도)
    - DTAM 비활성화 (구조 학습에 집중)
    - 기본 GAN + LPIPS 손실
    """
    args = dnnlib.EasyDict()

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
    # 데이터셋 설정 (HDRPhysicalDataset 사용)
    # ------------------------------------------
    if data is None:
        raise UserError('--data is required')

    # HDRPhysicalDataset 사용 (FP32 선형 HDR)
    args.training_set_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.HDRPhysicalDataset',
        path=data,
        target_height=512,
        target_width=1024,
        use_tilt_augment=True,  # 카메라 기울기 augmentation (Stage 1 전용)
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
        desc = f"stage1-{training_set.name}"
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
    # 모델 설정 (Full FP32 + Softplus)
    # ------------------------------------------
    desc += '-fp32-softplus'

    # Generator 설정
    args.G_kwargs = dnnlib.EasyDict(
        class_name='training.networks.Generator',
        z_dim=512,
        w_dim=512,
        mapping_kwargs=dnnlib.EasyDict(num_layers=8),
        synthesis_kwargs=dnnlib.EasyDict(
            # Full FP32 강제화
            num_fp16_res=0,
            conv_clamp=None,
            # HDR 모드 설정
            hdr_mode=True,
            hdr_activation='softplus',
            # 채널 설정
            channel_base=32768,
            channel_max=512,
            channels_dict={4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128, 512: 64},
        ),
    )

    # Discriminator 설정
    args.D_kwargs = dnnlib.EasyDict(
        class_name='training.networks.Discriminator',
        # Full FP32
        num_fp16_res=0,
        conv_clamp=None,
        # 채널 설정
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
    # RTX 5090 기준 배치 크기 설정
    if batch is None:
        batch = max(2, 8 // gpus)  # GPU당 2-4

    args.batch_size = batch
    args.batch_gpu = batch // gpus

    # 학습률 설정
    args.G_opt_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam',
        lr=0.002,
        betas=[0, 0.99],
        eps=1e-8,
    )
    args.D_opt_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam',
        lr=0.002,
        betas=[0, 0.99],
        eps=1e-8,
    )

    # R1 gamma 설정
    resolution = 512  # 512x1024
    if gamma is None:
        gamma = 0.0002 * (resolution ** 2) / batch
    args.loss_kwargs = dnnlib.EasyDict(
        class_name='training.loss.StyleGAN2Loss',
        r1_gamma=gamma,
    )

    # 학습 기간
    args.total_kimg = kimg
    args.ema_kimg = batch * 10 / 32
    args.ema_rampup = 0.05

    desc += f'-gamma{gamma:g}'
    desc += f'-kimg{kimg}'

    # ------------------------------------------
    # 증강 설정
    # ------------------------------------------
    if aug == 'ada':
        args.ada_target = target
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
    # 전이 학습
    # ------------------------------------------
    if resume is not None:
        args.resume_pkl = resume
        args.ada_kimg = 100
        args.ema_rampup = None
        desc += '-resume'

    # ------------------------------------------
    # 성능 옵션
    # ------------------------------------------
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32:
        args.allow_tf32 = True

    # 작업 이름
    args.task_name = 'Stage1-HDR-Structure'

    return desc, args

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

    # 학습 루프 실행
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
@click.option('--data', help='S2R-HDR 데이터셋 경로 (필수)', required=True, metavar='PATH')
@click.option('--outdir', help='출력 디렉토리 (필수)', required=True, metavar='DIR')

# 일반 옵션
@click.option('--gpus', help='GPU 개수 [기본값: 1]', type=int, default=1, metavar='INT')
@click.option('--snap', help='스냅샷 간격 [기본값: 100 ticks]', type=int, default=100, metavar='INT')
@click.option('--metrics', help='메트릭 목록 또는 "none" [기본값: fid50k_full]', type=CommaSeparatedList(), default='fid50k_full')
@click.option('--seed', help='랜덤 시드 [기본값: 0]', type=int, default=0, metavar='INT')
@click.option('-n', '--dry-run', help='학습 옵션만 출력하고 종료', is_flag=True)

# 데이터셋
@click.option('--subset', help='N개 이미지만 사용 [기본값: 전체]', type=int, metavar='INT')
@click.option('--mirror/--no-mirror', help='x-flip 증강 [기본값: True]', default=True)

# 학습 설정
@click.option('--kimg', help='학습 기간 (kimg) [기본값: 10000]', type=int, default=10000, metavar='INT')
@click.option('--batch', help='배치 크기 [기본값: GPU에 맞게 자동]', type=int, metavar='INT')
@click.option('--gamma', help='R1 gamma 오버라이드', type=float, metavar='FLOAT')

# 증강
@click.option('--aug', help='증강 모드 [기본값: ada]', type=click.Choice(['noaug', 'ada']), default='ada')
@click.option('--target', help='ADA 목표값 [기본값: 0.6]', type=float, default=0.6, metavar='FLOAT')
@click.option('--augpipe', help='증강 파이프라인 [기본값: bgc]', type=click.Choice(['blit', 'geom', 'color', 'bg', 'bgc', 'cyclic']), default='bgc')

# 전이 학습
@click.option('--resume', help='체크포인트에서 재개', type=str, metavar='PKL')

# 성능 옵션
@click.option('--workers', help='DataLoader 워커 수 [기본값: 4]', type=int, default=4, metavar='INT')
@click.option('--nobench', help='cuDNN 벤치마킹 비활성화', is_flag=True)
@click.option('--allow-tf32', help='TF32 허용', is_flag=True)

def main(ctx, outdir, dry_run, **config_kwargs):
    """
    Stage 1 학습: HDR 파노라마 구조 학습

    S2R-HDR 데이터셋을 사용하여 기본 구조를 학습합니다.
    Full FP32 정밀도와 Softplus 활성화를 사용합니다.

    \b
    예시:
        # 기본 학습 (1 GPU)
        python train_stage1.py --data ./data/s2r_hdr --outdir ./training-runs-stage1

        # 2 GPU, 배치 크기 4
        python train_stage1.py --data ./data/s2r_hdr --outdir ./training-runs-stage1 --gpus 2 --batch 4

        # 체크포인트에서 재개
        python train_stage1.py --data ./data/s2r_hdr --outdir ./training-runs-stage1 --resume ./checkpoint.pkl
    """
    dnnlib.util.Logger(should_flush=True)

    # Stage 1 정보 출력
    print()
    print('=' * 60)
    print('Stage 1: HDR 파노라마 구조 학습')
    print('=' * 60)
    print()
    print('특징:')
    print('  - 데이터셋: S2R-HDR (합성 HDR 파노라마)')
    print('  - 정밀도: Full FP32')
    print('  - 활성화: Softplus (비음수 물리적 휘도)')
    print('  - DTAM: OFF (비활성화)')
    print('  - 손실: GAN + LPIPS + L2')
    print()

    # 학습 옵션 설정
    try:
        run_desc, args = setup_stage1_training_kwargs(**config_kwargs)
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
    print(f'출력 디렉토리:   {args.run_dir}')
    print(f'학습 데이터:     {args.training_set_kwargs.path}')
    print(f'학습 기간:       {args.total_kimg} kimg')
    print(f'GPU 개수:        {args.num_gpus}')
    print(f'이미지 수:       {args.training_set_kwargs.max_size}')
    print(f'해상도:          {args.training_set_kwargs.resolution}')
    print(f'배치 크기:       {args.batch_size} (GPU당 {args.batch_gpu})')
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

    # Stage 1 설정 저장
    stage1_config = {
        'stage': 1,
        'description': 'Structure Learning with S2R-HDR',
        'dataset': 's2r_hdr',
        'fp32': True,
        'activation': 'softplus',
        'dtam_enabled': False,
        'loss_components': ['gan', 'lpips', 'l2'],
    }
    with open(os.path.join(args.run_dir, 'stage1_config.json'), 'wt') as f:
        json.dump(stage1_config, f, indent=2)

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
    print('Stage 1 학습 완료!')
    print('=' * 60)
    print()
    print(f'체크포인트 위치: {args.run_dir}')
    print()
    print('다음 단계:')
    print('  python train_stage2.py --stage1_ckpt <stage1_checkpoint.pkl> --data <laval_dataset>')
    print()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
