# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
HDR GAN Inversion: 입력 HDR 이미지를 Generator의 잠재 공간에 역투영

물리적 휘도(cd/m²)를 보존하면서 최적의 latent code w를 탐색합니다.
Stage 2 학습과 동일한 PhysicalLoss(DTAM+PU21)를 사용하여
고휘도 영역(광원)을 정확하게 복원합니다.

손실 구성:
    L = λ_phys * L_phys + λ_struct * L_struct + λ_reg * L_reg
    - L_phys: DTAM 가중 PU21 L1 손실 (물리적 휘도 정합)
    - L_struct: 톤매핑 후 VGG LPIPS (구조적/지각적 유사도)
    - L_reg: 노이즈 정규화 (StyleGAN2 표준)
"""

import copy
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PTI_utils import global_config, hyperparameters, paths_config
from PTI_utils import log_utils
import dnnlib
from PTI_utils import l2_loss
import PIL.Image

from training.loss import PhysicalLoss
from training.dtam import rgb_to_luminance


def _reinhard_tonemap(hdr, key=0.18, burn=10.0):
    """
    Reinhard 톤매핑 (VGG LPIPS 입력용)

    HDR → [0, 255] LDR 변환. 구조적 비교 전용.
    물리적 손실에는 사용하지 않는다.
    """
    luminance = rgb_to_luminance(hdr)  # (B, 1, H, W)
    lum_mean = torch.exp(torch.mean(torch.log(luminance + 1e-6)))
    lum_scaled = (key / lum_mean) * luminance
    ldr = hdr * (lum_scaled / (luminance + 1e-6)) * (1.0 + lum_scaled / (burn ** 2)) / (1.0 + lum_scaled)
    ldr = torch.clamp(ldr * 255.0, 0, 255)
    return ldr


def project(
        G,
        bbox,
        target: torch.Tensor,  # [C,H,W] HDR 물리적 휘도 스케일 (cd/m²)
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        lambda_phys=1.0,
        lambda_struct=1.0,
        verbose=False,
        device: torch.device,
        use_wandb=False,
        initial_w=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        w_name: str,
):
    """
    HDR GAN Inversion: 초점 마스킹 기반 latent code 탐색

    Generator 출력(Softplus, [0, ∞))의 bbox 영역을
    입력 HDR 이미지와 비교하여 최적의 w를 찾는다.

    Args:
        G: Generator 모델
        bbox: [row_min, row_max, col_min, col_max] — ERP 상 NFoV 영역
        target: [C, H, W] HDR 이미지 (cd/m² 스케일, FP32)
        lambda_phys: 물리적 손실 가중치 (DTAM+PU21)
        lambda_struct: 구조적 손실 가중치 (톤매핑 LPIPS)

    Returns:
        w_opt: [1, num_ws, 512] 최적화된 latent code
    """
    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()

    # W 공간 통계 계산
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg

    # 노이즈 버퍼 설정
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # VGG16 (구조적 비교용)
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # 물리적 손실 함수 (Stage 2 학습과 동일)
    phys_loss_fn = PhysicalLoss().to(device).eval()

    # 타겟 HDR 이미지 준비 (cd/m² 스케일 유지)
    target_hdr = target.unsqueeze(0).to(device).to(torch.float32)  # (1, C, H, W)

    # bbox 영역 추출 — 이 영역만 손실에 기여
    target_crop = target_hdr[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]

    # VGG용 타겟: 톤매핑 후 리사이즈
    target_tm = _reinhard_tonemap(target_crop)
    target_tm_resized = F.interpolate(target_tm, size=(128, 256), mode='area')
    target_features = vgg16(target_tm_resized, resize_images=False, return_lpips=True)

    # 최적화 변수 초기화
    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)

    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in tqdm(range(num_steps)):
        # 학습률 스케줄
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Generator 출력 (Softplus: [0, ∞))
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const', force_fp32=True)
        synth_hdr = torch.clamp(synth_images, min=0.0)  # Softplus 출력, 음수만 방지

        # bbox 영역 추출
        synth_crop = synth_hdr[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]

        # ── 물리적 손실: DTAM 가중 PU21 L1 (cd/m² 직접 비교) ──
        loss_phys = phys_loss_fn(synth_crop, target_crop)

        # ── 구조적 손실: 톤매핑 → VGG LPIPS ──
        synth_tm = _reinhard_tonemap(synth_crop)
        synth_tm_resized = F.interpolate(synth_tm, size=(128, 256), mode='area')
        synth_features = vgg16(synth_tm_resized, resize_images=False, return_lpips=True)
        loss_struct = (target_features - synth_features).square().sum()

        # ── 노이즈 정규화 ──
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        # ── 총 손실 ──
        loss = lambda_phys * loss_phys + lambda_struct * loss_struct + reg_loss * regularize_noise_weight

        if step % image_log_step == 0:
            with torch.no_grad():
                if use_wandb:
                    global_config.training_step += 1
                    wandb.log({f'first projection _{w_name}': loss.detach().cpu()}, step=global_config.training_step)
                    log_utils.log_image_from_w(w_opt.repeat([1, G.mapping.num_ws, 1]), G, w_name)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: phys {loss_phys:<4.2f} struct {loss_struct:<4.2f} loss {float(loss):<5.2f}')

        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    del G
    return w_opt.repeat([1, 18, 1])
