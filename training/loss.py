# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from PIL import Image
import math
from typing import Optional

# DTAM 및 PU21 임포트 (Stage 2에서 사용)
try:
    from training.dtam import DTAM, rgb_to_luminance
    from training.pu21 import PU21Encoder
except ImportError:
    DTAM = None
    PU21Encoder = None


#----------------------------------------------------------------------------
# 물리적 정합성 손실 함수들
#----------------------------------------------------------------------------

class PhysicalLoss(nn.Module):
    """
    물리적 정합성 손실 함수

    DTAM 가중치와 PU21 인코딩을 결합하여
    DGP 계산에 중요한 휘도 영역을 정확하게 학습합니다.

    L_phys = || W_DTAM ⊙ (PU21(Y_pred) - PU21(Y_gt)) ||_1

    Args:
        T_onset: DTAM 학습 시작 임계값 (cd/m²)
        T_peak: DTAM 최대 가중치 도달점 (cd/m²)
        alpha: DTAM 가중치 증폭 계수
        gamma: DTAM 곡률
        pu21_mode: PU21 인코딩 모드 ('simple', 'full')
    """

    def __init__(self,
                 T_onset: float = 300.0,
                 T_peak: float = 1000.0,
                 alpha: float = 9.0,
                 gamma: float = 2.0,
                 pu21_mode: str = 'simple'):
        super().__init__()

        if DTAM is None or PU21Encoder is None:
            raise ImportError("DTAM and PU21Encoder are required for PhysicalLoss")

        self.dtam = DTAM(T_onset=T_onset, T_peak=T_peak, alpha=alpha, gamma=gamma)
        self.pu21 = PU21Encoder(mode=pu21_mode)

    def forward(self,
                pred_rgb: torch.Tensor,
                gt_rgb: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        물리적 손실 계산

        Args:
            pred_rgb: 예측 RGB HDR 이미지 (B, 3, H, W), 물리적 휘도 스케일
            gt_rgb: 정답 RGB HDR 이미지 (B, 3, H, W), 물리적 휘도 스케일
            return_components: True면 구성 요소별 손실 반환

        Returns:
            loss: 물리적 손실 스칼라
            (선택) components: {'total', 'background', 'transition', 'glare'}
        """
        # RGB -> 휘도 변환
        pred_luminance = rgb_to_luminance(pred_rgb)
        gt_luminance = rgb_to_luminance(gt_rgb)

        # PU21 인코딩
        pred_pu = self.pu21(pred_luminance)
        gt_pu = self.pu21(gt_luminance)

        # DTAM 가중치
        weight = self.dtam(gt_luminance)

        # 가중치 적용 L1 손실
        diff = torch.abs(pred_pu - gt_pu)
        weighted_diff = weight * diff
        loss = weighted_diff.mean()

        if return_components:
            # 구간별 손실 분석
            bg_mask, trans_mask, glare_mask = self.dtam.get_zone_masks(gt_luminance)

            components = {
                'total': loss,
                'background': diff[bg_mask].mean() if bg_mask.any() else torch.tensor(0.0),
                'transition': diff[trans_mask].mean() if trans_mask.any() else torch.tensor(0.0),
                'glare': diff[glare_mask].mean() if glare_mask.any() else torch.tensor(0.0)
            }
            return loss, components

        return loss


class ConsistencyLoss(nn.Module):
    """
    구조적 일관성 손실 (Structural Consistency Loss)

    Stage 2 학습 시 Stage 1 모델과의 구조적 차이를 최소화합니다.
    LoRA를 통한 물리 보정이 구조적 지식을 파괴하는 것을 방지합니다.

    L_Consist = LPIPS(ToneMap(G_Stage1(z)), ToneMap(G_Stage2(z)))

    Args:
        lpips_net: LPIPS 네트워크 타입 ('alex', 'vgg', 'squeeze')
        tonemap_gamma: 톤맵 감마값
    """

    def __init__(self,
                 lpips_net: str = 'alex',
                 tonemap_gamma: float = 2.4):
        super().__init__()

        # LPIPS 손실 (lazy import)
        try:
            import lpips
            self.lpips = lpips.LPIPS(net=lpips_net)
            self.lpips.eval()
            for param in self.lpips.parameters():
                param.requires_grad = False
        except ImportError:
            print("Warning: lpips not installed. Using L1 loss as fallback.")
            self.lpips = None

        self.tonemap_gamma = tonemap_gamma

    def tonemap(self, hdr_image: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
        """
        HDR 이미지를 LDR로 톤매핑 (LPIPS 비교용)

        간단한 감마 톤매핑 사용 (Reinhard 등 복잡한 방식 대신)
        """
        if gamma is None:
            gamma = self.tonemap_gamma

        # 음수 방지
        hdr_image = torch.clamp(hdr_image, min=1e-6)

        # 감마 톤매핑 + 정규화
        # percentile 기반 정규화
        flat = hdr_image.view(hdr_image.shape[0], -1)
        percentile_95 = torch.quantile(flat, 0.95, dim=1, keepdim=True)
        percentile_95 = percentile_95.view(-1, 1, 1, 1).clamp(min=1.0)

        normalized = hdr_image / percentile_95
        tonemapped = torch.pow(normalized.clamp(0, 1), 1.0 / gamma)

        # [-1, 1] 범위로 변환 (LPIPS 입력 형식)
        return tonemapped * 2 - 1

    def forward(self,
                stage1_output: torch.Tensor,
                stage2_output: torch.Tensor) -> torch.Tensor:
        """
        구조적 일관성 손실 계산

        Args:
            stage1_output: Stage 1 Generator 출력 (동결됨)
            stage2_output: Stage 2 Generator 출력 (학습 중)

        Returns:
            loss: LPIPS 또는 L1 손실
        """
        # 톤매핑 적용
        stage1_tm = self.tonemap(stage1_output)
        stage2_tm = self.tonemap(stage2_output)

        if self.lpips is not None:
            # LPIPS 손실
            # LPIPS는 (B, C, H, W) 형태의 [-1, 1] 범위 입력 기대
            loss = self.lpips(stage1_tm, stage2_tm)
            return loss.mean()
        else:
            # Fallback: L1 손실
            return F.l1_loss(stage1_tm, stage2_tm)

    def to(self, device):
        """디바이스 이동 시 LPIPS도 함께 이동"""
        super().to(device)
        if self.lpips is not None:
            self.lpips = self.lpips.to(device)
        return self


class CombinedPhysicalLoss(nn.Module):
    """
    Stage 2 학습을 위한 통합 물리적 손실 함수

    L_Total = L_Phys + λ_Consist * L_Consist + L_GAN

    이 클래스는 L_Phys와 L_Consist만 담당합니다.
    L_GAN은 StyleGAN2Loss에서 처리됩니다.

    Args:
        phys_weight: 물리적 손실 가중치
        consist_weight: 일관성 손실 가중치
        **phys_kwargs: PhysicalLoss 파라미터
        **consist_kwargs: ConsistencyLoss 파라미터
    """

    def __init__(self,
                 phys_weight: float = 1.0,
                 consist_weight: float = 0.5,
                 T_onset: float = 300.0,
                 T_peak: float = 1000.0,
                 alpha: float = 9.0,
                 gamma: float = 2.0,
                 lpips_net: str = 'alex'):
        super().__init__()

        self.phys_weight = phys_weight
        self.consist_weight = consist_weight

        self.physical_loss = PhysicalLoss(
            T_onset=T_onset,
            T_peak=T_peak,
            alpha=alpha,
            gamma=gamma
        )
        self.consistency_loss = ConsistencyLoss(lpips_net=lpips_net)

    def forward(self,
                pred_rgb: torch.Tensor,
                gt_rgb: torch.Tensor,
                stage1_output: Optional[torch.Tensor] = None,
                stage2_output: Optional[torch.Tensor] = None) -> dict:
        """
        통합 손실 계산

        Args:
            pred_rgb: 예측 HDR 이미지
            gt_rgb: 정답 HDR 이미지
            stage1_output: Stage 1 출력 (일관성 손실용)
            stage2_output: Stage 2 출력 (일관성 손실용)

        Returns:
            losses: {'total', 'physical', 'consistency'} 딕셔너리
        """
        # 물리적 손실
        L_phys = self.physical_loss(pred_rgb, gt_rgb)

        # 일관성 손실 (Stage 1/2 출력이 제공된 경우에만)
        L_consist = torch.tensor(0.0, device=pred_rgb.device)
        if stage1_output is not None and stage2_output is not None:
            L_consist = self.consistency_loss(stage1_output, stage2_output)

        # 총 손실
        L_total = self.phys_weight * L_phys + self.consist_weight * L_consist

        return {
            'total': L_total,
            'physical': L_phys,
            'consistency': L_consist
        }


#----------------------------------------------------------------------------
class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    # def __init__(self, device, G_ldr2hdr, G_mapping, G_synthesis, D, D_, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
    def __init__(self, device, G_mapping, G_synthesis, D, D_, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.D_ = D_
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    # modified, ws+ position
    def run_G(self, z, c, sync):
        """
        생성자(Generator)를 실행합니다.
        Mapping Network를 통해 Z를 W로 변환하고, Synthesis Network를 통해 이미지를 생성합니다.
        LDR과 HDR 이미지를 모두 반환하는 것으로 보입니다.
        """
        with misc.ddp_sync(self.G_mapping, sync):
            # ws = self.G_mapping(z, c)
            ws = self.G_mapping(z, None)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        
        img_shared = img
        # [LDR 브랜치 (LDR Branch)]: 생성된 이미지를 [-1, 1] 범위로 클리핑하여 LDR 이미지 생성
        img_ldr = torch.clip(img_shared, -1, 1)
        # [HDR 브랜치 (HDR Branch)]: 생성된 이미지를 [-1, 10] (또는 그 이상) 범위로 클리핑하여 HDR 이미지(Highlight 보존) 생성
        img_hdr = torch.clip(img_shared, -1, 10)
        img_hdr = torch.clip(img_hdr, -1, 1e8)
        return img_ldr, img_hdr, ws, img

    # for ldr
    def run_D(self, img, c, sync, isRealImage=False):
        """
        LDR 이미지를 위한 판별자(Discriminator)를 실행합니다.
        """
        if isRealImage:
            img = img[:,:3,:,:]    ##(ldr,hdr)
        if self.augment_pipe is not None:
            img = self.augment_pipe(img, isRealImage=isRealImage)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    # for hdr
    def run_D_hdr(self, img, c, sync, isRealImage=False):
        """
        HDR 이미지를 위한 판별자(Discriminator)를 실행합니다.
        """
        if isRealImage:
            img = img[:,3:,:,:]  #(ldr,hdr) ######## run_D와 차이점
        if self.augment_pipe is not None:
            img = self.augment_pipe(img, isRealImage=isRealImage)
        with misc.ddp_sync(self.D_, sync):
            logits = self.D_(img, c)
        return logits


    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        """
        각 학습 단계(phase)에 따라 손실을 계산하고 그래디언트를 누적합니다.
        StyleLight는 LDR 판별자(D)와 HDR 판별자(D_)를 모두 사용하여, 
        생성자가 LDR의 사실감과 HDR의 광원 정보를 동시에 학습하도록 유도합니다.
        """
        # assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'D_main', 'D_reg', 'D_both']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dmain_ = (phase in ['D_main', 'D_both'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        do_Dr1_   = (phase in ['D_reg', 'D_both']) and (self.r1_gamma != 0)

        # Gmain: 생성된 이미지에 대한 로짓을 최대화합니다 (판별자를 속임).
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws, _ = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # Gpl에 의해 동기화될 수 있음.
                gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False,isRealImage=False)    ########  add isRealImage=False
                gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False,isRealImage=False)    ########  add isRealImage=False
                training_stats.report('Loss/scores/fake', gen_logits_)
                training_stats.report('Loss/signs/fake', gen_logits_.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                loss_Gmain_ = torch.nn.functional.softplus(-gen_logits_) # -log(sigmoid(gen_logits))
                # training_stats.report('Loss/G/loss', loss_Gmain+loss_Gmain_)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain+loss_Gmain_).mean().mul(gain).backward()

        # Gpl: 경로 길이 정규화(Path length regularization)를 적용합니다.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img_ldr, gen_img_hdr, gen_ws, img = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(img) / np.sqrt(img.shape[2] * img.shape[3])

                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    # pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                    pl_grads = torch.autograd.grad(outputs=[(img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                # pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                # print('pl_grads shape:',pl_grads.shape)
                if len(gen_ws.shape)==2:
                    pl_lengths = pl_grads.square().sum(1).sqrt()
                else:
                    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: 생성된 이미지에 대한 로짓을 최소화합니다.
        loss_Dgen = 0
        if do_Dmain:
        # if False:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws,_ = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                # gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                # loss_Dgen_ = torch.nn.functional.softplus(gen_logits_) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: 생성된 이미지에 대한 로짓을 최소화합니다. (HDR)
        loss_Dgen_ = 0
        if do_Dmain_:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws,_ = self.run_G(gen_z, gen_c, sync=False)
                # gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                training_stats.report('Loss/scores/fake', gen_logits_)
                training_stats.report('Loss/signs/fake', gen_logits_.sign())
                # loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                loss_Dgen_ = torch.nn.functional.softplus(gen_logits_) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward_'):
                # loss_Dgen.mean().mul(gain).backward()
                loss_Dgen_.mean().mul(gain).backward()
                # (0*loss_Dgen+loss_Dgen_).mean().mul(gain).backward()

        # Dmain: 실제 이미지에 대한 로짓을 최대화합니다.
        # Dr1: R1 정규화를 적용합니다.
        if do_Dmain or do_Dr1:
        # if False:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                # real_logits_ = self.run_D_hdr(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                # loss_Dreal_ = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    # loss_Dreal_ = torch.nn.functional.softplus(-real_logits_) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        # r1_grads_ = torch.autograd.grad(outputs=[real_logits_.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    # r1_penalty_ = r1_grads_.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    # loss_Dr1_ = r1_penalty_ * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # Dmain: 실제 이미지에 대한 로짓을 최대화합니다. (HDR)
        # Dr1: R1 정규화를 적용합니다. (HDR)
        if do_Dmain_ or do_Dr1_:
            name = 'Dreal_Dr1_' if do_Dmain_ and do_Dr1_ else 'Dreal_' if do_Dmain_ else 'Dr1_'
            with torch.autograd.profiler.record_function(name + '_forward_'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1_)
                # real_logits = self.run_D(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                real_logits_ = self.run_D_hdr(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                training_stats.report('Loss/scores/real_', real_logits_)
                training_stats.report('Loss/signs/real_', real_logits_.sign())

                # loss_Dreal = 0
                loss_Dreal_ = 0
                if do_Dmain_:
                    # loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    loss_Dreal_ = torch.nn.functional.softplus(-real_logits_) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_', loss_Dgen_ + loss_Dreal_)

                # loss_Dr1 = 0
                loss_Dr1_ = 0
                if do_Dr1_:
                    with torch.autograd.profiler.record_function('r1_grads_'), conv2d_gradfix.no_weight_gradients():
                        # r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        r1_grads_ = torch.autograd.grad(outputs=[real_logits_.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    # r1_penalty = r1_grads.square().sum([1,2,3])
                    r1_penalty_ = r1_grads_.square().sum([1,2,3])
                    # loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    loss_Dr1_ = r1_penalty_ * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty_', r1_penalty_)
                    training_stats.report('Loss/D/reg_', loss_Dr1_)

            with torch.autograd.profiler.record_function(name + '_backward_'):
                (real_logits_ * 0 + loss_Dreal_ + loss_Dr1_).mean().mul(gain).backward()



