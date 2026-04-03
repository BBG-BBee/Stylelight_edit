"""
HDR GAN Inversion Coach

NFoV HDR 입력으로부터 풀 파노라마를 생성하는 추론 파이프라인.
초점 마스킹 GAN 역전환 + PTI(Pivotal Tuning Inversion)으로
물리적 휘도(cd/m²)를 보존하면서 최적의 파노라마를 생성합니다.

흐름:
    1. HDR 이미지 로드 → log-domain 다운스케일(256×192) → ERP embed
    2. 초점 마스킹 GAN Inversion: bbox 영역의 PhysicalLoss로 최적 w 탐색
    3. PTI: Generator 가중치 미세 조정 (bbox 영역 정합)
    4. 최종 파노라마 생성 → HDR EXR 저장
"""

import os
import torch
from tqdm import tqdm
from PTI_utils import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from PTI_utils.log_utils import log_images_from_w

from skylibs.envmap import EnvironmentMap, rotation_matrix
import numpy as np
import PIL.Image
from PIL import Image, ImageDraw

from skylibs.hdrio import imread, imsave
from utils.hdr_utils import log_domain_resize

# TTA 관련 import
from training.tta_augment import TTAAugmentor, ExpAug, WBAug, FlipAug, IdentityAug
from training.s2r_adapter import (
    compute_uncertainty_from_outputs,
    apply_adaptive_scales,
    set_adapter_scales,
    get_adapter_scales
)


def _load_hdr_to_erp(image_path, vfov, erp_height=512):
    """
    HDR NFoV 이미지를 ERP에 embed

    Args:
        image_path: HDR 이미지 경로 (.exr, .hdr)
        erp_height: ERP 높이 (너비는 2배)
        vfov: 수직 화각 (도)

    Returns:
        masked_pano: (H, W, 3) FP32 ERP (cd/m² 스케일, 빈 영역은 0)
        bbox: [row_min, row_max, col_min, col_max]
    """
    # 1. HDR 로드 (물리적 휘도 보존)
    image_hdr = imread(image_path).astype(np.float32)
    image_hdr = np.maximum(image_hdr, 0.0)

    # 2. 256×192로 log-domain 다운스케일 (피크 휘도 보존)
    image_resized = log_domain_resize(image_hdr, 192, 256)

    # 3. 빈 ERP 생성 + 중앙 방향
    env = EnvironmentMap(erp_height, 'latlong')
    dcm = rotation_matrix(azimuth=0, elevation=0, roll=0)

    # 4. NFoV를 ERP에 embed (cd/m² 보존)
    masked_pano = env.Fov2MaskedPano(
        image_resized, vfov=vfov, rotation_matrix=dcm,
        ar=4. / 3., resolution=(256, 192),
        projection="perspective", mode="normal"
    )

    # 5. 마스크에서 bbox 추출
    mask = env.Fov2MaskedPano(
        image_resized, vfov=vfov, rotation_matrix=dcm,
        ar=4. / 3., resolution=(256, 192),
        projection="perspective", mode="mask"
    )
    mask_rows = np.any(mask > 0, axis=1)
    mask_cols = np.any(mask > 0, axis=0)
    row_min = np.argmax(mask_rows)
    row_max = len(mask_rows) - np.argmax(mask_rows[::-1])
    col_min = np.argmax(mask_cols)
    col_max = len(mask_cols) - np.argmax(mask_cols[::-1])
    bbox = [row_min, row_max, col_min, col_max]

    return masked_pano, bbox


class MyCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb, vfov):
        super().__init__(data_loader, use_wandb)
        self.vfov = vfov

    def train(self):
        """
        HDR GAN Inversion + PTI 추론

        각 NFoV HDR 이미지에 대해:
        1. HDR 로드 → ERP embed (물리적 휘도 보존)
        2. w_projector로 최적 latent code 탐색 (PhysicalLoss)
        3. PTI로 Generator 미세 조정
        4. Softplus HDR 출력 → EXR 저장
        """
        use_ball_holder = True

        for image_name in tqdm(self.data_loader):
            # ── HDR 입력 전처리 ──
            masked_pano, bbox = _load_hdr_to_erp(image_name, vfov=self.vfov)

            # (H, W, C) → (1, C, H, W) 텐서 변환 (cd/m² 유지)
            image = torch.from_numpy(masked_pano.transpose(2, 0, 1).copy()).to(global_config.device)
            image = image.unsqueeze(0).to(torch.float32)

            self.restart_training()
            name = os.path.splitext(os.path.basename(image_name))[0]

            # ── 초점 마스킹 GAN 역전환 ──
            # bbox 영역에서 PhysicalLoss(DTAM+PU21)로 최적 w 탐색
            w_pivot = self.calc_inversions(image, name, bbox)
            w_pivot = w_pivot.to(global_config.device)
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            real_images_batch = real_images_batch[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]

            # ── PTI: Generator 가중치 미세 조정 ──
            for i in tqdm(range(hyperparameters.max_pti_steps)):
                generated_images = self.forward(w_pivot)
                # Softplus 출력: [0, ∞), 음수만 방지
                gen_hdr = torch.clamp(generated_images, min=0.0)
                gen_crop = gen_hdr[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]

                loss, l2_loss_val, loss_lpips = self.calc_loss(
                    gen_crop, real_images_batch, name,
                    self.G, use_ball_holder, w_pivot
                )

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [name])

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1

            # ── HDR 파노라마 출력 (Softplus 직접 저장) ──
            generated_images = self.forward(w_pivot)
            hdr_output = torch.clamp(generated_images, min=0.0)  # cd/m² 스케일
            img_hdr_np = hdr_output.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
            imsave(f'{paths_config.checkpoints_dir}/{name}_test.exr', img_hdr_np)

    def train_with_tta(self, use_tta=True, adaptive_scale=True, num_augmentations=5):
        """
        TTA(Test-Time Augmentation)를 사용한 GAN Inversion + 동적 스케일 조절

        S2R-HDR 논문의 불확실성 기반 도메인 적응을 구현합니다.

        흐름:
        1. LFOV 이미지에 N개 증강 적용
        2. 각 증강된 LFOV에 대해 GAN Inversion 수행
        3. 각 w_pivot으로 파노라마 생성
        4. 파노라마들의 분산으로 불확실성 U(x) 계산
        5. 스케일 조절: scale1 = 1 - U(x), scale2 = 1 + U(x)
        6. 최종 파노라마 생성

        Args:
            use_tta: TTA 사용 여부 (False면 기본 train()과 동일)
            adaptive_scale: 불확실성 기반 스케일 조절 여부
            num_augmentations: TTA 증강 수 (기본값: 5)
        """
        if not use_tta:
            return self.train()

        use_ball_holder = True

        # TTA 증강기 초기화
        tta_augmentor = TTAAugmentor(augmentations=[
            IdentityAug(),
            ExpAug(param=0.3),
            ExpAug(param=-0.3),
            WBAug(gains=[1.05, 1.0, 0.95]),
            FlipAug(horizontal=True),
        ][:num_augmentations])

        print(f"[TTA] Augmentor initialized with {tta_augmentor.num_augmentations} augmentations")
        print(f"[TTA] Adaptive scale: {adaptive_scale}")

        for image_name in tqdm(self.data_loader):
            # ── HDR 입력 전처리 ──
            masked_pano, bbox = _load_hdr_to_erp(image_name, vfov=self.vfov)

            image = torch.from_numpy(masked_pano.transpose(2, 0, 1).copy()).to(global_config.device)
            image = image.unsqueeze(0).to(torch.float32)

            self.restart_training()
            name = os.path.splitext(os.path.basename(image_name))[0]

            # ── TTA: 여러 증강된 LFOV로 불확실성 계산 ──
            print(f"[TTA] Computing uncertainty for {name}...")

            augmented_inputs = tta_augmentor.generate_augmented_inputs(image)
            panoramas = []
            augmentations = []

            for aug_image, aug_module in augmented_inputs:
                w_pivot = self.calc_inversions(aug_image, name, bbox)
                w_pivot = w_pivot.to(global_config.device)

                with torch.no_grad():
                    panorama = self.forward(w_pivot)
                    panoramas.append(panorama)
                    augmentations.append(aug_module)

            aligned_panoramas = tta_augmentor.apply_inverse_transforms(panoramas, augmentations)

            uncertainty, variance_map = compute_uncertainty_from_outputs(
                aligned_panoramas,
                uncertainty_scale=0.05
            )

            print(f"[TTA] Uncertainty: {uncertainty.item():.6f}")

            if adaptive_scale:
                scale1, scale2 = apply_adaptive_scales(self.G, uncertainty)
                print(f"[TTA] Applied scales: scale1={scale1:.4f}, scale2={scale2:.4f}")
            else:
                scale1, scale2 = 1.0, 1.0

            # ── 최종 GAN Inversion + PTI ──
            w_pivot = self.calc_inversions(image, name, bbox)
            w_pivot = w_pivot.to(global_config.device)
            real_images_batch = image.to(global_config.device)
            real_images_batch = real_images_batch[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]

            for i in tqdm(range(hyperparameters.max_pti_steps), desc="PTI"):
                generated_images = self.forward(w_pivot)
                gen_hdr = torch.clamp(generated_images, min=0.0)
                gen_crop = gen_hdr[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]

                loss, l2_loss_val, loss_lpips = self.calc_loss(
                    gen_crop, real_images_batch, name,
                    self.G, use_ball_holder, w_pivot
                )

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0
                global_config.training_step += 1

            self.image_counter += 1

            # ── HDR 출력 저장 ──
            generated_images = self.forward(w_pivot)
            hdr_output = torch.clamp(generated_images, min=0.0)
            img_hdr_np = hdr_output.permute(0, 2, 3, 1)[0].detach().cpu().numpy()

            output_name = f'{name}_tta_s1{scale1:.2f}_s2{scale2:.2f}_u{uncertainty.item():.4f}'
            imsave(f'{paths_config.checkpoints_dir}/{output_name}.exr', img_hdr_np)
            print(f"[TTA] Saved: {output_name}.exr")
