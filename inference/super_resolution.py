"""
SwinIR 초해상도 모듈

512x512 HDR 이미지를 1024x1024로 업스케일링합니다.
HDR 선형 휘도를 보존하면서 초해상도를 수행합니다.

SwinIR 논문: "SwinIR: Image Restoration Using Swin Transformer"
https://arxiv.org/abs/2108.10257

사용법:
    from inference.super_resolution import SwinIRUpscaler

    upscaler = SwinIRUpscaler(scale=2, model_path='swinir_x2.pth')
    hr_image = upscaler(lr_image)  # (B, 3, 512, 512) -> (B, 3, 1024, 1024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SwinIRUpscaler(nn.Module):
    """
    SwinIR 기반 HDR 초해상도 모듈

    HDR 이미지의 물리적 휘도를 보존하면서 해상도를 증가시킵니다.

    Args:
        scale: 업스케일 배율 (기본값: 2)
        model_path: 사전 학습된 SwinIR 모델 경로
        device: 디바이스
        hdr_mode: HDR 모드 활성화 (휘도 보존)
    """

    def __init__(self,
                 scale: int = 2,
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 hdr_mode: bool = True):
        super().__init__()

        self.scale = scale
        self.device = device
        self.hdr_mode = hdr_mode

        # SwinIR 모델 로드 시도
        self.swinir = None
        if model_path is not None:
            self.swinir = self._load_swinir(model_path)

        # SwinIR가 없으면 Lanczos 폴백
        if self.swinir is None:
            print("Warning: SwinIR not available. Using Lanczos interpolation as fallback.")

    def _load_swinir(self, model_path: str) -> Optional[nn.Module]:
        """SwinIR 모델 로드"""
        try:
            # SwinIR 아키텍처 임포트 시도
            from models.swinir import SwinIR

            model = SwinIR(
                upscale=self.scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffle',
                resi_connection='1conv'
            )

            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'params' in checkpoint:
                model.load_state_dict(checkpoint['params'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(self.device)
            model.eval()

            # 추론 모드로 동결
            for param in model.parameters():
                param.requires_grad = False

            print(f"SwinIR 모델 로드 완료: {model_path}")
            return model

        except ImportError:
            print("Warning: SwinIR module not found.")
            return None
        except Exception as e:
            print(f"Warning: Failed to load SwinIR model: {e}")
            return None

    def _normalize_hdr(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HDR 이미지를 [0, 1] 범위로 정규화

        Args:
            image: HDR 이미지 (B, 3, H, W), 물리적 휘도 스케일

        Returns:
            normalized: 정규화된 이미지
            scale_factor: 복원을 위한 스케일 팩터
        """
        # 채널별 최대값 계산 (배치 내 최대)
        B = image.shape[0]
        scale_factor = image.view(B, 3, -1).max(dim=2)[0]  # (B, 3)
        scale_factor = scale_factor.view(B, 3, 1, 1).clamp(min=1e-6)

        normalized = image / scale_factor
        return normalized, scale_factor

    def _denormalize_hdr(self,
                         image: torch.Tensor,
                         scale_factor: torch.Tensor) -> torch.Tensor:
        """
        정규화된 이미지를 원래 HDR 스케일로 복원

        Args:
            image: 정규화된 이미지 (B, 3, H, W)
            scale_factor: 스케일 팩터 (B, 3, 1, 1)

        Returns:
            denormalized: HDR 이미지
        """
        return image * scale_factor

    def _lanczos_upscale(self, image: torch.Tensor) -> torch.Tensor:
        """Lanczos 보간으로 업스케일 (폴백)"""
        B, C, H, W = image.shape
        target_H = H * self.scale
        target_W = W * self.scale

        # PyTorch의 bicubic은 Lanczos와 유사한 결과
        upscaled = F.interpolate(
            image,
            size=(target_H, target_W),
            mode='bicubic',
            align_corners=False,
            antialias=True
        )

        return upscaled

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        HDR 이미지 초해상도

        Args:
            image: 입력 HDR 이미지 (B, 3, H, W), 물리적 휘도 스케일

        Returns:
            upscaled: 업스케일된 HDR 이미지 (B, 3, H*scale, W*scale)
        """
        if self.hdr_mode:
            # HDR 모드: 휘도 정규화 -> SR -> 휘도 복원
            normalized, scale_factor = self._normalize_hdr(image)

            if self.swinir is not None:
                with torch.no_grad():
                    upscaled_norm = self.swinir(normalized)
            else:
                upscaled_norm = self._lanczos_upscale(normalized)

            # 스케일 팩터 업스케일 (같은 값 유지)
            upscaled = self._denormalize_hdr(upscaled_norm, scale_factor)

        else:
            # 일반 모드: 직접 SR
            if self.swinir is not None:
                with torch.no_grad():
                    upscaled = self.swinir(image)
            else:
                upscaled = self._lanczos_upscale(image)

        # 음수 방지 (물리적 휘도)
        upscaled = torch.clamp(upscaled, min=0.0)

        return upscaled


def create_swinir_model(scale: int = 2,
                        model_path: Optional[str] = None,
                        device: str = 'cuda') -> SwinIRUpscaler:
    """
    SwinIR 업스케일러 생성 헬퍼 함수

    Args:
        scale: 업스케일 배율
        model_path: 모델 경로 (None이면 Lanczos 폴백)
        device: 디바이스

    Returns:
        upscaler: SwinIRUpscaler 인스턴스
    """
    return SwinIRUpscaler(
        scale=scale,
        model_path=model_path,
        device=device,
        hdr_mode=True
    )


class HDRSuperResolution(nn.Module):
    """
    HDR 전용 초해상도 파이프라인

    물리적 휘도 보존을 위한 추가 처리를 포함합니다:
    1. 로그 도메인 변환
    2. 초해상도
    3. 선형 도메인 복원
    4. 휘도 일관성 검증

    Args:
        upscaler: 기본 업스케일러 (SwinIR 등)
        use_log_domain: 로그 도메인 처리 사용
    """

    def __init__(self,
                 upscaler: Optional[nn.Module] = None,
                 use_log_domain: bool = False):
        super().__init__()

        self.upscaler = upscaler or SwinIRUpscaler()
        self.use_log_domain = use_log_domain
        self.epsilon = 1e-6

    def _to_log_domain(self, image: torch.Tensor) -> torch.Tensor:
        """선형 -> 로그 도메인 변환"""
        return torch.log(image + self.epsilon)

    def _to_linear_domain(self, image: torch.Tensor) -> torch.Tensor:
        """로그 -> 선형 도메인 변환"""
        return torch.exp(image) - self.epsilon

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        HDR 초해상도 수행

        Args:
            image: 입력 HDR 이미지 (B, 3, H, W)

        Returns:
            upscaled: 업스케일된 HDR 이미지
        """
        if self.use_log_domain:
            # 로그 도메인에서 처리
            log_image = self._to_log_domain(image)
            upscaled_log = self.upscaler(log_image)
            upscaled = self._to_linear_domain(upscaled_log)
        else:
            # 선형 도메인에서 직접 처리
            upscaled = self.upscaler(image)

        return upscaled

    def verify_luminance_preservation(self,
                                      lr_image: torch.Tensor,
                                      hr_image: torch.Tensor,
                                      tolerance: float = 0.05) -> dict:
        """
        휘도 보존 검증

        Args:
            lr_image: 저해상도 입력
            hr_image: 고해상도 출력
            tolerance: 허용 오차 비율

        Returns:
            metrics: {'mean_error', 'max_error', 'preserved'}
        """
        # 저해상도로 다운샘플링하여 비교
        B, C, H, W = lr_image.shape
        hr_downsampled = F.interpolate(
            hr_image, size=(H, W), mode='area'
        )

        # 휘도 계산 (ITU-R BT.709)
        lr_lum = 0.2126 * lr_image[:, 0] + 0.7152 * lr_image[:, 1] + 0.0722 * lr_image[:, 2]
        hr_lum = 0.2126 * hr_downsampled[:, 0] + 0.7152 * hr_downsampled[:, 1] + 0.0722 * hr_downsampled[:, 2]

        # 오차 계산
        abs_error = torch.abs(lr_lum - hr_lum)
        rel_error = abs_error / (lr_lum + self.epsilon)

        mean_error = rel_error.mean().item()
        max_error = rel_error.max().item()
        preserved = mean_error < tolerance

        return {
            'mean_error': mean_error,
            'max_error': max_error,
            'preserved': preserved
        }
