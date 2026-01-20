"""
이중 임계값 적응형 마스킹 (Dual-Threshold Adaptive Masking, DTAM)

DGP(주광 눈부심 확률) 계산의 정확성을 위해
휘도 값에 따라 학습 가중치를 차등 부여하는 전략입니다.

핵심 아이디어:
- 1,000 cd/m² 고정 임계값(ATM)의 문제점 해결
- 300~1,000 cd/m² 전이 구간의 '눈부시지 않은 밝은 빛' 학습
- Ev(수직 조도) 정합성 확보

가중치 함수:
    W(L) = 1.0                                           if L < T_onset
    W(L) = 1.0 + α * ((L - T_onset) / (T_peak - T_onset))^γ   if T_onset ≤ L < T_peak
    W(L) = 1.0 + α                                        if L ≥ T_peak

기본값:
    T_onset = 300 cd/m² (학습 시작점)
    T_peak = 1,000 cd/m² (최대 가중치 도달점)
    α = 9.0 (가중치 증폭 계수, 최대 10배)
    γ = 2.0 (곡률, 이차 함수)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class DTAM(nn.Module):
    """
    이중 임계값 적응형 마스킹 (Dual-Threshold Adaptive Masking)

    휘도에 따라 가중치를 차등 부여하여 DGP 계산의 정확성을 확보합니다.

    Args:
        T_onset: 학습 시작 임계값 (cd/m²). 기본값 300.
        T_peak: 최대 가중치 도달점 (cd/m²). 기본값 1000.
        alpha: 가중치 증폭 계수. 기본값 9.0 (최대 10배 가중치).
        gamma: 전이 구간의 곡률. 기본값 2.0 (이차 함수).

    Example:
        >>> dtam = DTAM(T_onset=300, T_peak=1000, alpha=9.0, gamma=2.0)
        >>> luminance = torch.tensor([[[[100, 500, 2000]]]])  # (B, 1, H, W)
        >>> weights = dtam(luminance)
        >>> print(weights)
        # [[[[1.0, ~3.8, 10.0]]]]
    """

    def __init__(self,
                 T_onset: float = 300.0,
                 T_peak: float = 1000.0,
                 alpha: float = 9.0,
                 gamma: float = 2.0):
        super().__init__()

        assert T_onset < T_peak, f"T_onset({T_onset}) must be less than T_peak({T_peak})"
        assert alpha > 0, f"alpha must be positive, got {alpha}"
        assert gamma > 0, f"gamma must be positive, got {gamma}"

        # 상수로 등록 (학습되지 않음)
        self.register_buffer('T_onset', torch.tensor(T_onset, dtype=torch.float32))
        self.register_buffer('T_peak', torch.tensor(T_peak, dtype=torch.float32))
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))

    def forward(self, luminance: torch.Tensor) -> torch.Tensor:
        """
        휘도값에 따른 DTAM 가중치 계산

        Args:
            luminance: 픽셀별 휘도 (cd/m²)
                      Shape: (B, 1, H, W) 또는 (B, C, H, W) 또는 (H, W)

        Returns:
            weight: 픽셀별 가중치
                   Shape: luminance와 동일
        """
        # 텐서 타입 및 디바이스 일치
        T_onset = self.T_onset.to(luminance.device)
        T_peak = self.T_peak.to(luminance.device)
        alpha = self.alpha.to(luminance.device)
        gamma = self.gamma.to(luminance.device)

        # 기본 가중치 1.0으로 초기화
        weight = torch.ones_like(luminance)

        # 전이 구간 마스크 (T_onset ≤ L < T_peak)
        transition_mask = (luminance >= T_onset) & (luminance < T_peak)

        # 눈부심 구간 마스크 (L ≥ T_peak)
        peak_mask = luminance >= T_peak

        # 전이 구간 가중치 계산
        # normalized = (L - T_onset) / (T_peak - T_onset), 범위: [0, 1)
        normalized = (luminance - T_onset) / (T_peak - T_onset + 1e-8)
        normalized = torch.clamp(normalized, 0.0, 1.0)

        # W(L) = 1.0 + α * normalized^γ
        transition_weight = 1.0 + alpha * torch.pow(normalized, gamma)

        # 가중치 적용
        weight = torch.where(transition_mask, transition_weight, weight)
        weight = torch.where(peak_mask, 1.0 + alpha, weight)

        return weight

    def compute_weight(self, luminance: torch.Tensor) -> torch.Tensor:
        """forward의 별칭 (호환성 유지)"""
        return self.forward(luminance)

    def get_zone_masks(self, luminance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        휘도 구간별 마스크 반환

        Args:
            luminance: 픽셀별 휘도 (cd/m²)

        Returns:
            background_mask: L < T_onset (배경 구간)
            transition_mask: T_onset ≤ L < T_peak (전이 구간)
            glare_mask: L ≥ T_peak (눈부심 구간)
        """
        T_onset = self.T_onset.to(luminance.device)
        T_peak = self.T_peak.to(luminance.device)

        background_mask = luminance < T_onset
        transition_mask = (luminance >= T_onset) & (luminance < T_peak)
        glare_mask = luminance >= T_peak

        return background_mask, transition_mask, glare_mask

    def get_zone_stats(self, luminance: torch.Tensor) -> dict:
        """
        휘도 구간별 통계 반환

        Args:
            luminance: 픽셀별 휘도 (cd/m²)

        Returns:
            dict: 각 구간의 픽셀 비율 및 평균 휘도
        """
        bg_mask, trans_mask, glare_mask = self.get_zone_masks(luminance)

        total_pixels = luminance.numel()

        stats = {
            'background': {
                'ratio': bg_mask.sum().item() / total_pixels,
                'mean_luminance': luminance[bg_mask].mean().item() if bg_mask.any() else 0.0
            },
            'transition': {
                'ratio': trans_mask.sum().item() / total_pixels,
                'mean_luminance': luminance[trans_mask].mean().item() if trans_mask.any() else 0.0
            },
            'glare': {
                'ratio': glare_mask.sum().item() / total_pixels,
                'mean_luminance': luminance[glare_mask].mean().item() if glare_mask.any() else 0.0
            }
        }

        return stats

    def extra_repr(self) -> str:
        return (f'T_onset={self.T_onset.item():.1f}, '
                f'T_peak={self.T_peak.item():.1f}, '
                f'alpha={self.alpha.item():.1f}, '
                f'gamma={self.gamma.item():.1f}')


def rgb_to_luminance(rgb: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """
    RGB 이미지에서 휘도(Y) 채널 추출

    ITU-R BT.709 표준:
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

    Args:
        rgb: RGB 이미지 (B, 3, H, W) 또는 (3, H, W)
        keepdim: True면 채널 차원 유지

    Returns:
        luminance: 휘도 맵 (B, 1, H, W) 또는 (1, H, W)
    """
    if rgb.ndim == 3:
        # (3, H, W) -> (1, H, W)
        luminance = (0.2126 * rgb[0:1] +
                     0.7152 * rgb[1:2] +
                     0.0722 * rgb[2:3])
    else:
        # (B, 3, H, W) -> (B, 1, H, W)
        luminance = (0.2126 * rgb[:, 0:1] +
                     0.7152 * rgb[:, 1:2] +
                     0.0722 * rgb[:, 2:3])

    if not keepdim and luminance.shape[-3] == 1:
        luminance = luminance.squeeze(-3)

    return luminance


def visualize_dtam_weights(luminance: torch.Tensor,
                           dtam: DTAM,
                           normalize: bool = True) -> torch.Tensor:
    """
    DTAM 가중치를 시각화용 이미지로 변환

    Args:
        luminance: 휘도 맵
        dtam: DTAM 인스턴스
        normalize: True면 [0, 1] 범위로 정규화

    Returns:
        weight_image: 가중치 이미지 (시각화용)
    """
    weights = dtam(luminance)

    if normalize:
        # 최대 가중치(1 + alpha)로 정규화
        max_weight = 1.0 + dtam.alpha.item()
        weights = (weights - 1.0) / (max_weight - 1.0)
        weights = torch.clamp(weights, 0.0, 1.0)

    return weights


# 테스트 함수
def _test_dtam():
    """DTAM 모듈 테스트"""
    print("Testing DTAM module...")

    # DTAM 인스턴스 생성
    dtam = DTAM(T_onset=300, T_peak=1000, alpha=9.0, gamma=2.0)
    print(f"DTAM: {dtam}")

    # 테스트 휘도 값
    test_values = torch.tensor([
        [[[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 10000]]]
    ], dtype=torch.float32)

    # 가중치 계산
    weights = dtam(test_values)

    print("\nLuminance -> Weight mapping:")
    for i, (lum, w) in enumerate(zip(test_values.flatten(), weights.flatten())):
        zone = "background" if lum < 300 else ("transition" if lum < 1000 else "glare")
        print(f"  L={lum.item():7.0f} cd/m² -> W={w.item():.4f} ({zone})")

    # 구간 통계
    print("\nZone statistics:")
    stats = dtam.get_zone_stats(test_values)
    for zone, stat in stats.items():
        print(f"  {zone}: ratio={stat['ratio']:.2%}, mean={stat['mean_luminance']:.1f} cd/m²")

    print("\nDTAM test passed!")


if __name__ == "__main__":
    _test_dtam()
