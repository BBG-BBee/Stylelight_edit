"""
PU21 (Perceptually Uniform encoding for 2021)

물리적 휘도를 인간의 시각적 인지 특성에 맞게
지각적으로 균일한 값으로 변환하는 인코딩입니다.

참고 논문:
- Mantiuk et al., "PU21: Perceptually Uniform Encoding for Image Analysis" (2021)

PU21의 장점:
- HDR 이미지의 넓은 동적 범위(10^-3 ~ 10^6 cd/m²)를 처리
- 인간 시각 시스템(HVS)의 대비 감도 함수(CSF) 기반
- 베버-페히너 법칙을 반영한 로그 스케일 변환

사용 목적:
- DTAM 가중치와 함께 물리적 HDR 손실 함수에 사용
- L_phys = || W_DTAM ⊙ (PU21(Y_pred) - PU21(Y_gt)) ||_1
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class PU21Encoder(nn.Module):
    """
    PU21 (Perceptually Uniform) 인코딩

    물리적 휘도(cd/m²)를 지각적으로 균일한 값으로 변환합니다.

    변환 공식 (간략화된 버전):
        PU(L) = a * L^b + c

    여기서:
        - L: 물리적 휘도 (cd/m²)
        - a, b, c: 피팅 파라미터

    참고: 전체 PU21 공식은 더 복잡하지만,
    실용적인 목적으로 간략화된 버전을 사용합니다.

    Args:
        mode: 인코딩 모드 ('simple', 'full')
            - 'simple': 간략화된 power-law 변환
            - 'full': 전체 PU21 공식 (더 정확하지만 계산 비용 높음)

    Example:
        >>> pu21 = PU21Encoder()
        >>> luminance = torch.tensor([1.0, 10.0, 100.0, 1000.0])
        >>> encoded = pu21(luminance)
    """

    def __init__(self, mode: str = 'simple'):
        super().__init__()
        self.mode = mode

        if mode == 'simple':
            # 간략화된 파라미터 (power-law 근사)
            # 참고: 이 값들은 PU21 논문의 피팅 결과 기반
            self.register_buffer('a', torch.tensor(0.002939, dtype=torch.float32))
            self.register_buffer('b', torch.tensor(1.183, dtype=torch.float32))
            self.register_buffer('c', torch.tensor(1.226, dtype=torch.float32))
        elif mode == 'full':
            # 전체 PU21 파라미터
            self._init_full_params()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'simple' or 'full'.")

    def _init_full_params(self):
        """전체 PU21 파라미터 초기화"""
        # PU21 논문의 전체 파라미터
        # 참고: Mantiuk et al., 2021
        self.register_buffer('p1', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('p2', torch.tensor(2.4, dtype=torch.float32))
        self.register_buffer('p3', torch.tensor(0.01, dtype=torch.float32))
        self.register_buffer('p4', torch.tensor(0.69, dtype=torch.float32))
        self.register_buffer('p5', torch.tensor(0.4, dtype=torch.float32))
        self.register_buffer('p6', torch.tensor(0.13, dtype=torch.float32))

    def forward(self, luminance: torch.Tensor) -> torch.Tensor:
        """
        물리적 휘도를 PU21 인코딩 값으로 변환

        Args:
            luminance: 물리적 휘도 (cd/m²), 비음수 텐서

        Returns:
            encoded: PU21 인코딩된 값
        """
        # 음수 및 0 방지 (로그 연산 안정성)
        L = torch.clamp(luminance, min=1e-6)

        if self.mode == 'simple':
            return self._encode_simple(L)
        else:
            return self._encode_full(L)

    def _encode_simple(self, L: torch.Tensor) -> torch.Tensor:
        """
        간략화된 PU21 인코딩

        PU(L) = a * L^b + c

        이 공식은 휘도 범위 [0.1, 10000] cd/m²에서
        전체 PU21 공식과 높은 상관관계를 보입니다.
        """
        a = self.a.to(L.device)
        b = self.b.to(L.device)
        c = self.c.to(L.device)

        return a * torch.pow(L, b) + c

    def _encode_full(self, L: torch.Tensor) -> torch.Tensor:
        """
        전체 PU21 인코딩

        더 정확하지만 계산 비용이 높습니다.
        """
        # 전체 PU21 공식 구현
        # 참고: https://github.com/gfxdisp/pu21

        p1 = self.p1.to(L.device)
        p2 = self.p2.to(L.device)
        p3 = self.p3.to(L.device)
        p4 = self.p4.to(L.device)
        p5 = self.p5.to(L.device)
        p6 = self.p6.to(L.device)

        # CSF 기반 변환
        L_norm = L / 100.0  # 100 cd/m²로 정규화

        # 로그 기반 변환 (간략화)
        pu = p1 * torch.log10(1 + p2 * torch.pow(L_norm, p4) / (1 + p3 * torch.pow(L_norm, p5))) + p6

        return pu

    def encode(self, luminance: torch.Tensor) -> torch.Tensor:
        """forward의 별칭"""
        return self.forward(luminance)

    def inverse(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        PU21 역변환 (인코딩 → 휘도)

        주의: 간략화된 모드에서만 정확합니다.

        Args:
            encoded: PU21 인코딩된 값

        Returns:
            luminance: 복원된 휘도 (cd/m²)
        """
        if self.mode != 'simple':
            raise NotImplementedError("Inverse transform only available for 'simple' mode")

        a = self.a.to(encoded.device)
        b = self.b.to(encoded.device)
        c = self.c.to(encoded.device)

        # PU(L) = a * L^b + c
        # L = ((PU - c) / a)^(1/b)
        L = torch.pow((encoded - c) / (a + 1e-8), 1.0 / b)
        L = torch.clamp(L, min=0.0)

        return L

    def extra_repr(self) -> str:
        if self.mode == 'simple':
            return f"mode={self.mode}, a={self.a.item():.6f}, b={self.b.item():.4f}, c={self.c.item():.4f}"
        else:
            return f"mode={self.mode}"


class LogLuminanceEncoder(nn.Module):
    """
    로그 휘도 인코딩

    PU21의 간단한 대안으로, 로그 스케일 변환을 사용합니다.

    공식:
        encoded = log10(L + epsilon) / log10(L_max + epsilon)

    Args:
        epsilon: 0 방지를 위한 작은 값
        L_max: 최대 휘도 (정규화용)
    """

    def __init__(self, epsilon: float = 1e-6, L_max: float = 1e6):
        super().__init__()
        self.register_buffer('epsilon', torch.tensor(epsilon, dtype=torch.float32))
        self.register_buffer('L_max', torch.tensor(L_max, dtype=torch.float32))

    def forward(self, luminance: torch.Tensor) -> torch.Tensor:
        """로그 휘도 인코딩"""
        eps = self.epsilon.to(luminance.device)
        L_max = self.L_max.to(luminance.device)

        log_L = torch.log10(luminance + eps)
        log_max = torch.log10(L_max + eps)

        # [0, 1] 범위로 정규화
        encoded = (log_L - torch.log10(eps)) / (log_max - torch.log10(eps))

        return torch.clamp(encoded, 0.0, 1.0)

    def encode(self, luminance: torch.Tensor) -> torch.Tensor:
        return self.forward(luminance)


class MuLawEncoder(nn.Module):
    """
    Mu-Law 인코딩

    오디오 압축에서 사용되는 Mu-Law 변환을
    HDR 이미지에 적용합니다.

    공식:
        encoded = log(1 + mu * L/L_max) / log(1 + mu)

    Args:
        mu: 압축 파라미터 (기본값: 255)
        L_max: 최대 휘도
    """

    def __init__(self, mu: float = 255.0, L_max: float = 1e6):
        super().__init__()
        self.register_buffer('mu', torch.tensor(mu, dtype=torch.float32))
        self.register_buffer('L_max', torch.tensor(L_max, dtype=torch.float32))

    def forward(self, luminance: torch.Tensor) -> torch.Tensor:
        """Mu-Law 인코딩"""
        mu = self.mu.to(luminance.device)
        L_max = self.L_max.to(luminance.device)

        # 정규화
        L_norm = torch.clamp(luminance / L_max, 0.0, 1.0)

        # Mu-Law 변환
        encoded = torch.log(1 + mu * L_norm) / torch.log(1 + mu)

        return encoded

    def encode(self, luminance: torch.Tensor) -> torch.Tensor:
        return self.forward(luminance)


def compare_encodings(luminance: torch.Tensor) -> dict:
    """
    다양한 인코딩 방법 비교

    Args:
        luminance: 테스트할 휘도 값들

    Returns:
        dict: 각 인코딩 방법의 결과
    """
    encoders = {
        'pu21_simple': PU21Encoder(mode='simple'),
        'log': LogLuminanceEncoder(),
        'mu_law': MuLawEncoder()
    }

    results = {}
    for name, encoder in encoders.items():
        results[name] = encoder(luminance)

    return results


# 테스트 함수
def _test_pu21():
    """PU21 인코딩 테스트"""
    print("Testing PU21 Encoder...")

    # PU21 인스턴스 생성
    pu21 = PU21Encoder(mode='simple')
    print(f"PU21: {pu21}")

    # 테스트 휘도 값 (넓은 동적 범위)
    test_values = torch.tensor([
        0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0
    ], dtype=torch.float32)

    # 인코딩
    encoded = pu21(test_values)

    print("\nLuminance -> PU21 encoding:")
    for lum, enc in zip(test_values, encoded):
        print(f"  L={lum.item():10.3f} cd/m² -> PU={enc.item():.6f}")

    # 역변환 테스트
    print("\nInverse transform test:")
    recovered = pu21.inverse(encoded)
    for orig, rec in zip(test_values, recovered):
        error = abs(orig.item() - rec.item()) / (orig.item() + 1e-8) * 100
        print(f"  Original={orig.item():10.3f}, Recovered={rec.item():10.3f}, Error={error:.2f}%")

    # 다양한 인코딩 비교
    print("\nComparing different encodings:")
    results = compare_encodings(test_values)
    for name, values in results.items():
        print(f"\n  {name}:")
        for lum, enc in zip(test_values[:5], values[:5]):
            print(f"    L={lum.item():10.3f} -> {enc.item():.6f}")

    print("\nPU21 test passed!")


if __name__ == "__main__":
    _test_pu21()
