"""
Test-Time Augmentation (TTA) 모듈

S2R-HDR 논문의 TTA 기법을 Stylelight에 적용합니다.

불확실성 기반 도메인 적응:
1. LFOV 입력에 여러 증강 적용
2. 각 증강된 입력으로 파노라마 생성
3. 출력들의 분산으로 불확실성(U(x)) 계산
4. 불확실성에 따라 스케일 동적 조절:
   - scale1 = 1 - U(x) (공유 브랜치)
   - scale2 = 1 + U(x) (전송 브랜치)

사용 시나리오:
- 기본: Laval 데이터로 파인튜닝 → TTA 불필요
- 선택: 새 도메인 추론 시 TTA 활성화 가능

참조: S2R-HDR/utils/aug.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Union
import random


class ExpAug(nn.Module):
    """
    노출 조정 증강 (Exposure Augmentation)

    HDR 이미지의 노출을 2^param 배율로 조정합니다.
    물리적 휘도 스케일을 유지하면서 밝기를 변화시킵니다.

    S2R-HDR 원본 참조: utils/aug.py ExpAug

    Args:
        param: 노출 파라미터 (결과 배율: 2^param)
               예: param=1.0 → 2배, param=-1.0 → 0.5배

    Example:
        >>> aug = ExpAug(param=0.5)
        >>> augmented = aug.transform(image)
        >>> restored = aug.transform_back(output)
    """

    def __init__(self, param: float = 0.0):
        super().__init__()
        self.param = param
        self.scale = 2.0 ** param

    def transform(self, img: torch.Tensor) -> torch.Tensor:
        """
        입력 이미지에 노출 증강 적용

        Args:
            img: HDR 이미지 (B, C, H, W) 또는 (C, H, W)

        Returns:
            augmented: 노출 조정된 이미지
        """
        return img * self.scale

    def transform_back(self, img: torch.Tensor) -> torch.Tensor:
        """
        출력 파노라마에 역변환 적용

        Args:
            img: 생성된 파노라마 (B, C, H, W)

        Returns:
            restored: 노출 복원된 파노라마
        """
        return img / self.scale

    def __repr__(self) -> str:
        return f"ExpAug(param={self.param}, scale={self.scale:.4f})"


class WBAug(nn.Module):
    """
    화이트 밸런스 증강 (White Balance Augmentation)

    RGB 채널별 가중치를 적용하여 색온도를 변화시킵니다.

    S2R-HDR 원본 참조: utils/aug.py WBAug

    Args:
        gains: RGB 채널별 게인 [R, G, B]
               예: [1.0, 0.9, 1.1] → R 유지, G 감소, B 증가

    Example:
        >>> aug = WBAug(gains=[1.0, 0.9, 1.1])
        >>> augmented = aug.transform(image)
        >>> restored = aug.transform_back(output)
    """

    def __init__(self, gains: List[float] = None):
        super().__init__()
        if gains is None:
            gains = [1.0, 1.0, 1.0]
        self.gains = torch.tensor(gains, dtype=torch.float32)

    def transform(self, img: torch.Tensor) -> torch.Tensor:
        """
        입력 이미지에 화이트밸런스 증강 적용

        Args:
            img: HDR 이미지 (B, C, H, W) 또는 (C, H, W)

        Returns:
            augmented: WB 조정된 이미지
        """
        gains = self.gains.to(img.device)

        if img.dim() == 3:
            # (C, H, W)
            return img * gains.view(3, 1, 1)
        else:
            # (B, C, H, W)
            return img * gains.view(1, 3, 1, 1)

    def transform_back(self, img: torch.Tensor) -> torch.Tensor:
        """
        출력 파노라마에 역변환 적용

        Args:
            img: 생성된 파노라마 (B, C, H, W)

        Returns:
            restored: WB 복원된 파노라마
        """
        gains = self.gains.to(img.device)

        if img.dim() == 3:
            return img / gains.view(3, 1, 1)
        else:
            return img / gains.view(1, 3, 1, 1)

    def __repr__(self) -> str:
        return f"WBAug(gains={self.gains.tolist()})"


class PermAug(nn.Module):
    """
    채널 순열 증강 (Channel Permutation Augmentation)

    RGB 채널 순서를 변경하여 색상 불변성을 테스트합니다.

    S2R-HDR 원본 참조: utils/aug.py PermAug

    Args:
        perm: 채널 순열 (예: [2, 0, 1] → BGR)
              None이면 랜덤 생성

    Example:
        >>> aug = PermAug(perm=[2, 0, 1])  # RGB → BGR
        >>> augmented = aug.transform(image)
        >>> restored = aug.transform_back(output)
    """

    # 가능한 모든 순열 (6가지)
    ALL_PERMUTATIONS = [
        [0, 1, 2],  # RGB (원본)
        [0, 2, 1],  # RBG
        [1, 0, 2],  # GRB
        [1, 2, 0],  # GBR
        [2, 0, 1],  # BRG
        [2, 1, 0],  # BGR
    ]

    def __init__(self, perm: List[int] = None):
        super().__init__()
        if perm is None:
            perm = random.choice(self.ALL_PERMUTATIONS[1:])  # 원본 제외
        self.perm = perm
        # 역순열 계산
        self.inverse_perm = [perm.index(i) for i in range(3)]

    def transform(self, img: torch.Tensor) -> torch.Tensor:
        """
        입력 이미지에 채널 순열 적용

        Args:
            img: HDR 이미지 (B, C, H, W) 또는 (C, H, W)

        Returns:
            augmented: 채널 순열된 이미지
        """
        if img.dim() == 3:
            return img[self.perm, :, :]
        else:
            return img[:, self.perm, :, :]

    def transform_back(self, img: torch.Tensor) -> torch.Tensor:
        """
        출력 파노라마에 역순열 적용

        Args:
            img: 생성된 파노라마 (B, C, H, W)

        Returns:
            restored: 채널 복원된 파노라마
        """
        if img.dim() == 3:
            return img[self.inverse_perm, :, :]
        else:
            return img[:, self.inverse_perm, :, :]

    def __repr__(self) -> str:
        return f"PermAug(perm={self.perm})"


class FlipAug(nn.Module):
    """
    뒤집기 증강 (Flip Augmentation)

    이미지를 수평으로 뒤집습니다.
    파노라마의 경우 수직 뒤집기는 물리적으로 무의미하므로 수평만 지원.

    S2R-HDR 원본 참조: utils/aug.py FlipAug

    Args:
        horizontal: 수평 뒤집기 여부 (기본값: True)

    Example:
        >>> aug = FlipAug(horizontal=True)
        >>> augmented = aug.transform(image)
        >>> restored = aug.transform_back(output)
    """

    def __init__(self, horizontal: bool = True):
        super().__init__()
        self.horizontal = horizontal

    def transform(self, img: torch.Tensor) -> torch.Tensor:
        """
        입력 이미지에 뒤집기 적용

        Args:
            img: HDR 이미지 (B, C, H, W) 또는 (C, H, W)

        Returns:
            augmented: 뒤집힌 이미지
        """
        if self.horizontal:
            return torch.flip(img, dims=[-1])  # 수평 (width 축)
        return img

    def transform_back(self, img: torch.Tensor) -> torch.Tensor:
        """
        출력 파노라마에 역뒤집기 적용 (대칭 연산)

        Args:
            img: 생성된 파노라마 (B, C, H, W)

        Returns:
            restored: 뒤집기 복원된 파노라마
        """
        if self.horizontal:
            return torch.flip(img, dims=[-1])
        return img

    def __repr__(self) -> str:
        return f"FlipAug(horizontal={self.horizontal})"


class IdentityAug(nn.Module):
    """
    항등 증강 (Identity Augmentation)

    원본 이미지를 그대로 반환합니다.
    TTA에서 원본을 포함할 때 사용합니다.
    """

    def __init__(self):
        super().__init__()

    def transform(self, img: torch.Tensor) -> torch.Tensor:
        return img

    def transform_back(self, img: torch.Tensor) -> torch.Tensor:
        return img

    def __repr__(self) -> str:
        return "IdentityAug()"


class TTAAugmentor(nn.Module):
    """
    Test-Time Augmentation 통합 관리자

    여러 증강을 조합하여 TTA 파이프라인을 구성합니다.
    불확실성 계산을 위한 다양한 변환 버전을 생성합니다.

    Args:
        augmentations: 사용할 증강 리스트 (None이면 기본 세트)
        include_identity: 원본 포함 여부 (기본값: True)

    Example:
        >>> tta = TTAAugmentor()
        >>> augmented_inputs = tta.generate_augmented_inputs(lfov_image)
        >>> # 각각 GAN Inversion → 파노라마 생성
        >>> panoramas = [generate(aug_in) for aug_in in augmented_inputs]
        >>> # 역변환 적용하여 정렬
        >>> aligned = tta.apply_inverse_transforms(panoramas)
        >>> # 분산 계산
        >>> uncertainty = torch.var(torch.stack(aligned), dim=0).mean()
    """

    @staticmethod
    def get_default_augmentations() -> List[nn.Module]:
        """
        기본 증강 세트 반환

        S2R-HDR에서 사용하는 증강들:
        - 원본 (IdentityAug)
        - 노출 변화 (ExpAug): +0.5, -0.5
        - 화이트밸런스 (WBAug): 따뜻한/차가운 톤
        - 수평 뒤집기 (FlipAug)
        """
        return [
            IdentityAug(),                          # 원본
            ExpAug(param=0.5),                      # 밝게 (1.41배)
            ExpAug(param=-0.5),                     # 어둡게 (0.71배)
            WBAug(gains=[1.1, 1.0, 0.9]),          # 따뜻한 톤
            WBAug(gains=[0.9, 1.0, 1.1]),          # 차가운 톤
            FlipAug(horizontal=True),              # 수평 뒤집기
        ]

    def __init__(self,
                 augmentations: List[nn.Module] = None,
                 include_identity: bool = True):
        super().__init__()

        if augmentations is None:
            augmentations = self.get_default_augmentations()

        # Identity가 없고 include_identity=True면 추가
        has_identity = any(isinstance(aug, IdentityAug) for aug in augmentations)
        if include_identity and not has_identity:
            augmentations = [IdentityAug()] + augmentations

        self.augmentations = nn.ModuleList(augmentations)

    @property
    def num_augmentations(self) -> int:
        """증강 수 반환"""
        return len(self.augmentations)

    def generate_augmented_inputs(self, image: torch.Tensor) -> List[Tuple[torch.Tensor, nn.Module]]:
        """
        입력 이미지의 여러 증강 버전 생성

        Args:
            image: LFOV 입력 이미지 (B, C, H, W) 또는 (C, H, W)

        Returns:
            augmented_list: [(증강된_이미지, 증강_모듈), ...] 리스트
        """
        results = []
        for aug in self.augmentations:
            aug_image = aug.transform(image)
            results.append((aug_image, aug))
        return results

    def apply_inverse_transforms(self,
                                  outputs: List[torch.Tensor],
                                  augmentations: List[nn.Module]) -> List[torch.Tensor]:
        """
        생성된 파노라마들에 역변환 적용하여 정렬

        Args:
            outputs: 각 증강 입력으로 생성된 파노라마 리스트
            augmentations: 적용된 증강 모듈 리스트 (generate_augmented_inputs에서 반환)

        Returns:
            aligned: 역변환이 적용되어 정렬된 파노라마 리스트
        """
        aligned = []
        for output, aug in zip(outputs, augmentations):
            aligned_output = aug.transform_back(output)
            aligned.append(aligned_output)
        return aligned

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        입력 이미지의 증강된 버전들 생성 (증강만, 역변환 없음)

        Args:
            image: LFOV 입력 이미지

        Returns:
            augmented_images: 증강된 이미지 리스트
        """
        return [aug.transform(image) for aug in self.augmentations]

    def __repr__(self) -> str:
        aug_names = [aug.__class__.__name__ for aug in self.augmentations]
        return f"TTAAugmentor(augmentations={aug_names})"


# ============================================================================
# 테스트 함수
# ============================================================================

def _test_tta_augment():
    """TTA 증강 모듈 단위 테스트"""
    print("=" * 60)
    print("Testing TTA Augmentation module...")
    print("=" * 60)

    # 테스트 이미지 생성
    test_image = torch.rand(1, 3, 256, 512)  # (B, C, H, W)

    # 1. ExpAug 테스트
    print("\n[Test 1] ExpAug...")
    exp_aug = ExpAug(param=1.0)
    aug_img = exp_aug.transform(test_image)
    restored = exp_aug.transform_back(aug_img)
    diff = (test_image - restored).abs().max()
    print(f"  Scale: {exp_aug.scale}")
    print(f"  Restoration diff: {diff:.2e}")
    assert diff < 1e-5, "ExpAug restoration failed"
    print("  PASS")

    # 2. WBAug 테스트
    print("\n[Test 2] WBAug...")
    wb_aug = WBAug(gains=[1.1, 1.0, 0.9])
    aug_img = wb_aug.transform(test_image)
    restored = wb_aug.transform_back(aug_img)
    diff = (test_image - restored).abs().max()
    print(f"  Gains: {wb_aug.gains.tolist()}")
    print(f"  Restoration diff: {diff:.2e}")
    assert diff < 1e-5, "WBAug restoration failed"
    print("  PASS")

    # 3. PermAug 테스트
    print("\n[Test 3] PermAug...")
    perm_aug = PermAug(perm=[2, 0, 1])  # RGB → BRG
    aug_img = perm_aug.transform(test_image)
    restored = perm_aug.transform_back(aug_img)
    diff = (test_image - restored).abs().max()
    print(f"  Perm: {perm_aug.perm}")
    print(f"  Inverse: {perm_aug.inverse_perm}")
    print(f"  Restoration diff: {diff:.2e}")
    assert diff < 1e-5, "PermAug restoration failed"
    print("  PASS")

    # 4. FlipAug 테스트
    print("\n[Test 4] FlipAug...")
    flip_aug = FlipAug(horizontal=True)
    aug_img = flip_aug.transform(test_image)
    restored = flip_aug.transform_back(aug_img)
    diff = (test_image - restored).abs().max()
    print(f"  Horizontal: {flip_aug.horizontal}")
    print(f"  Restoration diff: {diff:.2e}")
    assert diff < 1e-5, "FlipAug restoration failed"
    print("  PASS")

    # 5. TTAAugmentor 테스트
    print("\n[Test 5] TTAAugmentor...")
    tta = TTAAugmentor()
    print(f"  Augmentations: {tta.num_augmentations}")
    print(f"  {tta}")

    augmented_list = tta.generate_augmented_inputs(test_image)
    print(f"  Generated {len(augmented_list)} augmented inputs")

    # 역변환 테스트
    outputs = [aug_img for aug_img, _ in augmented_list]  # 실제로는 Generator 출력
    augmentations = [aug for _, aug in augmented_list]
    aligned = tta.apply_inverse_transforms(outputs, augmentations)
    print(f"  Applied inverse transforms to {len(aligned)} outputs")
    print("  PASS")

    # 6. 분산 계산 시뮬레이션
    print("\n[Test 6] Uncertainty calculation simulation...")
    # 실제로는 각기 다른 Generator 출력을 사용
    stacked = torch.stack(aligned, dim=0)  # (N, B, C, H, W)
    variance = torch.var(stacked, dim=0)  # (B, C, H, W)
    uncertainty = torch.mean(variance) * 0.1
    print(f"  Stacked shape: {stacked.shape}")
    print(f"  Variance shape: {variance.shape}")
    print(f"  Uncertainty (scaled): {uncertainty:.6f}")
    print("  PASS")

    print("\n" + "=" * 60)
    print("All TTA Augmentation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _test_tta_augment()
