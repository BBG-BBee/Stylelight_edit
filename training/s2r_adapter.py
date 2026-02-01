"""
S2R-Adapter (Sim-to-Real Adapter)

S2R-HDR 논문의 어댑터 구조를 StyleGAN2 기반 Stylelight에 적용합니다.

핵심 구조:
- 공유 브랜치 (Shared Branch): 원본 선형 변환 (Stage 1 지식 보존)
- 전송 브랜치 1 (Transfer Branch 1): 작은 rank r1 (미세 조정)
- 전송 브랜치 2 (Transfer Branch 2): 큰 rank r2 (광범위 적응)

출력 공식:
    h = linear_shared(x) + adapter_up1(adapter_down1(x)) * scale1
                         + adapter_up2(adapter_down2(x)) * scale2

적응 전략:
- Stage 1 (S2R-HDR): 구조적 지식 학습
- Stage 2 (Laval): scale1, scale2로 물리적 보정 조절

추론 시:
- 도메인 유사도에 따라 scale1, scale2 조절
- 합성 데이터와 유사: scale 낮춤 (구조 중심)
- 실측 데이터와 유사: scale 높임 (물리 보정 중심)

참조: S2R-HDR/models/SCTNet.py - InjectedLinear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
import copy


class S2RAdapterLinear(nn.Module):
    """
    S2R-Adapter for nn.Linear layers.

    3-브랜치 구조:
    - shared: 원본 Linear (동결 가능)
    - transfer1: 작은 rank (r1) - 미세 조정
    - transfer2: 큰 rank (r2) - 광범위 적응

    Args:
        original_layer: 원본 nn.Linear 레이어
        r1: 전송 브랜치 1의 rank (기본값: 1)
        r2: 전송 브랜치 2의 rank (기본값: 128)
        freeze_shared: 공유 브랜치 동결 여부 (기본값: True)
        init_scale1: scale1 초기값 (기본값: 1.0)
        init_scale2: scale2 초기값 (기본값: 1.0)
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r1: int = 1,
        r2: int = 128,
        freeze_shared: bool = True,
        init_scale1: float = 1.0,
        init_scale2: float = 1.0,
    ):
        super().__init__()

        if not isinstance(original_layer, nn.Linear):
            raise TypeError(f"S2RAdapterLinear requires nn.Linear, got {type(original_layer)}")

        in_features = original_layer.in_features
        out_features = original_layer.out_features
        has_bias = original_layer.bias is not None

        self.in_features = in_features
        self.out_features = out_features
        self.r1 = r1
        self.r2 = r2

        # ========== 공유 브랜치 (Shared Branch) ==========
        # 원본 가중치로 초기화
        self.linear_shared = nn.Linear(in_features, out_features, bias=has_bias)
        self.linear_shared.weight.data.copy_(original_layer.weight.data)
        if has_bias:
            self.linear_shared.bias.data.copy_(original_layer.bias.data)

        if freeze_shared:
            for param in self.linear_shared.parameters():
                param.requires_grad = False

        # ========== 전송 브랜치 1 (Transfer Branch 1, 작은 rank) ==========
        self.adapter_down1 = nn.Linear(in_features, r1, bias=False)
        self.adapter_up1 = nn.Linear(r1, out_features, bias=False)

        # S2R-HDR 논문 초기화: down은 normal(std=1/r^2), up은 zeros
        nn.init.normal_(self.adapter_down1.weight, std=1.0 / (r1 ** 2))
        nn.init.zeros_(self.adapter_up1.weight)

        # ========== 전송 브랜치 2 (Transfer Branch 2, 큰 rank) ==========
        self.adapter_down2 = nn.Linear(in_features, r2, bias=False)
        self.adapter_up2 = nn.Linear(r2, out_features, bias=False)

        nn.init.normal_(self.adapter_down2.weight, std=1.0 / (r2 ** 2))
        nn.init.zeros_(self.adapter_up2.weight)

        # ========== 스케일 파라미터 ==========
        # float 또는 learnable Parameter로 사용 가능
        self.scale1 = init_scale1
        self.scale2 = init_scale2

    def set_scales(self, scale1: float, scale2: float):
        """스케일 값 수동 설정 (추론 시 도메인 유사도에 따라)"""
        self.scale1 = scale1
        self.scale2 = scale2

    def get_scales(self) -> Tuple[float, float]:
        """현재 스케일 값 반환"""
        s1 = self.scale1.item() if isinstance(self.scale1, torch.Tensor) else self.scale1
        s2 = self.scale2.item() if isinstance(self.scale2, torch.Tensor) else self.scale2
        return s1, s2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        S2R-Adapter forward pass

        h = shared(x) + up1(down1(x)) * scale1 + up2(down2(x)) * scale2
        """
        # 공유 브랜치
        out_shared = self.linear_shared(x)

        # 전송 브랜치 1
        out_transfer1 = self.adapter_up1(self.adapter_down1(x))

        # 전송 브랜치 2
        out_transfer2 = self.adapter_up2(self.adapter_down2(x))

        # 스케일 적용 (float 또는 Tensor 모두 지원)
        if isinstance(self.scale1, (int, float)):
            return out_shared + out_transfer1 * self.scale1 + out_transfer2 * self.scale2
        else:
            # Tensor인 경우 device 맞춤
            device = x.device
            return (out_shared +
                    out_transfer1 * self.scale1.to(device) +
                    out_transfer2 * self.scale2.to(device))

    def get_adapter_parameters(self) -> List[nn.Parameter]:
        """어댑터 파라미터만 반환 (옵티마이저용)"""
        params = []
        params.extend(self.adapter_down1.parameters())
        params.extend(self.adapter_up1.parameters())
        params.extend(self.adapter_down2.parameters())
        params.extend(self.adapter_up2.parameters())
        # scale이 Parameter인 경우 추가
        if isinstance(self.scale1, nn.Parameter):
            params.append(self.scale1)
        if isinstance(self.scale2, nn.Parameter):
            params.append(self.scale2)
        return params

    def merge_weights(self) -> nn.Linear:
        """
        어댑터 가중치를 공유 브랜치에 병합

        추론 시 사용하여 추가 연산 오버헤드 제거

        Returns:
            merged_layer: 병합된 nn.Linear 레이어
        """
        merged = nn.Linear(
            self.in_features, self.out_features,
            bias=self.linear_shared.bias is not None
        )

        # 스케일 값 추출
        s1 = self.scale1.item() if isinstance(self.scale1, torch.Tensor) else self.scale1
        s2 = self.scale2.item() if isinstance(self.scale2, torch.Tensor) else self.scale2

        # W_merged = W_shared + scale1 * (up1 @ down1) + scale2 * (up2 @ down2)
        with torch.no_grad():
            delta_w1 = s1 * (self.adapter_up1.weight @ self.adapter_down1.weight)
            delta_w2 = s2 * (self.adapter_up2.weight @ self.adapter_down2.weight)

            merged.weight.data.copy_(
                self.linear_shared.weight.data + delta_w1 + delta_w2
            )

            if self.linear_shared.bias is not None:
                merged.bias.data.copy_(self.linear_shared.bias.data)

        return merged

    def extra_repr(self) -> str:
        s1, s2 = self.get_scales()
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'r1={self.r1}, r2={self.r2}, scale1={s1:.4f}, scale2={s2:.4f}')


class S2RAdapterFullyConnected(nn.Module):
    """
    S2R-Adapter for StyleGAN2's FullyConnectedLayer.

    FullyConnectedLayer는 표준 nn.Linear와 다르게:
    - weight_gain, bias_gain 적용
    - 활성화 함수 통합 (linear, lrelu 등)

    이 어댑터는 원본 FullyConnectedLayer를 유지하면서
    전송 브랜치의 출력을 더합니다.

    Args:
        original_layer: 원본 FullyConnectedLayer
        r1: 전송 브랜치 1의 rank (기본값: 1)
        r2: 전송 브랜치 2의 rank (기본값: 128)
        freeze_shared: 원본 레이어 동결 여부 (기본값: True)
        init_scale1: scale1 초기값 (기본값: 1.0)
        init_scale2: scale2 초기값 (기본값: 1.0)
    """

    def __init__(
        self,
        original_layer,  # FullyConnectedLayer
        r1: int = 1,
        r2: int = 128,
        freeze_shared: bool = True,
        init_scale1: float = 1.0,
        init_scale2: float = 1.0,
    ):
        super().__init__()

        # 원본 레이어 저장
        self.original = original_layer

        # 원본 레이어 차원 추출
        # FullyConnectedLayer: weight shape is [out_features, in_features]
        out_features, in_features = original_layer.weight.shape

        self.in_features = in_features
        self.out_features = out_features
        self.r1 = r1
        self.r2 = r2

        # 원본 레이어 동결
        if freeze_shared:
            for param in self.original.parameters():
                param.requires_grad = False

        # ========== 전송 브랜치 1 ==========
        self.adapter_down1 = nn.Linear(in_features, r1, bias=False)
        self.adapter_up1 = nn.Linear(r1, out_features, bias=False)

        nn.init.normal_(self.adapter_down1.weight, std=1.0 / (r1 ** 2))
        nn.init.zeros_(self.adapter_up1.weight)

        # ========== 전송 브랜치 2 ==========
        self.adapter_down2 = nn.Linear(in_features, r2, bias=False)
        self.adapter_up2 = nn.Linear(r2, out_features, bias=False)

        nn.init.normal_(self.adapter_down2.weight, std=1.0 / (r2 ** 2))
        nn.init.zeros_(self.adapter_up2.weight)

        # ========== 스케일 파라미터 ==========
        self.scale1 = init_scale1
        self.scale2 = init_scale2

    def set_scales(self, scale1: float, scale2: float):
        """스케일 값 수동 설정"""
        self.scale1 = scale1
        self.scale2 = scale2

    def get_scales(self) -> Tuple[float, float]:
        """현재 스케일 값 반환"""
        s1 = self.scale1.item() if isinstance(self.scale1, torch.Tensor) else self.scale1
        s2 = self.scale2.item() if isinstance(self.scale2, torch.Tensor) else self.scale2
        return s1, s2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FullyConnectedLayer와 호환되는 forward

        원본 레이어의 활성화/gain/bias 처리를 유지하면서
        어댑터 출력을 더함

        주의: 어댑터 출력은 활성화 함수 적용 전에 더해짐
        (원본의 활성화가 linear인 경우 문제없음)
        """
        # 원본 레이어 출력 (활성화 포함)
        out_original = self.original(x)

        # 어댑터 출력 (순수 선형 변환만)
        adapter_out1 = self.adapter_up1(self.adapter_down1(x))
        adapter_out2 = self.adapter_up2(self.adapter_down2(x))

        # 스케일 적용
        if isinstance(self.scale1, (int, float)):
            return out_original + adapter_out1 * self.scale1 + adapter_out2 * self.scale2
        else:
            device = x.device
            return (out_original +
                    adapter_out1 * self.scale1.to(device) +
                    adapter_out2 * self.scale2.to(device))

    def get_adapter_parameters(self) -> List[nn.Parameter]:
        """어댑터 파라미터만 반환"""
        params = []
        params.extend(self.adapter_down1.parameters())
        params.extend(self.adapter_up1.parameters())
        params.extend(self.adapter_down2.parameters())
        params.extend(self.adapter_up2.parameters())
        if isinstance(self.scale1, nn.Parameter):
            params.append(self.scale1)
        if isinstance(self.scale2, nn.Parameter):
            params.append(self.scale2)
        return params

    def extra_repr(self) -> str:
        s1, s2 = self.get_scales()
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'r1={self.r1}, r2={self.r2}, scale1={s1:.4f}, scale2={s2:.4f}')


# ============================================================================
# Helper Functions
# ============================================================================

def apply_s2r_adapter_to_generator(
    generator: nn.Module,
    r1: int = 1,
    r2: int = 128,
    freeze_shared: bool = True,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    init_scale1: float = 1.0,
    init_scale2: float = 1.0,
) -> Tuple[nn.Module, Dict[str, List[nn.Parameter]]]:
    """
    Generator에 S2R-Adapter 적용

    Args:
        generator: StyleGAN2 Generator
        r1: 전송 브랜치 1의 rank (기본값: 1, 미세 조정용)
        r2: 전송 브랜치 2의 rank (기본값: 128, 광범위 적응용)
        freeze_shared: 공유 브랜치 동결 여부 (기본값: True)
        target_modules: 적용 대상 모듈 이름 패턴 (기본값: ['affine'])
        exclude_modules: 제외 모듈 이름 패턴
        init_scale1, init_scale2: 스케일 초기값

    Returns:
        generator: S2R-Adapter가 적용된 Generator
        adapter_params: 어댑터 파라미터 딕셔너리
    """
    if target_modules is None:
        target_modules = ['affine']

    if exclude_modules is None:
        exclude_modules = []

    adapter_params = {}
    modules_to_replace = []

    # 대상 모듈 탐색
    for name, module in generator.named_modules():
        is_target = any(target in name for target in target_modules)
        is_excluded = any(exclude in name for exclude in exclude_modules)

        if is_target and not is_excluded:
            # FullyConnectedLayer인지 확인 (activation 속성으로 판별)
            if hasattr(module, 'weight') and hasattr(module, 'activation'):
                modules_to_replace.append((name, module, 'fc'))
            elif isinstance(module, nn.Linear):
                modules_to_replace.append((name, module, 'linear'))

    # S2R-Adapter로 교체
    for name, module, layer_type in modules_to_replace:
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(generator.named_modules())[parent_name]
        else:
            attr_name = parts[0]
            parent = generator

        # 어댑터 레이어 생성
        if layer_type == 'fc':
            adapter_layer = S2RAdapterFullyConnected(
                module, r1=r1, r2=r2, freeze_shared=freeze_shared,
                init_scale1=init_scale1, init_scale2=init_scale2
            )
        else:
            adapter_layer = S2RAdapterLinear(
                module, r1=r1, r2=r2, freeze_shared=freeze_shared,
                init_scale1=init_scale1, init_scale2=init_scale2
            )

        setattr(parent, attr_name, adapter_layer)

        # 파라미터 수집
        adapter_params[name] = adapter_layer.get_adapter_parameters()

    print(f"[S2R-Adapter] Applied to {len(modules_to_replace)} modules (r1={r1}, r2={r2})")

    return generator, adapter_params


def get_s2r_adapter_parameters(generator: nn.Module) -> List[nn.Parameter]:
    """
    Generator에서 S2R-Adapter 파라미터만 추출

    Args:
        generator: S2R-Adapter가 적용된 Generator

    Returns:
        params: 모든 어댑터 파라미터 리스트
    """
    params = []
    for module in generator.modules():
        if isinstance(module, (S2RAdapterLinear, S2RAdapterFullyConnected)):
            params.extend(module.get_adapter_parameters())
    return params


def freeze_non_adapter_parameters(generator: nn.Module) -> None:
    """
    S2R-Adapter 파라미터를 제외한 모든 파라미터 동결

    Args:
        generator: S2R-Adapter가 적용된 Generator
    """
    # 어댑터 파라미터 ID 수집
    adapter_param_ids = set()
    for module in generator.modules():
        if isinstance(module, (S2RAdapterLinear, S2RAdapterFullyConnected)):
            for param in module.get_adapter_parameters():
                adapter_param_ids.add(id(param))

    # 어댑터가 아닌 파라미터 동결
    frozen_count = 0
    for param in generator.parameters():
        if id(param) not in adapter_param_ids:
            param.requires_grad = False
            frozen_count += 1

    print(f"[S2R-Adapter] Frozen {frozen_count} non-adapter parameters")


def set_adapter_scales(
    generator: nn.Module,
    scale1: float,
    scale2: float
) -> None:
    """
    모든 S2R-Adapter 레이어의 스케일 설정

    추론 시 도메인 유사도에 따라 조절:
    - scale 낮음: 구조적 지식 중심 (합성 데이터와 유사)
    - scale 높음: 물리적 보정 중심 (실측 데이터와 유사)

    Args:
        generator: S2R-Adapter가 적용된 Generator
        scale1: 전송 브랜치 1 스케일
        scale2: 전송 브랜치 2 스케일
    """
    for module in generator.modules():
        if isinstance(module, (S2RAdapterLinear, S2RAdapterFullyConnected)):
            module.set_scales(scale1, scale2)


def get_adapter_scales(generator: nn.Module) -> List[Tuple[str, float, float]]:
    """
    모든 S2R-Adapter 레이어의 스케일 값 조회

    Args:
        generator: S2R-Adapter가 적용된 Generator

    Returns:
        scales: [(module_name, scale1, scale2), ...] 리스트
    """
    scales = []
    for name, module in generator.named_modules():
        if isinstance(module, (S2RAdapterLinear, S2RAdapterFullyConnected)):
            s1, s2 = module.get_scales()
            scales.append((name, s1, s2))
    return scales


def merge_adapter_weights(generator: nn.Module) -> nn.Module:
    """
    모든 S2R-Adapter 가중치를 원본에 병합 (추론 최적화)

    주의: S2RAdapterFullyConnected는 병합이 복잡하므로 현재 지원하지 않음.
          S2RAdapterLinear만 병합됨.

    Args:
        generator: S2R-Adapter가 적용된 Generator

    Returns:
        merged: 병합된 Generator (복사본)
    """
    merged = copy.deepcopy(generator)

    merged_count = 0
    skipped_count = 0

    for name, module in list(merged.named_modules()):
        if isinstance(module, S2RAdapterLinear):
            merged_layer = module.merge_weights()

            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(merged.named_modules())[parent_name]
            else:
                attr_name = parts[0]
                parent = merged

            setattr(parent, attr_name, merged_layer)
            merged_count += 1

        elif isinstance(module, S2RAdapterFullyConnected):
            # FullyConnectedLayer는 병합이 복잡하므로 스킵
            skipped_count += 1

    print(f"[S2R-Adapter] Merged {merged_count} layers, skipped {skipped_count} FullyConnected layers")

    return merged


def count_s2r_adapter_parameters(generator: nn.Module) -> Tuple[int, int, int]:
    """
    S2R-Adapter 파라미터 수 계산

    Args:
        generator: S2R-Adapter가 적용된 Generator

    Returns:
        adapter_params: 어댑터 파라미터 수
        total_params: 전체 파라미터 수
        trainable_params: 학습 가능한 파라미터 수
    """
    adapter_param_ids = set()
    adapter_params = 0

    for module in generator.modules():
        if isinstance(module, (S2RAdapterLinear, S2RAdapterFullyConnected)):
            for param in module.get_adapter_parameters():
                adapter_param_ids.add(id(param))
                adapter_params += param.numel()

    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)

    return adapter_params, total_params, trainable_params


def make_scales_learnable(
    generator: nn.Module,
    init_scale1: float = 1.0,
    init_scale2: float = 1.0,
) -> List[nn.Parameter]:
    """
    스케일 파라미터를 학습 가능하게 설정

    Args:
        generator: S2R-Adapter가 적용된 Generator
        init_scale1: scale1 초기값
        init_scale2: scale2 초기값

    Returns:
        scale_params: 학습 가능한 스케일 파라미터 리스트
    """
    scale_params = []

    for module in generator.modules():
        if isinstance(module, (S2RAdapterLinear, S2RAdapterFullyConnected)):
            # 기존 스케일 값을 Parameter로 변환
            module.scale1 = nn.Parameter(torch.tensor(init_scale1))
            module.scale2 = nn.Parameter(torch.tensor(init_scale2))
            scale_params.extend([module.scale1, module.scale2])

    print(f"[S2R-Adapter] Made {len(scale_params)} scale parameters learnable")

    return scale_params


# ============================================================================
# TTA: Uncertainty Calculation and Adaptive Scale
# ============================================================================

def compute_uncertainty_from_outputs(
    outputs: List[torch.Tensor],
    uncertainty_scale: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TTA 출력들의 분산을 이용한 불확실성 계산

    S2R-HDR 원본 공식 (train_adapter_without_gt.py:259-264):
        ps_label = torch.stack(ps_label_aug)
        variance = torch.var(ps_label)
        uncertainty = torch.mean(variance) * uncertainty_scale

    Args:
        outputs: Generator 출력 리스트 [원본, aug1, aug2, ...]
                 각 출력은 (B, C, H, W) 형태
        uncertainty_scale: 불확실성 스케일링 계수
                          - GT 있는 경우: 0.1 (S2R-HDR 기본값)
                          - GT 없는 경우: 0.05

    Returns:
        uncertainty: 스칼라 불확실성 값
        variance_map: 픽셀별 분산 맵 (B, C, H, W)

    Example:
        >>> outputs = [panorama1, panorama2, panorama3, ...]  # TTA 출력들
        >>> uncertainty, var_map = compute_uncertainty_from_outputs(outputs)
        >>> print(f"Uncertainty: {uncertainty:.4f}")
    """
    # 리스트 -> 스택
    stacked = torch.stack(outputs, dim=0)  # (N, B, C, H, W)

    # 분산 계산 (증강 축 방향)
    variance_map = torch.var(stacked, dim=0)  # (B, C, H, W)

    # 평균 불확실성
    uncertainty = torch.mean(variance_map) * uncertainty_scale

    return uncertainty, variance_map


def adaptive_scale_from_uncertainty(
    uncertainty: Union[torch.Tensor, float],
    base_scale1: float = 1.0,
    base_scale2: float = 1.0,
    clamp_range: Tuple[float, float] = (0.0, 2.0),
) -> Tuple[float, float]:
    """
    불확실성 기반 동적 스케일 계산

    S2R-HDR 원본 공식 (train_adapter_without_gt.py:262-268):
        if args.adaptive_scale:
            set_scale(model, 1 - uncertainty, 1 + uncertainty)

    불확실성이 높으면:
        - scale1 감소 → 공유 브랜치(구조) 의존도 낮춤
        - scale2 증가 → 전송 브랜치(적응) 강화

    불확실성이 낮으면:
        - scale1 유지 → 기존 구조 유지
        - scale2 유지 → 최소한의 적응

    Args:
        uncertainty: 불확실성 스칼라 (0~1 범위 권장)
        base_scale1: 기본 scale1 값
        base_scale2: 기본 scale2 값
        clamp_range: 스케일 클램핑 범위 (min, max)

    Returns:
        scale1, scale2: 조절된 스케일 값

    Example:
        >>> uncertainty = 0.3
        >>> scale1, scale2 = adaptive_scale_from_uncertainty(uncertainty)
        >>> print(f"scale1={scale1:.2f}, scale2={scale2:.2f}")
        # scale1=0.70, scale2=1.30
    """
    # 불확실성 값 추출
    if isinstance(uncertainty, torch.Tensor):
        U = uncertainty.item()
    else:
        U = float(uncertainty)

    # S2R-HDR 원본 공식
    scale1 = base_scale1 * (1.0 - U)
    scale2 = base_scale2 * (1.0 + U)

    # 클램핑
    scale1 = max(clamp_range[0], min(clamp_range[1], scale1))
    scale2 = max(clamp_range[0], min(clamp_range[1], scale2))

    return scale1, scale2


def apply_adaptive_scales(
    generator: nn.Module,
    uncertainty: Union[torch.Tensor, float],
    base_scale1: float = 1.0,
    base_scale2: float = 1.0,
) -> Tuple[float, float]:
    """
    불확실성 기반 동적 스케일 계산 및 Generator에 적용

    S2R-HDR 원본: set_scale(model, 1 - uncertainty, 1 + uncertainty)

    Args:
        generator: S2R-Adapter가 적용된 Generator
        uncertainty: 불확실성 값
        base_scale1, base_scale2: 기본 스케일 값

    Returns:
        scale1, scale2: 적용된 스케일 값

    Example:
        >>> uncertainty, _ = compute_uncertainty_from_outputs(outputs)
        >>> scale1, scale2 = apply_adaptive_scales(G, uncertainty)
        >>> print(f"Applied: scale1={scale1:.2f}, scale2={scale2:.2f}")
    """
    scale1, scale2 = adaptive_scale_from_uncertainty(
        uncertainty, base_scale1, base_scale2
    )

    # Generator에 스케일 적용
    set_adapter_scales(generator, scale1, scale2)

    return scale1, scale2


# ============================================================================
# Test Functions
# ============================================================================

def _test_s2r_adapter():
    """S2R-Adapter 모듈 단위 테스트"""
    print("=" * 60)
    print("Testing S2R-Adapter module...")
    print("=" * 60)

    # 1. S2RAdapterLinear 기본 동작 테스트
    print("\n[Test 1] S2RAdapterLinear basic forward...")
    linear = nn.Linear(512, 256)
    adapter = S2RAdapterLinear(linear, r1=1, r2=128, freeze_shared=True)
    x = torch.randn(4, 512)
    output = adapter(x)
    assert output.shape == (4, 256), f"Expected (4, 256), got {output.shape}"
    print(f"  Shape check: PASS ({output.shape})")

    # 2. 초기화 테스트 (시작 시 원본과 거의 동일)
    print("\n[Test 2] Initial output should match original...")
    with torch.no_grad():
        original_out = linear(x)
        adapter_out = adapter(x)
        diff = (original_out - adapter_out).abs().max()
        # up 레이어가 zeros로 초기화되므로 차이는 0에 가까워야 함
        assert diff < 1e-5, f"Initial output should match original, diff={diff}"
    print(f"  Initial output match: PASS (max diff={diff:.2e})")

    # 3. 스케일 조절 테스트
    print("\n[Test 3] Scale adjustment...")
    adapter.set_scales(0.5, 0.3)
    s1, s2 = adapter.get_scales()
    assert s1 == 0.5 and s2 == 0.3, f"Expected (0.5, 0.3), got ({s1}, {s2})"
    print(f"  Scale adjustment: PASS (scale1={s1}, scale2={s2})")

    # 4. 가중치 병합 테스트
    print("\n[Test 4] Weight merging...")
    adapter.set_scales(1.0, 1.0)
    # 먼저 어댑터에 약간의 학습을 시뮬레이션
    with torch.no_grad():
        adapter.adapter_up1.weight.fill_(0.01)
        adapter.adapter_up2.weight.fill_(0.001)

    merged = adapter.merge_weights()
    merged_out = merged(x)
    adapter_out = adapter(x)
    diff = (merged_out - adapter_out).abs().max()
    assert diff < 1e-5, f"Merged output should match adapter, diff={diff}"
    print(f"  Weight merging: PASS (max diff={diff:.2e})")

    # 5. 파라미터 수 확인
    print("\n[Test 5] Parameter count...")
    original_params = sum(p.numel() for p in linear.parameters())
    adapter_params_list = adapter.get_adapter_parameters()
    adapter_params = sum(p.numel() for p in adapter_params_list)
    print(f"  Original params: {original_params:,}")
    print(f"  Adapter params: {adapter_params:,}")
    print(f"  Adapter overhead: {adapter_params / original_params * 100:.2f}%")

    # 6. Generator 적용 테스트 (Mock)
    print("\n[Test 6] Apply to mock generator...")

    class MockGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.synthesis = nn.ModuleDict({
                'b4': nn.ModuleDict({
                    'conv1': nn.ModuleDict({
                        'affine': nn.Linear(512, 512)
                    }),
                    'conv2': nn.ModuleDict({
                        'affine': nn.Linear(512, 512)
                    })
                }),
                'b8': nn.ModuleDict({
                    'conv1': nn.ModuleDict({
                        'affine': nn.Linear(512, 512)
                    })
                })
            })

    G = MockGenerator()
    G, params = apply_s2r_adapter_to_generator(G, r1=1, r2=128)

    # 어댑터가 적용된 레이어 수 확인
    adapter_count = sum(1 for m in G.modules()
                       if isinstance(m, (S2RAdapterLinear, S2RAdapterFullyConnected)))
    assert adapter_count == 3, f"Expected 3 adapters, got {adapter_count}"
    print(f"  Applied to {adapter_count} layers: PASS")

    # 7. freeze_non_adapter_parameters 테스트
    print("\n[Test 7] Freeze non-adapter parameters...")
    freeze_non_adapter_parameters(G)

    trainable_params = sum(1 for p in G.parameters() if p.requires_grad)
    adapter_param_count = len(get_s2r_adapter_parameters(G))
    print(f"  Trainable params: {trainable_params}")
    print(f"  Adapter params: {adapter_param_count}")

    # 8. set_adapter_scales 테스트
    print("\n[Test 8] Set adapter scales globally...")
    set_adapter_scales(G, 0.7, 0.5)
    scales = get_adapter_scales(G)
    for name, s1, s2 in scales:
        assert s1 == 0.7 and s2 == 0.5, f"Scale mismatch at {name}"
    print(f"  Global scale setting: PASS ({len(scales)} layers)")

    print("\n" + "=" * 60)
    print("All S2R-Adapter tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _test_s2r_adapter()
