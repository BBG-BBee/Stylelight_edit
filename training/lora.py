"""
LoRA (Low-Rank Adaptation) 어댑터

Stage 2 학습에서 Generator의 물리적 정합성을 보정하기 위한
저랭크 적응 레이어입니다.

S2R-Adapter 전략:
- Stage 1에서 학습된 Generator 가중치를 동결(Freeze)
- LoRA 레이어만 Laval 데이터셋으로 학습
- 구조적 지식 보존 + 물리적 보정

LoRA 공식:
    h = W₀x + ΔWx = W₀x + BAx

    여기서:
    - W₀: 원본 가중치 (동결)
    - B: (out_features, rank) 행렬
    - A: (rank, in_features) 행렬
    - rank << min(in_features, out_features)

장점:
- 학습 파라미터 수 대폭 감소
- 원본 모델 구조 보존
- 과적합 방지
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import copy
import math


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 레이어

    원본 레이어의 출력에 저랭크 행렬의 곱을 더합니다.
    원본 레이어는 동결되고, LoRA 파라미터만 학습됩니다.

    Args:
        original_layer: 원본 레이어 (nn.Linear 또는 nn.Conv2d)
        rank: LoRA 랭크 (기본값: 16)
        alpha: 스케일링 파라미터 (기본값: 1.0)
        dropout: 드롭아웃 비율 (기본값: 0.0)

    Example:
        >>> linear = nn.Linear(512, 512)
        >>> lora_linear = LoRALayer(linear, rank=16)
        >>> x = torch.randn(1, 512)
        >>> output = lora_linear(x)
    """

    def __init__(self,
                 original_layer: nn.Module,
                 rank: int = 16,
                 alpha: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()

        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 원본 레이어 동결
        for param in self.original.parameters():
            param.requires_grad = False

        # 레이어 타입에 따른 처리
        if isinstance(original_layer, nn.Linear):
            self._init_linear_lora(original_layer)
        elif isinstance(original_layer, nn.Conv2d):
            self._init_conv2d_lora(original_layer)
        else:
            raise TypeError(f"Unsupported layer type: {type(original_layer)}")

        # 드롭아웃
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def _init_linear_lora(self, layer: nn.Linear):
        """Linear 레이어용 LoRA 초기화"""
        in_features = layer.in_features
        out_features = layer.out_features

        # A: (rank, in_features) - Kaiming 초기화
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: (out_features, rank) - 0으로 초기화 (시작 시 ΔW = 0)
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))

        self.layer_type = 'linear'

    def _init_conv2d_lora(self, layer: nn.Conv2d):
        """Conv2d 레이어용 LoRA 초기화"""
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        kernel_size = layer.kernel_size

        # Conv2d는 (out, in, kH, kW) 형태의 가중치를 가짐
        # LoRA를 1x1 conv로 근사

        # A: (rank, in_channels, 1, 1)
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_channels, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: (out_channels, rank, 1, 1)
        self.lora_B = nn.Parameter(torch.zeros(out_channels, self.rank, 1, 1))

        self.layer_type = 'conv2d'
        self.stride = layer.stride
        self.padding = layer.padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA 적용된 forward pass

        output = original(x) + scaling * B @ A @ x
        """
        # 원본 레이어 출력
        original_output = self.original(x)

        # LoRA 출력
        if self.layer_type == 'linear':
            # x: (batch, in_features)
            # A @ x: (batch, rank)
            # B @ (A @ x): (batch, out_features)
            lora_output = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        else:
            # Conv2d 케이스
            # 1x1 conv로 근사
            lora_output = F.conv2d(
                F.conv2d(self.dropout(x), self.lora_A),
                self.lora_B
            )

        return original_output + self.scaling * lora_output

    def merge_weights(self) -> nn.Module:
        """
        LoRA 가중치를 원본 가중치에 병합

        추론 시 사용하여 추가 연산 없이 결과를 얻을 수 있습니다.

        Returns:
            merged_layer: LoRA가 병합된 새로운 레이어
        """
        merged = copy.deepcopy(self.original)

        if self.layer_type == 'linear':
            # W_merged = W_original + scaling * B @ A
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            merged.weight.data += delta_w
        else:
            # Conv2d: (out, in, kH, kW)
            # 1x1 conv의 경우에만 정확한 병합 가능
            if self.original.kernel_size == (1, 1):
                delta_w = self.scaling * (self.lora_B @ self.lora_A.view(self.rank, -1)).view_as(self.original.weight)
                merged.weight.data += delta_w

        return merged

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


class LoRALinear(LoRALayer):
    """Linear 레이어 전용 LoRA (편의 클래스)"""

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, alpha: float = 1.0, dropout: float = 0.0,
                 bias: bool = True):
        original = nn.Linear(in_features, out_features, bias=bias)
        super().__init__(original, rank=rank, alpha=alpha, dropout=dropout)


def apply_lora_to_generator(generator: nn.Module,
                            rank: int = 16,
                            alpha: float = 1.0,
                            target_modules: Optional[List[str]] = None,
                            exclude_modules: Optional[List[str]] = None) -> Tuple[nn.Module, Dict[str, nn.Parameter]]:
    """
    Generator에 LoRA 레이어 적용

    Stage 1에서 학습된 Generator의 특정 레이어에 LoRA를 적용합니다.

    Args:
        generator: Stage 1 Generator
        rank: LoRA 랭크
        alpha: 스케일링 파라미터
        target_modules: LoRA를 적용할 모듈 이름 패턴 (기본값: affine 레이어)
        exclude_modules: 제외할 모듈 이름 패턴

    Returns:
        generator: LoRA가 적용된 Generator
        lora_params: LoRA 파라미터 딕셔너리 (옵티마이저용)

    Example:
        >>> G_stage1 = load_checkpoint('Checkpoint-Stage1.pkl')
        >>> G_stage2, lora_params = apply_lora_to_generator(G_stage1, rank=16)
        >>> optimizer = torch.optim.Adam(lora_params.values(), lr=1e-4)
    """
    if target_modules is None:
        # 기본값: SynthesisLayer의 affine 레이어
        target_modules = ['affine']

    if exclude_modules is None:
        exclude_modules = []

    lora_params = {}
    modules_to_replace = []

    # LoRA 적용 대상 모듈 찾기
    for name, module in generator.named_modules():
        # 타겟 모듈인지 확인
        is_target = any(target in name for target in target_modules)
        is_excluded = any(exclude in name for exclude in exclude_modules)

        if is_target and not is_excluded:
            if isinstance(module, nn.Linear):
                modules_to_replace.append((name, module))

    # LoRA 레이어로 교체
    for name, module in modules_to_replace:
        # 부모 모듈과 속성 이름 분리
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(generator.named_modules())[parent_name]
        else:
            attr_name = parts[0]
            parent = generator

        # LoRA 레이어 생성 및 교체
        lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
        setattr(parent, attr_name, lora_layer)

        # LoRA 파라미터 수집
        lora_params[f"{name}.lora_A"] = lora_layer.lora_A
        lora_params[f"{name}.lora_B"] = lora_layer.lora_B

    print(f"Applied LoRA to {len(modules_to_replace)} modules with rank={rank}")

    return generator, lora_params


def freeze_generator_except_lora(generator: nn.Module) -> None:
    """
    Generator의 LoRA 파라미터를 제외한 모든 파라미터 동결

    Args:
        generator: LoRA가 적용된 Generator
    """
    for name, param in generator.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_parameters(generator: nn.Module) -> List[nn.Parameter]:
    """
    Generator에서 LoRA 파라미터만 추출

    Args:
        generator: LoRA가 적용된 Generator

    Returns:
        lora_params: LoRA 파라미터 리스트
    """
    lora_params = []
    for name, param in generator.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
    return lora_params


def count_lora_parameters(generator: nn.Module) -> Tuple[int, int]:
    """
    LoRA 파라미터 수와 전체 파라미터 수 계산

    Args:
        generator: LoRA가 적용된 Generator

    Returns:
        lora_params: LoRA 파라미터 수
        total_params: 전체 파라미터 수
    """
    lora_params = 0
    total_params = 0

    for name, param in generator.named_parameters():
        total_params += param.numel()
        if 'lora_A' in name or 'lora_B' in name:
            lora_params += param.numel()

    return lora_params, total_params


def merge_lora_weights(generator: nn.Module) -> nn.Module:
    """
    Generator의 모든 LoRA 가중치를 원본에 병합

    추론 시 사용하여 추가 연산 오버헤드를 제거합니다.

    Args:
        generator: LoRA가 적용된 Generator

    Returns:
        merged_generator: LoRA가 병합된 Generator
    """
    merged = copy.deepcopy(generator)

    for name, module in list(merged.named_modules()):
        if isinstance(module, LoRALayer):
            # 병합된 레이어로 교체
            merged_layer = module.merge_weights()

            # 부모 모듈에서 교체
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(merged.named_modules())[parent_name]
            else:
                attr_name = parts[0]
                parent = merged

            setattr(parent, attr_name, merged_layer)

    return merged


# 테스트 함수
def _test_lora():
    """LoRA 모듈 테스트"""
    print("Testing LoRA module...")

    # 테스트용 Linear 레이어
    linear = nn.Linear(512, 512)

    # LoRA 적용
    lora_linear = LoRALayer(linear, rank=16, alpha=1.0)
    print(f"LoRA Linear: {lora_linear}")

    # Forward 테스트
    x = torch.randn(4, 512)
    output = lora_linear(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")

    # 파라미터 수 확인
    original_params = sum(p.numel() for p in linear.parameters())
    lora_added_params = lora_linear.lora_A.numel() + lora_linear.lora_B.numel()
    print(f"Original params: {original_params:,}")
    print(f"LoRA added params: {lora_added_params:,}")
    print(f"LoRA overhead: {lora_added_params / original_params * 100:.2f}%")

    # 가중치 병합 테스트
    merged = lora_linear.merge_weights()
    merged_output = merged(x)
    diff = (output - merged_output).abs().max()
    print(f"Merge test - Max difference: {diff.item():.6f}")

    print("\nLoRA test passed!")


if __name__ == "__main__":
    _test_lora()
