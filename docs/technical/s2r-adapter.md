# S2R-Adapter 기술 문서

> Sim-to-Real Adapter: 2-브랜치 구조 기반 효율적 도메인 적응

---

## 1. 개요

S2R-Adapter는 합성(Synthetic) 데이터로 사전학습된 모델을 실측(Real) 데이터에 효율적으로 적응시키기 위한 어댑터 구조입니다. 원본 모델의 가중치를 동결하고, 소수의 학습 가능한 파라미터만 추가하여 도메인 간 차이를 보정합니다.

### 1.1 핵심 특징

| 특징 | 설명 |
|------|------|
| **2-브랜치 구조** | 공유 브랜치 + 전송 브랜치 (Transfer1 + Transfer2) |
| **효율성** | 전체 파라미터의 약 7-8%만 학습 |
| **스케일 조절** | 일반 모드(고정/학습) 및 TTA 모드(동적) 지원 |
| **원본 보존** | 원본 모델 가중치 동결, 구조적 지식 유지 |

### 1.2 적용 대상

S2R-Adapter는 StyleGAN2 Generator의 **ModulatedConv2d 레이어 내 affine 변환**에 적용됩니다. 스타일 벡터를 변조하는 affine 레이어가 도메인 특성을 결정하는 핵심 요소이기 때문입니다.

---

## 2. 2-브랜치 구조

### 2.1 출력 공식

원본 affine 변환 $y = W_{orig} \cdot x + b_{orig}$에 대해, S2R-Adapter는 다음과 같이 출력을 변환합니다:

$$y_{adapted} = y_{shared} + \alpha_1 \cdot y_{transfer1} + \alpha_2 \cdot y_{transfer2}$$

### 2.2 공유 브랜치 (Share Branch)

| 항목 | 내용 |
|------|------|
| **구성** | 원본 Linear 레이어 (동결) |
| **수식** | $y_{shared} = W_{orig} \cdot x + b_{orig}$ |
| **역할** | Stage 1에서 학습된 구조적 지식 보존 |
| **학습** | 가중치 동결 (`requires_grad=False`) |

합성 데이터(S2R-HDR)로 학습한 파노라마의 공간 구조, 창문 위치, 조명 배치 등의 특성을 유지합니다.

### 2.3 전송 브랜치 (Transfer Branch)

전송 브랜치는 두 개의 Low-Rank Adapter로 구성됩니다:

| 구성요소 | Rank | 수식 | 역할 |
|----------|------|------|------|
| **Transfer1** | r1=1 | $y_{t1} = B_1 \cdot A_1 \cdot x$ | 전역적 도메인 이동 (휘도 스케일 등) |
| **Transfer2** | r2=128 | $y_{t2} = B_2 \cdot A_2 \cdot x$ | 세밀한 도메인 적응 (공간별 특성) |

**초기화 전략:**

| 행렬 | 초기화 | 의미 |
|------|--------|------|
| $A$ (down) | `normal(std=1/r²)` | 입력 특성 압축 |
| $B$ (up) | `zeros` | 학습 시작 시 원본과 동일한 출력 보장 |

### 2.4 스케일링 계수 (α₁, α₂)

스케일링 계수는 전송 브랜치의 기여도를 조절합니다.

**일반 모드 (학습/추론)**

| 설정 | 사용 시점 | 설명 |
|------|----------|------|
| **고정값** | Stage 2 학습 | scale1=1.0, scale2=1.0 유지 |
| **학습 가능** | 파인튜닝 | `make_scales_learnable()`로 최적값 학습 |
| **수동 설정** | 추론 | `set_adapter_scales()`로 직접 지정 |

**TTA 모드 (Test-Time Adaptation)**

추론 시점에 입력의 불확실성에 따라 동적으로 조절:

- $\alpha_1 = 1 - U(x)$ : 불확실성 높으면 공유 브랜치 의존도 감소
- $\alpha_2 = 1 + U(x)$ : 불확실성 높으면 전송 브랜치 강화

---

## 3. 적용 방법

```python
# Generator에 S2R-Adapter 적용
apply_s2r_adapter_to_generator(G, r1=1, r2=128)

# 원본 가중치 동결
freeze_non_adapter_parameters(G)
```

---

## 4. 스케일 조절

### 4.1 수동 스케일 설정

```python
set_adapter_scales(G, scale1=0.8, scale2=1.2)
```

### 4.2 스케일 학습 활성화

```python
make_scales_learnable(G)
params = get_s2r_adapter_parameters(G, include_scales=True)
```

---

## 5. Test-Time Adaptation (TTA)

### 5.1 개요

TTA는 추론 시점에 입력의 불확실성을 측정하여 스케일링 계수를 동적으로 조절하는 기법입니다.

!!! info "TTA 사용 시나리오"
    - Laval 데이터로 파인튜닝된 모델은 TTA가 **불필요**합니다 (기본 off)
    - 학습에 사용되지 않은 새로운 도메인에서 추론할 때 **선택적으로** 활성화

### 5.2 불확실성 계산

입력에 다양한 증강(노출, 화이트밸런스, 뒤집기 등)을 적용하고, 출력들의 분산으로 불확실성을 측정합니다:

$$U(x) = \text{scale} \cdot \mathbb{E}[\text{Var}(f_{aug}(x))]$$

### 5.3 권장 설정

| 시나리오 | TTA | 비고 |
|----------|-----|------|
| Laval 파인튜닝 후 추론 | OFF | 기본 설정 |
| 새 도메인 추론 | ON | 불확실성 측정 후 동적 조절 |
| 빠른 추론 필요 | OFF | TTA는 N배 추론 비용 |

---

## 6. 요약

| 항목 | 설명 |
|------|------|
| **구조** | 2-브랜치 (공유 + 전송) |
| **전송 브랜치** | Transfer1(r1=1) + Transfer2(r2=128) |
| **학습 파라미터** | 전체의 약 7-8% |
| **스케일 모드** | 고정 / 학습 가능 / TTA 동적 조절 |

**권장 워크플로우**:

1. Stage 1: S2R-HDR로 구조 학습
2. Stage 2: Laval로 S2R-Adapter 파인튜닝 (스케일 학습 포함)
3. 추론: 기본적으로 TTA off, 새 도메인에서만 선택적 활성화
