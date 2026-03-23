# 구현 결과 보고서

## 1. 개요

물리적 정합성 기반 HDR 파노라마 생성 모델의 구현이 완료되었습니다. 이 문서는 수정된 파일과 새로 생성된 파일, 그리고 각각의 변경 내용을 정리합니다.

총 4개의 기존 파일이 수정되었고, 11개의 새로운 파일이 생성되었습니다.

---

## 2. 수정된 기존 파일

### 2.1 training/dataset.py

**변경 목적**: 물리적 휘도를 보존하는 FP32 HDR 데이터 로더 추가

기존 `ImageFolderDataset`은 HDR 이미지를 톤매핑하여 [-1, 1] 범위의 LDR로 변환했습니다. 이 과정에서 절대 휘도 정보가 손실됩니다.

새로 추가된 `HDRPhysicalDataset` 클래스는 EXR/HDR 파일을 톤매핑 없이 그대로 로드합니다. 선형 휘도 스케일을 유지하면서 512×1024 해상도로 로그 도메인 리사이즈합니다.

주요 기능:
- 로그 도메인 리사이즈로 HDR 피크 휘도 보존
- ITU-R BT.709 표준에 따른 휘도 계산 메서드
- `use_tilt_augment=True` 옵션으로 카메라 기울기 증강 (Stage 1 전용)

---

### 2.2 training/networks.py

**변경 목적**: HDR 출력을 위한 Softplus 활성화 함수 적용

기존 `ToRGBLayer`는 선형 활성화를 사용하여 음수 값이 출력될 수 있었습니다. 물리적 휘도는 항상 0 이상이어야 합니다.

새로 추가된 `ToRGBLayerHDR` 클래스는 Softplus 활성화 함수를 사용하여 출력 범위를 [0, ∞)로 제한합니다. Softplus는 ReLU와 달리 0 근처에서 부드럽게 변화하여 그래디언트 흐름이 안정적입니다.

또한 `SynthesisBlock`에 `hdr_mode`와 `hdr_activation` 파라미터를 추가하여, HDR 모드가 활성화되면 자동으로 ToRGBLayerHDR을 사용하도록 했습니다.

---

### 2.3 train.py

**변경 목적**: HDR 물리 모드 설정 추가

391-406번 라인에 HDR 물리 모드 설정을 추가했습니다. 이 모드가 활성화되면:

- Generator와 Discriminator 모두 Full FP32로 강제됩니다 (`num_fp16_res=0`)
- 활성화 클램핑이 제거됩니다 (`conv_clamp=None`)
- Generator 출력층에 Softplus 활성화가 적용됩니다

이 설정은 HDR 이미지의 높은 동적 범위(10^-3 ~ 10^5 cd/m²)를 정확하게 표현하는 데 필수적입니다.

---

### 2.4 training/loss.py

**변경 목적**: 물리적 정합성 손실 함수 추가

파일 상단에 DTAM과 PU21 모듈을 임포트하고, 세 개의 새로운 손실 클래스를 추가했습니다.

**PhysicalLoss**: DTAM 가중치와 PU21 인코딩을 결합한 물리적 손실입니다. 예측 휘도와 정답 휘도를 각각 PU21로 인코딩한 후, DTAM 가중치를 곱한 L1 손실을 계산합니다. 전이 구간(300~1000 cd/m²)에서 더 높은 가중치가 적용됩니다.

**ConsistencyLoss**: Stage 2 학습 시 구조적 일관성을 유지하기 위한 손실입니다. Stage 1과 Stage 2 모델의 출력을 톤매핑한 후 LPIPS로 비교합니다. 이를 통해 물리적 보정 과정에서 구조가 과도하게 변형되는 것을 방지합니다.

**CombinedPhysicalLoss**: PhysicalLoss와 ConsistencyLoss를 통합한 클래스입니다. Stage 2 학습에서 사용됩니다.

---

## 3. 신규 생성 파일

### 3.1 Phase 1: 데이터 인프라

#### data_preprocessing/prepare_datasets.py

S2R-HDR과 Laval 데이터셋을 학습에 적합한 형태로 전처리하는 스크립트입니다.

**S2R-HDR 전처리**: EXR/HDR 파일을 512×1024 Equirectangular로 리사이즈합니다. 로그 도메인 리사이즈를 적용하고 음수 값을 제거한 후 FP32 EXR로 저장합니다.

**Laval 전처리**: HDR 파일 헤더에서 노출값(EXPOSURE)을 읽어 적용합니다. 이를 통해 상대적 휘도를 절대 휘도(cd/m²)로 변환합니다. 휘도 통계(최소, 최대, 평균, 중앙값)를 수집하여 데이터셋의 특성을 파악할 수 있습니다.

또한 전처리된 데이터셋의 유효성을 검사하는 `validate_dataset` 함수와 StyleGAN 학습용 `dataset.json`을 생성하는 함수도 포함되어 있습니다.

---

### 3.2 Phase 3: 핵심 기술

#### training/dtam.py

이중 임계값 적응형 마스킹(DTAM) 가중치 함수를 구현합니다.

DTAM은 휘도 구간에 따라 다른 학습 가중치를 부여합니다. DGP 계산에서 가장 중요한 전이 구간(300~1000 cd/m²)에 높은 가중치를 적용하여 모델이 이 영역을 정확하게 학습하도록 유도합니다.

가중치 함수는 S자 곡선을 따릅니다. 배경 구간에서는 가중치 1.0, 전이 구간에서는 1.0에서 10.0까지 점진적으로 증가, 눈부심 구간에서는 최대 가중치 10.0을 유지합니다.

파라미터 T_onset, T_peak, α, γ는 모두 조정 가능하며, 기본값은 DGP 계산 특성에 맞게 설정되어 있습니다.

#### training/pu21.py

PU21(Perceptually Uniform 2021) 인코딩을 구현합니다.

인간 시각 시스템은 휘도를 로그 스케일로 인지합니다. PU21은 물리적 휘도를 지각적으로 균일한 값으로 변환하여, 손실 함수가 인간 시각 특성에 맞게 오차를 측정하도록 합니다.

간단 모드(simple)와 전체 모드(full)를 지원합니다. 간단 모드는 세 개의 파라미터(a, b, c)를 사용하는 단순한 변환이고, 전체 모드는 더 정확한 변환을 제공합니다. 역변환 함수도 구현되어 있어 PU21 값을 다시 휘도로 변환할 수 있습니다.

#### training/s2r_adapter.py

S2R-Adapter (Sim-to-Real Adapter)를 구현합니다. S2R-HDR 논문의 2-브랜치 도메인 적응 구조입니다.

Stage 2 학습에서는 Stage 1의 구조적 지식을 보존하면서 물리적 보정만 수행해야 합니다. S2R-Adapter는 2-브랜치 구조로 효율적인 도메인 적응을 달성합니다:

**공유 브랜치 (Share Branch)**: 원본 가중치 동결 (Stage 1 지식 보존)

**전송 브랜치 (Transfer Branch)**:

- Transfer1 (r1=1): 전역적 도메인 이동
- Transfer2 (r2=128): 세밀한 도메인 적응

`S2RAdapterLinear` 클래스는 Linear 레이어용, `S2RAdapterFullyConnected` 클래스는 StyleGAN2의 FullyConnectedLayer용입니다. 두 클래스 모두 원본 레이어를 래핑하고, scale1/scale2로 전송 브랜치의 기여도를 조절합니다.

`apply_s2r_adapter_to_generator` 함수는 Generator의 affine 레이어에 S2R-Adapter를 자동으로 적용합니다. 2-브랜치 구조로 전체 파라미터의 약 7-8%만 학습하면서도 효과적인 보정이 가능합니다.

학습 완료 후 `merge_adapter_weights` 함수로 어댑터 가중치를 원본에 병합하면, 추론 시 추가 연산 없이 사용할 수 있습니다.

---

### 3.3 Phase 4: 학습 스크립트

#### train_stage1.py

Stage 1 학습(구조 학습)을 위한 스크립트입니다.

S2R-HDR 합성 데이터셋을 사용하여 HDR 파노라마의 구조적 특징을 학습합니다. 이 단계에서는 DTAM을 적용하지 않고 기본 GAN 손실과 LPIPS 손실만 사용합니다.

Full FP32 정밀도와 Softplus 활성화를 사용하여 물리적 휘도 범위를 정확하게 표현합니다. 다만 절대 휘도 스케일은 Stage 2에서 보정합니다.

CLI 옵션으로 데이터셋 경로, 배치 크기, 학습 기간, 증강 설정 등을 조정할 수 있습니다. 학습 중간에 스냅샷이 저장되며, 학습을 중단했다가 재개할 수 있습니다.

#### train_stage2.py

Stage 2 학습(물리 보정)을 위한 스크립트입니다.

Laval 실측 데이터셋을 사용하여 물리적 휘도 정합성을 보정합니다. Stage 1 체크포인트를 로드하고 S2R-Adapter를 적용한 후, DTAM과 PU21을 활용한 물리 손실로 학습합니다.

손실 함수는 세 가지 요소로 구성됩니다:
- L_phys: DTAM 가중치 적용 물리 손실
- L_consist: Stage 1과의 구조적 일관성 손실
- L_GAN: 기본 적대적 손실

CLI 옵션으로 S2R-Adapter 설정(r1, r2, scale), DTAM 파라미터, 손실 가중치 등을 조정할 수 있습니다.

---

### 3.4 Phase 5: 검증 파이프라인

#### inference/super_resolution.py

SwinIR 기반 HDR 초해상도 모듈입니다.

512×512 이미지를 1024×1024로 업스케일합니다. HDR 이미지의 물리적 휘도를 보존하기 위해 정규화-SR-역정규화 파이프라인을 사용합니다.

SwinIR 모델이 없는 환경에서도 작동하도록 Lanczos 보간 폴백을 제공합니다. 휘도 보존 검증 함수도 포함되어 있어, 업스케일 전후의 휘도 일치 여부를 확인할 수 있습니다.

#### inference/validation_pipeline.py

학습된 모델로 원형어안 HDR 파일을 생성하는 파이프라인입니다.

전체 워크플로우:
1. Generator가 512×1024 Equirectangular HDR 생성
2. 전방 180° 영역을 512×512로 크롭
3. SwinIR로 1024×1024로 업스케일
4. Angular Fisheye 프로젝션으로 변환
5. Radiance HDR 형식으로 저장

생성된 HDR 파일에는 evalglare가 인식할 수 있도록 VIEW 헤더(-vta -vv 180 -vh 180)가 삽입됩니다. 중간 결과(파노라마, 크롭, 업스케일)를 선택적으로 저장할 수 있어 디버깅에 유용합니다.

evalglare 실행은 사용자가 직접 수행합니다.

#### metrics/physical_metrics.py

물리적 정합성 평가 메트릭을 구현합니다.

**수직 조도(Ev) 오차율**: 파노라마 전체에서 수직면에 입사하는 조도를 적분하여 계산합니다. 목표는 10% 미만입니다.

**전이 구간 RMSE**: 300~1000 cd/m² 영역의 휘도 오차입니다. DGP에 가장 큰 영향을 미치는 구간으로, 이 값이 낮을수록 DGP 계산이 정확합니다.

**DGP 등급 분류 정확도**: Imperceptible/Perceptible/Disturbing/Intolerable 4등급 분류의 일치율입니다. 등급 경계값은 CIE 표준을 따릅니다.

추가로 휘도 히스토그램 유사도, 피크 휘도 정확도 등의 보조 메트릭도 제공합니다. 평가 결과를 표 형식으로 출력하는 리포트 함수도 포함되어 있습니다.

---

## 4. 파일 구조 요약

```
Stylelight_edit/
├── train.py                          [수정] HDR 모드 설정 추가
├── train_stage1.py                   [신규] Stage 1 학습 스크립트
├── train_stage2.py                   [신규] Stage 2 학습 스크립트
│
├── training/
│   ├── dataset.py                    [수정] HDRPhysicalDataset 추가
│   ├── networks.py                   [수정] ToRGBLayerHDR 추가
│   ├── loss.py                       [수정] Physical/Consistency Loss 추가
│   ├── dtam.py                       [신규] DTAM 가중치 함수
│   ├── pu21.py                       [신규] PU21 인코딩
│   └── s2r_adapter.py                [신규] S2R-Adapter (2-브랜치 도메인 적응)
│
├── inference/
│   ├── __init__.py                   [신규]
│   ├── super_resolution.py           [신규] SwinIR 초해상도
│   └── validation_pipeline.py        [신규] 검증 파이프라인
│
├── metrics/
│   └── physical_metrics.py           [신규] 물리적 평가 메트릭
│
└── data_preprocessing/
    ├── __init__.py                   [신규]
    └── prepare_datasets.py           [신규] 데이터 전처리
```

---

## 5. 핵심 기술 요약

| 기술 | 목적 | 구현 위치 |
|------|------|-----------|
| **DTAM** | 전이 구간(300~1000 cd/m²)에 높은 학습 가중치 부여 | training/dtam.py |
| **PU21** | 물리적 휘도를 지각적으로 균일한 값으로 변환 | training/pu21.py |
| **S2R-Adapter** | Stage 1 구조 보존하면서 효율적인 물리 보정 (2-브랜치) | training/s2r_adapter.py |
| **Softplus** | 비음수 물리적 휘도 출력 보장 | training/networks.py |
| **Full FP32** | HDR 동적 범위 정밀하게 표현 | train.py |

---

## 6. 의존성

기존 StyleLight 의존성 외에 추가로 필요한 패키지:

- **lpips**: Consistency Loss 계산용 (선택, 없으면 L1 손실로 대체)
- **swinir**: 초해상도용 (선택, 없으면 Lanczos 보간으로 대체)
