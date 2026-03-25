# StyleLight HDR 파노라마 생성

**물리적 정합성 기반 HDR 파노라마 생성 모델 문서**

---

## 프로젝트 개요

23mm 렌즈(약 63° 화각)의 **단일 NFoV HDR 이미지**로부터 **물리적으로 정확한 180° 반구형 HDR 파노라마**를 생성합니다.

### 핵심 목표

| 목표 | 설명 | 지표 |
|------|------|------|
| **DGP 정합성** | 주광 눈부심 확률 계산에 필요한 절대 휘도 정합성 확보 | - |
| **전이 구간 복원** | 300~1000 cd/m² 구간 정밀 복원 | RMSE_trans |
| **수직 조도 정확도** | Ev 오차율 최소화 | < 10% |

### 주요 기술

| 기술 | 설명 |
|------|------|
| **2-Stage 학습** | S2R-HDR(구조) → Laval(물리 보정) |
| **S2R-Adapter** | 2-브랜치 도메인 적응 구조 (r1=1, r2=128) |
| **DTAM** | 이중 임계값 적응형 마스킹 |
| **Full FP32** | 높은 동적 범위를 위한 정밀도 유지 |

---

## 문서 안내

### 시작하기

| 문서 | 설명 |
|------|------|
| [프로젝트 소개](getting-started/overview.md) | StyleLight 배경 및 현재 프로젝트 목표 |
| [환경 설정](getting-started/installation.md) | CUDA 12.1 + GCC 12 환경 구성 (Linux/WSL2) |

### 학습 가이드

| 문서 | 설명 |
|------|------|
| [학습 실행](training/guide.md) | Stage 1/2 학습 명령어 및 데이터 준비 |

### 구현

| 문서 | 설명 |
|------|------|
| [구현 계획](implementation/plan.md) | 마일스톤 M1~M6 정의 |
| [구현 결과](implementation/result.md) | 수정/생성 파일 목록 및 변경 내용 |

### 기술 문서

| 문서 | 설명 |
|------|------|
| [S2R-Adapter](technical/s2r-adapter.md) | 2-브랜치 도메인 적응, 스케일 학습, TTA |
| [DTAM 이론](technical/dtam.md) | 이중 임계값 적응형 마스킹 수식 및 원리 |
| [마일스톤](technical/milestones.md) | 프로젝트 목표 및 단계별 정의 |
| [통합 보고서](technical/integrated-report.md) | 이론, 아키텍처, 로드맵 통합 (Single Source of Truth) |

---

## 빠른 시작

### 1. 환경 설정

```bash
conda create -n StyleLight_conda python=3.8
conda activate StyleLight_conda

# PyTorch + CUDA 12.1
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2. Stage 1 학습 (구조 학습)

```bash
python train_stage1.py \
    --data ./data/s2r_hdr_processed \
    --outdir ./training-runs-stage1 \
    --gpus 1 \
    --batch 4 \
    --kimg 5000
```

### 3. Stage 2 학습 (물리 보정)

```bash
python train_stage2.py \
    --stage1_ckpt ./training-runs-stage1/network-snapshot.pkl \
    --data ./data/laval_processed \
    --outdir ./training-runs-stage2 \
    --adapter_r1 1 \
    --adapter_r2 128 \
    --dtam_onset 300 \
    --dtam_peak 1000
```

---

## 파일 구조

```
Stylelight_edit/
├── train_stage1.py           # Stage 1 학습 스크립트
├── train_stage2.py           # Stage 2 학습 스크립트
├── training/
│   ├── dtam.py               # DTAM 가중치 함수
│   ├── pu21.py               # PU21 인코딩
│   ├── s2r_adapter.py        # S2R-Adapter (2-브랜치, 스케일 조절, TTA 함수)
│   ├── tta_augment.py        # TTA 증강 모듈 (ExpAug, WBAug, FlipAug, PermAug)
│   └── coaches/
│       └── my_coach.py       # MyCoach (train_with_tta 포함)
├── inference/
│   └── validation_pipeline.py
└── metrics/
    └── physical_metrics.py   # 물리적 평가 메트릭
```

---

## 핵심 기술 요약

| 기술 | 목적 | 구현 위치 |
|------|------|-----------|
| **DTAM** | 전이 구간(300~1000 cd/m²)에 높은 학습 가중치 부여 | `training/dtam.py` |
| **PU21** | 물리적 휘도를 지각적으로 균일한 값으로 변환 | `training/pu21.py` |
| **S2R-Adapter** | Stage 1 구조 보존하면서 효율적인 물리 보정 | `training/s2r_adapter.py` |
| **스케일 학습** | 파인튜닝 시 scale1, scale2 자동 최적화 | `training/s2r_adapter.py` |
| **TTA** | 불확실성 기반 동적 스케일 조절 (선택적) | `training/tta_augment.py` |
| **Softplus** | 비음수 물리적 휘도 출력 보장 | `training/networks.py` |
| **Full FP32** | HDR 동적 범위 정밀하게 표현 | `train.py` |
