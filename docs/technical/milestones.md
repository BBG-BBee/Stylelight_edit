# 프로젝트 마일스톤

> 물리적 정합성 기반 HDR 뷰 확장 및 DGP 시뮬레이션

---

## 1. 프로젝트 정의 및 목표

본 프로젝트는 제한된 시야(NFoV, 약 63°)를 가진 단일 HDR 이미지(23mm 렌즈 등)를 입력받아, 물리적으로 정확한 **180° 반구형(Hemispherical) HDR 파노라마**를 생성하는 것을 목표로 한다.

### 1.1 핵심 목표

단순한 시각적 확장(Outpainting)을 넘어, **주광 눈부심 확률(DGP)** 계산의 정확성을 담보할 수 있는 **수직 조도($E_v$) 정합성**을 확보한다.

### 1.2 해결 과제

기존 모델들이 간과했던 **전이 구간(300 ~ 1,000 $cd/m^2$)**의 휘도 정보를 정밀하게 복원하여, 시뮬레이션 시 발생하는 **'거짓 양성(False Positive)'** 눈부심 오류를 제거한다.

---

## 2. 핵심 R&D 전략

### 2.1 모델 아키텍처: HDR-Native StyleGAN2

| 항목 | 내용 |
|------|------|
| **기반 모델** | StyleLight (GAN) 기반 - LDM의 VAE 정보 손실(Blurring)을 피하기 위해 선택 |
| **활성화 함수** | Tanh → **Softplus**로 교체 (물리적 휘도의 비음수성 및 무한 동적 범위 지원) |
| **해상도** | $512 \times 1024$ (Equirectangular) Native 출력 |

### 2.2 학습 전략: 2-Stage & S2R-Adapter

| 단계 | 데이터 | 목표 | DTAM |
|------|--------|------|------|
| **Stage 1** | S2R-HDR (합성) | 기하학적 구조 확장 능력 배양 | OFF |
| **Stage 2** | Laval Photometric (실측) | 물리적 도메인 적응 | ON |

**S2R-Adapter 구조:**
- 2-브랜치 도메인 적응 구조 (공유 브랜치 + 전송 브랜치)
- 전송 브랜치: Transfer1(r1=1) + Transfer2(r2=128)

### 2.3 핵심 기술: S2R-Adapter, DTAM 및 Full FP32

| 기술 | 설명 |
|------|------|
| **S2R-Adapter** | 2-브랜치 도메인 적응 구조로 Stage 1 지식 보존하면서 물리 보정 (r1=1, r2=128) |
| **DTAM** | 휘도에 따라 가중치를 차등 부여. $T_{onset}=300$, $T_{peak}=1,000$ $cd/m^2$ |
| **Full FP32** | RTX 5090 (32GB+ VRAM) 환경에서 모든 연산을 float32로 강제 |

---

## 3. 단계별 세부 마일스톤

### Milestone 1: 인프라 구축 및 데이터 파이프라인

**목표:** 물리적 정합성을 지원하는 하드웨어 및 소프트웨어 환경 구축

**세부 활동:**

1. **하드웨어 확정**
   - NVIDIA RTX 5090 확보 및 CUDA 환경 설정
   - Full FP32 메모리 할당 테스트

2. **데이터셋 준비**
   - Stage 1용: S2R-HDR (24k장, 합성)
   - Stage 2용: Laval Photometric Indoor HDR (1.7k장, $cd/m^2$ 보정)

3. **데이터 로더 개발**
   - pytorch360convert 커스터마이징
   - 톤매핑/감마 보정 없이 선형 Float32 텐서 직접 로드

---

### Milestone 2: 모델 아키텍처 수정 및 초기화

**목표:** HDR 데이터를 손실 없이 처리할 수 있는 모델 구조 변경

**세부 활동:**

1. **출력층 교체**
   - StyleGAN2 Generator의 ToRGB 레이어 활성화 함수를 **Softplus**로 변경

2. **FP32 강제화**
   - amp (Automatic Mixed Precision) 관련 코드 전면 제거
   - `torch.set_default_dtype(torch.float32)` 적용

3. **판별자 수정**
   - 180° HDR 이미지를 입력받아 리얼리티를 판별하는 $D_{180\_HDR}$ 구현

---

### Milestone 3: Stage 1 파인튜닝 (구조 학습)

**목표:** S2R-HDR 데이터셋을 활용하여 '잘린 이미지를 자연스럽게 확장'하는 능력 확보

**주요 전략:**

| 항목 | 설정 |
|------|------|
| **데이터** | S2R-HDR |
| **마스킹** | DTAM 비활성화 (OFF) - 합성 데이터의 광원 물리량이 실제와 다를 수 있음 |
| **손실 함수** | 기본 GAN 손실 + 지각적 손실 (LPIPS) |
| **결과물** | Checkpoint-Stage1 (구조적 이해도가 높은 기본 모델) |

---

### Milestone 4: Stage 2 파인튜닝 (물리적 정합성)

**목표:** Laval 실측 데이터를 통해 실제 빛의 세기와 분포($E_v$)를 학습

**주요 전략:**

| 항목 | 설정 |
|------|------|
| **데이터** | Laval Photometric Indoor HDR |
| **S2R-Adapter** | Stage 1의 $G$ 가중치 동결, 어댑터만 학습 |
| **DTAM** | 활성화 (ON) - $T_{onset}=300$, $T_{peak}=1000$ |

**손실 함수:**

$$\mathcal{L}_{Total} = \mathcal{L}_{Phys}(DTAM) + \lambda \mathcal{L}_{Consist} + \mathcal{L}_{GAN}$$

| 손실 | 역할 |
|------|------|
| $\mathcal{L}_{Phys}$ | DTAM 가중치가 적용된 물리적 L1 손실 |
| $\mathcal{L}_{Consist}$ | Stage 1 모델과의 구조적 차이를 제한하는 일관성 손실 |
| $\mathcal{L}_{GAN}$ | 기본 적대적 손실 |

**결과물:** Checkpoint-Final (물리적으로 보정된 최종 모델)

---

### Milestone 5: 검증 파이프라인 및 추론

**목표:** 해상도 손실 없이 evalglare 분석이 가능한 고품질 결과물 생성

**실행 프로세스 (6-Step Workflow):**

| 단계 | 작업 | 설명 |
|------|------|------|
| 1 | **생성** | $512 \times 1024$ Equirectangular HDR 생성 (FP32) |
| 2 | **크롭** | 전방 180° 영역 추출 ($512 \times 512$) |
| 3 | **초해상도** | $1024 \times 1024$로 업스케일링 (SwinIR 등) |
| 4 | **변환** | Angular Fisheye (-vta) 포맷으로 투영 변환 |
| 5 | **헤더 주입** | Radiance 헤더 (`VIEW= -vta -vv 180 -vh 180`) 삽입 |
| 6 | **DGP 산출** | evalglare 실행 |

---

### Milestone 6: 최종 평가

**목표:** 정량적/정성적 평가를 통한 모델 성능 검증

#### 정량적 평가

| 지표 | 설명 | 목표 |
|------|------|------|
| **$\Delta E_v$** | 수직 조도 오차율 | < 10% (가장 중요) |
| **$RMSE_{trans}$** | 전이 구간 (300~1,000 $cd/m^2$) 복원 정확도 | 최소화 |
| **DGP Class Accuracy** | 눈부심 등급 (Imperceptible ~ Intolerable) 분류 정확도 | 최대화 |
| **PU21-PSNR** | 지각적 품질 지표 | 기존 대비 향상 |
| **HDR-VDP-3 Q-Score** | HDR 품질 점수 (외부 도구 hdrvdp3 CLI 사용) | 기존 대비 향상 |

#### 정성적 평가

- Blender를 이용한 **IBL (Image-Based Lighting)** 렌더링 테스트
- 반사/그림자 품질 비교

---

## 4. 요약 타임라인

```
M1: 인프라 구축     →  M2: 모델 수정     →  M3: Stage 1 학습
        ↓                    ↓                    ↓
   데이터 준비          Softplus 적용         구조 학습
   FP32 환경 설정       FP32 강제화          DTAM OFF
        ↓                    ↓                    ↓
                         M4: Stage 2 학습
                              ↓
                    S2R-Adapter + DTAM ON
                         물리 보정
                              ↓
                    M5: 검증 파이프라인
                              ↓
                    M6: 최종 평가 및 검증
```
