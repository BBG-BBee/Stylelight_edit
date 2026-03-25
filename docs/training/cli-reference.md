# CLI 레퍼런스

학습 및 추론에 사용되는 명령어를 단계별로 정리합니다.

---

## 1. 데이터 전처리

### S2R-HDR 전처리

```bash
python data_preprocessing/prepare_datasets.py \
    --dataset s2r_hdr \
    --input_dir ./data/s2r_hdr_raw \
    --output_dir ./data/s2r_hdr_processed \
    --generate-json
```

### Laval 전처리

```bash
python data_preprocessing/prepare_datasets.py \
    --dataset laval \
    --input_dir ./data/laval_raw \
    --output_dir ./data/laval_processed \
    --generate-json
```

### 데이터셋 유효성 검사

```bash
python data_preprocessing/prepare_datasets.py \
    --dataset validate \
    --input_dir ./data/s2r_hdr_processed
```

### 전처리 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--dataset` | 데이터셋 유형 (`s2r_hdr`, `laval`, `validate`) | **필수** |
| `--input_dir` | 원본 HDR 파일 디렉토리 | **필수** |
| `--output_dir` | 출력 디렉토리 | `{input_dir}_processed` |
| `--height` | 목표 높이 | 512 |
| `--width` | 목표 너비 | 1024 |
| `--no-exposure` | 노출 보정 비적용 (Laval 전용) | False |
| `--generate-json` | dataset.json 생성 | False |

---

## 2. Stage 1 학습 (구조 학습)

### 기본 실행

```bash
python train_stage1.py \
    --data ./data/s2r_hdr_processed \
    --outdir ./training-runs-stage1 \
    --gpus 1 \
    --batch 4 \
    --kimg 10000
```

### Gradient Accumulation (VRAM 절약)

`--batch-gpu`로 GPU당 마이크로배치 크기를 지정하면, `batch / (batch_gpu × gpus)` 스텝 동안 그래디언트를 누적합니다.

```bash
# batch=16, batch_gpu=2 → 8스텝 누적
python train_stage1.py \
    --data ./data/s2r_hdr_processed \
    --outdir ./training-runs-stage1 \
    --batch 16 \
    --batch-gpu 2
```

### 체크포인트에서 재개

```bash
python train_stage1.py \
    --data ./data/s2r_hdr_processed \
    --outdir ./training-runs-stage1 \
    --resume ./training-runs-stage1/00000-stage1/network-snapshot-005000.pkl
```

### Stage 1 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--data` | S2R-HDR 데이터셋 경로 | **필수** |
| `--outdir` | 출력 디렉토리 | **필수** |
| `--gpus` | GPU 개수 | 1 |
| `--batch` | 배치 크기 | GPU에 맞게 자동 |
| `--batch-gpu` | GPU당 마이크로배치 크기 | `batch // gpus` |
| `--kimg` | 학습 기간 (kimg) | 10000 |
| `--gamma` | R1 gamma 오버라이드 | 자동 계산 |
| `--snap` | 스냅샷 간격 (ticks) | 100 |
| `--metrics` | 메트릭 목록 또는 "none" | fid50k_full |
| `--seed` | 랜덤 시드 | 0 |
| `--subset` | N개 이미지만 사용 | 전체 |
| `--mirror/--no-mirror` | x-flip 증강 | True |
| `--aug` | 증강 모드 (`noaug`, `ada`) | ada |
| `--target` | ADA 목표값 | 0.6 |
| `--augpipe` | 증강 파이프라인 | bgc |
| `--resume` | 체크포인트에서 재개 | - |
| `--workers` | DataLoader 워커 수 | 4 |
| `--nobench` | cuDNN 벤치마킹 비활성화 | False |
| `--allow-tf32` | TF32 허용 | False |
| `-n, --dry-run` | 설정만 출력하고 종료 | False |

---

## 3. Stage 2 학습 (물리 보정)

### 기본 실행

```bash
python train_stage2.py \
    --stage1_ckpt ./training-runs-stage1/00000-stage1/network-snapshot-005000.pkl \
    --data ./data/laval_processed \
    --outdir ./training-runs-stage2
```

### S2R-Adapter / DTAM 조정

```bash
python train_stage2.py \
    --stage1_ckpt ./stage1.pkl \
    --data ./data/laval_processed \
    --outdir ./training-runs-stage2 \
    --adapter_r1 1 \
    --adapter_r2 128 \
    --dtam_t_onset 300 \
    --dtam_t_peak 1000
```

### Gradient Accumulation

```bash
# batch=16, batch_gpu=2 → 8스텝 누적
python train_stage2.py \
    --stage1_ckpt ./stage1.pkl \
    --data ./data/laval_processed \
    --outdir ./training-runs-stage2 \
    --batch 16 \
    --batch-gpu 2
```

### 손실 가중치 조정

```bash
python train_stage2.py \
    --stage1_ckpt ./stage1.pkl \
    --data ./data/laval_processed \
    --outdir ./training-runs-stage2 \
    --lambda_phys 1.0 \
    --lambda_consist 0.5 \
    --lambda_gan 1.0
```

### Stage 2 옵션

**필수 옵션**

| 옵션 | 설명 |
|------|------|
| `--stage1_ckpt` | Stage 1 체크포인트 경로 |
| `--data` | Laval Photometric 데이터셋 경로 |
| `--outdir` | 출력 디렉토리 |

**S2R-Adapter 설정**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--adapter_r1` | 전송 브랜치 1 rank | 1 |
| `--adapter_r2` | 전송 브랜치 2 rank | 128 |
| `--freeze_shared/--no-freeze_shared` | 공유 브랜치 동결 | True |
| `--init_scale1` | scale1 초기값 | 1.0 |
| `--init_scale2` | scale2 초기값 | 1.0 |

**DTAM 설정**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--dtam_t_onset` | T_onset (cd/m²) | 300 |
| `--dtam_t_peak` | T_peak (cd/m²) | 1000 |
| `--dtam_alpha` | 가중치 증폭 계수 | 9.0 |
| `--dtam_gamma` | 곡률 | 2.0 |

**손실 가중치**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--lambda_phys` | Physical Loss 가중치 | 1.0 |
| `--lambda_consist` | Consistency Loss 가중치 | 0.5 |
| `--lambda_gan` | GAN Loss 가중치 | 1.0 |

**학습/성능 옵션**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--gpus` | GPU 개수 | 1 |
| `--batch` | 배치 크기 | GPU에 맞게 자동 |
| `--batch-gpu` | GPU당 마이크로배치 크기 | `batch // gpus` |
| `--kimg` | 학습 기간 (kimg) | 5000 |
| `--gamma` | R1 gamma 오버라이드 | 자동 계산 |
| `--snap` | 스냅샷 간격 (ticks) | 100 |
| `--seed` | 랜덤 시드 | 0 |
| `--aug` | 증강 모드 (`noaug`, `ada`) | ada |
| `--target` | ADA 목표값 | 0.6 |
| `-n, --dry-run` | 설정만 출력하고 종료 | False |

---

## 4. 추론

### 기본 실행

```bash
python inference/validation_pipeline.py \
    --checkpoint ./training-runs-stage2/network-snapshot.pkl \
    --output ./results/output.hdr
```

### SwinIR 초해상도 사용

```bash
python inference/validation_pipeline.py \
    --checkpoint ./training-runs-stage2/network-snapshot.pkl \
    --output ./results/output.hdr \
    --sr_model ./models/swinir.pth
```

### 중간 결과 저장 (디버깅)

```bash
python inference/validation_pipeline.py \
    --checkpoint ./training-runs-stage2/network-snapshot.pkl \
    --output ./results/output.hdr \
    --save_intermediate \
    --seed 42
```

### LFOV 입력 이미지 사용

```bash
python inference/validation_pipeline.py \
    --checkpoint ./training-runs-stage2/network-snapshot.pkl \
    --output ./results/output.hdr \
    --input ./photos/lfov_image.hdr \
    --vfov 63.0
```

### 추론 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--checkpoint` | 체크포인트 경로 | **필수** |
| `--output` | 출력 HDR 파일 경로 | output_fisheye.hdr |
| `--sr_model` | SwinIR 모델 경로 (없으면 Lanczos) | None |
| `--truncation` | Truncation psi | 0.7 |
| `--seed` | 랜덤 시드 (재현용) | None |
| `--save_intermediate` | 중간 결과 저장 | False |
| `--device` | 디바이스 | cuda |
| `--input` | LFOV 입력 이미지 경로 | None |
| `--vfov` | 입력 이미지 수직 화각 (°) | 63.0 |
| `--az` | 방위각 오프셋 (°) | 0.0 |
| `--el` | 고도 오프셋 (°) | 0.0 |

---

## 5. Dry Run (설정 확인)

학습 전에 `-n` 플래그로 설정을 확인할 수 있습니다. 실제 학습은 수행하지 않습니다.

```bash
# Stage 1 설정 확인
python train_stage1.py \
    --data ./data/s2r_hdr_processed \
    --outdir ./training-runs-stage1 \
    --batch 16 --batch-gpu 2 \
    -n

# Stage 2 설정 확인
python train_stage2.py \
    --stage1_ckpt ./stage1.pkl \
    --data ./data/laval_processed \
    --outdir ./training-runs-stage2 \
    --batch 16 --batch-gpu 2 \
    -n
```
