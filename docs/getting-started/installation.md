# StyleLight CUDA 환경 설정 가이드 (Windows/Linux)

이 가이드는 StyleLight 프로젝트를 Windows 및 Linux(WSL2 포함)에서 실행하기 위한 CUDA 환경 설정 방법을 설명합니다.

## 📋 필수 요구사항

### 공통
- **Python 3.8+**
- **PyTorch 2.1.0+ (CUDA 12.1 버전)**
- **NVIDIA GPU 및 드라이버**

### Windows
- **Windows 10/11**
- **Visual Studio 2019 또는 2022**
  - "C++를 사용한 데스크톱 개발" 워크로드 설치 필수

### Linux/WSL2
- **GCC 12 이하** (CUDA 12.x 호환 필수)
- **CUDA nvcc 버전이 PyTorch CUDA 버전과 일치해야 함**

---

## 🚀 빠른 시작

### 방법 1: 자동 설정 (권장)

프로젝트의 메인 Python 파일(`train.py`, `test_lighting.py`)은 **자동으로 환경을 설정**합니다.

```bash
# 바로 실행 가능!
conda activate linux_stylelight_conda  # 또는 사용하는 환경 이름
python test_lighting.py
python train.py
```

스크립트 실행 시 자동으로:
1. CUDA Toolkit 경로 탐지 (conda 환경 또는 시스템 설치)
2. 컴파일러 경로 탐지 (Windows: MSVC, Linux: GCC)
3. 필요한 환경변수 설정
4. **Linux: CCCL 헤더 경로 설정, TORCH_CUDA_ARCH_LIST 수정**

### 방법 2: 수동으로 환경 확인 및 설정

#### 1단계: 환경 진단

```bash
python setup_diagnoze.py
```

이 명령어는 다음을 확인합니다:
- PyTorch CUDA 사용 가능 여부
- CUDA Toolkit 설치 상태
- nvcc 컴파일러 (버전 호환성 포함)
- 컴파일러 (Windows: MSVC, Linux: GCC 버전)
- GPU 정보
- **Linux: CCCL 헤더 경로, TORCH_CUDA_ARCH_LIST**

#### 2단계: 환경 설정 (필요시)

```bash
python setup_env.py
```

이 명령어는:
- CUDA_HOME, CUDA_PATH 환경변수 설정
- nvcc를 PATH에 추가
- Windows: MSVC 컴파일러를 PATH에 추가
- **Linux: CPATH에 CCCL 헤더 경로 추가, TORCH_CUDA_ARCH_LIST 설정**

### 방법 3: 배치 스크립트 사용 (Windows)

```bash
setup_cuda_env.bat
```

더블클릭으로 실행 가능합니다.

---

## 🔧 상세 설정 방법

### CUDA Toolkit 설치

#### conda 환경 (권장)
```bash
# PyTorch 2.1.0이 CUDA 12.1로 빌드되었으므로 nvcc도 12.1 사용
conda install -c nvidia cuda-nvcc=12.1
conda install -c nvidia cuda-toolkit=12.1
```

**중요**: nvcc 버전과 PyTorch CUDA 버전이 일치해야 합니다!

#### 시스템 전역 설치
1. [NVIDIA CUDA Toolkit 다운로드](https://developer.nvidia.com/cuda-downloads)
2. 설치 후 환경변수 확인:
   - `CUDA_HOME`: CUDA 설치 경로
   - `PATH`: `$CUDA_HOME/bin` 포함 여부

### Visual Studio 설치 (Windows)

1. [Visual Studio 다운로드](https://visualstudio.microsoft.com/downloads/)
   - Visual Studio 2019 또는 2022 Community 버전 (무료)
2. 설치 시 **"C++를 사용한 데스크톱 개발"** 워크로드 반드시 선택
3. 설치 후 `setup_env.py`가 자동으로 컴파일러를 찾습니다

### GCC 설치 (Linux)

**중요**: CUDA 12.x는 GCC 12 이하만 지원합니다!

```bash
# conda 환경에 GCC 12 설치 (권장)
conda install -c conda-forge gxx_linux-64=12

# 또는 시스템에 설치
sudo apt install gcc-12 g++-12
```

---

## 📁 파일 설명

### 설정 스크립트

- **`setup_env.py`**: 범용 환경 설정 스크립트
  - 자동으로 CUDA와 컴파일러 경로 탐지
  - `verbose=False` 옵션으로 조용히 실행 가능
  - conda 환경, 시스템 설치 모두 지원
  - **Windows/Linux 모두 지원**

- **`setup_diagnoze.py`**: 환경 진단 스크립트
  - 현재 시스템의 CUDA 환경 상태 확인
  - 문제 발견 시 해결 방법 제시
  - **Windows/Linux 모두 지원**

- **`setup_cuda_env.bat`**: Windows 배치 스크립트
  - `setup_env.py`를 쉽게 실행
  - 더블클릭으로 실행 가능

### 메인 실행 파일

- **`train.py`**: GAN 학습 스크립트
  - 자동 환경 설정 포함 (`verbose=True`)

- **`test_lighting.py`**: 조명 테스트 스크립트
  - 자동 환경 설정 포함 (`verbose=True`)

---

## ❓ 문제 해결

### 공통 문제

#### "CUDA를 사용할 수 없습니다"

1. GPU 드라이버 확인:
   ```bash
   nvidia-smi
   ```

2. PyTorch CUDA 버전 확인:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. PyTorch를 CUDA 버전으로 재설치:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

#### "nvcc를 찾을 수 없습니다"

**해결 방법 1**: `setup_env.py` 실행
```bash
python setup_env.py
```

**해결 방법 2**: conda에 CUDA nvcc 설치
```bash
conda install -c nvidia cuda-nvcc=12.1
```

### Windows 전용 문제

#### "MSVC를 찾을 수 없습니다"

**해결 방법 1**: `setup_env.py` 실행 (자동 탐지)
```bash
python setup_env.py
```

**해결 방법 2**: Visual Studio 설치
1. Visual Studio 2019 또는 2022 설치
2. **"C++를 사용한 데스크톱 개발"** 워크로드 선택

**해결 방법 3**: x64 Native Tools Command Prompt 사용
1. 시작 메뉴 → "x64 Native Tools Command Prompt for VS" 검색
2. 해당 프롬프트에서 Python 스크립트 실행

### Linux/WSL2 전용 문제

#### "unsupported GNU version! gcc versions later than 12 are not supported"

GCC 버전이 너무 높습니다. CUDA 12.x는 GCC 12 이하만 지원합니다.

**해결 방법**:
```bash
conda install -c conda-forge gxx_linux-64=12
```

#### "thrust/complex.h: No such file or directory" 또는 "nv/target: No such file or directory"

CCCL 헤더 경로가 설정되지 않았습니다.

**해결 방법 1**: `setup_env.py` 실행 (자동 설정)
```bash
python setup_env.py
```

**해결 방법 2**: 수동 설정
```bash
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include/cccl:$CPATH
```

#### "Unknown CUDA arch (10.0) or GPU not supported"

TORCH_CUDA_ARCH_LIST에 PyTorch 2.x가 지원하지 않는 아키텍처가 포함되어 있습니다.

**해결 방법 1**: `setup_env.py` 실행 (자동 수정)
```bash
python setup_env.py
```

**해결 방법 2**: 수동 설정
```bash
export TORCH_CUDA_ARCH_LIST="8.6;8.9"
```

#### nvcc 버전과 PyTorch CUDA 버전 불일치

nvcc 버전이 PyTorch가 빌드된 CUDA 버전과 다릅니다.

**해결 방법**:
```bash
# PyTorch 2.1.0은 CUDA 12.1로 빌드됨
conda install -c nvidia cuda-nvcc=12.1
```

---

## 💡 팁

### 환경 설정 메시지 숨기기

환경 설정 메시지를 숨기려면:

```python
from setup_env import setup_cuda_environment
setup_cuda_environment(verbose=False)  # 조용히 실행
```

### 여러 conda 환경 사용

각 conda 환경마다 독립적으로 CUDA Toolkit을 설치할 수 있습니다:

```bash
# 환경 1
conda activate env1
conda install -c nvidia cuda-toolkit=11.8

# 환경 2
conda activate env2
conda install -c nvidia cuda-toolkit=12.1
```

`setup_env.py`는 자동으로 현재 활성화된 환경의 CUDA를 사용합니다.

### CUDA 커널 캐시 삭제

문제가 계속되면 캐시를 삭제하고 다시 시도하세요:

```bash
rm -rf ~/.cache/torch_extensions/
```

### VSCode에서 실행

VSCode 사용 시:
1. `.vscode/launch.json` 설정 불필요
2. 터미널에서 바로 실행:
   ```bash
   python test_lighting.py
   ```
3. 환경이 자동으로 설정됨

---

## 📞 지원

문제가 계속되면:

1. **`setup_diagnoze.py`** 실행하여 전체 환경 상태 확인
   ```bash
   python setup_diagnoze.py
   ```

2. 출력 결과를 확인하여 누락된 구성요소 설치

3. **`setup_env.py`** 실행하여 자동 설정 시도
   ```bash
   python setup_env.py
   ```

---

## 🎯 다른 컴퓨터에서 사용하기

1. 프로젝트 폴더 전체를 복사
2. 새 컴퓨터에서 Python 환경 설정 (conda)
3. 필요한 패키지 설치 (README.md 참조)
4. **Linux인 경우 추가 설치**:
   ```bash
   conda install -c conda-forge gxx_linux-64=12
   conda install -c nvidia cuda-nvcc=12.1
   ```
5. 바로 실행:
   ```bash
   python test_lighting.py
   ```

환경 설정이 자동으로 처리됩니다!

---

## 📝 주요 변경사항

이 프로젝트는 다음 파일들이 자동 환경 설정을 포함하도록 수정되었습니다:

- `train.py` - 실행 시 자동 환경 설정
- `test_lighting.py` - 실행 시 자동 환경 설정

추가된 설정 파일:
- `setup_env.py` - 범용 환경 설정 스크립트 **(Windows/Linux 지원)**
- `setup_diagnoze.py` - 환경 진단 스크립트 **(Windows/Linux 지원)**
- `setup_cuda_env.bat` - Windows 배치 스크립트

이제 **Windows와 Linux(WSL2 포함) 모두에서** 프로젝트를 복사하고 바로 실행할 수 있습니다!
