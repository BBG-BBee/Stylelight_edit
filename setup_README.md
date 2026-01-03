# StyleLight CUDA 환경 설정 가이드 (Windows)

이 가이드는 StyleLight 프로젝트를 Windows에서 실행하기 위한 CUDA 환경 설정 방법을 설명합니다.

## 📋 필수 요구사항

- **Windows 10/11**
- **Python 3.8+**
- **PyTorch (CUDA 버전)**
- **NVIDIA GPU 및 드라이버**
- **Visual Studio 2019 또는 2022**
  - "C++를 사용한 데스크톱 개발" 워크로드 설치 필수

---

## 🚀 빠른 시작

### 방법 1: 자동 설정 (권장) ⭐

프로젝트의 메인 Python 파일(`train.py`, `test_lighting.py`)은 **자동으로 환경을 설정**합니다.

```bash
# 바로 실행 가능!
python test_lighting.py
python train.py
```

스크립트 실행 시 자동으로:
1. CUDA Toolkit 경로 탐지 (conda 환경 또는 시스템 설치)
2. MSVC 컴파일러 경로 탐지
3. 필요한 환경변수 설정

### 방법 2: 수동으로 환경 확인 및 설정

#### 1단계: 환경 진단

```bash
python setup_diagnoze.py
```

이 명령어는 다음을 확인합니다:
- PyTorch CUDA 사용 가능 여부
- CUDA Toolkit 설치 상태
- nvcc 컴파일러
- MSVC 컴파일러
- GPU 정보

#### 2단계: 환경 설정 (필요시)

```bash
python setup_env.py
```

이 명령어는:
- CUDA_HOME, CUDA_PATH 환경변수 설정
- nvcc를 PATH에 추가
- MSVC 컴파일러를 PATH에 추가

### 방법 3: 배치 스크립트 사용

```bash
setup_cuda_env.bat
```

더블클릭으로 실행 가능합니다.

---

## 🔧 상세 설정 방법

### CUDA Toolkit 설치

#### conda 환경 (권장)
```bash
conda install -c nvidia cuda-toolkit
```

자동으로 현재 conda 환경에 CUDA Toolkit이 설치되며, `setup_env.py`가 자동으로 탐지합니다.

#### 시스템 전역 설치
1. [NVIDIA CUDA Toolkit 다운로드](https://developer.nvidia.com/cuda-downloads)
2. 설치 후 환경변수 확인:
   - `CUDA_HOME`: CUDA 설치 경로
   - `PATH`: `%CUDA_HOME%\bin` 포함 여부

### Visual Studio 설치

1. [Visual Studio 다운로드](https://visualstudio.microsoft.com/downloads/)
   - Visual Studio 2019 또는 2022 Community 버전 (무료)
2. 설치 시 **"C++를 사용한 데스크톱 개발"** 워크로드 반드시 선택
3. 설치 후 `setup_env.py`가 자동으로 컴파일러를 찾습니다

---

## 📁 파일 설명

### 설정 스크립트

- **`setup_env.py`**: 범용 환경 설정 스크립트
  - 자동으로 CUDA와 MSVC 경로 탐지
  - `verbose=False` 옵션으로 조용히 실행 가능
  - conda 환경, 시스템 설치 모두 지원

- **`setup_diagnoze.py`**: 환경 진단 스크립트
  - 현재 시스템의 CUDA 환경 상태 확인
  - 문제 발견 시 해결 방법 제시

- **`setup_cuda_env.bat`**: Windows 배치 스크립트
  - `setup_env.py`를 쉽게 실행
  - 더블클릭으로 실행 가능

### 메인 실행 파일

- **`train.py`**: GAN 학습 스크립트
  - 자동 환경 설정 포함 (`verbose=False`)

- **`test_lighting.py`**: 조명 테스트 스크립트
  - 자동 환경 설정 포함 (`verbose=False`)

---

## ❓ 문제 해결

### "CUDA를 사용할 수 없습니다"

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

### "nvcc를 찾을 수 없습니다"

**해결 방법 1**: `setup_env.py` 실행
```bash
python setup_env.py
```

**해결 방법 2**: conda에 CUDA Toolkit 설치
```bash
conda install -c nvidia cuda-toolkit
```

**해결 방법 3**: 시스템 환경변수 수동 설정
1. 시작 메뉴 → "환경 변수" 검색
2. 시스템 환경 변수에서:
   - `CUDA_HOME`: CUDA 설치 경로 설정
   - `Path`에 `%CUDA_HOME%\bin` 추가

### "MSVC를 찾을 수 없습니다"

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

---

## 💡 팁

### 환경 설정 메시지 숨기기

`train.py`와 `test_lighting.py`는 이미 `verbose=False`로 설정되어 있어 환경 설정 메시지가 표시되지 않습니다.

만약 다른 스크립트에서 사용하려면:

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
2. 새 컴퓨터에서 Python 환경 설정 (conda 또는 venv)
3. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   # 또는
   conda env create -f environment.yml
   ```
4. 바로 실행:
   ```bash
   python test_lighting.py
   ```

환경 설정이 자동으로 처리됩니다! 🎉

---

## 📝 주요 변경사항

이 프로젝트는 다음 파일들이 자동 환경 설정을 포함하도록 수정되었습니다:

- ✅ `train.py` - 실행 시 자동 환경 설정
- ✅ `test_lighting.py` - 실행 시 자동 환경 설정

추가된 설정 파일:
- 📄 `setup_env.py` - 범용 환경 설정 스크립트
- 📄 `setup_diagnoze.py` - 환경 진단 스크립트
- 📄 `setup_cuda_env.bat` - Windows 배치 스크립트

이제 **어떤 Windows 컴퓨터에서도** 프로젝트를 복사하고 바로 실행할 수 있습니다!
