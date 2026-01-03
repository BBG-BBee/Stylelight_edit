"""
CUDA 환경 진단 스크립트 (Windows 전용)
현재 시스템의 CUDA 환경 상태를 확인합니다.
"""
import torch
import os
import subprocess
import re
import sys

print("=" * 60)
print("CUDA 환경 진단 (Windows)")
print("=" * 60)

# 시스템 정보
print(f"\n[시스템 정보]")
print(f"  - OS: Windows")
print(f"  - Python: {sys.version.split()[0]}")
print(f"  - Python 경로: {sys.executable}")

# 1. PyTorch CUDA 버전
print(f"\n[1] PyTorch 정보")
print(f"  - PyTorch 버전: {torch.__version__}")
print(f"  - CUDA 사용 가능: {torch.cuda.is_available()}")
if hasattr(torch.version, 'cuda') and torch.version.cuda:
    print(f"  - PyTorch CUDA 버전: {torch.version.cuda}")
else:
    print(f"  - PyTorch CUDA 버전: 없음 (CPU 버전)")

if torch.cuda.is_available():
    print(f"  - GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

# 2. CUDA Toolkit 경로
print(f"\n[2] CUDA Toolkit")
cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
print(f"  - CUDA_HOME: {cuda_home if cuda_home else '설정되지 않음'}")

# 3. nvcc 확인
nvcc_found = False

try:
    result = subprocess.run(['nvcc.exe', '--version'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"  - nvcc 버전:")
        # 출력 형식이 다를 수 있으므로 안전하게 파싱
        lines = result.stdout.strip().splitlines()
        version_line = next((line for line in lines if "release" in line.lower()),
                          lines[-1] if lines else "정보 없음")
        print(f"    {version_line.strip()}")
        nvcc_found = True
except FileNotFoundError:
    print(f"  - nvcc: 찾을 수 없음 (PATH에 없음)")
except Exception as e:
    print(f"  - nvcc: 확인 실패 ({e})")

if cuda_home and not nvcc_found:
    # CUDA_HOME이 설정되어 있지만 nvcc를 못 찾는 경우
    nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc.exe')
    if os.path.exists(nvcc_path):
        try:
            result = subprocess.run([nvcc_path, '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                version_line = next((line for line in lines if "release" in line.lower()),
                                  lines[-1] if lines else "정보 없음")
                print(f"  - nvcc (CUDA_HOME에서 발견):")
                print(f"    {version_line.strip()}")
                print(f"  - 경고: nvcc가 PATH에 없습니다. setup_env.py를 사용하세요.")
                nvcc_found = True
        except Exception:
            pass

# 4. nvidia-smi (GPU 드라이버)
print(f"\n[3] NVIDIA Driver")
try:
    result = subprocess.run(['nvidia-smi'],
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        # 정규표현식으로 안전하게 버전 추출
        match = re.search(r'CUDA Version:\s*(\d+\.\d+)', result.stdout)
        driver_version = match.group(1) if match else "정보 없음"
        print(f"  - 지원 CUDA 버전: {driver_version}")

        # GPU 정보도 추출
        gpu_match = re.search(r'NVIDIA\s+([^\|]+)\s+\|', result.stdout)
        if gpu_match:
            print(f"  - GPU: {gpu_match.group(1).strip()}")
    else:
        print(f"  - nvidia-smi 실행 불가 (NVIDIA GPU 없음?)")
except FileNotFoundError:
    print(f"  - nvidia-smi: 찾을 수 없음 (NVIDIA Driver 미설치)")
except Exception as e:
    print(f"  - nvidia-smi 실행 불가: {e}")

# 5. Visual Studio MSVC
print(f"\n[4] 컴파일러 (MSVC)")
try:
    result = subprocess.run(['cl'],
                          capture_output=True, timeout=5)
    output = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ''
    if 'Microsoft' in output:
        # 버전 정보 추출
        version_match = re.search(r'Version\s+([\d.]+)', output)
        version = version_match.group(1) if version_match else "알 수 없음"
        print(f"  - MSVC: 설치됨 (버전 {version})")
    else:
        print(f"  - MSVC: 찾을 수 없음")
except FileNotFoundError:
    print(f"  - MSVC: 찾을 수 없음 (PATH에 없음)")
    print(f"    해결 방법:")
    print(f"    1. Visual Studio 2019/2022 설치")
    print(f"    2. 'C++를 사용한 데스크톱 개발' 워크로드 설치")
    print(f"    3. setup_env.py를 사용하여 자동 설정")
except Exception as e:
    print(f"  - MSVC 확인 실패: {e}")

print("\n" + "=" * 60)

# 진단 결과 요약
print("\n[진단 결과 요약]")
issues = []

if not torch.cuda.is_available():
    issues.append("PyTorch가 CUDA를 사용할 수 없습니다")

if not cuda_home:
    issues.append("CUDA_HOME 환경변수가 설정되지 않았습니다")

if not nvcc_found:
    issues.append("nvcc를 찾을 수 없습니다")

# MSVC 확인
try:
    subprocess.run(['cl'], capture_output=True, timeout=2)
except FileNotFoundError:
    issues.append("MSVC 컴파일러를 찾을 수 없습니다")
except:
    pass

if issues:
    print("\n경고: 다음 문제가 발견되었습니다:")
    for issue in issues:
        print(f"  - {issue}")
    print("\n해결 방법:")
    print("  1. setup_env.py를 실행하여 환경을 자동 설정하세요:")
    print(f"     python setup_env.py")
    print("  2. 또는 train.py/test_lighting.py에 이미 환경 설정이 포함되어 있습니다.")
else:
    print("\n모든 환경이 정상적으로 설정되었습니다!")

print("=" * 60)
