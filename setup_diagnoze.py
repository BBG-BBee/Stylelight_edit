"""
CUDA 환경 진단 스크립트 (Windows/Linux 지원)
현재 시스템의 CUDA 환경 상태를 확인합니다.
"""
import torch
import os
import subprocess
import re
import sys
import platform

def is_linux():
    """Linux 환경인지 확인"""
    return platform.system() == 'Linux'

print("=" * 60)
print(f"CUDA 환경 진단 ({'Linux' if is_linux() else 'Windows'})")
print("=" * 60)

# 시스템 정보
print(f"\n[시스템 정보]")
print(f"  - OS: {platform.system()} {platform.release()}")
print(f"  - Python: {sys.version.split()[0]}")
print(f"  - Python 경로: {sys.executable}")
if hasattr(sys, 'prefix'):
    print(f"  - Conda 환경: {sys.prefix}")

# 1. PyTorch CUDA 버전
print(f"\n[1] PyTorch 정보")
print(f"  - PyTorch 버전: {torch.__version__}")
print(f"  - CUDA 사용 가능: {torch.cuda.is_available()}")
pytorch_cuda_version = None
if hasattr(torch.version, 'cuda') and torch.version.cuda:
    pytorch_cuda_version = torch.version.cuda
    print(f"  - PyTorch CUDA 버전: {pytorch_cuda_version}")
else:
    print(f"  - PyTorch CUDA 버전: 없음 (CPU 버전)")

if torch.cuda.is_available():
    print(f"  - GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        cap = torch.cuda.get_device_capability(i)
        print(f"    Compute Capability: {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})")

# 2. CUDA Toolkit 경로
print(f"\n[2] CUDA Toolkit")
cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
print(f"  - CUDA_HOME: {cuda_home if cuda_home else '설정되지 않음'}")

# 3. nvcc 확인
nvcc_found = False
nvcc_version = None
nvcc_name = 'nvcc' if is_linux() else 'nvcc.exe'

try:
    result = subprocess.run([nvcc_name, '--version'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"  - nvcc 버전:")
        lines = result.stdout.strip().splitlines()
        version_line = next((line for line in lines if "release" in line.lower()),
                          lines[-1] if lines else "정보 없음")
        print(f"    {version_line.strip()}")
        nvcc_found = True
        # 버전 추출
        match = re.search(r'release\s+(\d+\.\d+)', version_line)
        if match:
            nvcc_version = match.group(1)
except FileNotFoundError:
    print(f"  - nvcc: 찾을 수 없음 (PATH에 없음)")
except Exception as e:
    print(f"  - nvcc: 확인 실패 ({e})")

if cuda_home and not nvcc_found:
    nvcc_path = os.path.join(cuda_home, 'bin', nvcc_name)
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
                match = re.search(r'release\s+(\d+\.\d+)', version_line)
                if match:
                    nvcc_version = match.group(1)
        except Exception:
            pass

# nvcc와 PyTorch CUDA 버전 비교
if nvcc_version and pytorch_cuda_version:
    nvcc_major = nvcc_version.split('.')[0]
    pytorch_major = pytorch_cuda_version.split('.')[0]
    if nvcc_major != pytorch_major:
        print(f"  - [경고] nvcc 버전({nvcc_version})과 PyTorch CUDA 버전({pytorch_cuda_version}) 불일치!")
        print(f"    CUDA 커널 JIT 컴파일 실패 가능성 있음")
    else:
        print(f"  - [OK] nvcc와 PyTorch CUDA 버전 호환됨")

# 4. nvidia-smi (GPU 드라이버)
print(f"\n[3] NVIDIA Driver")
try:
    result = subprocess.run(['nvidia-smi'],
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        match = re.search(r'CUDA Version:\s*(\d+\.\d+)', result.stdout)
        driver_version = match.group(1) if match else "정보 없음"
        print(f"  - 지원 CUDA 버전: {driver_version}")

        gpu_match = re.search(r'NVIDIA\s+([^\|]+)\s+\|', result.stdout)
        if gpu_match:
            print(f"  - GPU: {gpu_match.group(1).strip()}")
    else:
        print(f"  - nvidia-smi 실행 불가 (NVIDIA GPU 없음?)")
except FileNotFoundError:
    print(f"  - nvidia-smi: 찾을 수 없음 (NVIDIA Driver 미설치)")
except Exception as e:
    print(f"  - nvidia-smi 실행 불가: {e}")

# 5. 컴파일러 확인 (Linux: GCC, Windows: MSVC)
if is_linux():
    print(f"\n[4] 컴파일러 (GCC)")
    gcc_version = None
    try:
        # conda 환경의 GCC 먼저 확인
        conda_gcc = os.path.join(sys.prefix, 'bin', 'x86_64-conda-linux-gnu-cc')
        if os.path.exists(conda_gcc):
            result = subprocess.run([conda_gcc, '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                first_line = result.stdout.strip().split('\n')[0]
                print(f"  - Conda GCC: {first_line}")
                match = re.search(r'(\d+)\.\d+\.\d+', first_line)
                if match:
                    gcc_version = int(match.group(1))
        else:
            # 시스템 GCC
            result = subprocess.run(['gcc', '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                first_line = result.stdout.strip().split('\n')[0]
                print(f"  - GCC: {first_line}")
                match = re.search(r'(\d+)\.\d+\.\d+', first_line)
                if match:
                    gcc_version = int(match.group(1))
    except FileNotFoundError:
        print(f"  - GCC: 찾을 수 없음")
    except Exception as e:
        print(f"  - GCC 확인 실패: {e}")

    # GCC 버전 호환성 확인 (CUDA 12.x는 GCC 12 이하 필요)
    if gcc_version:
        if gcc_version > 12:
            print(f"  - [경고] GCC {gcc_version}은 CUDA 12.x와 호환되지 않습니다!")
            print(f"    해결: conda install -c conda-forge gxx_linux-64=12")
        else:
            print(f"  - [OK] GCC 버전 호환됨")
else:
    print(f"\n[4] 컴파일러 (MSVC)")
    try:
        result = subprocess.run(['cl'],
                              capture_output=True, timeout=5)
        output = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ''
        if 'Microsoft' in output:
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

# 6. TORCH_CUDA_ARCH_LIST 확인
print(f"\n[5] CUDA 아키텍처 설정")
arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '')
if arch_list:
    print(f"  - TORCH_CUDA_ARCH_LIST: {arch_list}")
    # 호환되지 않는 아키텍처 확인
    incompatible = ['10.0', '10.3', '11.0', '12.0', '12.1']
    found_incompatible = [a for a in incompatible if a in arch_list]
    if found_incompatible:
        print(f"  - [경고] PyTorch 2.x와 호환되지 않는 아키텍처 포함: {found_incompatible}")
        print(f"    setup_env.py가 자동으로 수정합니다")
else:
    print(f"  - TORCH_CUDA_ARCH_LIST: 설정되지 않음 (setup_env.py가 자동 설정)")

# 7. Linux CCCL 경로 확인
if is_linux() and hasattr(sys, 'prefix'):
    print(f"\n[6] CCCL 헤더 경로 (Linux)")
    cccl_path = os.path.join(sys.prefix, 'targets', 'x86_64-linux', 'include', 'cccl')
    if os.path.exists(cccl_path):
        print(f"  - CCCL 경로: {cccl_path}")
        thrust_exists = os.path.exists(os.path.join(cccl_path, 'thrust'))
        nv_exists = os.path.exists(os.path.join(cccl_path, 'nv'))
        print(f"  - thrust 헤더: {'있음' if thrust_exists else '없음'}")
        print(f"  - nv 헤더: {'있음' if nv_exists else '없음'}")

        cpath = os.environ.get('CPATH', '')
        if cccl_path in cpath:
            print(f"  - [OK] CPATH에 CCCL 경로 포함됨")
        else:
            print(f"  - [경고] CPATH에 CCCL 경로 없음")
            print(f"    setup_env.py가 자동으로 설정합니다")
    else:
        print(f"  - CCCL 경로: 찾을 수 없음")
        print(f"    cuda-cccl 패키지가 설치되어 있는지 확인하세요")

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

if nvcc_version and pytorch_cuda_version:
    nvcc_major = nvcc_version.split('.')[0]
    pytorch_major = pytorch_cuda_version.split('.')[0]
    if nvcc_major != pytorch_major:
        issues.append(f"nvcc({nvcc_version})와 PyTorch CUDA({pytorch_cuda_version}) 버전 불일치")

if is_linux():
    # GCC 버전 확인
    try:
        conda_gcc = os.path.join(sys.prefix, 'bin', 'x86_64-conda-linux-gnu-cc')
        gcc_cmd = conda_gcc if os.path.exists(conda_gcc) else 'gcc'
        result = subprocess.run([gcc_cmd, '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            match = re.search(r'(\d+)\.\d+\.\d+', result.stdout)
            if match and int(match.group(1)) > 12:
                issues.append("GCC 버전이 12보다 높음 (CUDA 12.x 호환 문제)")
    except:
        pass
else:
    # MSVC 확인
    try:
        subprocess.run(['cl'], capture_output=True, timeout=2)
    except FileNotFoundError:
        issues.append("MSVC 컴파일러를 찾을 수 없습니다")
    except:
        pass

if issues:
    print("\n[경고] 다음 문제가 발견되었습니다:")
    for issue in issues:
        print(f"  - {issue}")
    print("\n해결 방법:")
    print("  1. setup_env.py를 실행하여 환경을 자동 설정하세요:")
    print(f"     python setup_env.py")
    if is_linux():
        print("\n  Linux 추가 해결 방법:")
        print("  - GCC 버전 문제: conda install -c conda-forge gxx_linux-64=12")
        print("  - nvcc 버전 문제: conda install -c nvidia cuda-nvcc=12.1")
    print("\n  2. 또는 train.py/test_lighting.py에 이미 환경 설정이 포함되어 있습니다.")
else:
    print("\n모든 환경이 정상적으로 설정되었습니다!")

print("=" * 60)
