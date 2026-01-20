"""
CUDA 환경 변수를 Python 런타임에서 설정하는 스크립트 (Windows/Linux 지원)
train.py나 test_lighting.py 실행 전에 import하거나 직접 실행하세요

자동으로 CUDA와 컴파일러 경로를 탐지하여 설정합니다.
"""
import os
import sys
import subprocess
import platform

def is_linux():
    """Linux 환경인지 확인"""
    return platform.system() == 'Linux'

def find_cuda_home():
    """CUDA Toolkit 경로 자동 탐지 (Windows/Linux)"""

    nvcc_name = 'nvcc' if is_linux() else 'nvcc.exe'

    # 1. 환경변수에서 먼저 확인
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home and os.path.exists(cuda_home):
        nvcc_path = os.path.join(cuda_home, 'bin', nvcc_name)
        if os.path.exists(nvcc_path):
            return cuda_home

    # 2. conda 환경에서 찾기
    if hasattr(sys, 'prefix'):
        conda_nvcc = os.path.join(sys.prefix, 'bin', nvcc_name)
        if os.path.exists(conda_nvcc):
            return sys.prefix

    if is_linux():
        # 3a. Linux 표준 경로 탐색
        common_paths = [
            '/usr/local/cuda',
            '/usr/cuda',
        ]
        for base_path in common_paths:
            if os.path.exists(base_path):
                nvcc_path = os.path.join(base_path, 'bin', 'nvcc')
                if os.path.exists(nvcc_path):
                    return base_path
    else:
        # 3b. Windows 표준 경로 탐색
        common_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
        ]

        for base_path in common_paths:
            if os.path.exists(base_path):
                # 버전별로 정렬하여 최신 버전 찾기
                versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if versions:
                    versions.sort(reverse=True)
                    cuda_path = os.path.join(base_path, versions[0])
                    nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc.exe')
                    if os.path.exists(nvcc_path):
                        return cuda_path

    return None

def find_msvc():
    """Visual Studio MSVC 컴파일러 경로 자동 탐지 (Windows)"""

    # Visual Studio 설치 경로 탐색
    vs_base_paths = [
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio",
    ]

    # 가능한 버전들 (VS 18은 특수 케이스, 일반적으로는 2022, 2019 등)
    versions = ['18', '2022', '2019', '2017']
    editions = ['Community', 'Professional', 'Enterprise', 'BuildTools']

    found_paths = []

    for base_path in vs_base_paths:
        if not os.path.exists(base_path):
            continue

        for version in versions:
            version_path = os.path.join(base_path, version)
            if not os.path.exists(version_path):
                continue

            for edition in editions:
                msvc_tools_path = os.path.join(version_path, edition, 'VC', 'Tools', 'MSVC')
                if os.path.exists(msvc_tools_path):
                    # MSVC 버전들 찾기
                    msvc_versions = [d for d in os.listdir(msvc_tools_path)
                                    if os.path.isdir(os.path.join(msvc_tools_path, d))]

                    for msvc_ver in msvc_versions:
                        cl_exe = os.path.join(msvc_tools_path, msvc_ver, 'bin', 'Hostx64', 'x64', 'cl.exe')
                        if os.path.exists(cl_exe):
                            found_paths.append({
                                'version': msvc_ver,
                                'bin_path': os.path.dirname(cl_exe),
                                'vs_version': version,
                                'vs_edition': edition
                            })

    # 가장 최신 버전 반환
    if found_paths:
        found_paths.sort(key=lambda x: x['version'], reverse=True)
        return found_paths[0]

    return None

def setup_linux_cuda_paths(cuda_home, verbose=True):
    """Linux에서 CUDA CCCL 헤더 경로 설정 (thrust, cub, nv 포함)"""

    # conda 환경의 CCCL 경로 확인
    cccl_path = os.path.join(cuda_home, 'targets', 'x86_64-linux', 'include', 'cccl')
    if os.path.exists(cccl_path):
        current_cpath = os.environ.get('CPATH', '')
        if cccl_path not in current_cpath:
            os.environ['CPATH'] = f"{cccl_path}:{current_cpath}" if current_cpath else cccl_path
        if verbose:
            print(f"[OK] CCCL 경로 설정: {cccl_path}")
        return True
    return False

def setup_torch_cuda_arch(verbose=True):
    """PyTorch CUDA 아키텍처 설정 (RTX 40 시리즈 호환)"""

    # 기본적으로 RTX 30/40 시리즈를 위한 아키텍처 설정
    # sm_86: RTX 30xx, sm_89: RTX 40xx
    # TORCH_CUDA_ARCH_LIST 환경변수가 설정되지 않았거나 호환되지 않는 값이 있으면 수정
    current_arch = os.environ.get('TORCH_CUDA_ARCH_LIST', '')

    # 10.0 이상의 아키텍처가 있으면 PyTorch 2.x와 호환되지 않음
    if '10.0' in current_arch or '10.3' in current_arch or '11.0' in current_arch or '12.0' in current_arch:
        # 호환되는 아키텍처만 설정
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6;8.9'
        if verbose:
            print(f"[OK] TORCH_CUDA_ARCH_LIST 수정: 8.6;8.9")
        return True
    elif not current_arch:
        # 설정되어 있지 않으면 기본값 설정
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6;8.9'
        if verbose:
            print(f"[OK] TORCH_CUDA_ARCH_LIST 설정: 8.6;8.9")
        return True

    return False

def setup_cuda_environment(verbose=True):
    """CUDA 및 컴파일러 환경 변수 설정 (Windows/Linux)"""

    # OpenMP 중복 로드 오류 방지
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    platform_name = "Linux" if is_linux() else "Windows"

    if verbose:
        print("=" * 60)
        print(f"CUDA 환경 자동 설정 ({platform_name})")
        print("=" * 60)

    # CUDA 경로 찾기 및 설정
    cuda_home = find_cuda_home()
    msvc_info = None
    path_sep = ':' if is_linux() else ';'
    nvcc_name = 'nvcc' if is_linux() else 'nvcc.exe'

    if cuda_home:
        os.environ['CUDA_HOME'] = cuda_home
        os.environ['CUDA_PATH'] = cuda_home

        # PATH에 CUDA bin 추가
        cuda_bin = os.path.join(cuda_home, 'bin')
        if cuda_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = f"{cuda_bin}{path_sep}{os.environ.get('PATH', '')}"

        if verbose:
            print(f"CUDA_HOME: {cuda_home}")

            # nvcc 버전 확인
            try:
                result = subprocess.run([os.path.join(cuda_bin, nvcc_name), '--version'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
                    if version_line:
                        print(f"nvcc: {version_line[0].strip()}")
                print("[OK] nvcc 사용 가능")
            except Exception as e:
                print(f"[WARNING] nvcc 확인 실패: {e}")

        # Linux에서 CCCL 경로 설정
        if is_linux():
            setup_linux_cuda_paths(cuda_home, verbose)
            setup_torch_cuda_arch(verbose)
    else:
        if verbose:
            print("[WARNING] CUDA Toolkit을 찾을 수 없습니다")
            print("  - conda 환경에 CUDA가 설치되어 있는지 확인하세요")
            print("  - 또는 CUDA_HOME 환경변수를 수동으로 설정하세요")

    # Windows에서만 MSVC 찾기
    if not is_linux():
        msvc_info = find_msvc()

        if msvc_info:
            if msvc_info['bin_path'] not in os.environ.get('PATH', ''):
                os.environ['PATH'] = f"{msvc_info['bin_path']};{os.environ.get('PATH', '')}"

            if verbose:
                print(f"[OK] MSVC 찾음: {msvc_info['version']} (VS {msvc_info['vs_version']} {msvc_info['vs_edition']})")
        else:
            if verbose:
                print("[WARNING] MSVC를 찾을 수 없습니다")
                print("  해결 방법:")
                print("  1. Visual Studio 2019 또는 2022 설치")
                print("  2. 'C++를 사용한 데스크톱 개발' 워크로드 설치")
                print("  3. 또는 'x64 Native Tools Command Prompt'에서 실행")

    if verbose:
        print("=" * 60)
        print()

    return cuda_home is not None, msvc_info is not None if not is_linux() else True

if __name__ == "__main__":
    cuda_ok, msvc_ok = setup_cuda_environment()

    # 진단 스크립트도 실행
    print("\n진단 결과:")
    print("-" * 60)

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"PyTorch CUDA 버전: {torch.version.cuda}")

        if torch.cuda.is_available():
            print(f"GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("[ERROR] PyTorch가 설치되어 있지 않습니다")
    except Exception as e:
        print(f"[ERROR] PyTorch 확인 중 오류: {e}")

    print("-" * 60)

    # 결과 요약
    if cuda_ok and msvc_ok:
        print("\n[SUCCESS] 모든 환경이 정상적으로 설정되었습니다!")
    else:
        print("\n[WARNING] 일부 환경 설정이 완료되지 않았습니다.")
        print("위의 경고 메시지를 확인하세요.")
