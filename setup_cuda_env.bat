@echo off
REM ========================================
REM CUDA 환경 설정 배치 스크립트 (Windows)
REM 범용적으로 작동하도록 설계됨
REM ========================================

echo ========================================
echo CUDA 환경 자동 설정
echo ========================================
echo.

REM Python을 통해 setup_env.py 실행 (가장 간단하고 범용적)
echo Python setup_env.py를 실행하여 환경을 자동 설정합니다...
echo.

python setup_env.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 환경 설정 완료!
    echo ========================================
    echo.
    echo 이제 Python 스크립트를 실행할 수 있습니다:
    echo   python train.py
    echo   python test_lighting.py
    echo.
    echo 참고: train.py와 test_lighting.py는 자동으로 환경을 설정합니다.
    echo       이 배치 파일 없이도 바로 실행 가능합니다.
    echo.
) else (
    echo.
    echo [오류] setup_env.py 실행 실패
    echo Python이 설치되어 있고 PATH에 등록되어 있는지 확인하세요.
    echo.
)

pause
