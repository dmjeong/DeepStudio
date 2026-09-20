@echo off
setlocal
cd /d "%~dp0"
REM ═══════════════════════════════════════════════════
REM  Deep Vision Studio — Windows EXE 빌드 스크립트
REM ═══════════════════════════════════════════════════
REM
REM  사용법:
REM    build.bat         — EXE 빌드
REM    build.bat installer — EXE와 간단한 Windows 설치파일 빌드
REM    build.bat clean   — 빌드 산출물 삭제
REM
REM  이 스크립트는 requirements.txt의 데스크톱 런타임을 설치한 뒤
REM  PyInstaller로 EXE를 만듭니다. 이미 설치한 PyTorch(CPU/CUDA)는
REM  requirements.txt의 버전 조건을 만족하면 그대로 사용합니다.
REM
REM  GPU PyTorch를 선택 설치하려면 build.bat 실행 전에 다음 중 하나를 실행:
REM    [GPU] pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
REM    [CPU] pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
REM
REM  빌드 결과:
REM    dist\DeepVisionStudio\DeepVisionStudio.exe
REM
REM ═══════════════════════════════════════════════════

echo ================================================
echo  Deep Vision Studio - EXE Builder
echo ================================================

if "%1"=="clean" (
    echo Cleaning build artifacts...
    python build_exe.py clean
    goto :eof
)

REM 1. Python 및 GUI 런타임 확인. pip 대신 같은 Python의 -m pip을 사용해
REM    PyInstaller가 다른 가상환경의 PySide6를 참조하는 문제를 막는다.
python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.11+ is required and was not found on PATH.
    exit /b 1
)

echo Installing/verifying desktop build dependencies...
REM SAM2's optional CUDA extension needs a locally installed matching nvcc.
REM The PyTorch implementation remains functional without it.
set SAM2_BUILD_CUDA=0
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo DEPENDENCY INSTALL FAILED!
    exit /b 1
)

REM 1.5. 모든 기본 모델의 사전학습 가중치를 빌드 시 한 번만 받아 EXE에 포함한다.
REM 앱 실행 중 Settings/학습/추론 화면에서 모델 파일을 추가로 받지 않는다.
echo Downloading/verifying packaged basic-model weights...
python ..\python\prepare_builtin_assets.py --output builtin_assets
if errorlevel 1 (
    echo BASIC MODEL WEIGHT PREPARATION FAILED!
    exit /b 1
)

REM 2. 빌드 실행
echo Building EXE...
python build_exe.py
if errorlevel 1 (
    echo BUILD FAILED!
    exit /b 1
)

REM 3. 결과 확인
if exist "dist\DeepVisionStudio\DeepVisionStudio.exe" (
    echo.
    echo ================================================
    echo  BUILD SUCCESSFUL!
    echo  dist\DeepVisionStudio\DeepVisionStudio.exe
    echo ================================================
    echo.
    echo Run dist\DeepVisionStudio\DeepVisionStudio.exe to start the app.
) else (
    echo.
    echo  BUILD FAILED!
    echo.
    exit /b 1
)

if /I "%1"=="installer" (
    echo Building minimal Windows installer (app plus C++ and C# examples)...
    powershell -NoProfile -ExecutionPolicy Bypass -File "..\packaging\windows\build_simple_installer.ps1" ^
        -AppRoot "%CD%\dist\DeepVisionStudio" -OutputDirectory "%CD%\..\release"
    if errorlevel 1 (
        echo INSTALLER BUILD FAILED!
        exit /b 1
    )
)

pause
