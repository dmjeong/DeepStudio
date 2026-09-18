@echo off
setlocal
cd /d "%~dp0"
REM ═══════════════════════════════════════════════════
REM  Deep Vision Studio — Windows EXE 빌드 스크립트
REM ═══════════════════════════════════════════════════
REM
REM  사용법:
REM    build.bat         — EXE 빌드
REM    build.bat clean   — 빌드 산출물 삭제
REM
REM  필수 패키지:
REM    pip install pyinstaller
REM
REM  PyTorch 선행 설치 (택 1):
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

REM 1. PyInstaller 설치 확인
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
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

pause
