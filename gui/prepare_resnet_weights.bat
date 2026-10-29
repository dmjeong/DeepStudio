@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
REM Double-click: download/verify ResNet18, ResNet34 and ResNet50 only.
REM prepare_resnet_weights.bat offline: register existing official files, no download.
set "DVS_REGISTER_FLAG="
if /I "%~1"=="offline" set "DVS_REGISTER_FLAG=--register-existing"
python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.11 or later is required. Use the Python environment used for build.bat.
    pause
    exit /b 1
)
python ..\python\prepare_builtin_assets.py --output builtin_assets --only resnet18,resnet34,resnet50 %DVS_REGISTER_FLAG%
if errorlevel 1 (
    echo.
    echo Some weights could not be prepared. See each [failed] message above.
    echo Successfully verified files remain usable. No EXE build was started.
    pause
    exit /b 1
)
echo.
echo ResNet weights are ready in %CD%\builtin_assets\torchvision
echo Restart Deep Vision Studio to use them. No EXE rebuild is needed for source mode.
pause
exit /b 0
