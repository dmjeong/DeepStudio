@echo off
setlocal
cd /d "%~dp0"
REM No EXE rebuild, GPU runtime replacement or model-pack import is needed.
REM No arguments: all built-in weights. segment: six segmentation weights.
REM check: read-only validation. offline: register existing official files.
set "DVS_WEIGHT_OPTIONS="
set "DVS_WEIGHT_READONLY="
for %%A in (%*) do (
    if /I "%%~A"=="segment" (set "DVS_WEIGHT_OPTIONS=--group segment") else (
    if /I "%%~A"=="check" (set "DVS_WEIGHT_READONLY=--check") else (
    if /I "%%~A"=="offline" (set "DVS_WEIGHT_READONLY=--register-existing") else (
        echo Unknown option: %%~A. Use segment, check or offline.
        goto :failed
    )))
)
python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Use the same Python environment as the app.
    goto :failed
)
if not defined DVS_WEIGHT_READONLY (
    python -c "import truststore, certifi" >nul 2>&1
    if errorlevel 1 (
        python -m pip install truststore==0.10.4 certifi
        if errorlevel 1 goto :failed
    )
)
REM The Python tool resolves the same asset directory as the running app,
REM including DVS_BUILTIN_ASSETS_DIR. Do not hardcode a relative output here.
python ..\python\prepare_builtin_assets.py %DVS_WEIGHT_OPTIONS% %DVS_WEIGHT_READONLY%
if errorlevel 1 goto :failed
echo Model weights are ready. You can run python main.py.
if not defined DVS_WEIGHTS_NONINTERACTIVE pause
exit /b 0
:failed
echo Weight preparation/check failed. See the model-specific errors above.
echo Completed weights are preserved. After fixing the cause, run this file again.
if not defined DVS_WEIGHTS_NONINTERACTIVE pause
exit /b 1
