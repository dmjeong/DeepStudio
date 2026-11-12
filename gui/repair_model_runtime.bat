@echo off
setlocal
cd /d "%~dp0"
REM Optional first argument: the Python path printed by the app error.
set "DVS_MODEL_PYTHON=python"
if not "%~1"=="" set "DVS_MODEL_PYTHON=%~1"
"%DVS_MODEL_PYTHON%" repair_model_runtime.py
set "DVS_REPAIR_RESULT=%ERRORLEVEL%"
if not defined DVS_WEIGHTS_NONINTERACTIVE pause
exit /b %DVS_REPAIR_RESULT%
