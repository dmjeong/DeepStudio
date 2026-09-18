@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" goto active_venv
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" goto active_conda
if exist ".venv\Scripts\python.exe" goto local_venv
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 goto py311
python -c "import sys" >nul 2>&1
if not errorlevel 1 goto python_path
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 goto py3
echo Python not found. Install Python 3.11 64-bit and try again.
echo https://www.python.org/downloads/windows/
pause
exit /b 1

:active_venv
"%VIRTUAL_ENV%\Scripts\python.exe" start_web.py %*
set "STUDIO_EXIT_CODE=%ERRORLEVEL%"
goto finish
:active_conda
"%CONDA_PREFIX%\python.exe" start_web.py %*
set "STUDIO_EXIT_CODE=%ERRORLEVEL%"
goto finish
:local_venv
".venv\Scripts\python.exe" start_web.py %*
set "STUDIO_EXIT_CODE=%ERRORLEVEL%"
goto finish
:py311
py -3.11 start_web.py %*
set "STUDIO_EXIT_CODE=%ERRORLEVEL%"
goto finish
:python_path
python start_web.py %*
set "STUDIO_EXIT_CODE=%ERRORLEVEL%"
goto finish
:py3
py -3 start_web.py %*
set "STUDIO_EXIT_CODE=%ERRORLEVEL%"
:finish
if not "%STUDIO_EXIT_CODE%"=="0" pause
exit /b %STUDIO_EXIT_CODE%
