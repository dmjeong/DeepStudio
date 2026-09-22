@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
where py >nul 2>nul
if not errorlevel 1 goto use_py
where python >nul 2>nul
if not errorlevel 1 goto use_python
echo Python 3.8+ is required. Install Python or run setup_vs2017.py with your existing Python.
pause
exit /b 1

:use_py
py -3 "%~dp0setup_vs2017.py" %*
goto finished

:use_python
python "%~dp0setup_vs2017.py" %*

:finished
set "setup_result=%errorlevel%"
echo.
pause
exit /b %setup_result%
