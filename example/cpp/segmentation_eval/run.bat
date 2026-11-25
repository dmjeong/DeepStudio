@echo off
setlocal
cd /d "%~dp0"
if exist paths.bat call paths.bat
build\Release\segmentation_eval.exe
exit /b %errorlevel%
