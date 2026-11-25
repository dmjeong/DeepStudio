@echo off
REM Copy this file to paths.bat, then set your actual existing SDK paths.
set "ONNXRUNTIME_ROOT=C:\libraries\onnxruntime-win-x64-1.29.0"
set "OpenCV_DIR=C:\libraries\opencv\build"
REM Add the matching OpenCV DLL directory when running the EXE.
set "PATH=C:\libraries\opencv\build\x64\vc16\bin;%PATH%"
