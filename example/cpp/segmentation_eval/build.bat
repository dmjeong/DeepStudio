@echo off
setlocal
cd /d "%~dp0"
REM Set these to the folders you already use in your C++ solution.
REM ONNXRUNTIME_ROOT must contain include and lib; OpenCV_DIR contains OpenCVConfig.cmake.
if exist paths.bat call paths.bat
if not defined ONNXRUNTIME_ROOT (
  echo Copy paths.example.bat to paths.bat and enter your ONNX Runtime / OpenCV folders.
  exit /b 2
)
if not defined OpenCV_DIR (
  echo Set OpenCV_DIR in paths.bat.
  exit /b 2
)
cmake -S . -B build -A x64 -DONNXRUNTIME_ROOT="%ONNXRUNTIME_ROOT%" -DOpenCV_DIR="%OpenCV_DIR%"
if errorlevel 1 exit /b 1
cmake --build build --config Release --target segmentation_eval
if errorlevel 1 exit /b 1
echo Ready: build\Release\segmentation_eval.exe
echo Put model, images and masks beside the EXE. Edit evaluation.json, then run the EXE.
endlocal
