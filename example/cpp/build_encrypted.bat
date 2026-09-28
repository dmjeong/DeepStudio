@echo off
setlocal
cd /d "%~dp0"
echo Encrypted BW8 example - C++17, VS2017 15.9 or newer
echo Enter the ONNX Runtime folder containing include and lib.
set "DVS_ORT_PATH=%ONNXRUNTIME_ROOT%"
set /p "DVS_ORT_PATH=ONNX Runtime folder [%DVS_ORT_PATH%]: "
if not exist "%DVS_ORT_PATH%\include\onnxruntime_cxx_api.h" (
    echo ERROR: include\onnxruntime_cxx_api.h was not found.
    pause
    exit /b 1
)
where cmake >nul 2>nul
if errorlevel 1 (
    echo ERROR: Install the CMake component in Visual Studio, then run from its Developer Command Prompt.
    pause
    exit /b 1
)
cmake "-DONNXRUNTIME_ROOT=%DVS_ORT_PATH%" -P build_encrypted.cmake
if errorlevel 1 goto failed
echo Edit with_evision\encrypted.cpp to use your own model. Never use the example key for real models.
pause
exit /b 0
:failed
echo Build or execution failed. See the error above.
pause
exit /b 1
