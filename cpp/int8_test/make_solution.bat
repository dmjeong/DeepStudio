@echo off
rem ============================================================================
rem  INT8 test solution generator (Visual Studio 2022, x64)
rem
rem    make_solution.bat <ONNXRUNTIME_ROOT> <OpenCV_DIR> [vcpkg_root]
rem
rem    ONNXRUNTIME_ROOT : folder of onnxruntime-win-x64-<ver> (has include\ and lib\)
rem    OpenCV_DIR       : folder with OpenCVConfig.cmake (e.g. C:\opencv\build)
rem    vcpkg_root       : optional, for nlohmann-json (vcpkg install nlohmann-json:x64-windows)
rem
rem  Output: build\Int8Test.sln  (startup project: int8_compare)
rem  (ASCII-only comments on purpose: cmd.exe misreads UTF-8 text in .bat files)
rem ============================================================================
setlocal
if "%~2"=="" (
    echo usage: make_solution.bat ^<ONNXRUNTIME_ROOT^> ^<OpenCV_DIR^> [vcpkg_root]
    exit /b 2
)
set TOOLCHAIN=
if not "%~3"=="" set TOOLCHAIN=-DCMAKE_TOOLCHAIN_FILE="%~3\scripts\buildsystems\vcpkg.cmake"

cmake -S "%~dp0." -B "%~dp0build" -G "Visual Studio 17 2022" -A x64 ^
      -DONNXRUNTIME_ROOT="%~1" -DOpenCV_DIR="%~2" %TOOLCHAIN%
if errorlevel 1 exit /b 1

echo.
echo Solution: %~dp0build\Int8Test.sln
echo Build Release, then run: build\Release\int8_compare.exe fp32.json int8.json images_folder 8 5 20
endlocal
