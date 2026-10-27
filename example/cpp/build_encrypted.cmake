cmake_minimum_required(VERSION 3.14)
# Run by build_encrypted.bat. Keep build files outside Program Files and avoid
# sharing one CMake cache between a checkout, an installer, and different SDKs.
get_filename_component(source "${CMAKE_CURRENT_LIST_DIR}/with_evision" REALPATH)
get_filename_component(sdk "${ONNXRUNTIME_ROOT}" REALPATH)
string(SHA256 identity "${source}|${sdk}|$ENV{VSINSTALLDIR}|$ENV{CMAKE_GENERATOR}|$ENV{CMAKE_GENERATOR_TOOLSET}")
string(SUBSTRING "${identity}" 0 20 identity)
file(TO_CMAKE_PATH "$ENV{LOCALAPPDATA}" local_app_data)
if(NOT local_app_data)
    message(FATAL_ERROR "LOCALAPPDATA is not set. Run build_encrypted.bat on Windows.")
endif()
set(build "${local_app_data}/DeepVisionStudio/encrypted-bw8-build/${identity}")
message(STATUS "Build folder: ${build}")
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${source}" -B "${build}" -A x64
    "-DONNXRUNTIME_ROOT=${sdk}" RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "CMake configuration failed: ${status}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${build}" --config Release RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "C++ build failed: ${status}")
endif()
# WORKING_DIRECTORY also supports CTest versions before --test-dir was added.
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" -C Release --output-on-failure
    WORKING_DIRECTORY "${build}" RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Example tests failed: ${status}")
endif()
message(STATUS "Example: ${build}/Release/evision_encrypted_example.exe")
