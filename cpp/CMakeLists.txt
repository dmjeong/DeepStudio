# ═══════════════════════════════════════════════════════
#  Vision C++ 추론 엔진 - CMake 빌드 설정
# ═══════════════════════════════════════════════════════
#
#  빌드 방법:
#    mkdir build && cd build
#    cmake .. -DCMAKE_BUILD_TYPE=Release
#    make -j$(nproc)
#
#  의존성:
#    - OpenCV 4.x: sudo apt install libopencv-dev
#    - ONNX Runtime: https://github.com/microsoft/onnxruntime/releases
#      다운로드 후 ONNXRUNTIME_ROOT 설정
#
#  Windows (Visual Studio):
#    cmake .. -G "Visual Studio 17 2022" -A x64
#    cmake --build . --config Release
#
# ═══════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.14)
option(VISION_BUILD_DEMO "Build the optional highgui desktop demo" ON)
option(VISION_WITH_OPENVINO "Enable OpenVINO FP32 CPU classification" OFF)

project(VisionInference
    VERSION 5.7
    DESCRIPTION "Vision ONNX Runtime CPU 추론 엔진"
    LANGUAGES CXX
)

# C++17 필수 (std::filesystem 사용)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── 빌드 타입 기본값: Release ─────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "빌드 타입" FORCE)
endif()

# Release 최적화 플래그
# Toolchain Release defaults apply (MSVC /O2, GCC/Clang -O3).

# Install with apt install nlohmann-json3-dev or vcpkg install nlohmann-json:x64-windows.
find_package(nlohmann_json 3.2.0 CONFIG REQUIRED)

# ── OpenCV 찾기 ──────────────────────────────────────
set(VISION_OPENCV_COMPONENTS core imgproc imgcodecs)
if(VISION_BUILD_DEMO)
    list(APPEND VISION_OPENCV_COMPONENTS highgui)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${VISION_OPENCV_COMPONENTS})
find_package(Threads REQUIRED)
if(VISION_WITH_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()
message(STATUS "OpenCV 버전: ${OpenCV_VERSION}")
message(STATUS "OpenCV 라이브러리: ${OpenCV_LIBS}")

# ── ONNX Runtime 찾기 ────────────────────────────────
# 방법 1: ONNXRUNTIME_ROOT 환경변수 또는 CMake 변수
# 방법 2: 시스템 경로에서 자동 탐색
if(NOT DEFINED ONNXRUNTIME_ROOT)
    # 기본 경로 후보
    set(ONNXRUNTIME_SEARCH_PATHS
        "/usr/local"
        "/opt/onnxruntime"
        "$ENV{HOME}/onnxruntime"
        "C:/onnxruntime"
        "$ENV{ONNXRUNTIME_ROOT}"
    )
    foreach(search_path ${ONNXRUNTIME_SEARCH_PATHS})
        if(EXISTS "${search_path}/include/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_ROOT "${search_path}")
            break()
        endif()
        if(EXISTS "${search_path}/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_ROOT "${search_path}")
            break()
        endif()
    endforeach()
endif()

if(DEFINED ONNXRUNTIME_ROOT)
    message(STATUS "ONNX Runtime 경로: ${ONNXRUNTIME_ROOT}")
    # 헤더 경로 탐색
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
            "${ONNXRUNTIME_ROOT}/include"
            "${ONNXRUNTIME_ROOT}/include/onnxruntime"
            "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
    )
    # 라이브러리 탐색
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS
            "${ONNXRUNTIME_ROOT}/lib"
            "${ONNXRUNTIME_ROOT}/lib64"
    )
else()
    # 시스템 경로에서 탐색
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATH_SUFFIXES onnxruntime/core/session
    )
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime)
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime을 찾을 수 없습니다!\n"
        "해결 방법:\n"
        "  1. 다운로드: https://github.com/microsoft/onnxruntime/releases\n"
        "  2. cmake 실행 시 경로 지정:\n"
        "     cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime\n"
        "  3. 또는 환경변수 설정:\n"
        "     export ONNXRUNTIME_ROOT=/path/to/onnxruntime"
    )
endif()

message(STATUS "ONNX Runtime 헤더: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime 라이브러리: ${ONNXRUNTIME_LIB}")

if(MSVC AND MSVC_VERSION LESS 1920)
    include("${CMAKE_CURRENT_SOURCE_DIR}/include/ort_vs2017.cmake")
    dvs_ort_vs2017_headers("${ONNXRUNTIME_INCLUDE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ort-v141-include")
    set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ort-v141-include")
endif()

# ── 정적 라이브러리 빌드 (다른 프로젝트에서 사용 가능) ──
add_library(vision_inference STATIC
    src/vision_inference.cpp
    src/sam2_inference.cpp
    src/classification_worker.cpp
    include/classification_worker.h
    include/sam2_inference.h
    include/vision_inference.h
)
set_target_properties(vision_inference PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(vision_inference PUBLIC cxx_std_17)
if(MSVC)
    if(MSVC_VERSION LESS 1916)
        message(FATAL_ERROR "Use Visual Studio 2017 15.9 (v141 14.16) or newer")
    endif()
    target_compile_options(vision_inference PUBLIC /utf-8 /Zc:__cplusplus /Zc:noexceptTypes)
endif()

target_include_directories(vision_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(vision_inference PUBLIC
    nlohmann_json::nlohmann_json
    Threads::Threads
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
)

# Stable C ABI for C# and other languages. The C++17 implementation remains
# the owner of ONNX Runtime/OpenCV objects; callers only see opaque handles.
add_library(vision_runtime SHARED src/vision_runtime_c.cpp)
target_compile_features(vision_runtime PUBLIC cxx_std_17)
target_compile_definitions(vision_runtime PRIVATE VISION_RUNTIME_BUILD=1)
target_include_directories(vision_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(vision_runtime PRIVATE vision_inference)

if(VISION_WITH_OPENVINO)
    target_compile_definitions(vision_inference PUBLIC VISION_WITH_OPENVINO=1)
    target_link_libraries(vision_inference PUBLIC openvino::runtime)
endif()

add_executable(efficientnet_cpu_benchmark src/efficientnet_cpu_benchmark.cpp)
target_link_libraries(efficientnet_cpu_benchmark PRIVATE vision_inference)
add_executable(efficientnet_background_demo src/efficientnet_background_demo.cpp)
target_link_libraries(efficientnet_background_demo PRIVATE vision_inference)
if(VISION_BUILD_DEMO)
    add_executable(vision_demo src/main.cpp)
    target_link_libraries(vision_demo PRIVATE vision_inference)
    install(TARGETS vision_demo DESTINATION bin)
endif()
install(TARGETS efficientnet_cpu_benchmark efficientnet_background_demo DESTINATION bin)

# ── 설치 설정 ────────────────────────────────────────
install(TARGETS vision_inference DESTINATION lib)
install(TARGETS vision_runtime DESTINATION bin)
install(FILES include/classification_worker.h include/sam2_inference.h include/vision_inference.h include/vision_runtime_c.h include/opencv_preprocess.h DESTINATION include)

# ── 빌드 정보 출력 ───────────────────────────────────
message(STATUS "")
message(STATUS "═══════════════════════════════════════")
message(STATUS "  Vision 추론 엔진 빌드 설정")
message(STATUS "═══════════════════════════════════════")
message(STATUS "  빌드 타입  : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ 표준   : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV     : ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_LIB}")
message(STATUS "═══════════════════════════════════════")

include(CTest)
if(BUILD_TESTING)
    add_executable(preprocess_probe tests/preprocess_probe.cpp)
    target_include_directories(preprocess_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(preprocess_probe PRIVATE ${OpenCV_LIBS})
    add_executable(classification_worker_test tests/classification_worker_test.cpp)
    target_link_libraries(classification_worker_test PRIVATE vision_inference)
    add_test(NAME classification_worker_contract COMMAND classification_worker_test "${CMAKE_BINARY_DIR}/test-models")
    set_tests_properties(classification_worker_contract PROPERTIES TIMEOUT 30)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME generate_deployment_fixtures COMMAND "${Python3_EXECUTABLE}"
             "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_models.py" "${CMAKE_BINARY_DIR}/test-models")
    set_tests_properties(generate_deployment_fixtures PROPERTIES FIXTURES_SETUP deployment_models)
    set_tests_properties(classification_worker_contract PROPERTIES FIXTURES_REQUIRED deployment_models)
    add_executable(deployment_smoke tests/deployment_smoke.cpp)
    target_link_libraries(deployment_smoke PRIVATE vision_inference)
    add_test(NAME deployment_contracts COMMAND deployment_smoke "${CMAKE_BINARY_DIR}/test-models")
    set_tests_properties(deployment_contracts PROPERTIES FIXTURES_REQUIRED deployment_models)
    add_executable(c_api_contracts tests/c_api_contracts.cpp)
    target_link_libraries(c_api_contracts PRIVATE vision_runtime ${OpenCV_LIBS})
    add_test(NAME c_api_contracts COMMAND c_api_contracts "${CMAKE_BINARY_DIR}/test-models")
    set_tests_properties(c_api_contracts PROPERTIES FIXTURES_REQUIRED deployment_models)
endif()
