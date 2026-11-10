# ═══════════════════════════════════════════════════════
#  INT8 테스트 솔루션 (Visual Studio .sln 생성용 독립 CMake 프로젝트)
#
#    상위 SDK(cpp/)를 그대로 포함해 vision_inference 를 함께 빌드하고,
#    int8_compare.exe 하나를 만든다 (시작 프로젝트로 지정).
#
#  Windows:  make_solution.bat 실행 → build\Int8Test.sln 을 Visual Studio로 연다
#  Linux:    cmake -S . -B build -DONNXRUNTIME_ROOT=... && cmake --build build -j
# ═══════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.14)
project(Int8Test LANGUAGES CXX)

# SDK의 데모·계약 테스트는 이 솔루션에 필요 없다
set(VISION_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. vision_sdk)

add_executable(int8_compare int8_compare.cpp)
target_link_libraries(int8_compare PRIVATE vision_inference)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT int8_compare)

if(WIN32 AND DEFINED ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
    # 실행 파일 옆에 onnxruntime.dll 복사 (C:\Windows\System32 의 오래된 onnxruntime.dll 이 대신 잡히는 것 방지)
    add_custom_command(TARGET int8_compare POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
                $<TARGET_FILE_DIR:int8_compare>)
endif()
