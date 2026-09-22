cmake_minimum_required(VERSION 3.14)
project(OnnxExample LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(VISION_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(DEEPVISION_RUNTIME_SOURCE "" CACHE PATH
    "Path to the Vision C++ runtime source; the installer copies it beside this example")
if(NOT DEEPVISION_RUNTIME_SOURCE)
    # Repository checkout: example/cpp -> ../../cpp.
    set(DEEPVISION_RUNTIME_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../../cpp")
endif()
if(NOT EXISTS "${DEEPVISION_RUNTIME_SOURCE}/CMakeLists.txt")
    # Installed layout: Examples/cpp/vision-runtime.
    set(DEEPVISION_RUNTIME_SOURCE "${CMAKE_CURRENT_LIST_DIR}/vision-runtime")
endif()
if(NOT EXISTS "${DEEPVISION_RUNTIME_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "Vision C++ runtime source was not found. Set -DDEEPVISION_RUNTIME_SOURCE=<path>.")
endif()
# Import OpenCV targets in this directory as well: imported targets created only
# in the runtime subdirectory are not visible to the app-local copy commands.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/nlohmann" nlohmann)
add_subdirectory("${DEEPVISION_RUNTIME_SOURCE}" vision EXCLUDE_FROM_ALL)
add_executable(onnx_cpp_example main.cpp)
add_executable(onnx_cpp_self_test self_test.cpp)
if(WIN32)
    # PATH is searched after Windows system directories. Put the matching ORT
    # DLL next to each EXE so a system/other application's ORT is never selected.
    get_filename_component(_ort_lib_dir "${ONNXRUNTIME_LIB}" DIRECTORY)
    find_file(EXAMPLE_ORT_DLL NAMES onnxruntime.dll
        PATHS "${_ort_lib_dir}" "${_ort_lib_dir}/../bin"
              "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/bin" NO_DEFAULT_PATH)
    if(NOT EXAMPLE_ORT_DLL)
        message(FATAL_ERROR "Matching onnxruntime.dll not found. Set ONNXRUNTIME_ROOT to the extracted Windows SDK.")
    endif()
endif()
foreach(target onnx_cpp_example onnx_cpp_self_test)
    target_link_libraries(${target} PRIVATE vision_inference)
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8)
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_LIST_DIR}/../assets"
            "$<TARGET_FILE_DIR:${target}>/assets")
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${EXAMPLE_ORT_DLL}"
                "$<TARGET_FILE_DIR:${target}>/onnxruntime.dll")
        # Also support the official prebuilt OpenCV SDK without vcpkg's app-local
        # hook. Its component targets resolve to the shared opencv_world DLL.
        foreach(opencv_target IN LISTS OpenCV_LIBS)
            if(TARGET ${opencv_target})
                get_target_property(_opencv_type ${opencv_target} TYPE)
                if(_opencv_type STREQUAL "SHARED_LIBRARY")
                    add_custom_command(TARGET ${target} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "$<TARGET_FILE:${opencv_target}>" "$<TARGET_FILE_DIR:${target}>")
                endif()
            endif()
        endforeach()
    endif()
endforeach()
enable_testing()
add_test(NAME onnx_example_no_args COMMAND onnx_cpp_example)
add_test(NAME onnx_example_self_test COMMAND onnx_cpp_self_test)
# Intentionally run away from the EXE/assets directory, just like a host app/IDE.
set_tests_properties(onnx_example_no_args onnx_example_self_test PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}" TIMEOUT 60)
