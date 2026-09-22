# VS2017 C3615: microsoft/onnxruntime#20564. Copy, never modify the SDK.
# Standalone: cmake -DORT_INCLUDE=C:/onnxruntime/include -DORT_OUTPUT=C:/ort-v141 -P ort_vs2017.cmake
function(dvs_ort_vs2017_headers source destination)
    get_filename_component(source "${source}" ABSOLUTE)
    get_filename_component(destination "${destination}" ABSOLUTE)
    file(RELATIVE_PATH relative "${source}" "${destination}")
    if(NOT IS_ABSOLUTE "${relative}" AND NOT relative MATCHES "^\\.\\.(/|$)")
        message(FATAL_ERROR "ORT_OUTPUT must be outside the original SDK include directory")
    endif()
    file(READ "${source}/onnxruntime_cxx_api.h" header)
    # Only these four declarations lose compile-time evaluation. Function bodies,
    # FP32/FP64 inference, C ABI, and the runtime DLL remain untouched.
    foreach(declaration
        "constexpr explicit Float16_t(uint16_t v)"
        "constexpr static Float16_t FromBits(uint16_t v)"
        "constexpr explicit BFloat16_t(uint16_t v)"
        "static constexpr BFloat16_t FromBits(uint16_t v)")
        string(FIND "${header}" "${declaration}" position)
        if(position EQUAL -1)
            message(FATAL_ERROR "Unknown ONNX Runtime header layout; cannot safely apply VS2017 compatibility fix: ${declaration}")
        endif()
        string(REPLACE "constexpr " "" replacement "${declaration}")
        string(REPLACE "${declaration}" "${replacement}" header "${header}")
    endforeach()
    file(MAKE_DIRECTORY "${destination}")
    file(COPY "${source}/" DESTINATION "${destination}")
    file(WRITE "${destination}/onnxruntime_cxx_api.h" "${header}")
    message(STATUS "VS2017 ONNX Runtime compatibility headers: ${destination}")
endfunction()

if(CMAKE_SCRIPT_MODE_FILE)
    if(NOT ORT_INCLUDE OR NOT ORT_OUTPUT)
        message(FATAL_ERROR "Set -DORT_INCLUDE=<SDK include> -DORT_OUTPUT=<new folder>")
    endif()
    dvs_ort_vs2017_headers("${ORT_INCLUDE}" "${ORT_OUTPUT}")
endif()
