# Bundled, unmodified nlohmann/json v3.12.0 (MIT).
if(NOT TARGET nlohmann_json::nlohmann_json)
    add_library(dvs_example_json INTERFACE)
    target_include_directories(dvs_example_json INTERFACE "${CMAKE_CURRENT_LIST_DIR}/..")
    add_library(nlohmann_json::nlohmann_json ALIAS dvs_example_json)
endif()
