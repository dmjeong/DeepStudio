#include "vision_runtime_c.h"

#include <opencv2/core.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv)
{
    dv_session* session = nullptr;
    dv_result* result = nullptr;
    try
    {
        require(argc == 2, "Fixture directory required.");
        const auto config = (fs::u8path(argv[1]) / "classify.json").u8string();
        dv_session_options options{sizeof(dv_session_options), DV_ABI_VERSION, "onnxruntime", 1};
        require(dv_abi_version() == DV_ABI_VERSION, "ABI version mismatch.");
        require(dv_create_session(nullptr, &options, &session) == DV_STATUS_INVALID_ARGUMENT &&
                    std::string(dv_last_error(nullptr)).find("Invalid session") != std::string::npos,
                "C ABI invalid create arguments were not diagnosed.");
        require(dv_create_session(config.c_str(), &options, &session) == DV_STATUS_OK && session,
                "C ABI session creation failed.");
        cv::Mat gray = (cv::Mat_<uchar>(2, 3) << 0, 127, 255, 0, 127, 255);
        dv_image_view view{sizeof(dv_image_view), DV_ABI_VERSION, gray.data, gray.cols, gray.rows,
                           gray.channels(), static_cast<int32_t>(gray.step)};
        require(dv_infer(session, &view, &result) == DV_STATUS_OK && result,
                dv_last_error(session));
        require(result->kind == DV_RESULT_CLASSIFICATION && result->probability_count == 2,
                "C ABI classification result mismatch.");
        require(result->class_name_utf8 && std::string(result->class_name_utf8) == "OK \"quoted\"",
                "C ABI UTF-8 class name mismatch.");
        require(std::isfinite(result->confidence) && result->total_ms >= 0,
                "C ABI classification timing mismatch.");
        dv_release_result(result);
        result = nullptr;

        dv_image_view bad = view;
        bad.stride_bytes = 1;
        require(dv_infer(session, &bad, &result) == DV_STATUS_INVALID_ARGUMENT && !result,
                "Invalid C ABI image was accepted.");
        require(std::string(dv_last_error(session)).find("stride") != std::string::npos,
                "C ABI error detail was not preserved.");

        dv_close_session(session);
        session = nullptr;
        std::cout << "C ABI contract passed." << std::endl;
        return 0;
    }
    catch (const std::exception& error)
    {
        dv_release_result(result);
        dv_close_session(session);
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
