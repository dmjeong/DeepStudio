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

        dv_session* detect_session = nullptr;
        dv_result* detect_result = nullptr;
        const auto detect_config = (fs::u8path(argv[1]) / "detect.json").u8string();
        require(dv_create_session(detect_config.c_str(), &options, &detect_session) == DV_STATUS_OK,
                "C ABI detection session creation failed.");
        require(dv_infer(detect_session, &view, &detect_result) == DV_STATUS_OK && detect_result &&
                    detect_result->kind == DV_RESULT_DETECTION && detect_result->detection_count == 1,
                "C ABI detection result mismatch.");
        dv_release_result(detect_result);
        dv_close_session(detect_session);

        dv_session* redetr_session = nullptr;
        dv_result* redetr_result = nullptr;
        const auto redetr_config = (fs::u8path(argv[1]) / "redetr.json").u8string();
        require(dv_create_session(redetr_config.c_str(), &options, &redetr_session) == DV_STATUS_OK,
                "C ABI Re-DETR v4 session creation failed.");
        require(dv_infer(redetr_session, &view, &redetr_result) == DV_STATUS_OK && redetr_result &&
                    redetr_result->kind == DV_RESULT_DETECTION && redetr_result->detection_count == 1 &&
                    redetr_result->detections[0].class_id == 0 && redetr_result->detections[0].confidence > 0.99f,
                "C ABI Re-DETR v4 result mismatch.");
        dv_release_result(redetr_result);
        dv_close_session(redetr_session);

        dv_session* anomaly_session = nullptr;
        dv_result* anomaly_result = nullptr;
        const auto anomaly_config = (fs::u8path(argv[1]) / "anomaly.json").u8string();
        require(dv_create_session(anomaly_config.c_str(), &options, &anomaly_session) == DV_STATUS_OK,
                "C ABI anomaly session creation failed.");
        require(dv_infer(anomaly_session, &view, &anomaly_result) == DV_STATUS_OK && anomaly_result &&
                    anomaly_result->kind == DV_RESULT_ANOMALY && anomaly_result->anomaly_map_width == 3,
                "C ABI anomaly result mismatch.");
        dv_release_result(anomaly_result);
        dv_close_session(anomaly_session);

        dv_session* patchcore_session = nullptr;
        dv_result* patchcore_result = nullptr;
        const auto patchcore_config = (fs::u8path(argv[1]) / "patchcore.json").u8string();
        require(dv_create_session(patchcore_config.c_str(), &options, &patchcore_session) == DV_STATUS_OK,
                "C ABI PatchCore session creation failed.");
        require(dv_infer(patchcore_session, &view, &patchcore_result) == DV_STATUS_OK && patchcore_result &&
                    patchcore_result->kind == DV_RESULT_ANOMALY && patchcore_result->anomalous == 1,
                "C ABI PatchCore result mismatch.");
        dv_release_result(patchcore_result);
        dv_close_session(patchcore_session);

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
