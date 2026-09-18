#include "vision_runtime_c.h"

#include <opencv2/core.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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
        const auto bundle = fs::u8path(argv[1]) / "classify.dvdeploy";
        dv_session* bundle_session = nullptr;
        require(dv_create_session_from_bundle(bundle.u8string().c_str(), &options, &bundle_session) == DV_STATUS_OK &&
                    bundle_session,
                "C ABI deployment bundle session creation failed.");
        dv_close_session(bundle_session);
        // The native SDK must reject a changed graph before it opens an ONNX
        // session, because the Windows C# path does not run Python first.
        const auto bundle_graph = bundle / "classify.onnx";
        const auto original_graph = fs::u8path(argv[1]) / "classify.onnx";
        auto graph_bytes = std::ifstream(bundle_graph, std::ios::binary);
        std::string graph((std::istreambuf_iterator<char>(graph_bytes)), std::istreambuf_iterator<char>());
        require(!graph.empty(), "Deployment bundle graph fixture is empty.");
        graph[0] = static_cast<char>(graph[0] ^ 0x01);
        std::ofstream(bundle_graph, std::ios::binary)
            .write(graph.data(), static_cast<std::streamsize>(graph.size()));
        bundle_session = nullptr;
        require(dv_create_session_from_bundle(bundle.u8string().c_str(), &options, &bundle_session) == DV_STATUS_INVALID_ARGUMENT &&
                    bundle_session == nullptr && std::string(dv_last_error(nullptr)).find("checksum") != std::string::npos,
                "C ABI accepted a tampered deployment bundle.");
        fs::copy_file(original_graph, bundle_graph, fs::copy_options::overwrite_existing);
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

        dv_session* sam_session = nullptr;
        dv_result* sam_result = nullptr;
        dv_image_context* sam_context = nullptr;
        const auto sam_config = (fs::u8path(argv[1]) / "sam2.json").u8string();
        require(dv_create_session(sam_config.c_str(), &options, &sam_session) == DV_STATUS_OK,
                "C ABI SAM2 session creation failed.");
        require(dv_sam_encode(sam_session, &view, &sam_context) == DV_STATUS_OK && sam_context,
                "C ABI SAM2 encode failed.");
        const float points[] = {1.0f, 1.0f};
        const int32_t labels[] = {1};
        dv_sam_prompt prompt{sizeof(dv_sam_prompt), DV_ABI_VERSION, points, labels, 1,
                             nullptr, nullptr, 0, 0};
        require(dv_sam_segment(sam_session, sam_context, &prompt, &sam_result) == DV_STATUS_OK && sam_result &&
                    sam_result->kind == DV_RESULT_SEGMENTATION && sam_result->mask_width == 2 &&
                    sam_result->mask_height == 2,
                "C ABI SAM2 prompt result mismatch.");
        dv_release_result(sam_result);
        sam_result = nullptr;
        const float box[] = {0.0f, 0.0f, 1.0f, 1.0f};
        const float mask[] = {1.0f, 1.0f, 1.0f, 1.0f};
        dv_sam_prompt box_prompt{sizeof(dv_sam_prompt), DV_ABI_VERSION, nullptr, nullptr, 0,
                                 box, mask, 2, 2};
        require(dv_sam_segment(sam_session, sam_context, &box_prompt, &sam_result) == DV_STATUS_OK && sam_result &&
                    sam_result->kind == DV_RESULT_SEGMENTATION && sam_result->mask_width == 2 &&
                    sam_result->mask_height == 2,
                "C ABI SAM2 box/mask prompt result mismatch.");
        dv_release_result(sam_result);
        dv_release_image_context(sam_context);
        dv_close_session(sam_session);

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
