// C++17: copy this small wrapper into your application and link vision_inference.
#pragma once
#if defined(_MSC_VER)
// Manual VS projects must also add the SDK lib folder to the library search path.
#pragma comment(lib, "onnxruntime.lib")
#if _MSC_VER < 1916
#error Use Visual Studio 2017 15.9 (v141 14.16) or newer.
#endif
#if !defined(_MSVC_LANG) || _MSVC_LANG < 201703L
#error Enable C++17: Project Properties > C/C++ > Language > /std:c++17.
#endif
#if !defined(__cpp_noexcept_function_type)
#error Enable /Zc:noexceptTypes and remove /Zc:noexceptTypes- before including ONNX Runtime.
#endif
#endif
#include "vision_inference.h"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace example {

inline cv::Mat ReadImage(const std::filesystem::path& path) {
    // cv::imread(string) cannot reliably open Korean Windows paths.
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open image: " + path.u8string());
    const std::vector<unsigned char> bytes{
        std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (bytes.empty()) throw std::runtime_error("Empty image: " + path.u8string());
    auto image = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
    if (image.empty()) throw std::runtime_error("Cannot decode image: " + path.u8string());
    return image;
}

// Construct once during application startup, before enabling the inference UI.
// Construction includes one warm-up inference. Call on one thread at a time.
class Classifier {
public:
    explicit Classifier(const std::filesystem::path& model_json,
                        const cv::Mat& startup_image = cv::Mat()) {
        // JSON resolves the companion ONNX path and preserves verified settings.
        if (!model_.InitializeFromJson(model_json.u8string(), "onnxruntime"))
            throw std::runtime_error("Model/JSON load failed: " + model_json.u8string());
        if (model_.GetConfig().task != "classify")
            throw std::runtime_error("Classifier requires a classification export");
        const auto& config = model_.GetConfig();
        // Prefer a representative camera frame (same size/type as production).
        // Without one, exercise the complete pipeline with a model-sized image.
        const cv::Mat warmup = startup_image.empty()
            ? cv::Mat(config.input_height, config.input_width,
                      CV_MAKETYPE(CV_8U, config.input_channels), cv::Scalar::all(0))
            : startup_image;
        const auto start = std::chrono::steady_clock::now();
        const auto result = model_.Classify(warmup);
        warmup_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (result.class_id < 0 || !std::isfinite(result.confidence))
            throw std::runtime_error("Startup warm-up inference failed");
        // Discard this prediction. Only successful construction means ready.
    }

    ClassifyResult Infer(const cv::Mat& pixels) {
        // Original GRAY/BGR/BGRA pixels. Do not resize/normalize/softmax again.
        return model_.Classify(pixels);
    }

    ClassifyResult InferFile(const std::filesystem::path& image_path) {
        return Infer(ReadImage(image_path));
    }

    const InferenceConfig& Config() const { return model_.GetConfig(); }
    double WarmupMilliseconds() const { return warmup_ms_; }

private:
    VisionInference model_;
    double warmup_ms_ = 0;
};

} // namespace example
