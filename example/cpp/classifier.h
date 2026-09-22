// C++17: copy this small wrapper into your application and link vision_inference.
#pragma once
#include "vision_inference.h"
#include <opencv2/imgcodecs.hpp>
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

// Keep one instance for the model lifetime. Call on one thread at a time.
class Classifier {
public:
    explicit Classifier(const std::filesystem::path& model_json) {
        // JSON resolves the companion ONNX path and preserves verified settings.
        if (!model_.InitializeFromJson(model_json.u8string(), "onnxruntime"))
            throw std::runtime_error("Model/JSON load failed: " + model_json.u8string());
        if (model_.GetConfig().task != "classify")
            throw std::runtime_error("Classifier requires a classification export");
    }

    ClassifyResult Infer(const cv::Mat& pixels) {
        // Original GRAY/BGR/BGRA pixels. Do not resize/normalize/softmax again.
        return model_.Classify(pixels);
    }

    ClassifyResult InferFile(const std::filesystem::path& image_path) {
        return Infer(ReadImage(image_path));
    }

    const InferenceConfig& Config() const { return model_.GetConfig(); }

private:
    VisionInference model_;
};

} // namespace example
