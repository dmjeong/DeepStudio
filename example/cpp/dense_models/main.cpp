// C++17. No arguments: edit the constants and put the files beside the EXE.
#include "vision_inference.h"
#include "model_crypto.h"
#include "../example_paths.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

void SaveImage(const std::filesystem::path& path, const cv::Mat& image)
{
    std::vector<unsigned char> encoded;
    if (!cv::imencode(path.extension().u8string(), image, encoded))
        throw std::runtime_error("Image encoding failed");
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    if (!file) throw std::runtime_error("Image saving failed");
}

void SaveScores(const std::filesystem::path& directory, const cv::Mat& scores)
{
    if (scores.type() != CV_32FC1) throw std::runtime_error("Expected float32 score map");
    std::ofstream file(directory / "anomaly_scores.f32", std::ios::binary);
    for (int y = 0; y < scores.rows; ++y)
        file.write(reinterpret_cast<const char*>(scores.ptr<float>(y)), scores.cols * sizeof(float));
    if (!file) throw std::runtime_error("Score map saving failed");
    std::ofstream metadata(directory / "anomaly_scores.json");
    metadata << nlohmann::json{{"height", scores.rows}, {"width", scores.cols},
        {"dtype", "float32"}, {"layout", "HW"}, {"byte_order", "little"}}.dump(2);
    if (!metadata) throw std::runtime_error("Score metadata saving failed");
}

int main()
{
    constexpr bool encrypted = false;
    constexpr int threads = 4;
    try {
        const auto directory = example::ExecutableDirectory();
        VisionInference model; // Keep this object alive; initialize once.
        if (encrypted) {
            auto key = dvs_crypto::ReadKey(directory / "model.key");
            const dvs_crypto::Package package(directory / "model.dvsenc", key);
            dvs_crypto::Wipe(key.data(), key.size());
            if (!model.InitializeFromPackage(package, "onnxruntime", threads)) return 1;
        } else if (!model.InitializeFromJson((directory / "model.json").u8string(), "onnxruntime", threads)) return 1;
        // Replace this read with your application's BGR/gray cv::Mat buffer.
        // Do not crop/resize it again: JSON owns the training crop/tile contract.
        // filesystem I/O also supports Korean Windows paths.
        const auto image = cv::imdecode(dvs_crypto::ReadFile(directory / "image.png"), cv::IMREAD_UNCHANGED);
        if (image.empty()) throw std::runtime_error("image.png is missing beside the executable");
        SaveImage(directory / "original.png", image);
        if (model.GetConfig().task == "segment") {
            model.Segment(image); // Startup warmup, excluded from measurement.
            const auto result = model.Segment(image);
            SaveImage(directory / "classes.png", result.mask);
            std::cout << "preprocess_ms=" << result.preprocess_ms
                      << " model_ms=" << result.model_ms << " postprocess_ms=" << result.postprocess_ms
                      << " total_ms=" << result.inference_ms << '\n';
            for (const auto& defect : result.defects)
                std::cout << "class=" << defect.class_id << " area=" << defect.area
                          << " score=" << defect.score << '\n';
        } else if (model.GetConfig().task == "anomaly") {
            model.Anomaly(image);
            const auto result = model.Anomaly(image);
            // Raw scores, with dimensions; no optional TIFF codec dependency.
            SaveScores(directory, result.anomaly_map);
            std::cout << (result.anomalous ? "NG" : "OK") << " raw_score=" << result.score
                      << " threshold=" << result.threshold << " preprocess_ms=" << result.preprocess_ms
                      << " model_ms=" << result.model_ms << " postprocess_ms=" << result.postprocess_ms
                      << " total_ms=" << result.inference_ms << '\n';
        } else throw std::runtime_error("Choose a segmentation or anomaly model");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
