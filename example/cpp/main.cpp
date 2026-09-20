// C++17: one session, repeated classification, and a real ONNX self-test.
#include "vision_inference.h"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        const bool self_test = argc == 3 && std::string(argv[1]) == "--self-test";
        if (!self_test && (argc < 3 || argc > 4)) {
            std::cerr << "Usage: onnx_cpp_example <model.json> <image> [runs=20]\n"
                         "       onnx_cpp_example --self-test <example/assets>\n";
            return 2;
        }
        const auto assets = std::filesystem::u8path(argv[2]);
        const auto config = self_test ? (assets / "test.json").u8string() : argv[1];
        const auto image_path = self_test ? (assets / "white.pgm").u8string() : argv[2];
        int runs = 20;
        if (!self_test && argc == 4) {
            std::size_t used = 0;
            runs = std::stoi(argv[3], &used);
            if (used != std::string(argv[3]).size()) throw std::invalid_argument("Invalid runs");
        }
        if (runs < 1 || runs > 10000) throw std::invalid_argument("runs must be 1..10000");

        // Open JSON beside the ONNX. It carries preprocessing, classes and the
        // graph optimization/thread settings that passed export verification.
        VisionInference session;
        if (!session.InitializeFromJson(config, "onnxruntime"))
            throw std::runtime_error("Model/JSON load failed (see native error above)");
        if (session.GetConfig().task != "classify")
            throw std::runtime_error("This example expects a classification export");
        const cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
        if (image.empty()) throw std::runtime_error("Image load failed: " + image_path);
        if (image.depth() != CV_8U) throw std::runtime_error("Use an 8-bit image");
        // Supply original GRAY/BGR/BGRA pixels; the SDK resizes and normalizes.
        for (int i = 0; i < 3; ++i) session.Classify(image);
        std::vector<double> times;
        double pre = 0, model = 0, post = 0;
        ClassifyResult result;
        for (int i = 0; i < runs; ++i) {
            const auto start = std::chrono::steady_clock::now();
            result = session.Classify(image);
            times.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
            if (result.class_id < 0 || !std::isfinite(result.confidence))
                throw std::runtime_error("Invalid inference result");
            if (self_test && (result.class_id != 0 || result.probabilities.size() != 2 ||
                std::abs(result.probabilities[0] - 0.88079708f) > 0.00001f ||
                std::abs(result.probabilities[1] - 0.11920292f) > 0.00001f))
                throw std::runtime_error("Self-test output mismatch; check SDK/runtime settings");
            pre += result.preprocess_ms;
            model += result.model_ms;
            post += result.postprocess_ms;
        }
        std::sort(times.begin(), times.end());
        std::cout << std::fixed << std::setprecision(6)
                  << "class_id=" << result.class_id << " class=" << result.class_name
                  << " confidence=" << result.confidence << '\n'
                  << "runs=" << runs << " preprocess_ms=" << pre / runs
                  << " model_ms=" << model / runs << " postprocess_ms=" << post / runs
                  << " call_p50_ms=" << times[(runs - 1) / 2]
                  << " call_p95_ms=" << times[static_cast<int>(std::ceil(runs * .95)) - 1] << '\n';
        if (self_test) std::cout << "PASS: ONNX load, inference and saved runtime settings\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
