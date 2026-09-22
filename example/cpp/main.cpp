// C++17: no command-line arguments. Edit these settings, then run the EXE.
#include "example_paths.h"
#include "classifier.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    try {
        // Change only these three values for your exported classification model.
        // Absolute paths also work: std::filesystem::u8path(u8"C:/모델/model.json")
        const auto model_json = std::filesystem::u8path(u8"assets/test.json");
        const auto image_file = std::filesystem::u8path(u8"assets/white.pgm");
        const int runs = 20;
        if (runs < 1 || runs > 10000) throw std::invalid_argument("runs must be 1..10000");
        const auto base = example::ExecutableDirectory();
        std::cout << "ONNX Runtime " << OrtGetApiBase()->GetVersionString() << std::endl;
        example::Classifier session(base / model_json); // load once
        const cv::Mat image = example::ReadImage(base / image_file);
        // Supply original GRAY/BGR/BGRA pixels; the SDK resizes and normalizes.
        for (int i = 0; i < 3; ++i) session.Infer(image);
        std::vector<double> times;
        double pre = 0, model = 0, post = 0;
        ClassifyResult result;
        for (int i = 0; i < runs; ++i) {
            const auto start = std::chrono::steady_clock::now();
            result = session.Infer(image);
            times.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
            if (result.class_id < 0 || !std::isfinite(result.confidence))
                throw std::runtime_error("Invalid inference result");
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
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
