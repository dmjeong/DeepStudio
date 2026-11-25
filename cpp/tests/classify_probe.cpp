// Exported JSON + original image -> actual SDK probabilities, without repair.
#include "vision_inference.h"
#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include <iostream>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) return 2;
    try {
        VisionInference engine;
        if (argc == 4) {
            auto key = dvs_crypto::ReadKey(std::filesystem::u8path(argv[3]));
            dvs_crypto::Package package(std::filesystem::u8path(argv[1]), key);
            dvs_crypto::Wipe(key.data(), key.size());
            if (!engine.InitializeFromPackage(package, "onnxruntime", 1)) return 3;
        } else if (!engine.InitializeFromJson(argv[1], "", 1)) return 3;
        const auto image = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
        if (image.empty()) return 4;
        const auto result = engine.Classify(image);
        std::cout << std::setprecision(9) << "{\"class_id\":" << result.class_id
                  << ",\"probabilities\":[";
        for (size_t i = 0; i < result.probabilities.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << result.probabilities[i];
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 5;
    }
    return 0;
}
