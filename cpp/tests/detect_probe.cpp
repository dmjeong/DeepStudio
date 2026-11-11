#include "vision_inference.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <filesystem>
int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4) return 2;
        VisionInference model;
        if (argc == 4) {
            auto key = dvs_crypto::ReadKey(std::filesystem::u8path(argv[3]));
            dvs_crypto::Package package(std::filesystem::u8path(argv[1]), key);
            dvs_crypto::Wipe(key.data(), key.size());
            if (!model.InitializeFromPackage(package, "onnxruntime", 1)) return 3;
        } else if (!model.InitializeFromJson(argv[1], "onnxruntime", 1)) return 3;
        cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        auto result = model.Detect(image);
        nlohmann::json values = nlohmann::json::array();
        for (const auto& row : result.detections)
            values.push_back({{"class_id", row.class_id}, {"confidence", row.confidence},
                              {"bbox", {row.x1, row.y1, row.x2, row.y2}}});
        std::cout << nlohmann::json{{"detections", values}, {"inference_ms", result.inference_ms}}.dump() << std::endl;
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}
