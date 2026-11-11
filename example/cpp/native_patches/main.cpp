// No command-line arguments. Put the exported pair and image beside the executable.
#include "vision_inference.h"
#include "../example_paths.h"
#include <iostream>

int main() {
    try {
        const auto directory = example::ExecutableDirectory();
        VisionInference model; // Initialize once; reuse for all subsequent images.
        constexpr bool encrypted = false; // true: model.dvsenc + model.key
        if (encrypted) {
            auto key = dvs_crypto::ReadKey(directory / "model.key");
            dvs_crypto::Package package(directory / "model.dvsenc", key);
            dvs_crypto::Wipe(key.data(), key.size());
            if (!model.InitializeFromPackage(package, "onnxruntime", 4)) return 1;
        } else if (!model.InitializeFromJson((directory / "model.json").u8string(), "onnxruntime", 4)) return 1;
        const cv::Mat image = cv::imread((directory / "image.png").u8string(), cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("image.png is missing beside the executable");
        // Warm-up runs once at startup. It is excluded from the reported next run.
        if (model.GetConfig().task == "classify") {
            model.Classify(image);
            const auto result = model.Classify(image);
            std::cout << result.class_name << " confidence=" << result.confidence
                      << " total_ms=" << result.inference_ms << '\n';
        } else if (model.GetConfig().task == "detect") {
            model.Detect(image);
            const auto result = model.Detect(image);
            std::cout << "detections=" << result.detections.size() << " total_ms=" << result.inference_ms << '\n';
            for (const auto& box : result.detections)
                std::cout << "class=" << box.class_id << " score=" << box.confidence << " box="
                          << box.x1 << ',' << box.y1 << ',' << box.x2 << ',' << box.y2 << '\n';
        } else throw std::runtime_error("This example is for native classification/detection patches");
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}
