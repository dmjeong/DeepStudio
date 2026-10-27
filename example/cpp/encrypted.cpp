// C++17/OpenCV. All tasks, including SAM2, use the same encrypted package.
// No command-line arguments. Edit these paths in your own application.
#include "classifier.h"
#include "sam2_inference.h"
#include "example_paths.h"
#include <iostream>

int main() {
    try {
        const auto base = example::ExecutableDirectory();
        VisionInference engine;
        Sam2Inference sam;
        bool is_sam = false;
        // This block runs once at startup. Its temporary plaintext is wiped on exit.
        {
            auto key = dvs_crypto::ReadKey(base / "assets/example-only.key");
            // Also erase on authentication/initialization exceptions.
            struct EraseKey { dvs_crypto::Key& k; ~EraseKey() { dvs_crypto::Wipe(k.data(), k.size()); } } erase{key};
            dvs_crypto::Package package(base / "assets/test.dvsenc", key);
            dvs_crypto::Wipe(key.data(), key.size());
            is_sam = package.Config().value("backend", std::string()) == "sam2";
            if (!(is_sam ? sam.InitializeFromPackage(package) : engine.InitializeFromPackage(package)))
                throw std::runtime_error("Encrypted model initialization failed");
        }
        const cv::Mat image = example::ReadImage(base / "assets/white.pgm");
        auto infer = [&]() {
            if (is_sam) {
                auto context = sam.Encode(image);
                auto result = sam.Automatic(context, 2, 2);
                std::cout << "SAM2 mask: " << result.mask.cols << "x" << result.mask.rows << '\n';
            } else {
                const auto task = engine.GetConfig().task;
                if (task == "classify") {
                    auto result = engine.Classify(image);
                    std::cout << result.class_name << " " << result.confidence << " ms=" << result.inference_ms << '\n';
                } else if (task == "detect") {
                    auto result = engine.Detect(image);
                    std::cout << "detections=" << result.detections.size() << '\n';
                } else if (task == "segment") {
                    auto result = engine.Segment(image);
                    std::cout << "mask=" << result.mask.cols << "x" << result.mask.rows << '\n';
                } else if (task == "anomaly") {
                    auto result = engine.Anomaly(image);
                    std::cout << "anomaly_score=" << result.score << '\n';
                }
            }
        };
        infer(); // Warm up before enabling your inspection button.
        for (int i = 0; i < 20; ++i) infer(); // Keep both engines alive in your app.
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
