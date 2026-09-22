#include "../example_paths.h"
#include "classifier.h"
#include <iostream>

int main() {
    try {
        // Edit here; no command-line arguments. This fixture uses a synthetic BW8
        // buffer so the test driver does not require an eVision installation.
        const auto model_json = example::ExecutableDirectory() / "assets/test.json";
        const int width = 224, height = 224;
        const size_t pitch = 240; // padded rows, like a camera/ROI buffer
        std::vector<uint8_t> frame(pitch * height, 255);
        dvs_bw8::Classifier model(model_json, {frame.data(), width, height, pitch});
        std::cout << "READY warmup_ms=" << model.WarmupMilliseconds() << '\n';

        // In your application replace this call with model.InferEvision(image),
        // where image is your existing EImageBW8 or EROIBW8 object.
        const auto result = model.InferBW8(frame.data(), width, height, pitch);
        std::cout << "class=" << result.class_name << " confidence=" << result.confidence
                  << " inference_ms=" << result.inference_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
