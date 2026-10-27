// No arguments. Replace these example paths with your own package/key paths.
#include "classifier.h"
#include "../example_paths.h"
#include <iostream>

int main() {
    try {
        const auto base = example::ExecutableDirectory();
        auto key = dvs_crypto::ReadKey(base / "assets/example-only.key");
        // Also erase if the constructor rejects the key or model.
        struct EraseKey { dvs_crypto::Key& k; ~EraseKey() { dvs_crypto::Wipe(k.data(), k.size()); } } erase{key};
        // Initialization decrypts/authenticates and warms up ONCE.
        dvs_bw8::Classifier classifier(base / "assets/test.dvsenc", key);
        dvs_crypto::Wipe(key.data(), key.size());
        std::vector<uint8_t> frame(224 * 224, 255);
        // In your eVision project, use instead:
        // auto result = classifier.InferEvision(roi); // EROIBW8&
        for (int i = 0; i < 20; ++i) {
            auto result = classifier.InferBW8(frame.data(), 224, 224, 224);
            if (result.class_id != 0) throw std::runtime_error("Example prediction mismatch");
            std::cout << result.class_name << " " << result.confidence
                      << " inference_ms=" << result.inference_ms << '\n';
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
