// Standalone, no arguments, no eVision SDK required to run the public demo.
// In an existing CSeVision class, copy the member/functions, not this class/main.
#include "sevision_csv_example.h"
#include "../example_paths.h"
#include <iostream>

// Only for this buildable demo. The real application passes EROIBW8 directly.
struct DemoBW8 {
    std::vector<uint8_t> pixels = std::vector<uint8_t>(240 * 224, 255);
    int GetBitsPerPixel() const { return 8; }
    int GetColPitch() const { return 1; }
    int GetWidth() const { return 224; }
    int GetHeight() const { return 224; }
    size_t GetRowPitch() const { return 240; }
    const uint8_t* GetImagePtr(int x, int y) const { return pixels.data() + y * 240 + x; }
};

int main() {
    try {
        const auto assets = example::ExecutableDirectory() / "assets";
        const auto folder = assets / "filename-reload";
        CSeVision vision;
        std::string error;
        if (!vision.InitializeModel(folder / "models.csv", folder, assets / "example-only.key", error))
            throw std::runtime_error(error);

        // Put this one call in YOUR existing periodic thread. No extra watcher.
        if (!vision.ChangeModel(error)) std::cerr << error << '\n';
        DemoBW8 roi; // Replace with your existing EROIBW8.
        {
            auto thd_classifier = std::atomic_load(&vision.thd_classifier.at(0)); // No model reload here.
            const auto result = thd_classifier->InferEvision(roi); // No old Context passed.
            std::cout << "TOP: " << result.class_name << ", " << result.inference_ms << " ms\n";
            auto side = std::atomic_load(&vision.thd_classifier.at(1));
            std::cout << "SIDE: " << side->InferEvision(roi).class_name << '\n';
        } // Release frame snapshots; do not cache them permanently.
        // In your program: stop accepting images and join YOUR update/inference
        // threads before destroying vision. There is no StopModelWatch function.
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Model initialization/inference failed: " << error.what() << '\n';
        return 1;
    }
}
