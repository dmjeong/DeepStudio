// No command-line arguments. Demonstrates the same Snapshot used with EROIBW8.
#include "csv_model_watcher.h"
#include "../example_paths.h"
#include <iostream>

int main() {
    try {
        const auto folder = example::ExecutableDirectory() / "assets/reload";
        dvs_bw8::CsvModelWatcher models(folder / "models.csv", "chip", "../example-only.key");
        models.Start(); // Initial decrypt/load/warmup; later updates use one background thread.
        std::vector<uint8_t> pixels(224 * 224, 255);
        for (int i = 0; i < 10; ++i) {
            auto thd_classifier = models.Snapshot(); // A shared_ptr; no disk I/O or initialization.
            auto result = thd_classifier->InferBW8(pixels.data(), 224, 224, 224);
            // In the user's eVision worker: thd_classifier->InferEvision(roi);
            std::cout << thd_classifier->ModelVersion() << " " << result.class_name << '\n';
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        models.Stop(); // In a real app, join inference workers before destroying models.
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
