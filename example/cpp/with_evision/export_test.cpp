#include "../example_paths.h"
#include "classifier.h"
#include <iostream>

// An offset BW8 ROI with padded parent rows, using the same accessor contract as
// EROIBW8. No proprietary SDK is needed to test the production pointer adapter.
struct ExportROI {
    static constexpr int pitch = 32;
    static constexpr int width = 13;
    static constexpr int height = 9;
    std::vector<uint8_t> data = std::vector<uint8_t>(pitch * 16, 33);
    void* GetImagePtr(int x, int y) { return data.data() + (y + 2) * pitch + x + 3; }
    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
    int GetRowPitch() const { return pitch; }
    int GetBitsPerPixel() const { return 8; }
    int GetColPitch() const { return 1; }
};

void Check(bool passed, const char* message) {
    if (!passed) throw std::runtime_error(message);
}

int main() {
    try {
        const auto assets = example::ExecutableDirectory() / "export-test-assets";
        std::ifstream file(assets / "expected.json");
        if (!file) throw std::runtime_error("Cannot open export reference results");
        const auto samples = nlohmann::json::parse(file);
        // No hand-edited metadata: load exactly what export_checkpoint emitted.
        dvs_bw8::Classifier model(assets / "libreyolo.json");
        Check(std::isfinite(model.WarmupMilliseconds()), "Exported model warm-up failed");
        ExportROI image;
        for (const auto& sample : samples) {
            const auto pixel = sample.at("pixel").get<uint8_t>();
            for (int y = 0; y < image.GetHeight(); ++y)
                std::fill_n(static_cast<uint8_t*>(image.GetImagePtr(0, y)), image.GetWidth(), pixel);
            const auto expected = sample.at("probabilities").get<std::vector<float>>();
            for (int repeat = 0; repeat < 3; ++repeat) {
                const auto result = model.InferEvision(image);
                Check(result.class_id == sample.at("class_id").get<int>(), "Exported model class mismatch");
                Check(result.class_name == sample.at("class_name").get<std::string>(), "Exported class name mismatch");
                Check(result.probabilities.size() == expected.size(), "Exported probabilities shape mismatch");
                for (size_t i = 0; i < expected.size(); ++i)
                    Check(std::abs(result.probabilities[i] - expected[i]) < 1e-5f,
                          "Exported probability mismatch (including accidental double softmax)");
                Check(std::abs(result.confidence - expected[result.class_id]) < 1e-5f,
                      "Exported confidence mismatch");
            }
        }
        std::cout << "PASS: Studio LibreYOLO export -> JSON -> C++ load/warm-up -> BW8 ROI probabilities\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
