#include "../example_paths.h"
#include "classifier.h"
#include <iostream>

void Check(bool ok) { if (!ok) throw std::runtime_error("BW8 contract mismatch"); }
template<class F> void MustThrow(F f) {
    bool failed = false; try { f(); } catch (const std::exception&) { failed = true; } Check(failed);
}
// Only the documented BW8 accessors are used. Vendor SDK execution remains a
// separate check on the user's licensed Windows development machine.
struct BW8View {
    std::vector<uint8_t> data = std::vector<uint8_t>(240 * 224, 255);
    void* GetImagePtr(int x, int y) { return data.data() + y * 240 + x; }
    int GetWidth() const { return 224; }
    int GetHeight() const { return 224; }
    int GetRowPitch() const { return 240; }
    int GetBitsPerPixel() const { return bits; }
    int GetColPitch() const { return 1; }
    int bits = 8;
};
int main() {
    try {
        const auto path = example::ExecutableDirectory() / "assets/test.json";
        BW8View image;
        dvs_bw8::Classifier model(path);
        Check(std::isfinite(model.WarmupMilliseconds()));
        for (int i = 0; i < 20; ++i) {
            const auto result = model.InferEvision(image);
            Check(result.class_id == 0 && result.class_name == "white" &&
                  std::abs(result.confidence - 0.88079708f) < 1e-5f);
        }
        std::fill(image.data.begin(), image.data.end(), 0);
        Check(model.InferEvision(image).class_id == 1);
        image.bits = 16;
        MustThrow([&] { model.InferEvision(image); });
        image.bits = 8;
        MustThrow([&] { model.InferBW8(nullptr, 224, 224, 240); });
        MustThrow([&] { model.InferBW8(image.data.data(), 224, 224, 1); });
        MustThrow([&] { model.InferBW8(image.data.data(), -1, 224, 240); });
        MustThrow([&] { dvs_bw8::Classifier bad(path, {nullptr, 224, 224, 224}); });
        std::fill(image.data.begin(), image.data.end(), 255);
        Check(model.InferEvision(image).class_id == 0);
        dvs_bw8::Classifier rgb(path.parent_path() / "bw8_rgb.json");
        const auto rgb_result = rgb.InferEvision(image);
        const float average = ((1.f - .1f) / .5f + (1.f - .2f) / .6f + (1.f - .3f) / .7f) / 3.f;
        Check(rgb_result.class_id == 0 && std::abs(rgb_result.confidence -
              1.f / (1.f + std::exp(-2.f * average))) < 1e-4f);
        std::cout << "PASS: BW8 pointer/pitch, startup warm-up, repeated ONNX inference, invalid buffers\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
