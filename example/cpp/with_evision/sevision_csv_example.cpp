// Standalone, no arguments, no eVision SDK required to run the public demo.
// In an existing CSeVision class, copy the member/functions, not this class/main.
#include "csv_model_manager.h"
#include "../example_paths.h"
#include <iostream>

class CSeVision {
public:
    void InitializeModel(const std::filesystem::path& csv,
                         const std::filesystem::path& modelDirectory,
                         const std::filesystem::path& keyFile) {
        if (models_) throw std::logic_error("Models already initialized");
        auto next = std::make_unique<dvs_bw8::CsvModelManager>(csv, modelDirectory, keyFile);
        next->Start(); // Initial load + warmup; failure leaves models_ unset.
        models_ = std::move(next);
    }

    // Take ONE snapshot before inspecting a frame (before the OMP loop).
    std::shared_ptr<dvs_bw8::Classifier> GetModel(int index, const std::string& camtype) const {
        if (!models_) throw std::runtime_error("Models are not initialized");
        auto model = models_->Snapshot(index, camtype);
        if (!model) throw std::runtime_error("Unknown index/camtype or model not ready");
        return model;
    }
    std::string ModelStatus() const { return models_ ? models_->LastStatus() : "Not initialized"; }
    void StopModelWatch() { if (models_) models_->Stop(); }

private:
    // Owner thread initializes this once BEFORE starting inspection threads.
    // Never reset/reassign it while inference workers are running.
    std::unique_ptr<dvs_bw8::CsvModelManager> models_;
};

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
        vision.InitializeModel(folder / "models.csv", folder, assets / "example-only.key");
        DemoBW8 roi; // Replace with your existing EROIBW8.
        {
            auto thd_classifier = vision.GetModel(0, "TOP"); // No model reload here.
            const auto result = thd_classifier->InferEvision(roi); // No old Context passed.
            std::cout << "TOP: " << result.class_name << ", " << result.inference_ms << " ms\n";
            auto side = vision.GetModel(1, "SIDE");
            std::cout << "SIDE: " << side->InferEvision(roi).class_name << '\n';
        } // Release frame snapshots; do not cache them permanently.
        std::cout << vision.ModelStatus() << '\n';
        // In your program: stop accepting images and join inspection threads first.
        vision.StopModelWatch();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Model initialization/inference failed: " << error.what() << '\n';
        return 1;
    }
}
