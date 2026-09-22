#pragma once
#include <Open_eVision.h>
#include "classifier.h"

// Keep one instance as an application/inspection-class member.
// Construct at startup, NOT inside the inspection button handler.
// Open_eVision.h provides the namespace alias for the installed SDK version.
class EvisionRoiExample {
public:
    // Loads model.json + its ONNX model and completes one warm-up inference.
    explicit EvisionRoiExample(const std::filesystem::path& model_json)
        : classifier_(model_json) {}

    // roi is the application's existing ROI attached to its camera image.
    // No Attach/SetPlacement here: use the caller's ROI exactly as supplied.
    dvs_bw8::Result Inspect(Euresys::Open_eVision::EROIBW8& roi) {
        return classifier_.InferEvision(roi);
    }

private:
    dvs_bw8::Classifier classifier_;
};
