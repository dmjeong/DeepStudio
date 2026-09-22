#include "example_paths.h"
#include "classifier.h"
#include <chrono>
#include <cmath>
#include <iostream>

namespace {
void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void CheckWhite(const ClassifyResult& result) {
    Check(result.class_id == 0 && result.class_name == "white", "Wrong class");
    Check(result.probabilities.size() == 2 && std::isfinite(result.confidence) &&
          std::abs(result.confidence - 0.88079708f) < 0.00001f &&
          std::abs(result.probabilities[0] - 0.88079708f) < 0.00001f &&
          std::abs(result.probabilities[1] - 0.11920292f) < 0.00001f, "Wrong probabilities");
}
template<class F> void MustThrow(F action) {
    bool failed = false;
    try { action(); } catch (const std::exception&) { failed = true; }
    Check(failed, "Invalid input was accepted");
}
struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        std::filesystem::u8path(u8"DVS 한글 경로 " + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { std::filesystem::create_directories(path); }
    ~TemporaryDirectory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
}

int main() {
    try {
        std::cout << "ONNX Runtime " << OrtGetApiBase()->GetVersionString() << std::endl;
        const auto assets = example::ExecutableDirectory() / "assets";
        TemporaryDirectory temp;
        for (const auto* name : {"test.json", "test.onnx", "white.pgm"})
            std::filesystem::copy_file(assets / name, temp.path / name);
        // Loading a Unicode JSON path also verifies relative companion ONNX paths.
        example::Classifier model(temp.path / "test.json");
        Check(std::isfinite(model.WarmupMilliseconds()) && model.WarmupMilliseconds() >= 0,
              "Invalid startup warm-up timing");
        Check(model.Config().ort_graph_optimization_level == "disabled" &&
              model.Config().num_threads == 1, "Saved runtime settings lost");
        CheckWhite(model.InferFile(temp.path / "white.pgm"));
        {
            const auto image = example::ReadImage(temp.path / "white.pgm");
            example::Classifier prewarmed(temp.path / "test.json", image);
            CheckWhite(prewarmed.Infer(image));
        }
        // Invalid startup pixels fail during initialization, before UI readiness.
        MustThrow([&] {
            example::Classifier invalid(temp.path / "test.json",
                                        cv::Mat(224, 224, CV_32FC1, cv::Scalar(0)));
        });
        for (int channels : {1, 3, 4}) {
            const cv::Mat pixels(240, 320, CV_MAKETYPE(CV_8U, channels), cv::Scalar::all(255));
            // Non-contiguous camera ROI: checks stride handling, resize and reuse.
            const auto roi = pixels(cv::Rect(3, 4, 230, 225));
            for (int i = 0; i < 20; ++i) CheckWhite(model.Infer(roi));
        }
        const auto black = model.Infer(cv::Mat(224, 224, CV_8UC1, cv::Scalar(0)));
        Check(black.class_id == 1 && black.probabilities.size() == 2 &&
              std::abs(black.confidence - 1.0f) < 0.00001f,
              "Session reused a stale input or output");
        MustThrow([&] { model.Infer(cv::Mat()); });
        MustThrow([&] { model.InferFile(temp.path / "missing.png"); });
        std::ofstream(temp.path / "invalid.png") << "not an image";
        MustThrow([&] { model.InferFile(temp.path / "invalid.png"); });
        MustThrow([&] { example::Classifier invalid(temp.path / "missing.json"); });
        CheckWhite(model.InferFile(temp.path / "white.pgm"));
        std::cout << "PASS: startup warm-up, ONNX load, repeated inference, Unicode paths, camera ROI, error recovery\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
