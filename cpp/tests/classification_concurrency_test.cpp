// Shared-instance callers own distinct, immutable images. Input/output scratch
// belongs to VisionInference and must never mix frames across calling threads.
#include "vision_inference.h"
#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <thread>

static bool Same(const ClassifyResult& a, const ClassifyResult& b) {
    return a.class_id == b.class_id && a.class_name == b.class_name &&
           std::memcmp(&a.confidence, &b.confidence, sizeof(a.confidence)) == 0 &&
           a.probabilities.size() == b.probabilities.size() &&
           (a.probabilities.empty() || std::memcmp(a.probabilities.data(), b.probabilities.data(),
                                                  a.probabilities.size() * sizeof(float)) == 0);
}
static void Exercise(const std::filesystem::path& root, bool encrypted) {
    VisionInference model;
    if (encrypted) {
        auto key = dvs_crypto::ReadKey(root / "example-only.key");
        struct EraseKey { dvs_crypto::Key& key; ~EraseKey() { dvs_crypto::Wipe(key.data(), key.size()); } } erase{key};
        const dvs_crypto::Package package(root / "classify.dvsenc", key);
        if (!model.InitializeFromPackage(package)) throw std::runtime_error("Encrypted initialization failed");
    } else if (!model.InitializeFromJson((root / "classify.json").u8string())) {
        throw std::runtime_error("Original initialization failed");
    }
    std::array<cv::Mat, 4> images{
        cv::Mat(61, 71, CV_8UC1, cv::Scalar(0)),
        cv::Mat(63, 73, CV_8UC1, cv::Scalar(255)),
        cv::Mat(67, 79, CV_8UC3, cv::Scalar(255, 0, 0)),
        cv::Mat(69, 83, CV_8UC4, cv::Scalar(0, 255, 0, 255))
    };
    std::array<ClassifyResult, 4> expected;
    for (size_t i = 0; i < images.size(); ++i) expected[i] = model.Classify(images[i]);
    // The public fixture distinguishes bright and dark inputs; comparison to
    // an input-independent output would not reveal cross-frame contamination.
    if (expected[0].class_id == expected[1].class_id)
        throw std::runtime_error("Fixture must distinguish images");
    constexpr int workers = 8, repeats = 500;
    std::atomic<int> ready{0}, mismatches{0}, failures{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            ++ready;
            while (!start.load()) std::this_thread::yield();
            for (int i = 0; i < repeats; ++i) {
                const size_t sample = size_t(worker + i) % images.size();
                try {
                    if (!Same(model.Classify(images[sample]), expected[sample])) ++mismatches;
                } catch (...) { ++failures; }
            }
        });
    }
    while (ready.load() != workers) std::this_thread::yield();
    start = true;
    for (auto& thread : threads) thread.join();
    if (mismatches || failures)
        throw std::runtime_error("Concurrent classification changed predictions: mismatches=" +
                                 std::to_string(mismatches.load()) + ", failures=" +
                                 std::to_string(failures.load()));
    for (size_t i = 0; i < images.size(); ++i)
        if (!Same(model.Classify(images[i]), expected[i]))
            throw std::runtime_error("Serial prediction changed after concurrent calls");
    std::cout << "PASS: " << (encrypted ? "encrypted" : "plain")
              << " shared-instance " << workers * repeats << " predictions byte exact\n";
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Fixture directory required");
        const auto root = std::filesystem::u8path(argv[1]);
        Exercise(root, false);
        Exercise(root, true);
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
