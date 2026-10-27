// Stress the shared model with different inputs, sizes and padded row pitches.
#include "../example_paths.h"
#include "classifier.h"
#include <array>
#include <atomic>
#include <iostream>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

struct Frame {
    int width, height;
    size_t pitch;
    std::vector<uint8_t> data;
};
static bool Same(const dvs_bw8::Result& a, const dvs_bw8::Result& b) {
    return a.class_id == b.class_id && a.class_name == b.class_name &&
        a.confidence == b.confidence && a.probabilities == b.probabilities;
}
static void Exercise(dvs_bw8::Classifier& model) {
    constexpr int workers = 8, repeats = 200;
    std::array<Frame, workers> frames;
    std::array<dvs_bw8::Result, workers> expected;
    std::array<std::unique_ptr<dvs_bw8::Classifier::Context>, workers> contexts;
    for (int i = 0; i < workers; ++i) {
        auto& frame = frames[i];
        frame.width = 7 + i * 31; frame.height = 11 + i * 29; frame.pitch = frame.width + 17;
        frame.data.resize(frame.pitch * frame.height, 123); // Padding differs from image pixels.
        for (int y = 0; y < frame.height; ++y)
            std::fill_n(frame.data.data() + y * frame.pitch, frame.width, i % 2 ? 255 : 0);
        expected[i] = model.InferBW8(frame.data.data(), frame.width, frame.height, frame.pitch);
        contexts[i] = model.CreateContext();
    }
    if (expected[0].class_id == expected[1].class_id) throw std::runtime_error("Input-independent fixture");
    // Default calls automatically borrow independent scratch. Explicit Contexts
    // remain supported for callers that want to preallocate their own workspace.
    // Reusing one Context concurrently is also safe and serializes that Context.
    for (int mode = 0; mode < 3; ++mode) {
        std::atomic<int> ready{0}, errors{0};
        std::atomic<bool> start{false};
        std::vector<std::thread> threads;
        for (int worker = 0; worker < workers; ++worker) threads.emplace_back([&, worker] {
            ++ready;
            while (!start.load()) std::this_thread::yield();
            for (int i = 0; i < repeats; ++i) {
                const int sample = (i + worker) % workers;
                const auto& frame = frames[sample];
                try {
                    const auto result = mode == 0
                        ? model.InferBW8(frame.data.data(), frame.width, frame.height, frame.pitch)
                        : model.InferBW8(*contexts[mode == 1 ? worker : 0], frame.data.data(), frame.width, frame.height, frame.pitch);
                    if (!Same(result, expected[sample])) ++errors;
                } catch (...) { ++errors; }
            }
        });
        while (ready.load() != workers) std::this_thread::yield();
        start = true;
        for (auto& thread : threads) thread.join();
        if (errors.load()) throw std::runtime_error("Concurrent BW8 prediction mismatch");
    }
#ifdef _OPENMP
    std::atomic<int> errors{0};
#pragma omp parallel for num_threads(workers)
    for (int i = 0; i < workers * repeats; ++i) {
        const int sample = i % workers;
        const auto& frame = frames[sample];
        try {
            const auto result = model.InferBW8(*contexts[omp_get_thread_num()], frame.data.data(),
                                              frame.width, frame.height, frame.pitch);
            if (!Same(result, expected[sample])) ++errors;
        } catch (...) { ++errors; }
    }
    if (errors.load()) throw std::runtime_error("OpenMP prediction mismatch");
#endif
    // Caller counts change over time; no OMP thread ID or capacity is supplied
    // to the model. Busy workspaces must never be handed to a second caller.
    int dynamic_calls = 0;
    for (int callers : {3, 11, 2, 17}) {
        dynamic_calls += callers * 32;
        std::atomic<int> ready{0}, errors{0};
        std::atomic<bool> start{false};
        std::vector<std::thread> threads;
        for (int worker = 0; worker < callers; ++worker) threads.emplace_back([&, worker] {
            ++ready;
            while (!start.load()) std::this_thread::yield();
            for (int i = 0; i < 32; ++i) {
                const int sample = (i + worker) % workers;
                const auto& frame = frames[sample];
                try {
                    if (!Same(model.InferBW8(frame.data.data(), frame.width, frame.height, frame.pitch), expected[sample]))
                        ++errors;
                } catch (...) { ++errors; }
            }
        });
        while (ready.load() != callers) std::this_thread::yield();
        start = true;
        for (auto& thread : threads) thread.join();
        if (errors.load()) throw std::runtime_error("Changing caller count mixed predictions");
    }
#ifdef _OPENMP
    // Two independent OMP teams reuse local thread IDs concurrently. The
    // automatic API must not confuse team 0's worker 0 with team 1's worker 0.
    std::atomic<int> team_errors{0};
    std::array<std::thread, 2> teams;
    for (int team = 0; team < 2; ++team) teams[team] = std::thread([&, team] {
        const int team_workers = team == 0 ? 3 : 5;
#pragma omp parallel for num_threads(team_workers)
        for (int i = 0; i < 160; ++i) {
            const int sample = (i + team) % workers;
            const auto& frame = frames[sample];
            try {
                if (!Same(model.InferBW8(frame.data.data(), frame.width, frame.height, frame.pitch), expected[sample]))
                    ++team_errors;
            } catch (...) { ++team_errors; }
        }
    });
    for (auto& team : teams) team.join();
    if (team_errors.load()) throw std::runtime_error("Independent OMP teams mixed predictions");
#endif
    // Validation failure must release the Context's lock for the next call.
    bool rejected = false;
    try { model.InferBW8(*contexts[0], nullptr, 7, 11, 24); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Invalid buffer accepted");
    const auto& frame = frames[0];
    if (!Same(model.InferBW8(*contexts[0], frame.data.data(), frame.width, frame.height, frame.pitch), expected[0]))
        throw std::runtime_error("Context failed after an invalid buffer");
    // A failed automatic call returns its borrowed workspace through RAII.
    rejected = false;
    try { model.InferBW8(nullptr, 7, 11, 24); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected || !Same(model.InferBW8(frame.data.data(), frame.width, frame.height, frame.pitch), expected[0]))
        throw std::runtime_error("Automatic workspace failed after an invalid buffer");
    std::cout << "PASS: shared model, default/shared/per-worker Contexts, "
              << 3 * workers * repeats + dynamic_calls << " std::thread calls";
#ifdef _OPENMP
    std::cout << ", " << workers * repeats + 320 << " OpenMP calls";
#endif
    std::cout << '\n';
}
int main() {
    try {
        const auto root = example::ExecutableDirectory() / "assets";
        dvs_bw8::Classifier plain(root / "test.json");
        Exercise(plain);
        auto key = dvs_crypto::ReadKey(root / "example-only.key");
        struct EraseKey { dvs_crypto::Key& k; ~EraseKey() { dvs_crypto::Wipe(k.data(), k.size()); } } erase{key};
        dvs_bw8::Classifier encrypted(root / "test.dvsenc", key);
        dvs_crypto::Wipe(key.data(), key.size());
        Exercise(encrypted);
        auto foreign = plain.CreateContext();
        bool rejected = false;
        uint8_t pixel = 255;
        try { encrypted.InferBW8(*foreign, &pixel, 1, 1, 1); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Foreign Context accepted");
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
