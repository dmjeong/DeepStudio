// No arguments. One model, one private Context per worker. With /openmp on
// VS2017 this uses OpenMP; other builds use equivalent std::thread workers.
#include "../example_paths.h"
#include "classifier.h"
#include <array>
#include <atomic>
#include <iostream>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    try {
        constexpr int workers = 2, image_count = 16;
        const auto root = example::ExecutableDirectory() / "assets";
        auto key = dvs_crypto::ReadKey(root / "example-only.key");
        struct EraseKey { dvs_crypto::Key& k; ~EraseKey() { dvs_crypto::Wipe(k.data(), k.size()); } } erase{key};
        dvs_bw8::Classifier model(root / "test.dvsenc", key); // Decrypt/load/warm up once.
        dvs_crypto::Wipe(key.data(), key.size());
        std::array<std::vector<uint8_t>, image_count> frames;
        for (int i = 0; i < image_count; ++i) frames[i].resize(240 * 224, i % 2 ? 0 : 255);
        std::array<std::unique_ptr<dvs_bw8::Classifier::Context>, workers> contexts;
        for (auto& context : contexts) {
            context = model.CreateContext();
            model.InferBW8(*context, frames[0].data(), 224, 224, 240); // Prepare each worker's scratch once.
        }
        std::array<dvs_bw8::Result, image_count> results;
        std::atomic<int> errors{0};
        auto infer = [&](int index, int worker) {
            try {
                results[index] = model.InferBW8(*contexts[worker], frames[index].data(), 224, 224, 240);
                // In your application: model.InferEvision(*contexts[worker], rois[index]);
            } catch (...) { ++errors; }
        };
#ifdef _OPENMP
#pragma omp parallel for num_threads(workers)
        for (int i = 0; i < image_count; ++i) infer(i, omp_get_thread_num());
#else
        std::array<std::thread, workers> threads;
        for (int worker = 0; worker < workers; ++worker)
            threads[worker] = std::thread([&, worker] {
                for (int i = worker; i < image_count; i += workers) infer(i, worker);
            });
        for (auto& thread : threads) thread.join();
#endif
        if (errors.load()) throw std::runtime_error("Parallel inference failed");
        for (int i = 0; i < image_count; ++i) {
            if (results[i].class_id != i % 2) throw std::runtime_error("Parallel prediction mismatch");
            std::cout << i << " " << results[i].class_name << " " << results[i].confidence << '\n';
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
