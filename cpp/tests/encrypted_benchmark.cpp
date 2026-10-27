// Target-PC A/B benchmark: identical graph/settings, alternating order, no I/O
// inside the timed inference loop. Startup decryption is reported separately.
#include "vision_inference.h"
#include "model_crypto.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now()-start).count(); }
static nlohmann::json stats(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return {{"p50_ms", values[values.size()/2]}, {"p95_ms", values[size_t((values.size()-1)*.95)]},
            {"mean_ms", std::accumulate(values.begin(), values.end(), 0.0)/values.size()}};
}
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::invalid_argument("Usage: encrypted_benchmark plain.json encrypted.dvsenc model.key");
        VisionInference plain, encrypted;
        auto start = Clock::now();
        if (!plain.InitializeFromJson(argv[1])) throw std::runtime_error("Plain model failed");
        const double plain_startup = ms(start);
        start = Clock::now();
        {
            auto key = dvs_crypto::ReadKey(std::filesystem::u8path(argv[3]));
            const dvs_crypto::Package package(std::filesystem::u8path(argv[2]), key);
            dvs_crypto::Wipe(key.data(), key.size());
            if (!encrypted.InitializeFromPackage(package)) throw std::runtime_error("Encrypted model failed");
        }
        const double encrypted_startup = ms(start);
        if (plain.GetConfig().task != "classify") throw std::invalid_argument("This timing harness expects classification");
        const auto& cfg = plain.GetConfig();
        cv::Mat frame((std::max)(cfg.input_height, cfg.crop_height), (std::max)(cfg.input_width, cfg.crop_width),
                      CV_MAKETYPE(CV_8U, cfg.input_channels));
        cv::RNG rng(42); rng.fill(frame, cv::RNG::UNIFORM, 0, 256);
        for (int i = 0; i < 30; ++i) { plain.Classify(frame); encrypted.Classify(frame); }
        std::vector<double> p, e, pm, em;
        for (int i = 0; i < 400; ++i) {
            ClassifyResult a, b;
            auto one = [&]{ auto t=Clock::now(); a=plain.Classify(frame); p.push_back(ms(t)); pm.push_back(a.model_ms); };
            auto two = [&]{ auto t=Clock::now(); b=encrypted.Classify(frame); e.push_back(ms(t)); em.push_back(b.model_ms); };
            if (i%2) { two(); one(); } else { one(); two(); }
            if (a.class_id != b.class_id || a.probabilities != b.probabilities)
                throw std::runtime_error("Encrypted output changed");
        }
        std::cout << nlohmann::json{{"runs", 400}, {"plain_startup_ms", plain_startup},
            {"encrypted_startup_ms", encrypted_startup}, {"plain_pipeline", stats(p)}, {"encrypted_pipeline", stats(e)},
            {"plain_model", stats(pm)}, {"encrypted_model", stats(em)}, {"outputs_byte_equal", true},
            {"threads", cfg.num_threads}, {"optimizer", cfg.ort_graph_optimization_level},
            {"ort", OrtGetApiBase()->GetVersionString()}}.dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
