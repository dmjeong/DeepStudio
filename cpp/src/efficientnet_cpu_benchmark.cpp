// C++17 synchronous and in-process background latency measurement.
#include "classification_worker.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <nlohmann/json.hpp>

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

namespace {
int Positive(const std::string& text) {
    size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != text.size() || value < 1) throw std::invalid_argument("Expected a positive integer: " + text);
    return value;
}
Json Stats(std::vector<double> values, double target) {
    const auto raw = values;
    std::sort(values.begin(), values.end());
    auto percentile = [&](double p) {
        const double position = (values.size() - 1) * p;
        const auto lower = static_cast<size_t>(position);
        const auto upper = std::min(lower + 1, values.size() - 1);
        return values[lower] + (values[upper] - values[lower]) * (position - lower);
    };
    return {{"p50_ms", percentile(.5)}, {"p95_ms", percentile(.95)}, {"p99_ms", percentile(.99)},
            {"mean_ms", std::accumulate(values.begin(), values.end(), 0.0) / values.size()},
            {"max_ms", values.back()}, {"samples_ms", raw},
            {"within_target_fraction", static_cast<double>(std::count_if(values.begin(), values.end(),
                 [target](double value) { return value <= target; })) / values.size()}};
}
}

int main(int argc, char** argv) {
    try {
        std::string config_path, image_path, output, runtime = "onnxruntime", cpu_label = "unspecified";
        int threads = -1, warmup = 30, runs = 1000;
        double target = 8;
        bool synthetic = false, background = false, require_target = false;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("Missing value for " + flag);
                return argv[i];
            };
            if (flag == "--config") config_path = value();
            else if (flag == "--image") image_path = value();
            else if (flag == "--synthetic") synthetic = true;
            else if (flag == "--output") output = value();
            else if (flag == "--runtime") runtime = value();
            else if (flag == "--cpu-label") cpu_label = value();
            else if (flag == "--threads") threads = Positive(value());
            else if (flag == "--warmup") warmup = Positive(value());
            else if (flag == "--runs") runs = Positive(value());
            else if (flag == "--background") background = true;
            else if (flag == "--require-target") require_target = true;
            else if (flag == "--target-ms") {
                const auto text = value(); size_t used = 0;
                target = std::stod(text, &used);
                if (used != text.size() || !std::isfinite(target) || target <= 0)
                    throw std::invalid_argument("Positive finite target required.");
            } else if (flag == "--help") {
                std::cout << "--config model.json (--synthetic | --image frame.png) [--runtime onnxruntime|openvino] "
                          << "[--threads 4 --warmup 30 --runs 1000 --target-ms 8 --require-target --background --output report.json --cpu-label label]\n";
                return 0;
            } else throw std::invalid_argument("Unknown argument: " + flag);
        }
        if (config_path.empty() || (synthetic == !image_path.empty()))
            throw std::invalid_argument("Provide --config and exactly one of --synthetic or --image.");
        cv::setNumThreads(1);
        auto engine = std::make_unique<VisionInference>();
        if (!engine->InitializeFromJson(config_path, runtime, threads))
            throw std::runtime_error("Model initialization failed.");
        const auto config = engine->GetConfig();
        threads = config.num_threads; // An explicit CLI value overrides JSON; omission preserves it.
        if (config.task != "classify") throw std::invalid_argument("Classification model required.");
        cv::Mat image;
        if (synthetic) {
            image.create(std::max(config.input_height, config.crop_height), std::max(config.input_width, config.crop_width),
                         CV_MAKETYPE(CV_8U, config.input_channels));
            cv::RNG random(42);
            random.fill(image, cv::RNG::UNIFORM, 0, 256);
        } else {
            image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
            if (image.empty()) throw std::invalid_argument("Cannot decode input image.");
        }
        std::unique_ptr<ClassificationWorker> worker;
        if (background) {
            // Avoid keeping a second runtime's idle pool alive during measurement.
            engine.reset();
            worker = std::make_unique<ClassificationWorker>(config_path, runtime, threads, 2, warmup);
            worker->Ready().get();
        } else for (int i = 0; i < warmup; ++i) engine->Classify(image);
        std::vector<double> wall, pipeline, preprocess, model, postprocess, queue;
        ClassifyResult last;
        for (int i = 0; i < runs; ++i) {
            const auto start = Clock::now();
            double queue_ms = 0;
            if (worker) {
                auto result = worker->Submit(static_cast<uint64_t>(i), image).get();
                last = std::move(result.prediction);
                queue_ms = result.queue_ms;
            } else last = engine->Classify(image);
            wall.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
            if (last.class_id < 0) throw std::runtime_error("Inference returned no class.");
            pipeline.push_back(last.inference_ms);
            preprocess.push_back(last.preprocess_ms);
            model.push_back(last.model_ms);
            postprocess.push_back(last.postprocess_ms);
            queue.push_back(queue_ms);
        }
        Json report{{"schema_version", 1}, {"runtime", runtime}, {"threads", threads}, {"cpu_label", cpu_label},
                    {"cpp_standard", __cplusplus}, {"opencv_version", CV_VERSION},
                    {"onnxruntime_version", OrtGetApiBase()->GetVersionString()},
                    {"batch_size", 1}, {"input_shape", {1,config.input_channels,config.input_height,config.input_width}},
                    {"source_shape", {image.rows,image.cols,image.channels()}}, {"num_classes",config.num_classes},
                    {"precision", "FP32"}, {"synthetic_input", synthetic}, {"background", background},
                    {"warmup",warmup}, {"runs",runs}, {"scope", background
                        ? "Submit frame copy + queue + preprocessing + model + postprocessing + future wakeup; excludes decode/setup"
                        : "preprocessing + model + postprocessing; excludes decode/setup"},
                    {"wall", Stats(wall,target)}, {"pipeline", Stats(pipeline,target)},
                    {"preprocess", Stats(preprocess,target)}, {"model", Stats(model,target)},
                    {"postprocess", Stats(postprocess,target)}, {"queue", Stats(queue,target)},
                    {"last_class_id",last.class_id}, {"last_probabilities",last.probabilities}};
        report["target"] = {{"metric","wall.p95_ms"}, {"threshold_ms",target},
                            {"met_on_this_machine", report["wall"]["p95_ms"].get<double>() <= target},
                            {"all_measured_requests_within_target",report["wall"]["max_ms"].get<double>() <= target}};
        report["onnxruntime_options"] = {{"allow_spinning", config.ort_allow_spinning},
                                         {"dynamic_block_base", config.ort_dynamic_block_base}};
        if (!output.empty()) {
            const auto path = std::filesystem::u8path(output);
            if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
            std::ofstream file(path);
            if (!file) throw std::runtime_error("Cannot create report file.");
            file << report.dump(2) << '\n';
            if (!file) throw std::runtime_error("Cannot write report file.");
        }
        // Full raw timings are in the file. Keep terminal output short.
        report.erase("preprocess"); report.erase("model"); report.erase("postprocess"); report.erase("queue");
        report["wall"].erase("samples_ms"); report["pipeline"].erase("samples_ms");
        std::cout << report.dump(2) << '\n';
        return require_target && !report["target"]["met_on_this_machine"].get<bool>() ? 2 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
