// No command-line arguments. Paths are relative to evaluation.json beside the executable.
#include "vision_inference.h"
#include "metrics.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif
namespace fs = std::filesystem;
using nlohmann::json;

fs::path ExecutableDirectory() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    auto n = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (!n || n >= buffer.size()) throw std::runtime_error("Cannot locate executable");
    return fs::path(std::wstring(buffer.data(), n)).parent_path();
#elif defined(__APPLE__)
    uint32_t size = 0; _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size)) throw std::runtime_error("Cannot locate executable");
    return fs::canonical(buffer.data()).parent_path();
#else
    return fs::canonical("/proc/self/exe").parent_path();
#endif
}

cv::Mat ReadImage(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open image: " + path.u8string());
    auto length = stream.tellg();
    if (length <= 0) throw std::runtime_error("Empty image: " + path.u8string());
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), length)) throw std::runtime_error("Image read failed");
    auto image = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
    if (image.empty()) throw std::runtime_error("Cannot decode image: " + path.u8string());
    return image;
}

void WriteImage(const fs::path& path, const cv::Mat& image) {
    fs::create_directories(path.parent_path());
    std::vector<uint8_t> bytes;
    if (!cv::imencode(".png", image, bytes)) throw std::runtime_error("Cannot encode mask");
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!stream) throw std::runtime_error("Cannot save mask: " + path.u8string());
}

double Percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    double i = (values.size()-1)*p; size_t low = static_cast<size_t>(i);
    return values[low] + (values[std::min(low+1, values.size()-1)]-values[low])*(i-low);
}

int main() {
    try {
        cv::setNumThreads(1);
        // Optional environment override is for automated tests; normal use needs no arguments.
        const char* override_path = std::getenv("DVS_EVAL_CONFIG");
        auto config_path = override_path ? fs::absolute(fs::u8path(override_path)) : ExecutableDirectory()/"evaluation.json";
        std::ifstream settings(config_path);
        if (!settings) throw std::runtime_error("Cannot open " + config_path.u8string());
        json job; settings >> job;
        auto resolve = [&](const char* field) { return fs::absolute(config_path.parent_path()/fs::u8path(job.at(field).get<std::string>())); };
        auto model_path = resolve("model_json"), images = resolve("images"), masks = resolve("masks"), output = resolve("output");
        const int threads = job.value("threads", 4), warmup = job.value("warmup", 3), repeat = job.value("repeat", 5);
        if (threads < 1 || warmup < 1 || repeat < 1) throw std::invalid_argument("threads, warmup and repeat must be positive");
        if (!fs::is_directory(images) || !fs::is_directory(masks)) throw std::runtime_error("images/masks folder missing");
        if (fs::exists(output)) throw std::runtime_error("Output already exists; choose a new output folder");
        std::vector<fs::path> paths; std::set<fs::path> stems;
        for (const auto& entry : fs::recursive_directory_iterator(images)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return char(std::tolower(c)); });
            if (ext != ".png" && ext != ".bmp" && ext != ".jpg" && ext != ".jpeg" && ext != ".tif" && ext != ".tiff") continue;
            auto relative = entry.path().lexically_relative(images); relative.replace_extension(".png");
            if (!stems.insert(relative).second) throw std::runtime_error("Duplicate image stem: " + relative.u8string());
            if (!fs::is_regular_file(masks/relative)) throw std::runtime_error("Missing ground truth: " + (masks/relative).u8string());
            paths.push_back(entry.path());
        }
        if (paths.empty()) throw std::runtime_error("No images");
        std::sort(paths.begin(), paths.end());
        VisionInference engine;
        if (!engine.InitializeFromJson(model_path.u8string(), "onnxruntime", threads)) throw std::runtime_error("Model initialization failed");
        const auto& model = engine.GetConfig();
        if (model.task != "segment") throw std::invalid_argument("A semantic segmentation export is required");
        SegmentationMetrics aggregate(model.num_classes);
        json per_image = json::array(); std::vector<double> all_times;
        fs::create_directories(output);
        for (const auto& path : paths) {
            auto relative = path.lexically_relative(images); relative.replace_extension(".png");
            cv::Mat image = ReadImage(path), truth = ReadImage(masks/relative);
            if (truth.size() != image.size() || truth.type() != CV_8UC1)
                throw std::invalid_argument("Ground truth must be original-size grayscale class-ID PNG: " + relative.u8string());
            // Prepare once for each input shape before recording any timed calls.
            for (int i=0; i<warmup; ++i) engine.Segment(image);
            SegmentResult result; std::vector<double> times;
            for (int i=0; i<repeat; ++i) {
                auto begin = std::chrono::steady_clock::now();
                result = engine.Segment(image);
                times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
            }
            SegmentationMetrics sample(model.num_classes);
            sample.Add(truth, result.mask, result.valid_mask);
            aggregate.Add(truth, result.mask, result.valid_mask);
            auto row = sample.Report(model.class_names);
            if (row["evaluated_pixels"].get<uint64_t>() == 0) throw std::runtime_error("No evaluable pixels: " + relative.u8string());
            row["image"] = path.lexically_relative(images).generic_u8string();
            row["p50_ms"] = Percentile(times, .5); row["p95_ms"] = Percentile(times, .95); row["times_ms"] = times;
            per_image.push_back(row); all_times.insert(all_times.end(), times.begin(), times.end());
            WriteImage(output/"predictions"/relative, result.mask);
            // Original background remains visible; only predicted foreground is tinted.
            cv::Mat preview;
            if (image.depth() == CV_8U) {
                if (image.channels() == 1) cv::cvtColor(image, preview, cv::COLOR_GRAY2BGR);
                else if (image.channels() == 4) cv::cvtColor(image, preview, cv::COLOR_BGRA2BGR);
                else preview = image.clone();
                cv::Mat tinted; cv::addWeighted(preview, .7, result.color_mask, .3, 0, tinted);
                tinted.copyTo(preview, (result.mask != 0) & result.valid_mask);
                WriteImage(output/"overlays"/relative, preview);
            }
            std::cout << row["image"] << " : " << row["p50_ms"] << " ms\n";
        }
        auto report = aggregate.Report(model.class_names);
        report["images"] = per_image; report["model_json"] = model_path.u8string();
        report["threads"] = threads; report["warmup"] = warmup; report["repeat"] = repeat;
        report["p50_ms"] = Percentile(all_times,.5); report["p95_ms"] = Percentile(all_times,.95);
        report["mask_policy"] = "deployment mask after exported defect_rules; only valid_mask pixels; ignore GT 255";
        report["timing_scope"] = "Segment call: preprocessing + ONNX + postprocessing; excludes decode, load, warmup, metrics and output writes";
        std::ofstream csv(output/"classes.csv", std::ios::binary); csv << "\xEF\xBB\xBF" "class_id,name,iou,dice,tp,fp,fn,gt_pixels,predicted_pixels\r\n";
        for (const auto& row : report["per_class"]) {
            std::string name = row["name"].get<std::string>(), escaped;
            for (char c : name) { escaped += c; if (c=='"') escaped += c; }
            csv << row["class_id"] << ",\"" << escaped << "\",";
            for (const char* key : {"iou","dice","tp","fp","fn","gt_pixels","predicted_pixels"}) {
                if (!row[key].is_null()) csv << row[key];
                csv << (std::string(key)=="predicted_pixels" ? "\r\n" : ",");
            }
        }
        csv.close(); if (!csv) throw std::runtime_error("Cannot write classes.csv");
        // Publish the final report only after every image and CSV was saved.
        auto pending_report = output/"report.json.tmp";
        std::ofstream json_out(pending_report); json_out << report.dump(2) << '\n'; json_out.close();
        if (!json_out) throw std::runtime_error("Cannot write report.json");
        fs::rename(pending_report, output/"report.json");
        std::cout << "Foreground mIoU: " << report["miou_foreground"] << "\nSaved: " << output.u8string() << '\n';
        return 0; // Evaluation completed, not a declaration that the model is production-ready.
    } catch (const std::exception& e) { std::cerr << "Evaluation failed: " << e.what() << '\n'; return 2; }
}
