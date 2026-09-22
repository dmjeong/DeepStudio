#pragma once
#include "bw8_preprocess.h"
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace dvs_bw8 {
struct Result {
    int class_id;
    std::string class_name;
    float confidence;
    std::vector<float> probabilities;
    double preprocess_ms, model_ms, postprocess_ms, inference_ms;
};

// No OpenCV, Python, or application SDK dependency. An EImageBW8 remains owned
// by the caller. Keep its pixels alive and unchanged until InferBW8 returns.
// Create once at startup; do not call the same instance concurrently.
class Classifier {
public:
    explicit Classifier(const std::filesystem::path& json_path,
                        Image startup_image = {nullptr, 0, 0, 0}) {
        std::ifstream file(json_path);
        if (!file) throw std::runtime_error("Cannot open model JSON: " + json_path.u8string());
        const auto doc = nlohmann::json::parse(file);
        auto integer = [&](const char* name) {
            const auto& v = doc.at(name);
            if (!v.is_number_integer() || v.get<double>() < 0 || v.get<double>() > INT32_MAX)
                throw std::invalid_argument(std::string("Invalid integer: ") + name);
            return v.get<int>();
        };
        const int schema = integer("schema_version");
        if ((schema != 5 && schema != 6) || doc.at("task") != "classify")
            throw std::invalid_argument("Expected Studio schema 5/6 classification JSON");
        width_ = integer("input_width"); height_ = integer("input_height");
        const int channels = integer("input_channels"), classes = integer("num_classes");
        if (width_ < 1 || height_ < 1 || width_ > 65536 || height_ > 65536 ||
            (channels != 1 && channels != 3) || classes < 1)
            throw std::invalid_argument("Invalid model dimensions");
        const auto& prep = doc.at("preprocessing");
        if (prep.at("resize_implementation") != "opencv_linear_exact_v1" ||
            prep.at("interpolation") != "INTER_LINEAR_EXACT" || prep.at("antialias") != false ||
            prep.value("layout", std::string("NCHW")) != "NCHW" ||
            prep.value("value_scale", 255.0) != 255.0 ||
            prep.value("resize", std::string("bilinear")) != "bilinear" ||
            prep.value("color_order", channels == 1 ? std::string("GRAY") : std::string("RGB")) !=
                (channels == 1 ? "GRAY" : "RGB"))
            throw std::invalid_argument("Unsupported preprocessing contract; re-export from Studio");
        if (doc.contains("postprocessing") && doc.at("postprocessing").value("output", std::string("logits")) != "logits")
            throw std::invalid_argument("Expected classification logits");
        int crop_width = 0, crop_height = 0;
        if (prep.contains("center_crop") && !prep.at("center_crop").is_null()) {
            const auto& crop = prep.at("center_crop");
            if (!crop.at("width").is_number_integer() || !crop.at("height").is_number_integer() ||
                crop.at("width").get<double>() < 1 || crop.at("width").get<double>() > 65536 ||
                crop.at("height").get<double>() < 1 || crop.at("height").get<double>() > 65536)
                throw std::invalid_argument("Invalid center crop");
            crop_width = crop.at("width").get<int>(); crop_height = crop.at("height").get<int>();
        }
        auto mean = doc.at("normalize_mean").get<std::vector<float>>();
        auto stddev = doc.at("normalize_std").get<std::vector<float>>();
        if (mean.size() != static_cast<size_t>(channels)) throw std::invalid_argument("Wrong normalization channels");
        preprocess_ = std::make_unique<Preprocessor>(width_, height_, mean, stddev, crop_width, crop_height);
        names_ = doc.value("class_names", std::vector<std::string>{});
        if (!names_.empty() && names_.size() != static_cast<size_t>(classes))
            throw std::invalid_argument("Wrong class names count");
        input_name_ = doc.at("input_name").get<std::string>();
        output_name_ = doc.at("output_name").get<std::string>();
        Ort::SessionOptions options;
        const auto tuning = doc.value("onnxruntime", nlohmann::json::object());
        if (schema == 6 && (!tuning.contains("graph_optimization_level") || !doc.contains("num_threads")))
            throw std::invalid_argument("Missing verified ONNX Runtime settings");
        const auto level = tuning.value("graph_optimization_level", std::string("all"));
        if (level != "all" && level != "basic" && level != "disabled")
            throw std::invalid_argument("Unsupported graph optimization level");
        options.SetGraphOptimizationLevel(level == "all" ? ORT_ENABLE_ALL : level == "basic" ? ORT_ENABLE_BASIC : ORT_DISABLE_ALL);
        options.SetExecutionMode(ORT_SEQUENTIAL);
        const int threads = doc.contains("num_threads") ? integer("num_threads") : 0;
        if (threads) options.SetIntraOpNumThreads(threads);
        if (tuning.contains("allow_spinning"))
            options.AddConfigEntry("session.intra_op.allow_spinning", tuning.at("allow_spinning").get<bool>() ? "1" : "0");
        if (tuning.contains("dynamic_block_base")) {
            const auto& v = tuning.at("dynamic_block_base");
            if (!v.is_number_integer() || v.get<double>() < 0 || v.get<double>() > INT32_MAX)
                throw std::invalid_argument("Invalid dynamic block base");
            options.AddConfigEntry("session.dynamic_block_base", std::to_string(v.get<int>()).c_str());
        }
        auto model_path = std::filesystem::u8path(doc.at("model_path").get<std::string>());
        if (model_path.is_relative()) model_path = json_path.parent_path() / model_path;
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
            throw std::invalid_argument("Expected one input and one output");
        Ort::AllocatorWithDefaultOptions allocator;
        if (input_name_ != session_->GetInputNameAllocated(0, allocator).get() ||
            output_name_ != session_->GetOutputNameAllocated(0, allocator).get())
            throw std::invalid_argument("Tensor names do not match JSON");
        const auto input_type = session_->GetInputTypeInfo(0), output_type = session_->GetOutputTypeInfo(0);
        const auto input_info = input_type.GetTensorTypeAndShapeInfo(), output_info = output_type.GetTensorTypeAndShapeInfo();
        const auto shape = input_info.GetShape(), output_shape = output_info.GetShape();
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            shape.size() != 4 || (shape[0] > 0 && shape[0] != 1) || shape[1] != channels ||
            (shape[2] > 0 && shape[2] != height_) || (shape[3] > 0 && shape[3] != width_) ||
            output_shape.size() != 2 || (output_shape[0] > 0 && output_shape[0] != 1) ||
            (output_shape[1] > 0 && output_shape[1] != classes))
            throw std::invalid_argument("Tensor shape/type does not match JSON");
        input_.resize(static_cast<size_t>(width_) * height_ * channels); output_.resize(classes);
        const std::array<int64_t, 4> input_shape{1, channels, height_, width_};
        const std::array<int64_t, 2> result_shape{1, classes};
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory, input_.data(), input_.size(), input_shape.data(), 4);
        output_tensor_ = Ort::Value::CreateTensor<float>(memory, output_.data(), output_.size(), result_shape.data(), 2);
        std::vector<uint8_t> zero;
        if (!startup_image.data && startup_image.width == 0 && startup_image.height == 0 && startup_image.stride == 0) {
            const int w = (std::max)(width_, crop_width), h = (std::max)(height_, crop_height);
            zero.resize(static_cast<size_t>(w) * h);
            startup_image = {zero.data(), w, h, static_cast<size_t>(w)};
        }
        warmup_ms_ = InferBW8(startup_image.data, startup_image.width, startup_image.height, startup_image.stride).inference_ms;
    }

    Result InferBW8(const void* pixels, int width, int height, size_t row_pitch) {
        const auto start = Clock::now();
        preprocess_->Run({pixels, width, height, row_pitch}, input_);
        const auto prepared = Clock::now();
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
        const auto inferred = Clock::now();
        for (float value : output_) if (!std::isfinite(value)) throw std::runtime_error("Non-finite model output");
        Result result{};
        result.class_id = static_cast<int>(std::max_element(output_.begin(), output_.end()) - output_.begin());
        result.class_name = names_.empty() ? "class_" + std::to_string(result.class_id) : names_[result.class_id];
        result.probabilities.resize(output_.size());
        float sum = 0;
        for (size_t i = 0; i < output_.size(); ++i) {
            result.probabilities[i] = std::exp(output_[i] - output_[result.class_id]);
            sum += result.probabilities[i];
        }
        for (auto& value : result.probabilities) value /= sum;
        result.confidence = result.probabilities[result.class_id];
        const auto end = Clock::now();
        result.preprocess_ms = Milliseconds(prepared - start);
        result.model_ms = Milliseconds(inferred - prepared);
        result.postprocess_ms = Milliseconds(end - inferred);
        result.inference_ms = Milliseconds(end - start);
        return result;
    }

    // Instantiated in the user's eVision project; accepts EImageBW8/EROIBW8.
    template<class BW8Image> Result InferEvision(BW8Image& image) {
        if (image.GetBitsPerPixel() != 8 || image.GetColPitch() != 1)
            throw std::invalid_argument("InferEvision requires BW8, not a color or BW16 image");
        return InferBW8(image.GetImagePtr(0, 0), image.GetWidth(), image.GetHeight(), image.GetRowPitch());
    }
    double WarmupMilliseconds() const { return warmup_ms_; }
private:
    using Clock = std::chrono::steady_clock;
    static double Milliseconds(Clock::duration duration) { return std::chrono::duration<double, std::milli>(duration).count(); }
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "BW8"};
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Preprocessor> preprocess_;
    int width_ = 0, height_ = 0;
    std::string input_name_, output_name_;
    std::vector<std::string> names_;
    std::vector<float> input_, output_;
    Ort::Value input_tensor_{nullptr}, output_tensor_{nullptr};
    double warmup_ms_ = 0;
};
} // namespace dvs_bw8
