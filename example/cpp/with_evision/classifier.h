#pragma once
#if defined(_MSC_VER)
// Manual VS projects must also add the SDK lib folder to the library search path.
#pragma comment(lib, "onnxruntime.lib")
#if _MSC_VER < 1916
#error Use Visual Studio 2017 15.9 (v141 14.16) or newer.
#endif
#if !defined(_MSVC_LANG) || _MSVC_LANG < 201703L
#error Enable C++17: Project Properties > C/C++ > Language > /std:c++17.
#endif
#if !defined(__cpp_noexcept_function_type)
#error Enable /Zc:noexceptTypes and remove /Zc:noexceptTypes- before including ONNX Runtime.
#endif
#endif
#include "bw8_preprocess.h"
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
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
//
// Startup (once):  JSON -> ModelConfig -> Ort::Session -> bound tensors -> warm-up
// Per inference:   BW8 pixels -> Preprocessor -> input_ -> Run -> output_ -> softmax
class Classifier {
public:
    explicit Classifier(const std::filesystem::path& json_path,
                        Image startup_image = {nullptr, 0, 0, 0}) {
        const auto config = LoadConfig(json_path);
        width_ = config.width; height_ = config.height;
        input_name_ = config.input_name; output_name_ = config.output_name;
        preprocess_ = std::make_unique<Preprocessor>(width_, height_, config.mean, config.stddev,
                                                     config.crop_width, config.crop_height);
        // Resolve default labels once so inference never formats strings.
        names_ = config.class_names;
        if (names_.empty())
            for (int i = 0; i < config.classes; ++i) names_.push_back("class_" + std::to_string(i));

        session_ = std::make_unique<Ort::Session>(env_, config.model_path.c_str(), CreateSessionOptions(config));
        VerifySignature(config);
        BindTensors(config.channels, config.classes);

        // Warm-up on the caller's frame, or on a black frame large enough for the crop.
        std::vector<uint8_t> zero;
        if (!startup_image.data && startup_image.width == 0 && startup_image.height == 0 && startup_image.stride == 0) {
            const int w = (std::max)(width_, config.crop_width), h = (std::max)(height_, config.crop_height);
            zero.resize(static_cast<size_t>(w) * h);
            startup_image = {zero.data(), w, h, static_cast<size_t>(w)};
        }
        warmup_ms_ = InferBW8(startup_image.data, startup_image.width, startup_image.height, startup_image.stride).inference_ms;
    }

    Result InferBW8(const void* pixels, int width, int height, size_t row_pitch) {
        const auto start = Clock::now();
        preprocess_->Run({pixels, width, height, row_pitch}, input_);
        const auto prepared = Clock::now();
        // input_/output_ are pre-bound to the tensors, so Run writes in place.
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
        const auto inferred = Clock::now();
        Result result{};
        Softmax(result);
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
    using Json = nlohmann::json;

    // Everything read from the exported JSON, validated before ONNX Runtime starts.
    struct ModelConfig {
        int schema = 0, width = 0, height = 0, channels = 0, classes = 0;
        int crop_width = 0, crop_height = 0, threads = 0;
        std::vector<float> mean, stddev;
        std::vector<std::string> class_names;
        std::string input_name, output_name;
        std::filesystem::path model_path;
        Json runtime;  // optional "onnxruntime" tuning object
    };

    static double Milliseconds(Clock::duration duration) { return std::chrono::duration<double, std::milli>(duration).count(); }

    // Integer in [low, high]; rejects floats, booleans and strings.
    static int ReadInt(const Json& value, const char* name, double low = 0, double high = INT32_MAX) {
        if (!value.is_number_integer() || value.get<double>() < low || value.get<double>() > high)
            throw std::invalid_argument(std::string("Invalid integer: ") + name);
        return value.get<int>();
    }

    static ModelConfig LoadConfig(const std::filesystem::path& json_path) {
        std::ifstream file(json_path);
        if (!file) throw std::runtime_error("Cannot open model JSON: " + json_path.u8string());
        const auto doc = Json::parse(file);
        ModelConfig config;
        config.schema = ReadInt(doc.at("schema_version"), "schema_version");
        if ((config.schema != 5 && config.schema != 6) || doc.at("task") != "classify")
            throw std::invalid_argument("Expected Studio schema 5/6 classification JSON");
        config.width = ReadInt(doc.at("input_width"), "input_width");
        config.height = ReadInt(doc.at("input_height"), "input_height");
        config.channels = ReadInt(doc.at("input_channels"), "input_channels");
        config.classes = ReadInt(doc.at("num_classes"), "num_classes");
        if (config.width < 1 || config.height < 1 || config.width > 65536 || config.height > 65536 ||
            (config.channels != 1 && config.channels != 3) || config.classes < 1)
            throw std::invalid_argument("Invalid model dimensions");

        CheckPreprocessingContract(doc.at("preprocessing"), config.channels);
        if (doc.contains("postprocessing") && doc.at("postprocessing").value("output", std::string("logits")) != "logits")
            throw std::invalid_argument("Expected classification logits");
        const auto& prep = doc.at("preprocessing");
        if (prep.contains("center_crop") && !prep.at("center_crop").is_null()) {
            const auto& crop = prep.at("center_crop");
            try {
                config.crop_width = ReadInt(crop.at("width"), "center_crop.width", 1, 65536);
                config.crop_height = ReadInt(crop.at("height"), "center_crop.height", 1, 65536);
            } catch (const std::invalid_argument&) {
                throw std::invalid_argument("Invalid center crop");
            }
        }

        config.mean = doc.at("normalize_mean").get<std::vector<float>>();
        config.stddev = doc.at("normalize_std").get<std::vector<float>>();
        if (config.mean.size() != static_cast<size_t>(config.channels))
            throw std::invalid_argument("Wrong normalization channels");
        config.class_names = doc.value("class_names", std::vector<std::string>{});
        if (!config.class_names.empty() && config.class_names.size() != static_cast<size_t>(config.classes))
            throw std::invalid_argument("Wrong class names count");
        config.input_name = doc.at("input_name").get<std::string>();
        config.output_name = doc.at("output_name").get<std::string>();

        config.runtime = doc.value("onnxruntime", Json::object());
        if (config.schema == 6 && (!config.runtime.contains("graph_optimization_level") || !doc.contains("num_threads")))
            throw std::invalid_argument("Missing verified ONNX Runtime settings");
        config.threads = doc.contains("num_threads") ? ReadInt(doc.at("num_threads"), "num_threads") : 0;

        config.model_path = std::filesystem::u8path(doc.at("model_path").get<std::string>());
        if (config.model_path.is_relative()) config.model_path = json_path.parent_path() / config.model_path;
        return config;
    }

    // Only the resize/normalization contract implemented by Preprocessor is accepted.
    static void CheckPreprocessingContract(const Json& prep, int channels) {
        const std::string color = channels == 1 ? "GRAY" : "RGB";
        if (prep.at("resize_implementation") != "opencv_linear_exact_v1" ||
            prep.at("interpolation") != "INTER_LINEAR_EXACT" || prep.at("antialias") != false ||
            prep.value("layout", std::string("NCHW")) != "NCHW" ||
            prep.value("value_scale", 255.0) != 255.0 ||
            prep.value("resize", std::string("bilinear")) != "bilinear" ||
            prep.value("color_order", color) != color)
            throw std::invalid_argument("Unsupported preprocessing contract; re-export from Studio");
    }

    static Ort::SessionOptions CreateSessionOptions(const ModelConfig& config) {
        const auto& tuning = config.runtime;
        Ort::SessionOptions options;
        const auto level = tuning.value("graph_optimization_level", std::string("all"));
        if (level != "all" && level != "basic" && level != "disabled")
            throw std::invalid_argument("Unsupported graph optimization level");
        options.SetGraphOptimizationLevel(level == "all" ? ORT_ENABLE_ALL : level == "basic" ? ORT_ENABLE_BASIC : ORT_DISABLE_ALL);
        options.SetExecutionMode(ORT_SEQUENTIAL);
        if (config.threads) options.SetIntraOpNumThreads(config.threads);
        if (tuning.contains("allow_spinning"))
            options.AddConfigEntry("session.intra_op.allow_spinning", tuning.at("allow_spinning").get<bool>() ? "1" : "0");
        if (tuning.contains("dynamic_block_base")) {
            const auto& v = tuning.at("dynamic_block_base");
            if (!v.is_number_integer() || v.get<double>() < 0 || v.get<double>() > INT32_MAX)
                throw std::invalid_argument("Invalid dynamic block base");
            options.AddConfigEntry("session.dynamic_block_base", std::to_string(v.get<int>()).c_str());
        }
        return options;
    }

    // The ONNX graph must be exactly [1, C, H, W] float -> [1, classes] float.
    void VerifySignature(const ModelConfig& config) const {
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
            throw std::invalid_argument("Expected one input and one output");
        Ort::AllocatorWithDefaultOptions allocator;
        if (input_name_ != session_->GetInputNameAllocated(0, allocator).get() ||
            output_name_ != session_->GetOutputNameAllocated(0, allocator).get())
            throw std::invalid_argument("Tensor names do not match JSON");
        const auto input_type = session_->GetInputTypeInfo(0), output_type = session_->GetOutputTypeInfo(0);
        const auto input_info = input_type.GetTensorTypeAndShapeInfo(), output_info = output_type.GetTensorTypeAndShapeInfo();
        const auto shape = input_info.GetShape(), output_shape = output_info.GetShape();
        // Negative extents are dynamic axes and accept any value.
        auto matches = [](int64_t actual, int64_t expected) { return actual <= 0 || actual == expected; };
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            shape.size() != 4 || !matches(shape[0], 1) || shape[1] != config.channels ||
            !matches(shape[2], height_) || !matches(shape[3], width_) ||
            output_shape.size() != 2 || !matches(output_shape[0], 1) || !matches(output_shape[1], config.classes))
            throw std::invalid_argument("Tensor shape/type does not match JSON");
    }

    // Tensors wrap input_/output_ directly: no per-call tensor or copy.
    void BindTensors(int channels, int classes) {
        input_.resize(static_cast<size_t>(width_) * height_ * channels); output_.resize(classes);
        const std::array<int64_t, 4> input_shape{1, channels, height_, width_};
        const std::array<int64_t, 2> result_shape{1, classes};
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory, input_.data(), input_.size(), input_shape.data(), 4);
        output_tensor_ = Ort::Value::CreateTensor<float>(memory, output_.data(), output_.size(), result_shape.data(), 2);
    }

    // Numerically stable softmax over output_ (logits). The finiteness check and
    // argmax share one pass; exp/sum and normalization are the other two.
    void Softmax(Result& result) const {
        size_t best = 0;
        for (size_t i = 0; i < output_.size(); ++i) {
            if (!std::isfinite(output_[i])) throw std::runtime_error("Non-finite model output");
            if (output_[i] > output_[best]) best = i;  // first maximum wins, like max_element
        }
        const float peak = output_[best];
        result.probabilities.resize(output_.size());
        float sum = 0;
        for (size_t i = 0; i < output_.size(); ++i) {
            result.probabilities[i] = std::exp(output_[i] - peak);
            sum += result.probabilities[i];
        }
        for (auto& value : result.probabilities) value /= sum;
        result.class_id = static_cast<int>(best);
        result.class_name = names_[best];
        result.confidence = result.probabilities[best];
    }

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "BW8"};  // must outlive session_
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
