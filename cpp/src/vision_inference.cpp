/**
 * @file vision_inference.cpp
 * @brief Vision ONNX Runtime CPU 추론 엔진 구현
 *
 * 핵심 흐름:
 *   Initialize() → Preprocess() → Session.Run() → Postprocess()
 *
 * CPU 최적화 전략:
 * ┌──────────────────────────────────────────────────────────────┐
 * │ 1. ORT_ENABLE_ALL 그래프 최적화 (상수 폴딩, 노드 융합)     │
 * │ 2. Inter/Intra Op 스레드 수 자동 최적화                     │
 * │ 3. 메모리 패턴 최적화 (enable_mem_pattern)                  │
 * │ 4. CPU 메모리 아레나 사용 (enable_cpu_mem_arena)             │
 * └──────────────────────────────────────────────────────────────┘
 */

#include "vision_inference.h"
#include "opencv_preprocess.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <stdexcept>
#include <limits>
#include <utility>
#include <nlohmann/json.hpp>
#ifdef VISION_WITH_OPENVINO
#include <openvino/openvino.hpp>
#endif

struct VisionInference::OpenVINOState
{
#ifdef VISION_WITH_OPENVINO
    ov::Core core;
    ov::CompiledModel compiled;
    ov::InferRequest request;
#endif
};

// One owner calls Classify at a time (as enforced by ClassificationWorker).
// Tensor wrappers and buffers live for the model lifetime, never for one frame.
struct VisionInference::ClassificationState
{
    std::vector<float> input, output;
    Ort::Value input_tensor{nullptr}, output_tensor{nullptr};
    cv::Mat converted, resized;
    explicit ClassificationState(const InferenceConfig& config)
        : input(static_cast<size_t>(config.input_channels) * config.input_height * config.input_width),
          output(config.num_classes)
    {
        const std::vector<int64_t> shape{1, config.input_channels, config.input_height, config.input_width};
        const std::vector<int64_t> result_shape{1, config.num_classes};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        input_tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
        output_tensor = Ort::Value::CreateTensor<float>(memory, output.data(), output.size(), result_shape.data(), result_shape.size());
    }
};


// ═══════════════════════════════════════════════════
//  생성자 / 소멸자
// ═══════════════════════════════════════════════════

VisionInference::VisionInference()
    : m_env(ORT_LOGGING_LEVEL_WARNING, "Vision")  // 로깅 레벨 설정
{
}

VisionInference::~VisionInference()
{
    Release();
}


// ═══════════════════════════════════════════════════
//  초기화
// ═══════════════════════════════════════════════════

namespace {
GraphOptimizationLevel GraphLevel(const std::string& name)
{
    if (name == "all") return GraphOptimizationLevel::ORT_ENABLE_ALL;
    if (name == "basic") return GraphOptimizationLevel::ORT_ENABLE_BASIC;
    if (name == "disabled") return GraphOptimizationLevel::ORT_DISABLE_ALL;
    throw std::invalid_argument("graph_optimization_level must be all, basic or disabled.");
}

void ValidateConfig(const InferenceConfig& config)
{
    GraphLevel(config.ort_graph_optimization_level);
    if (config.ort_allow_spinning < -1 || config.ort_allow_spinning > 1 || config.ort_dynamic_block_base < 0)
        throw std::invalid_argument("Invalid ONNX Runtime threading options.");
    if (config.runtime != "onnxruntime" && config.runtime != "openvino")
        throw std::invalid_argument("Runtime must be onnxruntime or openvino.");
    if (config.runtime == "openvino" && config.task != "classify")
        throw std::invalid_argument("OpenVINO currently supports classification only.");
    if (config.crop_width < 0 || config.crop_height < 0 ||
        config.crop_width > 65536 || config.crop_height > 65536 ||
        ((config.crop_width == 0) != (config.crop_height == 0)))
        throw std::invalid_argument("Invalid center crop dimensions.");
    if (config.backend != "custom" && config.backend != "builtin" && config.backend != "patchcore" &&
        config.backend != "redetr_v4" &&
        !(config.backend == "efficientnet" && config.task == "classify"))
        throw std::invalid_argument("Supported backends: custom, builtin, patchcore, redetr_v4 and efficientnet classification.");
    if (config.backend == "patchcore" && config.task != "anomaly")
        throw std::invalid_argument("PatchCore backend requires anomaly task.");
    if (config.backend == "redetr_v4" && config.task != "detect")
        throw std::invalid_argument("Re-DETR backend requires detect task.");
    if (config.resize_mode != "stretch" || config.classification_output != "logits")
        throw std::invalid_argument("Custom and EfficientNet models require stretch resize and logits.");
    if (config.task != "classify" && config.task != "segment" &&
        config.task != "detect" && config.task != "anomaly")
        throw std::invalid_argument("Supported tasks: classify, segment, detect, anomaly.");
    if ((config.input_channels != 1 && config.input_channels != 3) ||
        config.input_height <= 0 || config.input_width <= 0 || config.num_classes <= 0 ||
        (config.task == "segment" && config.num_classes > 256) || config.num_threads < 0)
        throw std::invalid_argument("Invalid input shape, class count or thread count.");
    if (config.normalize_mean.size() != static_cast<size_t>(config.input_channels) ||
        config.normalize_std.size() != static_cast<size_t>(config.input_channels))
        throw std::invalid_argument("Normalization length must match input channels.");
    for (int c = 0; c < config.input_channels; ++c)
        if (!std::isfinite(config.normalize_mean[c]) || !std::isfinite(config.normalize_std[c]) ||
            config.normalize_std[c] <= 0)
            throw std::invalid_argument("Normalization values must be finite; std must be positive.");
    if (config.task == "detect") {
        if (config.detection_box_encoding != "normalized_cxcywh" &&
            config.detection_box_encoding != "normalized_xyxy")
            throw std::invalid_argument("Unsupported detection box encoding.");
        const bool redetr = config.backend == "redetr_v4";
        if ((!redetr && config.detection_objectness != "sigmoid") ||
            (redetr && config.detection_objectness != "none") ||
            (config.detection_class_scores != "sigmoid" && config.detection_class_scores != "softmax"))
            throw std::invalid_argument("Unsupported detection score activation.");
        if (!std::isfinite(config.detection_confidence_threshold) ||
            config.detection_confidence_threshold < 0.0f || config.detection_confidence_threshold > 1.0f ||
            !std::isfinite(config.detection_iou_threshold) || config.detection_iou_threshold < 0.0f ||
            config.detection_iou_threshold > 1.0f || config.detection_max_detections < 1)
            throw std::invalid_argument("Invalid detection thresholds or max detections.");
    }
    if (config.task == "anomaly" && !std::isfinite(config.anomaly_threshold))
        throw std::invalid_argument("Invalid anomaly threshold.");
    if (config.model_path.empty() || config.input_name.empty() || config.output_name.empty())
        throw std::invalid_argument("Model path and tensor names are required.");
    if (!config.output_names.empty() && (config.output_names.front() != config.output_name ||
        std::any_of(config.output_names.begin(), config.output_names.end(),
                    [](const std::string& name) { return name.empty(); })))
        throw std::invalid_argument("Output tensor names are invalid.");
    if (!config.class_names.empty() && config.class_names.size() != static_cast<size_t>(config.num_classes))
        throw std::invalid_argument("Class names do not match class count.");
    if (config.model_implementation_version != 0)
    {
        const bool legacy = config.model_implementation_version == 1 && config.input_channels == 1;
        if (config.backend != "efficientnet" ||
            (config.model_implementation_version != 1 && config.model_implementation_version != 2) ||
            config.stem_in_channels != (legacy ? 3 : config.input_channels) ||
            config.input_adapter != (legacy ? "rgb_repeat_inside_model" : "native"))
            throw std::invalid_argument("EfficientNet input/stem channel contract mismatch.");
    }
}

void ValidateOutput(const Ort::Value& output, const InferenceConfig& config, size_t rank)
{
    auto info = output.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    const int64_t expected_channels = config.task == "anomaly" ? config.input_channels : config.num_classes;
    const bool channel_shape_valid = rank == 3
        ? shape.size() == 3 && shape[2] == static_cast<int64_t>(5 + config.num_classes)
        : shape.size() == rank && shape[1] == expected_channels;
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != rank || shape[0] != 1 || !channel_shape_valid ||
        (rank == 3 && shape[1] <= 0) || (rank == 4 && (shape[2] <= 0 || shape[3] <= 0)))
        throw std::runtime_error("Model output does not match the deployment contract.");
    const float* values = output.GetTensorData<float>();
    for (size_t i = 0; i < info.GetElementCount(); ++i)
        if (!std::isfinite(values[i]))
            throw std::runtime_error("Model returned non-finite values.");
}
} // namespace

bool VisionInference::Initialize(const InferenceConfig& config)
{
    try
    {
        ValidateConfig(config);
        if (config.runtime == "openvino")
        {
#ifdef VISION_WITH_OPENVINO
            auto state = std::make_unique<OpenVINOState>();
#if defined(_WIN32) && defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
            auto model = state->core.read_model(std::filesystem::u8path(config.model_path).wstring());
#else
            auto model = state->core.read_model(config.model_path);
#endif
            if (model->inputs().size() != 1 || model->outputs().size() != 1)
                throw std::invalid_argument("Expected one OpenVINO input and output.");
            const auto input = model->input();
            const auto output = model->output();
            const ov::Shape expected{1, static_cast<size_t>(config.input_channels),
                                     static_cast<size_t>(config.input_height), static_cast<size_t>(config.input_width)};
            const auto shape = input.get_partial_shape();
            if (!input.get_names().count(config.input_name) || !output.get_names().count(config.output_name) ||
                input.get_element_type() != ov::element::f32 || output.get_element_type() != ov::element::f32 ||
                shape.rank().is_dynamic() || shape.rank().get_length() != 4 ||
                !shape.compatible(ov::PartialShape(expected)))
                throw std::invalid_argument("OpenVINO tensor names, types or input shape mismatch.");
            for (size_t i = 1; i < expected.size(); ++i)
                if (shape[i].is_dynamic() || shape[i].get_length() != static_cast<int64_t>(expected[i]))
                    throw std::invalid_argument("Only batch may be dynamic.");
            model->reshape(expected);
            if (model->output().get_shape() != ov::Shape{1, static_cast<size_t>(config.num_classes)})
                throw std::invalid_argument("OpenVINO output shape mismatch.");
            ov::AnyMap properties{ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                  ov::hint::inference_precision(ov::element::f32), ov::num_streams(1)};
            if (config.num_threads > 0) properties[ov::inference_num_threads.name()] = config.num_threads;
            state->compiled = state->core.compile_model(model, "CPU", properties);
            state->request = state->compiled.create_infer_request();
            InferenceConfig validated = config;
            m_config = std::move(validated);
            m_openvino = std::move(state);
            m_session.reset();
            m_classification.reset();
            m_bInitialized = true;
            return true;
#else
            throw std::invalid_argument("Rebuild with -DVISION_WITH_OPENVINO=ON to use OpenVINO.");
#endif
        }
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphLevel(config.ort_graph_optimization_level));
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        if (config.ort_allow_spinning >= 0)
            options.AddConfigEntry("session.intra_op.allow_spinning", config.ort_allow_spinning ? "1" : "0");
        if (config.ort_dynamic_block_base > 0)
            options.AddConfigEntry("session.dynamic_block_base", std::to_string(config.ort_dynamic_block_base).c_str());
        if (config.num_threads > 0) options.SetIntraOpNumThreads(config.num_threads);
        options.EnableMemPattern();
        options.EnableCpuMemArena();
        if (config.enable_profiling) options.EnableProfiling(ORT_TSTR("vision_profile"));
        const auto path = std::filesystem::u8path(config.model_path);
        auto session = std::make_unique<Ort::Session>(m_env, path.c_str(), options);
        const std::vector<std::string> expected_outputs = config.output_names.empty()
            ? std::vector<std::string>{config.output_name} : config.output_names;
        if (session->GetInputCount() != 1 || session->GetOutputCount() != expected_outputs.size())
            throw std::invalid_argument("Model input/output count does not match the deployment contract.");
        auto input_name = session->GetInputNameAllocated(0, m_allocator);
        if (config.input_name != input_name.get())
            throw std::invalid_argument("Tensor names do not match model.");
        for (size_t index = 0; index < expected_outputs.size(); ++index) {
            auto output_name = session->GetOutputNameAllocated(index, m_allocator);
            if (expected_outputs[index] != output_name.get())
                throw std::invalid_argument("Tensor names do not match model.");
        }
        const auto input_type = session->GetInputTypeInfo(0);
        const auto input = input_type.GetTensorTypeAndShapeInfo();
        const auto shape = input.GetShape();
        if (input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.size() != 4 ||
            (shape[0] > 0 && shape[0] != 1) || shape[1] != config.input_channels ||
            (shape[2] > 0 && shape[2] != config.input_height) ||
            (shape[3] > 0 && shape[3] != config.input_width))
            throw std::invalid_argument("Input shape/type does not match model.");
        const auto output_type = session->GetOutputTypeInfo(0);
        const auto output = output_type.GetTensorTypeAndShapeInfo();
        const auto out_shape = output.GetShape();
        const size_t rank = config.task == "classify" ? 2 : (config.task == "detect" ? 3 : 4);
        if (config.backend == "patchcore") {
            if (session->GetOutputCount() != 2 || output.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                out_shape.size() != 1 || (out_shape[0] > 0 && out_shape[0] != 1))
                throw std::invalid_argument("PatchCore score output does not match the deployment contract.");
            const auto map = session->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo();
            const auto map_shape = map.GetShape();
            // Some ONNX Runtime builds report a rank-0 type descriptor for a
            // constant secondary output even though the runtime tensor has its
            // declared shape.  Validate known metadata here and always repeat
            // the complete contract check on the actual Run() result below.
            if (!map_shape.empty() &&
                (map.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || map_shape.size() != 4 ||
                 map_shape[0] != 1 || (map_shape[1] > 0 && map_shape[1] != 1) ||
                 (map_shape[2] > 0 && map_shape[2] != config.input_height) ||
                 (map_shape[3] > 0 && map_shape[3] != config.input_width)))
                throw std::invalid_argument("PatchCore map output does not match the deployment contract.");
        } else if (config.backend == "redetr_v4") {
            if (session->GetOutputCount() != 2 || output.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                out_shape.size() != 3 || (out_shape[0] > 0 && out_shape[0] != 1) ||
                (out_shape[2] > 0 && out_shape[2] != 4))
                throw std::invalid_argument("Re-DETR boxes output does not match the deployment contract.");
            const auto logits = session->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo();
            const auto logits_shape = logits.GetShape();
            // As with PatchCore's constant map output, some ORT builds expose
            // no static type descriptor for a constant secondary output.  If
            // metadata is known, validate it here; the runtime tensor is
            // checked again in Detect() in all cases.
            if (!logits_shape.empty() &&
                (logits.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || logits_shape.size() != 3 ||
                 (logits_shape[0] > 0 && logits_shape[0] != 1) ||
                 (logits_shape[1] > 0 && out_shape[1] > 0 && logits_shape[1] != out_shape[1]) ||
                 (logits_shape[2] > 0 && logits_shape[2] != config.num_classes)))
                throw std::invalid_argument("Re-DETR logits output does not match the deployment contract.");
        } else if (output.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                   out_shape.size() != rank ||
                   (config.task == "classify" && out_shape[1] > 0 && out_shape[1] != config.num_classes) ||
                   (config.task == "segment" && out_shape[1] > 0 && out_shape[1] != config.num_classes) ||
                   (config.task == "detect" && out_shape[2] > 0 && out_shape[2] != 5 + config.num_classes) ||
                   (config.task == "anomaly" && out_shape[1] > 0 && out_shape[1] != config.input_channels))
            throw std::invalid_argument("Output shape/type does not match task.");
        // 전체 후보를 검증한 뒤 교체한다. 실패하면 기존 모델과 설정을 함께 유지한다.
        auto classification = config.task == "classify" ? std::make_unique<ClassificationState>(config) : nullptr;
        InferenceConfig validated = config;
        m_config = std::move(validated);
        m_session = std::move(session);
        m_classification = std::move(classification);
        m_openvino.reset();
        m_bInitialized = true;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Vision] Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool VisionInference::InitializeFromJson(const std::string& config_path,
                                        const std::string& runtime, int num_threads)
{
    try
    {
        const auto path = std::filesystem::u8path(config_path);
        std::ifstream file(path);
        if (!file) throw std::runtime_error("Cannot open configuration file.");
        const auto doc = nlohmann::json::parse(file);
        if (!doc.is_object()) throw std::invalid_argument("Configuration must be an object.");
        if (!doc.contains("schema_version") || !doc.at("schema_version").is_number_integer() ||
            (doc.at("schema_version") != 5 && doc.at("schema_version") != 6))
            throw std::invalid_argument("Re-export this model: supported schemas are 5 and 6.");
        if (doc.at("schema_version") == 6 &&
            (!doc.contains("onnxruntime") || !doc.at("onnxruntime").is_object() ||
             !doc.at("onnxruntime").contains("graph_optimization_level") || !doc.contains("num_threads")))
            throw std::invalid_argument("Schema 6 requires the verified ONNX Runtime settings.");
        InferenceConfig config;
        if (num_threads < -1) throw std::invalid_argument("Invalid thread override.");
        config.runtime = runtime.empty() ? doc.value("runtime", std::string("onnxruntime")) : runtime;
        if (doc.contains("onnxruntime")) {
            const auto& tuning = doc.at("onnxruntime");
            if (!tuning.is_object()) throw std::invalid_argument("onnxruntime must be an object.");
            config.ort_graph_optimization_level = tuning.value("graph_optimization_level", std::string("all"));
            if (tuning.contains("allow_spinning")) {
                if (!tuning.at("allow_spinning").is_boolean())
                    throw std::invalid_argument("allow_spinning must be boolean.");
                config.ort_allow_spinning = tuning.at("allow_spinning").get<bool>() ? 1 : 0;
            }
            if (tuning.contains("dynamic_block_base")) {
                const auto& value = tuning.at("dynamic_block_base");
                if (!value.is_number_integer() || value.get<double>() < 0 ||
                    value.get<double>() > std::numeric_limits<int>::max())
                    throw std::invalid_argument("dynamic_block_base must be a nonnegative int.");
                config.ort_dynamic_block_base = value.get<int>();
            }
        }
        config.backend = doc.value("backend", std::string("custom"));
        auto model_path = std::filesystem::u8path(doc.at("model_path").get<std::string>());
        if (model_path.is_relative()) model_path = path.parent_path() / model_path;
        config.model_path = model_path.lexically_normal().u8string();
        config.task = doc.at("task").get<std::string>();
        // JSON 정수 필드에 소수나 문자열이 들어오는 경우도 거부한다.
        auto integer = [&doc](const char* key) {
            const auto& value = doc.at(key);
            if (!value.is_number_integer()) throw std::invalid_argument(std::string(key) + " must be an integer.");
            const auto number = value.get<int64_t>();
            if (number < std::numeric_limits<int>::min() || number > std::numeric_limits<int>::max())
                throw std::invalid_argument("Integer field out of range.");
            return static_cast<int>(number);
        };
        config.num_classes = integer("num_classes");
        config.input_channels = integer("input_channels");
        config.input_height = integer("input_height");
        config.input_width = integer("input_width");
        config.input_name = doc.at("input_name").get<std::string>();
        config.output_name = doc.at("output_name").get<std::string>();
        if (doc.contains("output_names")) {
            config.output_names = doc.at("output_names").get<std::vector<std::string>>();
            if (config.output_names.empty() || config.output_names.front() != config.output_name)
                throw std::invalid_argument("output_names must start with output_name.");
        }
        if (config.output_names.empty()) config.output_names.push_back(config.output_name);
        if (config.backend == "efficientnet" && doc.contains("model_config"))
        {
            const auto& model = doc.at("model_config");
            if (!model.is_object() || !model.at("implementation_version").is_number_integer())
                throw std::invalid_argument("EfficientNet model definition is invalid.");
            const auto version = model.at("implementation_version").get<int64_t>();
            if (version != 1 && version != 2)
                throw std::invalid_argument("Unsupported EfficientNet implementation version.");
            config.model_implementation_version = static_cast<int>(version);
            const bool legacy = version == 1 && config.input_channels == 1;
            config.stem_in_channels = legacy ? 3 : config.input_channels;
            config.input_adapter = model.value("input_adapter", legacy ? std::string("rgb_repeat_inside_model") : std::string("native"));
            for (const auto& item : {std::make_pair("in_channels", config.input_channels),
                                     std::make_pair("stem_in_channels", config.stem_in_channels)})
                if (model.contains(item.first) && (!model.at(item.first).is_number_integer() ||
                    model.at(item.first).get<int64_t>() != item.second))
                    throw std::invalid_argument("EfficientNet input/stem metadata mismatch.");
        }
        config.normalize_mean = doc.at("normalize_mean").get<std::vector<float>>();
        config.normalize_std = doc.at("normalize_std").get<std::vector<float>>();
        config.class_names = doc.value("class_names", std::vector<std::string>{});
        if (doc.contains("num_threads")) config.num_threads = integer("num_threads");
        if (num_threads >= 0) config.num_threads = num_threads;
        config.enable_profiling = doc.value("enable_profiling", false);
        if (doc.contains("postprocessing"))
        {
            const auto& post = doc.at("postprocessing");
            if (!post.is_object()) throw std::invalid_argument("postprocessing must be an object.");
            config.classification_output = post.value("output", std::string("logits"));
            if (config.task == "detect")
            {
                config.detection_box_encoding = post.value("box_format", std::string("normalized_cxcywh"));
                config.detection_objectness = post.value("objectness", config.backend == "redetr_v4" ? std::string("none") : std::string("sigmoid"));
                config.detection_class_scores = post.value("class_scores", std::string("sigmoid"));
                config.detection_confidence_threshold = post.value("confidence_threshold", 0.25f);
                config.detection_iou_threshold = post.value("iou_threshold", 0.5f);
                config.detection_max_detections = post.value("max_detections", 300);
            }
            if (config.task == "anomaly")
                config.anomaly_threshold = post.value("threshold", 0.0f);
        }
        if (doc.contains("preprocessing"))
        {
            const auto& prep = doc.at("preprocessing");
            if (prep.is_object() && prep.contains("in_channels") &&
                (!prep.at("in_channels").is_number_integer() ||
                 prep.at("in_channels").get<int64_t>() != config.input_channels))
                throw std::invalid_argument("Conflicting preprocessing channels.");
            if (prep.is_object() && prep.contains("grayscale_adapter") && config.model_implementation_version != 0 &&
                prep.at("grayscale_adapter").get<std::string>() != config.input_adapter)
                throw std::invalid_argument("Conflicting grayscale adapter.");
            if (!prep.is_object() ||
                prep.value("resize_implementation", std::string("opencv_linear_exact_v1")) != "opencv_linear_exact_v1" ||
                prep.value("antialias", true) || prep.value("layout", std::string("NCHW")) != "NCHW")
                throw std::invalid_argument("Unsupported resize implementation.");
            if (!prep.contains("resize_implementation") || !prep.contains("antialias") ||
                prep.value("interpolation", std::string()) != "INTER_LINEAR_EXACT")
                throw std::invalid_argument("Schema 5 requires explicit resize metadata.");
            if (prep.contains("center_crop") && !prep.at("center_crop").is_null())
            {
                const auto& crop = prep.at("center_crop");
                if (!crop.is_object() || crop.size() != 2 ||
                    !crop.at("width").is_number_integer() || !crop.at("height").is_number_integer())
                    throw std::invalid_argument("Invalid center crop contract.");
                auto width = crop.at("width").get<int64_t>();
                auto height = crop.at("height").get<int64_t>();
                if (width < 1 || width > 65536 || height < 1 || height > 65536)
                    throw std::invalid_argument("Invalid center crop dimensions.");
                config.crop_width = static_cast<int>(width);
                config.crop_height = static_cast<int>(height);
            }
            if (prep.value("value_scale", 255.0) != 255.0 ||
                prep.value("resize", std::string("bilinear")) != "bilinear" ||
                prep.value("color_order", config.input_channels == 1 ? std::string("GRAY") : std::string("RGB")) !=
                    (config.input_channels == 1 ? "GRAY" : "RGB"))
                throw std::invalid_argument("Unsupported preprocessing contract.");
        }
        if (!doc.contains("preprocessing"))
            throw std::invalid_argument("Schema 5 requires preprocessing metadata.");
        return Initialize(config);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Vision] Configuration error: " << e.what() << std::endl;
        return false;
    }
}

void VisionInference::Release()
{
    m_classification.reset();
    m_session.reset();
    m_openvino.reset();
    m_bInitialized = false;
}


// ═══════════════════════════════════════════════════
//  전처리
// ═══════════════════════════════════════════════════

std::vector<float> VisionInference::Preprocess(const cv::Mat& source)
{
    std::vector<float> tensor(static_cast<size_t>(m_config.input_channels) * m_config.input_height * m_config.input_width);
    cv::Mat converted, resized;
    PreprocessInto(source, tensor.data(), tensor.size(), converted, resized);
    return tensor;
}

void VisionInference::PreprocessInto(const cv::Mat& source, float* data, size_t count,
                                   cv::Mat& converted, cv::Mat& resized)
{
    cv::Mat image = source;
    if (m_config.crop_width > 0)
    {
        if (source.cols < m_config.crop_width || source.rows < m_config.crop_height)
            throw std::invalid_argument("Center crop exceeds original image size.");
        image = source(cv::Rect((source.cols-m_config.crop_width)/2,
                               (source.rows-m_config.crop_height)/2,
                               m_config.crop_width, m_config.crop_height));
    }
    if (image.empty() || image.depth() != CV_8U)
        throw std::invalid_argument("Input must be a nonempty 8-bit image; map height data explicitly.");
    const int source_channels = image.channels();
    const int channels = m_config.input_channels;
    if (source_channels != 1 && source_channels != 3 && source_channels != 4)
        throw std::invalid_argument("Supported image channels: gray, BGR, BGRA.");
    const cv::Mat* pixels = &converted;
    if (channels == 1) {
        if (source_channels == 1) pixels = &image;
        else cv::cvtColor(image, converted, source_channels == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    } else {
        cv::cvtColor(image, converted, source_channels == 1 ? cv::COLOR_GRAY2RGB :
                                      source_channels == 3 ? cv::COLOR_BGR2RGB : cv::COLOR_BGRA2RGB);
    }
    vision_preprocess::resize_normalize_into(*pixels, m_config.input_width, m_config.input_height,
        m_config.normalize_mean, m_config.normalize_std, data, count, resized);
}


// ═══════════════════════════════════════════════════
//  후처리 유틸리티
// ═══════════════════════════════════════════════════

std::vector<float> VisionInference::Softmax(const std::vector<float>& logits)
{
    /**
     * Softmax 함수: 로짓 → 확률 분포
     *
     * softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)
     *
     * max를 빼서 수치적 안정성 확보 (overflow 방지)
     */
    std::vector<float> probs(logits.size());

    // 수치 안정성을 위해 최대값을 뺌
    float max_val = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i)
    {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }

    // 정규화
    for (auto& p : probs)
    {
        p /= sum;
    }

    return probs;
}

cv::Mat VisionInference::ArgmaxMask(const float* output_data,
                                     int num_classes,
                                     int height, int width)
{
    /**
     * 세그멘테이션 출력 → Argmax 마스크
     *
     * 출력 텐서: (1, C, H, W) - 각 클래스의 로짓
     * 마스크: (H, W) - 각 픽셀의 클래스 인덱스
     *
     * 각 픽셀에서 가장 큰 로짓을 가진 클래스가 해당 픽셀의 예측 클래스
     */
    cv::Mat mask(height, width, CV_8UC1);

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            float max_val = -std::numeric_limits<float>::infinity();
            int max_cls = 0;

            // 모든 클래스에 대해 최대값 찾기
            for (int c = 0; c < num_classes; ++c)
            {
                // CHW 레이아웃: [c * H * W + h * W + w]
                float val = output_data[c * height * width + h * width + w];
                if (val > max_val)
                {
                    max_val = val;
                    max_cls = c;
                }
            }

            mask.at<uchar>(h, w) = static_cast<uchar>(max_cls);
        }
    }

    return mask;
}

cv::Mat VisionInference::ColorizeMask(const cv::Mat& mask, int num_classes)
{
    /**
     * 클래스 인덱스 마스크 → 컬러 시각화
     *
     * 고정 팔레트 사용 (최대 20 클래스)
     */
    // 클래스별 색상 팔레트 (BGR)
    static const cv::Vec3b palette[] = {
        {0, 0, 0},       // 0: 배경 (검정)
        {0, 0, 255},     // 1: 빨강
        {0, 255, 0},     // 2: 초록
        {255, 0, 0},     // 3: 파랑
        {0, 255, 255},   // 4: 노랑
        {255, 0, 255},   // 5: 마젠타
        {255, 255, 0},   // 6: 시안
        {128, 0, 0},     // 7: 남색
        {0, 128, 0},     // 8: 어두운 초록
        {0, 0, 128},     // 9: 어두운 빨강
        {128, 128, 0},   // 10
        {0, 128, 128},   // 11
        {128, 0, 128},   // 12
        {192, 192, 192}, // 13
        {128, 128, 128}, // 14
        {64, 0, 0},      // 15
        {0, 64, 0},      // 16
        {0, 0, 64},      // 17
        {64, 64, 0},     // 18
        {0, 64, 64},     // 19
    };

    cv::Mat color_mask(mask.size(), CV_8UC3);
    for (int h = 0; h < mask.rows; ++h)
    {
        for (int w = 0; w < mask.cols; ++w)
        {
            int cls = mask.at<uchar>(h, w);
            cls = std::min(cls, 19); // 팔레트 범위 제한
            color_mask.at<cv::Vec3b>(h, w) = palette[cls];
        }
    }

    return color_mask;
}


// ═══════════════════════════════════════════════════
//  Classification 추론
// ═══════════════════════════════════════════════════

ClassifyResult VisionInference::Classify(const cv::Mat& image)
{
    ClassifyResult result;

    if (!m_bInitialized)
    {
        std::cerr << "[Vision] 모델이 초기화되지 않았습니다." << std::endl;
        return result;
    }

    auto start = std::chrono::steady_clock::now();

    if (m_config.task != "classify") throw std::logic_error("Classify requires a classification model.");

    // ── 1. 전처리 ────────────────────────────────────
    std::vector<float> logits;
    std::chrono::steady_clock::time_point model_start, model_end;
    if (m_openvino)
    {
#ifdef VISION_WITH_OPENVINO
        std::vector<float> input_tensor = Preprocess(image);
        auto tensor = m_openvino->request.get_input_tensor();
        std::copy(input_tensor.begin(), input_tensor.end(), tensor.data<float>());
        model_start = std::chrono::steady_clock::now();
        m_openvino->request.infer();
        model_end = std::chrono::steady_clock::now();
        const auto output = m_openvino->request.get_output_tensor();
        if (output.get_element_type() != ov::element::f32 ||
            output.get_shape() != ov::Shape{1, static_cast<size_t>(m_config.num_classes)})
            throw std::runtime_error("OpenVINO output contract mismatch.");
        const auto data = output.data<const float>();
        logits.assign(data, data + m_config.num_classes);
#endif
    }
    else
    {
        auto& state = *m_classification;
        PreprocessInto(image, state.input.data(), state.input.size(), state.converted, state.resized);
        const char* input_names[] = {m_config.input_name.c_str()};
        const char* output_names[] = {m_config.output_name.c_str()};
        model_start = std::chrono::steady_clock::now();
        m_session->Run(Ort::RunOptions{nullptr}, input_names, &state.input_tensor, 1, output_names, &state.output_tensor, 1);
        model_end = std::chrono::steady_clock::now();
        ValidateOutput(state.output_tensor, m_config, 2);
        logits.assign(state.output.begin(), state.output.end());
    }
    if (logits.empty() || std::any_of(logits.begin(), logits.end(), [](float v) { return !std::isfinite(v); }))
        throw std::runtime_error("Invalid classification output.");
    result.probabilities = Softmax(logits);
    // Avoid changing rank when FP32 softmax rounds unequal logits to a tie.
    result.class_id = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    result.confidence = result.probabilities[result.class_id];

    // 클래스 이름 매핑
    if (result.class_id < static_cast<int>(m_config.class_names.size()))
    {
        result.class_name = m_config.class_names[result.class_id];
    }
    else
    {
        result.class_name = "class_" + std::to_string(result.class_id);
    }

    // 소요 시간 계산
    auto end = std::chrono::steady_clock::now();
    result.preprocess_ms = std::chrono::duration<double, std::milli>(model_start - start).count();
    result.model_ms = std::chrono::duration<double, std::milli>(model_end - model_start).count();
    result.postprocess_ms = std::chrono::duration<double, std::milli>(end - model_end).count();
    result.inference_ms = std::chrono::duration<double, std::milli>(
        end - start
    ).count();

    return result;
}


// ═══════════════════════════════════════════════════
//  Segmentation 추론
// ═══════════════════════════════════════════════════

SegmentResult VisionInference::Segment(const cv::Mat& image)
{
    SegmentResult result;

    if (!m_bInitialized)
    {
        std::cerr << "[Vision] 모델이 초기화되지 않았습니다." << std::endl;
        return result;
    }

    if (m_config.task != "segment") throw std::logic_error("Segment requires a segmentation model.");
    auto start = std::chrono::steady_clock::now();

    // ── 1. 전처리 ────────────────────────────────────
    std::vector<float> input_tensor = Preprocess(image);

    // ── 2. 입력 텐서 ─────────────────────────────────
    std::vector<int64_t> input_shape = {
        1, m_config.input_channels,
        m_config.input_height, m_config.input_width
    };

    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor.data(),
        input_tensor.size(),
        input_shape.data(),
        input_shape.size()
    );

    // ── 3. 추론 실행 ─────────────────────────────────
    const char* input_names[] = { m_config.input_name.c_str() };
    const char* output_names[] = { m_config.output_name.c_str() };

    auto output_tensors = m_session->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_ort, 1,
        output_names, 1
    );

    // ── 4. 후처리 ────────────────────────────────────
    ValidateOutput(output_tensors[0], m_config, 4);
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    // 출력 형태: (1, num_classes, H, W)
    int out_classes = static_cast<int>(output_shape[1]);
    int out_height = static_cast<int>(output_shape[2]);
    int out_width = static_cast<int>(output_shape[3]);

    // Argmax 마스크 생성
    cv::Mat seg_mask = ArgmaxMask(output_data, out_classes,
                                   out_height, out_width);

    // 원본 이미지 크기로 리사이즈
    cv::Rect roi(0, 0, image.cols, image.rows);
    if (m_config.crop_width > 0)
        roi = cv::Rect((image.cols-m_config.crop_width)/2, (image.rows-m_config.crop_height)/2,
                       m_config.crop_width, m_config.crop_height);
    if (seg_mask.size() != roi.size())
    {
        cv::resize(seg_mask, seg_mask, roi.size(),
                   0, 0, cv::INTER_NEAREST);  // 마스크는 NEAREST!
    }

    result.mask = cv::Mat::zeros(image.size(), CV_8UC1);
    seg_mask.copyTo(result.mask(roi));
    result.valid_mask = cv::Mat::zeros(image.size(), CV_8UC1);
    result.valid_mask(roi).setTo(255);
    result.num_classes = out_classes;
    result.color_mask = cv::Mat::zeros(image.size(), CV_8UC3);
    ColorizeMask(seg_mask, out_classes).copyTo(result.color_mask(roi));

    // 클래스별 픽셀 수 집계
    result.pixel_counts.resize(out_classes, 0);
    for (int h = 0; h < seg_mask.rows; ++h)
    {
        for (int w = 0; w < seg_mask.cols; ++w)
        {
            int cls = seg_mask.at<uchar>(h, w);
            if (cls < out_classes)
                result.pixel_counts[cls]++;
        }
    }

    // 소요 시간
    auto end = std::chrono::steady_clock::now();
    result.inference_ms = std::chrono::duration<double, std::milli>(
        end - start
    ).count();

    return result;
}

float VisionInference::Sigmoid(float value)
{
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

float VisionInference::IntersectionOverUnion(const Detection& left, const Detection& right)
{
    const float x1 = std::max(left.x1, right.x1);
    const float y1 = std::max(left.y1, right.y1);
    const float x2 = std::min(left.x2, right.x2);
    const float y2 = std::min(left.y2, right.y2);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float left_area = std::max(0.0f, left.x2 - left.x1) * std::max(0.0f, left.y2 - left.y1);
    const float right_area = std::max(0.0f, right.x2 - right.x1) * std::max(0.0f, right.y2 - right.y1);
    const float denominator = left_area + right_area - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

DetectResult VisionInference::Detect(const cv::Mat& image)
{
    DetectResult result;
    if (!m_bInitialized) throw std::logic_error("Model is not initialized.");
    if (m_config.task != "detect") throw std::logic_error("Detect requires a detection model.");
    const auto start = std::chrono::steady_clock::now();
    const auto input_tensor = Preprocess(image);
    const auto preprocess_end = std::chrono::steady_clock::now();
    const std::vector<int64_t> input_shape{1, m_config.input_channels,
                                           m_config.input_height, m_config.input_width};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory_info,
        const_cast<float*>(input_tensor.data()), input_tensor.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {m_config.input_name.c_str()};
    const auto& configured_outputs = m_config.output_names.empty()
        ? std::vector<std::string>{m_config.output_name} : m_config.output_names;
    std::vector<const char*> output_names;
    for (const auto& name : configured_outputs) output_names.push_back(name.c_str());
    const auto model_start = std::chrono::steady_clock::now();
    auto outputs = m_session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                  output_names.data(), output_names.size());
    const auto model_end = std::chrono::steady_clock::now();
    const float* boxes = nullptr;
    const float* scores = nullptr;
    int64_t candidates = 0;
    if (m_config.backend == "redetr_v4") {
        if (outputs.size() != 2) throw std::runtime_error("Re-DETR output count mismatch.");
        const auto box_info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto box_shape = box_info.GetShape();
        const auto logit_info = outputs[1].GetTensorTypeAndShapeInfo();
        const auto logit_shape = logit_info.GetShape();
        if (box_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            logit_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            box_shape.size() != 3 || logit_shape.size() != 3 || box_shape[0] != 1 || logit_shape[0] != 1 ||
            box_shape[2] != 4 || logit_shape[2] != m_config.num_classes ||
            box_shape[1] <= 0 || logit_shape[1] != box_shape[1])
            throw std::runtime_error("Re-DETR output shape/type mismatch.");
        boxes = outputs[0].GetTensorData<float>();
        scores = outputs[1].GetTensorData<float>();
        candidates = box_shape[1];
        for (size_t index = 0; index < logit_info.GetElementCount(); ++index)
            if (!std::isfinite(scores[index])) throw std::runtime_error("Re-DETR returned non-finite scores.");
        for (size_t index = 0; index < box_info.GetElementCount(); ++index)
            if (!std::isfinite(boxes[index])) throw std::runtime_error("Re-DETR returned non-finite boxes.");
    } else {
        if (outputs.size() != 1) throw std::runtime_error("Detection output count mismatch.");
        ValidateOutput(outputs[0], m_config, 3);
        const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        boxes = outputs[0].GetTensorData<float>();
        candidates = shape[1];
    }
    const int stride = 5 + m_config.num_classes;
    const int roi_width = m_config.crop_width > 0 ? m_config.crop_width : image.cols;
    const int roi_height = m_config.crop_height > 0 ? m_config.crop_height : image.rows;
    const int roi_x = m_config.crop_width > 0 ? (image.cols - m_config.crop_width) / 2 : 0;
    const int roi_y = m_config.crop_height > 0 ? (image.rows - m_config.crop_height) / 2 : 0;
    std::vector<Detection> candidates_to_keep;
    candidates_to_keep.reserve(static_cast<size_t>(candidates));
    for (int64_t index = 0; index < candidates; ++index)
    {
        const float* row = boxes + index * (m_config.backend == "redetr_v4" ? 4 : stride);
        const float objectness = m_config.backend == "redetr_v4" && m_config.detection_objectness == "none"
            ? 1.0f : Sigmoid(row[4]);
        int class_id = 0;
        float best_class = -std::numeric_limits<float>::infinity();
        float score_sum = 0.0f;
        float score_max = -std::numeric_limits<float>::infinity();
        if (m_config.backend == "redetr_v4" && m_config.detection_class_scores == "softmax") {
            for (int cls = 0; cls < m_config.num_classes; ++cls)
                score_max = std::max(score_max, scores[index * m_config.num_classes + cls]);
            for (int cls = 0; cls < m_config.num_classes; ++cls)
                score_sum += std::exp(scores[index * m_config.num_classes + cls] - score_max);
        }
        for (int cls = 0; cls < m_config.num_classes; ++cls)
        {
            const float raw_score = m_config.backend == "redetr_v4"
                ? scores[index * m_config.num_classes + cls] : row[5 + cls];
            const float score = m_config.backend == "redetr_v4" && m_config.detection_class_scores == "softmax"
                ? std::exp(raw_score - score_max) / score_sum : Sigmoid(raw_score);
            if (score > best_class) { best_class = score; class_id = cls; }
        }
        const float confidence = objectness * best_class;
        if (!std::isfinite(confidence) || confidence < m_config.detection_confidence_threshold)
            continue;
        float x1, y1, x2, y2;
        if (m_config.detection_box_encoding == "normalized_cxcywh")
        {
            const float cx = row[0], cy = row[1], width = row[2], height = row[3];
            x1 = cx - width * 0.5f; y1 = cy - height * 0.5f;
            x2 = cx + width * 0.5f; y2 = cy + height * 0.5f;
        }
        else
        {
            x1 = row[0]; y1 = row[1]; x2 = row[2]; y2 = row[3];
        }
        Detection detection;
        detection.x1 = roi_x + std::clamp(x1, 0.0f, 1.0f) * roi_width;
        detection.y1 = roi_y + std::clamp(y1, 0.0f, 1.0f) * roi_height;
        detection.x2 = roi_x + std::clamp(x2, 0.0f, 1.0f) * roi_width;
        detection.y2 = roi_y + std::clamp(y2, 0.0f, 1.0f) * roi_height;
        detection.class_id = class_id;
        detection.confidence = confidence;
        if (detection.x2 > detection.x1 && detection.y2 > detection.y1)
            candidates_to_keep.push_back(detection);
    }
    std::stable_sort(candidates_to_keep.begin(), candidates_to_keep.end(),
                     [](const Detection& left, const Detection& right) {
                         return left.confidence > right.confidence;
                     });
    for (const auto& candidate : candidates_to_keep)
    {
        bool suppressed = false;
        for (const auto& selected : result.detections)
            if (candidate.class_id == selected.class_id &&
                IntersectionOverUnion(candidate, selected) > m_config.detection_iou_threshold)
            { suppressed = true; break; }
        if (!suppressed) {
            result.detections.push_back(candidate);
            if (static_cast<int>(result.detections.size()) >= m_config.detection_max_detections) break;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    result.preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - start).count();
    result.model_ms = std::chrono::duration<double, std::milli>(model_end - model_start).count();
    result.postprocess_ms = std::chrono::duration<double, std::milli>(end - model_end).count();
    result.inference_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

AnomalyResult VisionInference::Anomaly(const cv::Mat& image)
{
    AnomalyResult result;
    if (!m_bInitialized) throw std::logic_error("Model is not initialized.");
    if (m_config.task != "anomaly") throw std::logic_error("Anomaly requires an anomaly model.");
    const auto start = std::chrono::steady_clock::now();
    const auto input_tensor = Preprocess(image);
    const auto preprocess_end = std::chrono::steady_clock::now();
    const std::vector<int64_t> input_shape{1, m_config.input_channels,
                                           m_config.input_height, m_config.input_width};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory_info,
        const_cast<float*>(input_tensor.data()), input_tensor.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {m_config.input_name.c_str()};
    std::vector<const char*> output_names;
    const auto& configured_outputs = m_config.output_names.empty()
        ? std::vector<std::string>{m_config.output_name} : m_config.output_names;
    for (const auto& name : configured_outputs) output_names.push_back(name.c_str());
    const auto model_start = std::chrono::steady_clock::now();
    auto outputs = m_session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                  output_names.data(), output_names.size());
    const auto model_end = std::chrono::steady_clock::now();
    cv::Mat input_map;
    if (m_config.backend == "patchcore") {
        const auto score_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const auto score_type = outputs[0].GetTensorTypeAndShapeInfo().GetElementType();
        const auto map_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        const auto map_type = outputs[1].GetTensorTypeAndShapeInfo().GetElementType();
        if (score_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            map_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            score_shape.size() != 1 || score_shape[0] != 1 || map_shape.size() != 4 ||
            map_shape[0] != 1 || map_shape[1] != 1 || map_shape[2] <= 0 || map_shape[3] <= 0)
            throw std::runtime_error("PatchCore output shape mismatch.");
        const int map_height = static_cast<int>(map_shape[2]);
        const int map_width = static_cast<int>(map_shape[3]);
        input_map = cv::Mat(map_height, map_width, CV_32FC1);
        const float* map_data = outputs[1].GetTensorData<float>();
        std::memcpy(input_map.data, map_data, static_cast<size_t>(map_height) * map_width * sizeof(float));
        result.score = outputs[0].GetTensorData<float>()[0];
    } else {
        ValidateOutput(outputs[0], m_config, 4);
        const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (shape[2] != m_config.input_height || shape[3] != m_config.input_width)
            throw std::runtime_error("Anomaly reconstruction shape must match input shape.");
        const auto* reconstruction = outputs[0].GetTensorData<float>();
        input_map = cv::Mat(m_config.input_height, m_config.input_width, CV_32FC1, cv::Scalar(0));
        const size_t plane = static_cast<size_t>(m_config.input_height) * m_config.input_width;
        for (int y = 0; y < m_config.input_height; ++y)
            for (int x = 0; x < m_config.input_width; ++x)
            {
                float error = 0.0f;
                for (int channel = 0; channel < m_config.input_channels; ++channel)
                    error += std::abs(input_tensor[static_cast<size_t>(channel) * plane + y * m_config.input_width + x]
                                      - reconstruction[static_cast<size_t>(channel) * plane + y * m_config.input_width + x]);
                input_map.at<float>(y, x) = error / static_cast<float>(m_config.input_channels);
            }
    }
    const int roi_width = m_config.crop_width > 0 ? m_config.crop_width : image.cols;
    const int roi_height = m_config.crop_height > 0 ? m_config.crop_height : image.rows;
    const int roi_x = m_config.crop_width > 0 ? (image.cols - m_config.crop_width) / 2 : 0;
    const int roi_y = m_config.crop_height > 0 ? (image.rows - m_config.crop_height) / 2 : 0;
    cv::Mat resized;
    cv::resize(input_map, resized, cv::Size(roi_width, roi_height), 0, 0, cv::INTER_LINEAR);
    result.anomaly_map = cv::Mat::zeros(image.size(), CV_32FC1);
    resized.copyTo(result.anomaly_map(cv::Rect(roi_x, roi_y, roi_width, roi_height)));
    if (m_config.backend != "patchcore") {
        double maximum = 0.0;
        cv::minMaxLoc(result.anomaly_map, nullptr, &maximum);
        result.score = static_cast<float>(maximum);
    }
    result.threshold = m_config.anomaly_threshold;
    result.anomalous = result.score >= result.threshold;
    const auto end = std::chrono::steady_clock::now();
    result.preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - start).count();
    result.model_ms = std::chrono::duration<double, std::milli>(model_end - model_start).count();
    result.postprocess_ms = std::chrono::duration<double, std::milli>(end - model_end).count();
    result.inference_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}


// ═══════════════════════════════════════════════════
//  파일 경로 기반 추론
// ═══════════════════════════════════════════════════

ClassifyResult VisionInference::ClassifyFile(const std::string& image_path)
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (image.empty())
    {
        std::cerr << "[Vision] 이미지 로드 실패: " << image_path << std::endl;
        return ClassifyResult();
    }
    return Classify(image);
}

SegmentResult VisionInference::SegmentFile(const std::string& image_path)
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (image.empty())
    {
        std::cerr << "[Vision] 이미지 로드 실패: " << image_path << std::endl;
        return SegmentResult();
    }
    return Segment(image);
}

DetectResult VisionInference::DetectFile(const std::string& image_path)
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (image.empty()) throw std::runtime_error("Image load failed: " + image_path);
    return Detect(image);
}

AnomalyResult VisionInference::AnomalyFile(const std::string& image_path)
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (image.empty()) throw std::runtime_error("Image load failed: " + image_path);
    return Anomaly(image);
}


// ═══════════════════════════════════════════════════
//  모델 정보
// ═══════════════════════════════════════════════════

void VisionInference::PrintModelInfo() const
{
    if (!m_bInitialized) return;

    std::cout << "\n╔══════════════════════════════════════╗" << std::endl;
    std::cout << "║       Vision 모델 정보               ║" << std::endl;
    std::cout << "╠══════════════════════════════════════╣" << std::endl;
    std::cout << "║  태스크: " << m_config.task << std::endl;
    std::cout << "║  모델  : " << m_config.model_path << std::endl;
    std::cout << "║  입력  : (" << m_config.input_channels << ", "
              << m_config.input_height << ", " << m_config.input_width
              << ")" << std::endl;
    std::cout << "║  클래스: " << m_config.num_classes << std::endl;
    if (m_config.model_implementation_version != 0)
        std::cout << "║  첫 Conv: " << m_config.stem_in_channels
                  << "ch, 입력 처리: " << m_config.input_adapter << std::endl;

    std::cout << "║  Runtime: " << m_config.runtime << std::endl;
    if (m_openvino)
        std::cout << "║  CPU / FP32 / LATENCY / 1 stream" << std::endl;
    // 입출력 텐서 정보
    if (m_session) {
    size_t num_inputs = m_session->GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i)
    {
        auto name = m_session->GetInputNameAllocated(i, m_allocator);
        auto shape = m_session->GetInputTypeInfo(i)
                              .GetTensorTypeAndShapeInfo()
                              .GetShape();
        std::cout << "║  입력텐서[" << i << "]: " << name.get() << " [";
        for (size_t j = 0; j < shape.size(); ++j)
        {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    size_t num_outputs = m_session->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i)
    {
        auto name = m_session->GetOutputNameAllocated(i, m_allocator);
        auto shape = m_session->GetOutputTypeInfo(i)
                              .GetTensorTypeAndShapeInfo()
                              .GetShape();
        std::cout << "║  출력텐서[" << i << "]: " << name.get() << " [";
        for (size_t j = 0; j < shape.size(); ++j)
        {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    }
    if (!m_config.class_names.empty())
    {
        std::cout << "║  클래스 목록:" << std::endl;
        for (size_t i = 0; i < m_config.class_names.size(); ++i)
        {
            std::cout << "║    [" << i << "] "
                      << m_config.class_names[i] << std::endl;
        }
    }
    std::cout << "╚══════════════════════════════════════╝" << std::endl;
}
