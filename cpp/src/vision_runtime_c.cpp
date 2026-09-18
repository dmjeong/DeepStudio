#include "vision_runtime_c.h"

#include "sam2_inference.h"
#include "vision_inference.h"
#include <opencv2/core.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

struct dv_session {
    VisionInference engine;
    std::unique_ptr<Sam2Inference> sam2;
    mutable std::mutex mutex;
    std::string error;
};

struct dv_image_context {
    dv_session* owner = nullptr;
    Sam2ImageContext value;
};

namespace {
thread_local std::string g_last_create_error;
constexpr uint32_t kMinOptionsSize = static_cast<uint32_t>(sizeof(dv_session_options));
constexpr uint32_t kMinImageSize = static_cast<uint32_t>(sizeof(dv_image_view));
constexpr uint32_t kMinSamPromptSize = static_cast<uint32_t>(sizeof(dv_sam_prompt));
constexpr uint32_t kResultSize = static_cast<uint32_t>(sizeof(dv_result));

void clear_error(dv_session* session) {
    if (session) session->error.clear();
}

void set_error(dv_session* session, const char* message) {
    const char* value = message ? message : "Unknown runtime error.";
    if (session) session->error = value;
    else g_last_create_error = value;
}

void set_error(dv_session* session, const std::exception& error) {
    set_error(session, error.what());
}

dv_status classify_exception(dv_session* session) {
    try { throw; }
    catch (const std::bad_alloc&) { set_error(session, "Out of memory."); return DV_STATUS_OUT_OF_MEMORY; }
    catch (const std::invalid_argument& error) { set_error(session, error); return DV_STATUS_INVALID_ARGUMENT; }
    catch (const std::logic_error& error) { set_error(session, error); return DV_STATUS_NOT_READY; }
    catch (const std::exception& error) { set_error(session, error); return DV_STATUS_RUNTIME_ERROR; }
    catch (...) { set_error(session, "Unknown C++ exception."); return DV_STATUS_INTERNAL_ERROR; }
}

cv::Mat view_to_mat(const dv_image_view& view) {
    if (view.struct_size < kMinImageSize || view.abi_version != DV_ABI_VERSION || !view.data ||
        view.width <= 0 || view.height <= 0 || (view.channels != 1 && view.channels != 3 && view.channels != 4))
        throw std::invalid_argument("Invalid image view.");
    const int packed = view.width * view.channels;
    const int stride = view.stride_bytes ? view.stride_bytes : packed;
    if (stride < packed) throw std::invalid_argument("Image stride is smaller than one row.");
    return cv::Mat(view.height, view.width, CV_MAKETYPE(CV_8U, view.channels),
                   const_cast<uint8_t*>(view.data), static_cast<size_t>(stride));
}

std::unique_ptr<dv_result> make_classification(const ClassifyResult& source) {
    auto result = std::make_unique<dv_result>();
    result->struct_size = kResultSize;
    result->abi_version = DV_ABI_VERSION;
    result->kind = DV_RESULT_CLASSIFICATION;
    result->class_id = source.class_id;
    result->confidence = source.confidence;
    result->probability_count = static_cast<uint32_t>(source.probabilities.size());
    if (!source.probabilities.empty()) {
        result->probabilities = new float[source.probabilities.size()];
        std::copy(source.probabilities.begin(), source.probabilities.end(), result->probabilities);
    }
    result->class_name_utf8 = new char[source.class_name.size() + 1];
    std::memcpy(result->class_name_utf8, source.class_name.c_str(), source.class_name.size() + 1);
    result->total_ms = source.inference_ms;
    result->preprocess_ms = source.preprocess_ms;
    result->model_ms = source.model_ms;
    result->postprocess_ms = source.postprocess_ms;
    return result;
}

std::unique_ptr<dv_result> make_segmentation(const SegmentResult& source) {
    if (source.mask.empty() || source.mask.type() != CV_8UC1)
        throw std::runtime_error("Segmentation mask is invalid.");
    auto result = std::make_unique<dv_result>();
    result->struct_size = kResultSize;
    result->abi_version = DV_ABI_VERSION;
    result->kind = DV_RESULT_SEGMENTATION;
    result->mask_width = static_cast<uint32_t>(source.mask.cols);
    result->mask_height = static_cast<uint32_t>(source.mask.rows);
    result->mask_stride_bytes = result->mask_width;
    result->mask_classes = static_cast<uint32_t>(source.num_classes);
    const size_t bytes = static_cast<size_t>(source.mask.cols) * source.mask.rows;
    result->mask = new uint8_t[bytes];
    for (int y = 0; y < source.mask.rows; ++y)
        std::memcpy(result->mask + static_cast<size_t>(y) * source.mask.cols,
                    source.mask.ptr<uint8_t>(y), static_cast<size_t>(source.mask.cols));
    result->total_ms = source.inference_ms;
    return result;
}

std::unique_ptr<dv_result> make_detection(const DetectResult& source) {
    auto result = std::make_unique<dv_result>();
    result->struct_size = kResultSize;
    result->abi_version = DV_ABI_VERSION;
    result->kind = DV_RESULT_DETECTION;
    result->detection_count = static_cast<uint32_t>(source.detections.size());
    if (!source.detections.empty()) {
        result->detections = new dv_detection[source.detections.size()];
        for (size_t index = 0; index < source.detections.size(); ++index) {
            const auto& source_detection = source.detections[index];
            result->detections[index] = {
                source_detection.x1, source_detection.y1, source_detection.x2,
                source_detection.y2, source_detection.class_id, source_detection.confidence};
        }
    }
    result->total_ms = source.inference_ms;
    result->preprocess_ms = source.preprocess_ms;
    result->model_ms = source.model_ms;
    result->postprocess_ms = source.postprocess_ms;
    return result;
}

std::unique_ptr<dv_result> make_anomaly(const AnomalyResult& source) {
    if (source.anomaly_map.empty() || source.anomaly_map.type() != CV_32FC1)
        throw std::runtime_error("Anomaly map is invalid.");
    auto result = std::make_unique<dv_result>();
    result->struct_size = kResultSize;
    result->abi_version = DV_ABI_VERSION;
    result->kind = DV_RESULT_ANOMALY;
    result->anomaly_score = source.score;
    result->anomaly_threshold = source.threshold;
    result->anomalous = source.anomalous ? 1u : 0u;
    result->anomaly_map_width = static_cast<uint32_t>(source.anomaly_map.cols);
    result->anomaly_map_height = static_cast<uint32_t>(source.anomaly_map.rows);
    result->anomaly_map_stride_bytes = result->anomaly_map_width * sizeof(float);
    const size_t values = static_cast<size_t>(source.anomaly_map.cols) * source.anomaly_map.rows;
    result->anomaly_map = new float[values];
    for (int row = 0; row < source.anomaly_map.rows; ++row)
        std::memcpy(result->anomaly_map + static_cast<size_t>(row) * source.anomaly_map.cols,
                    source.anomaly_map.ptr<float>(row), static_cast<size_t>(source.anomaly_map.cols) * sizeof(float));
    result->total_ms = source.inference_ms;
    result->preprocess_ms = source.preprocess_ms;
    result->model_ms = source.model_ms;
    result->postprocess_ms = source.postprocess_ms;
    return result;
}

std::unique_ptr<dv_result> make_sam_result(const Sam2Result& source) {
    if (source.mask.empty() || source.mask.type() != CV_8UC1)
        throw std::runtime_error("SAM2 mask is invalid.");
    auto result = std::make_unique<dv_result>();
    result->struct_size = kResultSize;
    result->abi_version = DV_ABI_VERSION;
    result->kind = DV_RESULT_SEGMENTATION;
    result->mask_width = static_cast<uint32_t>(source.mask.cols);
    result->mask_height = static_cast<uint32_t>(source.mask.rows);
    result->mask_stride_bytes = result->mask_width;
    result->mask_classes = 1;
    const size_t bytes = static_cast<size_t>(source.mask.cols) * source.mask.rows;
    result->mask = new uint8_t[bytes];
    for (int row = 0; row < source.mask.rows; ++row)
        std::memcpy(result->mask + static_cast<size_t>(row) * source.mask.cols,
                    source.mask.ptr<uint8_t>(row), static_cast<size_t>(source.mask.cols));
    result->total_ms = source.total_ms;
    result->model_ms = source.model_ms;
    result->postprocess_ms = source.postprocess_ms;
    result->preprocess_ms = source.total_ms - source.model_ms - source.postprocess_ms;
    if (result->preprocess_ms < 0.0) result->preprocess_ms = 0.0;
    return result;
}

std::string bundle_config_path(const char* bundle_path_utf8) {
    if (!bundle_path_utf8 || !*bundle_path_utf8)
        throw std::invalid_argument("Deployment bundle path is empty.");
    const auto root = std::filesystem::u8path(bundle_path_utf8);
    if (!std::filesystem::is_directory(root) || std::filesystem::is_symlink(root))
        throw std::invalid_argument("Deployment bundle must be a directory.");
    std::ifstream stream(root / "manifest.json");
    if (!stream) throw std::invalid_argument("Deployment bundle manifest.json is missing.");
    const auto manifest = nlohmann::json::parse(stream);
    if (!manifest.is_object() || manifest.value("schema_version", 0) != 1)
        throw std::invalid_argument("Unsupported deployment bundle manifest.");
    const auto name = manifest.value("config", std::string());
    if (name.empty()) throw std::invalid_argument("Deployment bundle config is missing.");
    const auto relative = std::filesystem::u8path(name);
    if (relative.is_absolute() || relative.has_parent_path() && relative.parent_path() != ".")
        throw std::invalid_argument("Deployment bundle config path is unsafe.");
    const auto config = root / relative;
    if (!std::filesystem::is_regular_file(config) || std::filesystem::is_symlink(config))
        throw std::invalid_argument("Deployment bundle config is missing.");
    return config.lexically_normal().u8string();
}
} // namespace

extern "C" {

uint32_t dv_abi_version(void) { return DV_ABI_VERSION; }

const char* dv_status_name(dv_status status) {
    switch (status) {
    case DV_STATUS_OK: return "ok";
    case DV_STATUS_INVALID_ARGUMENT: return "invalid_argument";
    case DV_STATUS_NOT_READY: return "not_ready";
    case DV_STATUS_RUNTIME_ERROR: return "runtime_error";
    case DV_STATUS_OUT_OF_MEMORY: return "out_of_memory";
    default: return "internal_error";
    }
}

dv_status dv_create_session(const char* config_path_utf8, const dv_session_options* options,
                            dv_session** out_session) {
    if (out_session) *out_session = nullptr;
    g_last_create_error.clear();
    if (!config_path_utf8 || !*config_path_utf8 || !out_session) {
        g_last_create_error = "Invalid session arguments.";
        return DV_STATUS_INVALID_ARGUMENT;
    }
    try {
        if (options && (options->struct_size < kMinOptionsSize || options->abi_version != DV_ABI_VERSION))
            throw std::invalid_argument("Invalid session options.");
        const std::string runtime = options && options->runtime_utf8 ? options->runtime_utf8 : "onnxruntime";
        const int threads = options ? options->num_threads : -1;
        if (threads < -1) throw std::invalid_argument("Invalid thread count.");
        auto session = std::make_unique<dv_session>();
        if (Sam2Inference::LooksLikeConfig(config_path_utf8)) {
            session->sam2 = std::make_unique<Sam2Inference>();
            if (!session->sam2->InitializeFromJson(config_path_utf8, runtime, threads)) {
                set_error(session.get(), session->sam2->LastError().c_str());
                g_last_create_error = session->error;
                return DV_STATUS_RUNTIME_ERROR;
            }
        } else if (!session->engine.InitializeFromJson(config_path_utf8, runtime, threads)) {
            set_error(session.get(), "Vision configuration or model initialization failed.");
            g_last_create_error = session->error;
            return DV_STATUS_RUNTIME_ERROR;
        }
        *out_session = session.release();
        return DV_STATUS_OK;
    }
    catch (...) {
        // No session exists to carry an error string. The status remains the ABI's
        // stable signal; callers can use the configuration log for diagnostics.
        return classify_exception(nullptr);
    }
}

dv_status dv_create_session_from_bundle(const char* bundle_path_utf8,
                                        const dv_session_options* options,
                                        dv_session** out_session) {
    try {
        const auto config = bundle_config_path(bundle_path_utf8);
        return dv_create_session(config.c_str(), options, out_session);
    }
    catch (...) {
        if (out_session) *out_session = nullptr;
        return classify_exception(nullptr);
    }
}

dv_status dv_infer(dv_session* session, const dv_image_view* image, dv_result** out_result) {
    if (out_result) *out_result = nullptr;
    if (!session || !image || !out_result) return DV_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    clear_error(session);
    try {
        const auto pixels = view_to_mat(*image);
        std::unique_ptr<dv_result> result;
        if (session->sam2) {
            auto context = session->sam2->Encode(pixels);
            Sam2Prompt prompt;
            result = make_sam_result(session->sam2->Segment(context, prompt));
        } else if (!session->engine.IsReady()) throw std::logic_error("Session is not ready.");
        else if (session->engine.GetConfig().task == "classify")
            result = make_classification(session->engine.Classify(pixels));
        else if (session->engine.GetConfig().task == "segment")
            result = make_segmentation(session->engine.Segment(pixels));
        else if (session->engine.GetConfig().task == "detect")
            result = make_detection(session->engine.Detect(pixels));
        else if (session->engine.GetConfig().task == "anomaly")
            result = make_anomaly(session->engine.Anomaly(pixels));
        else
            throw std::invalid_argument("Unsupported C ABI result task.");
        *out_result = result.release();
        return DV_STATUS_OK;
    }
    catch (...) { return classify_exception(session); }
}

dv_status dv_sam_encode(dv_session* session, const dv_image_view* image,
                        dv_image_context** out_context) {
    if (out_context) *out_context = nullptr;
    if (!session || !image || !out_context) return DV_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    clear_error(session);
    try {
        if (!session->sam2) throw std::logic_error("The session is not a SAM2 session.");
        auto context = std::make_unique<dv_image_context>();
        context->owner = session;
        context->value = session->sam2->Encode(view_to_mat(*image));
        *out_context = context.release();
        return DV_STATUS_OK;
    } catch (...) { return classify_exception(session); }
}

dv_status dv_sam_segment(dv_session* session, const dv_image_context* context,
                         const dv_sam_prompt* prompt, dv_result** out_result) {
    if (out_result) *out_result = nullptr;
    if (!session || !context || !prompt || !out_result) return DV_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    clear_error(session);
    try {
        if (!session->sam2 || context->owner != session)
            throw std::invalid_argument("SAM2 image context does not belong to this session.");
        if (prompt->struct_size < kMinSamPromptSize || prompt->abi_version != DV_ABI_VERSION)
            throw std::invalid_argument("Invalid SAM2 prompt.");
        if (prompt->point_count > 0 && (!prompt->point_coords_xy || !prompt->point_labels))
            throw std::invalid_argument("SAM2 point arrays are incomplete.");
        if ((prompt->mask_input != nullptr) != (prompt->mask_width > 0 && prompt->mask_height > 0))
            throw std::invalid_argument("SAM2 mask dimensions are incomplete.");
        Sam2Prompt native;
        native.points.reserve(prompt->point_count);
        native.labels.reserve(prompt->point_count);
        for (uint32_t index = 0; index < prompt->point_count; ++index) {
            native.points.emplace_back(prompt->point_coords_xy[index * 2], prompt->point_coords_xy[index * 2 + 1]);
            native.labels.push_back(prompt->point_labels[index]);
        }
        if (prompt->box_xyxy) native.box_xyxy.assign(prompt->box_xyxy, prompt->box_xyxy + 4);
        if (prompt->mask_input) {
            native.mask_input = cv::Mat(static_cast<int>(prompt->mask_height),
                                        static_cast<int>(prompt->mask_width), CV_32FC1,
                                        const_cast<float*>(prompt->mask_input)).clone();
        }
        *out_result = make_sam_result(session->sam2->Segment(context->value, native)).release();
        return DV_STATUS_OK;
    } catch (...) { return classify_exception(session); }
}

const char* dv_last_error(const dv_session* session) {
    return session ? session->error.c_str() : g_last_create_error.c_str();
}

void dv_release_result(dv_result* result) {
    if (!result) return;
    delete[] result->probabilities;
    delete[] result->class_name_utf8;
    delete[] result->mask;
    delete[] result->detections;
    delete[] result->anomaly_map;
    delete result;
}

void dv_release_image_context(dv_image_context* context) { delete context; }

void dv_close_session(dv_session* session) { delete session; }

} // extern "C"
