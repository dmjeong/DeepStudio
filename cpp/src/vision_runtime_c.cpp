#include "vision_runtime_c.h"

#include "vision_inference.h"
#include <opencv2/core.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

struct dv_session {
    VisionInference engine;
    mutable std::mutex mutex;
    std::string error;
};

namespace {
thread_local std::string g_last_create_error;
constexpr uint32_t kMinOptionsSize = static_cast<uint32_t>(sizeof(dv_session_options));
constexpr uint32_t kMinImageSize = static_cast<uint32_t>(sizeof(dv_image_view));
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
        if (!session->engine.InitializeFromJson(config_path_utf8, runtime, threads)) {
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

dv_status dv_infer(dv_session* session, const dv_image_view* image, dv_result** out_result) {
    if (out_result) *out_result = nullptr;
    if (!session || !image || !out_result) return DV_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    clear_error(session);
    try {
        const auto pixels = view_to_mat(*image);
        if (!session->engine.IsReady()) throw std::logic_error("Session is not ready.");
        std::unique_ptr<dv_result> result;
        if (session->engine.GetConfig().task == "classify")
            result = make_classification(session->engine.Classify(pixels));
        else if (session->engine.GetConfig().task == "segment")
            result = make_segmentation(session->engine.Segment(pixels));
        else
            throw std::invalid_argument("The C ABI currently supports classify and segment.");
        *out_result = result.release();
        return DV_STATUS_OK;
    }
    catch (...) { return classify_exception(session); }
}

const char* dv_last_error(const dv_session* session) {
    return session ? session->error.c_str() : g_last_create_error.c_str();
}

void dv_release_result(dv_result* result) {
    if (!result) return;
    delete[] result->probabilities;
    delete[] result->class_name_utf8;
    delete[] result->mask;
    delete result;
}

void dv_close_session(dv_session* session) { delete session; }

} // extern "C"
