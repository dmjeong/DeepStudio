#include "vision_runtime_c.h"

#include "sam2_inference.h"
#include "vision_inference.h"
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <set>
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
    result->preprocess_ms = source.preprocess_ms;
    return result;
}

class Sha256 {
public:
    Sha256() : state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

    void update(const uint8_t* input, size_t size) {
        for (size_t index = 0; index < size; ++index) {
            buffer_[length_++] = input[index];
            if (length_ == buffer_.size()) {
                transform();
                bit_length_ += 512;
                length_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> final() {
        const size_t original_length = length_;
        size_t index = length_;
        buffer_[index++] = 0x80;
        if (index > 56) {
            while (index < buffer_.size()) buffer_[index++] = 0;
            transform();
            index = 0;
        }
        while (index < 56) buffer_[index++] = 0;
        const uint64_t total_bits = bit_length_ + static_cast<uint64_t>(original_length) * 8;
        for (int shift = 7; shift >= 0; --shift)
            buffer_[index++] = static_cast<uint8_t>((total_bits >> (shift * 8)) & 0xffu);
        transform();

        std::array<uint8_t, 32> digest{};
        for (size_t word = 0; word < state_.size(); ++word) {
            digest[word * 4] = static_cast<uint8_t>((state_[word] >> 24) & 0xffu);
            digest[word * 4 + 1] = static_cast<uint8_t>((state_[word] >> 16) & 0xffu);
            digest[word * 4 + 2] = static_cast<uint8_t>((state_[word] >> 8) & 0xffu);
            digest[word * 4 + 3] = static_cast<uint8_t>(state_[word] & 0xffu);
        }
        return digest;
    }

private:
    static constexpr std::array<uint32_t, 64> kRound = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    static uint32_t rotate_right(uint32_t value, uint32_t bits) {
        return (value >> bits) | (value << (32 - bits));
    }

    void transform() {
        std::array<uint32_t, 64> words{};
        for (size_t index = 0; index < 16; ++index) {
            const size_t offset = index * 4;
            words[index] = (static_cast<uint32_t>(buffer_[offset]) << 24) |
                           (static_cast<uint32_t>(buffer_[offset + 1]) << 16) |
                           (static_cast<uint32_t>(buffer_[offset + 2]) << 8) |
                           static_cast<uint32_t>(buffer_[offset + 3]);
        }
        for (size_t index = 16; index < words.size(); ++index) {
            const uint32_t s0 = rotate_right(words[index - 15], 7) ^
                                rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const uint32_t s1 = rotate_right(words[index - 2], 17) ^
                                rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t index = 0; index < words.size(); ++index) {
            const uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const uint32_t choose = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + choose + kRound[index] + words[index];
            const uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_;
    std::array<uint8_t, 64> buffer_{};
    size_t length_ = 0;
    uint64_t bit_length_ = 0;
};

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::invalid_argument("Deployment bundle file cannot be opened.");
    Sha256 sha;
    std::array<uint8_t, 1024 * 1024> buffer{};
    while (stream) {
        stream.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const auto count = stream.gcount();
        if (count > 0) sha.update(buffer.data(), static_cast<size_t>(count));
    }
    if (!stream.eof()) throw std::invalid_argument("Deployment bundle file cannot be read.");
    const auto digest = sha.final();
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const auto byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0x0f]);
    }
    return result;
}

std::filesystem::path bundle_relative_path(const std::string& value, const char* field) {
    if (value.empty() || value.find('\\') != std::string::npos || value.find('\0') != std::string::npos ||
        value.find("//") != std::string::npos)
        throw std::invalid_argument(std::string("Deployment bundle ") + field + " path is unsafe.");
    const auto path = std::filesystem::u8path(value);
    if (path.is_absolute() || path.has_root_name() || path.has_root_directory())
        throw std::invalid_argument(std::string("Deployment bundle ") + field + " path is unsafe.");
    for (const auto& component : path) {
        if (component == "." || component == ".." || component.empty())
            throw std::invalid_argument(std::string("Deployment bundle ") + field + " path is unsafe.");
    }
    return path;
}

void validate_bundle_references(const std::filesystem::path& root, const nlohmann::json& config) {
    auto check = [&](const std::string& value, const char* field) {
        const auto relative = bundle_relative_path(value, field);
        const auto target = (root / relative).lexically_normal();
        if (target.lexically_relative(root) != relative)
            throw std::invalid_argument(std::string("Deployment bundle ") + field + " escapes its root.");
        if (!std::filesystem::is_regular_file(target) || std::filesystem::is_symlink(target))
            throw std::invalid_argument(std::string("Deployment bundle referenced file is missing: ") + value);
    };
    if (config.contains("model_path")) {
        if (!config["model_path"].is_string()) throw std::invalid_argument("Deployment model_path must be a string.");
        check(config["model_path"].get<std::string>(), "model");
    }
    if (config.contains("contracts") && config["contracts"].is_object() &&
        config["contracts"].contains("graphs") && config["contracts"]["graphs"].is_object()) {
        for (const auto& item : config["contracts"]["graphs"].items()) {
            if (!item.value().is_object() || !item.value().contains("file") || !item.value()["file"].is_string())
                throw std::invalid_argument("Deployment graph contract is invalid.");
            check(item.value()["file"].get<std::string>(), "graph");
        }
    }
}

std::string bundle_config_path(const char* bundle_path_utf8) {
    if (!bundle_path_utf8 || !*bundle_path_utf8)
        throw std::invalid_argument("Deployment bundle path is empty.");
    auto root = std::filesystem::u8path(bundle_path_utf8);
    // The public C/C# APIs accept a relative bundle path. Resolve it once
    // before returning the config path so model_path and graph references are
    // interpreted relative to the bundle, never relative to the caller's
    // current working directory.
    if (root.is_relative()) root = std::filesystem::absolute(root);
    root = root.lexically_normal();
    if (!std::filesystem::is_directory(root) || std::filesystem::is_symlink(root))
        throw std::invalid_argument("Deployment bundle must be a directory.");
    std::ifstream stream(root / "manifest.json");
    if (!stream) throw std::invalid_argument("Deployment bundle manifest.json is missing.");
    const auto manifest = nlohmann::json::parse(stream);
    if (!manifest.is_object() || manifest.value("schema_version", 0) != 1)
        throw std::invalid_argument("Unsupported deployment bundle manifest.");
    if (!manifest.contains("files") || !manifest["files"].is_object())
        throw std::invalid_argument("Deployment bundle file manifest is missing.");
    std::set<std::string> expected;
    for (const auto& item : manifest["files"].items()) {
        const auto relative = bundle_relative_path(item.key(), "file");
        const auto name = relative.generic_u8string();
        if (!item.value().is_object() || !item.value().contains("size") || !item.value()["size"].is_number_unsigned() ||
            !item.value().contains("sha256") || !item.value()["sha256"].is_string() ||
            item.value()["sha256"].get<std::string>().size() != 64)
            throw std::invalid_argument("Deployment bundle checksum entry is invalid.");
        const auto checksum = item.value()["sha256"].get<std::string>();
        if (!std::all_of(checksum.begin(), checksum.end(), [](char value) {
                return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
            }))
            throw std::invalid_argument("Deployment bundle checksum entry is invalid.");
        if (!expected.insert(name).second) throw std::invalid_argument("Deployment bundle contains duplicate files.");
        const auto target = (root / relative).lexically_normal();
        if (!std::filesystem::is_regular_file(target) || std::filesystem::is_symlink(target) ||
            std::filesystem::file_size(target) != item.value()["size"].get<uintmax_t>() ||
            sha256_file(target) != checksum)
            throw std::invalid_argument(std::string("Deployment bundle checksum mismatch: ") + name);
    }
    std::set<std::string> actual;
    for (std::filesystem::recursive_directory_iterator iterator(root), end; iterator != end; ++iterator) {
        if (iterator->is_symlink()) throw std::invalid_argument("Deployment bundle symlinks are not allowed.");
        if (!iterator->is_regular_file()) continue;
        const auto relative = iterator->path().lexically_relative(root).generic_u8string();
        if (relative == "manifest.json") continue;
        actual.insert(relative);
    }
    if (actual != expected) throw std::invalid_argument("Deployment bundle file manifest does not cover every file.");
    const auto name = manifest.value("config", std::string());
    const auto relative = bundle_relative_path(name, "config");
    if (!expected.count(relative.generic_u8string()))
        throw std::invalid_argument("Deployment bundle config is not checksummed.");
    const auto config = root / relative;
    if (!std::filesystem::is_regular_file(config) || std::filesystem::is_symlink(config))
        throw std::invalid_argument("Deployment bundle config is missing.");
    std::ifstream config_stream(config);
    if (!config_stream) throw std::invalid_argument("Deployment bundle config cannot be opened.");
    const auto config_json = nlohmann::json::parse(config_stream);
    if (!config_json.is_object() || (config_json.value("schema_version", 0) != 5 &&
                                    config_json.value("schema_version", 0) != 6))
        throw std::invalid_argument("Deployment bundle config is unsupported.");
    validate_bundle_references(root, config_json);
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
            const auto overall_started = std::chrono::steady_clock::now();
            auto context = session->sam2->Encode(pixels);
            Sam2Prompt prompt;
            auto sam_result = session->sam2->Segment(context, prompt);
            sam_result.preprocess_ms += context.preprocess_ms;
            sam_result.model_ms += context.model_ms;
            sam_result.total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - overall_started).count();
            sam_result.postprocess_ms = std::max(
                0.0, sam_result.total_ms - sam_result.preprocess_ms - sam_result.model_ms);
            result = make_sam_result(sam_result);
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

dv_status dv_sam_auto_mask(dv_session* session, const dv_image_context* context,
                           uint32_t grid_width, uint32_t grid_height, float min_score,
                           dv_result** out_result) {
    if (out_result) *out_result = nullptr;
    if (!session || !context || !out_result) return DV_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    clear_error(session);
    try {
        if (!session->sam2 || context->owner != session)
            throw std::invalid_argument("SAM2 image context does not belong to this session.");
        auto result = session->sam2->Automatic(context->value, static_cast<int>(grid_width),
                                               static_cast<int>(grid_height), min_score);
        *out_result = make_sam_result(result).release();
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
