#include "sam2_inference.h"

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string required_text(const json& object, const char* key, const std::string& context)
{
    if (!object.contains(key) || !object.at(key).is_string() || object.at(key).get<std::string>().empty())
        throw std::invalid_argument(context + " requires " + key + ".");
    return object.at(key).get<std::string>();
}

std::vector<std::string> required_strings(const json& object, const char* key, const std::string& context)
{
    if (!object.contains(key) || !object.at(key).is_array() || object.at(key).empty())
        throw std::invalid_argument(context + " requires a non-empty " + key + " list.");
    std::vector<std::string> result;
    for (const auto& item : object.at(key)) {
        if (!item.is_string() || item.get<std::string>().empty())
            throw std::invalid_argument(context + " contains an invalid " + key + " entry.");
        result.push_back(item.get<std::string>());
    }
    return result;
}

void require_finite(const float* values, size_t count, const char* message)
{
    for (size_t index = 0; index < count; ++index)
        if (!std::isfinite(values[index])) throw std::runtime_error(message);
}

} // namespace

Sam2Inference::Sam2Inference() : m_env(ORT_LOGGING_LEVEL_WARNING, "DeepVisionStudio-SAM2") {}

Sam2Inference::~Sam2Inference() = default;

bool Sam2Inference::LooksLikeConfig(const std::string& config_path)
{
    try {
        std::ifstream file(fs::u8path(config_path));
        if (!file) return false;
        const auto document = json::parse(file);
        return document.is_object() && document.value("backend", std::string()) == "sam2";
    } catch (...) {
        return false;
    }
}

void Sam2Inference::Fail(const std::string& message)
{
    m_error = message;
    throw std::runtime_error(message);
}

bool Sam2Inference::InitializeFromJson(const std::string& config_path,
                                       const std::string& runtime, int num_threads)
{
    m_ready = false;
    m_error.clear();
    m_encoder.reset();
    m_decoder.reset();
    m_encoder_output_names.clear();
    m_encoder_output_index.clear();
    try {
        if (!runtime.empty() && runtime != "onnxruntime")
            throw std::invalid_argument("SAM2 supports the ONNX Runtime backend only.");
        if (num_threads < -1) throw std::invalid_argument("Invalid SAM2 thread count.");
        const auto path = fs::u8path(config_path);
        std::ifstream file(path);
        if (!file) throw std::runtime_error("Cannot open SAM2 configuration file.");
        const auto document = json::parse(file);
        if (!document.is_object() || document.value("backend", std::string()) != "sam2" ||
            document.value("task", std::string()) != "segment")
            throw std::invalid_argument("SAM2 configuration must use backend sam2 and task segment.");
        if (document.value("schema_version", 1) != 5)
            throw std::invalid_argument("SAM2 configuration requires schema 5.");
        m_input_name = document.value("input_name", std::string("input_image"));
        m_input_channels = document.value("input_channels", 3);
        m_input_height = document.value("input_height", 1024);
        m_input_width = document.value("input_width", 1024);
        m_normalize_mean = document.value("normalize_mean", m_normalize_mean);
        m_normalize_std = document.value("normalize_std", m_normalize_std);
        if (m_input_name.empty() || m_input_channels != 3 || m_input_height <= 0 || m_input_width <= 0 ||
            m_normalize_mean.size() != 3 || m_normalize_std.size() != 3)
            throw std::invalid_argument("SAM2 input and normalization metadata is invalid.");
        for (size_t index = 0; index < 3; ++index)
            if (!std::isfinite(m_normalize_mean[index]) || !std::isfinite(m_normalize_std[index]) ||
                m_normalize_std[index] <= 0.0f)
                throw std::invalid_argument("SAM2 normalization metadata is invalid.");

        const auto contracts = document.at("contracts");
        const auto graphs = contracts.at("graphs");
        auto parse_graph = [&](const char* name) {
            const auto graph = graphs.at(name);
            GraphContract result;
            auto graph_file = fs::u8path(required_text(graph, "file", std::string("SAM2 ") + name));
            if (graph_file.is_absolute() || graph_file.lexically_normal().string().find("..") != std::string::npos)
                throw std::invalid_argument(std::string("SAM2 ") + name + " graph path must be relative.");
            result.path = (path.parent_path() / graph_file).lexically_normal().u8string();
            result.outputs = required_strings(graph, "outputs", std::string("SAM2 ") + name);
            if (graph.contains("inputs")) {
                if (!graph.at("inputs").is_object())
                    throw std::invalid_argument(std::string("SAM2 ") + name + ".inputs must be an object.");
                for (const auto& item : graph.at("inputs").items()) {
                    if (!item.value().is_string() || item.value().get<std::string>().empty())
                        throw std::invalid_argument(std::string("SAM2 ") + name + ".inputs contains an invalid name.");
                    result.inputs.emplace(item.key(), item.value().get<std::string>());
                }
            }
            return result;
        };
        m_encoder_contract = parse_graph("encoder");
        m_decoder_contract = parse_graph("decoder");
        const auto encoder_image_input = m_encoder_contract.inputs.find("image");
        if (m_encoder_contract.inputs.size() != 1 || encoder_image_input == m_encoder_contract.inputs.end() ||
            encoder_image_input->second != m_input_name)
            throw std::invalid_argument("SAM2 encoder image input does not match the graph contract.");
        if (contracts.contains("mask_size")) {
            const auto mask_size = contracts.at("mask_size");
            if (!mask_size.is_array() || mask_size.size() != 2 || !mask_size[0].is_number_integer() ||
                !mask_size[1].is_number_integer())
                throw std::invalid_argument("SAM2 contracts.mask_size must be [height, width].");
            m_mask_height = mask_size[0].get<int>();
            m_mask_width = mask_size[1].get<int>();
        }
        if (m_mask_height <= 0 || m_mask_width <= 0 || m_mask_height > 4096 || m_mask_width > 4096)
            throw std::invalid_argument("SAM2 mask size is invalid.");
        m_prompt_coordinate_space = contracts.value("prompt_coordinate_space", std::string("resized_input"));
        if (m_prompt_coordinate_space != "resized_input" &&
            m_prompt_coordinate_space != "original_pixels")
            throw std::invalid_argument("SAM2 prompt_coordinate_space must be resized_input or original_pixels.");

        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        if (num_threads > 0) options.SetIntraOpNumThreads(num_threads);
        options.EnableMemPattern();
        options.EnableCpuMemArena();
        m_encoder = std::make_unique<Ort::Session>(m_env, fs::u8path(m_encoder_contract.path).c_str(), options);
        m_decoder = std::make_unique<Ort::Session>(m_env, fs::u8path(m_decoder_contract.path).c_str(), options);
        if (m_encoder->GetInputCount() != 1 || m_encoder->GetOutputCount() != m_encoder_contract.outputs.size())
            throw std::invalid_argument("SAM2 encoder input/output count does not match the contract.");
        auto encoder_input = m_encoder->GetInputNameAllocated(0, m_allocator);
        if (encoder_input.get() != m_input_name)
            throw std::invalid_argument("SAM2 encoder input name does not match the contract.");
        const auto encoder_shape = m_encoder->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (m_encoder->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || encoder_shape.size() != 4 ||
            (encoder_shape[0] > 0 && encoder_shape[0] != 1) ||
            (encoder_shape[1] > 0 && encoder_shape[1] != m_input_channels) ||
            (encoder_shape[2] > 0 && encoder_shape[2] != m_input_height) ||
            (encoder_shape[3] > 0 && encoder_shape[3] != m_input_width))
            throw std::invalid_argument("SAM2 encoder input shape/type does not match the contract.");
        for (size_t index = 0; index < m_encoder_contract.outputs.size(); ++index) {
            auto output = m_encoder->GetOutputNameAllocated(index, m_allocator);
            if (output.get() != m_encoder_contract.outputs[index])
                throw std::invalid_argument("SAM2 encoder output names do not match the contract.");
            m_encoder_output_names.push_back(m_encoder_contract.outputs[index]);
            m_encoder_output_index.emplace(m_encoder_contract.outputs[index], index);
        }
        if (m_decoder->GetOutputCount() != m_decoder_contract.outputs.size() || m_decoder_contract.outputs.empty())
            throw std::invalid_argument("SAM2 decoder output count does not match the contract.");
        for (size_t index = 0; index < m_decoder_contract.outputs.size(); ++index) {
            auto output = m_decoder->GetOutputNameAllocated(index, m_allocator);
            if (output.get() != m_decoder_contract.outputs[index])
                throw std::invalid_argument("SAM2 decoder output names do not match the contract.");
        }
        if (m_decoder_contract.inputs.size() != m_decoder->GetInputCount())
            throw std::invalid_argument("SAM2 decoder input count does not match the contract.");
        std::vector<std::string> mapped_decoder_inputs;
        mapped_decoder_inputs.reserve(m_decoder_contract.inputs.size());
        for (const auto& item : m_decoder_contract.inputs) {
            if (std::find(mapped_decoder_inputs.begin(), mapped_decoder_inputs.end(), item.second) !=
                mapped_decoder_inputs.end())
                throw std::invalid_argument("SAM2 decoder contract maps multiple semantics to one input.");
            mapped_decoder_inputs.push_back(item.second);
            bool found = false;
            for (size_t index = 0; index < m_decoder->GetInputCount(); ++index) {
                auto input = m_decoder->GetInputNameAllocated(index, m_allocator);
                if (input.get() == item.second) { found = true; break; }
            }
            if (!found) throw std::invalid_argument("SAM2 decoder input name does not match the contract.");
        }
        for (size_t index = 0; index < m_decoder->GetInputCount(); ++index) {
            auto input = m_decoder->GetInputNameAllocated(index, m_allocator);
            if (std::find(mapped_decoder_inputs.begin(), mapped_decoder_inputs.end(), input.get()) ==
                mapped_decoder_inputs.end())
                throw std::invalid_argument("SAM2 decoder has an input missing from the graph contract.");
        }
        const std::vector<std::string> supported_semantics{
            "image_embeddings", "image_features_0", "image_features_1", "point_coords",
            "point_labels", "mask_input", "has_mask_input", "orig_im_size"};
        for (const auto& item : m_decoder_contract.inputs) {
            if (std::find(supported_semantics.begin(), supported_semantics.end(), item.first) ==
                supported_semantics.end())
                throw std::invalid_argument("SAM2 decoder contract contains an unsupported semantic input.");
            if (item.first == "image_embeddings" || item.first == "image_features_0" ||
                item.first == "image_features_1") {
                if (m_encoder_output_index.find(item.second) == m_encoder_output_index.end())
                    throw std::invalid_argument("SAM2 decoder embedding is not an encoder output.");
            }
        }
        const std::vector<std::string> required_semantics{
            "image_embeddings", "point_coords", "point_labels", "mask_input",
            "has_mask_input", "orig_im_size"};
        for (const auto& semantic : required_semantics)
            if (m_decoder_contract.inputs.find(semantic) == m_decoder_contract.inputs.end())
                throw std::invalid_argument("SAM2 decoder graph contract is missing a required input.");
        ++m_generation;
        if (m_generation == 0) ++m_generation;
        m_ready = true;
        return true;
    } catch (const std::exception& error) {
        m_error = error.what();
        m_encoder.reset();
        m_decoder.reset();
        return false;
    }
}

std::vector<float> Sam2Inference::Preprocess(const cv::Mat& image) const
{
    if (image.empty() || image.depth() != CV_8U)
        throw std::invalid_argument("SAM2 image must be a non-empty 8-bit image.");
    cv::Mat rgb;
    if (image.channels() == 1) cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    else if (image.channels() == 3) cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    else if (image.channels() == 4) cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
    else throw std::invalid_argument("SAM2 image must have 1, 3 or 4 channels.");
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(m_input_width, m_input_height), 0, 0, cv::INTER_LINEAR_EXACT);
    std::vector<float> result(static_cast<size_t>(m_input_channels) * m_input_height * m_input_width);
    for (int y = 0; y < m_input_height; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < m_input_width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const size_t offset = static_cast<size_t>(channel) * m_input_height * m_input_width +
                                      static_cast<size_t>(y) * m_input_width + x;
                result[offset] = (static_cast<float>(row[x][channel]) / 255.0f - m_normalize_mean[channel]) /
                                 m_normalize_std[channel];
            }
        }
    }
    return result;
}

Sam2ImageContext Sam2Inference::Encode(const cv::Mat& image)
{
    if (!m_ready) throw std::logic_error("SAM2 model is not initialized.");
    const auto started = std::chrono::steady_clock::now();
    const auto tensor_data = Preprocess(image);
    const std::vector<int64_t> shape{1, m_input_channels, m_input_height, m_input_width};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(tensor_data.data()), tensor_data.size(),
                                                 shape.data(), shape.size());
    const char* input_names[] = {m_input_name.c_str()};
    std::vector<const char*> output_names;
    for (const auto& name : m_encoder_output_names) output_names.push_back(name.c_str());
    const auto model_started = std::chrono::steady_clock::now();
    auto outputs = m_encoder->Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                  output_names.data(), output_names.size());
    const auto model_finished = std::chrono::steady_clock::now();
    for (size_t index = 0; index < outputs.size(); ++index) {
        const auto info = outputs[index].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetShape().empty())
            throw std::runtime_error("SAM2 encoder returned an invalid embedding tensor.");
        require_finite(outputs[index].GetTensorData<float>(), info.GetElementCount(),
                       "SAM2 encoder returned non-finite embeddings.");
    }
    Sam2ImageContext context;
    context.owner = this;
    context.generation = m_generation;
    context.image_width = image.cols;
    context.image_height = image.rows;
    const auto finished = std::chrono::steady_clock::now();
    context.total_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    context.preprocess_ms = std::chrono::duration<double, std::milli>(model_started - started).count();
    context.model_ms = std::chrono::duration<double, std::milli>(model_finished - model_started).count();
    context.embeddings = std::move(outputs);
    return context;
}

Sam2Result Sam2Inference::Segment(const Sam2ImageContext& context, const Sam2Prompt& prompt)
{
    const auto started = std::chrono::steady_clock::now();
    if (!m_ready) throw std::logic_error("SAM2 model is not initialized.");
    if (context.owner != this || context.generation != m_generation ||
        context.embeddings.size() != m_encoder_output_names.size())
        throw std::invalid_argument("SAM2 image context does not belong to this encoder.");
    if (prompt.points.size() != prompt.labels.size())
        throw std::invalid_argument("SAM2 point and label counts differ.");
    if (!prompt.box_xyxy.empty() && prompt.box_xyxy.size() != 4)
        throw std::invalid_argument("SAM2 box prompt must contain four coordinates.");

    std::vector<cv::Point2f> points = prompt.points;
    std::vector<int64_t> labels(prompt.labels.begin(), prompt.labels.end());
    if (!prompt.box_xyxy.empty()) {
        points.emplace_back(prompt.box_xyxy[0], prompt.box_xyxy[1]);
        labels.push_back(2);
        points.emplace_back(prompt.box_xyxy[2], prompt.box_xyxy[3]);
        labels.push_back(3);
    }
    if (points.empty()) {
        points.emplace_back(0.0f, 0.0f);
        labels.push_back(-1);
    }
    std::vector<float> point_values;
    point_values.reserve(points.size() * 2);
    const float coordinate_scale_x = m_prompt_coordinate_space == "resized_input"
        ? static_cast<float>(m_input_width) / static_cast<float>(context.image_width) : 1.0f;
    const float coordinate_scale_y = m_prompt_coordinate_space == "resized_input"
        ? static_cast<float>(m_input_height) / static_cast<float>(context.image_height) : 1.0f;
    if (!(coordinate_scale_x > 0.0f) || !(coordinate_scale_y > 0.0f) ||
        !std::isfinite(coordinate_scale_x) || !std::isfinite(coordinate_scale_y))
        throw std::invalid_argument("SAM2 image context dimensions are invalid.");
    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y))
            throw std::invalid_argument("SAM2 point prompt must be finite.");
        point_values.push_back(point.x * coordinate_scale_x);
        point_values.push_back(point.y * coordinate_scale_y);
    }
    bool has_mask = !prompt.mask_input.empty();
    if (has_mask && (prompt.mask_input.type() != CV_32FC1 || prompt.mask_input.rows != m_mask_height ||
                     prompt.mask_input.cols != m_mask_width))
        throw std::invalid_argument("SAM2 mask prompt has an invalid shape or type.");
    std::vector<float> mask_values(static_cast<size_t>(m_mask_height) * m_mask_width, 0.0f);
    if (has_mask) {
        for (int row = 0; row < m_mask_height; ++row)
            std::copy(prompt.mask_input.ptr<float>(row), prompt.mask_input.ptr<float>(row) + m_mask_width,
                      mask_values.begin() + static_cast<size_t>(row) * m_mask_width);
    }
    std::vector<float> original_size{static_cast<float>(context.image_height), static_cast<float>(context.image_width)};
    std::vector<float> has_mask_value{has_mask ? 1.0f : 0.0f};
    std::vector<Ort::Value> input_values;
    std::vector<std::string> input_names;
    std::vector<Ort::Value> owned;
    input_values.reserve(m_decoder->GetInputCount());
    input_names.reserve(m_decoder->GetInputCount());
    owned.reserve(m_decoder->GetInputCount());
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t index = 0; index < m_decoder->GetInputCount(); ++index) {
        auto name = m_decoder->GetInputNameAllocated(index, m_allocator);
        const std::string tensor_name = name.get();
        input_names.push_back(tensor_name);
        std::string semantic = tensor_name;
        for (const auto& item : m_decoder_contract.inputs)
            if (item.second == tensor_name) { semantic = item.first; break; }
        auto embedding = m_encoder_output_index.find(tensor_name);
        if (embedding != m_encoder_output_index.end()) {
            const auto info = context.embeddings[embedding->second].GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.empty())
                throw std::invalid_argument("SAM2 embedding tensor must be a float tensor.");
            auto* data = const_cast<float*>(context.embeddings[embedding->second].GetTensorData<float>());
            input_values.push_back(Ort::Value::CreateTensor<float>(memory, data, info.GetElementCount(),
                                                                    shape.data(), shape.size()));
            continue;
        }
        if (semantic == "image_embeddings" || semantic == "image_features_0" || semantic == "image_features_1") {
            auto mapped = m_decoder_contract.inputs.find(semantic);
            if (mapped == m_decoder_contract.inputs.end())
                throw std::invalid_argument("SAM2 embedding input is not mapped in the contract.");
            auto embedding_index = m_encoder_output_index.find(mapped->second);
            if (embedding_index == m_encoder_output_index.end())
                throw std::invalid_argument("SAM2 embedding output is not present in the encoder graph.");
            const auto info = context.embeddings[embedding_index->second].GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.empty())
                throw std::invalid_argument("SAM2 embedding tensor must be a float tensor.");
            auto* data = const_cast<float*>(context.embeddings[embedding_index->second].GetTensorData<float>());
            input_values.push_back(Ort::Value::CreateTensor<float>(memory, data, info.GetElementCount(),
                                                                    shape.data(), shape.size()));
            continue;
        }
        if (semantic == "point_coords") {
            const std::vector<int64_t> shape{1, static_cast<int64_t>(points.size()), 2};
            owned.push_back(Ort::Value::CreateTensor<float>(memory, point_values.data(), point_values.size(),
                                                              shape.data(), shape.size()));
        } else if (semantic == "point_labels") {
            const std::vector<int64_t> shape{1, static_cast<int64_t>(labels.size())};
            owned.push_back(Ort::Value::CreateTensor<int64_t>(memory, labels.data(), labels.size(),
                                                               shape.data(), shape.size()));
        } else if (semantic == "mask_input") {
            const std::vector<int64_t> shape{1, 1, m_mask_height, m_mask_width};
            owned.push_back(Ort::Value::CreateTensor<float>(memory, mask_values.data(), mask_values.size(),
                                                              shape.data(), shape.size()));
        } else if (semantic == "has_mask_input") {
            const std::vector<int64_t> shape{1};
            owned.push_back(Ort::Value::CreateTensor<float>(memory, has_mask_value.data(), has_mask_value.size(),
                                                              shape.data(), shape.size()));
        } else if (semantic == "orig_im_size") {
            const std::vector<int64_t> shape{2};
            owned.push_back(Ort::Value::CreateTensor<float>(memory, original_size.data(), original_size.size(),
                                                              shape.data(), shape.size()));
        } else {
            throw std::invalid_argument("SAM2 decoder input is not described by the prompt contract: " + tensor_name);
        }
        input_values.push_back(std::move(owned.back()));
        owned.pop_back();
    }
    std::vector<const char*> input_name_ptrs;
    input_name_ptrs.reserve(input_names.size());
    for (const auto& name : input_names) input_name_ptrs.push_back(name.c_str());
    std::vector<const char*> output_name_ptrs;
    output_name_ptrs.reserve(m_decoder_contract.outputs.size());
    for (const auto& name : m_decoder_contract.outputs) output_name_ptrs.push_back(name.c_str());
    const auto model_start = std::chrono::steady_clock::now();
    auto outputs = m_decoder->Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), input_values.data(),
                                  input_values.size(), output_name_ptrs.data(), output_name_ptrs.size());
    const auto model_end = std::chrono::steady_clock::now();
    if (outputs.empty()) throw std::runtime_error("SAM2 decoder returned no mask output.");
    const auto mask_info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto mask_shape = mask_info.GetShape();
    if (mask_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        (mask_shape.size() != 4 && mask_shape.size() != 3 && mask_shape.size() != 2))
        throw std::runtime_error("SAM2 mask output has an unsupported type or rank.");
    int mask_count = 1;
    int mask_height = 0;
    int mask_width = 0;
    if (mask_shape.size() == 4) {
        if (mask_shape[0] != 1 || mask_shape[1] <= 0 || mask_shape[2] <= 0 || mask_shape[3] <= 0)
            throw std::runtime_error("SAM2 mask output shape is invalid.");
        mask_count = static_cast<int>(mask_shape[1]);
        mask_height = static_cast<int>(mask_shape[2]);
        mask_width = static_cast<int>(mask_shape[3]);
    } else if (mask_shape.size() == 3) {
        if (mask_shape[0] <= 0 || mask_shape[1] <= 0 || mask_shape[2] <= 0)
            throw std::runtime_error("SAM2 mask output shape is invalid.");
        mask_count = static_cast<int>(mask_shape[0]);
        mask_height = static_cast<int>(mask_shape[1]);
        mask_width = static_cast<int>(mask_shape[2]);
    } else {
        if (mask_shape[0] <= 0 || mask_shape[1] <= 0)
            throw std::runtime_error("SAM2 mask output shape is invalid.");
        mask_height = static_cast<int>(mask_shape[0]);
        mask_width = static_cast<int>(mask_shape[1]);
    }
    const float* logits = outputs[0].GetTensorData<float>();
    require_finite(logits, mask_info.GetElementCount(), "SAM2 decoder returned non-finite mask logits.");
    Sam2Result result;
    result.mask_count = mask_count;
    result.mask_height = mask_height;
    result.mask_width = mask_width;
    result.mask_logits.assign(logits, logits + mask_info.GetElementCount());
    if (outputs.size() > 1) {
        const auto score_info = outputs[1].GetTensorTypeAndShapeInfo();
        if (score_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("SAM2 score output must be float.");
        const float* scores = outputs[1].GetTensorData<float>();
        require_finite(scores, score_info.GetElementCount(), "SAM2 decoder returned non-finite scores.");
        result.scores.assign(scores, scores + score_info.GetElementCount());
    }
    if (!result.scores.empty()) {
        result.selected_mask = 0;
        for (int index = 1; index < std::min(mask_count, static_cast<int>(result.scores.size())); ++index)
            if (result.scores[index] > result.scores[result.selected_mask]) result.selected_mask = index;
    }
    result.mask = cv::Mat(mask_height, mask_width, CV_8UC1);
    const size_t plane = static_cast<size_t>(mask_height) * mask_width;
    const size_t selected_offset = static_cast<size_t>(result.selected_mask) * plane;
    for (int row = 0; row < mask_height; ++row) {
        auto* destination = result.mask.ptr<uint8_t>(row);
        for (int column = 0; column < mask_width; ++column)
            destination[column] = result.mask_logits[selected_offset + static_cast<size_t>(row) * mask_width + column] > 0.0f ? 1 : 0;
    }
    const auto end = std::chrono::steady_clock::now();
    result.preprocess_ms = std::chrono::duration<double, std::milli>(model_start - started).count();
    result.model_ms = std::chrono::duration<double, std::milli>(model_end - model_start).count();
    result.postprocess_ms = std::chrono::duration<double, std::milli>(end - model_end).count();
    result.total_ms = std::chrono::duration<double, std::milli>(end - started).count();
    return result;
}

Sam2Result Sam2Inference::Automatic(const Sam2ImageContext& context, int grid_width,
                                    int grid_height, float min_score)
{
    if (!m_ready) throw std::logic_error("SAM2 model is not initialized.");
    if (context.owner != this || context.generation != m_generation ||
        context.embeddings.size() != m_encoder_output_names.size())
        throw std::invalid_argument("SAM2 image context does not belong to this encoder.");
    if (grid_width < 1 || grid_height < 1 || grid_width > 32 || grid_height > 32)
        throw std::invalid_argument("SAM2 automatic-mask grid must be between 1 and 32 per axis.");
    if (std::isnan(min_score) || min_score > std::numeric_limits<float>::max())
        throw std::invalid_argument("SAM2 automatic-mask score threshold must be finite or -infinity.");

    const auto started = std::chrono::steady_clock::now();
    cv::Mat union_mask;
    Sam2Result best;
    float best_score = -std::numeric_limits<float>::infinity();
    double preprocess_ms = 0.0;
    double model_ms = 0.0;
    for (int row = 0; row < grid_height; ++row) {
        for (int column = 0; column < grid_width; ++column) {
            Sam2Prompt prompt;
            prompt.points.emplace_back(
                (static_cast<float>(column) + 0.5f) * static_cast<float>(context.image_width) /
                    static_cast<float>(grid_width),
                (static_cast<float>(row) + 0.5f) * static_cast<float>(context.image_height) /
                    static_cast<float>(grid_height));
            prompt.labels.push_back(1);
            auto candidate = Segment(context, prompt);
            preprocess_ms += candidate.preprocess_ms;
            model_ms += candidate.model_ms;
            const float score = candidate.scores.empty()
                ? 0.0f
                : candidate.scores[std::clamp(candidate.selected_mask, 0,
                                              static_cast<int>(candidate.scores.size()) - 1)];
            if (score > best_score) {
                best_score = score;
                best = candidate;
            }
            if (candidate.mask.empty() || score < min_score) continue;
            if (union_mask.empty()) union_mask = cv::Mat::zeros(candidate.mask.size(), CV_8UC1);
            if (union_mask.size() != candidate.mask.size())
                throw std::runtime_error("SAM2 automatic-mask candidates have different shapes.");
            cv::bitwise_or(union_mask, candidate.mask, union_mask);
        }
    }
    if (union_mask.empty()) {
        if (best.mask.empty()) throw std::runtime_error("SAM2 automatic-mask produced no masks.");
        union_mask = cv::Mat::zeros(best.mask.size(), CV_8UC1);
    }
    const auto finished = std::chrono::steady_clock::now();
    best.mask = std::move(union_mask);
    best.mask_count = 1;
    best.selected_mask = 0;
    best.total_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    best.preprocess_ms = preprocess_ms;
    best.model_ms = model_ms;
    best.postprocess_ms = std::max(0.0, best.total_ms - best.preprocess_ms - best.model_ms);
    return best;
}
