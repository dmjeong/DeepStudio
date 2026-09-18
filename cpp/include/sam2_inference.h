#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct Sam2Prompt
{
    std::vector<cv::Point2f> points;
    std::vector<int32_t> labels;
    std::vector<float> box_xyxy;
    cv::Mat mask_input; // optional CV_32FC1 low-resolution mask
};

class Sam2Inference;

struct Sam2ImageContext
{
    const Sam2Inference* owner = nullptr;
    int image_width = 0;
    int image_height = 0;
    std::vector<Ort::Value> embeddings;
};

struct Sam2Result
{
    cv::Mat mask; // selected low-resolution binary mask, CV_8UC1
    std::vector<float> mask_logits;
    std::vector<float> scores;
    int mask_count = 0;
    int mask_height = 0;
    int mask_width = 0;
    int selected_mask = 0;
    double total_ms = 0.0;
    double model_ms = 0.0;
    double postprocess_ms = 0.0;
};

class Sam2Inference
{
public:
    Sam2Inference();
    ~Sam2Inference();

    static bool LooksLikeConfig(const std::string& config_path);
    bool InitializeFromJson(const std::string& config_path,
                            const std::string& runtime = "",
                            int num_threads = -1);
    bool IsReady() const { return m_ready; }
    const std::string& LastError() const { return m_error; }

    Sam2ImageContext Encode(const cv::Mat& image);
    Sam2Result Segment(const Sam2ImageContext& context, const Sam2Prompt& prompt);
    // Run a deterministic positive-point grid and union accepted masks. This
    // is the native SDK's automatic-mask primitive; upstream video memory is
    // intentionally a separate contract and is not inferred from this call.
    Sam2Result Automatic(const Sam2ImageContext& context, int grid_width, int grid_height,
                         float min_score = -std::numeric_limits<float>::infinity());

private:
    struct GraphContract
    {
        std::string path;
        std::vector<std::string> outputs;
        std::unordered_map<std::string, std::string> inputs; // semantic -> tensor name
        std::unordered_map<std::string, std::string> embedding_inputs; // decoder semantic -> encoder output
    };

    std::vector<float> Preprocess(const cv::Mat& image) const;
    void Fail(const std::string& message);

    Ort::Env m_env;
    Ort::AllocatorWithDefaultOptions m_allocator;
    std::unique_ptr<Ort::Session> m_encoder;
    std::unique_ptr<Ort::Session> m_decoder;
    GraphContract m_encoder_contract;
    GraphContract m_decoder_contract;
    std::vector<std::string> m_encoder_output_names;
    std::unordered_map<std::string, size_t> m_encoder_output_index;
    std::string m_input_name = "input_image";
    int m_input_channels = 3;
    int m_input_height = 1024;
    int m_input_width = 1024;
    std::vector<float> m_normalize_mean{0.485f, 0.456f, 0.406f};
    std::vector<float> m_normalize_std{0.229f, 0.224f, 0.225f};
    int m_mask_height = 256;
    int m_mask_width = 256;
    bool m_ready = false;
    std::string m_error;
};
