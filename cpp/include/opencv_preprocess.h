#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace vision_preprocess {
inline void resize_normalize_into(const cv::Mat& source, int width, int height,
    const std::vector<float>& mean, const std::vector<float>& stddev,
    float* tensor, size_t count, cv::Mat& scratch) {
    const int channels = source.channels();
    if (source.empty() || source.depth() != CV_8U || (channels != 1 && channels != 3) ||
        width < 1 || height < 1 || mean.size() != static_cast<size_t>(channels) || stddev.size() != mean.size())
        throw std::invalid_argument("Invalid preprocessing dimensions.");
    const size_t plane = static_cast<size_t>(width) * height;
    if (!tensor || count != plane * channels)
        throw std::invalid_argument("Invalid preprocessing destination.");
    for (int c = 0; c < channels; ++c)
        if (!std::isfinite(mean[c]) || !std::isfinite(stddev[c]) || stddev[c] <= 0)
            throw std::invalid_argument("Invalid normalization coefficients.");
    // Same-size resize is an identity. Read rows directly, including strided ROIs.
    const cv::Mat* resized = &source;
    if (source.cols != width || source.rows != height) {
        cv::resize(source, scratch, cv::Size(width, height), 0, 0, cv::INTER_LINEAR_EXACT);
        resized = &scratch;
    }
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = resized->ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < channels; ++c)
                tensor[c * plane + static_cast<size_t>(y) * width + x] =
                    (static_cast<float>(row[x * channels + c]) / 255.0f - mean[c]) / stddev[c];
    }
}

inline std::vector<float> resize_normalize(const cv::Mat& source, int width, int height,
    const std::vector<float>& mean, const std::vector<float>& stddev) {
    if (width < 1 || height < 1 || (source.channels() != 1 && source.channels() != 3))
        throw std::invalid_argument("Invalid preprocessing dimensions.");
    std::vector<float> tensor(static_cast<size_t>(width) * height * source.channels());
    cv::Mat scratch;
    resize_normalize_into(source, width, height, mean, stddev, tensor.data(), tensor.size(), scratch);
    return tensor;
}

// RGB or gray bytes. Color conversion and resizing are performed by OpenCV.
inline std::vector<uint8_t> gray(const std::vector<uint8_t>& rgb, int width, int height) {
    if (width < 1 || height < 1 || rgb.size() != static_cast<size_t>(width) * height * 3)
        throw std::invalid_argument("Invalid RGB dimensions.");
    cv::Mat source(height, width, CV_8UC3, const_cast<uint8_t*>(rgb.data())), output;
    cv::cvtColor(source, output, cv::COLOR_RGB2GRAY);
    return {output.datastart, output.dataend};
}

inline std::vector<uint8_t> resize(const std::vector<uint8_t>& pixels,
    int width, int height, int channels, int target_width, int target_height) {
    if (width < 1 || height < 1 || target_width < 1 || target_height < 1 ||
        (channels != 1 && channels != 3) || pixels.size() != static_cast<size_t>(width) * height * channels)
        throw std::invalid_argument("Invalid OpenCV resize dimensions.");
    cv::Mat source(height, width, CV_MAKETYPE(CV_8U, channels), const_cast<uint8_t*>(pixels.data())), output;
    cv::resize(source, output, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR_EXACT);
    return {output.datastart, output.dataend};
}

inline std::vector<float> normalize(const std::vector<uint8_t>& pixels, int channels,
                                    const std::vector<float>& mean, const std::vector<float>& stddev) {
    if ((channels != 1 && channels != 3) || pixels.size() % channels ||
        mean.size() != static_cast<size_t>(channels) || stddev.size() != mean.size())
        throw std::invalid_argument("Invalid normalization dimensions.");
    for (int c = 0; c < channels; ++c)
        if (!std::isfinite(mean[c]) || !std::isfinite(stddev[c]) || stddev[c] <= 0)
            throw std::invalid_argument("Invalid normalization coefficients.");
    const size_t plane = pixels.size() / channels;
    std::vector<float> result(pixels.size());
    for (size_t pixel = 0; pixel < plane; ++pixel)
        for (int c = 0; c < channels; ++c)
            result[c * plane + pixel] = (static_cast<float>(pixels[pixel * channels + c]) / 255.0f - mean[c]) / stddev[c];
    return result;
}

inline int center_offset(int difference) {
    if (difference < 0) throw std::invalid_argument("Center crop would require padding.");
    // round(difference / 2.0), ties to even as in torchvision/Python.
    const int half = difference / 2;
    return half + ((difference % 2 && half % 2) ? 1 : 0);
}

inline std::vector<uint8_t> classification_resize(const std::vector<uint8_t>& pixels,
    int width, int height, int channels, int target_width, int target_height,
    const std::vector<int>& resize_size) {
    if (width < 1 || height < 1 || target_width < 1 || target_height < 1 ||
        (resize_size.size() != 1 && resize_size.size() != 2) ||
        std::any_of(resize_size.begin(), resize_size.end(), [](int v) { return v < 1; }))
        throw std::invalid_argument("Invalid classification resize dimensions.");
    int64_t new_width, new_height;
    if (resize_size.size() == 1) {
        const int short_edge = resize_size[0];
        new_width = width <= height ? short_edge : static_cast<int64_t>(short_edge) * width / height;
        new_height = height <= width ? short_edge : static_cast<int64_t>(short_edge) * height / width;
    } else {
        new_height = resize_size[0];
        new_width = resize_size[1];
    }
    if (new_width > std::numeric_limits<int>::max() || new_height > std::numeric_limits<int>::max())
        throw std::invalid_argument("Resized classification image is too large.");
    const int left = center_offset(static_cast<int>(new_width) - target_width);
    const int top = center_offset(static_cast<int>(new_height) - target_height);
    auto resized = resize(pixels, width, height, channels, static_cast<int>(new_width), static_cast<int>(new_height));
    std::vector<uint8_t> cropped(static_cast<size_t>(target_width) * target_height * channels);
    for (int y = 0; y < target_height; ++y) {
        const auto begin = resized.begin() + ((static_cast<size_t>(top) + y) * new_width + left) * channels;
        std::copy_n(begin, static_cast<size_t>(target_width) * channels,
                    cropped.begin() + static_cast<size_t>(y) * target_width * channels);
    }
    return cropped;
}
}  // namespace vision_preprocess
