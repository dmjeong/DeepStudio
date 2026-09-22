#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dvs_bw8 {
struct Image {
    const void* data;
    int width, height;
    size_t stride;
};
inline void Validate(Image image) {
    if (!image.data || image.width <= 0 || image.height <= 0 ||
        image.stride < static_cast<size_t>(image.width) ||
        image.stride > static_cast<size_t>(PTRDIFF_MAX) / static_cast<size_t>(image.height))
        throw std::invalid_argument("Invalid BW8 pointer, dimensions or row pitch");
}

// Independent scalar implementation of the exported uint8 bilinear contract:
// half-pixel coordinates, 8-bit interpolation weights, ties-to-even coefficient
// rounding, replicated borders and a single final rounding to uint8.
struct Axis { int first, second, weight; };
inline std::vector<Axis> MakeAxis(int source, int target) {
    std::vector<Axis> result(target);
    const double scale = 1.0 / (static_cast<double>(target) / source);
    for (int i = 0; i < target; ++i) {
        // Separate operations intentionally avoid FMA changing coefficient ties.
        volatile double product = (static_cast<double>(i) + 0.5) * scale;
        const double position = product - 0.5;
        const int left = static_cast<int>(std::floor(position));
        if (left < 0) result[i] = {0, 0, 0};
        else if (left >= source - 1) result[i] = {source - 1, source - 1, 0};
        else {
            const double value = (position - left) * 256.0;
            const int base = static_cast<int>(std::floor(value));
            const double fraction = value - base;
            const int weight = base + (fraction > 0.5 || (fraction == 0.5 && (base & 1)));
            result[i] = {left, left + 1, weight};
        }
    }
    return result;
}

class Preprocessor {
public:
    Preprocessor(int width, int height, std::vector<float> mean, std::vector<float> stddev,
                 int crop_width = 0, int crop_height = 0)
        : width_(width), height_(height), crop_width_(crop_width), crop_height_(crop_height),
          mean_(std::move(mean)), std_(std::move(stddev)) {
        if (width < 1 || height < 1 || width > 65536 || height > 65536 ||
            (mean_.size() != 1 && mean_.size() != 3) || mean_.size() != std_.size() ||
            crop_width < 0 || crop_height < 0 || (crop_width == 0) != (crop_height == 0))
            throw std::invalid_argument("Invalid preprocessing configuration");
        for (size_t c = 0; c < mean_.size(); ++c)
            if (!std::isfinite(mean_[c]) || !std::isfinite(std_[c]) || std_[c] <= 0)
                throw std::invalid_argument("Invalid normalization");
    }
    void Run(Image image, std::vector<float>& output) {
        Validate(image);
        auto pixels = static_cast<const uint8_t*>(image.data);
        if (crop_width_) {
            if (image.width < crop_width_ || image.height < crop_height_)
                throw std::invalid_argument("Center crop exceeds BW8 image");
            pixels += static_cast<size_t>((image.height - crop_height_) / 2) * image.stride +
                      (image.width - crop_width_) / 2;
            image.width = crop_width_; image.height = crop_height_;
        }
        if (source_width_ != image.width || source_height_ != image.height) {
            x_ = MakeAxis(image.width, width_); y_ = MakeAxis(image.height, height_);
            source_width_ = image.width; source_height_ = image.height;
        }
        const size_t plane = static_cast<size_t>(width_) * height_;
        output.resize(plane * mean_.size());
        for (int y = 0; y < height_; ++y) {
            const auto ay = y_[y];
            const auto* row0 = pixels + static_cast<size_t>(ay.first) * image.stride;
            const auto* row1 = pixels + static_cast<size_t>(ay.second) * image.stride;
            for (int x = 0; x < width_; ++x) {
                const auto ax = x_[x];
                const unsigned top = row0[ax.first] * (256 - ax.weight) + row0[ax.second] * ax.weight;
                const unsigned bottom = row1[ax.first] * (256 - ax.weight) + row1[ax.second] * ax.weight;
                const unsigned pixel = (top * (256 - ay.weight) + bottom * ay.weight + 32768) >> 16;
                const float value = static_cast<float>(pixel) / 255.0f;
                for (size_t c = 0; c < mean_.size(); ++c)
                    output[c * plane + static_cast<size_t>(y) * width_ + x] = (value - mean_[c]) / std_[c];
            }
        }
    }
private:
    int width_, height_, crop_width_, crop_height_, source_width_ = 0, source_height_ = 0;
    std::vector<float> mean_, std_;
    std::vector<Axis> x_, y_;
};
} // namespace dvs_bw8
