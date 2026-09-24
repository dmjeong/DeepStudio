#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dvs_bw8 {
// Caller-owned 8-bit grayscale view. stride is the row pitch in bytes.
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
// One entry per output coordinate: blend source[first] and source[second] with
// weight/256 on the second sample. Built once per source size, O(target).
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

// Center crop -> separable uint8 bilinear resize -> per-channel normalization
// into an NCHW float tensor. Buffers are reused between calls, so steady-state
// inference performs no allocation.
//
// The resize runs as two passes, like OpenCV's INTER_LINEAR_EXACT:
//   1. horizontal: one source row -> width_ fixed-point values (x256),
//   2. vertical:   blend two horizontal rows -> one uint8 output row.
// Horizontal rows are cached by source row index, so neighbouring output rows
// that read the same source rows (always the case when enlarging) reuse them
// instead of interpolating them again; rows nobody shares take a fused
// single-loop path. The integer arithmetic is exactly the same as the direct
// per-pixel formula, so results are bit-identical.
class Preprocessor {
public:
    Preprocessor(int width, int height, std::vector<float> mean, std::vector<float> stddev,
                 int crop_width = 0, int crop_height = 0)
        : width_(width), height_(height), crop_width_(crop_width), crop_height_(crop_height),
          channel_count_(mean.size()),
          mean_(std::move(mean)), std_(std::move(stddev)) {
        if (width < 1 || height < 1 || width > 65536 || height > 65536 ||
            (mean_.size() != 1 && mean_.size() != 3) || mean_.size() != std_.size() ||
            crop_width < 0 || crop_height < 0 || (crop_width == 0) != (crop_height == 0))
            throw std::invalid_argument("Invalid preprocessing configuration");
        for (size_t c = 0; c < mean_.size(); ++c)
            if (!std::isfinite(mean_[c]) || !std::isfinite(std_[c]) || std_[c] <= 0)
                throw std::invalid_argument("Invalid normalization");
        BuildNormalizationLuts();
        for (auto& row : rows_) row.resize(width_);
    }

    void Run(Image image, std::vector<float>& output) {
        Validate(image);
        const auto* pixels = ApplyCenterCrop(image);
        UpdateAxes(image.width, image.height);

        const size_t plane = static_cast<size_t>(width_) * height_;
        const size_t total = plane * channel_count_;
        if (output.size() != total) output.resize(total);

        // Cached rows belong to the previous frame; the pixels may have changed.
        cached_row_ = {-1, -1};
        // Locals, not members, inside the hot loops: a uint8_t store may alias
        // any object, which otherwise forces the compiler to reload this->...
        const int width = width_;
        float* out = output.data();
        if (channel_count_ == 1) {
            // Single plane: normalize inside the resize loop, no extra pass.
            const float* lut = luts_[0].data();
            for (int y = 0; y < height_; ++y, out += width)
                ResizeRow(pixels, image.stride, y, [out, lut](int x, unsigned v) { out[x] = lut[v]; });
            return;
        }
        // Three planes: the same gray value is normalized into R, G and B.
        // Each plane row is written contiguously, three streams in parallel.
        const float *lut0 = luts_[0].data(), *lut1 = luts_[1].data(), *lut2 = luts_[2].data();
        for (int y = 0; y < height_; ++y, out += width) {
            float *r = out, *g = out + plane, *b = out + 2 * plane;
            ResizeRow(pixels, image.stride, y, [=](int x, unsigned v) {
                r[x] = lut0[v]; g[x] = lut1[v]; b[x] = lut2[v];
            });
        }
    }

private:
    // uint8 -> normalized float for every possible pixel value: 256 entries
    // replace one divide and one subtract per output element.
    void BuildNormalizationLuts() {
        for (size_t c = 0; c < channel_count_; ++c) {
            luts_[c].resize(256);
            for (int v = 0; v < 256; ++v)
                luts_[c][v] = (static_cast<float>(v) / 255.0f - mean_[c]) / std_[c];
        }
    }

    // Moves the view to the centered crop window; the pitch stays the parent's.
    const uint8_t* ApplyCenterCrop(Image& image) const {
        auto pixels = static_cast<const uint8_t*>(image.data);
        if (!crop_width_) return pixels;
        if (image.width < crop_width_ || image.height < crop_height_)
            throw std::invalid_argument("Center crop exceeds BW8 image");
        pixels += static_cast<size_t>((image.height - crop_height_) / 2) * image.stride +
                  (image.width - crop_width_) / 2;
        image.width = crop_width_; image.height = crop_height_;
        return pixels;
    }

    // Coordinate tables depend only on the source size. Fixed-size ROIs build
    // them once; a size change rebuilds them in O(width_ + height_).
    void UpdateAxes(int source_width, int source_height) {
        if (source_width_ == source_width && source_height_ == source_height) return;
        x_ = MakeAxis(source_width, width_); y_ = MakeAxis(source_height, height_);
        source_width_ = source_width; source_height_ = source_height;
    }

    // Produces one output row, passing each resized uint8 value to store(x, v).
    // Per row, picks the cheaper plan:
    //  - shared rows (enlarging / mild shrinking): separable pass through the
    //    two-slot row cache, so each source row is interpolated only once;
    //  - private rows (strong shrinking, same size): one fused loop straight
    //    from the source, avoiding the intermediate uint16 buffer entirely.
    // Both plans evaluate the identical integer formula.
    template<class Store>
    void ResizeRow(const uint8_t* pixels, size_t stride, int y, Store store) {
        const Axis ay = y_[y];
        // A zero vertical weight ignores the second row, so it is never read.
        const bool one_row = ay.weight == 0 || ay.second == ay.first;
        const bool next_shares = y + 1 < height_ &&
            (y_[y + 1].first == ay.first || y_[y + 1].first == ay.second);
        const bool cached = IsCached(ay.first) || (!one_row && IsCached(ay.second));
        const unsigned top_weight = 256 - ay.weight, bottom_weight = ay.weight;
        const int width = width_;

        if (next_shares || cached) {
            const uint16_t* top = HorizontalRow(pixels, stride, ay.first, -1);
            const uint16_t* bottom = one_row ? top : HorizontalRow(pixels, stride, ay.second, ay.first);
            for (int x = 0; x < width; ++x)
                store(x, (top[x] * top_weight + bottom[x] * bottom_weight + 32768) >> 16);
            return;
        }

        const Axis* axis = x_.data();
        const uint8_t* row0 = pixels + static_cast<size_t>(ay.first) * stride;
        if (one_row) {
            // (top * 256 + 32768) >> 16 with a zero bottom weight.
            for (int x = 0; x < width; ++x) {
                const Axis ax = axis[x];
                const unsigned top = row0[ax.first] * (256 - ax.weight) + row0[ax.second] * ax.weight;
                store(x, (top * 256 + 32768) >> 16);
            }
            return;
        }
        const uint8_t* row1 = pixels + static_cast<size_t>(ay.second) * stride;
        for (int x = 0; x < width; ++x) {
            const Axis ax = axis[x];
            const unsigned top = row0[ax.first] * (256 - ax.weight) + row0[ax.second] * ax.weight;
            const unsigned bottom = row1[ax.first] * (256 - ax.weight) + row1[ax.second] * ax.weight;
            store(x, (top * top_weight + bottom * bottom_weight + 32768) >> 16);
        }
    }

    bool IsCached(int source_y) const {
        return cached_row_[0] == source_y || cached_row_[1] == source_y;
    }

    // Returns the horizontally interpolated source row (values x256, max 65280
    // so uint16 is enough). Two slots suffice because y_ is non-decreasing:
    // each output row needs rows (first, second) and the next row's first is
    // usually the previous second. 'keep' is the row that must not be evicted.
    const uint16_t* HorizontalRow(const uint8_t* pixels, size_t stride, int source_y, int keep) {
        for (int slot = 0; slot < 2; ++slot)
            if (cached_row_[slot] == source_y) return rows_[slot].data();
        // Miss: overwrite the slot that does not hold 'keep'.
        const int slot = cached_row_[0] == keep ? 1 : 0;
        const uint8_t* src = pixels + static_cast<size_t>(source_y) * stride;
        uint16_t* dst = rows_[slot].data();
        const Axis* axis = x_.data();
        const int width = width_;
        for (int x = 0; x < width; ++x) {
            const Axis ax = axis[x];
            dst[x] = static_cast<uint16_t>(src[ax.first] * (256 - ax.weight) + src[ax.second] * ax.weight);
        }
        cached_row_[slot] = source_y;
        return dst;
    }

    int width_, height_, crop_width_, crop_height_, source_width_ = 0, source_height_ = 0;
    size_t channel_count_;
    std::vector<float> mean_, std_;
    std::array<std::vector<float>, 3> luts_;
    std::vector<Axis> x_, y_;
    std::array<std::vector<uint16_t>, 2> rows_;  // horizontal pass ring buffer
    std::array<int, 2> cached_row_{{-1, -1}};    // source row held by each slot
};
} // namespace dvs_bw8
