#include "opencv_preprocess.h"
#include <cstring>
#include <iostream>
#include <limits>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        // Every byte value in every channel, with different channel values and
        // non-contiguous row stride. No inference/model tolerance can hide a change.
        for (const int channels : {1, 3}) {
            cv::Mat backing(19, 23, CV_MAKETYPE(CV_8U, channels), cv::Scalar::all(77));
            cv::Mat pixels = backing(cv::Rect(2, 1, 16, 16));
            require(!pixels.isContinuous(), "Fixture must have padded rows.");
            for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 16; ++x)
                    for (int c = 0; c < channels; ++c)
                        pixels.ptr<uint8_t>(y)[x * channels + c] = static_cast<uint8_t>(y * 16 + x + c * 73);
            const cv::Mat original = backing.clone();
            for (const bool padded : {false, true}) {
                const int width = padded ? 21 : 16, height = padded ? 18 : 16;
                const size_t plane = static_cast<size_t>(width) * height;
                for (const bool identity : {false, true}) {
                    std::vector<float> mean(channels, 0.449f), stddev(channels, 0.226f);
                    if (channels == 3) { mean = {0.485f, 0.456f, 0.406f}; stddev = {0.229f, 0.224f, 0.225f}; }
                    if (identity) { mean.assign(channels, 0.0f); stddev.assign(channels, 1.0f); }
                    const auto actual = vision_preprocess::normalize_padded_u8(pixels, width, height, mean, stddev);
                    require(actual.size() == plane * channels, "Wrong tensor size.");
                    for (int c = 0; c < channels; ++c)
                        for (int y = 0; y < height; ++y)
                            for (int x = 0; x < width; ++x) {
                                float expected = 0.0f;
                                if (x < pixels.cols && y < pixels.rows)
                                    expected = (static_cast<float>(pixels.ptr<uint8_t>(y)[x * channels + c]) /
                                                255.0f - mean[c]) / stddev[c];
                                const float value = actual[c * plane + static_cast<size_t>(y) * width + x];
                                require(std::memcmp(&value, &expected, sizeof(float)) == 0,
                                        "Normalization or padding changed bits.");
                            }
                }
            }
            require(cv::norm(backing, original, cv::NORM_INF) == 0, "Caller image was modified.");
        }
        const cv::Mat pixels(2, 3, CV_8UC1, cv::Scalar(128));
        for (const float invalid_std : {0.0f, -1.0f, std::numeric_limits<float>::infinity()}) {
            bool rejected = false;
            try { vision_preprocess::normalize_padded_u8(pixels, 3, 2, {0.0f}, {invalid_std}); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Invalid normalization accepted.");
        }
        bool rejected = false;
        try { vision_preprocess::normalize_padded_u8(pixels, 2, 2, {0.0f}, {1.0f}); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Input larger than destination accepted.");
        std::cout << "Whole-region normalization: bitwise parity, padding and stride passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
