#include "bw8_preprocess.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <random>

int main() {
    try {
        std::mt19937 rng(71);
        int cases = 0;
        for (int k = 0; k < 300; ++k) {
            const int w = k == 12 ? 448 : k == 18 ? 224 : k < 12 ? k + 1 : 1 + rng() % 1024;
            const int h = k == 12 ? 448 : k == 18 ? 224 : k < 12 ? 1 : 1 + rng() % 768;
            const int dw = k % 3 == 0 ? 224 : 1 + rng() % 400;
            const int dh = k % 3 == 0 ? 224 : 1 + rng() % 300;
            const size_t stride = w + 17;
            std::vector<uint8_t> pixels(stride * h);
            for (auto& p : pixels) p = static_cast<uint8_t>(rng());
            for (int channels : {1, 3}) {
                const std::vector<float> mean(channels, .37f), stddev(channels, .23f);
                const int cw = k % 2 ? std::max(1, w / 2) : 0;
                const int ch = k % 2 ? std::max(1, h / 2) : 0;
                dvs_bw8::Preprocessor pre(dw, dh, mean, stddev, cw, ch);
                std::vector<float> actual;
                // Run a different frame first: cached rows must not leak into the next Run.
                std::vector<uint8_t> previous(pixels.size());
                for (size_t i = 0; i < pixels.size(); ++i) previous[i] = static_cast<uint8_t>(255 - pixels[i]);
                pre.Run({previous.data(), w, h, stride}, actual);
                pre.Run({pixels.data(), w, h, stride}, actual);
                cv::Mat src(h, w, CV_8UC1, pixels.data(), stride), resized;
                if (cw) src = src(cv::Rect((w - cw) / 2, (h - ch) / 2, cw, ch));
                if (channels == 3) {
                    cv::Mat rgb;
                    cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB);
                    src = rgb;
                }
                cv::resize(src, resized, cv::Size(dw, dh), 0, 0, cv::INTER_LINEAR_EXACT);
                for (int c = 0; c < channels; ++c)
                    for (int y = 0; y < dh; ++y)
                        for (int x = 0; x < dw; ++x) {
                            const float expected = (resized.ptr<uint8_t>(y)[x * channels + c] / 255.0f - mean[c]) / stddev[c];
                            if (std::abs(expected - actual[(c * dh + y) * dw + x]) > 1e-6f)
                                throw std::runtime_error("Preprocess mismatch at case " + std::to_string(k));
                        }
                ++cases;
            }
        }
        // A native ROI must preserve the centered pixels and parent row pitch.
        const int width = 7, height = 5, pitch = 11;
        std::vector<uint8_t> frame(pitch * height);
        for (size_t i = 0; i < frame.size(); ++i) frame[i] = static_cast<uint8_t>(i);
        std::vector<float> values;
        dvs_bw8::Preprocessor cropped(3, 3, {0.f}, {1.f}, 3, 3, true);
        cropped.Run({frame.data(), width, height, pitch}, values);
        for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 3; ++x)
                if (values[y * 3 + x] != frame[(y + 1) * pitch + x + 2] / 255.f)
                    throw std::runtime_error("Native centered BW8 pixels differ");
        dvs_bw8::Preprocessor native(3, 3, {0.f}, {1.f}, 0, 0, true);
        try {
            native.Run({frame.data(), width, height, pitch}, values);
            throw std::runtime_error("Wrong-size native BW8 ROI accepted");
        } catch (const std::invalid_argument&) {
            // The model requires a 3 x 3 ROI from the caller.
        }
        std::cout << "PASS: " << cases << " BW8 resize/stride/crop/normalization comparisons\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
