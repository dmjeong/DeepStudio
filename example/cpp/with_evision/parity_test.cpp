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
        std::cout << "PASS: " << cases << " BW8 resize/stride/crop/normalization comparisons\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
