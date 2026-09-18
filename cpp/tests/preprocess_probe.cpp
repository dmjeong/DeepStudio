#include "opencv_preprocess.h"
#include <iostream>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// Binary stdin/stdout bridge for the Python OpenCV parity tests.
int main(int argc, char** argv) {
    if (argc != 8 && argc != 10) return 2;
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    const int width = std::stoi(argv[1]), height = std::stoi(argv[2]);
    const int channels = std::stoi(argv[3]), out_width = std::stoi(argv[4]), out_height = std::stoi(argv[5]);
    const bool grayscale = std::stoi(argv[6]), normalized = std::stoi(argv[7]);
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * channels);
    if (!std::cin.read(reinterpret_cast<char*>(pixels.data()), pixels.size())) return 3;
    if (grayscale && channels == 3) {
        pixels = vision_preprocess::gray(pixels, width, height);
    }
    const int output_channels = grayscale ? 1 : channels;
    const bool native = argc == 10;
    std::vector<int> resize_size;
    if (native) {
        resize_size.push_back(std::stoi(argv[8]));
        if (std::stoi(argv[9]) > 0) resize_size.push_back(std::stoi(argv[9]));
    }
    auto resized = native
        ? vision_preprocess::classification_resize(pixels, width, height, output_channels, out_width, out_height, resize_size)
        : vision_preprocess::resize(pixels, width, height, output_channels, out_width, out_height);
    if (normalized) {
        auto tensor = vision_preprocess::normalize(resized, output_channels,
            native ? std::vector<float>{0,0,0} : output_channels == 1 ? std::vector<float>{.449f} : std::vector<float>{.485f,.456f,.406f},
            native ? std::vector<float>{1,1,1} : output_channels == 1 ? std::vector<float>{.226f} : std::vector<float>{.229f,.224f,.225f});
        if (!native) {
            cv::Mat source(height, width, CV_MAKETYPE(CV_8U, output_channels), pixels.data());
            tensor = vision_preprocess::resize_normalize(source, out_width, out_height,
                output_channels == 1 ? std::vector<float>{.449f} : std::vector<float>{.485f,.456f,.406f},
                output_channels == 1 ? std::vector<float>{.226f} : std::vector<float>{.229f,.224f,.225f});
        }
        std::cout.write(reinterpret_cast<const char*>(tensor.data()), tensor.size() * sizeof(float));
    } else std::cout.write(reinterpret_cast<const char*>(resized.data()), resized.size());
}
