// Test harness: load the actual exporter JSON without repairing it in the test.
#include "vision_inference.h"
#include "model_crypto.h"
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--runtime-version") {
        std::cout << OrtGetApiBase()->GetVersionString() << '\n';
        return 0;
    }
    if (argc != 4 && argc != 5) return 2; // config.json|model.dvsenc image map.f32 [key]
    try {
        VisionInference engine;
        if (argc == 5) {
            auto key = dvs_crypto::ReadKey(std::filesystem::u8path(argv[4]));
            const dvs_crypto::Package package(std::filesystem::u8path(argv[1]), key);
            dvs_crypto::Wipe(key.data(), key.size());
            if (!engine.InitializeFromPackage(package, "onnxruntime", 1)) return 3;
        } else if (!engine.InitializeFromJson(argv[1], "", 1)) return 3;
        const auto image = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
        if (image.empty()) return 4;
        const auto result = engine.Anomaly(image);
        if (result.anomaly_map.type() != CV_32FC1 || result.anomaly_map.size() != image.size()) return 5;
        std::ofstream map(argv[3], std::ios::binary);
        for (int row = 0; row < result.anomaly_map.rows; ++row)
            map.write(reinterpret_cast<const char*>(result.anomaly_map.ptr<float>(row)),
                      static_cast<std::streamsize>(result.anomaly_map.cols) * sizeof(float));
        map.close();
        if (!map) return 5;
        std::printf("{\"score\":%.9g,\"threshold\":%.9g,\"anomalous\":%s}\n",
                    result.score, result.threshold, result.anomalous ? "true" : "false");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 6;
    }
    return 0;
}
