// 분할 배포 계약 확인용: 설정(JSON)과 영상을 받아 Segment 결과를 내보낸다 (Python 일치 테스트가 쓴다).
//   segment_probe <config.json|model.dvsenc> <image> <mask.png> [threads=1] [repeat=1] [key]
//   표준 출력: 첫 줄 "removed <n>", 이후 결함마다 "class_id x y w h area score"
//   표준 오류: "segment_ms <중앙값>" (모델 로드 제외, 반복 실행의 중앙값)
#include "vision_inference.h"
#include "model_crypto.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char** argv)
{
    if (argc < 4 || argc > 7) {
        std::cerr << "usage: segment_probe <config.json|model.dvsenc> <image> <mask.png> [threads] [repeat] [key]\n";
        return 2;
    }
    VisionInference engine;
    const int threads = argc > 4 ? std::atoi(argv[4]) : 1, repeat = argc > 5 ? std::max(1, std::atoi(argv[5])) : 1;
    try {
        if (argc == 7) {
            auto key = dvs_crypto::ReadKey(std::filesystem::u8path(argv[6]));
            const dvs_crypto::Package package(std::filesystem::u8path(argv[1]), key);
            dvs_crypto::Wipe(key.data(), key.size());
            if (!engine.InitializeFromPackage(package, "onnxruntime", threads)) return 3;
        } else if (!engine.InitializeFromJson(argv[1], "", threads)) return 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 3;
    }
    const cv::Mat image = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
    if (image.empty()) return 4;
    try {
        SegmentResult result;
        std::vector<double> times;
        for (int run = 0; run < repeat; ++run) {
            result = engine.Segment(image);
            times.push_back(result.inference_ms);
        }
        std::sort(times.begin(), times.end());
        std::fprintf(stderr, "segment_ms %.1f\n", times[times.size() / 2]);
        if (!cv::imwrite(argv[3], result.mask)) return 5;
        std::printf("removed %d\n", result.removed_defects);
        for (const auto& defect : result.defects)
            std::printf("%d %d %d %d %d %d %.4f\n", defect.class_id, defect.box.x, defect.box.y,
                        defect.box.width, defect.box.height, defect.area, defect.score);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 6;
    }
    return 0;
}
