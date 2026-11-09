// ═══════════════════════════════════════════════════════════════════════
//  INT8 테스트 솔루션: 같은 영상에서 FP32 모델과 INT8 모델을 C++ SDK로 돌려 비교한다.
//
//    int8_compare <fp32.json> <int8.json> <영상 또는 폴더> [스레드=0(자동)] [반복=5] [최소면적=20]
//
//    영상마다 ─┬─ FP32 Segment ×반복 → 시간 중앙값, 마스크, 결함 목록
//             └─ INT8 Segment ×반복 → 시간 중앙값, 마스크, 결함 목록
//             ─▶ 마스크 픽셀 일치율 · 결함 짝짓기(같은 클래스 + 상자 IoU ≥ 0.3)
//
//    결함 목록은 모델 JSON에 판정 기준(postprocessing.defect_rules)이 있을 때만 나온다
//    (데스크톱에서 '오검출 0 기준 찾기'로 저장한 뒤 내보낸 모델). 없으면 마스크 일치율만 본다.
//    최소면적: 비교할 때 이보다 작은 덩어리는 무시한다 (경계에 걸린 1~10 px 조각은 INT8에서 흔들린다).
//
//    종료 코드: 0 = 모든 결함 짝이 맞음, 1 = 한쪽에만 있는 결함 있음, 2 = 사용법/로드 오류
// ═══════════════════════════════════════════════════════════════════════
#include "vision_inference.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;
using Defect = SegmentResult::Defect;

namespace {

// 폴더면 안의 영상 파일 전부(이름순), 파일이면 그 하나
std::vector<fs::path> ListImages(const fs::path& input)
{
    std::vector<fs::path> images;
    if (fs::is_regular_file(input)) return {input};
    for (const auto& entry : fs::directory_iterator(input)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (ext == ".png" || ext == ".bmp" || ext == ".jpg" || ext == ".jpeg" || ext == ".tif" || ext == ".tiff")
            images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    return images;
}

// Segment를 반복 실행해 마지막 결과와 시간 중앙값(ms)을 돌려준다 (첫 실행은 예열이라 버린다)
SegmentResult Run(VisionInference& engine, const cv::Mat& image, int repeat, double& median_ms)
{
    engine.Segment(image);  // 예열: 메모리 할당·스레드 풀 준비
    std::vector<double> times;
    SegmentResult result;
    for (int i = 0; i < repeat; ++i) {
        result = engine.Segment(image);
        times.push_back(result.inference_ms);
    }
    std::sort(times.begin(), times.end());
    median_ms = times[times.size() / 2];
    return result;
}

double BoxIoU(const cv::Rect& a, const cv::Rect& b)
{
    const double inter = (a & b).area();
    const double uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 4 || argc > 7) {
        std::cerr << "usage: int8_compare <fp32.json> <int8.json> <image|folder> [threads=0] [repeat=5] [min_area=20]\n";
        return 2;
    }
    const int threads = argc > 4 ? std::atoi(argv[4]) : 0;         // 0 = 물리 코어 수 자동
    const int repeat = argc > 5 ? std::max(1, std::atoi(argv[5])) : 5;
    const int min_area = argc > 6 ? std::max(1, std::atoi(argv[6])) : 20;

    VisionInference fp32, int8;
    if (!fp32.InitializeFromJson(argv[1], "", threads) || !int8.InitializeFromJson(argv[2], "", threads)) {
        std::cerr << "모델 로드 실패 (JSON 옆에 .onnx가 있는지, onnxruntime.dll 버전이 맞는지 확인)\n";
        return 2;
    }
    const auto images = ListImages(argv[3]);
    if (images.empty()) {
        std::cerr << "영상이 없습니다: " << argv[3] << "\n";
        return 2;
    }

    std::printf("스레드 %s · 반복 %d · 비교 최소 면적 %d px\n", threads ? std::to_string(threads).c_str() : "자동",
                repeat, min_area);
    std::printf("%-28s %10s %10s %7s %12s %6s %8s %8s\n", "영상", "FP32 ms", "INT8 ms", "배속",
                "픽셀 일치", "짝", "FP32만", "INT8만");

    double sum_fp32 = 0, sum_int8 = 0;
    int matched_total = 0, only_total = 0, rules_seen = 0;
    for (const auto& path : images) {
        const cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            std::printf("%-28s 읽기 실패\n", path.filename().string().c_str());
            continue;
        }
        double ms_a = 0, ms_b = 0;
        SegmentResult a, b;
        try {
            a = Run(fp32, image, repeat, ms_a);
            b = Run(int8, image, repeat, ms_b);
        } catch (const std::exception& error) {
            std::printf("%-28s 오류: %s\n", path.filename().string().c_str(), error.what());
            continue;
        }
        sum_fp32 += ms_a;
        sum_int8 += ms_b;

        // ① 마스크 픽셀 일치율 (판정 기준과 무관하게 항상 비교 가능)
        const double agree = 1.0 - double(cv::countNonZero(a.mask != b.mask)) / double(a.mask.total());

        // ② 결함 짝짓기: 최소 면적 이상만, 같은 클래스 + 상자 IoU ≥ 0.3
        std::vector<Defect> da, db;
        for (const auto& d : a.defects) if (d.area >= min_area) da.push_back(d);
        for (const auto& d : b.defects) if (d.area >= min_area) db.push_back(d);
        if (!a.defects.empty() || !b.defects.empty()) ++rules_seen;
        std::vector<bool> used(db.size(), false);
        std::vector<std::string> notes;
        int matched = 0;
        for (const auto& d : da) {
            int hit = -1;
            for (size_t j = 0; j < db.size() && hit < 0; ++j)
                if (!used[j] && db[j].class_id == d.class_id && BoxIoU(d.box, db[j].box) >= 0.3) hit = int(j);
            if (hit >= 0) { used[hit] = true; ++matched; }
            else notes.push_back("    FP32만: class " + std::to_string(d.class_id) + " 면적 " + std::to_string(d.area) +
                                 " 점수 " + std::to_string(d.score).substr(0, 4));
        }
        for (size_t j = 0; j < db.size(); ++j)
            if (!used[j]) notes.push_back("    INT8만: class " + std::to_string(db[j].class_id) + " 면적 " +
                                          std::to_string(db[j].area) + " 점수 " + std::to_string(db[j].score).substr(0, 4));
        const int only_a = int(da.size()) - matched, only_b = int(db.size()) - (matched);
        matched_total += matched;
        only_total += only_a + only_b;

        std::printf("%-28s %10.1f %10.1f %6.2fx %11.5f %6d %8d %8d\n", path.filename().string().c_str(), ms_a, ms_b,
                    ms_b > 0 ? ms_a / ms_b : 0.0, agree, matched, only_a, only_b);
        for (const auto& note : notes) std::printf("%s\n", note.c_str());
    }

    const double n = double(images.size());
    std::printf("\n평균 FP32 %.1f ms · INT8 %.1f ms → %.2f배 빠름\n", sum_fp32 / n, sum_int8 / n,
                sum_int8 > 0 ? sum_fp32 / sum_int8 : 0.0);
    if (rules_seen == 0)
        std::printf("※ 결함 목록이 비어 있습니다: 모델 JSON에 판정 기준(defect_rules)이 없으면 마스크 일치율만 비교됩니다.\n");
    std::printf("결함: 짝 맞음 %d개 · 한쪽에만 %d개 → %s\n", matched_total, only_total,
                only_total == 0 ? "통과 (INT8 사용 가능)" : "확인 필요 (위 목록의 결함을 영상에서 보세요)");
    return only_total == 0 ? 0 : 1;
}
