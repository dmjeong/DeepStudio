// ═══════════════════════════════════════════════════════
//  영상 평가 예제: 내보낸 ONNX 모델(JSON)에 영상을 넣어 C++ SDK 결과와 시간을 본다.
//
//    image_eval [model.json] [영상 또는 폴더] [--threads N] [--repeat N] [--roi x,y,w,h] [--output 폴더] [--no-save]
//
//    인자 없이 실행하면 아래 '기본값'을 쓴다 → 소스의 경로 두 개만 고쳐서 바로 돌릴 수 있다.
//
//    model.json ─▶ InitializeFromJson ─▶ 태스크에 맞는 함수 한 번 호출
//                                          ├─ segment : Segment()  결함 목록 · 마스크
//                                          ├─ anomaly : Anomaly()  점수 · 히트맵 (PatchCore 포함)
//                                          ├─ detect  : Detect()   상자 목록
//                                          └─ classify: Classify() 클래스 · 신뢰도
//    영상마다 ─▶ 판정 · 시간(ms) 출력 ─▶ 결과 그림 저장 ─▶ eval.csv
//    폴더 안의 하위 폴더(예: ok/, ng/)는 묶음으로 보고, 묶음별 판정 수를 마지막에 보여 준다.
//    --roi 를 주면 영상에서 그 사각형만 잘라 모델에 넣는다 (결함 좌표와 결과 그림은 잘라 낸 영역 기준).
//
//    종료 코드: 0 = 모든 영상 실행 완료, 1 = 일부 영상 오류(읽기 실패 · 크기 불일치 등), 2 = 사용법/모델 오류
// ═══════════════════════════════════════════════════════
#include "vision_inference.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#define NOMINMAX             // windows.h 의 min/max 매크로가 std::min/std::max 를 깨지 않게
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

// ── 기본값: 인자 없이 실행할 때 쓴다. 여기 두 경로만 고치면 된다 ──
const char* const kDefaultModel = "model.json";    // 내보낸 모델 JSON (같은 폴더에 .onnx)
const char* const kDefaultImages = "images";       // 영상 파일 하나 또는 폴더
const char* const kDefaultOutput = "eval_result";  // 결과 그림과 eval.csv 를 두는 폴더

struct Options
{
    fs::path model = kDefaultModel, images = kDefaultImages, output = kDefaultOutput;
    int threads = 0;   // 0 = 자동(물리 코어 수)
    int repeat = 1;    // 영상마다 반복 횟수. 시간은 중앙값
    bool save = true;  // 결과 그림 저장
    cv::Rect roi;      // 비어 있으면 영상 전체. 판정 영역만 학습한 모델은 같은 영역을 준다
};

struct Row
{
    std::string image, group, verdict, detail;
    double ms = 0.0;
};

// SDK는 경로를 UTF-8로 받는다. 한글 폴더도 되도록 경로는 항상 이 함수로 문자열로 바꾼다.
std::string Utf8(const fs::path& path)
{
    const auto text = path.u8string();
    return std::string(text.begin(), text.end());
}

// cv::imread 는 Windows에서 한글 경로를 못 읽는다 → 파일을 직접 읽어 메모리에서 푼다
cv::Mat ReadImage(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary);
    const std::vector<uchar> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return bytes.empty() ? cv::Mat() : cv::imdecode(bytes, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
}

bool WritePng(const fs::path& path, const cv::Mat& image)
{
    std::vector<uchar> bytes;
    if (!cv::imencode(".png", image, bytes)) return false;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return file.good();
}

// 파일이면 그 하나, 폴더면 하위 폴더까지 영상 전부 (이름순)
std::vector<fs::path> ListImages(const fs::path& input)
{
    if (fs::is_regular_file(input)) return {input};
    std::vector<fs::path> images;
    if (!fs::is_directory(input)) return images;
    for (const auto& entry : fs::recursive_directory_iterator(input)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (ext == ".png" || ext == ".bmp" || ext == ".jpg" || ext == ".jpeg" || ext == ".tif" || ext == ".tiff")
            images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    return images;
}

// 결과 그림의 바탕: 어떤 입력(흑백 · 16비트 · 알파)이든 8비트 BGR로
cv::Mat ToBgr8(const cv::Mat& image)
{
    cv::Mat view = image;
    if (view.depth() != CV_8U) {
        double low = 0, high = 0;
        cv::minMaxLoc(view.reshape(1), &low, &high);
        view.convertTo(view, CV_8U, high > low ? 255.0 / (high - low) : 1.0, high > low ? -low * 255.0 / (high - low) : 0.0);
    }
    if (view.channels() == 1) cv::cvtColor(view, view, cv::COLOR_GRAY2BGR);
    else if (view.channels() == 4) cv::cvtColor(view, view, cv::COLOR_BGRA2BGR);
    return view.clone();
}

std::string ClassName(const InferenceConfig& config, int class_id)
{
    if (class_id >= 0 && class_id < static_cast<int>(config.class_names.size())) return config.class_names[class_id];
    return "class " + std::to_string(class_id);
}

std::string Fixed(double value, int digits)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.*f", digits, value);
    return buffer;
}

// 영상 한 장을 태스크에 맞게 추론한다. 판정 · 요약을 row 에, 저장할 그림을 picture 에 채운다.
// SDK가 잰 시간(전처리 + 모델 + 후처리)을 돌려준다.
double Infer(VisionInference& engine, const cv::Mat& image, bool draw, Row& row, cv::Mat& picture)
{
    const InferenceConfig& config = engine.GetConfig();
    const cv::Scalar red(0, 0, 255);
    if (config.task == "segment") {
        const SegmentResult result = engine.Segment(image);
        long long defect_pixels = 0;
        for (size_t c = 1; c < result.pixel_counts.size(); ++c) defect_pixels += result.pixel_counts[c];
        if (config.defect_rules.empty()) {
            // 판정 기준이 없는 모델: 결함 목록이 나오지 않으므로 결함 픽셀이 하나라도 있으면 NG로 본다
            row.verdict = defect_pixels > 0 ? "NG" : "OK";
            row.detail = "결함 픽셀 " + std::to_string(defect_pixels) + " (판정 기준 없음)";
        } else {
            row.verdict = result.defects.empty() ? "OK" : "NG";
            row.detail = "결함 " + std::to_string(result.defects.size()) + "개";
            for (const auto& defect : result.defects)
                row.detail += " | " + ClassName(config, defect.class_id) + " 면적 " + std::to_string(defect.area) +
                              " 점수 " + Fixed(defect.score, 2);
        }
        if (draw) {
            picture = ToBgr8(image);
            cv::Mat blended;
            cv::addWeighted(picture, 0.55, result.color_mask, 0.45, 0.0, blended);
            blended.copyTo(picture, result.mask > 0);              // 결함으로 본 픽셀만 색을 입힌다
            for (const auto& defect : result.defects) cv::rectangle(picture, defect.box, red, 2);
        }
        return result.inference_ms;
    }
    if (config.task == "anomaly") {
        const AnomalyResult result = engine.Anomaly(image);
        row.verdict = result.anomalous ? "NG" : "OK";
        row.detail = "점수 " + Fixed(result.score, 4) + " / 임계값 " + Fixed(result.threshold, 4);
        if (draw) {
            // 히트맵 색 범위를 임계값 기준으로 고정한다(0 ~ 임계값의 2배): 정상 영상의 잡음이 빨갛게 보이지 않는다
            double high = result.threshold > 0 ? 2.0 * result.threshold : 0.0;
            if (high <= 0) cv::minMaxLoc(result.anomaly_map, nullptr, &high);
            cv::Mat heat;
            result.anomaly_map.convertTo(heat, CV_8U, high > 0 ? 255.0 / high : 0.0);
            cv::applyColorMap(heat, heat, cv::COLORMAP_JET);
            cv::addWeighted(ToBgr8(image), 0.5, heat, 0.5, 0.0, picture);
        }
        return result.inference_ms;
    }
    if (config.task == "detect") {
        const DetectResult result = engine.Detect(image);
        row.verdict = result.detections.empty() ? "OK" : "NG";
        row.detail = "검출 " + std::to_string(result.detections.size()) + "개";
        if (draw) picture = ToBgr8(image);
        for (const auto& box : result.detections) {
            row.detail += " | " + ClassName(config, box.class_id) + " " + Fixed(box.confidence, 2);
            if (draw)
                cv::rectangle(picture, cv::Point(cvRound(box.x1), cvRound(box.y1)),
                              cv::Point(cvRound(box.x2), cvRound(box.y2)), red, 2);
        }
        return result.inference_ms;
    }
    const ClassifyResult result = engine.Classify(image);           // classify: 판정 = 클래스 이름, 그림 없음
    row.verdict = result.class_name.empty() ? ClassName(config, result.class_id) : result.class_name;
    row.detail = "신뢰도 " + Fixed(result.confidence, 4);
    return result.inference_ms;
}

std::string Csv(const std::string& text)
{
    std::string quoted = "\"";
    for (const char c : text) quoted += c == '"' ? std::string("\"\"") : std::string(1, c);
    return quoted + "\"";
}

bool ParseArgs(int argc, char** argv, Options& options)
{
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--threads" && has_value) options.threads = std::max(0, std::atoi(argv[++i]));
        else if (arg == "--repeat" && has_value) options.repeat = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--output" && has_value) options.output = argv[++i];
        else if (arg == "--roi" && has_value) {
            cv::Rect& roi = options.roi;
            std::istringstream text(argv[++i]);
            char comma = 0;
            if (!(text >> roi.x >> comma >> roi.y >> comma >> roi.width >> comma >> roi.height) || roi.empty() ||
                roi.x < 0 || roi.y < 0)
                return false;
        }
        else if (arg == "--no-save") options.save = false;
        else if (arg.rfind("--", 0) == 0) return false;
        else if (positional == 0) { options.model = arg; ++positional; }
        else if (positional == 1) { options.images = arg; ++positional; }
        else return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);  // 소스가 UTF-8(/utf-8)이라 콘솔도 UTF-8로 맞춘다 (한글이 깨지지 않게)
#endif
    Options options;
    if (!ParseArgs(argc, argv, options)) {
        std::cerr << "usage: image_eval [model.json] [image|folder] [--threads N] [--repeat N] [--roi x,y,w,h] [--output dir] [--no-save]\n";
        return 2;
    }

    VisionInference engine;
    if (!engine.InitializeFromJson(Utf8(options.model), "", options.threads)) {
        std::cerr << "모델 로드 실패: " << Utf8(options.model)
                  << " (JSON 옆에 .onnx가 있는지, onnxruntime.dll 버전이 맞는지 확인)\n";
        return 2;
    }
    const std::vector<fs::path> images = ListImages(options.images);
    if (images.empty()) {
        std::cerr << "영상이 없습니다: " << Utf8(options.images) << "\n";
        return 2;
    }
    const bool is_folder = fs::is_directory(options.images);
    const bool draw = options.save && engine.GetConfig().task != "classify";
    if (options.save) fs::create_directories(options.output);

    std::printf("모델 %s · 태스크 %s · 스레드 %s · 반복 %d · 영상 %zu장\n", Utf8(options.model.filename()).c_str(),
                engine.GetConfig().task.c_str(), options.threads ? std::to_string(options.threads).c_str() : "자동",
                options.repeat, images.size());
    std::printf("%9s  %-6s  %s\n", "ms", "판정", "영상 · 요약");

    std::vector<Row> rows;
    bool warmed = false;
    for (const fs::path& path : images) {
        Row row;
        const fs::path relative = is_folder ? fs::relative(path, options.images) : path.filename();
        row.image = Utf8(relative);
        // 묶음 = 첫 하위 폴더 이름 (ok/a.png → ok). 폴더 바로 아래 영상은 묶음 없음
        row.group = relative.has_parent_path() ? Utf8(*relative.begin()) : "";
        cv::Mat image = ReadImage(path);
        cv::Mat picture;
        if (image.empty()) {
            row.verdict = "오류";
            row.detail = "영상을 읽지 못했습니다";
        } else if (!options.roi.empty() && (options.roi & cv::Rect(0, 0, image.cols, image.rows)) != options.roi) {
            row.verdict = "오류";
            row.detail = "--roi 가 영상(" + std::to_string(image.cols) + "x" + std::to_string(image.rows) + ") 밖으로 나갑니다";
        } else {
            if (!options.roi.empty()) image = image(options.roi);   // 복사 없이 그 영역만 본다
            try {
                if (!warmed) {                     // 예열: 첫 실행의 메모리 할당 · 스레드 준비를 시간에서 뺀다
                    Row unused;
                    cv::Mat none;
                    Infer(engine, image, false, unused, none);
                    warmed = true;
                }
                std::vector<double> times;
                for (int run = 0; run < options.repeat; ++run) {
                    Row current = row;
                    times.push_back(Infer(engine, image, draw && run + 1 == options.repeat, current, picture));
                    if (run + 1 == options.repeat) row = current;
                }
                std::sort(times.begin(), times.end());
                row.ms = times[times.size() / 2];
            } catch (const std::exception& error) {
                row.verdict = "오류";
                row.detail = error.what();
            }
        }
        std::printf("%9.1f  %-6s  %s · %s\n", row.ms, row.verdict.c_str(), row.image.c_str(), row.detail.c_str());
        if (!picture.empty()) {
            // 하위 폴더 구조를 이름에 남긴다: ok/a.png → ok_a.png
            std::string name = row.image;
            std::replace(name.begin(), name.end(), '/', '_');
            std::replace(name.begin(), name.end(), '\\', '_');
            if (!WritePng(options.output / fs::u8path(name).replace_extension(".png"), picture))
                std::fprintf(stderr, "결과 그림 저장 실패: %s\n", name.c_str());
        }
        rows.push_back(row);
    }

    // ── 요약: 시간 통계 + 묶음별 판정 수 ──
    std::vector<double> times;
    std::map<std::string, std::map<std::string, int>> groups;
    int errors = 0;
    for (const Row& row : rows) {
        ++groups[row.group][row.verdict];
        if (row.verdict != "오류") times.push_back(row.ms);
        else ++errors;
    }
    std::sort(times.begin(), times.end());
    if (!times.empty()) {
        double sum = 0;
        for (const double value : times) sum += value;
        std::printf("\n시간(ms): 평균 %.1f · 중앙값 %.1f · 최대 %.1f  (전처리 + 모델 + 후처리, 영상 읽기 제외)\n",
                    sum / double(times.size()), times[times.size() / 2], times.back());
    }
    std::printf("판정:\n");
    for (const auto& group : groups) {
        std::string line;
        for (const auto& verdict : group.second) line += "  " + verdict.first + " " + std::to_string(verdict.second);
        std::printf("  %-12s%s\n", group.first.empty() ? "(전체)" : group.first.c_str(), line.c_str());
    }

    if (options.save) {
        std::ofstream csv(options.output / "eval.csv", std::ios::binary);
        csv << "\xEF\xBB\xBF" << "image,group,verdict,ms,detail\n";   // BOM: 엑셀이 UTF-8 한글을 바로 읽는다
        for (const Row& row : rows)
            csv << Csv(row.image) << ',' << Csv(row.group) << ',' << Csv(row.verdict) << ',' << Fixed(row.ms, 2) << ','
                << Csv(row.detail) << "\n";
        std::printf("결과 저장: %s\n", Utf8(options.output).c_str());
    }
    return errors ? 1 : 0;
}
