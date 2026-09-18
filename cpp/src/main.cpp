/**
 * @file main.cpp
 * @brief Vision C++ 추론 데모 프로그램
 *
 * 사용법:
 *   ./vision_demo --config inference_config.json --image test.jpg
 *   ./vision_demo --config inference_config.json --dir ./test_images/
 *
 * 빌드 (CMake):
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j4
 *
 * 실행 흐름:
 * ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
 * │ 인자    │───►│ 모델     │───►│ 이미지    │───►│ 결과     │
 * │ 파싱    │    │ 초기화   │    │ 추론      │    │ 출력     │
 * └─────────┘    └──────────┘    └──────────┘    └──────────┘
 */

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cctype>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "vision_inference.h"

namespace fs = std::filesystem;


// ═══════════════════════════════════════════════════
//  명령행 인자 파서
// ═══════════════════════════════════════════════════

struct AppArgs
{
    std::string config_path;      ///< 설정 파일 경로
    std::string image_path;       ///< 단일 이미지 경로
    std::string dir_path;         ///< 이미지 디렉토리 경로
    bool show_result = false;     ///< 결과 시각화 표시
    bool save_result = true;      ///< 결과 파일 저장
    std::string output_dir = "./results"; ///< 결과 저장 디렉토리
};

AppArgs ParseArgs(int argc, char* argv[])
{
    AppArgs args;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)
            args.config_path = argv[++i];
        else if (arg == "--image" && i + 1 < argc)
            args.image_path = argv[++i];
        else if (arg == "--dir" && i + 1 < argc)
            args.dir_path = argv[++i];
        else if (arg == "--show")
            args.show_result = true;
        else if (arg == "--no-save")
            args.save_result = false;
        else if (arg == "--output" && i + 1 < argc)
            args.output_dir = argv[++i];
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Vision 추론 데모\n"
                      << "사용법:\n"
                      << "  --config <path>   설정 파일 (inference_config.json)\n"
                      << "  --image <path>    단일 이미지 추론\n"
                      << "  --dir <path>      디렉토리 내 전체 이미지 추론\n"
                      << "  --show            결과 시각화 (GUI)\n"
                      << "  --no-save         결과 파일 저장 안 함\n"
                      << "  --output <path>   결과 저장 디렉토리\n"
                      << std::endl;
            exit(0);
        }
    }

    return args;
}


// ═══════════════════════════════════════════════════
//  Classification 결과 출력
// ═══════════════════════════════════════════════════

void PrintClassifyResult(const ClassifyResult& result,
                         const std::string& filename = "")
{
    std::cout << "┌──────────────────────────────────────┐" << std::endl;
    if (!filename.empty())
        std::cout << "│ 파일: " << filename << std::endl;
    std::cout << "│ 결과: " << result.class_name
              << " (ID: " << result.class_id << ")" << std::endl;
    std::cout << "│ 신뢰도: " << (result.confidence * 100.0f)
              << "%" << std::endl;
    std::cout << "│ 추론 시간: " << result.inference_ms
              << " ms" << std::endl;

    // 상위 3개 클래스 확률
    std::cout << "│ 확률 분포:" << std::endl;
    std::vector<std::pair<int, float>> ranked;
    for (size_t i = 0; i < result.probabilities.size(); ++i)
    {
        ranked.push_back({static_cast<int>(i), result.probabilities[i]});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](auto& a, auto& b) { return a.second > b.second; });

    int top_k = std::min(3, static_cast<int>(ranked.size()));
    for (int i = 0; i < top_k; ++i)
    {
        std::cout << "│   [" << ranked[i].first << "] "
                  << (ranked[i].second * 100.0f) << "%" << std::endl;
    }
    std::cout << "└──────────────────────────────────────┘" << std::endl;
}


// ═══════════════════════════════════════════════════
//  Segmentation 결과 출력 & 시각화
// ═══════════════════════════════════════════════════

void PrintSegmentResult(const SegmentResult& result,
                        const std::string& filename = "")
{
    std::cout << "┌──────────────────────────────────────┐" << std::endl;
    if (!filename.empty())
        std::cout << "│ 파일: " << filename << std::endl;
    std::cout << "│ 클래스 수: " << result.num_classes << std::endl;
    std::cout << "│ 마스크 크기: " << result.mask.cols << "×"
              << result.mask.rows << std::endl;
    std::cout << "│ 추론 시간: " << result.inference_ms
              << " ms" << std::endl;

    // 클래스별 픽셀 비율
    int total_pixels = result.mask.rows * result.mask.cols;
    std::cout << "│ 클래스별 점유율:" << std::endl;
    for (size_t i = 0; i < result.pixel_counts.size(); ++i)
    {
        float ratio = 100.0f * result.pixel_counts[i] / total_pixels;
        if (ratio > 0.01f)  // 0.01% 이상만 표시
        {
            std::cout << "│   [" << i << "] "
                      << result.pixel_counts[i] << " px ("
                      << ratio << "%)" << std::endl;
        }
    }
    std::cout << "└──────────────────────────────────────┘" << std::endl;
}

cv::Mat OverlaySegmentation(const cv::Mat& image,
                             const cv::Mat& color_mask,
                             float alpha = 0.5f)
{
    /**
     * 원본 이미지에 세그멘테이션 마스크 오버레이
     *
     * overlay = α × color_mask + (1-α) × original
     */
    cv::Mat overlay;
    cv::addWeighted(image, 1.0 - alpha, color_mask, alpha, 0, overlay);
    return overlay;
}


// ═══════════════════════════════════════════════════
//  이미지 파일 수집
// ═══════════════════════════════════════════════════

std::vector<std::string> CollectImages(const std::string& dir_path)
{
    std::vector<std::string> images;

    // 지원 확장자
    static const std::vector<std::string> exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    };

    for (const auto& entry : fs::directory_iterator(dir_path))
    {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        // 소문자 변환
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        for (const auto& valid_ext : exts)
        {
            if (ext == valid_ext)
            {
                images.push_back(entry.path().string());
                break;
            }
        }
    }

    std::sort(images.begin(), images.end());
    return images;
}


// ═══════════════════════════════════════════════════
//  메인 함수
// ═══════════════════════════════════════════════════

int main(int argc, char* argv[])
{
    // ── 인자 파싱 ────────────────────────────────────
    AppArgs args = ParseArgs(argc, argv);

    if (args.config_path.empty())
    {
        std::cerr << "오류: --config 인자를 지정해주세요." << std::endl;
        std::cerr << "사용법: ./vision_demo --config inference_config.json "
                  << "--image test.jpg" << std::endl;
        return 1;
    }

    // ── 추론 엔진 초기화 ─────────────────────────────
    std::cout << "\n══════════════════════════════════════" << std::endl;
    std::cout << "   Vision C++ 추론 데모" << std::endl;
    std::cout << "══════════════════════════════════════\n" << std::endl;

    VisionInference engine;
    if (!engine.InitializeFromJson(args.config_path))
    {
        std::cerr << "모델 초기화 실패!" << std::endl;
        return 1;
    }

    // ── 결과 디렉토리 생성 ────────────────────────────
    if (args.save_result)
    {
        fs::create_directories(args.output_dir);
    }

    // ── 이미지 수집 ──────────────────────────────────
    std::vector<std::string> image_paths;
    if (!args.image_path.empty())
    {
        image_paths.push_back(args.image_path);
    }
    else if (!args.dir_path.empty())
    {
        image_paths = CollectImages(args.dir_path);
        std::cout << "발견된 이미지: " << image_paths.size() << "개\n"
                  << std::endl;
    }
    else
    {
        std::cerr << "오류: --image 또는 --dir 인자를 지정해주세요."
                  << std::endl;
        return 1;
    }

    double total_time = 0.0;
    size_t processed_count = 0;
    size_t failed_count = 0;
    for (const auto& path : image_paths)
    {
        try
        {
            cv::Mat image = cv::imread(path, cv::IMREAD_ANYCOLOR | cv::IMREAD_IGNORE_ORIENTATION);
            if (image.empty()) throw std::runtime_error("Image load failed: " + path);
            const std::string filename = fs::path(path).filename().string();
            cv::Mat result_image;
            if (engine.GetConfig().task == "segment")
            {
                auto result = engine.Segment(image);
                PrintSegmentResult(result, filename);
                total_time += result.inference_ms;
                cv::Mat display_image;
                if (image.channels() == 1) cv::cvtColor(image, display_image, cv::COLOR_GRAY2BGR);
                else display_image = image;
                result_image = OverlaySegmentation(display_image, result.color_mask);
                if (args.save_result)
                {
                    const auto mask_path = fs::path(args.output_dir) / (fs::path(path).stem().string() + "_mask.png");
                    if (!cv::imwrite(mask_path.string(), result.mask))
                        throw std::runtime_error("Mask save failed.");
                }
            }
            else
            {
                auto result = engine.Classify(image);
                PrintClassifyResult(result, filename);
                total_time += result.inference_ms;
                if (image.channels() == 1) cv::cvtColor(image, result_image, cv::COLOR_GRAY2BGR);
                else result_image = image.clone();
                const std::string text = result.class_name + " (" +
                    std::to_string(int(result.confidence * 100)) + "%)";
                cv::putText(result_image, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                            1.0, cv::Scalar(0, 255, 0), 2);
            }
            if (args.save_result && !cv::imwrite((fs::path(args.output_dir) / ("result_" + filename)).string(), result_image))
                throw std::runtime_error("Result save failed.");
            ++processed_count;
            if (args.show_result)
            {
                cv::imshow("Vision Result", result_image);
                if (cv::waitKey(0) == 27) break;
            }
        }
        catch (const std::exception& error)
        {
            ++failed_count;
            std::cerr << path << ": " << error.what() << std::endl;
        }
    }
    std::cout << "Processed: " << processed_count << ", failed: " << failed_count << std::endl;
    if (processed_count)
        std::cout << "Mean preprocessing + inference + postprocessing: "
                  << total_time / processed_count << " ms/image" << std::endl;
    return failed_count || processed_count == 0 ? 1 : 0;
}
