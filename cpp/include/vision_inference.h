/**
 * @file vision_inference.h
 * @brief Vision ONNX Runtime CPU 추론 엔진
 *
 * 아키텍처 개요:
 * ┌────────────────────────────────────────────────────────────────┐
 * │                   Vision C++ 추론 시스템                       │
 * ├────────────────────────────────────────────────────────────────┤
 * │                                                                │
 * │  ┌───────────┐    ┌───────────────┐    ┌──────────────────┐   │
 * │  │ 이미지     │───►│ 전처리         │───►│ ONNX Runtime     │   │
 * │  │ (OpenCV)  │    │ (리사이즈+정규화)│    │ (CPU 추론)      │   │
 * │  └───────────┘    └───────────────┘    └────────┬─────────┘   │
 * │                                                  │             │
 * │                    ┌─────────────────────────────┼─────────┐   │
 * │                    │                             │         │   │
 * │              ┌─────▼─────┐              ┌───────▼───────┐ │   │
 * │              │ Classify  │              │ Segmentation  │ │   │
 * │              │ Result    │              │ Mask          │ │   │
 * │              │ (클래스명) │              │ (마스크맵)    │ │   │
 * │              └───────────┘              └───────────────┘ │   │
 * │                    │                             │         │   │
 * │                    └─────────────────────────────┘         │   │
 * │                              결과 반환                     │   │
 * └────────────────────────────────────────────────────────────────┘
 *
 * 의존성:
 * - ONNX Runtime (CPU): https://github.com/microsoft/onnxruntime
 * - OpenCV: 이미지 I/O 및 전처리
 * - nlohmann/json: 설정 파일 파싱
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>

// ONNX Runtime 헤더
#include <onnxruntime_cxx_api.h>

// OpenCV 헤더
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


// ═══════════════════════════════════════════════════
//  추론 설정 구조체
// ═══════════════════════════════════════════════════

/**
 * @brief ONNX 모델 추론에 필요한 설정값
 *
 * JSON 설정 파일(inference_config.json)에서 로드하거나
 * 코드에서 직접 생성 가능
 */
struct InferenceConfig
{
    // ── 모델 설정 ──
    std::string model_path;       ///< ONNX 모델 파일 경로
    std::string task;             ///< "classify" 또는 "segment"
    int num_classes = 10;         ///< 클래스 수

    // ── 입력 설정 ──
    int input_channels = 1;      ///< 입력 채널 (1: 그레이, 3: RGB)
    int input_height = 224;       ///< 입력 높이
    int input_width = 224;        ///< 입력 너비
    int crop_width = 0;           ///< Original-pixel center crop, both zero disables
    int crop_height = 0;

    // ── ONNX 텐서 이름 ──
    std::string input_name = "input_image";     ///< 입력 텐서 이름
    std::string output_name = "class_logits";   ///< 출력 텐서 이름
    std::vector<std::string> output_names;       ///< multi-output contracts (PatchCore, Re-DETR v4)

    // ── 정규화 파라미터 (학습 시 사용한 값과 동일해야 함!) ──
    std::vector<float> normalize_mean = {0.449f};
    std::vector<float> normalize_std  = {0.226f};

    // ── 클래스 이름 매핑 ──
    std::vector<std::string> class_names;

    // ── 추론 옵션 ──
    int num_threads = 0;          ///< CPU 스레드 수 (0: 자동)
    std::string runtime = "onnxruntime"; ///< onnxruntime or optional openvino (classification)
    std::string ort_graph_optimization_level = "all"; ///< all, basic, disabled; loaded from deployment JSON
    int ort_allow_spinning = -1;  ///< -1: SDK default, 0: sleep, 1: spin
    int ort_dynamic_block_base = 0; ///< 0: SDK default; e.g. 4: dynamic work partitioning
    bool enable_profiling = false; ///< 프로파일링 활성화
    std::string backend = "custom"; ///< 배포 모델 구현
    int model_implementation_version = 0; ///< 0: 구형 JSON 또는 직접 설정으로 내부 구조 미상
    int stem_in_channels = 0;
    std::string input_adapter = "unspecified";
    std::string resize_mode = "stretch";
    std::vector<int> resize_size; ///< Native classification Resize argument: [short] or [H,W]
    std::string classification_output = "logits"; ///< logits or probabilities
    std::string detection_box_encoding = "normalized_cxcywh";
    std::string detection_objectness = "sigmoid";
    std::string detection_class_scores = "sigmoid";
    float detection_confidence_threshold = 0.25f;
    float detection_iou_threshold = 0.5f;
    int detection_max_detections = 300;
    float anomaly_threshold = 0.0f;
};


// ═══════════════════════════════════════════════════
//  추론 결과 구조체
// ═══════════════════════════════════════════════════

/**
 * @brief Classification 추론 결과
 */
struct ClassifyResult
{
    int class_id = -1;             ///< 예측 클래스 인덱스
    std::string class_name;        ///< 예측 클래스 이름
    float confidence = 0.0f;       ///< 신뢰도 (softmax 확률)
    std::vector<float> probabilities; ///< 전체 클래스 확률 분포
    double inference_ms = 0.0;     ///< 추론 소요 시간 (밀리초)
    double preprocess_ms = 0.0;
    double model_ms = 0.0;         ///< CPU runtime inference only
    double postprocess_ms = 0.0;
};

/**
 * @brief Segmentation 추론 결과
 */
struct SegmentResult
{
    cv::Mat valid_mask;            ///< Inspected pixels (255), outside ROI (0)
    cv::Mat mask;                  ///< 세그멘테이션 마스크 (H×W, CV_8UC1)
    cv::Mat color_mask;            ///< 컬러 마스크 (시각화용, H×W, CV_8UC3)
    int num_classes = 0;           ///< 클래스 수
    std::vector<int> pixel_counts; ///< 클래스별 픽셀 수
    double inference_ms = 0.0;     ///< 추론 소요 시간
};

struct Detection
{
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    int class_id = -1;
    float confidence = 0.0f;
};

struct DetectResult
{
    std::vector<Detection> detections;
    double inference_ms = 0.0;
    double preprocess_ms = 0.0;
    double model_ms = 0.0;
    double postprocess_ms = 0.0;
};

struct AnomalyResult
{
    cv::Mat anomaly_map;            ///< 원본 이미지 크기의 CV_32FC1 map
    float score = 0.0f;
    float threshold = 0.0f;
    bool anomalous = false;
    double inference_ms = 0.0;
    double preprocess_ms = 0.0;
    double model_ms = 0.0;
    double postprocess_ms = 0.0;
};


// ═══════════════════════════════════════════════════
//  Vision 추론 엔진
// ═══════════════════════════════════════════════════

/**
 * @brief Vision ONNX Runtime 기반 추론 클래스
 *
 * 사용법:
 * @code
 *   // 1. Load exported EfficientNet or Custom model metadata.
 *   VisionInference engine;
 *   if (!engine.InitializeFromJson("model.json")) { // 에러 처리 }
 *
 *   // 2. 추론 실행
 *   cv::Mat image = cv::imread("test.png", cv::IMREAD_ANYCOLOR | cv::IMREAD_IGNORE_ORIENTATION);
 *   ClassifyResult result = engine.Classify(image);
 *   std::cout << "클래스: " << result.class_name
 *             << " (" << result.confidence << ")" << std::endl;
 * @endcode
 */
class VisionInference
{
public:
    VisionInference();
    ~VisionInference();

    // ── 초기화 ──
    /**
     * @brief 모델 로드 및 ONNX Runtime 세션 초기화
     * @param config 추론 설정
     * @return 성공 여부
     */
    bool Initialize(const InferenceConfig& config);

    /**
     * @brief JSON 설정 파일에서 초기화
     * @param config_path inference_config.json 경로
     * @return 성공 여부
     */
    bool InitializeFromJson(const std::string& config_path,
                            const std::string& runtime = "", int num_threads = -1);

    /**
     * @brief 리소스 해제
     */
    void Release();

    /**
     * @brief 초기화 상태 확인
     */
    bool IsReady() const { return m_bInitialized; }
    const InferenceConfig& GetConfig() const { return m_config; }

    // ── 추론 ──
    /**
     * @brief Classification 추론
     * @param image 입력 이미지 (GRAY, BGR 또는 BGRA, OpenCV)
     * @return 분류 결과
     */
    // Uses reusable scratch/tensors: serialize calls on each instance, or use ClassificationWorker.
    ClassifyResult Classify(const cv::Mat& image);

    /**
     * @brief Segmentation 추론
     * @param image 입력 이미지 (GRAY, BGR 또는 BGRA, OpenCV)
     * @return 세그멘테이션 결과
     */
    SegmentResult Segment(const cv::Mat& image);
    DetectResult Detect(const cv::Mat& image);
    AnomalyResult Anomaly(const cv::Mat& image);

    /**
     * @brief 파일 경로로 Classification
     * @param image_path 이미지 파일 경로
     * @return 분류 결과
     */
    ClassifyResult ClassifyFile(const std::string& image_path);

    /**
     * @brief 파일 경로로 Segmentation
     * @param image_path 이미지 파일 경로
     * @return 세그멘테이션 결과
     */
    SegmentResult SegmentFile(const std::string& image_path);
    DetectResult DetectFile(const std::string& image_path);
    AnomalyResult AnomalyFile(const std::string& image_path);

    // ── 정보 조회 ──
    /**
     * @brief 모델 정보 출력
     */
    void PrintModelInfo() const;

private:
    // ── 전처리 ──
    /**
     * @brief OpenCV 이미지 → ONNX 입력 텐서 변환
     *
     * 전처리 순서:
     * 1. BGR → RGB 변환
     * 2. 리사이즈 (input_height × input_width)
     * 3. [0, 255] → [0, 1] 스케일링
     * 4. ImageNet 정규화 (mean, std)
     * 5. HWC → CHW 레이아웃 변환 (ONNX 표준)
     * 6. 배치 차원 추가 (BCHW)
     */
    std::vector<float> Preprocess(const cv::Mat& image);
    void PreprocessInto(const cv::Mat& image, float* data, size_t count,
                        cv::Mat& converted, cv::Mat& resized);

    /**
     * @brief Softmax 계산 (Classification 후처리)
     */
    std::vector<float> Softmax(const std::vector<float>& logits);

    /**
     * @brief Argmax 마스크 생성 (Segmentation 후처리)
     */
    cv::Mat ArgmaxMask(const float* output_data, int num_classes,
                       int height, int width);

    /**
     * @brief 클래스별 컬러 마스크 생성 (시각화)
     */
    cv::Mat ColorizeMask(const cv::Mat& mask, int num_classes);
    static float Sigmoid(float value);
    static float IntersectionOverUnion(const Detection& left, const Detection& right);

    // ── 멤버 변수 ──
    InferenceConfig m_config;            ///< 추론 설정
    bool m_bInitialized = false;         ///< 초기화 상태

    // ONNX Runtime 객체
    Ort::Env m_env;                      ///< ONNX Runtime 환경
    std::unique_ptr<Ort::Session> m_session;  ///< 추론 세션
    struct ClassificationState;
    std::unique_ptr<ClassificationState> m_classification;
    struct OpenVINOState;
    std::unique_ptr<OpenVINOState> m_openvino;
    Ort::AllocatorWithDefaultOptions m_allocator; ///< 메모리 할당기
};
