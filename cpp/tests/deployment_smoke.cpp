#include "vision_inference.h"
#include "sam2_inference.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv)
{
    try
    {
        require(argc == 2, "Fixture directory required.");
        const fs::path root = fs::u8path(argv[1]);
        VisionInference engine;
        require(engine.InitializeFromJson((root / "classify.json").u8string()), "Classification load failed.");
        cv::Mat gray = (cv::Mat_<uchar>(2, 3) << 0, 127, 255, 0, 127, 255);
        const double mean = ((0.0 + 127.0 + 255.0) / (3 * 255) - 0.449) / 0.226;
        const double probability = 1.0 / (1.0 + std::exp(-2 * mean));
        auto check_classification = [&](const cv::Mat& image) {
            const auto result = engine.Classify(image);
            require(std::abs(result.probabilities[0] - probability) < 1e-5, "Channel/normalization mismatch.");
            require(result.class_name == "OK \"quoted\"", "JSON escaped class name lost.");
        };
        check_classification(gray);
        require(engine.InitializeFromJson((root / "builtin.json").u8string()), "Built-in backend load failed.");
        check_classification(gray);
        require(engine.InitializeFromJson((root / "classify.json").u8string()), "Classification reload failed.");
        cv::Mat bgr, bgra;
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        cv::cvtColor(gray, bgra, cv::COLOR_GRAY2BGRA);
        check_classification(bgr);
        check_classification(bgra);
        // Alternating colors/sizes/strides must not reuse stale pixels or mutate caller/result buffers.
        const auto owned = engine.Classify(gray);
        const auto original_bgr = bgr.clone();
        engine.Classify(cv::Mat(5, 7, CV_8UC3, cv::Scalar(0, 0, 0)));
        check_classification(bgr);
        check_classification(gray);
        cv::Mat padded(4, 7, CV_8UC1, cv::Scalar(0));
        gray.copyTo(padded(cv::Rect(2, 1, 3, 2)));
        check_classification(padded(cv::Rect(2, 1, 3, 2)));
        require(cv::norm(bgr, original_bgr, cv::NORM_INF) == 0, "Preprocessing changed caller pixels.");
        require(owned.probabilities == engine.Classify(gray).probabilities, "ONNX result buffer was reused.");
        auto efficientnet_config = engine.GetConfig();
        efficientnet_config.backend = "efficientnet";
        require(engine.Initialize(efficientnet_config), "EfficientNet classification contract rejected.");
        check_classification(gray);
        cv::imwrite((root / "gray.png").string(), gray);
        require(std::abs(engine.ClassifyFile((root / "gray.png").string()).probabilities[0] - probability) < 1e-5,
                "Grayscale file inference mismatch.");

        const auto timed = engine.Classify(gray);
        require(timed.preprocess_ms >= 0 && timed.model_ms >= 0 && timed.postprocess_ms >= 0,
                "Negative classification timing.");
        require(std::abs(timed.inference_ms - timed.preprocess_ms - timed.model_ms - timed.postprocess_ms) < 1e-6,
                "Classification stage timings do not sum to total.");

        auto invalid = engine.GetConfig();
        invalid.model_path = (root / "missing.onnx").u8string();
        invalid.input_channels = 3;
        invalid.normalize_mean = {0, 0, 0};
        invalid.normalize_std = {1, 1, 1};
        require(!engine.Initialize(invalid), "Invalid reinitialization accepted.");
        require(engine.IsReady() && engine.GetConfig().input_channels == 1, "Previous valid state not retained.");
        check_classification(gray);

        nlohmann::json config;
        std::ifstream input_config(root / "classify.json");
        input_config >> config;
        input_config.close();
        config["model_path"] = fs::absolute(root / "classify.onnx").u8string();
        std::ofstream(root / "absolute.json") << config;
        require(engine.InitializeFromJson((root / "absolute.json").u8string()), "Absolute model path rejected.");
        auto incompatible = config;
        incompatible["schema_version"] = 4;
        std::ofstream(root / "incompatible.json") << incompatible;
        require(!engine.InitializeFromJson((root / "incompatible.json").u8string()), "Legacy preprocessing accepted.");
        incompatible["schema_version"] = 5;
        incompatible["preprocessing"]["resize_implementation"] = "pillow_u8_bilinear";
        std::ofstream(root / "incompatible.json") << incompatible;
        require(!engine.InitializeFromJson((root / "incompatible.json").u8string()), "Pillow preprocessing accepted.");
        incompatible["preprocessing"]["resize_implementation"] = "opencv_linear_exact_v1";
        incompatible["preprocessing"]["antialias"] = true;
        std::ofstream(root / "incompatible.json") << incompatible;
        require(!engine.InitializeFromJson((root / "incompatible.json").u8string()), "Incompatible antialias flag accepted.");
        incompatible["preprocessing"]["antialias"] = false;
        std::ofstream(root / "compatible.json") << incompatible;
        require(engine.InitializeFromJson((root / "compatible.json").u8string()), "OpenCV contract rejected.");
        check_classification(gray);
        config["num_classes"] = "two";
        std::ofstream(root / "invalid.json") << config;
        require(!engine.InitializeFromJson((root / "invalid.json").u8string()), "Invalid JSON type accepted.");
        check_classification(gray);

        auto tuned = incompatible;
        tuned["onnxruntime"] = {{"allow_spinning", false}, {"dynamic_block_base", 4}};
        std::ofstream(root / "tuned.json") << tuned;
        require(engine.InitializeFromJson((root / "tuned.json").u8string(), "onnxruntime", 2), "Thread tuning rejected.");
        require(engine.GetConfig().ort_allow_spinning == 0 && engine.GetConfig().ort_dynamic_block_base == 4,
                "Thread tuning not applied.");
        check_classification(gray);
        for (const auto& bad : {nlohmann::json{{"allow_spinning", 1}},
                               nlohmann::json{{"dynamic_block_base", -1}},
                               nlohmann::json{{"dynamic_block_base", 1.5}},
                               nlohmann::json{{"dynamic_block_base", 4294967296ULL}}}) {
            tuned["onnxruntime"] = bad;
            std::ofstream(root / "tuned.json") << tuned;
            require(!engine.InitializeFromJson((root / "tuned.json").u8string()), "Invalid tuning accepted.");
            check_classification(gray);
        }

        auto gray_contract = incompatible;
        gray_contract["backend"] = "efficientnet";
        gray_contract["model_config"] = {{"implementation_version", 2}, {"in_channels", 1},
                                          {"stem_in_channels", 1}, {"input_adapter", "native"}};
        std::ofstream(root / "gray_contract.json") << gray_contract;
        require(engine.InitializeFromJson((root / "gray_contract.json").u8string()), "Native gray contract rejected.");
        require(engine.GetConfig().stem_in_channels == 1, "Native gray stem metadata lost.");
        check_classification(gray);
        gray_contract["model_config"]["stem_in_channels"] = 3;
        std::ofstream(root / "gray_contract.json") << gray_contract;
        require(!engine.InitializeFromJson((root / "gray_contract.json").u8string()), "Conflicting gray stem accepted.");
        gray_contract["model_config"]["stem_in_channels"] = 1;
        gray_contract["preprocessing"]["in_channels"] = 3;
        std::ofstream(root / "gray_contract.json") << gray_contract;
        require(!engine.InitializeFromJson((root / "gray_contract.json").u8string()), "Conflicting preprocessing channels accepted.");
        check_classification(gray);

        require(engine.InitializeFromJson((root / "segment.json").u8string()), "Segmentation load failed.");
        const auto segmented = engine.Segment(bgra);
        require(segmented.mask.rows == 2 && segmented.mask.cols == 3, "Segmentation shape mismatch.");
        require(segmented.mask.at<uchar>(0, 0) == 1 && segmented.mask.at<uchar>(0, 1) == 0 &&
                segmented.mask.at<uchar>(0, 2) == 0, "Wrong segmentation labels.");
        bool rejected = false;
        try { engine.Classify(gray); } catch (const std::logic_error&) { rejected = true; }
        require(rejected, "Classification accepted segmentation model.");
        auto cropped = engine.GetConfig();
        cropped.crop_width = 3;
        cropped.crop_height = 2;
        require(engine.Initialize(cropped), "Cropped segmentation load failed.");
        cv::Mat larger(7, 10, CV_8UC4, cv::Scalar(50,50,50,255));
        bgra.copyTo(larger(cv::Rect(3,2,3,2)));
        const auto roi_result = engine.Segment(larger);
        require(cv::countNonZero(roi_result.valid_mask) == 6, "Wrong inspected ROI area.");
        require(cv::countNonZero(roi_result.mask(cv::Rect(3,2,3,2)) != segmented.mask) == 0,
                "Cropped segmentation differs from explicit ROI.");
        require(roi_result.pixel_counts == segmented.pixel_counts, "Pixels outside ROI counted as background.");
        require(engine.InitializeFromJson((root / "detect.json").u8string()), "Detection load failed.");
        const auto detected = engine.Detect(gray);
        require(detected.detections.size() == 1, "Detection NMS/count mismatch.");
        require(detected.detections[0].class_id == 0 && detected.detections[0].confidence > 0.99f,
                "Detection score/class decode mismatch.");
        require(detected.detections[0].x1 > 0.0f && detected.detections[0].x2 < gray.cols,
                "Detection coordinate decode mismatch.");
        require(engine.InitializeFromJson((root / "redetr.json").u8string()), "Re-DETR v4 load failed.");
        const auto redetr = engine.Detect(gray);
        require(redetr.detections.size() == 1, "Re-DETR v4 detection count mismatch.");
        require(redetr.detections[0].class_id == 0 && redetr.detections[0].confidence > 0.99f,
                "Re-DETR v4 two-output decode mismatch.");
        require(engine.InitializeFromJson((root / "redetr_softmax.json").u8string()), "Re-DETR softmax load failed.");
        const auto redetr_softmax = engine.Detect(gray);
        require(redetr_softmax.detections.size() == 1 && redetr_softmax.detections[0].class_id == 0 &&
                    redetr_softmax.detections[0].confidence > 0.99f,
                "Re-DETR softmax decode mismatch.");
        std::ifstream sam_config_file(root / "sam2.json");
        auto invalid_sam_config = nlohmann::json::parse(sam_config_file);
        invalid_sam_config["contracts"]["graphs"]["decoder"]["inputs"].erase("image_features_1");
        const auto invalid_sam_path = root / "sam2_missing_decoder_input.json";
        std::ofstream invalid_sam_file(invalid_sam_path);
        invalid_sam_file << invalid_sam_config.dump();
        invalid_sam_file.close();
        Sam2Inference invalid_sam;
        require(!invalid_sam.InitializeFromJson(invalid_sam_path.u8string(), "onnxruntime", 2) &&
                    !invalid_sam.LastError().empty(),
                "SAM2 must reject an incomplete decoder input contract during initialization.");
        Sam2Inference sam2;
        require(sam2.InitializeFromJson((root / "sam2.json").u8string(), "onnxruntime", 2),
                "SAM2 encoder/decoder load failed.");
        const auto image_context = sam2.Encode(bgr);
        Sam2Prompt sam_prompt;
        sam_prompt.points.emplace_back(1.0f, 1.0f);
        sam_prompt.labels.push_back(1);
        const auto sam_result = sam2.Segment(image_context, sam_prompt);
        require(sam_result.mask.size() == cv::Size(2, 2) && sam_result.selected_mask == 1,
                "SAM2 selected mask shape/index mismatch.");
        require(cv::countNonZero(sam_result.mask) == 4 && sam_result.scores.size() == 2,
                "SAM2 prompt decode mismatch.");
        Sam2Prompt sam_box_prompt;
        sam_box_prompt.box_xyxy = {0.0f, 0.0f, 1.0f, 1.0f};
        sam_box_prompt.mask_input = cv::Mat::ones(2, 2, CV_32FC1);
        const auto sam_box_result = sam2.Segment(image_context, sam_box_prompt);
        require(sam_box_result.mask.size() == cv::Size(2, 2) && cv::countNonZero(sam_box_result.mask) == 4,
                "SAM2 box/mask prompt decode mismatch.");
        require(engine.InitializeFromJson((root / "anomaly.json").u8string()), "Anomaly load failed.");
        const auto anomaly = engine.Anomaly(gray);
        require(anomaly.anomaly_map.type() == CV_32FC1 && anomaly.anomaly_map.size() == gray.size(),
                "Anomaly map shape mismatch.");
        require(anomaly.score == 0.0f && !anomaly.anomalous, "Anomaly reconstruction score mismatch.");
        require(engine.InitializeFromJson((root / "patchcore.json").u8string()), "PatchCore load failed.");
        const auto patchcore = engine.Anomaly(gray);
        require(patchcore.anomaly_map.type() == CV_32FC1 && patchcore.score == 0.75f && patchcore.anomalous,
                "PatchCore score/map contract mismatch.");
        require(engine.InitializeFromJson((root / "classify.json").u8string()), "Classification reload failed.");
        auto classify_crop = engine.GetConfig();
        classify_crop.crop_width = 3;
        classify_crop.crop_height = 2;
        require(engine.Initialize(classify_crop), "Cropped classification load failed.");
        check_classification(larger);
        cv::Mat full_width_roi(6, 3, CV_8UC1, cv::Scalar(33));
        gray.copyTo(full_width_roi(cv::Rect(0, 2, 3, 2)));
        check_classification(full_width_roi);
        rejected = false;
        try { engine.Classify(cv::Mat(1,1,CV_8UC1)); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Oversized crop accepted.");
        check_classification(larger);
        require(!engine.InitializeFromJson((root / "unsupported.json").u8string()), "Unsupported model accepted.");
        check_classification(larger);
        require(engine.InitializeFromJson((root / "classify.json").u8string()), "Switch back to Custom failed.");
        check_classification(gray);
        require(engine.InitializeFromJson((root / "close_logits.json").u8string(), "onnxruntime", 1), "Close logits load failed.");
        const auto close = engine.Classify(gray);
        require(close.class_id == 1 && close.probabilities[0] == close.probabilities[1], "Softmax rounding changed rank.");
#ifdef VISION_WITH_OPENVINO
        require(engine.InitializeFromJson((root / "classify.json").u8string(), "openvino", 1), "OpenVINO load failed.");
        engine.PrintModelInfo();
        check_classification(gray);
        check_classification(bgr);
        check_classification(bgra);
        const auto saved = engine.Classify(gray);
        engine.Classify(cv::Mat(2, 3, CV_8UC1, cv::Scalar(0)));
        require(saved.probabilities[0] == engine.Classify(gray).probabilities[0], "OpenVINO result did not own its output.");
        auto broken = engine.GetConfig(); broken.output_name = "wrong";
        require(!engine.Initialize(broken), "OpenVINO accepted wrong output name.");
        check_classification(gray);
        broken = engine.GetConfig(); broken.num_classes = 3; broken.class_names.clear();
        require(!engine.Initialize(broken), "OpenVINO accepted wrong class count.");
        check_classification(gray);
        require(!engine.InitializeFromJson((root / "segment.json").u8string(), "openvino", 1), "OpenVINO accepted segmentation.");
        check_classification(gray);
        require(engine.InitializeFromJson((root / "classify.json").u8string(), "onnxruntime", 1), "Runtime switch back failed.");
#else
        require(!engine.InitializeFromJson((root / "classify.json").u8string(), "openvino", 1), "Missing OpenVINO silently fell back.");
#endif
        std::cout << "Deployment contract smoke passed." << std::endl;
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
