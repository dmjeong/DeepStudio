#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

// Rows = ground truth, columns = prediction. 255 means ignored ground truth.
class SegmentationMetrics {
public:
    explicit SegmentationMetrics(int classes) {
        if (classes < 1 || classes > 255) throw std::invalid_argument("Class count must be 1..255");
        counts_.assign(classes, std::vector<uint64_t>(classes));
    }
    void Add(const cv::Mat& truth, const cv::Mat& prediction, const cv::Mat& valid) {
        if (truth.type() != CV_8UC1 || prediction.type() != CV_8UC1 || valid.type() != CV_8UC1 ||
            truth.size() != prediction.size() || truth.size() != valid.size())
            throw std::invalid_argument("Masks must be original-size, single-channel uint8 class IDs");
        for (int y = 0; y < truth.rows; ++y) {
            const auto* gt = truth.ptr<uint8_t>(y);
            const auto* pred = prediction.ptr<uint8_t>(y);
            const auto* roi = valid.ptr<uint8_t>(y);
            for (int x = 0; x < truth.cols; ++x) {
                if (gt[x] != 255 && gt[x] >= counts_.size()) throw std::invalid_argument("Unknown ground-truth class ID");
                if (!roi[x] || gt[x] == 255) continue;
                if (pred[x] >= counts_.size()) throw std::invalid_argument("Unknown predicted class ID");
                ++counts_[gt[x]][pred[x]];
            }
        }
    }
    nlohmann::json Report(const std::vector<std::string>& names) const {
        using nlohmann::json;
        json rows = json::array(); uint64_t total = 0, correct = 0;
        double sum = 0, foreground = 0; int used = 0, fg_used = 0;
        for (size_t c = 0; c < counts_.size(); ++c) {
            uint64_t gt = 0, pred = 0, tp = counts_[c][c];
            for (size_t j = 0; j < counts_.size(); ++j) { gt += counts_[c][j]; pred += counts_[j][c]; }
            uint64_t uni = gt + pred - tp;
            json iou = nullptr, dice = nullptr;
            if (uni) {
                double value = double(tp) / double(uni);
                iou = value; dice = 2.0 * double(tp) / double(gt + pred);
                sum += value; ++used;
                if (c) { foreground += value; ++fg_used; }
            }
            rows.push_back({{"class_id", c}, {"name", c < names.size() ? names[c] : std::to_string(c)},
                {"gt_pixels", gt}, {"predicted_pixels", pred}, {"tp", tp}, {"fp", pred-tp}, {"fn", gt-tp},
                {"iou", iou}, {"dice", dice}});
            total += gt; correct += tp;
        }
        return {{"per_class", rows}, {"confusion_matrix", counts_}, {"evaluated_pixels", total},
            {"pixel_accuracy", total ? json(double(correct)/double(total)) : json(nullptr)},
            {"miou_including_background", used ? json(sum/used) : json(nullptr)},
            {"miou_foreground", fg_used ? json(foreground/fg_used) : json(nullptr)}};
    }
private:
    std::vector<std::vector<uint64_t>> counts_;
};
