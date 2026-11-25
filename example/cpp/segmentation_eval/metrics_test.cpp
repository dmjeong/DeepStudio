#include "metrics.h"
#include <cmath>
#include <iostream>
void Require(bool value) { if (!value) throw std::runtime_error("Metrics regression"); }
int main() {
    try {
        cv::Mat truth = (cv::Mat_<uint8_t>(1,6) << 0,1,2,0,255,1);
        cv::Mat pred = (cv::Mat_<uint8_t>(1,6) << 3,1,1,0,2,0);
        cv::Mat valid = (cv::Mat_<uint8_t>(1,6) << 255,255,255,255,255,0);
        SegmentationMetrics m(5); m.Add(truth,pred,valid);
        auto r = m.Report({"background","crack","FM","other","absent"});
        Require(r["evaluated_pixels"] == 4);
        Require(r["per_class"][0]["iou"] == .5);
        Require(r["per_class"][1]["iou"] == .5);
        Require(r["per_class"][2]["iou"] == 0);
        Require(r["per_class"][3]["iou"] == 0); // false positive with no GT remains in the mean
        Require(r["per_class"][4]["iou"].is_null());
        Require(std::abs(r["miou_foreground"].get<double>() - 1.0/6) < 1e-12);
        Require(r["miou_including_background"] == .25);
        SegmentationMetrics empty(2);
        Require(empty.Report({})["pixel_accuracy"].is_null());
        std::cout << "metrics passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
