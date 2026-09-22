"""평가 수식과 입력 계약 회귀 테스트. 모델/네트워크 없이 실행 가능."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "gui/core/metrics.py"
_SPEC = importlib.util.spec_from_file_location("reviewed_metrics", _PATH)
metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(metrics)


class AnomalyMetricTests(unittest.TestCase):
    def test_tied_scores_do_not_depend_on_order(self):
        for labels in ([1, 0], [0, 1]):
            meter = metrics.AnomalyMetrics()
            meter.update(np.array([0.5, 0.5]), np.array(labels))
            result = meter.compute()
            self.assertEqual(result["auroc"], 0.5)
            self.assertEqual(result["roc_curve"], {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})

    def test_auc_matches_pairwise_probability_with_mixed_ties(self):
        scores = np.array([1, 1, 2, 4, 4, 4, 7], dtype=float)
        labels = np.array([0, 1, 1, 0, 1, 0, 1])
        pos, neg = scores[labels == 1], scores[labels == 0]
        expected = np.mean((pos[:, None] > neg) + 0.5 * (pos[:, None] == neg))
        self.assertAlmostEqual(metrics.AnomalyMetrics()._compute_auroc(scores, labels), expected)

    def test_rare_defect_threshold_can_exceed_95th_percentile(self):
        meter = metrics.AnomalyMetrics()
        labels = np.zeros(100, dtype=int)
        labels[-1] = 1
        meter.update(np.arange(100), labels)
        result = meter.compute()
        self.assertEqual(result["optimal_threshold"], 99)
        self.assertEqual(result["f1"], 1)
        np.testing.assert_array_equal(result["confusion_matrix"], [[99, 0], [0, 1]])

    def test_empty_or_single_class_is_unavailable_not_chance_quality(self):
        for scores, labels in (([], []), ([0.1, 0.2], [0, 0]), ([1], [1])):
            meter = metrics.AnomalyMetrics()
            meter.update(np.array(scores), np.array(labels))
            result = meter.compute()
            self.assertFalse(result["evaluable"])
            self.assertIsNone(result["auroc"])
            self.assertIsNone(result["optimal_threshold"])
            self.assertIsNone(result["f1"])

    def test_image_labels_and_mismatched_lengths_are_rejected(self):
        for labels in (np.zeros((2, 3, 4, 4)), np.zeros(3)):
            with self.assertRaises(ValueError):
                metrics.AnomalyMetrics().update(np.array([0.1, 0.2]), labels)

    def test_nonfinite_score_or_nonbinary_label_is_rejected(self):
        for scores, labels in (([np.nan], [0]), ([np.inf], [1]), ([0.2], [2])):
            with self.assertRaises(ValueError):
                metrics.AnomalyMetrics().update(np.array(scores), np.array(labels))


class OtherMetricTests(unittest.TestCase):
    def test_tiny_identical_boxes_keep_scale_invariant_iou(self):
        for method in ("voc11", "interp101"):
            with self.subTest(method=method):
                meter = metrics.DetectionMetrics(1, ap_method=method)
                box = [.1, .1, .1 + 1 / 2048, .1 + 1 / 2048]
                self.assertEqual(meter._compute_iou(box, box), 1.0)
                target = {"class_id": 0, "bbox": box}
                meter.update([{**target, "confidence": .9}], [target])
                result = meter.compute()
                self.assertEqual(result["recall"], 1.0)
                self.assertGreater(result["mAP_50_95"], .99)
        self.assertEqual(meter._compute_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    def test_detection_tp_count_does_not_truncate_float(self):
        meter = metrics.DetectionMetrics(1)
        for i in range(22):
            ground_truth = {"class_id": 0, "bbox": [0, 0, 10, 10]}
            prediction = {**ground_truth, "confidence": 0.9}
            meter.update([prediction] if i < 15 else [], [ground_truth])
        result = meter.compute()
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 15 / 22)

    def test_absent_detection_class_does_not_halve_perfect_ap(self):
        meter = metrics.DetectionMetrics(2)
        gt = {"class_id": 0, "bbox": [0, 0, 10, 10]}
        meter.update([{**gt, "confidence": 0.9}], [gt])
        self.assertEqual(meter.compute()["mAP_50"], 1)

    def test_segmentation_ignores_255_and_absent_classes(self):
        meter = metrics.SegmentationMetrics(3)
        meter.update(np.array([[0, 1], [0, 123]]), np.array([[0, 1], [0, 255]]))
        result = meter.compute()
        self.assertEqual(result["mIoU"], 1)
        self.assertEqual(result["pixel_accuracy"], 1)
        self.assertEqual(meter.total_pixels, 3)

    def test_segmentation_rejects_shape_broadcast_and_invalid_class(self):
        for prediction, target in ((np.zeros((2, 1)), np.zeros((2, 2))),
                                   (np.array([[2]]), np.array([[0]]))):
            with self.assertRaises(ValueError):
                metrics.SegmentationMetrics(2).update(prediction, target)

    def test_classification_rejects_length_truncation_and_invalid_labels(self):
        for prediction, target in (([0], [0, 1]), ([2], [0])):
            with self.assertRaises(ValueError):
                metrics.ClassificationMetrics(2).update(np.array(prediction), np.array(target))

    def test_classification_excludes_zero_support_class_from_macro_metrics(self):
        meter = metrics.ClassificationMetrics(3, ["a", "b", "empty"])
        meter.update(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]))
        result = meter.compute()
        self.assertEqual(result["accuracy"], .75)
        self.assertAlmostEqual(result["precision_macro"], (2 / 3 + 1) / 2)
        self.assertEqual(result["recall_macro"], .75)
        self.assertAlmostEqual(result["f1_macro"], (.8 + 2 / 3) / 2)
        self.assertEqual(result["per_class"][2], {
            "name": "empty", "precision": None, "recall": None, "f1": None, "support": 0,
        })

    def test_segmentation_excludes_class_without_ground_truth_even_if_predicted(self):
        meter = metrics.SegmentationMetrics(3, ["a", "b", "empty"])
        meter.update(np.array([[0, 2]]), np.array([[0, 1]]))
        result = meter.compute()
        self.assertEqual(result["pixel_accuracy"], .5)
        self.assertEqual(result["mIoU"], .5)
        self.assertEqual(result["dice_score"], .5)
        self.assertIsNone(result["per_class"][2]["iou"])
        self.assertIsNone(result["per_class"][2]["dice"])

    def test_detection_reports_absent_class_as_unavailable(self):
        meter = metrics.DetectionMetrics(2, ["object", "empty"])
        gt = {"class_id": 0, "bbox": [0, 0, 1, 1]}
        meter.update([{**gt, "confidence": .9}], [gt])
        result = meter.compute()
        self.assertEqual(result["mAP_50"], 1)
        self.assertIsNone(result["per_class"][1]["ap_50"])
        self.assertIsNone(result["per_class"][1]["ap_50_95"])


if __name__ == "__main__":
    unittest.main()
