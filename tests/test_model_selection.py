"""선정 지표와 freeze 범위의 실제 동작을 검증한다."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
from core.model_selection import selection_policy
from core.comparison_evaluation import evaluate_predictions
from core.benchmark import latency_summary, measure_inference
from core.inference_types import InferenceResult


class SelectionTests(unittest.TestCase):

    def test_task_selection_validates_supported_metrics_and_finite_scores(self):
        policy = selection_policy(SimpleNamespace(selection_metric="f1_macro"), "custom", "classify")
        self.assertEqual(policy.metric, "f1_macro")
        for value in [None, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                policy.value({"f1_macro": value})
        with self.assertRaises(ValueError):
            selection_policy(SimpleNamespace(selection_metric="accuracy"), "custom", "segment")


    def test_imbalanced_majority_predictions_do_not_have_perfect_macro_f1(self):
        validator = SimpleNamespace(names=["OK", "NG"], pred=[np.zeros((100, 1), dtype=int)],
                                    targets=[np.asarray([0] * 99 + [1])])
        records = [{"image_id": str(i), "prediction": 0, "target": int(target)} for i, target in enumerate(validator.targets[0])]
        metrics = evaluate_predictions("classify", ["OK", "NG"], records)["metrics"]
        self.assertEqual(metrics["accuracy"], .99)
        self.assertEqual(metrics["recall_macro"], .5)
        self.assertLess(metrics["f1_macro"], .5)


class ComparisonTests(unittest.TestCase):
    def test_both_engine_predictions_use_identical_classification_evaluator(self):
        records = [{"image_id": "a", "prediction": 0, "target": 0},
                   {"image_id": "b", "prediction": 0, "target": 1}]
        self.assertEqual(evaluate_predictions("classify", ["OK", "NG"], records)["metrics"]["accuracy"], .5)
        with self.assertRaises(ValueError):
            evaluate_predictions("classify", ["OK", "NG"], records + records)

    def test_overlapping_targets_are_matched_once_each(self):
        boxes = [{"class_id": 0, "bbox": [.1, .1, .5, .5]},
                 {"class_id": 0, "bbox": [.12, .12, .52, .52]}]
        predictions = [{**boxes[0], "confidence": confidence} for confidence in [.9, .8]]
        report = evaluate_predictions("detect", ["defect"],
                                      [{"image_id": "a", "target": boxes, "prediction": predictions}])
        self.assertEqual(report["metrics"]["recall"], 1.0)

    def test_detector_comparison_records_ap_algorithm_and_keeps_empty_predictions(self):
        box = {"class_id": 0, "bbox": [.1, .1, .3, .3]}
        records = [{"image_id": "a", "target": [box], "prediction": [{**box, "confidence": .9}]},
                   {"image_id": "b", "target": [box], "prediction": []}]
        report = evaluate_predictions("detect", ["defect"], records)
        self.assertEqual(report["metrics"]["recall"], .5)
        self.assertIn("101", report["ap_method"])
        with self.assertRaises(ValueError):
            evaluate_predictions("segment", ["defect"], records)


class BenchmarkTests(unittest.TestCase):
    def test_percentiles_and_failed_samples_are_not_fabricated(self):
        result = latency_summary([.001, .002, .003])
        self.assertEqual(result["p50_ms"], 2)
        self.assertAlmostEqual(result["p95_ms"], 2.9)
        engine = SimpleNamespace(infer=lambda path: InferenceResult(path, "error", "", "ERROR", error="broken"))
        with self.assertRaisesRegex(RuntimeError, "broken"):
            measure_inference(engine, ["a"], warmup=0, repeats=1)

    def test_warmup_excluded_and_gradcam_measured_separately(self):
        calls = []
        def infer(path):
            calls.append(path)
            return InferenceResult(path, "ok", "classify", "OK", inference_sec=.001,
                                   gradcam_sec=.005, inference_status="completed", gradcam_status="completed")
        prepared = []
        report = measure_inference(SimpleNamespace(infer=infer), ["a", "b"], warmup=2, repeats=3,
                                   prepare_measurement=lambda: prepared.append(len(calls)))
        self.assertEqual(prepared, [2])
        self.assertEqual(len(calls), 5)
        self.assertEqual(report["summary"]["inference"]["samples"], 3)
        self.assertEqual(report["summary"]["gradcam"]["mean_ms"], 5)
