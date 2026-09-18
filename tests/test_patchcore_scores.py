"""Bounded scores, fixed model scale, legacy review and decision preservation."""
from copy import deepcopy
from dataclasses import asdict, replace
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]
from patchcore_scores import score_normalization, normalize_score, normalize_threshold, normalized_details, validate_normalization
from core.inference_types import make_anomaly_result
from core.inference_review import review_page


class PatchCoreScoreTests(unittest.TestCase):
    def test_saved_boundary_order_and_batch_independent_scale(self):
        spec = score_normalization(3)
        distances = [0, .01, 1, math.nextafter(3., 0.), 3., math.nextafter(3., 4.), 10, 1e308]
        scores = [normalize_score(d, spec) for d in distances]
        self.assertEqual(scores[0], 0)
        self.assertEqual(scores[4], .5)
        self.assertTrue(all(0 <= s < 1 for s in scores))
        self.assertEqual(scores, sorted(set(scores)))
        for distance, score in zip(distances, scores):
            self.assertEqual(score >= .5, distance >= 3)
        shuffled = [distances[i] for i in (7, 0, 4)]
        self.assertEqual([normalize_score(d, spec) for d in shuffled], [scores[i] for i in (7, 0, 4)])
        self.assertEqual(normalize_score(1e308, score_normalization(1e308)), .5)
        self.assertEqual(normalize_score(0, score_normalization(1e-300)), 0)

    def test_uncalibrated_zero_and_invalid_metadata(self):
        for threshold in (None, 0, -1):
            spec = score_normalization(threshold)
            self.assertEqual(spec["source"], "unit_distance_fallback")
            self.assertEqual(normalize_threshold(threshold, spec), None if threshold is None else 0)
        for bad in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError): normalize_score(bad, score_normalization(3))
        for scale in (0, -1, True, None, float("nan"), float("inf")):
            with self.assertRaises(ValueError): validate_normalization({"method": "distance_ratio_v1", "scale": scale})
        with self.assertRaises(ValueError): validate_normalization({"method": "per_image_minmax", "scale": 1})

    def test_legacy_cache_and_new_results_share_review_units_without_mutation(self):
        raw = [{**asdict(make_anomaly_result(f"{i}.png", score, 3)), "index": i,
                "heatmap": {"kind": "PatchCore"}} for i, score in enumerate((0, 1, 3, 30))]
        before = deepcopy(raw)
        page = review_page(raw, sort="score", descending=True, limit=2)
        self.assertTrue(page["score_normalized"])
        self.assertEqual(page["saved_thresholds"], [.5])
        self.assertEqual(page["counts"]["NG"], 2)
        self.assertEqual(page["results"][0]["details"]["raw_score"], 30)
        self.assertEqual(page["results"][0]["details"]["raw_threshold"], 3)
        self.assertEqual(review_page(raw, threshold=0)["counts"]["NG"], 4)
        self.assertEqual(review_page(raw, threshold=1)["counts"]["OK"], 4)
        self.assertEqual(review_page(raw, threshold=.5, decision="NG")["filtered_total"], 2)
        for threshold in (-.1, 1.1):
            with self.assertRaises(ValueError): review_page(raw, threshold=threshold)
        self.assertEqual(raw, before)
        spec = score_normalization(3)
        new = replace(make_anomaly_result("new.png", .5, .5), details=normalized_details(3, 3, spec))
        self.assertEqual(review_page([{**asdict(new), "index": 0}])["results"][0]["score"], .5)
        generic = {**asdict(make_anomaly_result("custom.png", 4, 3)), "index": 0}
        self.assertFalse(review_page([generic], threshold=5)["score_normalized"])
        self.assertEqual(review_page([generic], threshold=5)["results"][0]["score"], 4)


if __name__ == "__main__":
    unittest.main()
