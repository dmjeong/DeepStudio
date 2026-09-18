"""CSV의 최종 Best 행과 실제 가중치의 이름/내용 일치 검증."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
from core.training_artifacts import publish_best


class TrainingArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"actual-best-model")

    def publish(self, **overrides):
        kwargs = dict(epoch=2, metric="accuracy", value=.8123456789, direction="max",
                      engine="custom", task="classify", policy="strict_improvement_first_tie")
        kwargs.update(overrides)
        return publish_best(self.checkpoint, self.root,
                            {"epoch": [1, 2, 3], "train_loss": [.8, .5, .1], "val_loss": [.7, .6, .2]}, **kwargs)

    def rows(self, name):
        with (self.root / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_named_copy_and_csv_use_saved_epoch_not_minimum_loss(self):
        named, details = self.publish()
        self.assertEqual(Path(named).name, "best_accuracy_0.812346_epoch_2.pt")
        self.assertEqual(Path(named).read_bytes(), self.checkpoint.read_bytes())
        rows = self.rows("results.csv")
        self.assertEqual([row["is_best"] for row in rows], ["0", "1", "0"])
        best = self.rows("best_result.csv")[0]
        self.assertEqual((best["train_loss"], best["val_loss"]), ("0.5", "0.6"))
        self.assertEqual(float(best["selection_value"]), .8123456789)
        self.assertEqual(details["value"], .8123456789)
        with (self.root / "best_selection.json").open(encoding="utf-8-sig") as stream:
            self.assertEqual(json.load(stream), details)


    def test_zero_score_and_unavailable_patchcore_are_distinct(self):
        zero, _ = self.publish(value=0)
        unavailable, _ = self.publish(metric="unavailable", value=None, direction="single_fit", engine="patchcore")
        self.assertIn("0.000000", zero)
        self.assertIn("unavailable_NA", unavailable)
        self.assertEqual(self.rows("best_result.csv")[0]["selection_value"], "")


    def test_missing_checkpoint_nonfinite_and_unknown_epoch_are_not_best(self):
        for kwargs in ({"value": float("nan")}, {"epoch": 0}):
            with self.assertRaises(ValueError):
                self.publish(**kwargs)
        self.checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            self.publish()
        self.assertFalse((self.root / "best_result.csv").exists())

    def test_failed_copy_preserves_canonical_and_existing_named_checkpoint(self):
        named, _ = self.publish()
        before = Path(named).read_bytes()
        with patch("core.training_artifacts.shutil.copyfile", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.publish()
        self.assertEqual(Path(named).read_bytes(), before)
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertFalse(list(self.root.glob(".best-*")))


if __name__ == "__main__":
    unittest.main()
