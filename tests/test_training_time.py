"""에폭/전체 시간의 단조 시계, CSV, UI 로그 일치 검증."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
from core.training_time import TrainingClock, format_hms, record_epoch_time
from core.training_artifacts import publish_best
from tests import test_best_epoch_backend as best_contracts


class TrainingTimeTests(unittest.TestCase):
    def test_hms_has_no_day_wrap_and_preserves_unknown(self):
        for seconds, expected in ((0, "00:00:00"), (59.99, "00:00:59"),
                                  (3661.9, "01:01:01"), (90061, "25:01:01"),
                                  (None, "N/A"), (float("nan"), "N/A")):
            with self.subTest(seconds=seconds):
                self.assertEqual(format_hms(seconds), expected)

    def test_epoch_total_csv_and_log_share_exact_clock_values(self):
        now = [100.]
        clock = TrainingClock(lambda: now[0])
        logs, history = [], {"epoch": [1, 2], "accuracy": [.9, .8]}
        now[0] = 110.  # 모델과 데이터 준비 10초
        clock.start_epoch()
        now[0] = 175.25
        record_epoch_time(clock, history, logs.append, 1)
        now[0] = 177.
        clock.start_epoch()
        now[0] = 237.5
        record_epoch_time(clock, history, logs.append, 2)
        self.assertEqual(history["epoch_time_sec"], [65.25, 60.5])
        self.assertEqual(history["elapsed_time_sec"], [75.25, 137.5])
        self.assertIn("소요 00:01:05 | 누적 00:01:15", logs[0])
        now[0] = 245.75  # 최종 평가와 가중치 저장 완료
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"best")
            _, details = publish_best(checkpoint, root, history, epoch=1, metric="accuracy", value=.9,
                                      direction="max", engine="custom", task="classify", policy="strict", timing=clock)
            with (root / "results.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["epoch_time_hms"], "00:01:05")
            self.assertEqual(rows[1]["epoch_time_hms"], "00:01:00")
            self.assertEqual(rows[0]["run_total_time_hms"], "00:02:25")
            self.assertEqual(float(rows[1]["run_total_time_sec"]), 145.75)
            self.assertEqual(details["total_seconds"], 145.75)
            with (root / "training_summary.csv").open(encoding="utf-8-sig") as stream:
                summary = next(csv.DictReader(stream))
            self.assertEqual(summary["total_training_time_hms"], "00:02:25")
            self.assertEqual(summary["epochs_in_session"], "2")



if __name__ == "__main__":
    unittest.main()
