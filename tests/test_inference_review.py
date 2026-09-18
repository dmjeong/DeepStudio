"""Numeric sorting, dataset labels, threshold boundaries and stable Qt selection."""
from dataclasses import asdict, replace
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))
from core.inference_types import make_anomaly_result, InferenceResult
from core.inference_review import effective_result, review_page, source_class, restored_result


def records():
    return [{**asdict(replace(make_anomaly_result(f"/data/test/{'good' if i % 2 == 0 else 'defect'}/image-{i:03}.png", i / 10, 3),
                             source_class="good" if i % 2 == 0 else "defect", inference_sec=i / 1000)), "index": i} for i in range(75)]


class ReviewTests(unittest.TestCase):
    def test_desktop_restores_legacy_patchcore_job_as_normalized_result(self):
        stored = {**asdict(make_anomaly_result("legacy.png", 6, 3)), "index": 0,
                  "heatmap": {"kind": "PatchCore"}}
        result = restored_result(stored)
        self.assertEqual(result.score, 6 / 9)
        self.assertEqual(result.threshold, .5)
        self.assertEqual(result.details["raw_score"], 6)
        self.assertEqual(result.details["raw_threshold"], 3)
        self.assertEqual(stored["score"], 6)
        self.assertEqual(stored["threshold"], 3)

    def test_equal_threshold_is_ng_raw_result_and_timings_preserved(self):
        raw = replace(make_anomaly_result("image", 2.5, 3), inference_sec=.4, elapsed_sec=.7)
        adjusted = effective_result(raw, 2.5)
        self.assertTrue(adjusted.summary.startswith("NG"))
        self.assertEqual(adjusted.score, raw.score)
        self.assertEqual(adjusted.inference_sec, .4)
        self.assertEqual(adjusted.elapsed_sec, .7)
        self.assertTrue(raw.summary.startswith("OK"))
        self.assertEqual(raw.threshold, 3)
        self.assertEqual(effective_result(raw), raw)
        uncalibrated = make_anomaly_result("image", 2.5, None)
        self.assertEqual(effective_result(uncalibrated).status, "uncalibrated")
        self.assertTrue(effective_result(uncalibrated, 2).summary.startswith("NG"))
        error = InferenceResult("image", "error", "anomaly", "ERROR", score=0, error="failed")
        self.assertEqual(effective_result(error, 999).summary, "ERROR")
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError): effective_result(raw, bad)

    def test_global_numeric_sort_before_page_and_combined_filters(self):
        raw = records()
        result = review_page(raw, sort="score", descending=True, limit=60)
        self.assertEqual(result["results"][0]["index"], 74)
        self.assertEqual(result["results"][-1]["index"], 15)
        self.assertEqual(review_page(raw, sort="score", descending=True, offset=60)["results"][0]["index"], 14)
        result = review_page(raw, threshold=1.2, decision="NG", class_name="good", search="image-01", selected_only=True, selected=[12, 13, 14, 20])
        self.assertEqual([r["index"] for r in result["results"]], [12, 14])
        self.assertEqual(raw[12]["threshold"], 3)
        self.assertEqual(review_page(raw, selected_only=True)["filtered_total"], 0)
        self.assertEqual(review_page(raw, threshold=0)["counts"]["NG"], 75)
        self.assertEqual(review_page(raw, threshold=100)["counts"]["OK"], 75)

    def test_dataset_class_only_from_known_split_roots(self):
        self.assertEqual(source_class("/data/test/scratch/a.png", {"data": {"root": "/data"}}), "scratch")
        self.assertEqual(source_class("/data/test/a.png", {"data": {"root": "/data"}}), "")
        self.assertEqual(source_class("/other/scratch/a.png", {"data": {"root": "/data"}}), "")
        self.assertEqual(source_class(r"C:\data\val\empty\a.png", {"data": {"root": r"C:\data"}}), "empty")


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "Qt runtime required")
class ReviewQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_normalized_threshold_range_stays_fixed_and_restores_on_model_switch(self):
        from widgets.inference_review import InferenceReview
        widget = InferenceReview()
        self.addCleanup(widget.deleteLater)
        widget.set_saved_threshold(.5, normalized=True)
        widget.set_context(["a.png"])
        result = replace(make_anomaly_result("a.png", .5, .5),
                         details={"engine": "patchcore", "score_space": "normalized_0_1"})
        widget.update_result(result)
        self.assertEqual((widget.value.minimum(), widget.value.maximum()), (0, 1))
        self.assertEqual(widget._slider_range, (0, 1))
        widget.override.setChecked(True)
        widget.slider.setValue(600)
        self.assertEqual(widget.model.item(0, 4).text(), "OK")
        widget.slider.setValue(500)
        self.assertEqual(widget.model.item(0, 4).text(), "NG")
        widget.value.setValue(2)
        self.assertEqual(widget.value.value(), 1)
        widget.update_result(replace(result, image_path="b.png", score=.999))
        self.assertEqual(widget._slider_range, (0, 1))
        widget.reset.click()
        self.assertEqual(widget.value.value(), .5)
        widget.set_context([])
        widget.set_saved_threshold(3)
        self.assertEqual(widget.value.value(), 3)
        self.assertGreater(widget.value.maximum(), 1)

    def test_table_numeric_sort_filter_selection_and_threshold_reset(self):
        from PySide6.QtCore import Qt
        from widgets.inference_review import InferenceReview
        widget = InferenceReview()
        self.addCleanup(widget.deleteLater)
        raw = records()
        paths = [row["image_path"] for row in raw]
        widget.set_context(paths)
        for row in raw:
            widget.update_result(InferenceResult(**{k: v for k, v in row.items() if k != "index"}))
        widget.table.sortByColumn(5, Qt.SortOrder.DescendingOrder)
        self.assertEqual(widget.proxy.index(0, 1).data(), 75)
        selected = []
        widget.image_selected.connect(selected.append)
        widget.table.setCurrentIndex(widget.proxy.index(0, 3))
        self.assertEqual(selected[-1], paths[74])
        widget.model.item(74, 0).setCheckState(Qt.CheckState.Checked)
        widget.selected_only.setChecked(True)
        self.assertEqual(widget.proxy.rowCount(), 1)
        widget.override.setChecked(True)
        widget.value.setValue(7.4)
        self.assertEqual(widget.proxy.index(0, 4).data(), "NG")
        widget.value.setValue(7.5)
        self.assertEqual(widget.proxy.index(0, 4).data(), "OK")
        widget.reset.click()
        self.assertEqual(widget.proxy.index(0, 4).data(), "NG")
        self.assertEqual(widget.raw[paths[74]].threshold, 3)
        widget.clear_selection.click()
        self.assertEqual(widget.proxy.rowCount(), 0)
        widget.selected_only.setChecked(False)
        widget.class_filter.setCurrentIndex(widget.class_filter.findData("good"))
        widget.decision_filter.setCurrentIndex(widget.decision_filter.findData("OK"))
        widget.search.setText("image-01")
        self.assertEqual(widget.proxy.rowCount(), 5)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "torchvision", "cv2", "matplotlib")), "Full desktop runtime required")
    def test_loaded_anomaly_table_fits_laptop_viewport_with_real_theme(self):
        from PySide6.QtCore import QPoint
        from widgets.inference_widget import InferenceWidget
        widget = InferenceWidget()
        try:
            widget.setStyleSheet((ROOT / "gui/resources/styles/dark_theme.qss").read_text(encoding="utf-8").replace("{ICON_DIR}", (ROOT / "gui/resources/icons").as_posix()))
            widget.model_info.setText("PatchCore 준비 완료 | 메모리 뱅크: 150,000개 패치 | 백본: wide_resnet50_2 | 디바이스: CPU")
            widget.batch_time_label.setText("추론 완료: 75/75장 | 오류 0장 | 전체 12.50초\n추론 평균 120 ms | Grad-CAM 미지원")
            widget.batch_time_label.show()
            widget.result_review.set_saved_threshold(3)
            rows = records()
            widget.result_review.set_context([row["image_path"] for row in rows])
            for row in rows:
                widget.result_review.update_result(InferenceResult(**{k: v for k, v in row.items() if k != "index"}))
            for width, height in ((1100, 650), (1440, 850)):
                widget.resize(width, height)
                widget.show()
                self.app.processEvents()
                table = widget.result_review.table
                self.assertLessEqual(widget.height(), height, "minimum heights must not force the page off screen")
                self.assertLessEqual(widget.width(), width, "controls must fit the available content width")
                self.assertLessEqual(widget.model_group.height(), 90)
                self.assertFalse(widget.result_card.isVisible())
                self.assertGreaterEqual(table.viewport().height(), 240, f"eight batch rows must fit: input={widget.model_group.height()}, review={widget.result_review.height()}, threshold={widget.result_review.threshold_panel.height()}, status={widget.batch_time_label.height()}")
                origin = table.mapTo(widget, QPoint(0, 0))
                self.assertLessEqual(origin.y() + table.height(), height)
                self.assertLessEqual(origin.x() + table.width(), width)
                self.assertEqual(widget.result_review.proxy.rowCount(), 75)
        finally:
            widget.close()
            widget.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
