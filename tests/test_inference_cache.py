"""디스크 캐시 수명, 원시 지도 복원, 클릭 시 모델 재실행 방지를 확인한다."""

import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "python"))
from core.inference_cache import InferencePreviewCache

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
GUI_AVAILABLE = all(importlib.util.find_spec(name) is not None
                    for name in ("PySide6", "torch", "torchvision", "cv2", "matplotlib"))


class PreviewCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = InferencePreviewCache()
        self.addCleanup(self.cache.clear)
        self.original = np.full((60, 90, 3), 120, dtype=np.uint8)
        self.activation = np.linspace(0, 1, 6, dtype=np.float32).reshape(2, 3)
        self.heatmap = {"original": self.original, "activation": self.activation,
                        "kind": "Grad-CAM", "info": "target: layer", "base_image": None}

    def test_independent_images_restore_pixels_maps_and_failure_message(self):
        self.cache.put("first.png", self.original, self.heatmap, "first")
        self.cache.put("second.png", self.original + 1, info="Grad-CAM 실패: sample error")
        self.original[:] = 0
        first = self.cache.get("first.png")
        self.assertTrue(np.all(first["heatmap"]["original"] == 120))
        np.testing.assert_array_equal(first["heatmap"]["activation"], self.activation)
        second = self.cache.get("second.png")
        self.assertTrue(np.all(second["preview"] == 121))
        self.assertIsNone(second["heatmap"])
        self.assertIn("sample error", second["info"])
        first["heatmap"]["activation"][:] = 0
        np.testing.assert_array_equal(self.cache.get("first.png")["heatmap"]["activation"], self.activation)

    def test_grid_reads_bounded_arrays_and_detail_preserves_full_resolution(self):
        image = np.full((700, 1300, 3), 80, dtype=np.uint8)
        heatmap = dict(self.heatmap, original=image, base_image=image + 1)
        self.cache.put("large.png", image, heatmap)
        thumbnail = self.cache.get("large.png", thumbnail=True)
        self.assertLessEqual(max(thumbnail["preview"].shape[:2]), 512)
        self.assertLessEqual(max(thumbnail["heatmap"]["original"].shape[:2]), 512)
        self.assertEqual(self.cache.get("large.png")["preview"].shape, image.shape)
        self.assertFalse(any(isinstance(value, np.ndarray)
                             for entry in self.cache._entries.values() for value in entry.values()))

    def test_valid_coverage_survives_detail_and_grid_cache_without_false_color(self):
        from core.heatmap import render_heatmap
        image = np.full((700, 1300, 3), 80, np.uint8)
        mask = np.zeros(image.shape[:2], bool)
        mask[:, 300:1000] = True
        heatmap = dict(self.heatmap, original=image, valid_mask=mask,
                       activation=np.zeros(image.shape[:2], np.float32))
        self.cache.put("cropped.png", image, heatmap)
        for thumbnail in (False, True):
            with self.subTest(thumbnail=thumbnail):
                cached = self.cache.get("cropped.png", thumbnail=thumbnail)["heatmap"]
                valid = cached["valid_mask"]
                self.assertEqual(valid.dtype, np.bool_)
                self.assertEqual(valid.shape, cached["original"].shape[:2])
                self.assertTrue(valid.any())
                self.assertFalse(valid[:, 0].any())
                self.assertFalse(valid[:, -1].any())
                _, overlay = render_heatmap(cached["activation"], cached["original"],
                                             alpha=1.0, valid_mask=valid)
                np.testing.assert_array_equal(overlay[~valid], cached["original"][~valid])
                self.assertTrue(np.all(overlay[valid] == [0, 0, 127]))
        np.testing.assert_array_equal(self.cache.get("cropped.png")["heatmap"]["valid_mask"], mask)
        mask[:] = False
        self.assertTrue(self.cache.get("cropped.png")["heatmap"]["valid_mask"].any())

    def test_clear_deletes_files_and_old_results_cannot_reappear(self):
        self.cache.put("first.png", self.original, self.heatmap)
        folder = Path(self.cache._directory.name)
        self.assertTrue(folder.is_dir())
        self.cache.clear()
        self.assertFalse(folder.exists())
        self.assertIsNone(self.cache.get("first.png"))
        self.cache.put("second.png", self.original)
        self.assertIsNone(self.cache.get("first.png"))

    def test_unreadable_image_can_cache_error_metadata_without_pixels(self):
        self.cache.put("missing.png", None, info="file unreadable")
        entry = self.cache.get("missing.png")
        self.assertIsNone(entry["preview"])
        self.assertEqual(entry["info"], "file unreadable")

    def test_disk_failure_removes_previous_entry_instead_of_showing_stale_output(self):
        self.cache.put("first.png", self.original)
        with patch("core.inference_cache.np.savez", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.cache.put("first.png", self.original + 1)
        self.assertIsNone(self.cache.get("first.png"))




@unittest.skipUnless(GUI_AVAILABLE, "Qt와 모델 실행 의존성 필요")
class CachedSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from widgets.inference_widget import InferenceWidget, InferenceResult
        self.widget = InferenceWidget()
        self.Result = InferenceResult
        self.temp = tempfile.TemporaryDirectory()
        self.paths = [str(Path(self.temp.name) / f"image-{i}.png") for i in range(2)]
        for i, path in enumerate(self.paths):
            Image.new("RGB", (30, 20), (90 + i * 20, 90, 90)).save(path)
        self.widget.model = object()
        self.widget.infer_btn.setEnabled(True)
        self.widget._batch_images = self.paths[:]
        self.widget._clear_results()
        self.widget.show()
        self.application.processEvents()

    def tearDown(self):
        self.widget._preview_cache.clear()
        self.widget._heatmap_timer.stop()
        self.widget.hide()
        self.widget.deleteLater()
        self.application.processEvents()
        self.temp.cleanup()

    def backend(self, image_path):
        index = self.paths.index(image_path)
        original = np.full((20, 30, 3), 90 + index * 20, dtype=np.uint8)
        activation = np.array([[0, 1], [0.25, 0.5]], dtype=np.float32)
        self.widget._set_heatmap_preview(image_path, original, activation, "Grad-CAM", f"image {index}")
        return self.Result(image_path, "ok", "classify", f"class-{index} 90%",
                           details={"class_names": ["class-0", "class-1"],
                                    "probabilities": [0.9, 0.1] if index == 0 else [0.1, 0.9]})

    def test_pre_start_clicks_do_not_infer_and_show_waiting_state(self):
        with patch.object(self.widget, "_run_custom_inference", side_effect=self.backend) as model:
            self.widget._on_grid_cell_clicked(0)
            self.widget.result_review.select_path(self.widget._batch_images[1])
            model.assert_not_called()
        self.assertIsNone(self.widget._current_result)
        self.assertIn("추론 시작", self.widget.result_card.content_label.text())

    def test_clicks_restore_original_times_probabilities_and_per_image_maps(self):
        with patch.object(self.widget, "_run_custom_inference", side_effect=self.backend) as model:
            self.widget._run_batch_inference()
            self.assertEqual(model.call_count, 2)
            elapsed = self.widget._batch_results[0].elapsed_sec
            self.widget._on_grid_cell_clicked(0)
            self.assertEqual(self.widget._heatmap_cache["image_path"], self.paths[0])
            self.assertIn("class-0", self.widget.result_card.content_label.text())
            self.assertEqual(self.widget._last_infer_time, elapsed)
            self.widget.result_review.select_path(self.widget._batch_images[1])
            self.assertEqual(self.widget._heatmap_cache["image_path"], self.paths[1])
            self.assertIn("class-1", self.widget.result_card.content_label.text())
            self.widget._on_grid_cell_clicked(0)
            self.assertEqual(model.call_count, 2)
        self.widget.heatmap_lower_spin.setValue(70)
        self.widget._render_cached_heatmap()
        self.assertEqual(self.widget._current_result.image_path, self.paths[0])
        self.widget._on_grid_cell_clicked(1)
        self.widget._on_grid_cell_clicked(0)
        self.assertEqual(self.widget._heatmap_cache["image_path"], self.paths[0])
        self.assertEqual(self.widget.heatmap_lower_spin.value(), 70)

    def test_grid_uses_cached_heatmap_and_result_clear_drops_old_maps(self):
        with patch.object(self.widget, "_run_custom_inference", side_effect=self.backend):
            self.widget._run_batch_inference()
        self.widget._switch_view_mode("grid")
        image = self.widget._grid_cells[0].thumb_label._source_pixmap.toImage()
        self.assertNotEqual(image.pixelColor(29, 0).getRgb()[:3], (90, 90, 90))
        self.widget._clear_results()
        self.widget._run_custom_inference = Mock(side_effect=AssertionError("unexpected inference"))
        self.widget._on_grid_cell_clicked(0)
        self.assertIsNone(self.widget._current_result)
        self.assertIsNone(self.widget._heatmap_cache)
        self.widget._run_custom_inference.assert_not_called()

    def test_cached_cam_failure_message_and_score_survive_selection(self):
        def failed_cam(path):
            self.widget._show_image(path)
            self.widget.gradcam_info.setText("Grad-CAM 실패: gradient missing")
            return self.Result(path, "ok", "anomaly", "NG 0.2", score=0.2, threshold=0.1)
        with patch.object(self.widget, "_run_custom_inference", side_effect=failed_cam):
            self.widget._run_batch_inference()
        self.widget._on_grid_cell_clicked(0)
        self.assertIn("gradient missing", self.widget.gradcam_info.text())
        self.assertEqual(self.widget._current_result.score, 0.2)
        self.assertIn("NG", self.widget.result_card.content_label.text())

    def test_threshold_changes_card_grid_and_table_without_inference_or_cache_changes(self):
        def backend(path):
            self.widget._show_image(path)
            return self.Result(path, "ok", "anomaly", "NG 0.2", score=.2, threshold=.1, inference_sec=.012)
        with patch.object(self.widget, "_run_custom_inference", side_effect=backend):
            self.widget._run_batch_inference()
        self.widget._on_grid_cell_clicked(0)
        raw = self.widget._inference_results[self.paths[0]]
        with patch.object(self.widget, "_run_custom_inference", side_effect=AssertionError("must reuse scores")) as model:
            review = self.widget.result_review
            review.override.setChecked(True)
            review.value.setValue(.3)
            self.assertTrue(self.widget._current_result.summary.startswith("OK"))
            self.assertIn("판정: OK", self.widget.result_card.content_label.text())
            self.assertTrue(self.widget._grid_cells[0].thumb_label._result_text.startswith("OK"))
            self.assertEqual(review.model.item(0, 4).text(), "OK")
            self.assertEqual(self.widget._current_result.inference_sec, .012)
            self.widget._switch_view_mode("grid")
            self.assertTrue(self.widget._grid_cells[0].thumb_label._result_text.startswith("OK"))
            review.reset.click()
            self.assertTrue(self.widget._current_result.summary.startswith("NG"))
            self.assertEqual(self.widget._inference_results[self.paths[0]], raw)
            model.assert_not_called()

    def test_missing_cache_never_triggers_model_and_has_explicit_error(self):
        with patch.object(self.widget, "_run_custom_inference", side_effect=self.backend):
            self.widget._run_batch_inference()
        self.widget._preview_cache.clear()
        with patch.object(self.widget, "_run_custom_inference") as model:
            self.widget._on_grid_cell_clicked(0)
            model.assert_not_called()
        self.assertIn("캐시 불러오기 실패", self.widget.gradcam_info.text())
        self.assertEqual(self.widget._current_result.image_path, self.paths[0])


if __name__ == "__main__":
    unittest.main()
