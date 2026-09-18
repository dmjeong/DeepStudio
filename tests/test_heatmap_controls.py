"""실제 Qt 컨트롤이 캐시된 반응 맵만 갱신하고 판정은 유지하는지 확인한다."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "python"))
GUI_AVAILABLE = all(importlib.util.find_spec(name) is not None
                    for name in ("PySide6", "torch", "torchvision", "cv2", "matplotlib"))


@unittest.skipUnless(GUI_AVAILABLE, "Qt와 모델 실행 의존성 필요")
class HeatmapControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from widgets.inference_widget import InferenceWidget, InferenceResult
        self.widget = InferenceWidget()
        self.widget.resize(1100, 800)
        self.widget.show()
        self.application.processEvents()
        self.original = np.full((2, 4, 3), 100, dtype=np.uint8)
        self.activation = np.array([[0, 0.2, 0.6, 1], [0, 0.2, 0.6, 1]], dtype=np.float32)
        self.widget._current_image = "current.png"
        self.widget.gradcam_checkbox.setChecked(True)
        self.result = InferenceResult("current.png", "ok", "anomaly", "NG 0.1234",
                                      "#E05555", 0.1234, 0.1, elapsed_sec=0.025)
        self.widget._present_result(self.result)
        self.widget._set_heatmap_preview("current.png", self.original, self.activation,
                                        "Grad-CAM", "test")

    def tearDown(self):
        self.widget.hide()
        self.widget._heatmap_timer.stop()
        self.widget.deleteLater()
        self.application.processEvents()

    def settle(self):
        from PySide6.QtTest import QTest
        QTest.qWait(80)
        self.application.processEvents()

    def pixel(self, x, y=0):
        image = self.widget.image_label._source_pixmap.toImage()
        return image.pixelColor(x, y).getRgb()[:3]

    def test_range_changes_pixels_without_running_model_or_changing_decision(self):
        self.widget._run_single_inference = Mock(side_effect=AssertionError("unexpected inference"))
        before = self.pixel(1)
        self.widget.heatmap_lower_spin.setValue(50)
        self.settle()
        self.assertNotEqual(before, self.pixel(1))
        self.assertEqual(self.pixel(1), (100, 100, 100))
        self.assertNotEqual(self.pixel(2), (100, 100, 100))
        self.widget._run_single_inference.assert_not_called()
        self.assertIs(self.widget._current_result, self.result)
        self.assertEqual(self.widget.image_label._result_text, "NG 0.1234")
        np.testing.assert_array_equal(self.widget._heatmap_cache["activation"], self.activation)

    def test_full_range_includes_zero_and_explains_unseen_crop(self):
        self.widget._run_single_inference = Mock(side_effect=AssertionError("unexpected inference"))
        mask = np.ones(self.activation.shape, bool)
        mask[:, -1] = False
        self.widget.gradcam_alpha_combo.setCurrentIndex(self.widget.gradcam_alpha_combo.findData(1.0))
        self.widget._set_heatmap_preview("current.png", self.original, self.activation,
                                        "Grad-CAM", "test", valid_mask=mask)
        self.settle()
        self.assertEqual(self.pixel(0), (0, 0, 127))
        self.assertEqual(self.pixel(3), (100, 100, 100))
        self.assertIn("75.0%", self.widget.gradcam_info.text())
        self.assertIn("중앙 잘림", self.widget.gradcam_info.text())
        self.widget.heatmap_lower_spin.setValue(1)
        self.settle()
        self.assertEqual(self.pixel(0), (100, 100, 100))
        self.widget.heatmap_lower_spin.setValue(0)
        self.settle()
        self.assertEqual(self.pixel(0), (0, 0, 127))
        self.widget._run_single_inference.assert_not_called()
        self.assertIs(self.widget._current_result, self.result)

    def test_upper_range_saturation_alpha_zero_and_reset(self):
        self.widget.gradcam_alpha_combo.setCurrentIndex(self.widget.gradcam_alpha_combo.findData(1.0))
        self.widget.heatmap_upper_spin.setValue(50)
        self.settle()
        self.assertEqual(self.pixel(2), self.pixel(3))
        self.widget.gradcam_alpha_combo.setCurrentIndex(self.widget.gradcam_alpha_combo.findData(0.0))
        self.settle()
        self.assertTrue(all(self.pixel(x) == (100, 100, 100) for x in range(4)))
        self.widget.heatmap_lower_spin.setValue(40)
        self.widget.heatmap_reset_btn.click()
        self.settle()
        self.assertEqual((self.widget.heatmap_lower_spin.value(), self.widget.heatmap_upper_spin.value()),
                         (0, 100))
        self.assertEqual(self.widget.gradcam_alpha_combo.currentData(), 0.5)

    def test_crossed_endpoints_are_adjusted_to_valid_range(self):
        self.widget.heatmap_upper_spin.setValue(20)
        self.widget.heatmap_lower_spin.setValue(90)
        self.assertEqual(self.widget.heatmap_upper_spin.value(), 91)
        self.widget.heatmap_upper_spin.setValue(1)
        self.assertEqual(self.widget.heatmap_lower_spin.value(), 0)
        self.settle()
        self.assertIs(self.widget._current_result, self.result)

    def test_toggle_preserves_native_result_image_and_cached_map(self):
        base = np.full_like(self.original, 33)
        self.widget._set_heatmap_preview("current.png", self.original, self.activation,
                                        "Grad-CAM", "native", base_image=base)
        self.widget.gradcam_checkbox.setChecked(False)
        self.settle()
        self.assertEqual(self.pixel(3), (33, 33, 33))
        self.widget.gradcam_checkbox.setChecked(True)
        self.settle()
        self.assertNotEqual(self.pixel(3), (33, 33, 33))
        self.assertEqual(self.widget.image_label._result_text, self.result.summary)

    def test_clearing_or_changing_image_prevents_stale_timer_render(self):
        self.widget.heatmap_lower_spin.setValue(40)
        self.widget._clear_results()
        self.assertIsNone(self.widget._heatmap_cache)
        self.assertFalse(self.widget._heatmap_timer.isActive())
        self.assertEqual(self.pixel(3), (100, 100, 100))
        self.widget.image_label.setText("new image")
        self.settle()
        self.assertEqual(self.widget.image_label.text(), "new image")
        self.widget._set_heatmap_preview("current.png", self.original, self.activation,
                                        "Grad-CAM", "test")
        self.widget._current_image = "other.png"
        self.widget.image_label.setText("other image")
        self.widget.heatmap_lower_spin.setValue(60)
        self.settle()
        self.assertEqual(self.widget.image_label.text(), "other image")

    def test_patchcore_uses_range_without_gradcam_checkbox_or_threshold_change(self):
        self.widget.gradcam_checkbox.setChecked(False)
        self.widget._anomaly_threshold = 0.1
        self.widget._set_heatmap_preview("current.png", self.original, self.activation,
                                        "PatchCore", "score 0.1234")
        self.widget.heatmap_lower_spin.setValue(50)
        self.settle()
        self.assertEqual(self.pixel(1), (100, 100, 100))
        self.assertNotEqual(self.pixel(3), (100, 100, 100))
        self.assertEqual(self.widget._anomaly_threshold, 0.1)
        self.assertIs(self.widget._current_result, self.result)


if __name__ == "__main__":
    unittest.main()
