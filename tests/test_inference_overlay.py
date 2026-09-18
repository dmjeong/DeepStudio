"""실제 Qt 추론 화면에서 결과 배지, 배치 선택, 보기 전환을 검증한다."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "python"))
GUI_AVAILABLE = all(importlib.util.find_spec(name) is not None
                    for name in ("PySide6", "torch", "torchvision", "cv2", "matplotlib"))


@unittest.skipUnless(GUI_AVAILABLE, "Qt와 모델 실행 의존성 필요")
class InferenceOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from widgets.inference_widget import InferenceWidget, InferenceResult
        self.result_type = InferenceResult
        self.temp = tempfile.TemporaryDirectory()
        self.paths = [str(Path(self.temp.name) / f"image_{i}.png") for i in range(2)]
        for path in self.paths:
            Image.new("RGB", (320, 180), "white").save(path)
        self.widget = InferenceWidget()
        self.widget.resize(1100, 700)
        self.widget.show()
        self.application.processEvents()

    def tearDown(self):
        self.widget.hide()
        self.widget.deleteLater()
        self.application.processEvents()
        self.temp.cleanup()

    def backend(self, filename):
        # 학습 모델 대신 정해진 출력과 시각화로 실제 화면 연결만 검증한다.
        rendered = np.full((180, 320, 3), (20, 90, 160), dtype=np.uint8)
        self.widget._display_numpy_image(rendered)
        return self.result_type(filename, "ok", "classify", "정상 98.3%", "#34C759")

    def test_every_engine_presents_summary_above_rendered_image(self):
        from unittest.mock import patch
        for engine, method in (
            ("model", "_run_custom_inference"),
            ("_patchcore_model", "_run_patchcore_inference"),
        ):
            with self.subTest(engine=engine):
                self.widget.model = None
                self.widget._patchcore_model = None
                setattr(self.widget, engine, object())
                with patch.object(self.widget, method, side_effect=self.backend):
                    result = self.widget._run_single_inference(self.paths[0])
                self.assertEqual(result.status, "ok")
                self.assertEqual(self.widget.image_label._result_text, "정상 98.3%")
                source = self.widget.image_label._source_pixmap.toImage()
                self.assertEqual(source.pixelColor(0, 0).getRgb()[:3], (20, 90, 160))

    def test_classification_result_card_keeps_every_probability(self):
        from widgets.inference_widget import ResultCard
        card = ResultCard(self.widget)
        card.show_classification_result(["good", "scratch", "dent", "stain"], np.asarray([.4, .3, .2, .1]))
        self.assertEqual(card.prob_layout.count(), 4)
        card.deleteLater()

    def test_task_details_panel_matches_result_task(self):
        from widgets.inference_widget import InferenceResult
        cases = (
            ("classify", {"class_names": ["good", "scratch"], "probabilities": [.8, .2]}, "분류 클래스별 확률"),
            ("detect", {"detections": [{"class_id": 0, "confidence": .9}], "class_names": ["scratch"]}, "검출 결과 (1개)"),
            ("segment", {"pixel_counts": {0: 90, 1: 10}}, "세그멘테이션 픽셀 분포"),
            ("anomaly", {}, "이상 탐지 결과"),
        )
        for task, details, title in cases:
            with self.subTest(task=task):
                result = InferenceResult("image.png", "ok", task, "결과", score=.8 if task == "anomaly" else None,
                                         threshold=.5 if task == "anomaly" else None, details=details)
                self.widget._update_task_details(result)
                self.assertTrue(self.widget.task_details_group.isVisible())
                self.assertEqual(self.widget.task_details_group.title(), title)

    def test_grid_caches_bounded_preview_without_shrinking_detail_image(self):
        from PySide6.QtGui import QColor, QPixmap
        from widgets.inference_widget import GridCell
        pixmap = QPixmap(2048, 1024)
        pixmap.fill(QColor("white"))
        cell = GridCell(0, self.widget)
        cell.set_image(pixmap)
        cell.set_result("정상 98.3%")
        self.assertEqual(cell.thumb_label._source_pixmap.size().width(), 512)
        self.assertEqual(cell.thumb_label._source_pixmap.size().height(), 256)
        self.assertEqual(cell.thumb_label._result_text, "정상 98.3%")
        self.widget.image_label.setPixmap(pixmap)
        self.assertEqual(self.widget.image_label._source_pixmap.width(), 2048)
        cell.deleteLater()

    def test_grid_and_list_click_restore_success_or_error_without_inference(self):
        from unittest.mock import patch
        self.widget.model = object()
        self.widget._batch_images = self.paths[:]
        self.widget._clear_results()
        def backend(filename):
            if filename == self.paths[1]:
                raise ValueError("bad model")
            return self.backend(filename)
        with patch.object(self.widget, "_run_custom_inference", side_effect=backend) as model:
            self.widget._run_batch_inference()
            self.assertEqual(model.call_count, 2)
            saved_time = self.widget._batch_results[0].elapsed_sec
            self.widget._switch_view_mode("grid")
            self.widget._on_grid_cell_clicked(0)
            self.assertEqual(self.widget._view_mode, "single")
            self.assertEqual(self.widget.image_label._result_text, "정상 98.3%")
            self.assertEqual(self.widget._last_infer_time, saved_time)
            self.widget.result_review.select_path(self.widget._batch_images[1])
            self.assertEqual(model.call_count, 2)
        self.assertEqual(self.widget._batch_results[1].status, "error")
        self.assertEqual(self.widget.image_label._result_text, "ERROR")
        self.assertEqual(self.widget._grid_cells[1].thumb_label._result_text, "ERROR")
        self.assertIn("ERROR", self.widget.result_review.model.item(1, 4).text())
        self.assertIn("bad model", self.widget.result_card.content_label.text())

    def test_batch_and_view_switch_preserve_matching_image_heatmap_and_result(self):
        from unittest.mock import patch
        self.widget.model = object()
        self.widget._batch_images = self.paths[:]
        self.widget._clear_results()
        with patch.object(self.widget, "_run_custom_inference", side_effect=self.backend):
            self.widget._run_batch_inference()
        self.assertEqual(self.widget._current_image, self.paths[-1])
        self.assertEqual(self.widget._current_result.image_path, self.paths[-1])
        before = self.widget.image_label._source_pixmap.toImage()
        self.widget._switch_view_mode("grid")
        self.widget._switch_view_mode("single")
        self.assertEqual(self.widget.image_label._source_pixmap.toImage(), before)
        self.assertEqual(self.widget.image_label._result_text, "정상 98.3%")
        self.widget._clear_results()
        self.assertEqual(self.widget.image_label._result_text, "")
        self.assertTrue(all(cell.thumb_label._result_text == "" for cell in self.widget._grid_cells))


if __name__ == "__main__":
    unittest.main()
