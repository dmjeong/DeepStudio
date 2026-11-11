"""추론 화면의 '원본 비교': 결과 그림 옆의 원본은 항상 지금 보는 영상의 원본이어야 한다.

    결과 목록에서 다른 영상을 고른다 ─▶ 결과 그림 = 그 영상의 결과
                                      └▶ 원본 칸   = 그 영상의 원본   (이전 영상이 남으면 안 된다)
    흑백 · 16비트 · TIFF 원본도 결과 그림과 같은 방식으로 8비트 RGB로 보여 준다.
"""

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
COLORS = [(200, 30, 30), (30, 30, 200)]          # 영상 0 = 빨강, 영상 1 = 파랑
RESULT = (20, 160, 60)                           # 결과 그림 = 초록


def center(label):
    """라벨이 보관한 그림의 가운데 픽셀 (R, G, B)."""
    image = label._source_pixmap.toImage()
    color = image.pixelColor(image.width() // 2, image.height() // 2)
    return color.red(), color.green(), color.blue()


@unittest.skipUnless(GUI_AVAILABLE, "Qt와 모델 실행 의존성 필요")
class OriginalCompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from widgets.inference_widget import InferenceWidget, InferenceResult
        self.result_type = InferenceResult
        self.temp = tempfile.TemporaryDirectory()
        self.paths = [str(Path(self.temp.name) / f"image_{i}.png") for i in range(2)]
        for path, color in zip(self.paths, COLORS):
            Image.new("RGB", (320, 180), color).save(path)
        self.widget = InferenceWidget()
        self.widget.resize(1100, 700)
        self.widget.show()
        self.application.processEvents()

    def tearDown(self):
        self.widget.hide()
        self.widget.deleteLater()
        self.application.processEvents()
        self.temp.cleanup()

    def preview_backend(self, filename):
        """분할 · 검출처럼 히트맵 없이 결과 그림만 내는 엔진."""
        self.widget._display_numpy_image(np.full((180, 320, 3), RESULT, dtype=np.uint8))
        return self.result_type(filename, "ok", "segment", "OK", "#34C759")

    def run_batch(self):
        from unittest.mock import patch
        self.widget.model = object()
        self.widget._batch_images = self.paths[:]
        self.widget._clear_results()
        with patch.object(self.widget, "_run_custom_inference", side_effect=self.preview_backend):
            self.widget._run_batch_inference()

    def test_selecting_another_result_shows_its_own_original(self):
        self.widget.compare_check.setChecked(True)
        self.run_batch()
        for index in (0, 1, 0):
            self.widget._display_cached_result(self.paths[index])
            self.assertEqual(center(self.widget.image_label), RESULT)
            self.assertEqual(center(self.widget.original_label), COLORS[index])
            # 마우스 위치의 픽셀 값도 지금 보는 영상의 원본에서 읽는다
            self.assertEqual(tuple(self.widget.image_label._pixel_source[90, 160]), COLORS[index])

    def test_original_is_loaded_when_compare_is_switched_on_later(self):
        self.run_batch()
        self.widget._display_cached_result(self.paths[0])
        self.widget._display_cached_result(self.paths[1])
        self.widget.compare_check.setChecked(True)
        self.assertEqual(center(self.widget.original_label), COLORS[1])

    def test_grayscale_and_16bit_originals_are_shown_like_the_result(self):
        folder = Path(self.temp.name)
        gray = folder / "gray.png"
        Image.fromarray(np.full((180, 320), 77, np.uint8)).save(gray)
        deep = folder / "deep.tif"
        Image.fromarray(np.full((180, 320), 40000, np.uint16)).save(deep)
        self.widget.compare_check.setChecked(True)
        self.widget._current_image = str(gray)
        self.widget._show_image(str(gray))
        self.assertEqual(center(self.widget.original_label), (77, 77, 77))
        self.assertEqual(center(self.widget.image_label), (77, 77, 77))
        self.widget._current_image = str(deep)
        self.widget._show_image(str(deep))
        expected = round(40000 / 257)                    # 16비트 → 8비트 (결과 그림과 같은 환산)
        self.assertEqual(center(self.widget.original_label), (expected,) * 3)
        self.assertEqual(int(self.widget.original_label._pixel_source[90, 160]), 40000)   # 픽셀 값은 원래 정밀도

    def test_heatmap_original_in_grayscale_is_not_read_as_rgb(self):
        """히트맵 캐시의 원본이 흑백(H×W)이어도 RGB로 잘못 읽어 줄무늬가 되지 않는다."""
        self.widget.compare_check.setChecked(True)
        self.widget._current_image = self.paths[0]
        self.widget._heatmap_cache = {"image_path": self.paths[0], "original": np.full((180, 320), 90, np.uint8)}
        self.widget._display_numpy_image(np.full((180, 320, 3), RESULT, dtype=np.uint8))
        self.assertEqual(self.widget.original_label._source_pixmap.width(), 320)
        self.assertEqual(center(self.widget.original_label), (90, 90, 90))


if __name__ == "__main__":
    unittest.main()
