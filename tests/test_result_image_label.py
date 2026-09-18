"""실제 Qt 페인팅으로 결과 배지의 갱신, 위치와 초기화를 검사한다."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))
QT_AVAILABLE = importlib.util.find_spec("PySide6") is not None


@unittest.skipUnless(QT_AVAILABLE, "PySide6 필요")
class ResultImageLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from widgets.result_image_label import ResultImageLabel
        self.label = ResultImageLabel()
        self.label.resize(320, 240)
        self.label.show()
        self.application.processEvents()

    def tearDown(self):
        self.label.close()
        self.label.deleteLater()
        self.application.processEvents()

    def source(self, width=640, height=320):
        from PySide6.QtGui import QColor, QPixmap
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("white"))
        return pixmap

    def capture(self):
        self.application.processEvents()
        return self.label.grab().toImage()

    def pixel(self, image, x, y):
        ratio = image.devicePixelRatioF()
        return image.pixelColor(round(x * ratio), round(y * ratio))

    def changed_pixels(self, before, after, rect):
        return sum(self.pixel(before, x, y) != self.pixel(after, x, y)
                   for y in range(rect.top(), rect.bottom() + 1)
                   for x in range(rect.left(), rect.right() + 1))

    def test_result_added_after_image_is_visible_without_reloading(self):
        self.label.setPixmap(self.source())
        before = self.capture()
        self.label.set_result("NG 98.7%", "#E05555")
        after = self.capture()
        badge = self.label._badge_rect()
        self.assertGreater(self.changed_pixels(before, after, badge), 30)
        self.assertEqual(self.label.toolTip(), "NG 98.7%")
        self.assertEqual(self.label._source_pixmap.toImage().pixelColor(10, 10).name(), "#ffffff")
        # 배지는 화면 페인트에만 존재하며 QLabel 픽스맵에는 남지 않는다.
        self.assertEqual(self.label.pixmap().toImage().pixelColor(10, 10).name(), "#ffffff")

    def test_result_before_image_and_later_update_are_both_visible(self):
        self.label.set_result("OK", "#34C759")
        self.label.setPixmap(self.source())
        before = self.capture()
        self.label.set_result("ERROR", "#E05555")
        after = self.capture()
        self.assertGreater(self.changed_pixels(before, after, self.label._badge_rect()), 10)
        self.assertEqual(self.label.toolTip(), "ERROR")

    def test_resize_preserves_source_and_badge_at_image_top_left(self):
        self.label.setPixmap(self.source())
        self.label.set_result("정상 99.8%")
        self.capture()
        old_height = self.label._badge_rect().height()
        for width, height in ((480, 300), (220, 400), (640, 240)):
            self.label.resize(width, height)
            image = self.capture()
            image_rect, badge = self.label._image_rect(), self.label._badge_rect()
            self.assertEqual((self.label._source_pixmap.width(), self.label._source_pixmap.height()),
                             (640, 320))
            self.assertEqual(badge.topLeft().x(), image_rect.left() + 4)
            self.assertEqual(badge.topLeft().y(), image_rect.top() + 4)
            self.assertEqual(badge.height(), old_height)
            self.assertTrue(self.label.contentsRect().contains(badge))
            self.assertAlmostEqual(image_rect.width() / image_rect.height(), 2, delta=0.02)
            # 흰 이미지 위 배경이 실제로 어두워졌는지 확인한다.
            pixel = self.pixel(image, badge.left() + 5, badge.top() + 5)
            self.assertLess(pixel.red(), 220)

    def test_clear_and_placeholder_remove_image_result_and_tooltip(self):
        self.label.setPixmap(self.source())
        self.label.set_result("OK")
        self.label.clear()
        self.assertTrue(self.label._source_pixmap.isNull())
        self.assertTrue(self.label._badge_rect().isEmpty())
        self.assertEqual(self.label.toolTip(), "")
        self.label.setPixmap(self.source())
        self.label.set_result("NG")
        self.label.setText("이미지를 선택해 주세요")
        self.label.resize(400, 300)
        self.capture()
        self.assertEqual(self.label.text(), "이미지를 선택해 주세요")
        self.assertTrue(self.label._source_pixmap.isNull())
        self.assertTrue(self.label._badge_rect().isEmpty())
        self.assertEqual(self.label.toolTip(), "")

    def test_long_unicode_name_is_bounded_and_full_text_is_available(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFontMetrics
        self.label.setPixmap(self.source(200, 600))
        text = "검사 대상 표면 파손 클래스_" * 15 + " 98.5%"
        self.label.set_result(text)
        image = self.capture()
        badge = self.label._badge_rect()
        image_rect = self.label._image_rect()
        self.assertLessEqual(badge.right(), image_rect.right())
        self.assertEqual(self.label.toolTip(), text)
        metrics = QFontMetrics(self.label._badge_font())
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, badge.width() - 12)
        self.assertLess(len(elided), len(text))
        self.assertLessEqual(metrics.horizontalAdvance(elided), badge.width() - 12)
        self.assertLess(self.pixel(image, badge.left() + 5, badge.top() + 5).red(), 220)

    def test_error_badge_can_be_shown_over_unreadable_image_placeholder(self):
        self.label.setText("이미지 표시 불가")
        before = self.capture()
        self.label.set_result("ERROR", "#E05555")
        after = self.capture()
        self.assertGreater(self.changed_pixels(before, after, self.label._badge_rect()), 30)
        self.label.resize(400, 300)
        self.capture()
        self.assertEqual(self.label.text(), "이미지 표시 불가")
        self.assertEqual(self.label.toolTip(), "ERROR")
        self.assertFalse(self.label._badge_rect().isEmpty())

    def test_removing_result_restores_original_pixels(self):
        self.label.setPixmap(self.source())
        before = self.capture()
        self.label.set_result("NG 100%", "#E05555")
        self.capture()
        self.label.set_result()
        after = self.capture()
        self.assertEqual(before, after)
        self.assertEqual(self.label.toolTip(), "")


if __name__ == "__main__":
    unittest.main()
