"""그리기 기록이 없는 마스크 픽셀('PNG 1' 영역)을 작업실에서 지운다.

    마스크 PNG = 픽셀(학습 정답) + 그리기 기록. 클래스 변경·외부 마스크처럼 기록이 없는 픽셀은
    작업실에서 클래스마다 한 영역('PNG <번호>')으로 보인다.
    이전: 지우면 픽셀은 지워지는데 목록의 'PNG 1'이 그대로 남아 삭제가 안 된 것처럼 보였고,
         같은 클래스로 새로 그린 객체까지 함께 지워졌다.
"""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core.annotation_tools import AnnotationDraftStore
from core.mask_annotations import MaskDocument


@pytest.fixture
def dialog(tmp_path):
    from widgets.segmentation_annotation import SegmentationAnnotationDialog
    app = QApplication.instance() or QApplication([])
    Image.new("RGB", (60, 40), "gray").save(tmp_path / "a.png")
    base = np.zeros((40, 60), np.uint8)
    base[5:15, 5:20] = 1      # 원본 층 shift
    base[25:35, 2:10] = 2     # 원본 층 fm
    drawn = {"kind": "polygon", "class_id": 1, "object_id": "drawn",          # 나중에 그린 shift 객체
             "points": [40 / 60, 20 / 40, 55 / 60, 20 / 40, 55 / 60, 35 / 40, 40 / 60, 35 / 40]}
    document = MaskDocument(60, 40, 3, base=base, shapes=[drawn])
    widget = SegmentationAnnotationDialog(tmp_path / "a.png", ["background", "shift", "fm"], document,
                                          lambda *_: None, draft_store=AnnotationDraftStore(tmp_path / "drafts"))
    widget.show()
    app.processEvents()
    yield widget, app
    widget.close()
    # close() writes an unsaved draft asynchronously. Deleting its owning
    # dialog before that thread finishes aborts Qt on Windows.
    from PySide6.QtTest import QTest
    import time
    deadline = time.monotonic() + 10
    while widget.draft_worker is not None and time.monotonic() < deadline:
        QTest.qWait(10)
    assert widget.draft_worker is None and not widget.isVisible()
    widget.deleteLater()


def _labels(widget):
    return [widget.table.item(row, 0).text() for row in range(widget.table.rowCount())]


def test_deleting_a_png_layer_removes_its_row_and_keeps_drawn_objects(dialog):
    widget, app = dialog
    assert "PNG 1" in _labels(widget) and "PNG 2" in _labels(widget)
    drawn_pixels = int((widget.document.render() == 1).sum()) - 150     # 원본 층 shift 150px을 뺀 그린 객체
    widget._remove(-3)                                                    # PNG 1 (원본 층 shift)
    app.processEvents()
    assert "PNG 1" not in _labels(widget) and "PNG 2" in _labels(widget)
    pixels = widget.document.render()
    assert not pixels[5:15, 5:20].any()                                   # 원본 층 shift는 지워졌고
    assert int((pixels == 1).sum()) == drawn_pixels > 0                   # 그린 shift 객체는 남는다
    widget._undo()
    app.processEvents()
    assert "PNG 1" in _labels(widget) and (widget.document.render()[5:15, 5:20] == 1).all()


def test_base_classes_left_follows_erase_and_cover():
    base = np.zeros((10, 10), np.uint8)
    base[:5] = 1
    document = MaskDocument(10, 10, 3, base=base)
    assert document.base_classes_left() == (1,)
    cover = document.raster_shape(document.base_layer_mask(1), 2)         # 다른 클래스로 덮으면
    document.append_shape(cover)
    assert document.base_classes_left() == ()                              # 원본 층은 남지 않는다
