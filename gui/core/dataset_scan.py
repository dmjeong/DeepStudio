"""데이터 인덱스와 축소 이미지를 GUI 밖에서 준비하는 취소 가능한 스캔."""
import numpy as np
from PySide6.QtCore import QThread, Signal, QSize, Qt, QCoreApplication, QRectF
from PySide6.QtGui import QImage, QImageReader, QPainter, QPen, QColor
from core.dataset_editor import scan_dataset
from core.dataset_regions import class_rgb

THUMBNAILS_PER_SPLIT = 200


def compose_overlay(thumbnail, overlay):
    """썸네일 위에 클래스 영역을 작업실과 같은 색으로 겹친다 (분할: 반투명 채움, 탐지: 박스 윤곽)."""
    if thumbnail is None or thumbnail.isNull():
        return None
    result = thumbnail.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(result)
    try:
        if overlay[0] == "mask":
            classes = overlay[1]
            height, width = classes.shape
            palette = np.zeros((256, 4), np.uint8)  # BGRA (Format_ARGB32 little-endian)
            for class_id in range(1, 255):
                red, green, blue = class_rgb(class_id)
                palette[class_id] = [blue, green, red, 165]
            rgba = np.ascontiguousarray(palette[classes])
            layer = QImage(rgba.data, width, height, width * 4, QImage.Format.Format_ARGB32).copy()
            painter.drawImage(QRectF(0, 0, result.width(), result.height()), layer)
        else:
            _, (width, height), boxes = overlay
            sx, sy = result.width() / width, result.height() / height
            for class_id, (x, y, w, h) in boxes:
                pen = QPen(QColor(*class_rgb(class_id)))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawRect(QRectF(x * sx, y * sy, max(1.0, w * sx), max(1.0, h * sy)))
    finally:
        painter.end()
    return result

_active_scans = set()
_shutdown_connected = False


def stop_scans():
    workers = tuple(_active_scans)
    for worker in workers:
        worker.requestInterruption()
    for worker in workers:
        worker.wait(2000)


class DatasetScan(QThread):
    ready = Signal(object, object)
    failed = Signal(str)

    def __init__(self, project):
        super().__init__()
        self.project = project

    def start(self):
        global _shutdown_connected
        if not _shutdown_connected and QCoreApplication.instance() is not None:
            QCoreApplication.instance().aboutToQuit.connect(stop_scans)
            _shutdown_connected = True
        _active_scans.add(self)
        self.finished.connect(self._release)
        super().start()

    def _release(self):
        _active_scans.discard(self)
        self.deleteLater()

    def run(self):
        try:
            object_task = self.project.task in ("detect", "obb", "segment")
            index = scan_dataset(self.project, self.isInterruptionRequested,
                                 overlay_limit=THUMBNAILS_PER_SPLIT if object_task else 0)
            raw_overlays = index.pop("overlays", {})
            index["overlays"] = {}
            thumbnails = {}
            for split in ("train", "val", "test"):
                for row in [r for r in index["images"] if r["split"] == split][:THUMBNAILS_PER_SPLIT]:
                    if self.isInterruptionRequested():
                        return
                    reader = QImageReader(row["path"])
                    size = reader.size()
                    if size.isValid():
                        size.scale(QSize(132, 120), Qt.AspectRatioMode.KeepAspectRatio)
                        reader.setScaledSize(size)
                    thumbnails[row["path"]] = reader.read()
                    if row["path"] in raw_overlays:
                        composed = compose_overlay(thumbnails[row["path"]], raw_overlays[row["path"]])
                        if composed is not None:
                            index["overlays"][row["path"]] = composed
            self.ready.emit(index, thumbnails)
        except InterruptedError:
            pass
        except Exception as exc:
            self.failed.emit(str(exc))
