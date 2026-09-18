"""데이터 인덱스와 축소 이미지를 GUI 밖에서 준비하는 취소 가능한 스캔."""
from PySide6.QtCore import QThread, Signal, QSize, Qt, QCoreApplication
from PySide6.QtGui import QImageReader
from core.dataset_editor import scan_dataset

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
            index = scan_dataset(self.project, self.isInterruptionRequested)
            thumbnails = {}
            for split in ("train", "val", "test"):
                for row in [r for r in index["images"] if r["split"] == split][:200]:
                    if self.isInterruptionRequested():
                        return
                    reader = QImageReader(row["path"])
                    size = reader.size()
                    if size.isValid():
                        size.scale(QSize(132, 120), Qt.AspectRatioMode.KeepAspectRatio)
                        reader.setScaledSize(size)
                    thumbnails[row["path"]] = reader.read()
            self.ready.emit(index, thumbnails)
        except InterruptedError:
            pass
        except Exception as exc:
            self.failed.emit(str(exc))
