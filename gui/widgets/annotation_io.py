"""Background IO used by the local annotation workspace."""

from threading import Event

from PySide6.QtCore import QObject, QRunnable, Qt, QThread, QThreadPool, Signal
from PySide6.QtGui import QImage, QImageReader
from PySide6.QtWidgets import QApplication


class AnnotationWrite(QThread):
    def __init__(self, action, parent):
        super().__init__(parent)
        self.action, self.error, self.value = action, None, None
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def shutdown(self):
        self.requestInterruption()
        self.wait()

    def run(self):
        try:
            self.value = self.action()
        except Exception as exc:  # noqa: BLE001 - worker boundary reports failures to the GUI.
            self.error = str(exc)
        finally:
            self.action = None


class _ThumbnailJob(QRunnable):
    def __init__(self, loader):
        super().__init__()
        self.loader = loader

    def run(self):
        try:
            self.loader.run()
        finally:
            self.loader = None


class ThumbnailLoader(QObject):
    ready = Signal(object)
    finished = Signal()
    _pool = None

    def __init__(self, paths, state_provider=None):
        # Reuse a bounded pair of native IO threads across image navigation.
        # A closed editor's remaining thumbnails cannot create unlimited
        # decoder threads, and finished jobs release their state provider.
        super().__init__(QApplication.instance())
        if ThumbnailLoader._pool is None:
            ThumbnailLoader._pool = QThreadPool(QApplication.instance())
            ThumbnailLoader._pool.setMaxThreadCount(2)
            ThumbnailLoader._pool.setExpiryTimeout(-1)
        self.paths = list(paths)
        self.state_provider = state_provider
        self._cancelled, self._done = Event(), Event()
        self._done.set()
        self._started = False
        self.finished.connect(self.deleteLater)
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def start(self):
        if self._started:
            raise RuntimeError("썸네일 작업은 한 번만 시작할 수 있습니다.")
        self._started = True
        self._done.clear()
        self._pool.start(_ThumbnailJob(self))

    def requestInterruption(self):
        self._cancelled.set()

    def isInterruptionRequested(self):
        return self._cancelled.is_set()

    def isRunning(self):
        return self._started and not self._done.is_set()

    def wait(self, milliseconds=None):
        return self._done.wait(None if milliseconds is None else milliseconds / 1000)

    def shutdown(self):
        self.requestInterruption()
        self.wait()

    def run(self):
        try:
            for index, path in self.paths:
                if self.isInterruptionRequested():
                    break
                try:
                    reader = QImageReader(str(path))
                    size = reader.size()
                    if size.isValid():
                        reader.setScaledSize(
                            size.scaled(64, 48, Qt.AspectRatioMode.KeepAspectRatio)
                        )
                    image = reader.read()
                    state = (
                        self.state_provider(path) if self.state_provider else "미작업"
                    )
                    if image.isNull():
                        state = "오류"
                except Exception:  # noqa: BLE001 - a bad thumbnail must not abort the pool.
                    image, state = QImage(), "오류"
                self.ready.emit((index, image, state))
        finally:
            self.state_provider = None
            self.paths = []
            self._done.set()
            self.finished.emit()
