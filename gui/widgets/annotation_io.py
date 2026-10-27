"""Background IO used by the local annotation workspace."""

import os
from collections import OrderedDict
from threading import Event, Lock

from PySide6.QtCore import QObject, QRunnable, Qt, QThread, QThreadPool, Signal
from PySide6.QtGui import QImage, QImageReader
from PySide6.QtWidgets import QApplication


def _file_key(path):
    """Identify one version of an image file; an edited file gets a new key."""
    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    return resolved, stat.st_mtime_ns, stat.st_size


class _ImageCache:
    """Small thread-safe LRU of decoded QImages shared by every workspace window.

    QImage is implicitly shared with atomic reference counting, so an image
    decoded on an IO thread can be handed to the GUI thread without a copy.
    """

    def __init__(self, max_entries, max_bytes):
        self._items, self._lock = OrderedDict(), Lock()
        self._max_entries, self._max_bytes = max_entries, max_bytes

    def get(self, key):
        with self._lock:
            image = self._items.get(key)
            if image is not None:
                self._items.move_to_end(key)
            return image

    def pop(self, key):
        with self._lock:
            return self._items.pop(key, None)

    def put(self, key, image):
        if image.isNull() or image.sizeInBytes() > self._max_bytes:
            return
        with self._lock:
            self._items[key] = image
            self._items.move_to_end(key)
            total = sum(item.sizeInBytes() for item in self._items.values())
            while self._items and (len(self._items) > self._max_entries or total > self._max_bytes):
                _, dropped = self._items.popitem(last=False)
                total -= dropped.sizeInBytes()

    def clear(self):
        with self._lock:
            self._items.clear()


# 64×48 썸네일은 장당 약 12KB다. 이미지를 넘길 때마다 이웃 21장을 원본 해상도로
# 다시 디코딩하지 않도록 창 사이에서 공유한다 (PNG는 축소 디코딩이 안 돼 장당 수백 ms).
_thumbnails = _ImageCache(max_entries=4096, max_bytes=64 * 1024 * 1024)

# 다음·이전 원본 이미지. 4096² RGB는 장당 64MB라 2장, 256MB로 제한한다.
_prefetched = _ImageCache(max_entries=2, max_bytes=256 * 1024 * 1024)


class _PrefetchJob(QRunnable):
    def __init__(self, path, key, done):
        super().__init__()
        self.path, self.key, self.done = path, key, done

    def run(self):
        try:
            # Same reader defaults as QPixmap(path). A file edited during decoding
            # gets a new key, so take() never returns these pixels for it.
            _prefetched.put(self.key, QImageReader(self.path).read())
        except Exception:  # noqa: BLE001 - the editor falls back to decoding itself.
            pass
        finally:
            ImagePrefetcher._finished(self.key, self.done)


class ImagePrefetcher:
    """Decode the neighbouring originals while the user edits the current one."""

    _pool = None
    _pending = {}
    _lock = Lock()

    @classmethod
    def request(cls, paths):
        if cls._pool is None:
            # One thread: never compete with the two thumbnail threads or the GUI.
            cls._pool = QThreadPool(QApplication.instance())
            cls._pool.setMaxThreadCount(1)
        for path in paths:
            try:
                key = _file_key(path)
            except OSError:
                continue
            with cls._lock:
                if key in cls._pending or _prefetched.get(key) is not None:
                    continue
                done = cls._pending[key] = Event()
            cls._pool.start(_PrefetchJob(str(path), key, done))

    @classmethod
    def take(cls, path, timeout=5.0):
        """Return the prefetched image if it is still the file's current version.

        A decode already in progress is awaited instead of repeated: it is
        always closer to completion than a new decode on the GUI thread.
        """
        try:
            key = _file_key(path)
        except OSError:
            return None
        with cls._lock:
            done = cls._pending.get(key)
        if done is not None:
            done.wait(timeout)
        return _prefetched.pop(key)

    @classmethod
    def _finished(cls, key, done):
        with cls._lock:
            if cls._pending.get(key) is done:
                del cls._pending[key]
        done.set()


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
                    key = _file_key(path)
                    image = _thumbnails.get(key)
                    if image is None:
                        reader = QImageReader(str(path))
                        size = reader.size()
                        if size.isValid():
                            reader.setScaledSize(
                                size.scaled(64, 48, Qt.AspectRatioMode.KeepAspectRatio)
                            )
                        image = reader.read()
                        _thumbnails.put(key, image)
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
