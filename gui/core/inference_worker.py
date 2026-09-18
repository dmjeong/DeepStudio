"""추론 중에도 Qt 이벤트 처리를 유지하는 취소 가능한 계산 워커."""

import time
from dataclasses import replace
from PySide6.QtCore import QThread, Signal


class InferenceWorker(QThread):
    result_ready = Signal(object)
    progress = Signal(int, int)
    failed = Signal(str)

    def __init__(self, engine, paths, cache, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.paths = tuple(paths)
        self.cache = cache
        self.elapsed_sec = 0.0
        self.error = ""
        self.cancelled = False

    def stop(self):
        self.cancelled = True
        self.requestInterruption()

    def run(self):
        started = time.perf_counter()
        try:
            if callable(getattr(self.engine, "prepare", None)):
                self.engine.prepare()
            for index, path in enumerate(self.paths):
                if self.isInterruptionRequested():
                    break
                image_started = time.perf_counter()
                result = self.engine.infer(path)
                try:
                    self.cache.put(path, self.engine._current_preview_rgb,
                                   self.engine._heatmap_cache, self.engine.info)
                except Exception as exc:
                    raise OSError(f"결과 이미지 캐시 저장 실패 ({path}): {exc}") from exc
                finally:
                    # 미리보기 저장 실패도 이미 계산한 판정과 측정값을 지우지 않는다.
                    result = replace(result, elapsed_sec=time.perf_counter() - image_started)
                    self.result_ready.emit(result)
                    self.progress.emit(index + 1, len(self.paths))
        except Exception as exc:
            self.error = str(exc)
            self.failed.emit(self.error)
        finally:
            self.elapsed_sec = time.perf_counter() - started
