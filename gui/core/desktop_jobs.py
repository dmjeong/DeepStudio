"""EXE와 개발 실행이 공유하는 프로세스 작업 관리자 및 영구 추론 캐시."""
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal

from core.job_manager import desktop_manager as desktop_manager

FORCE_STOP_AFTER_SEC = 30.0  # 중지를 요청한 뒤 이 시간 안에 끝나지 않으면 멈춘 것으로 보고 강제로 끝낸다


def force_stop(manager, job_id):
    """멈춘 계산 프로세스(와 데이터 로더 자식 프로세스)를 끝내고 작업을 '취소'로 기록한다 (데스크톱 전용).

    학습은 배치 사이에서 중지 요청을 확인한다. 그래도 끝나지 않으면 데이터 로더나 장치에서 멈춘 것이라
    기다려도 끝나지 않는다. 결과 기록을 먼저 남겨 두면 작업 관리자가 '비정상 종료' 대신 '취소'로 정리한다.
    """
    import json
    import psutil
    process = getattr(manager, "process", None)
    if process is None or process.poll() is not None:
        return False
    result = manager.directory(job_id) / "result.json"
    if not result.exists():
        result.write_text(json.dumps({"status": "cancelled", "error": "", "output": {},
                                      "note": "중지 요청 후 응답이 없어 강제로 끝냈습니다"}, ensure_ascii=False),
                          encoding="utf-8")
    try:
        parent = psutil.Process(process.pid)
        tree = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return False
    for item in tree:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(tree, timeout=10)
    return True


class DesktopJob(QThread):
    event = Signal(str, object)
    failed = Signal(str)
    completed = Signal(object)

    def __init__(self, kind, payload, parent=None):
        super().__init__(parent)
        self.kind, self.payload = kind, payload
        self.job_id = None
        self.cancelled = False
        self.cancel_requested_at = None
        self.error = ""
        self.elapsed_sec = 0.0

    def stop(self):
        self.cancelled = True
        if self.job_id:
            desktop_manager().cancel(self.job_id)

    def run(self):
        from webapp.jobs import TERMINAL
        try:
            manager = desktop_manager()
            job = manager.start(self.kind, self.payload)
            self.job_id = job["id"]
            offset = 0
            import time
            while True:
                if self.cancelled:
                    manager.cancel(self.job_id)
                    if self.cancel_requested_at is None:
                        self.cancel_requested_at = time.monotonic()
                    elif time.monotonic() - self.cancel_requested_at > FORCE_STOP_AFTER_SEC:
                        if force_stop(manager, self.job_id):
                            self.event.emit("log_message", ["중지 요청 후 30초 동안 응답이 없어 학습 프로세스를 강제로 끝냈습니다."])
                        self.cancel_requested_at = float("inf")   # 한 번만
                view = manager.read(self.job_id, offset)
                offset = view["offset"]
                for item in view["events"]:
                    self.event.emit(item["event"], item["args"])
                if view["job"]["status"] in TERMINAL and len(view["events"]) < 250:
                    self.elapsed_sec = view["job"].get("duration_sec", 0)
                    self.error = view["job"].get("error", "")
                    if self.error:
                        self.failed.emit(self.error)
                    self.completed.emit(view["job"])
                    break
                self.msleep(100)
        except Exception as exc:
            self.error = str(exc)
            self.failed.emit(self.error)
            # ``failed`` is informational; the owner also needs the same
            # terminal event used by JobManager failures to release controls
            # and the QThread reference.  Without this, a startup/read error
            # could leave a model-pack button disabled indefinitely.
            self.completed.emit({"status": "failed", "error": self.error, "output": {}})


class PersistentInferenceCache:
    """작업 폴더의 NPZ/메타데이터를 그대로 읽고 clear에서는 원본을 삭제하지 않는다."""
    def __init__(self, directory=None):
        self.root = Path(directory) / "results" if directory else None
        self.entries = {}
        if self.root:
            from webapp.storage import read_json
            for path in sorted(self.root.glob("*.json"), key=lambda p: int(p.stem)):
                try:
                    row = read_json(path)
                    self.entries[row["image_path"]] = row
                except (OSError, ValueError, KeyError):
                    continue

    def clear(self):
        self.entries.clear()

    def get(self, image_path, *, thumbnail=False):
        row = self.entries.get(image_path)
        if row is None or self.root is None or not row.get("cache_ready"):
            return None
        if thumbnail and (self.root / f"{row['index']}.thumb.png").is_file():
            with Image.open(self.root / f"{row['index']}.thumb.png") as image:
                return {"preview": np.array(image), "heatmap": None, "info": row.get("info", "")}
        with np.load(self.root / f"{row['index']}.npz", allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        heatmap = row.get("heatmap")
        if heatmap:
            heatmap = {**heatmap, "image_path": image_path, "original": arrays["original"],
                       "activation": arrays["activation"], "base_image": arrays.get("base_image"),
                       "valid_mask": arrays.get("valid_mask")}
        return {"preview": arrays.get("preview"), "heatmap": heatmap, "info": row.get("info", "")}


def inference_result(row):
    # Persisted jobs from releases before normalized PatchCore scores contain
    # raw distance plus heatmap.kind. Convert the display copy before dropping
    # storage-only fields while leaving the JSON file untouched.
    from core.inference_review import restored_result
    return restored_result(row)


class DesktopInferenceJob(DesktopJob):
    result_ready = Signal(object)
    progress = Signal(int, int)

    def __init__(self, payload, cache, parent=None):
        super().__init__("infer", payload, parent)
        self.paths = tuple(payload["images"])
        self.cache = cache
        self.event.connect(self._event)

    def _event(self, name, args):
        if name == "inference_result":
            row = args[0]
            self.cache.root = desktop_manager().directory(self.job_id) / "results"
            self.cache.entries[row["image_path"]] = row
            self.result_ready.emit(inference_result(row))
        elif name == "progress_updated":
            self.progress.emit(*args)
