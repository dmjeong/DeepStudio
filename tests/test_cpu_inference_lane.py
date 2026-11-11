"""같은 앱에서 학습 중 CPU 추론: 주 작업 줄이 차 있어도 CPU 모델 검사 · 추론은 CPU 추론 줄에서 함께 돈다.

    주 작업 줄 (학습 · 내보내기 · 추론)   ── 한 번에 하나
    CPU 추론 줄 (학습 중의 CPU 추론)      ── 한 번에 하나, 같은 작업 폴더 · 같은 작업 기록
    GPU 추론은 학습 중에 막는다 (GPU 메모리를 나눠 쓰면 학습이 중단될 수 있다).
"""

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gui"), str(ROOT / "python")]
pytest.importorskip("psutil")


@pytest.fixture
def lanes(tmp_path, monkeypatch):
    from core import job_manager
    monkeypatch.setenv("DEEP_STUDIO_DESKTOP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(job_manager, "_manager", None)
    monkeypatch.setattr(job_manager, "_cpu_manager", None)
    yield job_manager
    for manager in (job_manager._manager, job_manager._cpu_manager):
        if manager is not None:
            if manager.process is None:
                manager.active_id = None        # 테스트가 '실행 중'으로만 표시해 둔 가짜 작업
            manager.close()


def test_idle_main_lane_is_used_as_before(lanes):
    assert lanes.inference_manager("infer", "cpu") is lanes.desktop_manager()
    assert lanes.inference_manager("inspect_model", "cuda:0") is lanes.desktop_manager()


def test_cpu_inference_gets_its_own_lane_while_this_app_trains(lanes):
    main = lanes.desktop_manager()
    main.active_id = "0" * 32                                   # 이 앱의 학습이 주 작업 줄을 차지한 상태
    for kind in ("inspect_model", "infer"):
        lane = lanes.inference_manager(kind, "cpu")
        assert lane is lanes.desktop_cpu_manager() and lane is not main
    assert lane.root == main.root                               # 작업 폴더와 작업 기록은 함께 쓴다
    with pytest.raises(RuntimeError, match="CPU로 바꾸면"):
        lanes.inference_manager("infer", "cuda:0")              # GPU 추론은 학습과 함께 돌리지 않는다
    lane.active_id = "1" * 32                                   # CPU 추론 줄도 한 번에 하나
    with pytest.raises(RuntimeError, match="CPU 추론이 이미 실행 중"):
        lanes.inference_manager("infer", "cpu")


def test_cpu_job_runs_while_the_main_lane_is_busy_and_shows_in_history(lanes):
    """실제 작업 프로세스를 띄운다: 주 작업 줄이 차 있고 계산 잠금이 잡혀 있어도 CPU 줄의 작업은 끝까지 돈다."""
    from webapp.jobs import TERMINAL
    from webapp.locking import exclusive_file
    main = lanes.desktop_manager()
    with exclusive_file(main.root / "compute.lock"):            # 학습 프로세스가 잡고 있는 잠금
        main.active_id = "0" * 32
        lane = lanes.inference_manager("inspect_model", "cpu")
        job = lane.start("inspect_model", {"weights": str(lanes._state_dir() / "missing.pt"), "device": "cpu"})
        deadline = time.monotonic() + 120
        while lane.read(job["id"])["job"]["status"] not in TERMINAL and time.monotonic() < deadline:
            time.sleep(0.2)
        finished = lane.read(job["id"])["job"]
        main.active_id = None
    # 가중치가 없어 검사는 실패하지만, 잠금 때문에 막힌 것이 아니라 작업이 실제로 실행된 결과다
    assert finished["status"] == "failed"
    assert "다른 계산 프로세스" not in finished["error"] and "missing.pt" in finished["error"]
    assert job["id"] in {row["id"] for row in main.list(limit=100)}      # 주 작업 줄의 작업 기록에도 보인다


def test_desktop_job_talks_to_the_lane_it_was_given(lanes):
    pytest.importorskip("PySide6")
    from core.desktop_jobs import DesktopInferenceJob, DesktopJob
    lane = lanes.desktop_cpu_manager()
    assert DesktopJob("inspect_model", {}, manager=lane).manager is lane
    assert DesktopJob("inspect_model", {}).manager is lanes.desktop_manager()
    assert DesktopInferenceJob({"images": []}, cache=None, manager=lane).manager is lane


def test_inference_screen_loads_a_model_on_the_cpu_lane_during_training(lanes, tmp_path, monkeypatch):
    """추론 화면: 이 앱의 학습이 주 작업 줄을 차지해도 '모델 로드'가 경고 없이 CPU 추론 줄로 간다."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    pytest.importorskip("torch")
    from unittest.mock import MagicMock
    from PySide6.QtWidgets import QApplication, QMessageBox
    application = QApplication.instance() or QApplication([])
    from core import desktop_jobs
    from widgets.inference_widget import InferenceWidget

    started = []

    class FakeJob:
        """작업 프로세스를 띄우지 않고 어느 줄로 보냈는지만 기록한다."""
        def __init__(self, kind, payload, parent=None, *, manager=None):
            started.append((kind, payload["device"], manager))
            self.failed = self.completed = self.finished = MagicMock()

        def start(self):
            pass

    monkeypatch.setattr(desktop_jobs, "DesktopJob", FakeJob)
    warning = MagicMock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    widget = InferenceWidget()
    try:
        lanes.desktop_manager().active_id = "0" * 32             # 학습 중
        assert widget.infer_device_combo.currentData() == "cpu"
        widget._start_model_inspection(str(weights))
        assert not warning.called
        assert started == [("inspect_model", "cpu", lanes.desktop_cpu_manager())]
    finally:
        widget._model_inspection = None
        widget.deleteLater()
        application.processEvents()
