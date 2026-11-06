"""학습이 멈춘 것처럼 보이고 중지도 안 되던 문제.

    이전: 중지 요청을 에폭 시작 때만 확인 → 첫 에폭이 길거나 멈추면 중지가 먹지 않음
          진행 기록은 에폭이 끝날 때만 → 느린 것과 멈춘 것을 구분할 수 없음
    이후: 배치마다 중지 확인 · 데스크톱 로그에 장치, 첫 배치, 30초마다 진행
          데스크톱 작업은 중지 후 30초 안에 끝나지 않으면 프로세스를 강제로 끝내고 '취소'로 기록
"""

import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
torch = pytest.importorskip("torch")


def _dataset(root):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)
        for index in range(2):
            mask = np.zeros((64, 64), np.uint8)
            mask[20:30, 20:30] = 1
            Image.new("L", (64, 64), 100).save(root / "images" / split / f"{index}.png")
            Image.fromarray(mask).save(root / "masks" / split / f"{index}.png")
    return root


def test_stop_request_ends_training_inside_the_epoch(tmp_path):
    from train_builtin import train_builtin
    calls, logs = [], []

    def should_stop():
        calls.append(1)
        return len(calls) > 3          # 첫 에폭의 몇 배치 뒤 중지

    train_builtin("unet_lite", _dataset(tmp_path / "data"), num_classes=2, input_size=32, in_channels=1, epochs=5,
                  batch_size=1, output_dir=tmp_path / "run", device="cpu", num_workers=0,
                  class_names=["background", "spot"], patch_training=True, patches_per_image=8,
                  class_report=True, should_stop=should_stop, log=lambda m: logs.append(str(m)))
    state = json.loads((tmp_path / "run" / "training.json").read_text(encoding="utf-8"))
    assert state["cancelled"] and state["completed_epochs"] == 0      # 첫 에폭을 끝내지 않고 멈춤
    assert any("에폭 1 도중에 멈춥니다" in line for line in logs)
    assert any("학습 장치: cpu (CPU)" in line for line in logs)
    assert any("에폭 1 학습: 첫 배치 완료" in line for line in logs)


def test_progress_reports_every_interval_with_time_left():
    from train_builtin import _BatchProgress
    now = [0.0]
    lines = []
    progress = _BatchProgress(lines.append, "에폭 1 학습", 100, "cuda", clock=lambda: now[0])
    for done in range(1, 101):
        now[0] = done * 1.0                              # 배치당 1초
        progress(done)
    assert lines[0].startswith("  에폭 1 학습: 첫 배치 완료 1.00초 (cuda)")
    assert any("에폭 1 학습 31/100 · 배치당 1.00초 · 남은 시간 약 1분" in line for line in lines)
    assert len(lines) == 1 + 3                           # 첫 배치 + 30초마다 (31, 61, 91)


def test_force_stop_kills_a_stuck_job_tree_and_records_cancelled(tmp_path):
    from core.desktop_jobs import force_stop
    script = ("import subprocess, sys, time; "
              "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); time.sleep(120)")
    process = subprocess.Popen([sys.executable, "-c", script])
    time.sleep(1.0)
    import psutil
    children = psutil.Process(process.pid).children(recursive=True)
    manager = SimpleNamespace(process=process, directory=lambda job_id: tmp_path)
    assert force_stop(manager, "job")
    assert process.wait(timeout=10) is not None
    assert all(not child.is_running() or child.status() == psutil.STATUS_ZOMBIE for child in children)
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["status"] == "cancelled"
    assert not force_stop(manager, "job")                # 이미 끝난 프로세스는 건드리지 않는다
