"""패치 학습의 '원본 그대로' 모드: 자르지 않고 판정 영역 전체(직사각형)를 샘플 하나로 학습 (데스크톱).

    원본 100×70 ─▶ 정규화 ─▶ 오른쪽·아래 0 채움 ─▶ 128×96 텐서 (32의 배수)   마스크 채움 = 255(학습 제외)
                                                  └─ 데스크톱 '한 번에' 추론·C++ whole_image와 같은 입력
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
torch = pytest.importorskip("torch")
SIZE = (100, 70)   # 가로 × 세로 (직사각형, 32의 배수 아님)


def _dataset(root, count=2):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            pixels = np.full((SIZE[1], SIZE[0], 3), 120, np.uint8)
            mask = np.zeros((SIZE[1], SIZE[0]), np.uint8)
            pixels[30:36, 60:70], mask[30:36, 60:70] = 230, 1
            Image.fromarray(pixels).save(root / "images" / split / f"{index}.png")
            Image.fromarray(mask).save(root / "masks" / split / f"{index}.png")
    return root


def test_whole_image_sample_is_the_full_rectangle_padded_like_inference(tmp_path):
    from dataset import PatchSegmentationDataset
    root = _dataset(tmp_path / "d")
    common = dict(patch_size=64, num_classes=2, in_channels=3, whole_image=True, flip_prob=0, rotation=0,
                  color_jitter=0)
    train = PatchSegmentationDataset(str(root / "images" / "train"), str(root / "masks" / "train"),
                                     is_train=True, patches_per_image=16, **common)
    val = PatchSegmentationDataset(str(root / "images" / "val"), str(root / "masks" / "val"), is_train=False,
                                   **common)
    assert len(train) == 2 and len(val) == 2                 # 이미지 1장 = 샘플 1개 (이미지당 패치 수 무시)
    image, mask = train[0]
    assert tuple(image.shape) == (3, 96, 128) and tuple(mask.shape) == (96, 128)
    assert not image[:, 70:].any() and not image[:, :, 100:].any()    # 정규화 뒤 0 = 추론 채움과 같다
    assert (mask[70:] == 255).all() and (mask[:, 100:] == 255).all()  # 채운 곳은 학습 제외
    assert int((mask == 1).sum()) == 60                                # 결함은 원본 해상도 그대로 (6×10)
    expected = (120 / 255 - 0.485) / 0.229
    assert abs(float(image[0, 0, 0]) - expected) < 1e-5               # 줄이거나 늘리지 않음


def test_whole_image_training_runs_and_keeps_the_inference_contract(tmp_path):
    from train_builtin import train_builtin
    from core.inference_loading import load_inference_engine
    root = _dataset(tmp_path / "d")
    logs = []
    best = train_builtin("unet_lite", root, num_classes=2, input_size=32, in_channels=3, epochs=1, batch_size=2,
                         output_dir=tmp_path / "run", device="cpu", num_workers=0,
                         class_names=["background", "spot"], patch_training=True, whole_image=True,
                         horizontal_flip=0, rotation=0, color_jitter=0, log=logs.append)
    assert any("원본 그대로 학습" in line and "128×96" in line for line in logs)
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert checkpoint["preprocessing"]["tiling"]["region"] == list(SIZE)   # 판정 영역 = 원본 크기
    engine = load_inference_engine(best, segment_review={"min_area": 1, "min_score": 0, "split_dirs": []})
    details = engine.infer(str(root / "images" / "val" / "0.png")).details
    assert sum(details["pixel_counts"].values()) == SIZE[0] * SIZE[1]


@pytest.mark.parametrize("model_id,patch,message", [("unet_lite", False, "패치 학습"),
                                                    ("dinov2_vits14_seg", True, "DINOv2")])
def test_whole_image_rejects_unsupported_settings(tmp_path, model_id, patch, message):
    from train_builtin import train_builtin
    with pytest.raises(ValueError, match=message):
        train_builtin(model_id, _dataset(tmp_path / "d"), num_classes=2, input_size=56, in_channels=3, epochs=1,
                      batch_size=1, output_dir=tmp_path / "run", device="cpu", num_workers=0,
                      class_names=["background", "spot"], patch_training=patch, whole_image=True,
                      log=lambda _m: None)


@pytest.mark.parametrize("desktop,expected", [(True, True), (False, False)])
def test_only_desktop_training_requests_use_whole_image(tmp_path, monkeypatch, desktop, expected):
    """웹 학습 요청은 프로젝트에 켜져 있어도 기존 패치 학습 그대로."""
    import webapp.worker as worker
    from core.project import ProjectManager
    project = ProjectManager.create_new("whole", "segment", str(tmp_path / "p"), ["background", "spot"])
    _dataset(Path(project.data.root))
    project.model.model_id = "unet_lite"
    project.training.patch_training, project.training.patch_whole_image = True, True
    seen = {}

    def capture(model_id, data_root, **kwargs):
        seen.update(kwargs)
        return tmp_path / "missing.pt"

    import train_builtin as module
    monkeypatch.setattr(module, "train_builtin", capture)
    monkeypatch.setattr(worker, "train_builtin", capture, raising=False)
    context = SimpleNamespace(emit=lambda *a: None, cancelled=lambda: False)
    worker._train_builtin_project(context, project, "cpu", class_report=desktop)
    assert seen["whole_image"] is expected


def test_training_page_whole_image_checkbox(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from core.project import ProjectManager
    from widgets.training_widget import TrainingWidget
    project = ProjectManager.create_new("whole-ui", "segment", str(tmp_path / "p"), ["background", "shift"])
    widget = TrainingWidget()
    widget.set_project(project)
    widget.model_id_combo.setCurrentIndex(widget.model_id_combo.findData("unet_lite"))
    widget.patch_check.setChecked(True)
    assert widget.patch_whole_check.isEnabled()
    widget.patch_whole_check.setChecked(True)
    assert not widget.patch_count_spin.isEnabled() and "자르지 않고" in widget.patch_info.text()
    widget._sync_config()
    assert project.training.patch_whole_image is True
    widget.patch_check.setChecked(False)               # 패치 학습을 끄면 저장값도 꺼진다
    widget._sync_config()
    assert project.training.patch_whole_image is False
    widget.deleteLater()
