"""분할 패치 학습(원본 해상도에서 잘라 학습)과 타일 추론을 검증한다."""

from pathlib import Path
import random
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from dataset import PatchSegmentationDataset, create_segmentation_loaders, patch_origins  # noqa: E402


def _dataset(root, sizes=((700, 300), (500, 260)), defect=(600, 250, 3, 2)):
    """크기가 다른 이미지 2장. 첫 장에만 작은 결함(3×2px) 하나, 둘째 장은 마스크 없음(전부 배경)."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        for index, (w, h) in enumerate(sizes):
            pixels = np.full((h, w), 100, np.uint8)
            Image.fromarray(pixels, "L").save(root / "images" / split / f"{index}.png")
        x, y, dw, dh = defect
        mask = np.zeros((sizes[0][1], sizes[0][0]), np.uint8)
        mask[y:y + dh, x:x + dw] = 1
        Image.fromarray(mask, "L").save(root / "masks" / split / "0.png")
    return root


def test_patch_origins_cover_the_axis_and_stay_inside():
    assert patch_origins(1504, 512, 512) == [0, 512, 992]
    assert patch_origins(3500, 512, 384)[-1] == 3500 - 512
    assert patch_origins(300, 512, 512) == [0]


def test_training_patches_are_native_resolution_and_find_the_small_defect(tmp_path):
    root = _dataset(tmp_path)
    dataset = PatchSegmentationDataset(str(root / "images" / "train"), str(root / "masks" / "train"),
                                       patch_size=128, patches_per_image=40, in_channels=1, is_train=True,
                                       flip_prob=0, num_classes=2)
    assert len(dataset) == 80 and dataset.input_size == (128, 128)
    random.seed(0)
    hits = 0
    for index in range(40):  # 첫 장의 패치들
        image, mask = dataset[index]
        assert image.shape == (1, 128, 128) and mask.shape == (128, 128)
        assert set(mask.unique().tolist()) <= {0, 1}
        hits += int((mask == 1).sum() == 6)  # 3×2px 결함이 줄지 않고 통째로 들어 있음
    # 패치 절반은 결함 중심 → 700×300에서 6px 결함이 40개 중 대략 절반에 들어 있다 (무작위면 1~2개)
    assert 12 <= hits <= 32, hits
    # 마스크 없는 이미지는 전부 배경
    assert all(int(dataset[40 + i][1].sum()) == 0 for i in range(5))
    # 클래스 가중치는 원본 해상도 픽셀 수
    assert dataset.class_pixel_counts(2) == [700 * 300 + 500 * 260 - 6, 6]


def test_validation_tiles_cover_every_pixel_once_except_edge_overlap(tmp_path):
    root = _dataset(tmp_path)
    dataset = PatchSegmentationDataset(str(root / "images" / "val"), str(root / "masks" / "val"),
                                       patch_size=128, in_channels=1, is_train=False, num_classes=2)
    # 700×300 → 가로 6 × 세로 3, 500×260 → 4 × 3 (마지막 타일은 끝에 맞춤)
    assert len(dataset) == 6 * 3 + 4 * 3
    covered = np.zeros((300, 700), int)
    for index, x, y in dataset.tiles:
        if index == 0:
            covered[y:y + 128, x:x + 128] += 1
    assert covered.min() == 1
    defect = sum(int((dataset[i][1] == 1).sum()) for i in range(18))
    assert defect >= 6  # 결함 6px가 검증 타일에 빠짐없이 들어간다


def test_small_image_patch_is_padded_with_ignore(tmp_path):
    root = _dataset(tmp_path, sizes=((100, 80), (90, 70)), defect=(10, 10, 3, 3))
    dataset = PatchSegmentationDataset(str(root / "images" / "train"), str(root / "masks" / "train"),
                                       patch_size=128, patches_per_image=1, in_channels=1, flip_prob=0,
                                       num_classes=2)
    _image, mask = dataset[0]
    assert int((mask == 255).sum()) == 128 * 128 - 100 * 80  # 이미지 밖은 학습 제외


def test_loader_switches_to_patches_only_when_asked(tmp_path):
    root = _dataset(tmp_path)
    train, val = create_segmentation_loaders(str(root), input_size=(64, 64), batch_size=2, in_channels=1,
                                             num_classes=2, patch_size=128, patches_per_image=3)
    assert isinstance(train.dataset, PatchSegmentationDataset) and len(train.dataset) == 6
    assert next(iter(train))[0].shape[-2:] == (128, 128)
    plain, _ = create_segmentation_loaders(str(root), input_size=(64, 64), batch_size=2, in_channels=1,
                                           num_classes=2)
    assert not isinstance(plain.dataset, PatchSegmentationDataset)


class _LocalModel(torch.nn.Module):
    """완전 합성곱(3×3) 모델: 타일 경계에서 멀면 타일 추론과 전체 추론이 같아야 한다."""

    task, num_classes = "segment", 2

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def test_tiled_inference_matches_whole_image_including_seams():
    from core.inference_engine import InferenceOperations
    torch.manual_seed(0)
    engine = InferenceOperations()
    engine.model = _LocalModel().eval()
    image = np.random.default_rng(0).integers(0, 255, (300, 700), dtype=np.uint8)
    tiled = engine._tiled_probabilities(image, {"tile": 128, "overlap": 32}, [0.5], [0.25], "cpu")
    whole = torch.from_numpy(image).float().div(255).sub(.5).div(.25)[None, None]
    with torch.no_grad():
        expected = torch.softmax(engine.model(whole), dim=1)[0]
    assert tiled.shape == (2, 300, 700)
    # 3×3 합성곱은 타일 가장자리 1px에서만 틀린다. 가장자리 띠를 이웃 타일의 안쪽이 채우므로
    # 이음매를 포함해 이미지 전체가 한 번에 추론한 결과와 같다.
    assert torch.allclose(tiled, expected, atol=1e-3)
    assert engine._tile_count == len(patch_origins(300, 128, 96)) * len(patch_origins(700, 128, 96))


def test_checkpoints_record_the_tiling_contract():
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from checkpoint import make_checkpoint_metadata, tiling_contract
    contract = tiling_contract(512)
    assert contract == {"tile": 512, "overlap": 128, "blend": "center_weighted_softmax"} and tiling_contract(0) is None
    custom = make_checkpoint_metadata("segment", 2, ["background", "scratch"], 512, 1, tiling=contract)
    assert custom["tiling"] == contract and custom["preprocessing"]["tiling"] == contract
    with pytest.raises(ValueError):
        make_checkpoint_metadata("classify", 2, ["a", "b"], 512, 1, tiling=contract)
    model = build_builtin_model("unet_resnet18", 2, 1).eval()
    builtin = make_builtin_checkpoint("unet_resnet18", model, num_classes=2, input_size=512, in_channels=1,
                                      class_names=["background", "scratch"], tiling=contract)
    assert builtin["preprocessing"]["tiling"] == contract


def test_dataset_size_warning_uses_native_scale_for_patch_training():
    from types import SimpleNamespace
    from core.dataset_regions import training_scale
    project = SimpleNamespace(task="segment", training=SimpleNamespace(
        patch_training=True, input_size=512, patchcore_crop_enabled=False))
    assert training_scale(project, (3500, 1504)) == (1.0, 1.0)
    project.training.patch_training = False
    sx, sy = training_scale(project, (3500, 1504))
    assert sx == pytest.approx(512 / 3500) and sy == pytest.approx(512 / 1504)


def test_training_page_saves_patch_settings_for_segmentation(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from core.project import ProjectManager
    from widgets.training_widget import TrainingWidget

    QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("patch", "segment", str(tmp_path / "p"), ["background", "scratch"])
    project.model.model_id = "unet_resnet18"
    project.training.training_mode = "builtin_finetune"
    widget = TrainingWidget()
    widget.set_project(project)
    widget.show()
    widget.model_id_combo.setCurrentIndex(widget.model_id_combo.findData("unet_resnet18"))
    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("builtin_finetune"))
    assert widget.patch_check.isVisibleTo(widget) and not widget.patch_count_spin.isVisibleTo(widget)
    widget.input_size_spin.setValue(512)
    widget.patch_check.setChecked(True)
    widget.patch_count_spin.setValue(24)
    assert widget.patch_count_spin.isVisibleTo(widget) and "512×512" in widget.patch_info.text()
    widget._sync_config()
    assert project.training.patch_training is True and project.training.patches_per_image == 24
    # 저장·다시 열기에도 유지된다
    ProjectManager.save(project)
    reloaded = ProjectManager.load(ProjectManager.get_active_filepath(project))
    assert reloaded.training.patch_training is True and reloaded.training.patches_per_image == 24
    widget.close()
    widget.deleteLater()
