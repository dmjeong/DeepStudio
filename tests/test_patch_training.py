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
    assert contract == {"tile": 512, "overlap": 128, "blend": "center_weighted_softmax", "scale": 1.0,
                        "interpolation": "INTER_AREA"}
    assert tiling_contract(0) is None and tiling_contract(512, .5)["scale"] == .5
    with pytest.raises(ValueError):
        tiling_contract(512, 1.5)
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
    # 배율을 쓰면 그 배율 (25%에서 1px 미세 크랙은 0.25px → 경고 대상)
    project.training.patch_scale = .25
    assert training_scale(project, (3500, 1504)) == (.25, .25)
    from core.dataset_regions import training_footprint, training_resolution_label
    assert training_footprint({"area": 300, "width": 1}, training_scale(project, (3500, 1504)))[1] < 1
    assert training_resolution_label(project) == "패치 배율 25%"
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
    assert widget.patch_count_spin.isVisibleTo(widget) and "512px 패치" in widget.patch_info.text()
    widget.pc_crop_check.setChecked(True)
    widget.pc_crop_width_spin.setValue(3500)
    widget.pc_crop_height_spin.setValue(1504)
    widget.patch_scale_combo.setCurrentIndex(widget.patch_scale_combo.findData(.5))
    # 배율을 고르면 줄인 크기와 추론 계산량을 바로 보여 준다
    assert "1750×752" in widget.patch_info.text() and "25%" in widget.patch_info.text()
    assert "⚠" not in widget.patch_info.text()
    # 25%면 875×376: 512 패치가 영역보다 커서 입력 크기를 줄이라고 알린다
    widget.patch_scale_combo.setCurrentIndex(widget.patch_scale_combo.findData(.25))
    assert "875×376" in widget.patch_info.text() and "352 이하" in widget.patch_info.text()
    widget.patch_scale_combo.setCurrentIndex(widget.patch_scale_combo.findData(.5))
    widget._sync_config()
    assert project.training.patch_training is True and project.training.patches_per_image == 24
    assert project.training.patch_scale == .5
    # 저장·다시 열기에도 유지된다
    ProjectManager.save(project)
    reloaded = ProjectManager.load(ProjectManager.get_active_filepath(project))
    assert reloaded.training.patch_training is True and reloaded.training.patches_per_image == 24
    assert reloaded.training.patch_scale == .5
    # 끄면(체크 해제) 이전과 같은 전체 이미지 학습
    widget.patch_check.setChecked(False)
    widget._sync_config()
    assert project.training.patch_training is False and not widget.patch_scale_combo.isVisibleTo(widget)
    widget.close()
    widget.deleteLater()



def test_whole_image_pass_matches_tiles_and_crops_the_padding():
    """완전 합성곱 모델은 판정 영역 전체를 한 번에 넣어도 된다 (32의 배수로 채웠다가 잘라 낸다)."""
    from core.inference_engine import InferenceOperations
    torch.manual_seed(0)
    engine = InferenceOperations()
    engine.model = _LocalModel().eval()
    image = np.random.default_rng(1).integers(0, 255, (301, 699), dtype=np.uint8)  # 32의 배수가 아님
    whole = engine._patch_probabilities(image, {"tile": 128, "overlap": 32}, [.5], [.25], "cpu")
    assert engine._patch_run == "whole" and engine._tile_count == 1 and whole.shape == (2, 301, 699)
    engine._patch_mode = "tiles"
    tiled = engine._patch_probabilities(image, {"tile": 128, "overlap": 32}, [.5], [.25], "cpu")
    assert engine._patch_run == "tiles" and engine._tile_count > 1
    # 오른쪽·아래 채움 경계(원래 이미지 끝)는 타일도 이미지 끝이라 같은 0 채움 → 전체가 같다
    assert torch.allclose(whole, tiled, atol=1e-3)


def test_out_of_memory_on_the_whole_image_falls_back_to_tiles():
    from core.inference_engine import InferenceOperations

    class Hungry(_LocalModel):
        def forward(self, x):
            if x.shape[-1] > 128:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return super().forward(x)

    engine = InferenceOperations()
    engine.model = Hungry().eval()
    image = np.zeros((200, 300), np.uint8)
    probabilities = engine._patch_probabilities(image, {"tile": 128, "overlap": 32}, [.5], [.25], "cpu")
    assert engine._patch_run == "tiles_after_oom" and probabilities.shape == (2, 200, 300)

    class Broken(_LocalModel):
        def forward(self, x):
            raise RuntimeError("shape mismatch")  # 메모리 문제가 아닌 오류는 숨기지 않는다

    engine.model = Broken().eval()
    with pytest.raises(RuntimeError, match="shape mismatch"):
        engine._patch_probabilities(image, {"tile": 128, "overlap": 32}, [.5], [.25], "cpu")


def test_scale_shrinks_training_images_but_keeps_thin_defects(tmp_path):
    root = _dataset(tmp_path, sizes=((800, 400), (600, 300)), defect=(100, 200, 300, 1))  # 1px 흠집
    dataset = PatchSegmentationDataset(str(root / "images" / "val"), str(root / "masks" / "val"),
                                       patch_size=128, in_channels=1, is_train=False, num_classes=2, scale=.5)
    image, mask, _foreground = dataset._load_native(0)
    assert image.shape == (200, 400) and mask.shape == (200, 400)
    # 1px 흠집을 절반으로 줄여도(nearest면 절반이 사라짐) 이어진 선으로 남는다
    assert int((mask == 1).sum()) >= 140 and dataset._sizes == [(400, 200), (300, 150)]
    with pytest.raises(ValueError):
        PatchSegmentationDataset(str(root / "images" / "val"), str(root / "masks" / "val"), patch_size=128,
                                 in_channels=1, is_train=False, num_classes=2, scale=0)


def _crack_columns(root, xs=(40, 49, 58, 67)):
    """400×200, 밝기 120 배경에 밝기 20(대비 100)의 1px 세로 크랙. x % 4 = 0·1·2·3 (4픽셀 묶음 안의 자리)."""
    for kind in ("images", "masks"):
        (root / kind / "val").mkdir(parents=True, exist_ok=True)
    pixels, mask = np.full((200, 400), 120, np.uint8), np.zeros((200, 400), np.uint8)
    for x in xs:
        pixels[:, x], mask[:, x] = 20, 1
    Image.fromarray(pixels, "L").save(root / "images" / "val" / "0.png")
    Image.fromarray(mask, "L").save(root / "masks" / "val" / "0.png")
    return root


def test_scaled_training_images_keep_1px_cracks_at_every_position(tmp_path):
    """배율 25%: 픽셀 평균 축소라 1px 크랙이 어느 자리에 있어도 영상에 남는다.

    선형 보간(9.33 첫 빌드)은 4픽셀 중 가운데 2개만 읽어, 자리 0·3의 크랙이 영상에서 사라졌다.
    마스크(작은 결함 보존 축소)에는 남으므로 '보이지 않는 곳을 결함이라 배우는' 학습이 됐다.
    """
    root = _crack_columns(tmp_path)
    xs = (40, 49, 58, 67)

    def contrast(interpolation):
        dataset = PatchSegmentationDataset(str(root / "images" / "val"), str(root / "masks" / "val"),
                                           patch_size=32, in_channels=1, is_train=False, num_classes=2,
                                           scale=.25, interpolation=interpolation)
        image, mask, _foreground = dataset._load_native(0)
        assert image.shape == mask.shape == (50, 100)
        assert all((mask[:, x // 4] == 1).all() for x in xs)  # 정답은 두 방식 모두 같은 열에 남는다
        return [120 - int(image[:, x // 4].min()) for x in xs]

    assert contrast("INTER_AREA") == [25, 25, 25, 25]
    assert contrast("INTER_LINEAR_EXACT") == [0, 50, 50, 0]
    with pytest.raises(ValueError, match="보간"):
        PatchSegmentationDataset(str(root / "images" / "val"), str(root / "masks" / "val"), patch_size=32,
                                 in_channels=1, is_train=False, num_classes=2, scale=.25, interpolation="NEAREST")


def test_scaled_model_input_uses_the_interpolation_it_was_trained_with(tmp_path):
    """새 모델은 픽셀 평균 축소로 추론한다. 보간이 기록되지 않은 9.33 첫 빌드 모델은 그때 학습한 선형 보간 그대로."""
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from checkpoint import tiling_contract
    from core.inference_loading import load_inference_engine
    image = tmp_path / "part.png"
    pixels = np.full((300, 700), 120, np.uint8)
    pixels[:, 400] = 20  # 4픽셀 묶음의 첫 자리: 선형 보간 25% 축소가 버리는 자리
    Image.fromarray(pixels, "L").save(image)
    model = build_builtin_model("unet_resnet18", 2, 1).eval()
    new = tiling_contract(128, .25)
    assert new["interpolation"] == "INTER_AREA"
    old = {key: value for key, value in new.items() if key != "interpolation"}
    gaps = {}
    for name, tiling in (("new", new), ("old", old)):
        checkpoint = make_builtin_checkpoint("unet_resnet18", model, num_classes=2, input_size=128, in_channels=1,
                                             class_names=["background", "crack"], tiling=tiling)
        torch.save(checkpoint, tmp_path / f"{name}.pt")
        engine = load_inference_engine(tmp_path / f"{name}.pt", segment_review={"min_area": 1, "min_score": 0,
                                                                              "split_dirs": []})

        def capture(inputs, name=name):
            # 줄인 판정 영역 175×75: 크랙 열(400 // 4 = 100)과 옆 열(98)의 입력 차이
            gaps[name] = float(inputs[0, 0, :75, 98].mean() - inputs[0, 0, :75, 100].mean())
            return torch.zeros(inputs.shape[0], 2, *inputs.shape[-2:])

        engine.model.forward = capture
        assert engine.infer(str(image)).status == "ok"
        # 이전 방식 모델은 다시 학습하라고 정보 줄에서 알린다
        assert ("이전 축소 방식" in engine.info) is (name == "old")
    assert gaps["new"] > 1e-3 and abs(gaps["old"]) < 1e-6


def test_scaled_patch_model_reports_defects_in_original_image_coordinates(tmp_path):
    """배율 50% 모델: 추론은 줄인 해상도, 결함 좌표·면적은 원본 이미지 기준."""
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from checkpoint import tiling_contract
    from core.inference_loading import load_inference_engine
    torch.manual_seed(0)
    model = build_builtin_model("unet_resnet18", 2, 1).eval()
    checkpoint = make_builtin_checkpoint("unet_resnet18", model, num_classes=2, input_size=128, in_channels=1,
                                         class_names=["background", "scratch"], tiling=tiling_contract(128, .5))
    torch.save(checkpoint, tmp_path / "scaled.pt")
    image = tmp_path / "part.png"
    Image.fromarray(np.full((300, 700), 120, np.uint8), "L").save(image)
    engine = load_inference_engine(tmp_path / "scaled.pt", segment_review={"min_area": 1, "min_score": 0,
                                                                           "split_dirs": []})

    def one_defect(inputs):  # 줄인 해상도(350×150)에서 (100..109, 40..44)만 결함
        out = torch.zeros(inputs.shape[0], 2, *inputs.shape[-2:])
        out[:, 0] = 5
        out[:, 1, 40:45, 100:110] = 9
        return out

    engine.model.forward = one_defect
    result = engine.infer(str(image))
    assert result.status == "ok", result.error
    assert result.details["tiling"]["run"] == "whole" and result.details["tiling"]["inference_size"] == [350, 150]
    defect, = result.details["defects"]
    # 줄인 좌표 ×2 = 원본 좌표 (nearest 블록 그대로)
    assert (defect["x"], defect["y"], defect["w"], defect["h"], defect["area"]) == (200, 80, 20, 10, 200)
    # 픽셀 분포·미리보기는 원본 크기(판정 영역)에서 센다
    assert sum(result.details["pixel_counts"].values()) == 300 * 700
    assert "배율 50%" in engine.info and "이미지 한 번에" in engine.info


def test_custom_csp_segmentation_gets_rotation_colour_and_patch_settings(tmp_path, monkeypatch):
    """Custom CSP 분할 로더: 회전·색상 증강(이전엔 좌우반전만)과 패치 학습 설정이 실제로 전달된다."""
    import dataset as dataset_module
    from core.project import ProjectManager
    from core.trainer import TrainWorker
    root = _dataset(tmp_path / "data")
    project = ProjectManager.create_new("csp", "segment", str(tmp_path / "p"), ["background", "scratch"])
    project.data.root = str(root)
    cfg = project.training
    cfg.augmentation.rotation, cfg.augmentation.color_jitter = 12.0, .15
    cfg.patch_training, cfg.input_size, cfg.patches_per_image, cfg.patch_scale = True, 128, 5, .5
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capturing loader arguments")

    monkeypatch.setattr(dataset_module, "create_segmentation_loaders", capture)
    worker = TrainWorker(project)
    try:
        worker._create_dataloaders("segment", project.data, cfg)
    except RuntimeError:
        pass
    assert seen["rotation"] == 12.0 and seen["color_jitter"] == .15
    assert (seen["patch_size"], seen["patches_per_image"], seen["patch_scale"]) == (128, 5, .5)
    metadata = worker._checkpoint_metadata()
    assert metadata["tiling"] == {"tile": 128, "overlap": 32, "blend": "center_weighted_softmax", "scale": .5,
                                  "interpolation": "INTER_AREA"}
