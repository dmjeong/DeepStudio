"""Run the original-size mode through the desktop training workers."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")


def _project(tmp_path, model_id, *, crop):
    from core.project import ProjectManager

    project = ProjectManager.create_new(
        "Original", "segment", str(tmp_path / "project"), ["background", "crack"])
    project.model.model_id = model_id
    project.model.backbone_channels = [8, 16, 32, 64, 128]
    project.model.csp_depth = [1, 1, 1, 1]
    cfg = project.training
    cfg.training_mode = "builtin_scratch" if model_id else "custom"
    cfg.original_size_training = True
    cfg.input_size = 32
    cfg.epochs, cfg.batch_size, cfg.device = 1, 2, "cpu"
    if crop:
        cfg.patchcore_crop_enabled = True
        cfg.patchcore_crop_width, cfg.patchcore_crop_height = 85, 61
    root = Path(project.data.root)
    for split in ("train", "val"):
        for i in range(2):
            image = np.full((91, 117, 3), 40 + 60 * i, dtype=np.uint8)
            mask = np.zeros((91, 117), dtype=np.uint8)
            image[40:52, 55], mask[40:52, 55] = 240, 1
            Image.fromarray(image).save(root / "images" / split / f"{i}.png")
            Image.fromarray(mask).save(root / "masks" / split / f"{i}.png")
    return project


@pytest.mark.parametrize("model_id,crop,expected_region,expected_input", [
    ("unet_lite", False, (117, 91), (96, 128)),
    ("", True, (85, 61), (64, 96)),
])
def test_desktop_worker_keeps_original_region(tmp_path, model_id, crop, expected_region, expected_input):
    from checkpoint import checkpoint_original_size_region
    from core.inference_loading import load_inference_engine
    from core.training_modes import validate_training_options

    project = _project(tmp_path, model_id, crop=crop)
    validate_training_options(project)
    if model_id:
        from webapp.worker import _train_builtin_project
        context = SimpleNamespace(emit=lambda *_: None, cancelled=lambda: False)
        record = _train_builtin_project(context, project, "cpu", class_report=True)
    else:
        from core.trainer import TrainWorker
        worker = TrainWorker(project)
        errors = []
        worker.signals.training_error.connect(errors.append)
        worker.debug_num_workers = 0
        worker.run()
        assert not errors, errors
        record = project.runs[-1]
    assert record.status == "completed"
    saved = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint_original_size_region(saved) == expected_region
    assert tuple(saved["preprocessing"]["input_size"]) == expected_input
    assert "tiling" not in saved
    assert project.training.input_size == 32
    if crop:
        assert saved["center_crop"] == {"width": 85, "height": 61}

    image_path = Path(project.data.root) / "images" / "val" / "0.png"
    engine = load_inference_engine(record.checkpoint_path)
    seen = []
    hook = engine.model.register_forward_pre_hook(lambda _module, args: seen.append(args[0].shape))
    try:
        result = engine.infer(str(image_path))
    finally:
        hook.remove()
    assert result.status == "ok", result.error
    assert [tuple(shape[-2:]) for shape in seen] == [expected_input]
    assert sum(result.details["pixel_counts"].values()) == expected_region[0] * expected_region[1]
    if crop:
        assert result.details["crop_box"] == [16, 15, 101, 76]


def test_original_size_cannot_be_combined_with_patches_or_used_for_sam2(tmp_path):
    from core.training_modes import validate_training_options

    project = _project(tmp_path, "unet_lite", crop=False)
    project.training.patch_training = True
    with pytest.raises(ValueError, match="패치 학습은 함께"):
        validate_training_options(project)
    project.training.patch_training = False
    project.model.model_id = "sam2_hiera_tiny"
    project.training.training_mode = "sam2_finetune"
    with pytest.raises(ValueError, match="원본 크기 학습을 지원하지"):
        validate_training_options(project)


def test_resume_requires_the_same_original_region_even_with_same_padding(tmp_path):
    from train_builtin import train_builtin

    project = _project(tmp_path, "unet_lite", crop=False)
    root = Path(project.data.root)
    kwargs = dict(num_classes=2, class_names=["background", "crack"],
                  epochs=1, batch_size=2, original_size_training=True, log=lambda _: None)
    train_builtin("unet_lite", root, output_dir=tmp_path / "first", **kwargs)
    paths = [path for kind in ("images", "masks") for split in ("train", "val")
             for path in (root / kind / split).glob("*.png")]
    for path in paths:
        with Image.open(path) as image:
            cropped = image.crop((0, 0, 116, 91))
        cropped.save(path)
    kwargs["epochs"] = 2
    with pytest.raises(ValueError, match="판정 영역을 유지"):
        train_builtin("unet_lite", root, output_dir=tmp_path / "second",
                      resume=tmp_path / "first" / "last.pt", **kwargs)
