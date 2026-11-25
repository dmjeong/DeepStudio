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
    ("unet_lite", True, (85, 61), (64, 96)),
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
        if model_id:
            effective = record.config_snapshot["best_selection"]["hyperparameters"]["effective"]
            assert effective["center_crop"] == saved["center_crop"]

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


def test_original_size_excludes_patches_and_checks_sam2_crop(tmp_path):
    from core.training_modes import validate_training_options

    project = _project(tmp_path, "unet_lite", crop=False)
    project.training.patch_training = True
    with pytest.raises(ValueError, match="패치 학습은 함께"):
        validate_training_options(project)
    project.training.patch_training = False
    project.model.model_id = "sam2_hiera_tiny"
    project.training.training_mode = "sam2_finetune"
    validate_training_options(project)
    project.training.patchcore_crop_enabled = True
    with pytest.raises(ValueError, match="중앙 크롭"):
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


@pytest.mark.parametrize(("platform", "device", "original", "workers"), [
    ("win32", "cuda", True, 0),
    ("win32", "cuda", False, 8),
    ("linux", "cuda", True, 8),
    ("win32", "cpu", True, 0),
])
def test_original_size_auto_loader_workers(monkeypatch, platform, device, original, workers):
    import train_builtin
    from types import SimpleNamespace
    monkeypatch.setattr(train_builtin, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(train_builtin.os, "cpu_count", lambda: 16)
    assert train_builtin._default_num_workers(device, original) == workers


def test_custom_reconstruction_native_train_and_onnx(tmp_path):
    import json
    import onnxruntime as ort
    from core.project import ProjectManager
    from core.trainer import TrainWorker
    from core.inference_loading import load_inference_engine
    from export_onnx import export_checkpoint

    project = ProjectManager.create_new("native", "anomaly", str(tmp_path / "project"))
    project.model.model_id = ""
    project.training.anomaly_method = "reconstruction"
    cfg = project.training
    cfg.original_size_training, cfg.training_mode = True, "custom"
    cfg.input_size, cfg.epochs, cfg.batch_size, cfg.in_channels, cfg.device = 32, 1, 2, 1, "cpu"
    project.model.backbone_channels = [8, 16, 32, 64, 96]
    project.model.csp_depth = [1] * 4
    folder = Path(project.data.root) / "train" / "good"
    for i in range(3):
        pixels = np.random.default_rng(i).integers(0, 256, (65, 95), np.uint8)
        Image.fromarray(pixels).save(folder / f"{i}.png")
    worker = TrainWorker(project)
    worker.debug_num_workers = 0
    errors = []
    worker.signals.training_error.connect(errors.append)
    worker.run()
    assert not errors, errors
    checkpoint = project.runs[-1].checkpoint_path
    assert "_original_" in str(checkpoint)   # 실행 폴더 이름: 채운 크기(96x65)가 아니라 원본 크기 학습 표시
    engine = load_inference_engine(checkpoint, runtime="pytorch")
    assert engine.infer(str(folder / "0.png")).status in {"ok", "uncalibrated"}
    wrong = tmp_path / "wrong.png"
    Image.fromarray(pixels[:, :-1]).save(wrong)
    assert "95×65" in engine.infer(str(wrong)).error
    output = tmp_path / "native-anomaly.onnx"
    export_checkpoint(checkpoint, output, verify=True)
    config = json.loads(output.with_suffix(".json").read_text())
    assert config["preprocessing"]["original_size_region"] == [95, 65]
    sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    reconstructed, = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 1, 65, 95), np.float32)})
    assert reconstructed.shape == (1, 1, 65, 95)


def test_custom_segmentation_cli_native_checkpoint(tmp_path, monkeypatch):
    import train_segmentation
    from checkpoint import checkpoint_original_size_region
    from export_onnx import load_custom_model, resolve_checkpoint_spec

    project = _project(tmp_path, "", crop=False)
    args = ["train_segmentation.py", "--data_root", project.data.root, "--num_classes", "2",
            "--epochs", "1", "--batch_size", "2", "--input_size", "32",
            "--original_size_training", "--save_dir", str(tmp_path), "--exp_name", "native"]
    monkeypatch.setattr(sys, "argv", [*args, "--patch_training"])
    with pytest.raises(SystemExit) as conflict:
        train_segmentation.parse_args()
    assert conflict.value.code == 2
    monkeypatch.setattr(sys, "argv", args)
    train_segmentation.main()
    for name in ("best", "last"):
        saved = torch.load(tmp_path / "native" / (name + ".pt"), weights_only=True)
        assert checkpoint_original_size_region(saved) == (117, 91)
        assert saved["input_size"] == [96, 128]
        spec = resolve_checkpoint_spec(saved)
        with torch.inference_mode():
            out = load_custom_model(saved, spec)(torch.zeros(1, 3, 96, 128))
        assert out.shape == (1, 2, 96, 128)
