"""Exercise AMP from the built-in training settings through the actual epoch loop."""

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("cv2")


@pytest.mark.parametrize("task,model_id", [("classify", "resnet18"), ("segment", "unet_lite")])
@pytest.mark.parametrize("requested,effective", [(False, False), (True, False), (True, True)])
def test_builtin_worker_reports_requested_and_applied_amp(tmp_path, monkeypatch, task, model_id, requested, effective):
    import train_builtin as training
    from core.project import ProjectManager
    from webapp.worker import _train_builtin_project

    classes = ["background", "crack"] if task == "segment" else ["good", "crack"]
    project = ProjectManager.create_new("AMP", task, str(tmp_path / "project"), classes)
    project.model.model_id = model_id
    project.training.training_mode = "builtin_scratch"
    project.training.use_amp = requested
    device = "cuda:0" if effective or not requested else "cpu"

    def fake_train(selected_model, data_root, *, output_dir, use_amp, **kwargs):
        assert selected_model == model_id
        assert use_amp is requested
        assert kwargs["device"] == device
        checkpoint = output_dir / "best.pt"
        torch.save({"metric": 0.25, "epoch": 0}, checkpoint)
        (output_dir / "training.json").write_text(
            json.dumps({"best_epoch": 1, "training_config": {"use_amp": effective}}),
            encoding="utf-8",
        )
        return checkpoint

    monkeypatch.setattr(training, "train_builtin", fake_train)
    context = SimpleNamespace(emit=lambda *_: None, cancelled=lambda: False)
    record = _train_builtin_project(context, project, device, class_report=True)
    assert record.status == "completed"
    assert record.config_snapshot["training"]["use_amp"] is effective
    with Path(record.checkpoint_path).with_name("best_result.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        row = next(csv.DictReader(stream))
    assert row["hp_training_use_amp"] == str(requested)
    assert row["hp_effective_use_amp"] == str(effective)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_original_size_builtin_epoch_applies_amp_only_on_supported_cuda(tmp_path, monkeypatch, device):
    import train_builtin as training

    use_cuda = device.startswith("cuda")
    if use_cuda and (not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 7):
        pytest.skip("CUDA FP16-capable GPU required")

    data_root = tmp_path / "data"
    image = np.full((37, 69), 70, dtype=np.uint8)
    mask = np.zeros(image.shape, dtype=np.uint8)
    image[8:20, 30:37], mask[8:20, 30:37] = 200, 1
    for split in ("train", "val"):
        for folder, pixels in (("images", image), ("masks", mask)):
            directory = data_root / folder / split
            directory.mkdir(parents=True)
            Image.fromarray(pixels).save(directory / "sample.png")

    real_build = training.build_builtin_model
    observed = []

    def observe_build(*args, **kwargs):
        model = real_build(*args, **kwargs)
        first_conv = next(module for module in model.modules() if isinstance(module, torch.nn.Conv2d))

        def observe_forward(module, inputs, output):
            observed.append((module.training, output.dtype, output.device.type))

        first_conv.register_forward_hook(observe_forward)
        return model

    monkeypatch.setattr(training, "build_builtin_model", observe_build)
    run_dir = tmp_path / "run"
    options = dict(
        num_classes=2, class_names=["background", "crack"], in_channels=1,
        original_size_training=True, batch_size=1, device=device, use_amp=True,
        num_workers=0, horizontal_flip=0, rotation=0, color_jitter=0, log=lambda _: None,
    )
    best = training.train_builtin("unet_lite", data_root, epochs=1, output_dir=run_dir, **options)
    device_type = torch.device(device).type
    train_dtype = torch.float16 if use_cuda else torch.float32
    assert observed == [(True, train_dtype, device_type), (False, torch.float32, device_type)]
    state = json.loads((run_dir / "training.json").read_text(encoding="utf-8"))
    assert state["training_config"]["use_amp"] is use_cuda
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    assert checkpoint["original_size_region"] == [69, 37]
    assert bool(checkpoint.get("scaler_state_dict")) is use_cuda
    if use_cuda:
        last = run_dir / "last.pt"
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        checkpoint["scaler_state_dict"]["growth_interval"] = 17
        torch.save(checkpoint, last)
        resume_dir = tmp_path / "resumed"
        training.train_builtin("unet_lite", data_root, epochs=2, output_dir=resume_dir, resume=last, **options)
        resumed = torch.load(resume_dir / "last.pt", map_location="cpu", weights_only=True)
        assert resumed["epoch"] == 1
        assert resumed["scaler_state_dict"]["growth_interval"] == 17


@pytest.mark.parametrize("task", ["classify", "segment"])
def test_epoch_with_scaler_updates_weights_and_grows_scale(task):
    """Use PyTorch's CPU FP16 kernels to exercise scaled updates in CPU CI."""
    from segmentation_loss import SegmentationLoss
    from train_builtin import _classification_epoch, _segmentation_epoch

    try:
        scaler = torch.amp.GradScaler("cpu", init_scale=16, growth_interval=1)
        with torch.amp.autocast("cpu", dtype=torch.float16):
            probe = torch.nn.Conv2d(1, 2, 3, padding=1)(torch.ones((1, 1, 8, 8)))
        if probe.dtype != torch.float16:
            pytest.skip("CPU FP16 autocast unavailable")
        scaler.scale(probe.float().mean()).backward()
    except (AttributeError, RuntimeError, TypeError, NotImplementedError) as error:
        pytest.skip(f"CPU FP16 or GradScaler unavailable: {error}")

    conv = torch.nn.Conv2d(1, 2, 3, padding=1)
    inputs = torch.ones((2, 1, 8, 8))
    if task == "classify":
        model = torch.nn.Sequential(conv, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1))
        labels = torch.zeros((2,), dtype=torch.long)
        criterion = torch.nn.CrossEntropyLoss()
        epoch = _classification_epoch
    else:
        model = conv
        labels = torch.zeros((2, 8, 8), dtype=torch.long)
        criterion = SegmentationLoss(2, loss="ce_dice")
        epoch = _segmentation_epoch
    observed = []
    conv.register_forward_hook(lambda _module, _inputs, output: observed.append(output.dtype))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = conv.weight.detach().clone()
    loss, metric = epoch(model, [(inputs, labels)], criterion, optimizer, scaler=scaler)
    assert observed == [torch.float16]
    assert math.isfinite(loss) and math.isfinite(metric)
    assert not torch.equal(conv.weight.detach(), before)
    assert scaler.get_scale() == 32
