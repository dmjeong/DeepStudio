"""Real native training/export contracts; no downloaded weights or accuracy claims."""
import json
import os
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo_classification import FullFrameTransform, MEAN, STD, preprocessing_contract


def edge_image():
    pixels = np.full((121, 337, 3), 80, np.uint8)
    pixels[:12] = (255, 0, 0)
    pixels[-12:] = (0, 255, 0)
    pixels[:, :24] = (0, 0, 255)
    pixels[:, -24:] = (255, 255, 0)
    return pixels


def test_all_four_edges_survive_train_val_and_spawn(tmp_path):
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    pixels = edge_image()
    transform = FullFrameTransform(224)
    expected = transform(pixels)
    restored = (expected.numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)) * 255
    np.testing.assert_allclose(restored[0, 112], (255, 0, 0), atol=1e-4)
    np.testing.assert_allclose(restored[-1, 112], (0, 255, 0), atol=1e-4)
    np.testing.assert_allclose(restored[112, 0], (0, 0, 255), atol=1e-4)
    np.testing.assert_allclose(restored[112, -1], (255, 255, 0), atol=1e-4)
    for _ in range(10):
        actual = FullFrameTransform(224, augment=True)(pixels)
        assert torch.equal(actual, expected) or torch.equal(actual, expected.flip(-1))
    assert torch.equal(pickle.loads(pickle.dumps(transform))(pixels), expected)
    folder = tmp_path / "class_a"
    folder.mkdir()
    Image.fromarray(pixels).save(folder / "edge.png")
    loader = DataLoader(ImageFolder(tmp_path, transform=transform), batch_size=1,
                        num_workers=1, multiprocessing_context="spawn")
    batches = list(loader)
    assert torch.equal(batches[0][0][0], expected)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    pytest.importorskip("libreyolo")
    from upstream_models import build_upstream_model
    from libreyolo_training import train_classifier
    root = tmp_path_factory.mktemp("libre-full-frame")
    data = root / "data"
    for split, count in (("train", 4), ("val", 2)):
        for label in ("a", "b"):
            folder = data / split / label
            folder.mkdir(parents=True)
            for index in range(count):
                pixels = edge_image()
                if label == "b":
                    pixels = pixels[:, ::-1]
                Image.fromarray(pixels).save(folder / f"{index}.png")
    with torch.random.fork_rng():
        torch.manual_seed(11)
        model = build_upstream_model("libreyolo_classify_mobilenetv4_small", num_classes=2,
                                     device="cpu", pretrained=False)
        settings = dict(data=str(data), epochs=1, batch=4, imgsz=224, workers=0,
                        device="cpu", project=str(root), name="train", exist_ok=True,
                        amp=False, val_loss=True, eval_interval=1, seed=11)
        result = train_classifier(model, **settings)
    return root, model, result, settings


def test_real_training_validation_resume_and_saved_contract(trained):
    from libreyolo_training import train_classifier, FullFrameValidator, ValidationConfig
    from upstream_models import load_upstream_checkpoint
    root, wrapper, result, settings = trained
    best = Path(result["best_checkpoint"])
    assert best.is_file()
    last = best.with_name("last.pt")
    for path in (best, last):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["studio_preprocessing"] == preprocessing_contract(224)
        assert saved["epoch"] == 0
    validator = FullFrameValidator(wrapper, ValidationConfig(data=settings["data"],
        imgsz=224, device="cpu", num_workers=0, split="val"))
    loader = validator._setup_dataloader()
    assert isinstance(loader.dataset._impl.transform, FullFrameTransform)
    assert not loader.dataset._impl.transform.augment
    for _ in range(3):
        actual = loader.dataset[0][0]
        assert torch.equal(actual, FullFrameTransform(224)(edge_image()))
    resumed = load_upstream_checkpoint(last, device="cpu")
    next_result = train_classifier(resumed, resume=True, **{**settings, "epochs": 2, "name": "resume"})
    next_last = Path(next_result["last_checkpoint"])
    assert torch.load(next_last, map_location="cpu", weights_only=False)["epoch"] == 1


def test_quantization_calibration_uses_same_full_input(trained):
    from libreyolo.export.calibration import CalibrationDataLoader
    root, wrapper, _, _ = trained
    reader = object.__new__(CalibrationDataLoader)
    reader._preprocess_fn = wrapper._get_preprocess_numpy()
    reader.imgsz = (224, 224)
    tensor = reader._preprocess(root / "data/val/a/0.png")
    assert isinstance(tensor, np.ndarray)
    np.testing.assert_array_equal(tensor, FullFrameTransform(224)(edge_image()).numpy())


def test_app_training_entry_keeps_explicit_roi_in_best_and_last(trained, monkeypatch):
    from upstream_models import train_upstream_model
    from libreyolo_training import train_classifier
    from center_crop import checkpoint_center_crop
    root, _, _, _ = trained
    crop = {"width": 201, "height": 95}
    def run(wrapper, **settings):
        # Keep CI independent of CPU count. Production creates this same trainer.
        return train_classifier(wrapper, **settings, workers=0)
    monkeypatch.setattr("libreyolo_training.train_classifier", run)
    events = []
    result = train_upstream_model("libreyolo_classify_mobilenetv4_small",
        data_root=root / "data", class_names=["a", "b"], output_dir=root / "roi",
        epochs=1, batch_size=4, learning_rate=.001, device="cpu", weights=None,
        resume=False, use_amp=False, patience=10, emit=events.append, center_crop=crop)
    for key in ("best_checkpoint", "last_checkpoint"):
        saved = torch.load(result[key], map_location="cpu", weights_only=False)
        assert checkpoint_center_crop(saved) == crop
        assert saved["studio_preprocessing"] == preprocessing_contract(224)
    assert any(e.get("policy") == "libreyolo_full_frame_v1" for e in events)


def test_old_training_resume_rejected_but_weights_load(trained):
    from libreyolo_training import FullFrameTrainer
    from upstream_models import load_upstream_checkpoint
    root, wrapper, result, settings = trained
    saved = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    saved.pop("studio_preprocessing")
    legacy = root / "legacy.pt"
    torch.save(saved, legacy)
    loaded = load_upstream_checkpoint(legacy, device="cpu")
    trainer = FullFrameTrainer(model=loaded.model, wrapper_model=loaded, size=loaded.size,
                               num_classes=2, **{**settings, "name": "old-resume"})
    with pytest.raises(ValueError, match="기존 가중치로 추가 학습"):
        trainer.resume(legacy)
    for key, value in wrapper.model.state_dict().items():
        assert torch.equal(value.cpu(), loaded.model.state_dict()[key].cpu()), key
    with pytest.raises(ValueError, match="계약"):
        preprocessing_contract(224, {**saved.get("studio_preprocessing", {}), "policy": "unknown"})


@pytest.mark.parametrize("crop", [None, {"width": 201, "height": 95}])
def test_actual_export_image_parity_and_cpp_when_available(trained, tmp_path, crop):
    import onnxruntime as ort
    from export_onnx import export_checkpoint
    from core.inference_loading import load_inference_engine
    root, _, result, _ = trained
    saved = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    saved["center_crop"] = crop
    checkpoint = tmp_path / "model.pt"
    torch.save(saved, checkpoint)
    pixels = edge_image()
    image = tmp_path / "edge.png"
    Image.fromarray(pixels).save(image)
    engine = load_inference_engine(checkpoint, gradcam=True)
    prediction = engine.infer(str(image))
    assert prediction.status == "ok", prediction.error
    assert prediction.gradcam_status == "completed", engine.info
    box = prediction.details["crop_box"]
    assert box == ([68, 13, 269, 108] if crop else [0, 0, 337, 121])
    expected_mask = np.zeros(pixels.shape[:2], bool)
    expected_mask[box[1]:box[3], box[0]:box[2]] = True
    np.testing.assert_array_equal(engine._heatmap_cache["valid_mask"], expected_mask)
    entire = load_inference_engine(checkpoint, input_region={"mode": "full"})
    assert entire.infer(str(image)).details["crop_box"] == [0, 0, 337, 121]
    output = export_checkpoint(checkpoint, tmp_path / "model.onnx", log=lambda _: None)
    config = json.loads(Path(output["config_path"]).read_text())
    assert config["preprocessing"]["policy"] == "libreyolo_full_frame_v1"
    assert config["preprocessing"]["center_crop"] == crop
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(output["output_path"], options, providers=["CPUExecutionProvider"])
    x = FullFrameTransform(224)(pixels[box[1]:box[3], box[0]:box[2]]).unsqueeze(0)
    logits = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
    np.testing.assert_allclose(probabilities, prediction.details["probabilities"], atol=1e-5, rtol=1e-4)
    probe = os.environ.get("CLASSIFY_PROBE")
    if probe:
        actual = subprocess.run([probe, output["config_path"], str(image)], capture_output=True,
                                text=True, timeout=60, check=True)
        native = json.loads(actual.stdout)
        np.testing.assert_allclose(native["probabilities"], probabilities, atol=1e-5, rtol=1e-4)
        assert native["class_id"] == int(probabilities.argmax())
