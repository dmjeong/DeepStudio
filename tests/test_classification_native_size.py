"""Native classifier pixels, model checkpoints and deployment geometry."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dataset import create_classification_loaders, get_normalize_params
from export_onnx import export_checkpoint


def images(root, *, explicit_validation, shape=(65, 95)):
    pixels = np.arange(shape[0] * shape[1], dtype=np.uint8).reshape(shape)
    for split in (("train", "val") if explicit_validation else ("train",)):
        for label in ("OK", "crack"):
            path = root / split / label
            path.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(path / "sample.png")
    return pixels


@pytest.mark.parametrize("explicit_validation", [True, False])
def test_classifier_native_preserves_exact_pixels_and_validates_all_files(tmp_path, explicit_validation):
    pixels = images(tmp_path, explicit_validation=explicit_validation)
    def load():
        return create_classification_loaders(str(tmp_path), input_size=(32, 32), original_size=True,
                                             in_channels=1, flip_prob=0, rotation=0, color_jitter=0)
    train, val, _ = load()
    mean, std = get_normalize_params(1)
    expected = (torch.from_numpy(pixels.copy()).float() / 255 - mean[0]) / std[0]
    for loader in (train, val):
        assert loader.dataset.original_size == (65, 95)
        tensor, _ = next(iter(loader))
        torch.testing.assert_close(tensor[0, 0], expected, atol=0, rtol=0)
    split = "val" if explicit_validation else "train"
    Image.fromarray(pixels[:, :-1]).save(tmp_path / split / "crack" / "sample.png")
    with pytest.raises(ValueError, match="크기가 모두 같아야"):
        load()


@pytest.mark.parametrize("model_id", ["resnet18", "efficientnet_b0"])
def test_classification_native_train_reload_infer_export(tmp_path, model_id):
    import onnxruntime as ort
    from core.project import ProjectManager
    from core.inference_loading import load_inference_engine

    project = ProjectManager.create_new("native", "classify", str(tmp_path / "project"), ["OK", "crack"])
    root = Path(project.data.root)
    pixels = images(root, explicit_validation=True)
    if model_id == "resnet18":
        from train_builtin import train_builtin
        saved_file = train_builtin(model_id, root, input_size=32, num_classes=2, in_channels=1,
                                   num_workers=0, batch_size=2, epochs=1, optimizer_name="sgd",
                                   original_size_training=True, output_dir=tmp_path / "run", log=lambda _: None)
    else:
        from core.efficientnet_trainer import EfficientNetTrainWorker
        project.model.model_id = model_id
        cfg = project.training
        cfg.original_size_training = True
        cfg.efficientnet_model = model_id
        cfg.training_mode = "efficientnet_scratch"
        cfg.input_size, cfg.epochs, cfg.batch_size, cfg.in_channels, cfg.device = 32, 1, 2, 1, "cpu"
        worker = EfficientNetTrainWorker(project)
        worker.debug_num_workers = 0
        errors = []
        worker.signals.training_error.connect(errors.append)
        worker.run()
        assert not errors, errors
        saved_file = project.runs[-1].checkpoint_path
    ck = torch.load(saved_file, map_location="cpu", weights_only=model_id != "efficientnet_b0")
    assert list(ck["input_size"]) == [65, 95]
    assert ck["preprocessing"]["original_size_region"] == ck["original_size_region"] == [95, 65]
    engine = load_inference_engine(saved_file, runtime="pytorch")
    assert engine.infer(str(root / "val" / "crack" / "sample.png")).status == "ok"
    wrong = tmp_path / "wrong.png"
    Image.fromarray(pixels[:, :-1]).save(wrong)
    assert "95×65" in engine.infer(str(wrong)).error
    dest = tmp_path / "native.onnx"
    export_checkpoint(saved_file, dest, verify=True)
    config = json.loads(dest.with_suffix(".json").read_text())
    assert (config["input_width"], config["input_height"]) == (95, 65)
    assert config["preprocessing"]["original_size_region"] == [95, 65]
    session = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[-2:] == [65, 95]
    if model_id == "efficientnet_b0":
        from onnx_classifier import ImageClassifier
        processor = ImageClassifier(dest.with_suffix(".json"))
        assert processor.preprocess(pixels).shape == (1, 1, 65, 95)
        with pytest.raises(ValueError, match="95×65"):
            processor.preprocess(pixels[:, :-1])


def test_precision_uses_native_pixels_after_center_crop_and_rejects_wrong_shape():
    from precision_benchmark import _classification_tensors

    pixels = np.arange(69 * 99, dtype=np.uint8).reshape(69, 99)
    prep = {"original_size_region": [95, 65], "center_crop": {"width": 95, "height": 65}}
    cfg = {"input_channels": 1, "input_width": 95, "input_height": 65,
           "normalize_mean": [0], "normalize_std": [1], "preprocessing": prep}
    tensor, = _classification_tensors(cfg, pixels)
    np.testing.assert_array_equal(tensor[0, 0], pixels[2:67, 2:97].astype(np.float32) / 255)
    prep.pop("center_crop")
    with pytest.raises(ValueError, match="95×65"):
        list(_classification_tensors(cfg, pixels))
