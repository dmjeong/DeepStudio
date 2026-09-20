import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from core.inference_engine import InferenceEngine
from core.inference_loading import load_inference_engine


class _Classifier:
    def predict(self, *args, **kwargs):
        return [SimpleNamespace(probs=SimpleNamespace(data=np.array([.1, .9], dtype=np.float32)))]


class _Detector:
    def predict(self, *args, **kwargs):
        boxes = SimpleNamespace(xyxyn=np.array([[.1, .2, .6, .8]], dtype=np.float32),
                                 conf=np.array([.75], dtype=np.float32), cls=np.array([1], dtype=np.float32))
        return [SimpleNamespace(boxes=boxes)]


def _engine(model, task):
    return InferenceEngine({"_upstream_model": model, "_upstream_task": task,
                            "_infer_device": "cpu", "_input_size": (224, 224),
                            "class_names": ["OK", "NG"]}, gradcam=False)


def test_upstream_classification_pt_result_uses_native_probabilities(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (12, 8)).save(image)
    result = _engine(_Classifier(), "classify").infer(str(image))
    assert result.status == "ok"
    assert result.summary == "NG 90.0%"
    assert result.details["runtime"] == "libreyolo-pytorch"


def test_upstream_detection_pt_result_converts_normalized_boxes_for_existing_preview(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (12, 8)).save(image)
    engine = _engine(_Detector(), "detect")
    result = engine.infer(str(image))
    assert result.status == "ok"
    detection = result.details["detections"][0]
    assert detection["class_id"] == 1
    assert detection["confidence"] == .75
    np.testing.assert_allclose(detection["bbox"], [.1, .2, .6, .8])
    assert engine._current_preview_rgb.shape == (8, 12, 3)


def test_upstream_pt_checkpoint_uses_public_libreyolo_factory(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "libreyolo.pt"
    torch.save({"model_family": "yolo9", "task": "detect", "nc": 2,
                "names": {0: "OK", 1: "NG"}, "imgsz": 640}, checkpoint_path)
    calls = []

    class _NativeModel:
        FAMILY = "yolo9"

    def factory(path, *, device):
        calls.append((path, device))
        return _NativeModel()

    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=factory))
    engine = load_inference_engine(checkpoint_path, device="cpu", gradcam=False)
    assert engine._upstream_model.FAMILY == "yolo9"
    assert engine._upstream_task == "detect"
    assert engine.class_names == ["OK", "NG"]
    assert engine._input_size == (640, 640)
    assert calls == [(str(checkpoint_path), "cpu")]
