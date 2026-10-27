"""Real built-in architectures through the encrypted public export API.

No pretrained download: encryption parity is a byte/storage property, not a
claim about training accuracy. Upstream/SAM2 contract fixtures are separate.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

import export_onnx
from model_crypto import create_key, load_package, read_key

# Keep model contract tests offline. ORT 1.29's macOS telemetry HTTP worker can
# outlive shutdown and abort in DebugEventSource::DispatchEvent; it is not part
# of inference/export and must not upload events from this test process.
ort.disable_telemetry_events()


@pytest.mark.parametrize("model_id,channels", [
    ("efficientnet_b0", 1), ("efficientnet_b0", 3),
    ("efficientnet_b1", 1), ("efficientnet_b1", 3),
    ("resnet18", 1), ("resnet50", 3), ("convnext_v1_tiny", 3),
    ("deeplabv3plus_resnet34", 3), ("unet_resnet18", 1),
])
def test_real_architecture_encrypted_export(tmp_path, monkeypatch, model_id, channels):
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(73)
        size = 64
        if model_id.startswith("efficientnet"):
            from efficientnet import EfficientNet
            from checkpoint import make_checkpoint_metadata
            model = EfficientNet(model_id, 2, channels).eval()
            ckpt = make_checkpoint_metadata("classify", 2, ["정상", "불량"], (size, size), channels)
            ckpt.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
        else:
            from builtin_models import build_builtin_model, make_builtin_checkpoint
            model = build_builtin_model(model_id, 2, channels).eval()
            ckpt = make_builtin_checkpoint(model_id, model, num_classes=2,
                                          input_size=size, in_channels=channels, class_names=["정상", "불량"])
        checkpoint = tmp_path / "model.pt"
        torch.save(ckpt, checkpoint)
        key_path = create_key(tmp_path / "private.key")
        # Capture the actual unencrypted output inside staging, then let the
        # production wrapper encrypt, verify and remove its temporary files.
        real_export = export_onnx._export_checkpoint
        reference = {}
        def capture(*args, **kwargs):
            result = real_export(*args, **kwargs)
            reference["model"] = Path(result["output_path"]).read_bytes()
            reference["config"] = json.loads(Path(result["config_path"]).read_text())
            return result
        monkeypatch.setattr(export_onnx, "_export_checkpoint", capture)
        result = export_onnx.export_checkpoint(checkpoint, tmp_path / "model.dvsenc",
                    encryption_key_path=key_path, dynamic_batch=True, log=lambda _: None)
        config, graphs = load_package(result["output_path"], read_key(key_path))
        assert config == reference["config"]
        restored = graphs[config["model_path"]]
        assert restored == reference["model"]
        assert result["encrypted"] and result["verification"] == "passed"
        assert not list(tmp_path.glob("*.onnx")) and not list(tmp_path.glob("*.json"))
        assert not list(tmp_path.glob(".encrypted-export-*"))
        from onnx_session import cpu_session_options, deployment_session_settings
        settings = deployment_session_settings(config)
        sessions = [ort.InferenceSession(data, cpu_session_options(**settings), providers=["CPUExecutionProvider"])
                    for data in (reference["model"], restored)]
        for batch in (1, 2):
            x = np.random.default_rng(7).normal(size=(batch, channels, size, size)).astype(np.float32)
            outputs = [s.run(None, {s.get_inputs()[0].name: x}) for s in sessions]
            for expected, actual in zip(*outputs):
                np.testing.assert_array_equal(expected, actual)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("family", ["mobilenetv4", "yolo9", "rtdetrv4"])
def test_encrypted_libreyolo_uses_upstream_names_and_logits(tmp_path, monkeypatch, family):
    class Tiny(torch.nn.Module):
        def forward(self, images):
            value = images.mean((1, 2, 3))
            if family == "yolo9":
                return torch.stack((value, value, value, value, value, -value), 1).unsqueeze(2)
            if family == "rtdetrv4":
                return {"pred_logits": torch.stack((value, -value), 1).unsqueeze(1),
                        "pred_boxes": torch.stack((value, value, value, value), 1).unsqueeze(1)}
            return torch.stack((value, -value), 1)
    class FakeLibreYOLOClassifier:
        FAMILY = family
        model = Tiny().eval()
        def export(self, format, *, output_path, imgsz, opset, dynamic, simplify, device):
            torch.onnx.export(self.model, torch.zeros(1, 3, 8, 8), output_path,
                              input_names=["images"], output_names=["pred_logits", "pred_boxes"] if family == "rtdetrv4" else ["output"], opset_version=opset,
                              dynamo=False)
            return output_path
    import upstream_models
    checkpoint = {"model_family": family, "task": "classify" if family == "mobilenetv4" else "detect", "nc": 2,
                  "names": {0: "ok", 1: "ng"}, "imgsz": 8}
    source = tmp_path / "libre.pt"; torch.save(checkpoint, source)
    monkeypatch.setattr(upstream_models, "load_upstream_checkpoint", lambda *a, **k: FakeLibreYOLOClassifier())
    key = create_key(tmp_path / "secret.key")
    result = export_onnx.export_checkpoint(source, tmp_path / "libre.dvsenc", encryption_key_path=key, log=lambda _: None)
    doc, graphs = load_package(result["output_path"], read_key(key))
    s = ort.InferenceSession(graphs[doc["model_path"]], providers=["CPUExecutionProvider"])
    assert doc["input_name"] == s.get_inputs()[0].name == "images"
    if family == "mobilenetv4":
        assert doc["postprocessing"]["output"] == "logits"
    else:
        assert doc["task"] == "detect" and result["verification"] == "passed"


class TinyDetector(torch.nn.Module):
    def forward(self, images):
        v = images.mean((1, 2, 3))
        return torch.stack((v, v, v, v), 1).unsqueeze(1), torch.stack((v, -v), 1).unsqueeze(1)


class TinyEncoder(torch.nn.Module):
    def forward(self, images):
        return images.mean(1, keepdim=True)


class TinyDecoder(torch.nn.Module):
    def forward(self, image_embeddings, point_coords, point_labels, mask_input, has_mask_input, orig_im_size):
        bias = point_coords.mean() + point_labels.float().mean() * .001 + mask_input.mean() * .001
        bias = bias + has_mask_input.mean() * .001 + orig_im_size.mean() * .000001
        masks = image_embeddings[:, :, :2, :2] + bias
        return masks, masks.mean((1, 2, 3)).reshape(-1, 1)


@pytest.mark.parametrize("variant", ["Small", "Medium", "Large"])
def test_redetr_encrypted_multiple_outputs(tmp_path, variant):
    checkpoint = {"type": "redetr_v4", "backend": "redetr_v4", "task": "detect", "variant": variant,
                  "num_classes": 2, "in_channels": 3, "input_size": [8, 8], "class_names": ["a", "b"],
                  "model_config": {"boxes_format": "normalized_cxcywh", "score_activation": "softmax"},
                  "model": TinyDetector().eval()}
    source = tmp_path / "detector.pt"; torch.save(checkpoint, source)
    key = create_key(tmp_path / "key")
    result = export_onnx.export_checkpoint(source, tmp_path / "model.dvsenc", encryption_key_path=key, log=lambda _: None)
    doc, graphs = load_package(result["output_path"], read_key(key))
    s = ort.InferenceSession(graphs[doc["model_path"]], providers=["CPUExecutionProvider"])
    assert [o.name for o in s.get_outputs()] == doc["output_names"] == ["pred_boxes", "pred_logits"]
    assert doc["variant"] == variant and result["verification"] == "passed"
    assert len(s.run(None, {"input_image": np.ones((1, 3, 8, 8), np.float32)})) == 2


@pytest.mark.parametrize("variant", ["Hiera Tiny", "Hiera Small", "Hiera Base+", "Hiera Large"])
def test_sam2_encrypted_encoder_decoder_real_export_contract(tmp_path, variant):
    source = tmp_path / "sam.pt"
    torch.save({"type": "sam2", "backend": "sam2", "task": "segment", "variant": variant,
                "input_size": [2, 3], "mask_size": [2, 2],
                "encoder": TinyEncoder().eval(), "decoder": TinyDecoder().eval()}, source)
    key = create_key(tmp_path / "key")
    result = export_onnx.export_checkpoint(source, tmp_path / "sam.dvsenc", encryption_key_path=key, log=lambda _: None)
    doc, graphs = load_package(result["output_path"], read_key(key))
    assert len(graphs) == 2 and doc["variant"] == variant and result["verification"] == "passed"
    contracts = doc["contracts"]["graphs"]
    encoder = ort.InferenceSession(graphs[contracts["encoder"]["file"]], providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(graphs[contracts["decoder"]["file"]], providers=["CPUExecutionProvider"])
    embedding = encoder.run(None, {"input_image": np.ones((1, 3, 2, 3), np.float32)})[0]
    outputs = decoder.run(None, {"image_embeddings": embedding,
        "point_coords": np.array([[[1, 1]]], np.float32), "point_labels": np.array([[1]], np.int64),
        "mask_input": np.zeros((1, 1, 2, 2), np.float32), "has_mask_input": np.zeros((1,), np.float32),
        "orig_im_size": np.array([2, 3], np.float32)})
    assert len(outputs) == 2 and np.isfinite(outputs[0]).all()
    assert not list(tmp_path.glob("*.onnx")) and not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("model_id", ["patchcore_resnet18", "patchcore_wide_resnet50_2"])
def test_patchcore_encrypted_bank_and_score_contract(tmp_path, monkeypatch, model_id):
    from patchcore import PatchCore, STANDARD_SCORE_DEFINITION
    class Backbone(torch.nn.Module):
        def forward(self, images): return images[:, :2, ::4, ::4]
    class Bank:
        backbone = Backbone()
        _avg_pool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        memory_bank = torch.randn(4, 2, generator=torch.Generator().manual_seed(7))
        n_neighbors = 2
        input_size = 8
        anomaly_threshold = .1
        score_definition = STANDARD_SCORE_DEFINITION
        center_crop = None
        preprocessing = "opencv_full_range_v2"
    monkeypatch.setattr(PatchCore, "load", lambda *a, **k: Bank())
    source = tmp_path / "patch.pt"; torch.save({"type": "patchcore", "model_id": model_id}, source)
    key = create_key(tmp_path / "key")
    result = export_onnx.export_checkpoint(source, tmp_path / "patch.dvsenc", encryption_key_path=key, log=lambda _: None)
    doc, graphs = load_package(result["output_path"], read_key(key))
    s = ort.InferenceSession(graphs[doc["model_path"]], providers=["CPUExecutionProvider"])
    assert result["verification"] == "passed" and len(s.get_outputs()) == 2
    assert doc["postprocessing"]["n_neighbors"] == 2
    assert all(np.isfinite(v).all() for v in s.run(None, {s.get_inputs()[0].name: np.ones((1, 3, 8, 8), np.float32)}))
