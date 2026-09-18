"""Export the complete fixed-bank PatchCore inference graph to ONNX.

The graph includes the frozen backbone, average-pooling, fixed memory bank,
vectorized kNN, bilinear map resize, Gaussian smoothing and image score.  It
is intentionally a fixed batch-1 profile so the bank and spatial dimensions
can be validated before a native SDK loads it.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCoreOnnxWrapper(nn.Module):
    def __init__(self, patchcore, *, sigma: float = 4.0):
        super().__init__()
        if patchcore.memory_bank is None:
            raise ValueError("PatchCore memory bank is required before ONNX export")
        bank = patchcore.memory_bank.detach().float().cpu()
        if bank.ndim != 2 or bank.shape[0] < 1 or bank.shape[0] > 4096:
            raise ValueError("PatchCore ONNX requires a memory bank with 1..4096 rows")
        if patchcore.n_neighbors < 1 or patchcore.n_neighbors > bank.shape[0]:
            raise ValueError("PatchCore neighbor count exceeds the memory bank")
        if sigma <= 0:
            raise ValueError("Gaussian sigma must be positive")
        self.backbone = patchcore.backbone.cpu().eval()
        self.avg_pool = patchcore._avg_pool.cpu().eval()
        self.register_buffer("memory_bank", bank)
        self.n_neighbors = int(patchcore.n_neighbors)
        kernel_size = int(2 * torch.ceil(torch.tensor(3.0 * sigma)).item() + 1)
        coordinates = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel = torch.exp(-0.5 * (coordinates / sigma) ** 2)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel_horizontal", kernel.reshape(1, 1, 1, -1))
        self.register_buffer("kernel_vertical", kernel.reshape(1, 1, -1, 1))

    def forward(self, images: torch.Tensor):
        if images.ndim != 4 or images.shape[0] < 1:
            raise ValueError("PatchCore input must be NCHW")
        features = self.avg_pool(self.backbone(images))
        batch, channels, height, width = features.shape
        query = features.permute(0, 2, 3, 1).reshape(-1, channels)
        bank = self.memory_bank.to(query.dtype)
        distances = (query.square().sum(1, keepdim=True) + bank.square().sum(1).unsqueeze(0)
                     - 2.0 * torch.matmul(query, bank.transpose(0, 1))).clamp_min(0.0).sqrt()
        nearest = torch.topk(distances, k=self.n_neighbors, largest=False, dim=1).values.mean(dim=1)
        distance_map = nearest.reshape(batch, 1, height, width)
        anomaly_map = F.interpolate(distance_map, size=(images.shape[2], images.shape[3]),
                                    mode="bilinear", align_corners=False)
        padding = self.kernel_horizontal.shape[-1] // 2
        anomaly_map = F.conv2d(anomaly_map, self.kernel_horizontal, padding=(0, padding))
        anomaly_map = F.conv2d(anomaly_map, self.kernel_vertical, padding=(padding, 0))
        return anomaly_map.reshape(batch, images.shape[2], images.shape[3]).amax(dim=(1, 2)), anomaly_map


def _export_graph(model, dummy, output: Path, opset: int):
    exporter_options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    if "external_data" in inspect.signature(torch.onnx.export).parameters:
        exporter_options["external_data"] = False
    torch.onnx.export(model, dummy, str(output), export_params=True, opset_version=opset,
                      do_constant_folding=True, input_names=["input_image"],
                      output_names=["anomaly_score", "anomaly_map"], **exporter_options)


def export_patchcore_model(patchcore, output_path: str | Path, *, verify: bool = True,
                           opset: int = 17, log=print) -> dict:
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".onnx":
        raise ValueError("PatchCore output must use .onnx suffix")
    if not 11 <= opset <= 17:
        raise ValueError("PatchCore exporter supports ONNX opset 11..17")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = PatchCoreOnnxWrapper(patchcore).eval()
    size = int(patchcore.input_size)
    threshold = patchcore.anomaly_threshold
    threshold = 0.0 if threshold is None else float(threshold)
    if not torch.isfinite(torch.tensor(threshold)):
        raise ValueError("PatchCore anomaly threshold must be finite")
    dummy = torch.randn(1, 3, size, size, generator=torch.Generator().manual_seed(42))
    with tempfile.TemporaryDirectory(prefix=".patchcore-onnx-", dir=output.parent) as directory:
        staged = Path(directory) / output.name
        log(f"PatchCore ONNX 그래프 생성: bank={patchcore.memory_bank.shape[0]}, k={patchcore.n_neighbors}")
        _export_graph(model, dummy, staged, opset)
        import onnx
        onnx.checker.check_model(str(staged))
        if verify:
            import onnxruntime as ort
            with torch.inference_mode():
                expected = model(dummy)
            actual = ort.InferenceSession(str(staged), providers=["CPUExecutionProvider"]).run(
                None, {"input_image": dummy.numpy()})
            for reference, converted in zip(expected, actual):
                if not torch.allclose(reference, torch.from_numpy(converted), atol=1e-4, rtol=1e-4):
                    raise ValueError("PatchCore ONNX numeric verification failed")
        os.replace(staged, output)
    config = {
        "schema_version": 5, "backend": "patchcore", "task": "anomaly",
        "model_path": output.name, "output_name": "anomaly_score",
        "output_names": ["anomaly_score", "anomaly_map"], "input_channels": 3,
        "input_height": size, "input_width": size, "num_classes": 1,
        "normalize_mean": [0.485, 0.456, 0.406], "normalize_std": [0.229, 0.224, 0.225],
        "class_names": [], "preprocessing": {"input_size": [size, size], "in_channels": 3,
            "resize_implementation": "opencv_linear_exact_v1", "interpolation": "INTER_LINEAR_EXACT",
            "antialias": False, "layout": "NCHW", "resize": "bilinear", "value_scale": 255.,
        "color_order": "RGB"}, "verification": "passed" if verify else "skipped",
        "cpp_supported": True,
        "postprocessing": {"score": "patchcore_smoothed_knn_max", "threshold": threshold,
                            "memory_bank_size": int(patchcore.memory_bank.shape[0]),
                            "n_neighbors": int(patchcore.n_neighbors)},
    }
    config_path = output.with_suffix(".json")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_path": str(output), "config_path": str(config_path),
            "file_size_mb": output.stat().st_size / (1024 * 1024), "backend": "patchcore",
            "task": "anomaly", "verification": config["verification"], "cpp_supported": True}


def export_patchcore_checkpoint(checkpoint_path, output_path, *, verify=True, opset=17,
                                simplify=False, log=print):
    from patchcore import PatchCore
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("type") != "patchcore":
        raise ValueError("PatchCore checkpoint required")
    if simplify:
        log("PatchCore 단순화는 그래프 검증 후 별도 도구에서 수행해야 합니다.")
    return export_patchcore_model(PatchCore.load(str(checkpoint_path), device="cpu"), output_path,
                                  verify=verify, opset=opset, log=log)
