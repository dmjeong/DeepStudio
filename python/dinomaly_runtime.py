"""Shared Dinomaly checkpoint/deployment contract and native inference adapter."""

import copy
from pathlib import Path

import cv2
import numpy as np
import torch


def deployment_config(checkpoint, model_path="model.onnx"):
    size = int(checkpoint["input_size"])
    channels = int(checkpoint["in_channels"])
    mean, std = (
        ([0.449], [0.226])
        if channels == 1
        else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    prep = {
        "input_size": [size, size],
        "in_channels": channels,
        "resize": "bilinear",
        "resize_mode": "stretch",
        "resize_implementation": "opencv_linear_exact_v1",
        "interpolation": "INTER_LINEAR_EXACT",
        "antialias": False,
        "layout": "NCHW",
        "color_order": "GRAY" if channels == 1 else "RGB",
        "value_scale": 255.0,
        "center_crop": checkpoint.get("center_crop"),
    }
    if checkpoint.get("tiling"):
        prep["tiling"] = copy.deepcopy(checkpoint["tiling"])
    return {
        "schema_version": 8,
        "backend": "dinomaly",
        "task": "anomaly",
        "model_path": model_path,
        "input_name": "input_image",
        "output_name": "anomaly_score",
        "output_names": ["anomaly_score", "anomaly_map"],
        "input_channels": channels,
        "input_height": size,
        "input_width": size,
        "num_classes": 1,
        "class_names": [],
        "normalize_mean": mean,
        "normalize_std": std,
        "preprocessing": prep,
        "cpp_supported": True,
        "postprocessing": {
            "score": "dinomaly_cosine_map_max_v1",
            "threshold": checkpoint.get("anomaly_threshold"),
        },
        "calibration": checkpoint.get("calibration", {}),
    }


def read_bgr(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ValueError(f"Dinomaly 입력은 8비트 이미지여야 합니다: {path}")
    return image


class DinomalyRuntime:
    name = "Dinomaly"
    preprocessing = "legacy_pil_rgb"

    def __init__(self, checkpoint, device="cpu"):
        from dinomaly import load_checkpoint

        self.model = load_checkpoint(checkpoint, device)
        self.device = torch.device(device)
        self.config = deployment_config(checkpoint)
        self.input_size = checkpoint["input_size"]
        self.center_crop = checkpoint.get("center_crop")
        self.anomaly_threshold = checkpoint.get("anomaly_threshold")
        self.calibration = checkpoint.get("calibration", {})

    @classmethod
    def load(cls, path, device="cpu"):
        return cls(torch.load(path, map_location="cpu", weights_only=True), device)

    def get_score_normalization(self):
        from patchcore_scores import score_normalization

        return score_normalization(self.anomaly_threshold)

    @property
    def normalized_threshold(self):
        from patchcore_scores import normalize_threshold

        return normalize_threshold(
            self.anomaly_threshold, self.get_score_normalization()
        )

    def get_info(self):
        return {
            "model": "Dinomaly DINOv2-S/14",
            "model_id": "dinomaly_vits14",
            "input_size": self.input_size,
            "score_definition": "dinomaly_cosine_map_max_v1",
            "calibration": self.calibration,
            "center_crop": self.center_crop,
        }

    def read_original(self, path):
        image = read_bgr(path)
        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
            if image.ndim == 2
            else cv2.COLOR_BGR2RGB
            if image.shape[2] == 3
            else cv2.COLOR_BGRA2RGB,
        )

    @torch.inference_mode()
    def predict_from_path(self, path, *, center_crop=None):
        from dense_deployment import predict_dense

        config = copy.deepcopy(self.config)
        config["preprocessing"]["center_crop"] = center_crop

        def run(tensor):
            return [
                value.cpu().numpy()
                for value in self.model(torch.from_numpy(tensor).to(self.device))
            ]

        maps = predict_dense(config, read_bgr(path), run)
        return np.asarray([maps.max()], np.float32), maps


def export_dinomaly_checkpoint(
    checkpoint_path, output_path, *, verify=True, opset=17, log=print
):
    import json
    import os
    import tempfile

    import onnx
    import onnxruntime as ort
    from dinomaly import load_checkpoint
    from export_onnx import export_to_onnx, validate_outputs, verification_tolerances

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = deployment_config(checkpoint, Path(output_path).name)
    threshold = config["postprocessing"]["threshold"]
    if threshold is None or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Dinomaly 정상 기준 보정 후 내보내세요")
    model = load_checkpoint(checkpoint)
    dummy = torch.randn(
        1,
        config["input_channels"],
        config["input_height"],
        config["input_width"],
        generator=torch.Generator().manual_seed(42),
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".dinomaly-export-", dir=output.parent
    ) as directory:
        staged = Path(directory) / output.name
        export_to_onnx(
            model,
            dummy,
            staged,
            opset,
            False,
            "anomaly",
            output_names=config["output_names"],
        )
        onnx.checker.check_model(str(staged))
        tolerance = verification_tolerances("anomaly")
        if verify:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            session = ort.InferenceSession(
                str(staged), options, providers=["CPUExecutionProvider"]
            )
            for probe in (dummy, torch.zeros_like(dummy), torch.ones_like(dummy)):
                with torch.inference_mode():
                    expected = model(probe)
                actual = session.run(None, {config["input_name"]: probe.numpy()})
                for a, b in zip(expected, actual):
                    validate_outputs(a.numpy(), b, **tolerance)
        config["verification"] = "passed" if verify else "skipped"
        config["export"] = {
            "onnxruntime_version": ort.__version__,
            "opset": opset,
            "precision": "float32",
            "dynamic_batch": False,
            "verification_tolerance": tolerance,
            "verification_reference": "original_checkpoint_pytorch",
        }
        manifest = Path(directory) / output.with_suffix(".json").name
        manifest.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(staged, output)
        os.replace(manifest, output.with_suffix(".json"))
    log(f"Dinomaly FP32 ONNX 완료: {output}")
    return {
        "output_path": str(output),
        "config_path": str(output.with_suffix(".json")),
        "task": "anomaly",
        "backend": "dinomaly",
        "verification": config["verification"],
        "cpp_supported": True,
        "file_size_mb": output.stat().st_size / 1024**2,
    }
