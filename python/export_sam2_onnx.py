"""Export a checkpoint-provided SAM2 encoder and prompt decoder pair.

The exporter intentionally accepts modules from the selected local checkpoint
instead of downloading SAM2 weights.  It produces the two graphs and the
schema-5 manifest consumed by ``Sam2Inference``.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import os
import tempfile

import torch

# Encoder and decoder are verified with the same bounded FP32 profile used by
# the generic exporter. It is recorded in sam2.json for release audits.
from export_onnx import validate_outputs, verification_tolerances

VERIFICATION_TOLERANCE = verification_tolerances("segment")


class Sam2ExportError(ValueError):
    pass


def _module(checkpoint, key: str):
    import torch.nn as nn
    value = checkpoint.get(key)
    if not isinstance(value, nn.Module):
        raise Sam2ExportError(f"SAM2 checkpoint requires an nn.Module in {key}")
    return value.cpu().eval()


def _outputs(value, name: str):
    if isinstance(value, dict):
        values = list(value.values())
    elif isinstance(value, (tuple, list)):
        values = list(value)
    else:
        values = [value]
    if not values or any(not isinstance(item, torch.Tensor) for item in values):
        raise Sam2ExportError(f"SAM2 {name} returned no tensor outputs")
    return values


def _export(model, args, kwargs, output: Path, *, input_names, output_names, opset: int,
            dynamic_axes=None):
    options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    if "external_data" in inspect.signature(torch.onnx.export).parameters:
        options["external_data"] = False
    torch.onnx.export(model, args, str(output), kwargs=kwargs, export_params=True,
                      opset_version=opset, do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, **options)


def export_sam2_model(checkpoint: dict, output_dir: str | Path, *, verify: bool = True,
                      opset: int = 17, log=print, bundle_output: str | Path | None = None) -> dict:
    if not isinstance(checkpoint, dict) or checkpoint.get("backend", checkpoint.get("type")) != "sam2":
        raise Sam2ExportError("SAM2 checkpoint required")
    if checkpoint.get("task", "segment") != "segment":
        raise Sam2ExportError("SAM2 checkpoint task must be segment")
    if not 11 <= opset <= 17:
        raise Sam2ExportError("SAM2 exporter supports ONNX opset 11..17")
    size = checkpoint.get("input_size", [1024, 1024])
    if isinstance(size, int):
        size = [size, size]
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(int(value) <= 0 for value in size):
        raise Sam2ExportError("SAM2 input_size must be [height, width]")
    channels = int(checkpoint.get("in_channels", 3))
    if channels != 3:
        raise Sam2ExportError("SAM2 encoder requires three input channels")
    mask_size = checkpoint.get("mask_size", [256, 256])
    if not isinstance(mask_size, (list, tuple)) or len(mask_size) != 2 or any(int(value) <= 0 for value in mask_size):
        raise Sam2ExportError("SAM2 mask_size must be [height, width]")
    encoder, decoder = _module(checkpoint, "encoder"), _module(checkpoint, "decoder")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_tensor = torch.randn(1, channels, int(size[0]), int(size[1]), generator=torch.Generator().manual_seed(42))
    with torch.inference_mode():
        embedding_values = _outputs(encoder(input_tensor), "encoder")
    if len(embedding_values) != 1:
        raise Sam2ExportError("SAM2 encoder export currently requires one image_embeddings output")
    embedding = embedding_values[0].detach().float()
    point_coords = torch.zeros((1, 1, 2), dtype=torch.float32)
    point_labels = torch.full((1, 1), -1, dtype=torch.int64)
    mask_input = torch.zeros((1, 1, int(mask_size[0]), int(mask_size[1])), dtype=torch.float32)
    has_mask_input = torch.zeros((1,), dtype=torch.float32)
    orig_im_size = torch.tensor([float(size[0]), float(size[1])], dtype=torch.float32)
    decoder_args = (embedding, point_coords, point_labels, mask_input, has_mask_input, orig_im_size)
    with torch.inference_mode():
        decoder_values = _outputs(decoder(*decoder_args), "decoder")
    if len(decoder_values) < 1 or len(decoder_values) > 2:
        raise Sam2ExportError("SAM2 decoder must return mask logits and optional quality scores")
    if decoder_values[0].ndim not in {2, 3, 4}:
        raise Sam2ExportError("SAM2 mask logits must have rank 2, 3 or 4")
    if len(decoder_values) == 1:
        decoder_values.append(torch.ones((1, 1), dtype=torch.float32))
        class DecoderWithScore(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
            def forward(self, *values):
                result = _outputs(self.wrapped(*values), "decoder")
                return result[0], torch.ones((values[0].shape[0], 1), device=values[0].device)
        decoder_for_export = DecoderWithScore(decoder).eval()
    else:
        decoder_for_export = decoder

    encoder_file = output / "sam2_encoder.onnx"
    decoder_file = output / "sam2_decoder.onnx"
    config_file = output / "sam2.json"
    with tempfile.TemporaryDirectory(prefix=".sam2-export-", dir=output) as directory:
        stage = Path(directory)
        staged_encoder, staged_decoder = stage / encoder_file.name, stage / decoder_file.name
        log(f"SAM2 encoder/decoder ONNX 생성: 입력={tuple(input_tensor.shape)}")
        _export(encoder, (input_tensor,), {}, staged_encoder,
                input_names=["input_image"], output_names=["image_embeddings"], opset=opset)
        _export(decoder_for_export, decoder_args, {}, staged_decoder,
                input_names=["image_embeddings", "point_coords", "point_labels", "mask_input",
                             "has_mask_input", "orig_im_size"],
                output_names=["low_res_mask_logits", "iou_predictions"], opset=opset)
        import onnx
        onnx.checker.check_model(str(staged_encoder))
        onnx.checker.check_model(str(staged_decoder))
        if verify:
            import onnxruntime as ort
            encoder_session = ort.InferenceSession(str(staged_encoder), providers=["CPUExecutionProvider"])
            decoder_session = ort.InferenceSession(str(staged_decoder), providers=["CPUExecutionProvider"])
            encoded = encoder_session.run(None, {"input_image": input_tensor.numpy()})
            decoded = decoder_session.run(None, {
                "image_embeddings": encoded[0], "point_coords": point_coords.numpy(),
                "point_labels": point_labels.numpy(), "mask_input": mask_input.numpy(),
                "has_mask_input": has_mask_input.numpy(), "orig_im_size": orig_im_size.numpy(),
            })
            for expected, actual in zip(decoder_values[:2], decoded):
                # CPU graph fusion can reassociate FP32 arithmetic.  Keep a
                # bounded deployment tolerance instead of rejecting a valid
                # graph for sub-millilogit drift.
                try:
                    validate_outputs(expected.detach().float().numpy(), actual,
                                     **VERIFICATION_TOLERANCE)
                except ValueError as exc:
                    raise Sam2ExportError(str(exc)) from exc
        os.replace(staged_encoder, encoder_file)
        os.replace(staged_decoder, decoder_file)
    config = {
        "schema_version": 5, "backend": "sam2", "task": "segment",
        "model_path": encoder_file.name, "output_name": "low_res_mask_logits",
        "output_names": ["low_res_mask_logits", "iou_predictions"], "num_classes": 1,
        "input_channels": channels, "input_height": int(size[0]), "input_width": int(size[1]),
        "input_name": "input_image", "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225], "class_names": [],
        "preprocessing": {"input_size": [int(size[0]), int(size[1])], "in_channels": channels,
            "resize_implementation": "opencv_linear_exact_v1", "interpolation": "INTER_LINEAR_EXACT",
            "antialias": False, "layout": "NCHW", "value_scale": 255., "color_order": "RGB"},
        "verification": "passed" if verify else "skipped",
        "export": {"opset": opset, "precision": "float32", "dynamic_batch": False,
                   "verification_tolerance": dict(VERIFICATION_TOLERANCE),
                   "verification_reference": "exported_pytorch_graph"},
        "cpp_supported": True,
        "contracts": {"graphs": {
            "encoder": {"file": encoder_file.name, "inputs": {"image": "input_image"},
                         "outputs": ["image_embeddings"]},
            "decoder": {"file": decoder_file.name, "inputs": {
                "image_embeddings": "image_embeddings", "point_coords": "point_coords",
                "point_labels": "point_labels", "mask_input": "mask_input",
                "has_mask_input": "has_mask_input", "orig_im_size": "orig_im_size"},
                "outputs": ["low_res_mask_logits", "iou_predictions"]}},
            "prompt_types": ["point", "box", "mask"], "video_state": False,
            "prompt_coordinate_space": "resized_input",
            "automatic_mask": {"mode": "positive_point_grid_union", "max_grid": 32},
            "mask_size": [int(mask_size[0]), int(mask_size[1])]},
    }
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"output_dir": str(output), "encoder_path": str(encoder_file),
            "decoder_path": str(decoder_file), "config_path": str(config_file),
            "backend": "sam2", "task": "segment", "verification": config["verification"],
            "cpp_supported": True, "verification_tolerance": dict(VERIFICATION_TOLERANCE)}
    if bundle_output is not None:
        from model_runtime.deployment_bundle import build_deployment_bundle
        bundle = build_deployment_bundle(output, bundle_output)
        result["bundle_path"] = str(bundle)
        log(f"배포 번들 생성: {bundle}")
    return result


def export_sam2_checkpoint(checkpoint_path, output_dir, *, verify=True, opset=17, log=print,
                           bundle_output=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return export_sam2_model(checkpoint, output_dir, verify=verify, opset=opset, log=log,
                             bundle_output=bundle_output)
