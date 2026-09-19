"""Export a checkpoint-provided SAM2 encoder and prompt decoder pair.

The exporter intentionally accepts modules from the selected local checkpoint
instead of downloading SAM2 weights.  It produces the two graphs and the
schema-5 manifest consumed by ``Sam2Inference``.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

# Encoder and decoder are verified with the same bounded FP32 profile used by
# the generic exporter. It is recorded in sam2.json for release audits.
from export_onnx import validate_outputs, verification_tolerances

VERIFICATION_TOLERANCE = verification_tolerances("segment")
SAM2_VARIANTS = frozenset({"Hiera Tiny", "Hiera Small", "Hiera Base+", "Hiera Large"})
SAM2_ENCODER_OUTPUTS = ("image_embeddings", "image_features_0", "image_features_1")
SAM2_PROMPT_INPUTS = ("point_coords", "point_labels", "mask_input", "has_mask_input", "orig_im_size")
SAM2_DECODER_OUTPUTS = ("low_res_mask_logits", "iou_predictions")


class Sam2ExportError(ValueError):
    pass


def _module(checkpoint, key: str):
    from torch import nn
    value = checkpoint.get(key)
    if not isinstance(value, nn.Module):
        raise Sam2ExportError(f"SAM2 checkpoint requires an nn.Module in {key}")
    return value.cpu().eval()


def _encoder_outputs(value, checkpoint: dict) -> list[tuple[str, torch.Tensor]]:
    """Name encoder tensors using the same semantics consumed by the C++ runtime."""
    configured_names = checkpoint.get("encoder_output_names")
    if isinstance(value, Mapping):
        names = list(value.keys())
        values = list(value.values())
        if configured_names is not None and list(configured_names) != names:
            raise Sam2ExportError("encoder_output_names must match the encoder mapping keys and order")
    elif isinstance(value, (tuple, list)):
        values = list(value)
        if configured_names is None:
            if len(values) != 1:
                raise Sam2ExportError(
                    "SAM2 tuple/list encoder outputs require encoder_output_names"
                )
            names = ["image_embeddings"]
        else:
            names = list(configured_names)
    else:
        values = [value]
        names = ["image_embeddings"] if configured_names is None else list(configured_names)

    if (not values or len(names) != len(values) or
            any(not isinstance(name, str) or not name for name in names) or
            len(set(names)) != len(names)):
        raise Sam2ExportError("SAM2 encoder output names must be unique and match its tensor outputs")
    if any(name not in SAM2_ENCODER_OUTPUTS for name in names) or "image_embeddings" not in names:
        raise Sam2ExportError(
            "SAM2 encoder outputs must use image_embeddings and optional image_features_0/image_features_1"
        )
    if any(not isinstance(item, torch.Tensor) for item in values):
        raise Sam2ExportError("SAM2 encoder returned no tensor outputs")
    return list(zip(names, values))


def _decoder_contract(checkpoint: dict, encoder_names: list[str]):
    semantic_names = [*encoder_names, *SAM2_PROMPT_INPUTS]
    order = checkpoint.get("decoder_input_order", semantic_names)
    if (not isinstance(order, (list, tuple)) or len(order) != len(semantic_names) or
            any(not isinstance(name, str) for name in order) or
            len(set(order)) != len(order) or set(order) != set(semantic_names)):
        raise Sam2ExportError(
            "decoder_input_order must list every encoder output and prompt input exactly once"
        )
    configured_names = checkpoint.get("decoder_input_names", {})
    if not isinstance(configured_names, Mapping):
        raise Sam2ExportError("decoder_input_names must map semantic input names to ONNX input names")
    input_names = {semantic: configured_names.get(semantic, semantic) for semantic in semantic_names}
    if (any(not isinstance(name, str) or not name for name in input_names.values()) or
            len(set(input_names.values())) != len(input_names)):
        raise Sam2ExportError("SAM2 decoder ONNX input names must be non-empty and unique")
    if not set(configured_names).issubset(input_names):
        raise Sam2ExportError("decoder_input_names contains an unknown semantic input")
    if any(input_names[name] != name for name in encoder_names):
        raise Sam2ExportError(
            "SAM2 decoder embedding input names must match the corresponding encoder output names"
        )
    return list(order), input_names


def _decoder_output_spec(value, checkpoint: dict):
    configured_names = checkpoint.get("decoder_output_names")
    if isinstance(value, Mapping):
        if configured_names is None:
            output_map = {name: name for name in value}
        elif isinstance(configured_names, Mapping):
            output_map = dict(configured_names)
        else:
            raise Sam2ExportError("mapping decoder outputs require decoder_output_names as a semantic-name map")
        if (any(not isinstance(name, str) or name not in SAM2_DECODER_OUTPUTS for name in output_map) or
                any(not isinstance(name, str) or not name for name in output_map.values()) or
                len(set(output_map.values())) != len(output_map) or
                not set(output_map.values()).issubset(value) or
                "low_res_mask_logits" not in output_map):
            raise Sam2ExportError("SAM2 decoder output mapping must identify mask logits and optional quality")
        return "mapping", output_map

    values = list(value) if isinstance(value, (tuple, list)) else [value]
    if configured_names is None:
        names = list(SAM2_DECODER_OUTPUTS[:len(values)])
    elif isinstance(configured_names, (list, tuple)):
        names = list(configured_names)
    else:
        raise Sam2ExportError("sequence decoder outputs require decoder_output_names as an ordered list")
    if (not values or len(values) > len(SAM2_DECODER_OUTPUTS) or len(names) != len(values) or
            any(not isinstance(name, str) or name not in SAM2_DECODER_OUTPUTS for name in names) or
            len(set(names)) != len(names) or "low_res_mask_logits" not in names):
        raise Sam2ExportError("SAM2 decoder must return mask logits and optional quality scores")
    return "sequence", names


def _normalize_decoder_outputs(value, output_spec):
    kind, names = output_spec
    if kind == "mapping":
        values = {semantic: value[key] for semantic, key in names.items()}
    else:
        raw_values = list(value) if isinstance(value, (tuple, list)) else [value]
        values = dict(zip(names, raw_values))
    if any(not isinstance(item, torch.Tensor) for item in values.values()):
        raise Sam2ExportError("SAM2 decoder returned non-tensor outputs")
    return [values[name] for name in SAM2_DECODER_OUTPUTS if name in values]


def _validate_onnx_contract(onnx, path: Path, input_names: list[str], output_names: list[str]):
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    initializer_names = {item.name for item in graph.graph.initializer}
    actual_inputs = [item.name for item in graph.graph.input if item.name not in initializer_names]
    actual_outputs = [item.name for item in graph.graph.output]
    if actual_inputs != input_names:
        raise Sam2ExportError(
            f"SAM2 ONNX inputs do not match the runtime contract: expected {input_names}, got {actual_inputs}"
        )
    if actual_outputs != output_names:
        raise Sam2ExportError(
            f"SAM2 ONNX outputs do not match the runtime contract: expected {output_names}, got {actual_outputs}"
        )


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
    variant = checkpoint.get("variant")
    if variant not in SAM2_VARIANTS:
        raise Sam2ExportError("SAM2 variant must be Hiera Tiny, Hiera Small, Hiera Base+ or Hiera Large")
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
        named_embedding_values = _encoder_outputs(encoder(input_tensor), checkpoint)
    encoder_names = [name for name, _ in named_embedding_values]
    embedding_values = {name: value.detach().float() for name, value in named_embedding_values}
    decoder_input_order, decoder_input_names = _decoder_contract(checkpoint, encoder_names)
    point_coords = torch.zeros((1, 1, 2), dtype=torch.float32)
    point_labels = torch.full((1, 1), -1, dtype=torch.int64)
    mask_input = torch.zeros((1, 1, int(mask_size[0]), int(mask_size[1])), dtype=torch.float32)
    has_mask_input = torch.zeros((1,), dtype=torch.float32)
    orig_im_size = torch.tensor([float(size[0]), float(size[1])], dtype=torch.float32)
    decoder_values_by_semantic = {
        **embedding_values,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
        "orig_im_size": orig_im_size,
    }
    decoder_args = tuple(decoder_values_by_semantic[name] for name in decoder_input_order)
    with torch.inference_mode():
        raw_decoder_values = decoder(*decoder_args)
    decoder_output_spec = _decoder_output_spec(raw_decoder_values, checkpoint)
    decoder_values = _normalize_decoder_outputs(raw_decoder_values, decoder_output_spec)
    if len(decoder_values) < 1 or len(decoder_values) > 2:
        raise Sam2ExportError("SAM2 decoder must return mask logits and optional quality scores")
    if decoder_values[0].ndim not in {2, 3, 4}:
        raise Sam2ExportError("SAM2 mask logits must have rank 2, 3 or 4")
    class DecoderWithScore(torch.nn.Module):
        def __init__(self, wrapped, output_spec, include_score):
            super().__init__()
            self.wrapped = wrapped
            self.output_spec = output_spec
            self.include_score = include_score
        def forward(self, *values):
            raw = self.wrapped(*values)
            outputs = _normalize_decoder_outputs(raw, self.output_spec)
            mask_logits = outputs[0]
            scores = outputs[1] if self.include_score else torch.ones(
                (mask_logits.shape[0], 1), device=mask_logits.device)
            return mask_logits, scores
    decoder_for_export = DecoderWithScore(decoder, decoder_output_spec, len(decoder_values) == 2).eval()
    if len(decoder_values) == 1:
        decoder_values.append(torch.ones((1, 1), dtype=torch.float32))

    encoder_file = output / "sam2_encoder.onnx"
    decoder_file = output / "sam2_decoder.onnx"
    config_file = output / "sam2.json"
    with tempfile.TemporaryDirectory(prefix=".sam2-export-", dir=output) as directory:
        stage = Path(directory)
        staged_encoder, staged_decoder = stage / encoder_file.name, stage / decoder_file.name
        log(f"SAM2 encoder/decoder ONNX 생성: 입력={tuple(input_tensor.shape)}")
        _export(encoder, (input_tensor,), {}, staged_encoder,
                input_names=["input_image"], output_names=encoder_names, opset=opset)
        _export(decoder_for_export, decoder_args, {}, staged_decoder,
                input_names=[decoder_input_names[name] for name in decoder_input_order],
                output_names=list(SAM2_DECODER_OUTPUTS), opset=opset,
                dynamic_axes={
                    decoder_input_names["point_coords"]: {1: "num_points"},
                    decoder_input_names["point_labels"]: {1: "num_points"},
                })
        import onnx
        _validate_onnx_contract(onnx, staged_encoder, ["input_image"], encoder_names)
        _validate_onnx_contract(
            onnx, staged_decoder,
            [decoder_input_names[name] for name in decoder_input_order], list(SAM2_DECODER_OUTPUTS),
        )
        if verify:
            import onnxruntime as ort
            encoder_session = ort.InferenceSession(str(staged_encoder), providers=["CPUExecutionProvider"])
            decoder_session = ort.InferenceSession(str(staged_decoder), providers=["CPUExecutionProvider"])
            encoded = encoder_session.run(None, {"input_image": input_tensor.numpy()})
            for point_count in (1, 2, 3, 8):
                case_values = dict(decoder_values_by_semantic)
                if point_count > 1:
                    case_values["point_coords"] = torch.linspace(
                        0.0, float(max(size)), point_count * 2, dtype=torch.float32
                    ).reshape(1, point_count, 2)
                    case_values["point_labels"] = torch.ones(
                        (1, point_count), dtype=torch.int64
                    )
                case_args = tuple(case_values[name] for name in decoder_input_order)
                with torch.inference_mode():
                    expected_values = decoder_for_export(*case_args)
                decoded = decoder_session.run(None, {
                    decoder_input_names[name]: encoded[encoder_names.index(name)]
                    if name in embedding_values else case_values[name].detach().cpu().numpy()
                    for name in decoder_input_order
                })
                for expected, actual in zip(expected_values, decoded):
                    try:
                        validate_outputs(expected.detach().float().numpy(), actual,
                                         **VERIFICATION_TOLERANCE)
                    except ValueError as exc:
                        raise Sam2ExportError(
                            f"SAM2 decoder verification failed for {point_count} prompt points: {exc}"
                        ) from exc
        os.replace(staged_encoder, encoder_file)
        os.replace(staged_decoder, decoder_file)
    config = {
        "schema_version": 5, "backend": "sam2", "task": "segment", "variant": variant,
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
                   "dynamic_prompt_points": True,
                   "verification_tolerance": dict(VERIFICATION_TOLERANCE),
                   "verification_reference": "exported_pytorch_graph"},
        "cpp_supported": True,
        "contracts": {"graphs": {
            "encoder": {"file": encoder_file.name, "inputs": {"image": "input_image"},
                         "outputs": encoder_names},
            "decoder": {"file": decoder_file.name,
                "inputs": {name: decoder_input_names[name] for name in decoder_input_order},
                "outputs": list(SAM2_DECODER_OUTPUTS)}},
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
