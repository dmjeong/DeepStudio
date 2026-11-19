"""FP16 deployment must validate every public output before replacing existing files."""

import json
from pathlib import Path

import numpy as np
import pytest

import export_onnx


def _two_output_reference(checkpoint, output, *args, **kwargs):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    shape = ["batch", 1, 4, 4]
    nodes = [
        helper.make_node("Mul", ["image", "w"], ["scaled"]),
        helper.make_node("Add", ["image", "offset"], ["shifted"]),
    ]
    graph = helper.make_graph(nodes, "two_heads",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ("scaled", "shifted")],
        [numpy_helper.from_array(np.array(.123456, np.float32), "w"),
         numpy_helper.from_array(np.array(.765432, np.float32), "offset")])
    onnx.save(helper.make_model(graph, ir_version=9, opset_imports=[helper.make_opsetid("", 17)]), output)
    manifest = output.with_suffix(".json")
    manifest.write_text(json.dumps({
        "backend": "custom", "task": "detect", "verification": "passed", "model_path": output.name,
        "cpp_supported": True, "input_name": "image", "output_names": ["scaled", "shifted"],
        "num_threads": 1, "input_channels": 1, "input_height": 4, "input_width": 4,
        "export": {"precision": "float32", "opset": 17, "dynamic_batch": True},
    }))
    return {"config_path": str(manifest), "output_path": str(output)}


def test_public_fp16_checks_all_heads_and_batch_sizes(tmp_path, monkeypatch):
    import onnx

    monkeypatch.setattr(export_onnx, "_export_checkpoint", _two_output_reference)
    output = tmp_path / "model_fp16.onnx"
    result = export_onnx.export_checkpoint(tmp_path / "source.pt", output, precision="float16", dynamic_batch=True)
    cfg = json.loads(Path(result["config_path"]).read_text())
    assert cfg["verification"] == result["verification"] == "fp16_compared"
    assert cfg["export"]["io_precision"] == "float32"
    assert cfg["export"]["precision"] == "float16"
    assert cfg["model_path"] == output.name
    report = cfg["export"]["fp16_comparison"]
    assert report == result["fp16_comparison"]
    assert set(report["max_abs_error"]) == {"scaled", "shifted"}
    assert report["probes"] == ["seeded", "zero", "batch_2"]
    assert report["validation_images"] == 0
    graph = onnx.load(output).graph
    assert all(i.data_type == onnx.TensorProto.FLOAT16 for i in graph.initializer)
    assert all(i.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for i in [*graph.input, *graph.output])


def test_wrong_second_fp16_head_preserves_previous_deployment(tmp_path, monkeypatch):
    import benchmark_fp16
    import onnx
    from onnx import numpy_helper

    convert = benchmark_fp16.convert_benchmark_fp16
    def corrupt(*args):
        manifest = convert(*args)
        model = manifest.parent / json.loads(manifest.read_text())["model_path"]
        graph = onnx.load(model)
        weight = next(w for w in graph.graph.initializer if w.name == "offset")
        weight.CopyFrom(numpy_helper.from_array(np.array(1.5, np.float16), "offset"))
        onnx.save(graph, model)
        return manifest

    monkeypatch.setattr(export_onnx, "_export_checkpoint", _two_output_reference)
    monkeypatch.setattr(benchmark_fp16, "convert_benchmark_fp16", corrupt)
    output = tmp_path / "model.onnx"
    output.write_bytes(b"last verified graph")
    output.with_suffix(".json").write_bytes(b"last verified config")
    with pytest.raises(ValueError, match="FP16 검증 실패.*shifted"):
        export_onnx.export_checkpoint(tmp_path / "original.pt", output, precision="float16")
    assert output.read_bytes() == b"last verified graph"
    assert output.with_suffix(".json").read_bytes() == b"last verified config"
    assert not list(tmp_path.glob(".fp16-export-*"))


def test_locked_sidecar_rolls_model_back(tmp_path, monkeypatch):
    import export_fp16_onnx as fp16

    stage = tmp_path / "stage"
    stage.mkdir()
    output, staged = [root / "model.onnx" for root in (tmp_path, stage)]
    for path, content in [(output, "previous"), (staged, "next")]:
        path.write_text(content)
        path.with_suffix(".json").write_text(content)
    replace = fp16.os.replace
    def simulate_windows_lock(src, dst):
        if dst == output.with_suffix(".json"):
            raise PermissionError("locked")
        return replace(src, dst)
    monkeypatch.setattr(fp16.os, "replace", simulate_windows_lock)
    with pytest.raises(PermissionError, match="locked"):
        fp16._publish(staged, staged.with_suffix(".json"), output)
    assert output.read_text() == output.with_suffix(".json").read_text() == "previous"


@pytest.mark.parametrize(("options", "message"), [
    ({"verify": False}, "출력 검증"),
    ({"allow_precision_fallback": True}, "FP64"),
    ({"sam2_model_id": "sam2_hiera_tiny"}, "SAM2"),
])
def test_fp16_rejects_unsupported_requests_before_export(tmp_path, options, message):
    output = tmp_path / "model.onnx"
    with pytest.raises(ValueError, match=message):
        export_onnx.export_checkpoint(tmp_path / "absent.pt", output, precision="float16", **options)
    assert not output.exists() and not output.with_suffix(".json").exists()


def test_concurrent_export_refuses_to_touch_models_or_diagnostics(tmp_path):
    output = tmp_path / "model.onnx"
    diagnostic = output.with_suffix(".export-error.json")
    diagnostic.write_text("active job")
    with export_onnx._destination_lock(output):
        with pytest.raises(ValueError, match="진행 중"):
            export_onnx.export_checkpoint(tmp_path / "source.pt", output, precision="float16", log=lambda _: None)
    assert diagnostic.read_text() == "active job" and not output.exists()


def test_segmentation_validates_stitched_masks_per_image(tmp_path, monkeypatch):
    import cv2
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def segmentation_model(checkpoint, output, *args, **kwargs):
        graph = helper.make_graph([helper.make_node("Conv", ["image", "w"], ["mask"])], "seg",
            [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 32, 32])],
            [helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 2, 32, 32])],
            [numpy_helper.from_array(np.array([1., -.125], np.float32).reshape(2, 1, 1, 1), "w")])
        onnx.save(helper.make_model(graph, ir_version=9, opset_imports=[helper.make_opsetid("", 17)]), output)
        cfg = {
            "backend": "builtin", "task": "segment", "model_path": output.name, "verification": "passed",
            "input_name": "image", "output_name": "mask", "input_channels": 1,
            "input_height": 32, "input_width": 32, "normalize_mean": [.5], "normalize_std": [.2],
            "num_classes": 2, "num_threads": 1, "cpp_supported": True, "export": {"precision": "float32"},
            "preprocessing": {"tiling": {"tile": 32, "overlap": 16, "blend": "center_weighted_softmax"}},
        }
        manifest = output.with_suffix(".json")
        manifest.write_text(json.dumps(cfg))
        return {"config_path": str(manifest)}

    monkeypatch.setattr(export_onnx, "_export_checkpoint", segmentation_model)
    images = tmp_path / "images"
    images.mkdir()
    for i in range(2):
        assert cv2.imwrite(str(images / f"{i}.png"), np.full((32, 64), i, np.uint8))
    result = export_onnx.export_checkpoint(tmp_path / "source.pt", tmp_path / "seg.onnx",
                                          precision="float16", validation_dir=images)
    report = result["fp16_comparison"]
    assert report["validation_images"] == 2 and report["validation_windows"] == 6
    assert len(report["segmentation"]["real_images"]) == 2
    for row in report["segmentation"]["real_images"]:
        assert row["pixels"] == 32 * 64 and row["foreground_iou_by_class_id"]["1"] == 1.0


def test_small_logit_errors_cannot_erase_a_real_defect():
    from export_fp16_onnx import _compare_masks, _max_error

    left = np.zeros((1, 2, 32, 32), np.float32)
    left[0, 1, 1, 1] = .00001
    right = np.zeros_like(left)
    assert _max_error(left, right, atol=.01, rtol=.01, label="check") < .001
    with pytest.raises(ValueError, match="클래스 1 IoU"):
        _compare_masks(left, right, label="crack.png", real_image=True)
    assert _compare_masks(left, right, label="seeded", real_image=False)["matched"] == 1023
    left[:, 1] = .00001
    with pytest.raises(ValueError, match="픽셀 일치율"):
        _compare_masks(left, right, label="seeded", real_image=False)


@pytest.mark.parametrize(("dtype", "size"), [(np.uint8, (33, 32)), (np.uint16, (32, 32))])
def test_whole_image_fp16_verification_enforces_cpp_region_contract(dtype, size):
    from export_fp16_onnx import _real_segment_maps

    cfg = {"preprocessing": {"whole_image": {"region": [32, 32], "scaled": [32, 32]}}}
    with pytest.raises(ValueError, match=r"C\+\+ 판정 영역"):
        _real_segment_maps(cfg, np.ones(size, dtype), lambda _: pytest.fail("bad input was accepted"))
