import json

import numpy as np
import pytest

from benchmark_fp16 import convert_benchmark_fp16


def test_benchmark_fp16_has_half_weights_and_float_cpu_io(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper
    from onnx_session import cpu_session_options

    source = tmp_path / "source.onnx"
    manifest = tmp_path / "source.json"
    shape = [1, 1, 4, 4]
    weight = numpy_helper.from_array(np.array([[[[.123456]]]], dtype=np.float32), "w")
    graph = helper.make_graph([helper.make_node("Conv", ["image", "w"], ["logits"])], "test",
                              [helper.make_tensor_value_info("image", TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, shape)], [weight])
    model = helper.make_model(graph, ir_version=9, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, source)
    manifest.write_text(json.dumps({"model_path": source.name, "input_name": "image", "output_names": ["logits"],
                                    "verification": "passed", "export": {"precision": "float32"}}))
    original = source.read_bytes()
    fp16_manifest = convert_benchmark_fp16(manifest, tmp_path / "output")
    config = json.loads(fp16_manifest.read_text())
    fp16_model = fp16_manifest.parent / config["model_path"]
    assert config["export"]["precision"] == "float16"
    assert config["verification"] == "benchmark_only"
    assert source.read_bytes() == original
    converted = onnx.load(fp16_model)
    onnx.checker.check_model(converted)
    assert converted.graph.initializer[0].data_type == TensorProto.FLOAT16

    session = ort.InferenceSession(str(fp16_model), cpu_session_options(num_threads=2),
                                   providers=["CPUExecutionProvider"])
    logits, = session.run(["logits"], {"image": np.ones(shape, dtype=np.float32)})
    assert logits.dtype == np.float32 and logits.shape == tuple(shape)
    np.testing.assert_allclose(logits, np.float16(.123456).astype(np.float32), atol=1e-7, rtol=0)


def test_dinov2_fp16_folds_position_resize_and_attention_scale_before_conversion(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper
    from onnx_session import cpu_session_options

    shape = ["batch", 1, 4, 4]
    grid = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    nodes = [
        helper.make_node("Resize", ["grid", "", "scales"], ["position"], mode="nearest"),
        helper.make_node("Sqrt", ["head_dim"], ["scale"]),
        helper.make_node("Mul", ["image", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "position"], ["logits"]),
    ]
    graph = helper.make_graph(nodes, "dino", [helper.make_tensor_value_info("image", TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, shape)],
                              [numpy_helper.from_array(grid, "grid"),
                               numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "scales"),
                               numpy_helper.from_array(np.array(64, np.float32), "head_dim")])
    model = helper.make_model(graph, ir_version=9, opset_imports=[helper.make_opsetid("", 17)])
    source = tmp_path / "dino.onnx"
    onnx.save(model, source)
    manifest = source.with_suffix(".json")
    manifest.write_text(json.dumps({"model_path": source.name, "architecture": "dinov2_vitb14_seg",
                                    "input_name": "image", "output_name": "logits"}))
    result = convert_benchmark_fp16(manifest, tmp_path / "benchmark")
    path = result.parent / json.loads(result.read_text())["model_path"]
    converted = onnx.load(path)
    assert not {n.op_type for n in converted.graph.node}.intersection({"Resize", "Sqrt"})
    assert any(v.data_type == TensorProto.FLOAT16 for v in converted.graph.initializer)
    session = ort.InferenceSession(str(path), cpu_session_options(num_threads=1), providers=["CPUExecutionProvider"])
    inputs = np.ones((2, 1, 4, 4), dtype=np.float32)
    actual, = session.run(["logits"], {"image": inputs})
    np.testing.assert_array_equal(actual, inputs * 8 + grid.repeat(2, axis=2).repeat(2, axis=3))


@pytest.mark.parametrize("dynamic_grid", [False, True])
def test_mask2former_shared_position_grid_converts_to_runnable_fp16(tmp_path, dynamic_grid):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper
    from onnx_session import cpu_session_options

    # The row and column CumSum nodes in a position embedding share one Cast.
    ids = np.arange(4, dtype=np.int32).reshape(2, 2)
    shape = ["batch", 1, 2, 2]
    initializers = [numpy_helper.from_array(np.array(-2, np.int64), "axis0"),
                    numpy_helper.from_array(np.array(-1, np.int64), "axis1")]
    nodes = []
    if dynamic_grid:
        initializers.append(numpy_helper.from_array(np.array(.25, np.float32), "threshold"))
        nodes.append(helper.make_node("Greater", ["image", "threshold"], ["ids"]))
    else:
        initializers.append(numpy_helper.from_array(ids, "ids"))
    graph = helper.make_graph(nodes + [
        helper.make_node("Cast", ["ids"], ["grid"], to=TensorProto.FLOAT),
        helper.make_node("CumSum", ["grid", "axis0"], ["y"]),
        helper.make_node("CumSum", ["grid", "axis1"], ["x"]),
        helper.make_node("Add", ["x", "y"], ["position"]),
        helper.make_node("Add", ["image", "position"], ["logits"]),
    ], "shared_grid", [helper.make_tensor_value_info("image", TensorProto.FLOAT, shape)],
       [helper.make_tensor_value_info("logits", TensorProto.FLOAT, shape)],
       initializers)
    onnx.save(helper.make_model(graph, ir_version=9, opset_imports=[helper.make_opsetid("", 17)]),
              tmp_path / "source.onnx")
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({"model_path": "source.onnx", "architecture": "mask2former_swin_small",
                                    "input_name": "image", "output_name": "logits"}))
    config = convert_benchmark_fp16(manifest, tmp_path / "benchmark")
    model = config.parent / json.loads(config.read_text())["model_path"]
    session = ort.InferenceSession(str(model), cpu_session_options(num_threads=1), providers=["CPUExecutionProvider"])
    images = np.arange(-4, 4, dtype=np.float32).reshape(2, 1, 2, 2)
    actual, = session.run(None, {"image": images})
    grid = images > .25 if dynamic_grid else ids
    np.testing.assert_array_equal(actual, images + grid.cumsum(-2) + grid.cumsum(-1))


def _measure_with_unresponsive_fp16(path, cfg, threads, backend, paths, warmup, repeats,
                                    *, reference, baseline, cache, progress):
    import os
    import time

    progress(f"워밍업 1/{warmup} · {paths[0].name}")
    if backend == "onnxruntime" and cfg["export"]["precision"] == "float16":
        if cfg.get("simulate_crash"):
            os._exit(19)
        time.sleep(60)
    if not reference:
        assert baseline == ["fp32"]
    return {"samples": 1, "p50_ms": 1.0, "model_p50_ms": .5}, ["fp32"] if reference else None


@pytest.mark.parametrize("crash", [False, True])
def test_fp16_native_worker_failure_keeps_completed_rows_and_runs_remaining_comparisons(tmp_path, monkeypatch, crash):
    import multiprocessing
    from PIL import Image
    import precision_benchmark

    monkeypatch.setattr(precision_benchmark, "_measure", _measure_with_unresponsive_fp16)
    Image.new("RGB", (16, 16)).save(tmp_path / "시험.png")
    for name in ("float32", "float16", "int8"):
        settings = {"cpp_supported": True, "task": "classify", "backend": "custom",
                    "input_width": 16, "input_height": 16, "export": {"precision": name}}
        if name == "float16" and crash:
            settings["simulate_crash"] = True
        (tmp_path / f"{name}.json").write_text(json.dumps(settings), encoding="utf-8")

    existing_children = {child.pid for child in multiprocessing.active_children()}
    rows = []
    result = precision_benchmark.benchmark_precisions(
        tmp_path / "float32.json", tmp_path, thread_counts=(1,),
        fp16_path=tmp_path / "float16.json", int8_path=tmp_path / "int8.json",
        backends=("onnxruntime", "openvino"), image_count=1, warmup=2, repeats=1,
        stage_timeout=5, workspace=tmp_path, on_row=rows.append, log=lambda _: None,
    )
    assert result == rows
    assert [(row["backend"], row["precision"]) for row in rows] == [
        (backend, precision) for precision in ("FP32", "FP16", "INT8") for backend in ("onnxruntime", "openvino")
    ]
    assert rows[2]["status"] == ("실행 실패" if crash else "시간 초과")
    assert "워밍업 1/2 · 시험.png" in rows[2]["detail"]
    assert ("종료 코드 19" if crash else "5초") in rows[2]["detail"]
    assert all(row["status"] == "완료" and row["samples"] == 1 for row in rows[:2] + rows[3:])
    assert not list(tmp_path.glob("dvs-precision-benchmark-*"))
    assert {child.pid for child in multiprocessing.active_children()} <= existing_children
