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
