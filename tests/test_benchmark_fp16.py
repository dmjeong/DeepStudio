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
