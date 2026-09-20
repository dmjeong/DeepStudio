"""Regenerate the tiny synthetic ONNX self-test. Requires onnx; no pretrained weights."""
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main():
    root = Path(__file__).resolve().parent / "assets"
    root.mkdir(exist_ok=True)
    initializers = [helper.make_tensor(name, TensorProto.FLOAT, shape, values)
                    for name, shape, values in (
                        ("w", [1, 1, 1, 1], [1e8]), ("b", [1], [-1e8]),
                        ("scale", [1], [1]), ("beta", [1], [1]),
                        ("mean", [1], [0]), ("variance", [1], [1]))]
    # White input: 1 * 1e8 - 1e8 + 1 = 1, logits [1, -1].
    # A Conv/BN fusion can lose the +1 in FP32. Thus this fixture also detects
    # consumers ignoring JSON's saved graph_optimization_level="disabled".
    nodes = [helper.make_node("Conv", ["input_image", "w", "b"], ["conv"], kernel_shape=[1, 1]),
             helper.make_node("BatchNormalization", ["conv", "scale", "beta", "mean", "variance"], ["bn"], epsilon=0.),
             helper.make_node("ReduceMean", ["bn"], ["positive"], axes=[2, 3], keepdims=0),
             helper.make_node("Neg", ["positive"], ["negative"]),
             helper.make_node("Concat", ["positive", "negative"], ["class_logits"], axis=1)]
    graph = helper.make_graph(nodes, "example_self_test",
        [helper.make_tensor_value_info("input_image", TensorProto.FLOAT, [1, 1, 224, 224])],
        [helper.make_tensor_value_info("class_logits", TensorProto.FLOAT, [1, 2])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, root / "test.onnx")
    config = {
        "schema_version": 6, "backend": "custom", "task": "classify",
        "model_path": "test.onnx", "num_classes": 2, "class_names": ["white", "other"],
        "input_channels": 1, "input_height": 224, "input_width": 224,
        "input_name": "input_image", "output_name": "class_logits",
        "normalize_mean": [0.], "normalize_std": [1.], "num_threads": 1,
        "onnxruntime": {"graph_optimization_level": "disabled"}, "cpp_supported": True,
        "preprocessing": {"resize_implementation": "opencv_linear_exact_v1", "resize": "bilinear",
                          "interpolation": "INTER_LINEAR_EXACT", "antialias": False,
                          "layout": "NCHW", "value_scale": 255., "color_order": "GRAY"},
    }
    (root / "test.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    pixels = bytes([255]) * (224 * 224)
    (root / "white.raw").write_bytes(pixels)
    (root / "white.pgm").write_bytes(b"P5\n224 224\n255\n" + pixels)


if __name__ == "__main__":
    main()
