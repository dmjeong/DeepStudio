"""네트워크 다운로드 없이 C++ 규약 테스트용 작은 ONNX 그래프 생성."""
import json
from pathlib import Path
import sys

import onnx
from onnx import TensorProto, helper

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
inputs = [helper.make_tensor_value_info("input_image", TensorProto.FLOAT, [1, 1, 2, 3])]
for task in ("classify", "segment"):
    if task == "classify":
        nodes = [helper.make_node("ReduceMean", ["input_image"], ["mean"], axes=[2, 3], keepdims=0),
                 helper.make_node("Neg", ["mean"], ["negative"]),
                 helper.make_node("Concat", ["mean", "negative"], ["class_logits"], axis=1)]
        output = helper.make_tensor_value_info("class_logits", TensorProto.FLOAT, [1, 2])
    else:
        nodes = [helper.make_node("Neg", ["input_image"], ["negative"]),
                 helper.make_node("Concat", ["input_image", "negative"], ["seg_mask"], axis=1)]
        output = helper.make_tensor_value_info("seg_mask", TensorProto.FLOAT, [1, 2, 2, 3])
    model = helper.make_model(helper.make_graph(nodes, task, inputs, [output]),
                              opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    model_name = f"{task}.onnx"
    onnx.save(model, root / model_name)
    config = {"schema_version": 5, "backend": "custom", "model_path": model_name,
              "task": task, "num_classes": 2, "input_channels": 1,
              "input_height": 2, "input_width": 3,
              "input_name": "input_image", "output_name": output.name,
              "normalize_mean": [0.449], "normalize_std": [0.226],
              "class_names": ['OK "quoted"', 'NG'],
              "preprocessing": {"resize_implementation": "opencv_linear_exact_v1", "resize": "bilinear",
                                "interpolation": "INTER_LINEAR_EXACT", "antialias": False,
                                "layout": "NCHW", "value_scale": 255., "color_order": "GRAY"}}
    (root / f"{task}.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

# Minimal retired-backend manifest: the loader must reject it without executing an unsupported model.
retired = json.loads((root / "classify.json").read_text(encoding="utf-8"))
retired["backend"] = "unsupported"
(root / "unsupported.json").write_text(json.dumps(retired), encoding="utf-8")

# Tiny unequal logits expose ranking errors after FP32 softmax rounds to a tie.
values = helper.make_tensor("tiny_logits", TensorProto.FLOAT, [1, 2], [-1e-9, 1e-9])
node = helper.make_node("Constant", [], ["class_logits"], value=values)
output = helper.make_tensor_value_info("class_logits", TensorProto.FLOAT, [1, 2])
model = helper.make_model(helper.make_graph([node], "close_logits", inputs, [output]),
                         opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, root / "close_logits.onnx")
config = json.loads((root / "classify.json").read_text(encoding="utf-8"))
config["model_path"] = "close_logits.onnx"
(root / "close_logits.json").write_text(json.dumps(config), encoding="utf-8")
