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
    if task == "classify":
        builtin_config = dict(config)
        builtin_config["backend"] = "builtin"
        (root / "builtin.json").write_text(json.dumps(builtin_config, ensure_ascii=False), encoding="utf-8")

# Generic detection contract: one normalized cx/xy box, objectness and two
# class logits.  It exercises C++ sigmoid decoding and class-aware NMS without
# downloading a model.
detect_values = helper.make_tensor("detections", TensorProto.FLOAT, [1, 1, 7],
                                   [0.5, 0.5, 0.5, 0.5, 10.0, 10.0, -10.0])
detect_node = helper.make_node("Constant", [], ["detections"], value=detect_values)
detect_output = helper.make_tensor_value_info("detections", TensorProto.FLOAT, [1, 1, 7])
detect_model = helper.make_model(helper.make_graph([detect_node], "detect", inputs, [detect_output]),
                                 opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(detect_model)
onnx.save(detect_model, root / "detect.onnx")
detect_config = json.loads((root / "classify.json").read_text(encoding="utf-8"))
detect_config.update({"model_path": "detect.onnx", "task": "detect", "num_classes": 2,
                      "output_name": "detections", "class_names": ["box", "other"],
                      "postprocessing": {"box_format": "normalized_cxcywh", "objectness": "sigmoid",
                                         "class_scores": "sigmoid", "confidence_threshold": 0.25,
                                         "iou_threshold": 0.5, "max_detections": 10}})
(root / "detect.json").write_text(json.dumps(detect_config, ensure_ascii=False), encoding="utf-8")

# Reconstruction anomaly contract: identity output yields a zero error map.
anomaly_node = helper.make_node("Identity", ["input_image"], ["reconstruction"])
anomaly_output = helper.make_tensor_value_info("reconstruction", TensorProto.FLOAT, [1, 1, 2, 3])
anomaly_model = helper.make_model(helper.make_graph([anomaly_node], "anomaly", inputs, [anomaly_output]),
                                  opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(anomaly_model)
onnx.save(anomaly_model, root / "anomaly.onnx")
anomaly_config = json.loads((root / "classify.json").read_text(encoding="utf-8"))
anomaly_config.update({"model_path": "anomaly.onnx", "task": "anomaly", "num_classes": 1,
                       "output_name": "reconstruction", "class_names": [],
                       "postprocessing": {"score": "mean_absolute_reconstruction_error",
                                          "threshold": 0.1, "map": "per_pixel_mean_absolute_error"}})
(root / "anomaly.json").write_text(json.dumps(anomaly_config, ensure_ascii=False), encoding="utf-8")

# PatchCore-style two-output contract: score plus a single-channel anomaly map.
patch_score = helper.make_tensor("anomaly_score_value", TensorProto.FLOAT, [1], [0.75])
patch_map = helper.make_tensor("anomaly_map_value", TensorProto.FLOAT, [1, 1, 2, 3],
                               [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
patch_nodes = [helper.make_node("Constant", [], ["anomaly_score"], value=patch_score),
               helper.make_node("Constant", [], ["anomaly_map"], value=patch_map)]
patch_outputs = [helper.make_tensor_value_info("anomaly_score", TensorProto.FLOAT, [1]),
                 helper.make_tensor_value_info("anomaly_map", TensorProto.FLOAT, [1, 1, 2, 3])]
patch_model = helper.make_model(helper.make_graph(patch_nodes, "patchcore", inputs, patch_outputs),
                               opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(patch_model)
onnx.save(patch_model, root / "patchcore.onnx")
patch_config = dict(anomaly_config)
patch_config.update({"model_path": "patchcore.onnx", "backend": "patchcore",
                     "output_name": "anomaly_score", "output_names": ["anomaly_score", "anomaly_map"],
                     "postprocessing": {"score": "patchcore_smoothed_knn_max", "threshold": 0.5,
                                        "memory_bank_size": 4, "n_neighbors": 2},
                     "cpp_supported": True})
(root / "patchcore.json").write_text(json.dumps(patch_config, ensure_ascii=False), encoding="utf-8")

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
