"""네트워크 다운로드 없이 C++ 규약 테스트용 작은 ONNX 그래프 생성."""
import json
import hashlib
from pathlib import Path
import shutil
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
                                "layout": "NCHW", "value_scale": 255., "color_order": "GRAY"},
              "cpp_supported": True}
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

# Re-DETR v4 two-output contract: normalized cx/xy boxes and class logits.
# Keeping this graph constant makes the native contract test deterministic and
# avoids bundling model weights or a third-party checkpoint.
redetr_boxes = helper.make_tensor("pred_boxes_value", TensorProto.FLOAT, [1, 1, 4],
                                  [0.5, 0.5, 0.5, 0.5])
redetr_logits = helper.make_tensor("pred_logits_value", TensorProto.FLOAT, [1, 1, 2],
                                   [10.0, -10.0])
redetr_nodes = [helper.make_node("Constant", [], ["pred_boxes"], value=redetr_boxes),
                helper.make_node("Constant", [], ["pred_logits"], value=redetr_logits)]
redetr_outputs = [helper.make_tensor_value_info("pred_boxes", TensorProto.FLOAT, [1, 1, 4]),
                  helper.make_tensor_value_info("pred_logits", TensorProto.FLOAT, [1, 1, 2])]
redetr_model = helper.make_model(helper.make_graph(redetr_nodes, "redetr_v4", inputs, redetr_outputs),
                                 opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(redetr_model)
onnx.save(redetr_model, root / "redetr.onnx")
redetr_config = dict(detect_config)
redetr_config.update({"backend": "redetr_v4", "model_path": "redetr.onnx",
                      "output_name": "pred_boxes", "output_names": ["pred_boxes", "pred_logits"],
                      "postprocessing": {"box_format": "normalized_cxcywh", "objectness": "none",
                                         "class_scores": "sigmoid", "confidence_threshold": 0.25,
                                         "iou_threshold": 0.5, "max_detections": 10}})
(root / "redetr.json").write_text(json.dumps(redetr_config, ensure_ascii=False), encoding="utf-8")
redetr_softmax_config = json.loads(json.dumps(redetr_config))
redetr_softmax_config["postprocessing"]["class_scores"] = "softmax"
(root / "redetr_softmax.json").write_text(json.dumps(redetr_softmax_config, ensure_ascii=False), encoding="utf-8")

# Minimal SAM2-style encoder/decoder graphs.  They keep every declared prompt
# input in the graph contract while returning deterministic masks/scores, so
# the C++/C ABI multi-graph lifecycle is tested without shipping weights.
sam_input = helper.make_tensor_value_info("input_image", TensorProto.FLOAT, [1, 3, 2, 3])
sam_embedding = helper.make_tensor_value_info("image_embeddings", TensorProto.FLOAT, [1, 1, 1, 1])
sam_encoder = helper.make_model(
    helper.make_graph([helper.make_node("ReduceMean", ["input_image"], ["image_embeddings"],
                                           axes=[1, 2, 3], keepdims=1)],
                      "sam2_encoder", [sam_input], [sam_embedding]),
    opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(sam_encoder)
onnx.save(sam_encoder, root / "sam2_encoder.onnx")
decoder_inputs = [
    helper.make_tensor_value_info("image_embeddings", TensorProto.FLOAT, [1, 1, 1, 1]),
    helper.make_tensor_value_info("point_coords", TensorProto.FLOAT, [1, "points", 2]),
    helper.make_tensor_value_info("point_labels", TensorProto.INT64, [1, "points"]),
    helper.make_tensor_value_info("mask_input", TensorProto.FLOAT, [1, 1, 2, 2]),
    helper.make_tensor_value_info("has_mask_input", TensorProto.FLOAT, [1]),
    helper.make_tensor_value_info("orig_im_size", TensorProto.FLOAT, [2]),
]
sam_logits_value = helper.make_tensor("sam_logits_value", TensorProto.FLOAT, [1, 2, 2, 2],
                                      [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
sam_scores_value = helper.make_tensor("sam_scores_value", TensorProto.FLOAT, [1, 2], [0.1, 0.9])
sam_decoder = helper.make_model(
    helper.make_graph([
        helper.make_node("Constant", [], ["low_res_mask_logits"], value=sam_logits_value),
        helper.make_node("Constant", [], ["iou_predictions"], value=sam_scores_value),
    ], "sam2_decoder", decoder_inputs,
       [helper.make_tensor_value_info("low_res_mask_logits", TensorProto.FLOAT, [1, 2, 2, 2]),
        helper.make_tensor_value_info("iou_predictions", TensorProto.FLOAT, [1, 2])]),
    opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(sam_decoder)
onnx.save(sam_decoder, root / "sam2_decoder.onnx")
sam_config = {
    "schema_version": 5, "backend": "sam2", "task": "segment", "model_path": "sam2_encoder.onnx",
    "num_classes": 1, "input_channels": 3, "input_height": 2, "input_width": 3,
    "input_name": "input_image", "output_name": "low_res_mask_logits",
    "normalize_mean": [0.485, 0.456, 0.406], "normalize_std": [0.229, 0.224, 0.225],
    "class_names": [], "preprocessing": {"resize_implementation": "opencv_linear_exact_v1",
        "resize": "bilinear", "interpolation": "INTER_LINEAR_EXACT", "antialias": False,
        "layout": "NCHW", "value_scale": 255., "color_order": "RGB"},
    "contracts": {"graphs": {
        "encoder": {"file": "sam2_encoder.onnx", "outputs": ["image_embeddings"]},
        "decoder": {"file": "sam2_decoder.onnx", "inputs": {
            "image_embeddings": "image_embeddings", "point_coords": "point_coords",
            "point_labels": "point_labels", "mask_input": "mask_input",
            "has_mask_input": "has_mask_input", "orig_im_size": "orig_im_size"},
            "outputs": ["low_res_mask_logits", "iou_predictions"]}},
        "prompt_types": ["point", "box", "mask"], "video_state": False, "mask_size": [2, 2]},
}
(root / "sam2.json").write_text(json.dumps(sam_config, ensure_ascii=False), encoding="utf-8")

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

# A native SDK fixture must be self-verifying without importing Python at
# runtime. Keep the manifest intentionally small while covering both the
# config and graph bytes.
bundle = root / "classify.dvdeploy"
shutil.rmtree(bundle, ignore_errors=True)
bundle.mkdir()
for name in ("classify.json", "classify.onnx"):
    (bundle / name).write_bytes((root / name).read_bytes())
files = {}
for path in sorted(bundle.iterdir()):
    files[path.name] = {"size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
(bundle / "manifest.json").write_text(json.dumps({
    "schema_version": 1, "bundle_type": "onnx-deployment", "config": "classify.json",
    "backend": "custom", "task": "classify", "verification": "passed", "files": files,
}, sort_keys=True), encoding="utf-8")
