import test from "node:test";
import assert from "node:assert/strict";
import { modelModes } from "../src/model-options.ts";

test("다른 모델에 EfficientNet 가중치 선택을 표시하지 않는다", () => {
  for (const [task, id] of [["classify", "resnet18"], ["classify", "convnext_v1_tiny"],
    ["segment", "unet_resnet18"], ["segment", "sam2_hiera_large"], ["detect", "re_detr_v4_small"]]) {
    const modes = modelModes(task, id, id);
    assert.ok(modes.every(([key, label]) => !key.startsWith("efficientnet") && !label.includes("EfficientNet")));
  }
  assert.equal(modelModes("classify", "resnet50", "ResNet 50")[0][0], "builtin_finetune");
  assert.equal(modelModes("segment", "deeplabv3plus_resnet34", "DeepLab")[0][0], "builtin_finetune");
  assert.equal(modelModes("classify", "efficientnet_b1", "EfficientNet B1")[0][0], "efficientnet_finetune");
});
