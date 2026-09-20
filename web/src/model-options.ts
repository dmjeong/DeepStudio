export const builtinAdapters = new Set([
  "resnet18", "resnet50", "convnext_v1_tiny", "deeplabv3plus_resnet34", "unet_resnet18",
]);

export function modelModes(task: string, modelId: string, name: string): [string, string][] {
  if (builtinAdapters.has(modelId)) return [
    ["builtin_finetune", `${name} ImageNet 가중치로 시작`],
    ["builtin_transfer", `${name} 로컬 가중치로 시작`],
    ["custom", `${name} 처음부터 학습`],
  ];
  if (task === "classify" && (modelId.startsWith("efficientnet_") || !modelId)) return [
    ["efficientnet_finetune", "EfficientNet 사전학습 모델로 시작"],
    ["efficientnet_transfer", "EfficientNet 내 가중치로 추가 학습"],
    ["efficientnet_resume", "EfficientNet 중단한 학습 재개"],
    ["custom", "Custom CSP"],
  ];
  return [["custom", modelId && !modelId.startsWith("patchcore_") ? `${name} 모델 팩` : "Custom CSP"]];
}
