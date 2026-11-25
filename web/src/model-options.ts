export const builtinAdapters = new Set([
  "dinov2_vits14_cls", "dinov2_vitb14_cls", "dinov2_vits14", "dinov2_vits14_seg", "dinov2_vitb14_seg",
  "unet_lite", "unet_mobilenetv2", "unet_mobilenetv4_small", "unet_shufflenetv2", "unet_efficientnet_b1",
  "bisenetv2", "stdc1", "efficientvit_b0_seg", "efficientvit_b1_seg", "sctnet_s", "eomt_dinov2_small", "segformer_b2",
  "resnet18", "resnet50", "convnext_v1_tiny", "deeplabv3plus_resnet34", "unet_resnet18", "mask2former_swin_small",
]);

export function modelModes(task: string, modelId: string, name: string): [string, string][] {
  if (task === "segment" && modelId.startsWith("sam2_hiera_")) return [
    ["sam2_finetune", "SAM2 기본 제공 가중치로 미세조정"],
    ["sam2_transfer", "SAM2 로컬 미세조정 체크포인트로 시작"],
  ];
  if (modelId === "dinomaly_vits14") return [
    ["builtin_finetune", "DINOv2 백본으로 시작"],
    ["builtin_transfer", "로컬 가중치로 시작"],
    ["builtin_resume", "중단 재개"],
  ];
  if (modelId === "unet_lite" || modelId === "segformer_b2") return [
    ["builtin_scratch", "처음부터 학습"],
    ["builtin_transfer", "로컬 가중치로 시작"],
    ["builtin_resume", "중단 재개"],
  ];
  if (modelId === "mask2former_swin_small") return [
    ["builtin_finetune", `${name} ImageNet 백본으로 시작`],
    ["builtin_transfer", `${name} 로컬 가중치로 시작`],
    ["builtin_resume", `${name} 중단 재개`],
    ["builtin_scratch", `${name} 처음부터 학습`],
  ];
  if ((modelId.startsWith("libreyolo_") || modelId.startsWith("re_detr_v4_")) && ["classify", "detect"].includes(task)) return [
    ["upstream_finetune", `${name} 사전학습 가중치로 시작`],
    ["upstream_transfer", `${name} 로컬 가중치로 추가 학습`],
    ["upstream_resume", `${name} 중단 재개`],
    ["upstream_scratch", `${name} 처음부터 학습`],
  ];
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
