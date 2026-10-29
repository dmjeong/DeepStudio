"""Built-in architectures with explicit ImageNet/local initialization.

Construction stays weight-free by default, including checkpoint restoration
and ONNX export. Only an explicit pretrained request may fetch weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BuiltinModelSpec:
    model_id: str
    task: str
    default_size: tuple[int, int]
    input_channels: tuple[int, ...]


BUILTIN_MODEL_SPECS = {
    "resnet18": BuiltinModelSpec("resnet18", "classify", (224, 224), (1, 3)),
    "resnet50": BuiltinModelSpec("resnet50", "classify", (224, 224), (1, 3)),
    "convnext_v1_tiny": BuiltinModelSpec("convnext_v1_tiny", "classify", (224, 224), (3,)),
    "deeplabv3plus_resnet34": BuiltinModelSpec("deeplabv3plus_resnet34", "segment", (512, 512), (3,)),
    "unet_resnet18": BuiltinModelSpec("unet_resnet18", "segment", (512, 512), (1, 3)),
    # DINOv2 (자기지도 ViT, 패치 14px). 224·448은 14와 32의 공배수라 학습 화면의 입력 크기 단계(32)와 맞는다.
    "dinov2_vits14_cls": BuiltinModelSpec("dinov2_vits14_cls", "classify", (224, 224), (1, 3)),
    "dinov2_vitb14_cls": BuiltinModelSpec("dinov2_vitb14_cls", "classify", (224, 224), (1, 3)),
    "dinov2_vits14_seg": BuiltinModelSpec("dinov2_vits14_seg", "segment", (448, 448), (1, 3)),
    "dinov2_vitb14_seg": BuiltinModelSpec("dinov2_vitb14_seg", "segment", (448, 448), (1, 3)),
}

# 새로 학습하는 머리 부분(나머지는 사전학습 백본: 백본 학습률 배수·동결 대상)과,
# 클래스 순서가 다른 체크포인트로 추가 학습할 때 버리는 마지막 분류층.
HEAD_PREFIXES = {
    "resnet18": ("fc.",), "resnet50": ("fc.",),
    "convnext_v1_tiny": ("classifier.",),
    "deeplabv3plus_resnet34": ("aspp.", "low_projection.", "decoder."),
    "unet_resnet18": ("dec", "head."),
    "dinov2_vits14_cls": ("head.",), "dinov2_vitb14_cls": ("head.",),
    "dinov2_vits14_seg": ("neck.", "detail", "fuse", "head."),
    "dinov2_vitb14_seg": ("neck.", "detail", "fuse", "head."),
}
CLASS_HEAD_PREFIX = {
    "resnet18": "fc.", "resnet50": "fc.", "convnext_v1_tiny": "classifier.2.",
    "deeplabv3plus_resnet34": "decoder.6.", "unet_resnet18": "head.",
    "dinov2_vits14_cls": "head.", "dinov2_vitb14_cls": "head.",
    "dinov2_vits14_seg": "head.", "dinov2_vitb14_seg": "head.",
}

# Pin a concrete revision rather than torchvision's mutable DEFAULT alias.
IMAGENET_WEIGHTS = {
    "resnet18": ("ResNet18_Weights", "IMAGENET1K_V1"),
    "resnet34": ("ResNet34_Weights", "IMAGENET1K_V1"),
    "resnet50": ("ResNet50_Weights", "IMAGENET1K_V2"),
    "convnext_tiny": ("ConvNeXt_Tiny_Weights", "IMAGENET1K_V1"),
}


def _torchvision_base(name, *, pretrained=False, weights_path=None):
    from torchvision import models
    from builtin_assets import builtin_asset_path

    if pretrained and weights_path:
        raise ValueError("ImageNet과 로컬 가중치를 동시에 지정할 수 없습니다")
    source = None
    provenance = {"source": "random", "backbone": name}
    if pretrained:
        enum, revision = IMAGENET_WEIGHTS[name]
        weights = getattr(getattr(models, enum), revision)
        key = {"resnet18": "resnet18", "resnet34": "resnet34", "resnet50": "resnet50",
               "convnext_tiny": "convnext_v1_tiny"}[name]
        source = builtin_asset_path(key)
        provenance.update(source="imagenet", weights=f"{enum}.{revision}", url=weights.url)
    elif weights_path:
        source = Path(weights_path)
        provenance.update(source="local_backbone", file=source.name)
    base = getattr(models, name)(weights=None)
    if source is not None:
        payload = torch.load(source, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        try:
            base.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"{name} 백본과 가중치 구조가 일치하지 않습니다: {source}") from exc
    base.weight_provenance = provenance
    return base

_GRADCAM_TARGET_LAYERS = {
    "resnet18": "layer4",
    "resnet50": "layer4",
    "convnext_v1_tiny": "features.7",
    "deeplabv3plus_resnet34": "layer4",
    "unet_resnet18": "enc4",
    "dinov2_vits14_cls": "feature_map", "dinov2_vitb14_cls": "feature_map",  # 패치 토큰을 펼친 (B, D, h, w) 지도
    "dinov2_vits14_seg": "neck", "dinov2_vitb14_seg": "neck",
}


def get_builtin_spec(model_id: str) -> BuiltinModelSpec:
    try:
        return BUILTIN_MODEL_SPECS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported built-in model adapter: {model_id}") from exc


def _replace_conv_in_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if in_channels == conv.in_channels:
        return conv
    replacement = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size,
                            conv.stride, conv.padding, dilation=conv.dilation,
                            groups=conv.groups, bias=conv.bias is not None,
                            padding_mode=conv.padding_mode)
    with torch.no_grad():
        if in_channels == 1 and conv.in_channels == 3:
            # Preserve the response to a gray image replicated across RGB.
            replacement.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(replacement.weight, mode="fan_out", nonlinearity="relu")
        if replacement.bias is not None:
            replacement.bias.copy_(conv.bias)
    return replacement


def _classification_model(model_id: str, num_classes: int, in_channels: int, **weights) -> nn.Module:
    if model_id == "resnet18":
        model = _torchvision_base("resnet18", **weights)
        model.conv1 = _replace_conv_in_channels(model.conv1, in_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_id == "resnet50":
        model = _torchvision_base("resnet50", **weights)
        model.conv1 = _replace_conv_in_channels(model.conv1, in_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_id == "convnext_v1_tiny":
        model = _torchvision_base("convnext_tiny", **weights)
        first = model.features[0][0]
        model.features[0][0] = _replace_conv_in_channels(first, in_channels)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_id in ("dinov2_vits14_cls", "dinov2_vitb14_cls"):
        return DinoV2Classifier(model_id.removeprefix("dinov2_").removesuffix("_cls"), num_classes, in_channels, **weights)
    raise ValueError(f"Unsupported classification adapter: {model_id}")


def _dinov2_backbone(variant, in_channels, *, pretrained=False, weights_path=None):
    """DINOv2 백본. pretrained면 공식 가중치(설치본·사용자 폴더), weights_path면 같은 키의 로컬 파일."""
    from builtin_assets import SOURCE_URLS, builtin_asset_path
    from dinov2 import VARIANTS, DinoV2ViT, load_pretrained

    if pretrained and weights_path:
        raise ValueError("공식 가중치와 로컬 가중치를 동시에 지정할 수 없습니다")
    backbone = DinoV2ViT(**VARIANTS[variant])
    provenance = {"source": "random", "backbone": f"dinov2_{variant}"}
    if pretrained:
        asset = f"dinov2_{variant}"
        load_pretrained(backbone, builtin_asset_path(asset))
        provenance.update(source="dinov2_lvd142m", weights=asset, url=SOURCE_URLS[asset])
    elif weights_path:
        load_pretrained(backbone, Path(weights_path))
        provenance.update(source="local_backbone", file=Path(weights_path).name)
    # 1채널(흑백): RGB 필터 합 → 회색을 RGB로 복제했을 때와 같은 응답 (ResNet과 같은 규칙)
    backbone.patch_embed.proj = _replace_conv_in_channels(backbone.patch_embed.proj, in_channels)
    backbone.weight_provenance = provenance
    return backbone


class DinoV2Classifier(nn.Module):
    """DINOv2 분류: [CLS 토큰, 패치 토큰 평균]을 이어 붙여 선형 분류 (DINOv2 선형 평가와 같은 특징)."""

    def __init__(self, variant: str, num_classes: int, in_channels: int = 3, **weights):
        super().__init__()
        self.backbone = _dinov2_backbone(variant, in_channels, **weights)
        self.weight_provenance = self.backbone.weight_provenance
        self.feature_map = nn.Identity()  # Grad-CAM이 붙는 (B, D, h, w) 패치 특징 지도
        self.head = nn.Linear(self.backbone.embed_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from dinov2 import tokens_to_map

        cls, patches, grid, _ = self.backbone(x)
        patch_map = self.feature_map(tokens_to_map(patches, grid))
        return self.head(torch.cat((cls, patch_map.mean(dim=(2, 3))), dim=1))


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()
        branches: list[nn.Module] = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        ]
        for dilation in (6, 12, 18):
            branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(branches), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeepLabV3PlusResNet34(nn.Module):
    """A torchvision ResNet-34 encoder with an explicit V3+ ASPP decoder."""

    def __init__(self, num_classes: int, in_channels: int = 3, **weights):
        super().__init__()
        base = _torchvision_base("resnet34", **weights)
        self.weight_provenance = base.weight_provenance
        base.conv1 = _replace_conv_in_channels(base.conv1, in_channels)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4
        self.aspp = _ASPP(512, 128)
        self.low_projection = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(176, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.interpolate(self.aspp(x), size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat((x, self.low_projection(low)), dim=1))
        return F.interpolate(x, size=original_size, mode="bilinear", align_corners=False)


class UNetResNet18(nn.Module):
    """U-Net decoder using the four spatial stages of a ResNet-18 encoder."""

    def __init__(self, num_classes: int, in_channels: int = 3, **weights):
        super().__init__()
        base = _torchvision_base("resnet18", **weights)
        self.weight_provenance = base.weight_provenance
        base.conv1 = _replace_conv_in_channels(base.conv1, in_channels)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.dec4 = _ConvBlock(512 + 256, 256)
        self.dec3 = _ConvBlock(256 + 128, 128)
        self.dec2 = _ConvBlock(128 + 64, 64)
        self.dec1 = _ConvBlock(64 + 64, 64)
        self.head = nn.Conv2d(64, num_classes, 1)

    @staticmethod
    def _merge(x: torch.Tensor, skip: torch.Tensor, block: nn.Module) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return block(torch.cat((x, skip), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        stem = self.stem(x)
        e1 = self.enc1(self.pool(stem))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        y = self._merge(e4, e3, self.dec4)
        y = self._merge(y, e2, self.dec3)
        y = self._merge(y, e1, self.dec2)
        y = self._merge(y, stem, self.dec1)
        return self.head(F.interpolate(y, size=original_size, mode="bilinear", align_corners=False))


class DinoV2Segmenter(nn.Module):
    """DINOv2 분할: ViT 의미 특징(1/14) + 얕은 합성곱 세부 경로(1/2·1/4)로 원본 해상도까지 복원.

        이미지 ─▶ DINOv2 블록 3·6·9·12 (1/14) ─▶ 이어 붙임 ─▶ neck 1×1 (256)
           │                                              │ 1/4로 확대
           ├─▶ detail1 3×3 s2 (32, 1/2) ─▶ detail2 3×3 s2 (64, 1/4) ─┴▶ fuse4 ─▶ 1/2로 확대 ─▶ fuse2 ─▶ head
           └──────────────────────────────────────────────────────────────────┘ (detail1과 합침)

    14px 패치 특징만으로는 경계가 14px 단위로 뭉개진다. 세부 경로(처음부터 학습)가 가장자리·가는 결함 위치를 준다.
    ViT 주의(attention)는 이미지 전체를 보므로 패치 학습 모델은 학습과 같은 크기의 타일로 추론한다(global_attention).
    """

    LAYERS = (2, 5, 8, 11)

    def __init__(self, variant: str, num_classes: int, in_channels: int = 3, **weights):
        super().__init__()
        self.backbone = _dinov2_backbone(variant, in_channels, **weights)
        self.weight_provenance = self.backbone.weight_provenance
        dim = self.backbone.embed_dim
        self.neck = nn.Sequential(nn.Conv2d(dim * len(self.LAYERS), 256, 1, bias=False),
                                  nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.detail1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.detail2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.fuse4 = _ConvBlock(256 + 64, 128)
        self.fuse2 = _ConvBlock(128 + 32, 64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from dinov2 import tokens_to_map

        height, width = int(x.shape[-2]), int(x.shape[-1])  # 고정 크기 ONNX 추출에서도 상수
        # 먼저 14의 배수로 채워(오른쪽·아래 0) 모든 경로를 같은 크기에서 계산하고, 마지막에 원래 크기로 자른다.
        # 그래야 ViT 격자(채운 크기 / 14)와 세부 경로·출력 픽셀이 정확히 겹친다.
        x = F.pad(x, (0, (-width) % 14, 0, (-height) % 14))
        _, _, grid, layers = self.backbone(x, layers=self.LAYERS)
        semantic = self.neck(torch.cat([tokens_to_map(tokens, grid) for tokens in layers], dim=1))
        half = self.detail1(x)
        quarter = self.detail2(half)
        y = F.interpolate(semantic, size=quarter.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse4(torch.cat((y, quarter), dim=1))
        y = F.interpolate(y, size=half.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse2(torch.cat((y, half), dim=1))
        y = F.interpolate(self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y[..., :height, :width]


def build_builtin_model(model_id: str, num_classes: int, in_channels: int = 3, *,
                        pretrained: bool = False, backbone_weights=None) -> nn.Module:
    """Construct the adapter; pretrained/local loading must be requested explicitly."""
    spec = get_builtin_spec(model_id)
    if isinstance(num_classes, bool) or int(num_classes) != num_classes or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if in_channels not in spec.input_channels:
        raise ValueError(f"{model_id} does not support {in_channels} input channels")
    if spec.task == "classify":
        model = _classification_model(model_id, int(num_classes), in_channels,
                                      pretrained=pretrained, weights_path=backbone_weights)
    elif model_id == "deeplabv3plus_resnet34":
        model = DeepLabV3PlusResNet34(int(num_classes), in_channels,
                                    pretrained=pretrained, weights_path=backbone_weights)
    elif model_id == "unet_resnet18":
        model = UNetResNet18(int(num_classes), in_channels,
                            pretrained=pretrained, weights_path=backbone_weights)
    elif model_id in ("dinov2_vits14_seg", "dinov2_vitb14_seg"):
        model = DinoV2Segmenter(model_id.removeprefix("dinov2_").removesuffix("_seg"), int(num_classes),
                                in_channels, pretrained=pretrained, weights_path=backbone_weights)
    else:
        raise ValueError(f"Unsupported segmentation adapter: {model_id}")

    # The desktop inference path consumes these attributes for every model.
    # Attach the same contract to fresh and checkpoint-loaded built-ins.
    model.task = spec.task
    model.in_channels = int(in_channels)
    model.num_classes = int(num_classes)
    model.gradcam_target_layer = _GRADCAM_TARGET_LAYERS[model_id]
    # ViT는 주의(attention)가 입력 전체를 본다: 패치 학습 모델을 큰 영역에 한 번에 넣으면 학습과 다른 함수가 된다.
    # 추론 엔진은 이 표시가 있으면 학습한 패치 크기의 타일로 추론한다 (합성곱 모델은 한 번에).
    model.global_attention = model_id.startswith("dinov2_")
    # ViT 미세조정은 합성곱보다 훨씬 작은 백본 학습률이 필요하다. 클래스당 10장 분류(20 epoch)에서
    # 백본 lr 1e-4(기본 배수 0.1)는 정확도가 우연 수준(0.33)으로 무너졌고, 1e-5(×0.1 더)는 0.97이었다.
    model.backbone_lr_scale = 0.1 if model_id.startswith("dinov2_") else 1.0
    return model


def make_builtin_checkpoint(model_id: str, model: nn.Module, *, num_classes: int,
                            input_size: int | Iterable[int] | None = None,
                            in_channels: int = 3, class_names: Iterable[str] = (),
                            tiling: dict | None = None) -> dict:
    """Create a self-describing checkpoint for the generic ONNX exporter.

    tiling: 패치 학습 모델의 타일 추론 규칙 (checkpoint.tiling_contract). None이면 전체 이미지 리사이즈.
    """
    spec = get_builtin_spec(model_id)
    if model.training:
        raise ValueError("model must be in eval mode before deployment checkpoint creation")
    size = list(spec.default_size if input_size is None else
                ((input_size, input_size) if isinstance(input_size, int) else input_size))
    if len(size) != 2 or any(isinstance(v, bool) or int(v) != v or int(v) < 32 for v in size):
        raise ValueError("input_size must contain two integers >= 32")
    class_names = list(class_names)
    if len(class_names) not in (0, num_classes):
        raise ValueError("class_names must be empty or match num_classes")
    payload = {
        "type": "builtin_model", "backend": "builtin", "model_id": model_id,
        "task": spec.task, "num_classes": int(num_classes), "in_channels": int(in_channels),
        "input_size": [int(size[0]), int(size[1])], "class_names": class_names,
        "model_state_dict": model.state_dict(),
        "model_config": {"model_id": model_id},
        "weight_provenance": dict(getattr(model, "weight_provenance", {})),
        "preprocessing": {"input_size": [int(size[0]), int(size[1])],
                           "in_channels": int(in_channels), "mean": [0.449] if in_channels == 1 else [0.485, 0.456, 0.406],
                           "std": [0.226] if in_channels == 1 else [0.229, 0.224, 0.225],
                           "resize": "bilinear", "resize_implementation": "opencv_linear_exact_v1",
                           "interpolation": "INTER_LINEAR_EXACT", "antialias": False,
                           "layout": "NCHW", "value_scale": 255.0,
                           "color_order": "GRAY" if in_channels == 1 else "RGB"},
    }
    if tiling:
        if spec.task != "segment":
            raise ValueError("타일 추론은 분할 모델만 지원합니다")
        payload["tiling"] = dict(tiling)
        payload["preprocessing"]["tiling"] = dict(tiling)
    return payload


def load_builtin_checkpoint(checkpoint: dict) -> nn.Module:
    model_id = checkpoint.get("model_id") or (checkpoint.get("model_config") or {}).get("model_id")
    if not isinstance(model_id, str):
        raise ValueError("builtin checkpoint model_id is required")
    model = build_builtin_model(model_id, int(checkpoint["num_classes"]), int(checkpoint["in_channels"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.weight_provenance = dict(checkpoint.get("weight_provenance", {}))
    return model.cpu().eval()
