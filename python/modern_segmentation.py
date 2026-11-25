"""CPU-portable dense models. No runtime downloads; logits always match input size.

EfficientViT uses Han Lab's multi-scale additive/MBConv segmentation head.
SCTNet retains the official CFBlock/DAPPM backbone. EoMT is in eomt.py.
Full-resolution detail branches are Studio additions, not published benchmarks.
"""

from pathlib import Path
from typing import ClassVar

import torch
from segmentation_models import _block, _small_batch_norm
from torch import nn
from torch.nn import functional as F

MODEL_IDS = (
    "efficientvit_b0_seg",
    "efficientvit_b1_seg",
    "sctnet_s",
    "eomt_dinov2_small",
)


def load_weights(module, asset, pretrained, weights_path):
    from builtin_assets import builtin_asset_path, file_sha256

    if pretrained and weights_path:
        raise ValueError("사전학습과 로컬 백본을 동시에 선택할 수 없습니다")
    source = (
        builtin_asset_path(asset)
        if pretrained
        else Path(weights_path)
        if weights_path
        else None
    )
    if source is None:
        return {"source": "random", "backbone": asset}
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    state = dict(state)
    # Official SCT-S ImageNet archive also contains a detached classification
    # head named literally "eight" / "ias". Accept only those exact shapes.
    if asset == "sctnet_s_backbone":
        for key, shape in (("eight", (1000, 128)), ("ias", (1000,))):
            if key in state:
                if tuple(state[key].shape) != shape:
                    raise ValueError(f"SCTNet 분류 헤드 형식 오류: {key}")
                del state[key]
    if any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("백본 가중치에 비유한 값이나 잘못된 텐서가 있습니다")
    module.load_state_dict(state, strict=True)
    return {
        "source": "pretrained" if pretrained else "local_backbone",
        "asset": asset,
        "sha256": file_sha256(source),
        "file": source.name,
    }


class DetailFusion(nn.Module):
    """Keep an explicit stride-one path for narrow defects."""

    def __init__(self, in_channels, features, classes):
        super().__init__()
        self.detail = nn.Sequential(_block(in_channels, 8), _block(8, 8))
        self.project = nn.Conv2d(features, 16, 1)
        self.head = nn.Conv2d(24, classes, 1)

    def forward(self, features, image):
        features = F.interpolate(
            self.project(features),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.head(torch.cat([features, self.detail(image)], dim=1))


class EfficientViTSegmenter(nn.Module):
    global_attention = True

    def __init__(
        self, variant, classes, channels, *, pretrained=False, weights_path=None
    ):
        super().__init__()
        import timm
        from builtin_models import _replace_conv_in_channels
        from timm.models.efficientvit_mit import MBConv

        name = f"efficientvit_{variant}.r224_in1k"
        full = timm.create_model(name, pretrained=False)
        self.weight_provenance = load_weights(
            full, f"efficientvit_{variant}", pretrained, weights_path
        )
        # Keep native module names (features_only rewrites names). Classification head is discarded.
        self.backbone = full
        self.backbone.head = nn.Identity()
        self.backbone.stem.in_conv.conv = _replace_conv_in_channels(
            self.backbone.stem.in_conv.conv, channels
        )
        widths = (32, 64, 128) if variant == "b0" else (64, 128, 256)
        width, depth = (32, 1) if variant == "b0" else (64, 3)
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, width, 1, bias=False), nn.BatchNorm2d(width))
            for c in widths
        )
        self.middle = nn.ModuleList(
            MBConv(
                width,
                width,
                expand_ratio=4,
                act_layer=(nn.Hardswish, nn.Hardswish, None),
            )
            for _ in range(depth)
        )
        self.expand = nn.Sequential(
            nn.Conv2d(width, width * 4, 1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.Hardswish(),
        )
        self.fusion = DetailFusion(channels, width * 4, classes)
        _small_batch_norm(self)

    def forward(self, x):
        f = self.backbone.stem(x)
        features = []
        for stage in self.backbone.stages:
            f = stage(f)
            features.append(f)
        selected = features[-3:]
        size = selected[0].shape[-2:]
        merged = sum(
            F.interpolate(p(f), size=size, mode="bilinear", align_corners=False)
            for p, f in zip(self.projections, selected)
        )
        for block in self.middle:
            merged = merged + block(merged)
        return self.fusion(self.expand(merged), x)


class SCTNetSegmenter(nn.Module):
    global_attention = (
        True  # spatial softmax and pooled context depend on the input extent
    )
    training_loss_config: ClassVar[dict] = {"auxiliary_segmentation_weight": 0.4}

    def __init__(self, classes, channels, *, pretrained=False, weights_path=None):
        super().__init__()
        from builtin_models import _replace_conv_in_channels
        from segmentation_backbones.sctnet import SCTNet

        self.backbone = SCTNet(base_channels=32, spp_channels=64)
        self.weight_provenance = load_weights(
            self.backbone, "sctnet_s_backbone", pretrained, weights_path
        )
        self.backbone.conv1[0] = _replace_conv_in_channels(
            self.backbone.conv1[0], channels
        )
        self.head = DetailFusion(channels, 128, classes)
        self.aux = nn.Conv2d(64, classes, 1)
        _small_batch_norm(self)

    def forward(self, x):
        return self.head(self.backbone(x)[0], x)

    def forward_train(self, x):
        main, aux, _ = self.backbone(x)
        return (
            self.head(main, x),
            [
                F.interpolate(
                    self.aux(aux),
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            ],
            None,
        )


def build_modern_segmenter(model_id, classes, channels, **weights):
    if model_id.startswith("efficientvit_"):
        return EfficientViTSegmenter(
            model_id.split("_")[1], classes, channels, **weights
        )
    if model_id == "sctnet_s":
        return SCTNetSegmenter(classes, channels, **weights)
    if model_id == "eomt_dinov2_small":
        from eomt import EoMTSegmenter

        return EoMTSegmenter(classes, channels, **weights)
    raise ValueError(f"Unknown modern segmenter: {model_id}")
