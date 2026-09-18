"""Built-in model adapters with weight-free, reproducible architectures.

The product catalog names the model family and variant; this module is the
single Python construction point used by training/export code.  No pretrained
weights are downloaded here.  Callers may load a separately licensed
``state_dict`` after constructing an adapter and must keep its source metadata
in the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    nn.init.kaiming_normal_(replacement.weight, mode="fan_out", nonlinearity="relu")
    if replacement.bias is not None:
        nn.init.zeros_(replacement.bias)
    return replacement


def _classification_model(model_id: str, num_classes: int, in_channels: int) -> nn.Module:
    from torchvision import models

    if model_id == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = _replace_conv_in_channels(model.conv1, in_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_id == "resnet50":
        model = models.resnet50(weights=None)
        model.conv1 = _replace_conv_in_channels(model.conv1, in_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_id == "convnext_v1_tiny":
        model = models.convnext_tiny(weights=None)
        first = model.features[0][0]
        model.features[0][0] = _replace_conv_in_channels(first, in_channels)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported classification adapter: {model_id}")


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

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        from torchvision import models

        base = models.resnet34(weights=None)
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

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        from torchvision import models

        base = models.resnet18(weights=None)
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


def build_builtin_model(model_id: str, num_classes: int, in_channels: int = 3) -> nn.Module:
    """Construct a model without fetching or embedding pretrained weights."""
    spec = get_builtin_spec(model_id)
    if isinstance(num_classes, bool) or int(num_classes) != num_classes or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if in_channels not in spec.input_channels:
        raise ValueError(f"{model_id} does not support {in_channels} input channels")
    if spec.task == "classify":
        return _classification_model(model_id, int(num_classes), in_channels)
    if model_id == "deeplabv3plus_resnet34":
        return DeepLabV3PlusResNet34(int(num_classes), in_channels)
    if model_id == "unet_resnet18":
        return UNetResNet18(int(num_classes), in_channels)
    raise ValueError(f"Unsupported segmentation adapter: {model_id}")


def make_builtin_checkpoint(model_id: str, model: nn.Module, *, num_classes: int,
                            input_size: int | Iterable[int] | None = None,
                            in_channels: int = 3, class_names: Iterable[str] = ()) -> dict:
    """Create a self-describing checkpoint for the generic ONNX exporter."""
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
    return {
        "type": "builtin_model", "backend": "builtin", "model_id": model_id,
        "task": spec.task, "num_classes": int(num_classes), "in_channels": int(in_channels),
        "input_size": [int(size[0]), int(size[1])], "class_names": class_names,
        "model_state_dict": model.state_dict(),
        "model_config": {"model_id": model_id},
        "preprocessing": {"input_size": [int(size[0]), int(size[1])],
                           "in_channels": int(in_channels), "mean": [0.449] if in_channels == 1 else [0.485, 0.456, 0.406],
                           "std": [0.226] if in_channels == 1 else [0.229, 0.224, 0.225],
                           "resize": "bilinear", "resize_implementation": "opencv_linear_exact_v1",
                           "interpolation": "INTER_LINEAR_EXACT", "antialias": False,
                           "layout": "NCHW", "value_scale": 255.0,
                           "color_order": "GRAY" if in_channels == 1 else "RGB"},
    }


def load_builtin_checkpoint(checkpoint: dict) -> nn.Module:
    model_id = checkpoint.get("model_id") or (checkpoint.get("model_config") or {}).get("model_id")
    if not isinstance(model_id, str):
        raise ValueError("builtin checkpoint model_id is required")
    model = build_builtin_model(model_id, int(checkpoint["num_classes"]), int(checkpoint["in_channels"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.cpu().eval()
