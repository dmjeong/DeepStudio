"""Native comparison models: thin U-Nets, BiSeNetV2 and STDC1.

Constructors never download weights. All deployment forwards return only
full-resolution logits; auxiliary/boundary heads run only in forward_train.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MODEL_IDS = (
    "unet_mobilenetv4_small",
    "unet_shufflenetv2",
    "unet_efficientnet_b1",
    "bisenetv2",
    "stdc1",
)
SCRATCH_IDS = frozenset({"bisenetv2", "stdc1"})
TIMM_BACKBONES = {
    "unet_mobilenetv4_small": (
        "mobilenetv4_conv_small.e2400_r224_in1k",
        "mobilenetv4_small_encoder",
    ),
    "unet_efficientnet_b1": (
        "efficientnet_b1.ra4_e3600_r240_in1k",
        "efficientnet_b1_encoder",
    ),
}


class SmallBatchNorm2d(nn.BatchNorm2d):
    """A pooled 1x1 context must also train with a final one-image batch."""

    def forward(self, x):
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(x)


def _small_batch_norm(module):
    for name, child in list(module.named_children()):
        if type(child) is nn.BatchNorm2d:
            replacement = SmallBatchNorm2d(
                child.num_features,
                child.eps,
                child.momentum,
                child.affine,
                child.track_running_stats,
            )
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            _small_batch_norm(child)


def _block(cin, cout, kernel=3):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
        SmallBatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class _TimmEncoder(nn.Module):
    def __init__(self, model_id, in_channels, pretrained, weights_path):
        super().__init__()
        import timm
        from builtin_assets import builtin_asset_path
        from builtin_models import _replace_conv_in_channels

        name, asset = TIMM_BACKBONES[model_id]
        if pretrained and weights_path:
            raise ValueError("사전학습과 로컬 백본 가중치를 동시에 지정할 수 없습니다")
        source = (
            builtin_asset_path(asset)
            if pretrained
            else Path(weights_path)
            if weights_path
            else None
        )
        base = timm.create_model(name, pretrained=False, features_only=True)
        self.weight_provenance = {
            "source": "random",
            "backbone": name,
            "timm": timm.__version__,
        }
        if source is not None:
            state = torch.load(source, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
            # Validate the full official classifier first, then copy every encoder
            # tensor. A partial or unrelated state_dict cannot silently succeed.
            full = timm.create_model(name, pretrained=False)
            try:
                full.load_state_dict(state, strict=True)
                base.load_state_dict(
                    {key: state[key] for key in base.state_dict()}, strict=True
                )
            except (RuntimeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"{name} 공식 백본과 가중치 구조가 다릅니다: {source}"
                ) from exc
            self.weight_provenance.update(
                source="imagenet" if pretrained else "local_backbone",
                file=source.name,
                asset=asset,
            )
        base.conv_stem = _replace_conv_in_channels(base.conv_stem, in_channels)
        self.base = base
        self.is_mobile = model_id == "unet_mobilenetv4_small"
        if self.is_mobile:
            # Like the existing V2 U-Net, omit the classification-only final
            # pointwise projection (960 channels for V4, 1280 for V2).
            self.base.blocks = self.base.blocks[:4]
            self.channels = (32, 32, 64, 96, 128)
        else:
            self.channels = tuple(base.feature_info.channels())

    def forward(self, x):
        if not self.is_mobile:
            return self.base(x)
        x = self.base.act1(self.base.bn1(self.base.conv_stem(x)))
        features = [x]
        for block in self.base.blocks:
            x = block(x)
            features.append(x)
        return features


class _ShuffleEncoder(nn.Module):
    def __init__(self, in_channels, pretrained, weights_path):
        super().__init__()
        from builtin_models import _torchvision_base, _replace_conv_in_channels

        base = _torchvision_base(
            "shufflenet_v2_x1_0", pretrained=pretrained, weights_path=weights_path
        )
        base.conv1[0] = _replace_conv_in_channels(base.conv1[0], in_channels)
        self.layers = nn.ModuleList(
            [base.conv1, base.maxpool, base.stage2, base.stage3, base.stage4]
        )
        self.channels = (24, 24, 116, 232, 464)
        self.weight_provenance = base.weight_provenance

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class ThinUNet(nn.Module):
    def __init__(
        self,
        model_id,
        num_classes,
        in_channels=3,
        *,
        pretrained=False,
        weights_path=None,
    ):
        super().__init__()
        self.encoder = (
            _ShuffleEncoder(in_channels, pretrained, weights_path)
            if model_id == "unet_shufflenetv2"
            else _TimmEncoder(model_id, in_channels, pretrained, weights_path)
        )
        self.weight_provenance = self.encoder.weight_provenance
        # Channel-shuffle reshapes must retain Shape operators so calibration
        # can use smaller tiles than the fixed deployment ROI.
        self.export_dynamic_spatial = model_id == "unet_shufflenetv2"
        self.dec = nn.ModuleList()
        channels = self.encoder.channels
        ci = channels[-1]
        for skip, co in zip(reversed(channels[:-1]), (64, 64, 32, 16)):
            self.dec.append(nn.Sequential(_block(ci + skip, co, 1), _block(co, co)))
            ci = co
        self.head = nn.Conv2d(ci, num_classes, 1)
        _small_batch_norm(self)

    def forward(self, x):
        skips = self.encoder(x)
        y = skips[-1]
        for skip, block in zip(reversed(skips[:-1]), self.dec):
            y = block(
                torch.cat(
                    (
                        F.interpolate(
                            y,
                            size=skip.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                        skip,
                    ),
                    dim=1,
                )
            )
        return F.interpolate(
            self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False
        )


class BiSeNetV2Segmenter(nn.Module):
    training_loss_config = {"auxiliary_segmentation_weight": 0.4}

    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        from segmentation_backbones.bisenetv2 import BiSeNetV2
        from builtin_models import _replace_conv_in_channels

        self.net = BiSeNetV2(num_classes, aux_mode="train")
        self.net.detail.S1[0].conv = _replace_conv_in_channels(
            self.net.detail.S1[0].conv, in_channels
        )
        self.net.segment.S1S2.conv.conv = _replace_conv_in_channels(
            self.net.segment.S1S2.conv.conv, in_channels
        )
        self.weight_provenance = {"source": "random", "backbone": "bisenetv2"}
        _small_batch_norm(self)

    def _features(self, x):
        x = F.pad(x, (0, (-x.shape[-1]) % 32, 0, (-x.shape[-2]) % 32))
        detail = self.net.detail(x)
        f2, f3, f4, f5, semantic = self.net.segment(x)
        return self.net.head(self.net.bga(detail, semantic)), (f2, f3, f4, f5)

    def forward(self, x):
        main, _ = self._features(x)
        return main[..., : x.shape[-2], : x.shape[-1]]

    def forward_train(self, x):
        main, features = self._features(x)
        heads = (self.net.aux2, self.net.aux3, self.net.aux4, self.net.aux5_4)

        def crop(y):
            return y[..., : x.shape[-2], : x.shape[-1]]

        return crop(main), [crop(head(f)) for head, f in zip(heads, features)], None


class STDCSegmenter(nn.Module):
    training_loss_config = {
        "auxiliary_segmentation_weight": 0.4,
        "boundary_weight": 1.0,
    }

    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        from segmentation_backbones.stdc_model import BiSeNet
        from builtin_models import _replace_conv_in_channels

        base = BiSeNet(
            "STDCNet813", num_classes, pretrain_model="", use_boundary_8=True
        )
        self.cp, self.ffm = base.cp, base.ffm
        self.head, self.aux16, self.aux32 = (
            base.conv_out,
            base.conv_out16,
            base.conv_out32,
        )
        self.boundary = base.conv_out_sp8
        first = self.cp.backbone.features[0]
        first.conv = _replace_conv_in_channels(first.conv, in_channels)
        self.weight_provenance = {"source": "random", "backbone": "STDCNet813"}
        _small_batch_norm(self)

    def _features(self, x):
        _, _, f8, _, cp8, cp16 = self.cp(x)
        main = self.head(self.ffm(f8, cp8))
        return F.interpolate(
            main, size=x.shape[-2:], mode="bilinear", align_corners=True
        ), (f8, cp8, cp16)

    def forward(self, x):
        return self._features(x)[0]

    def forward_train(self, x):
        main, (f8, cp8, cp16) = self._features(x)
        aux = [
            F.interpolate(
                head(f), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            for head, f in ((self.aux16, cp8), (self.aux32, cp16))
        ]
        boundary = F.interpolate(
            self.boundary(f8), size=x.shape[-2:], mode="bilinear", align_corners=True
        )
        return main, aux, boundary


def boundary_loss(logits, masks):
    """Class-ID-independent detail targets. Ignore excluded pixels and their edges.

    STDC-style BCE + Dice detail supervision. Neighbour disagreement instead of
    a Laplacian of numeric IDs avoids changing targets when class IDs are renamed.
    """
    valid = masks != 255
    target = torch.zeros_like(valid)
    for dy, dx in ((0, 1), (1, 0)):
        left = masks[:, : masks.shape[1] - dy or None, : masks.shape[2] - dx or None]
        right = masks[:, dy:, dx:]
        edge = (left != right) & (left != 255) & (right != 255)
        target[:, : masks.shape[1] - dy or None, : masks.shape[2] - dx or None] |= edge
        target[:, dy:, dx:] |= edge
    # Exclude the neighbourhood of ignored pixels from boundary supervision.
    valid = valid & (F.max_pool2d((~valid).float().unsqueeze(1), 3, 1, 1)[:, 0] == 0)
    x = logits[:, 0].float()
    truth = target.float()
    bce = (
        F.binary_cross_entropy_with_logits(x, truth, reduction="none") * valid
    ).sum() / valid.sum().clamp(min=1)
    pred = x.sigmoid() * valid
    truth = truth * valid
    dice = 1 - (2 * (pred * truth).sum() + 1) / (pred.sum() + truth.sum() + 1)
    return bce + dice


def build_segment_candidate(
    model_id, num_classes, in_channels, *, pretrained=False, weights_path=None
):
    if model_id in SCRATCH_IDS:
        if pretrained or weights_path:
            raise ValueError(
                "이 모델은 기본 사전학습 파일을 제공하지 않습니다. 처음부터 학습하거나 같은 모델의 Studio .pt를 선택하세요."
            )
        cls = BiSeNetV2Segmenter if model_id == "bisenetv2" else STDCSegmenter
        return cls(num_classes, in_channels)
    return ThinUNet(
        model_id,
        num_classes,
        in_channels,
        pretrained=pretrained,
        weights_path=weights_path,
    )
