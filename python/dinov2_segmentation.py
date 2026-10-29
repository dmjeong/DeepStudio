"""DINOv2 ViT-S/14 with a native-resolution semantic segmentation decoder.

Backbone adapted from facebookresearch/dinov2 (Apache-2.0).
Copyright (c) Meta Platforms, Inc. and affiliates.
Changes: only the non-register small architecture; standard PyTorch attention;
no remote code, xformers, DINO training machinery or implicit downloads.
The decoder/gray-input adapter are Deep Vision Studio additions.
See licenses/DINOv2-APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.
"""

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
# Release pin computed from the official HTTPS artifact, not a publisher signature.
DINOV2_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 384, kernel_size=14, stride=14)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(384, 1152)
        self.proj = nn.Linear(384, 384)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, 6, 64).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # PyTorch dispatches an efficient CPU/CUDA kernel; ONNX lowers to
        # standard MatMul/Softmax operations, with no custom runtime plugin.
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(y.transpose(1, 2).reshape(batch, tokens, channels))


class _LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(384))

    def forward(self, x):
        return x * self.gamma


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 1536)
        self.fc2 = nn.Linear(1536, 384)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(384, eps=1e-6)
        self.attn = _Attention()
        self.ls1 = _LayerScale()
        self.norm2 = nn.LayerNorm(384, eps=1e-6)
        self.mlp = _Mlp()
        self.ls2 = _LayerScale()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoV2SmallBackbone(nn.Module):
    """Exact official checkpoint key/shape contract (ViT-S/14, no registers)."""

    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, 384))  # 518/14 = 37
        self.patch_embed = _PatchEmbed()
        self.blocks = nn.ModuleList([_Block() for _ in range(12)])
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.mask_token = nn.Parameter(torch.zeros(1, 384), requires_grad=False)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def _position(self, height, width, dtype):
        # Spatial dimensions are deliberately fixed in the exported model.
        # Keep Meta's historical 0.1 interpolation offset for weight fidelity.
        height, width = int(height), int(width)
        if height == 518 and width == 518:
            return self.pos_embed.to(dtype)
        patches = self.pos_embed[:, 1:].float().reshape(1, 37, 37, 384).permute(0, 3, 1, 2)
        patches = F.interpolate(patches, scale_factor=((height // 14 + 0.1) / 37,
                                                       (width // 14 + 0.1) / 37),
                                mode="bicubic", align_corners=False, antialias=False)
        patches = patches.permute(0, 2, 3, 1).reshape(1, -1, 384)
        return torch.cat((self.pos_embed[:, :1], patches), dim=1).to(dtype)

    def forward(self, x):
        batch, _, height, width = x.shape
        tokens = self.patch_embed(x)
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1), tokens), dim=1)
        tokens = tokens + self._position(height, width, tokens.dtype)
        outputs = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in (2, 5, 8, 11):
                y = self.norm(tokens)[:, 1:]
                outputs.append(y.transpose(1, 2).reshape(batch, 384, height // 14, width // 14))
        return outputs


class DinoV2Segmentation(nn.Module):
    """Pretrained semantic features plus a trainable, stride-one detail path."""

    def __init__(self, num_classes, in_channels=3, *, pretrained=False, weights_path=None):
        super().__init__()
        if pretrained and weights_path:
            raise ValueError("사전학습과 로컬 가중치를 동시에 지정할 수 없습니다")
        if in_channels not in (1, 3):
            raise ValueError("DINOv2 입력 채널은 1 또는 3이어야 합니다")
        self.in_channels = in_channels
        self.backbone = DinoV2SmallBackbone()
        self.weight_provenance = {"source": "random", "backbone": "dinov2_vits14"}
        source = weights_path
        if pretrained:
            from builtin_assets import builtin_asset_path
            source = builtin_asset_path("dinov2_vits14")
        if source is not None:
            payload = torch.load(source, map_location="cpu", weights_only=True)
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            try:
                self.backbone.load_state_dict(state, strict=True)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(f"DINOv2 ViT-S/14 공식 백본과 가중치 구조가 다릅니다: {source}") from exc
            self.weight_provenance.update(source="dinov2_lvd142m" if pretrained else "local_backbone",
                                          file=Path(source).name)
            if pretrained:
                self.weight_provenance.update(url=DINOV2_URL, sha256=DINOV2_SHA256, license="Apache-2.0")
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        self.projections = nn.ModuleList([nn.Conv2d(384, 32, 1) for _ in range(4)])
        self.fusion = nn.Sequential(nn.Conv2d(128, 64, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.detail = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Conv2d(80, 32, 1), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.ReLU())
        self.head = nn.Conv2d(32, num_classes, 1)

    def backbone_input(self, x):
        # Existing gray deployment normalization -> exactly replicated RGB
        # ImageNet normalization. The same adapter is part of the ONNX graph.
        return ((x * 0.226 + 0.449) - self.rgb_mean) / self.rgb_std if self.in_channels == 1 else x

    def forward(self, x):
        height, width = x.shape[-2:]
        rgb = self.backbone_input(x)
        rgb = F.pad(rgb, (0, (14 - width % 14) % 14, 0, (14 - height % 14) % 14), mode="replicate")
        features = self.backbone(rgb)
        semantic = self.fusion(torch.cat([projection(y) for projection, y in zip(self.projections, features)], dim=1))
        # Upsample the padded field, then crop. Resizing straight to the
        # unpadded size would shift the pixel alignment at the right/bottom.
        semantic = F.interpolate(semantic, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        semantic = semantic[:, :, :height, :width]
        return self.head(self.decoder(torch.cat((semantic, self.detail(x)), dim=1)))
