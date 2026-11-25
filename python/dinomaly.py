"""Dinomaly DINOv2-S/14 port, Apache-2.0 (licenses/Dinomaly-LICENSE.txt).

guojiajeremy/Dinomaly 1f252be0: frozen layers 2..9, dropout MLP bottleneck,
eight linear-attention reconstruction blocks and two fused cosine feature maps.
The small, non-register encoder is an explicit deployment variant, not the
published ViT-B/register benchmark. No nearest-neighbour memory bank is used.
"""

import torch
from eomt import prepare_tokens
from torch import nn
from torch.nn import functional as F


class NoiseMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(self.drop(x))))))


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=6):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, dim = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(b, n, 3, self.heads, dim // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        q, k = F.elu(q) + 1, F.elu(k) + 1
        kv = k.transpose(-2, -1) @ v
        denominator = (
            (q * k.sum(-2, keepdim=True)).sum(-1, keepdim=True).clamp_min(1e-8)
        )
        return self.proj(((q @ kv) / denominator).transpose(1, 2).reshape(b, n, dim))


class DecoderBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-8)
        self.attn = LinearAttention(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-8)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Dinomaly(nn.Module):
    def __init__(self, channels=3, *, pretrained=False, weights_path=None):
        super().__init__()
        from builtin_models import _dinov2_backbone

        self.encoder = _dinov2_backbone(
            "vits14", channels, pretrained=pretrained, weights_path=weights_path
        )
        self.encoder.requires_grad_(False)
        self.weight_provenance = self.encoder.weight_provenance
        dim = self.encoder.embed_dim
        self.bottleneck = NoiseMLP(dim)
        self.decoder = nn.ModuleList(DecoderBlock(dim) for _ in range(8))
        for module in (self.bottleneck, self.decoder):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.trunc_normal_(child.weight, std=0.01)
                    nn.init.zeros_(child.bias)
        self.in_channels = channels
        self.task = "anomaly"

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def features(self, image):
        with torch.no_grad():
            x, grid = prepare_tokens(self.encoder, image)
            encoded = []
            for i, block in enumerate(self.encoder.blocks[:10]):
                x = block(x)
                if i >= 2:
                    encoded.append(x)
        x = self.bottleneck(torch.stack(encoded).mean(0))
        decoded = []
        for block in self.decoder:
            x = block(x)
            decoded.append(x)
        decoded.reverse()

        def groups(features):
            return [
                torch.stack(features[start : start + 4])
                .mean(0)[:, 1:]
                .transpose(1, 2)
                .reshape(image.shape[0], -1, *grid)
                for start in (0, 4)
            ]

        return groups(encoded), groups(decoded)

    def training_loss(self, image, hard_fraction=0.9):
        en, de = self.features(image)
        losses = []
        for source, reconstructed in zip(en, de):
            if self.training and hard_fraction > 0:
                with torch.no_grad():
                    distances = 1 - F.cosine_similarity(source, reconstructed, dim=1)
                    threshold = (
                        distances.flatten()
                        .topk(max(1, int(distances.numel() * (1 - hard_fraction))))
                        .values[-1]
                    )
                    weights = torch.where(distances >= threshold, 1.0, 0.1).unsqueeze(1)
                reconstructed.register_hook(
                    lambda grad, weights=weights: grad * weights
                )
            losses.append(
                (
                    1
                    - F.cosine_similarity(
                        source.flatten(1), reconstructed.flatten(1), dim=1
                    )
                ).mean()
            )
        return torch.stack(losses).mean()

    def forward(self, image):
        en, de = self.features(image)
        anomaly = (
            torch.stack(
                [1 - F.cosine_similarity(a, b, dim=1, eps=1e-8) for a, b in zip(en, de)]
            )
            .mean(0)
            .unsqueeze(1)
        )
        anomaly = F.interpolate(
            anomaly,
            size=tuple(((s + 13) // 14) * 14 for s in image.shape[-2:]),
            mode="bilinear",
            align_corners=True,
        )
        anomaly = anomaly[..., : image.shape[-2], : image.shape[-1]]
        # Native cosine map; no per-image normalization that could erase normal/NG separation.
        anomaly = anomaly.clamp_min(0)
        return anomaly.flatten(1).amax(1), anomaly


def load_checkpoint(checkpoint, device="cpu"):
    if (
        checkpoint.get("model_id") != "dinomaly_vits14"
        or checkpoint.get("architecture_version") != 1
    ):
        raise ValueError("지원하지 않는 Dinomaly 체크포인트 구조입니다")
    model = Dinomaly(int(checkpoint["in_channels"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.weight_provenance = checkpoint.get("weight_provenance", {})
    return model.to(device).eval()
