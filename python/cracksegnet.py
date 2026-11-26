"""CrackSegNet Studio: original, paper-inspired dense segmentation implementation.

Reference: Yu, Mei, Zhu (2026), doi:10.3390/rs18213712. This is NOT the
authors' implementation or a reproduction of their reported scores. See
docs/CRACKSEGNET_STUDIO.md for the exact architecture and deviations.

No external graph/deformable-convolution extension or pretrained download.
Deformable sampling uses standard ONNX GridSample (opset >= 16).
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from segmentation_models import SmallBatchNorm2d


class ConvAct(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, dilation=1):
        super().__init__(
            nn.Conv2d(cin, cout, kernel, padding=dilation * (kernel // 2),
                      dilation=dilation, bias=False),
            SmallBatchNorm2d(cout), nn.ReLU(inplace=False),
        )


class DeformConv2d(nn.Module):
    """Stride-1, 3x3 deformable convolution; unmodulated, one offset group.

    Offsets are interleaved (dy, dx), as in torchvision.ops.deform_conv2d.
    Pack nine sampling positions along width for ONE GridSample invocation.
    Neither images nor learned offsets are detached: both receive gradients.
    """
    def __init__(self, cin, cout):
        super().__init__()
        self.offset = nn.Conv2d(cin, 18, 3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.weight = nn.Parameter(torch.empty(cout, cin, 3, 3))
        self.bias = nn.Parameter(torch.zeros(cout))
        nn.init.kaiming_uniform_(self.weight, a=5 ** .5)
        self.register_buffer("kernel_y", torch.tensor([-1., -1., -1., 0., 0., 0., 1., 1., 1.]))
        self.register_buffer("kernel_x", torch.tensor([-1., 0., 1., -1., 0., 1., -1., 0., 1.]))

    def forward(self, x):
        offsets = self.offset(x)
        batch, channels, height, width = x.shape
        # FP32 coordinates also during AMP; subpixel locations must not be
        # rounded to bf16/half (particularly on large native-resolution inputs).
        with torch.autocast(device_type=x.device.type, enabled=False):
            offsets = offsets.float().reshape(batch, 9, 2, height, width)
            dy = offsets[:, :, 0].permute(0, 2, 3, 1)
            dx = offsets[:, :, 1].permute(0, 2, 3, 1)
            yy = torch.arange(height, device=x.device, dtype=torch.float32).reshape(1, height, 1, 1)
            xx = torch.arange(width, device=x.device, dtype=torch.float32).reshape(1, 1, width, 1)
            y = (yy + self.kernel_y.float() + dy + .5) * (2. / height) - 1.
            z = (xx + self.kernel_x.float() + dx + .5) * (2. / width) - 1.
            grid = torch.stack((z, y), dim=-1).reshape(batch, height, width * 9, 2)
            sampled = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros",
                                    align_corners=False)
            sampled = sampled.reshape(batch, channels, height, width, 9)
            sampled = sampled.permute(0, 1, 4, 2, 3).reshape(batch, channels * 9, height, width)
            return F.conv2d(sampled, self.weight.float().flatten(1)[:, :, None, None], self.bias.float())


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 8)
        self.channel = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(),
                                     nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        channel = torch.sigmoid(self.channel(x.mean((2, 3), keepdim=True)) +
                                self.channel(x.amax((2, 3), keepdim=True)))
        x = x * channel
        spatial = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)
        return x * torch.sigmoid(self.spatial(spatial))


class MultiScaleBlock(nn.Module):
    def __init__(self, cin, cout, *, deformable=False):
        super().__init__()
        hidden = max(8, cout // 4)
        self.reduce = ConvAct(cin, hidden, 1)
        self.pointwise = ConvAct(hidden, hidden, 1)
        self.spatial = ConvAct(hidden, hidden)
        self.dilated = ConvAct(hidden, hidden, dilation=2)
        self.deform = DeformConv2d(hidden, hidden) if deformable else ConvAct(hidden, hidden, dilation=3)
        self.fuse = ConvAct(4 * hidden, cout, 1)
        self.attention = CBAM(cout)
        self.shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        reduced = self.reduce(x)
        combined = torch.cat((self.pointwise(reduced), self.spatial(reduced),
                              self.dilated(reduced), self.deform(reduced)), 1)
        return F.relu(self.shortcut(x) + self.attention(self.fuse(combined)))


class LocalGraphAttention(nn.Module):
    """Single-head graph attention over a dense grid's 8 neighbors and self.

    Missing boundary neighbors are masked (never wrap across image borders).
    Fixed local edges avoid sparse-scatter custom operators in ONNX/C++.
    """
    OFFSETS = ((0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1))

    def __init__(self, cin, cout=128):
        super().__init__()
        import math
        self.project = nn.Conv2d(cin, cout, 1, bias=False)
        self.source_score = nn.Conv2d(cout, 1, 1, bias=False)
        self.target_score = nn.Conv2d(cout, 1, 1, bias=False)
        self.edge_score = nn.Linear(4, 1, bias=False)
        self.register_buffer("edges", torch.tensor([
            [dy, dx, math.hypot(dy, dx), math.atan2(dy, dx)] for dy, dx in self.OFFSETS
        ], dtype=torch.float32))
        self.norm = nn.LayerNorm(cout)

    def forward(self, x):
        x = self.project(x)
        height, width = x.shape[-2:]
        padded = F.pad(x, (1, 1, 1, 1))
        valid = F.pad(torch.ones_like(x[:, :1]), (1, 1, 1, 1))
        center_score = self.source_score(x)
        edge_scores = self.edge_score(self.edges).flatten()
        neighbors, scores = [], []
        for i, (dy, dx) in enumerate(self.OFFSETS):
            neighbor = padded[:, :, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
            mask = valid[:, :, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
            score = F.leaky_relu(center_score + self.target_score(neighbor) + edge_scores[i], .2)
            scores.append(torch.where(mask > 0, score, torch.full_like(score, -10000.)))
            neighbors.append(neighbor)
        weights = torch.softmax(torch.stack(scores, 2), dim=2)
        aggregated = (torch.stack(neighbors, 2) * weights).sum(2)
        return F.relu(self.norm(aggregated.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))


class UpBlock(nn.Module):
    def __init__(self, cin, cout, *, deformable):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(cin, cout * 4, 1), nn.PixelShuffle(2))
        self.fuse = MultiScaleBlock(cout * 2, cout, deformable=deformable)

    def forward(self, x, skip):
        return self.fuse(torch.cat((self.up(x), skip), 1))


class CrackSegNetStudio(nn.Module):
    global_attention = True  # CBAM depends on the entire training window.
    export_dynamic_spatial = True
    minimum_onnx_opset = 16

    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        widths = (16, 24, 48, 96, 160, 256)
        self.stem = MultiScaleBlock(in_channels, widths[0])
        self.down = nn.ModuleList([
            nn.Sequential(nn.PixelUnshuffle(2), MultiScaleBlock(cin * 4, cout, deformable=i >= 3))
            for i, (cin, cout) in enumerate(zip(widths, widths[1:]))
        ])
        self.graph1 = LocalGraphAttention(widths[-1])
        self.graph2 = LocalGraphAttention(128)
        self.graph_fuse = ConvAct(256, widths[-1], 1)
        self.up = nn.ModuleList([
            UpBlock(cin, cout, deformable=i < 2)
            for i, (cin, cout) in enumerate(zip(reversed(widths[1:]), reversed(widths[:-1])))
        ])
        self.head = nn.Conv2d(widths[0], num_classes, 1)
        self.weight_provenance = {
            "source": "random", "implementation": "cracksegnet_studio_v1",
            "reference": "https://doi.org/10.3390/rs18213712",
            "official_implementation": False, "pretrained": False,
        }

    def forward(self, x):
        height, width = x.shape[-2:]
        # Padding, NOT resizing: keep the original pixel scale and crop logits.
        x = F.pad(x, (0, (32 - width % 32) % 32, 0, (32 - height % 32) % 32))
        x = self.stem(x)
        skips = [x]
        for down in self.down:
            x = down(x)
            skips.append(x)
        a = self.graph1(x)
        b = self.graph2(a)
        x = x + self.graph_fuse(torch.cat((a, b), 1))
        for up, skip in zip(self.up, reversed(skips[:-1])):
            x = up(x, skip)
        return self.head(x)[:, :, :height, :width]
