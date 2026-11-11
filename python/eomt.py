"""EoMT DINOv2-S semantic adapter (MIT; licenses/EoMT-LICENSE.txt).

Derived from tue-mps/eomt aa017c34: query tokens enter the last four ViT blocks,
mask-guided self attention, query class/mask heads and stride-two ScaleBlock.
Studio adaptation: variable input padding, full-pixel Hungarian loss (no random
point sampling), and a stride-one detail residual for narrow inspection defects.
"""

import torch
from torch import nn
from torch.nn import functional as F


def prepare_tokens(backbone, image):
    h, w = int(image.shape[-2]), int(image.shape[-1])
    ph, pw = (-h) % 14, (-w) % 14
    x = F.pad(image, (0, pw, 0, ph))
    grid = ((h + ph) // 14, (w + pw) // 14)
    tokens = backbone.patch_embed(x)
    tokens = torch.cat([backbone.cls_token.expand(x.shape[0], -1, -1), tokens], dim=1)
    return tokens + backbone._position(*grid, tokens.dtype), grid


class EoMTSegmenter(nn.Module):
    global_attention = True
    backbone_lr_scale = 0.1

    def __init__(self, classes, channels, *, pretrained=False, weights_path=None):
        super().__init__()
        from builtin_models import _dinov2_backbone
        from timm.layers import LayerNorm2d

        self.backbone = _dinov2_backbone(
            "vits14", channels, pretrained=pretrained, weights_path=weights_path
        )
        self.weight_provenance = self.backbone.weight_provenance
        dim = self.backbone.embed_dim
        self.num_queries = max(16, classes * 4)
        self.classes = classes
        self.q = nn.Embedding(self.num_queries, dim)
        self.class_head = nn.Linear(dim, classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, 2),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim),
        )
        self.detail = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, classes, 1)
        )

    def _predict(self, tokens, grid):
        tokens = self.backbone.norm(tokens)
        queries = tokens[:, : self.num_queries]
        patches = tokens[:, self.num_queries + 1 :].transpose(1, 2)
        features = patches.reshape(tokens.shape[0], tokens.shape[-1], *grid)
        masks = torch.einsum(
            "bqc,bchw->bqhw", self.mask_head(queries), self.upscale(features)
        )
        return masks, self.class_head(queries)

    def query_outputs(self, image):
        x, grid = prepare_tokens(self.backbone, image)
        predictions = []
        start = len(self.backbone.blocks) - 4
        for i, block in enumerate(self.backbone.blocks):
            if i < start:
                x = block(x)
                continue
            if i == start:
                x = torch.cat(
                    [self.q.weight.unsqueeze(0).expand(x.shape[0], -1, -1), x], dim=1
                )
            masks, classes = self._predict(x, grid)
            if self.training:
                predictions.append((masks, classes))
            allowed = (
                F.interpolate(
                    masks, size=grid, mode="bilinear", align_corners=False
                ).flatten(2)
                > 0
            )
            # Query/prefix columns are always visible; patch rows see all tokens.
            left = torch.ones(
                (x.shape[0], self.num_queries, self.num_queries + 1),
                dtype=torch.bool,
                device=x.device,
            )
            top = torch.cat([left, allowed], dim=2)
            bottom = torch.ones(
                (x.shape[0], grid[0] * grid[1] + 1, x.shape[1]),
                dtype=torch.bool,
                device=x.device,
            )
            mask = torch.cat([top, bottom], dim=1).unsqueeze(1)
            normalized = block.norm1(x)
            b, n, dim = normalized.shape
            heads = block.attn.num_heads
            qkv = block.attn.qkv(normalized).reshape(b, n, 3, heads, dim // heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attention = block.attn.proj(attention.transpose(1, 2).reshape(b, n, dim))
            x = x + block.ls1(attention)
            x = x + block.ls2(block.mlp(block.norm2(x)))
        predictions.append(self._predict(x, grid))
        return predictions

    def semantic(self, prediction, image):
        masks, classes = prediction
        scores = torch.einsum(
            "bqc,bqhw->bchw", classes.softmax(-1)[..., :-1], masks.sigmoid()
        )
        scores = F.interpolate(
            scores,
            size=tuple(((s + 13) // 14) * 14 for s in image.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        scores = scores[..., : image.shape[-2], : image.shape[-1]]
        # Log scores expose the usual softmax/logits contract to CE, ONNX and C++.
        return scores.clamp_min(1e-7).log() + self.detail(image)

    def forward(self, image):
        return self.semantic(self.query_outputs(image)[-1], image)

    def compute_training_loss(self, image, masks, criterion):
        predictions = self.query_outputs(image)
        logits = self.semantic(predictions[-1], image)
        loss = criterion(logits, masks)
        loss = loss + sum(
            query_loss(*prediction, masks) for prediction in predictions
        ) / len(predictions)
        return logits, loss


def query_loss(pred_masks, pred_classes, target):
    """One target mask per semantic class, background included, ignore pixels excluded."""
    from scipy.optimize import linear_sum_assignment

    total = pred_masks.sum() * 0
    for batch in range(target.shape[0]):
        valid = target[batch] != 255
        if not valid.any():
            continue
        labels = torch.unique(target[batch][valid])
        masks = F.interpolate(
            pred_masks[batch : batch + 1],
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, :, valid]
        truth = (target[batch][None] == labels[:, None, None])[:, valid].float()
        probabilities = masks.sigmoid()
        dice_cost = 1 - (2 * probabilities @ truth.T + 1) / (
            probabilities.sum(1, keepdim=True) + truth.sum(1)[None] + 1
        )
        bce_cost = (
            F.softplus(masks).mean(1, keepdim=True) - masks @ truth.T / truth.shape[1]
        )
        cost = (
            5 * (dice_cost + bce_cost) - 2 * pred_classes[batch].softmax(-1)[:, labels]
        )
        row, col = linear_sum_assignment(cost.detach().float().cpu().numpy())
        row = torch.as_tensor(row, device=target.device)
        col = torch.as_tensor(col, device=target.device)
        classes = torch.full(
            (pred_masks.shape[1],),
            pred_classes.shape[-1] - 1,
            dtype=torch.long,
            device=target.device,
        )
        classes[row] = labels[col]
        weights = torch.ones(pred_classes.shape[-1], device=target.device)
        weights[-1] = 0.1
        total = total + 2 * F.cross_entropy(
            pred_classes[batch], classes, weight=weights
        )
        total = (
            total
            + 5 * F.binary_cross_entropy_with_logits(masks[row], truth[col])
            + 5 * dice_cost[row, col].mean()
        )
    return total / target.shape[0]
