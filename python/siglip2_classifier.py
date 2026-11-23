"""SigLIP 2 vision-only classifier with Deep Vision Studio preprocessing."""

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn


class SigLIP2Classifier(nn.Module):
    global_attention = True
    backbone_lr_scale = 0.1

    def __init__(self, num_classes, in_channels, *, pretrained=False, weights_path=None):
        super().__init__()
        import timm
        from builtin_assets import SOURCE_URLS, builtin_asset_path
        from dataset import get_normalize_params

        if pretrained and weights_path:
            raise ValueError("공식 가중치와 로컬 가중치를 동시에 지정할 수 없습니다")
        self.backbone = timm.create_model(
            "vit_so400m_patch16_siglip_384.v2_webli", pretrained=False, num_classes=0,
            dynamic_img_size=True, dynamic_img_pad=True,
        )
        self.weight_provenance = {"source": "random", "backbone": "siglip2_so400m"}
        if pretrained:
            asset = "siglip2_so400m_cls"
            weights_path = builtin_asset_path(asset)
            self.weight_provenance.update(source="siglip2_webli", weights=asset, url=SOURCE_URLS[asset])
        elif weights_path:
            self.weight_provenance.update(source="local_backbone", file=Path(weights_path).name)
        if weights_path:
            from transformer_segmenters import load_local_tensors
            self.backbone.load_state_dict(load_local_tensors(weights_path), strict=True)
        mean, std = get_normalize_params(in_channels)
        # The shared image loader normalizes to ImageNet (or grayscale).
        # Undo that affine transform and apply the encoder's [-1, 1] convention.
        self.register_buffer("input_scale", 2 * torch.tensor(std).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("input_bias", 2 * torch.tensor(mean).view(1, -1, 1, 1) - 1, persistent=False)
        self.in_channels = in_channels
        self.feature_map = nn.Identity()
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    @contextmanager
    def prepare_onnx_export(self, input_shape):
        """Fold fixed-size positions using the same timm resampling as PyTorch."""
        from timm.layers import resample_abs_pos_embed

        patch_embed = self.backbone.patch_embed
        grid = tuple((int(size) + patch - 1) // patch
                     for size, patch in zip(input_shape[-2:], patch_embed.patch_size))
        base_grid = patch_embed.grid_size
        position = self.backbone.pos_embed
        if grid == base_grid:
            yield
            return
        with torch.no_grad():
            resized = resample_abs_pos_embed(position, grid, old_size=base_grid, num_prefix_tokens=0)
        self.backbone.pos_embed = nn.Parameter(resized, requires_grad=False)
        patch_embed.grid_size = grid
        try:
            yield
        finally:
            patch_embed.grid_size = base_grid
            self.backbone.pos_embed = position

    def forward(self, images):
        images = images * self.input_scale + self.input_bias
        if self.in_channels == 1:
            images = images.expand(-1, 3, -1, -1)
        tokens = self.backbone.forward_features(images)
        rows = (images.shape[-2] + 15) // 16
        cols = (images.shape[-1] + 15) // 16
        spatial = self.feature_map(tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], rows, cols))
        embeddings = self.backbone.forward_head(spatial.flatten(2).transpose(1, 2), pre_logits=True)
        return self.head(embeddings)
