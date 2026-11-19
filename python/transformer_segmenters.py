"""Offline semantic training adapters for the Apache-2.0 Transformers implementations."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


MODEL_IDS = ("mask2former_swin_small", "segformer_b2")
# Keys in the original Hugging Face state dict, before our "net." prefix.
TASK_HEADS = {
    "mask2former_swin_small": ("class_predictor.", "criterion.empty_weight"),
    "segformer_b2": ("decode_head.classifier.",),
}


def load_local_tensors(path):
    if Path(path).suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _mask_points(target, limit):
    """Pixel centres in the stride-32 padded frame; 255 and padding never enter the loss."""
    height, width = target.shape
    flat = target.flatten()
    valid = (flat != 255).nonzero().flatten()
    labels = flat[valid].unique()
    if not labels.numel():
        return None
    if limit < labels.numel():
        raise ValueError("Mask2Former의 점 샘플 수가 이미지에 있는 클래스 수보다 작습니다.")
    count = min(limit, valid.numel())
    foreground = labels[labels != 0]
    # Preserve a training signal for a one-pixel defect even on small crops where every pixel fits.
    # The remaining three quarters keep the region's normal background and negative examples.
    reserved = []
    quota = max(1, count // (4 * max(1, foreground.numel())))
    for label in foreground:
        locations = (flat == label).nonzero().flatten()
        indices = torch.randint(locations.numel(), (quota,), device=flat.device)
        reserved.append(locations[indices])
    reserved = torch.cat(reserved) if reserved else valid[:0]
    uniform = valid[torch.randperm(valid.numel(), device=flat.device)[:count - reserved.numel()]]
    indices = torch.cat((reserved, uniform))
    # A mostly foreground crop can have a single background pixel. Keep its mask matchable.
    if labels[0] == 0 and not (flat[indices] == 0).any():
        indices[-1] = (flat == 0).nonzero()[0, 0]
    valid = indices
    padded_h, padded_w = ((size + 31) // 32 * 32 for size in (height, width))
    x = (valid % width + .5) * (2. / padded_w) - 1
    y = (valid // width + .5) * (2. / padded_h) - 1
    grid = torch.stack((x, y), dim=-1)[None, :, None]
    return labels, (flat[valid] == labels[:, None]).float(), grid


def _decoder_query_loss(out, target, native):
    from scipy.optimize import linear_sum_assignment
    from transformers.models.mask2former.modeling_mask2former import (
        dice_loss, pair_wise_dice_loss, pair_wise_sigmoid_cross_entropy_loss, sigmoid_cross_entropy_loss,
    )

    stages = [(aux["masks_queries_logits"], aux["class_queries_logits"])
              for aux in (out.auxiliary_logits or [])]
    stages.append((out.masks_queries_logits, out.class_queries_logits))
    points = [_mask_points(image, native.num_points) for image in target]
    total = out.class_queries_logits.float().sum() * 0.
    weights = native.weight_dict
    with torch.autocast(device_type=target.device.type, enabled=False):
        for masks, classes in stages:
            for i, sample in enumerate(points):
                if sample is None:
                    continue
                labels, truth, grid = sample
                cls = classes[i].float()
                # Match each decoder stage separately. Retain gradient buffers only for assigned masks.
                with torch.no_grad():
                    sampled = F.grid_sample(masks[i:i + 1].float(), grid, padding_mode="border",
                                            align_corners=False)[0, :, :, 0]
                    cost = (weights["loss_mask"] * pair_wise_sigmoid_cross_entropy_loss(sampled, truth)
                            + weights["loss_dice"] * pair_wise_dice_loss(sampled, truth)
                            - weights["loss_cross_entropy"] * cls.softmax(-1)[:, labels])
                    row, col = linear_sum_assignment(cost.cpu().numpy())
                    row = torch.as_tensor(row, device=target.device)
                    col = torch.as_tensor(col, device=target.device)
                expected = torch.full((cls.shape[0],), cls.shape[1] - 1, device=target.device, dtype=torch.long)
                expected[row] = labels[col]
                matched = F.grid_sample(masks[i, row][None].float(), grid, padding_mode="border",
                                        align_corners=False)[0, :, :, 0]
                total = (total + weights["loss_cross_entropy"] * F.cross_entropy(cls, expected, native.empty_weight)
                         + weights["loss_mask"] * sigmoid_cross_entropy_loss(matched, truth[col], len(row))
                         + weights["loss_dice"] * dice_loss(matched, truth[col], len(row)))
    return total / (len(stages) * max(1, sum(sample is not None for sample in points)))


class TransformerSegmenter(nn.Module):
    global_attention = True

    def __init__(self, model_id, classes, channels):
        super().__init__()
        self.model_id = model_id
        if model_id == "segformer_b2":
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
            config = SegformerConfig(num_labels=classes, depths=[3, 4, 6, 3],
                                     hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768)
            self.net = SegformerForSemanticSegmentation(config)
        elif model_id == "mask2former_swin_small":
            from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig
            backbone = SwinConfig(depths=[2, 2, 18, 2], drop_path_rate=0.3,
                                  out_features=["stage1", "stage2", "stage3", "stage4"])
            config = Mask2FormerConfig(backbone_config=backbone, num_labels=classes)
            self.net = Mask2FormerForUniversalSegmentation(config)
            # Encoder-only pretraining leaves the query decoder untrained. Start its
            # content and position slots at the scale used by the original decoder.
            # A full segmentation checkpoint restores both tensors after construction.
            decoder = self.net.model.transformer_module
            decoder.queries_features.reset_parameters()
            decoder.queries_embedder.reset_parameters()
        else:
            raise ValueError(f"Unknown transformer segmenter: {model_id}")
        self.gray = channels == 1
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)
        self.weight_provenance = {"source": "random", "architecture": model_id}

    def initialize_from_weights(self, state, path):
        """Initialize from a local HF semantic model, or the exact HF encoder alone."""
        if not isinstance(state, dict):
            raise ValueError("Hugging Face 모델 또는 백본의 state_dict가 필요합니다.")
        state = state.get("state_dict", state)
        if not isinstance(state, dict):
            raise ValueError("가중치의 state_dict 형식이 잘못되었습니다.")
        encoder = (self.net.segformer if self.model_id == "segformer_b2"
                   else self.net.model.pixel_level_module.encoder)
        if self.model_id == "mask2former_swin_small" and "swin.layernorm.weight" in state:
            # The classifier has one final norm. SwinBackbone has one per stage.
            source_weights = {k.removeprefix("swin."): v for k, v in state.items() if k.startswith("swin.")}
            for suffix in ("weight", "bias"):
                source_weights["hidden_states_norms.stage4." + suffix] = source_weights.pop("layernorm." + suffix)
            missing, extra = encoder.load_state_dict(source_weights, strict=False)
            expected = {f"hidden_states_norms.stage{i}.{suffix}"
                        for i in (1, 2, 3) for suffix in ("weight", "bias")}
            if extra or set(missing) != expected:
                raise ValueError("Swin-Small 백본의 레이어 또는 차원이 다릅니다.")
            source = "local_swin_backbone"
        elif self.model_id == "segformer_b2" and "classifier.weight" in state:
            # MiT-B2 image classification checkpoint, using the HF key layout.
            backbone = {key.removeprefix("segformer."): value for key, value in state.items()
                        if key.startswith("segformer.")}
            encoder.load_state_dict(backbone, strict=True)
            source = "local_mit_backbone"
        elif set(state) == set(encoder.state_dict()):
            encoder.load_state_dict(state, strict=True)
            source = "local_hf_encoder"
        else:
            heads = TASK_HEADS[self.model_id]
            matched = {k: v for k, v in state.items() if not k.startswith(heads)}
            expected = {k for k in self.net.state_dict() if not k.startswith(heads)}
            if set(matched) != expected:
                raise ValueError(f"{self.model_id}의 Hugging Face 분할 모델 또는 백본 가중치가 필요합니다.")
            self.net.load_state_dict(matched, strict=False)  # class order in external files is unknown
            source = "local_hf_segmentation"
        self.weight_provenance.update(source=source, file=Path(path).name)

    def _outputs(self, image, *, supervise_decoder=False):
        if self.gray:
            image = (image * .226 + .449 - self.rgb_mean) / self.rgb_std
        h, w = image.shape[-2:]
        # Align the four feature levels in the pixel decoder, including odd-sized ROIs.
        image = F.pad(image, (0, (-w) % 32, 0, (-h) % 32))
        options = {"output_auxiliary_logits": True} if supervise_decoder else {}
        out = self.net(pixel_values=image, return_dict=True, **options)
        if self.model_id == "segformer_b2":
            logits = out.logits
        else:
            with torch.autocast(device_type=image.device.type, enabled=False):
                classes = out.class_queries_logits.float().softmax(-1)[..., :-1]
                masks = out.masks_queries_logits.float().sigmoid()
                scores = torch.einsum("bqc,bqhw->bchw", classes, masks)
            logits = scores
        logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        if self.model_id == "mask2former_swin_small":
            logits = logits.clamp_min(1e-8).log()
        return logits[..., :h, :w], out

    def forward(self, image):
        return self._outputs(image)[0]

    def compute_training_loss(self, image, target, criterion):
        supervise_decoder = self.model_id == "mask2former_swin_small"
        logits, out = self._outputs(image, supervise_decoder=supervise_decoder)
        loss = criterion(logits, target)
        if supervise_decoder:
            loss = loss + _decoder_query_loss(out, target, self.net.criterion)
        return logits, loss
