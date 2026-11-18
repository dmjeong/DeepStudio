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

    def _outputs(self, image):
        if self.gray:
            image = (image * .226 + .449 - self.rgb_mean) / self.rgb_std
        h, w = image.shape[-2:]
        # Align the four feature levels in the pixel decoder, including odd-sized ROIs.
        image = F.pad(image, (0, (-w) % 32, 0, (-h) % 32))
        out = self.net(pixel_values=image, return_dict=True)
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
        logits, out = self._outputs(image)
        loss = criterion(logits, target)
        if self.model_id == "mask2former_swin_small":
            from eomt import query_loss
            loss = loss + query_loss(out.masks_queries_logits, out.class_queries_logits,
                                     target, pad_stride=32, max_points=12544)
        return logits, loss
