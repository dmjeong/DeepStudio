"""Geometry shared by native-resolution SAM2 image training and the paired SDK graphs."""

import torch
import torch.nn.functional as F

from checkpoint import padded_region_size


def configure_native_region(model, region):
    """Align Hiera's feature pyramids while keeping every source pixel and prompt coordinate."""
    # All four shipped SAM2.1 image trunks use an 8-token position window at
    # stride 4 and three 2x pooling stages. The FPN therefore aligns at 32px.
    height, width = padded_region_size(region)
    stride = int(model.backbone_stride)
    prompt = model.sam_prompt_encoder
    prompt.input_image_size = (height, width)
    prompt.image_embedding_size = (height // stride, width // stride)
    prompt.mask_input_size = tuple(value * 4 for value in prompt.image_embedding_size)
    model.dvs_original_size_region = tuple(region)


def native_mask_size(region):
    width, height = region
    return (height + 3) // 4, (width + 3) // 4


def pad_image(tensor, region):
    width, height = region
    rows, cols = padded_region_size(region)
    if (height, width) == (rows, cols):
        return tensor
    return F.pad(tensor, (0, cols - width, 0, rows - height))


def visible_logits(tensor, region):
    """Return logits over the true source field of view at roughly quarter resolution."""
    height, width = region[1], region[0]
    rows, cols = native_mask_size(region)
    result = tensor[..., :rows, :cols]
    if height % 4 or width % 4:
        # Part of the last decoder pixel covers padding. Sample the real pixel
        # centres before reducing onto the SDK's low-resolution mask grid.
        result = F.interpolate(result, scale_factor=4, mode="bilinear", align_corners=False)
        result = F.interpolate(result[..., :height, :width], size=(rows, cols),
                               mode="bilinear", align_corners=False)
    return result


def pad_mask_prompt(tensor, region, encoded_mask_size):
    """Lift a prompt in the public mask coordinates onto the padded decoder grid."""
    width, height = region
    rows, cols = native_mask_size(region)
    if height % 4 or width % 4:
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        tensor = F.pad(tensor, (0, cols * 4 - width, 0, rows * 4 - height))
        tensor = F.interpolate(tensor, size=(rows, cols), mode="bilinear", align_corners=False)
    if (rows, cols) == encoded_mask_size:
        return tensor
    return F.pad(tensor, (0, encoded_mask_size[1] - cols, 0, encoded_mask_size[0] - rows))


def image_embeddings(model, padded_tensor):
    """Features shared by fine-tuning, automatic inspection and exported encoder."""
    features = model.forward_image(padded_tensor)["backbone_fpn"]
    image_embed = features[2]
    if model.directly_add_no_mem_embed:
        image_embed = image_embed + model.no_mem_embed.reshape(1, -1, 1, 1)
    return image_embed, features[0], features[1]


@torch.inference_mode()
def encode_native_image(model, rgb):
    """The official predictor resizes to a square; feed native RGB into the same image encoder."""
    from torchvision.transforms.functional import normalize, to_tensor

    tensor = normalize(to_tensor(rgb), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    batch = pad_image(tensor.unsqueeze(0).to(model.device), model.dvs_original_size_region)
    return image_embeddings(model, batch)
