"""Deterministic extra normal views; never used by evaluation or inference."""
import math

import numpy as np


def augmentation_config(settings=None):
    result = {}
    for name, maximum in (("horizontal_flip", 1.0), ("rotation", 180.0), ("color_jitter", 1.0)):
        value = float(settings.get(name, 0.0) if isinstance(settings, dict)
                      else getattr(settings, name, 0.0))
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"PatchCore 증강 {name}: 0..{maximum} 범위 필요")
        result[name] = value
    return result


def augment_normal_tensor(tensor, settings, *, seed, sample_index, view_index):
    import cv2
    import torch
    from patchcore_data import IMAGENET_MEAN, IMAGENET_STD
    from torchvision.transforms import functional as tf

    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_index, view_index]))
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    rgb = (tensor * std + mean).clamp(0, 1)
    if rng.random() < settings["horizontal_flip"]:
        rgb = tf.hflip(rgb)
    degrees = settings["rotation"]
    if degrees:
        pixels = rgb.permute(1, 2, 0).numpy()
        height, width = pixels.shape[:2]
        matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2),
                                        float(rng.uniform(-degrees, degrees)), 1.0)
        rotated = cv2.warpAffine(pixels, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        rgb = torch.from_numpy(rotated).permute(2, 0, 1)
    jitter = settings["color_jitter"]
    if jitter:
        operations = [(tf.adjust_brightness, rng.uniform(1 - jitter, 1 + jitter)),
                      (tf.adjust_contrast, rng.uniform(1 - jitter, 1 + jitter)),
                      (tf.adjust_saturation, rng.uniform(1 - jitter, 1 + jitter)),
                      (tf.adjust_hue, rng.uniform(-jitter / 2, jitter / 2))]
        for index in rng.permutation(len(operations)):
            operation, factor = operations[index]
            rgb = operation(rgb, float(factor))
    return ((rgb - mean) / std).contiguous()
