"""국소 표면 합성: 다중 스케일 마스크, 재현 가능한 시드, 정확한 변경 마스크.

학습된 생성 모델이 아니다. 실제 불량의 질감을 참조 이미지로 제공할 수 있다.
원본의 8/16비트 정밀도와 ROI 밖 픽셀을 보존한다.
"""

from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
from PIL import Image


class DefectType(Enum):
    SCRATCH = "scratch"
    STAIN = "stain"
    CUTOUT = "cutout"
    NOISE = "noise"
    TEXTURE = "texture"
    ELASTIC = "elastic"
    COLOR_SHIFT = "color"


DEFECT_INFO = {
    DefectType.SCRATCH: {"name": "스크래치", "desc": "곡선과 가변 굵기, 긁힘의 명암 경계", "group": "기본"},
    DefectType.STAIN: {"name": "얼룩", "desc": "다중 스케일의 불규칙 오염과 부식", "group": "기본"},
    DefectType.CUTOUT: {"name": "패치 결손", "desc": "불규칙 파임과 가장자리 명암", "group": "기본"},
    DefectType.NOISE: {"name": "노이즈", "desc": "국소 입자와 거친 표면", "group": "기본"},
    DefectType.TEXTURE: {"name": "텍스처 교체", "desc": "참조 질감 또는 자체 패치의 불규칙 합성", "group": "고급"},
    DefectType.ELASTIC: {"name": "탄성 변형", "desc": "국소적인 휨과 표면 변형", "group": "고급"},
    DefectType.COLOR_SHIFT: {"name": "색상 이상", "desc": "원래 질감을 유지하는 국소 변색", "group": "고급"},
}


@dataclass
class DefectParams:
    intensity: float = 0.5
    size_ratio: float = 0.15
    count: int = 1
    types: list = field(default_factory=lambda: [DefectType.SCRATCH])
    seed: int | None = None
    feather: float = 0.25
    roughness: float = 0.65
    mix_types: bool = False


@dataclass
class DefectSample:
    image: np.ndarray
    mask: np.ndarray
    types: list
    seed: int
    recipe: dict


def _resize(field, shape):
    return np.asarray(Image.fromarray(field.astype(np.float32)).resize(
        (shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)


def _fractal(shape, rng):
    """작은 격자를 여러 해상도로 합성하여 반복 타원 형태를 줄인다."""
    noise = np.zeros(shape, dtype=np.float32)
    for size, weight in ((3, 1.0), (7, .5), (15, .25), (31, .125)):
        noise += weight * _resize(rng.uniform(-1, 1, (size, size)), shape)
    return noise / 1.875


def _random_region(h, w, size_ratio, rng=None):
    rng = rng or np.random.default_rng()
    if min(h, w) < 1 or not 0 < size_ratio <= 1:
        raise ValueError("이미지 크기와 영역 비율 오류")
    area = h * w * size_ratio
    aspect = float(rng.uniform(.5, 2))
    rw = min(w, max(1, int(np.sqrt(area * aspect))))
    rh = min(h, max(1, int(np.sqrt(area / aspect))))
    return int(rng.integers(w - rw + 1)), int(rng.integers(h - rh + 1)), rw, rh


def _smooth_mask(h, w, x, y, rw, rh):
    yy, xx = np.mgrid[:h, :w]
    radius = ((xx - x - (rw - 1) / 2) / max(rw / 2, .5)) ** 2
    radius += ((yy - y - (rh - 1) / 2) / max(rh / 2, .5)) ** 2
    return np.clip((1 - radius) / .3, 0, 1).astype(np.float32)


def _sample_bilinear(patch, dx, dy):
    h, w = patch.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    xx, yy = np.clip(xx + dx, 0, w - 1), np.clip(yy + dy, 0, h - 1)
    x0, y0 = xx.astype(int), yy.astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    ax, ay = (xx - x0)[..., None], (yy - y0)[..., None]
    return ((1 - ay) * ((1 - ax) * patch[y0, x0] + ax * patch[y0, x1])
            + ay * ((1 - ax) * patch[y1, x0] + ax * patch[y1, x1]))


def _validate_image(image):
    image = np.asarray(image)
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError("불량 합성 입력은 uint8 또는 uint16이어야 합니다")
    if image.ndim not in (2, 3) or image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError("불량 합성 입력은 회색조 또는 RGB여야 합니다")
    if min(image.shape[:2]) < 1:
        raise ValueError("빈 이미지는 합성할 수 없습니다")
    return image


def generate_sample(image, params, defect_type=None, *, roi_mask=None, texture=None):
    """합성 이미지, 실제 변경 픽셀 마스크(0/255), 재현 정보를 반환한다.

    ROI 마스크는 원본과 같은 HxW이며 양수인 픽셀만 합성한다.
    size_ratio는 패턴의 후보 사각형 면적이며 실제 결함 면적과 다르다.
    """
    image = _validate_image(image)
    if not 0 < params.intensity <= 1 or not 0 < params.size_ratio <= 1:
        raise ValueError("강도와 크기 비율은 0 초과 1 이하여야 합니다")
    if not isinstance(params.count, int) or not 1 <= params.count <= 100:
        raise ValueError("패턴 수는 1~100이어야 합니다")
    if not 0 <= params.feather <= 1 or not 0 <= params.roughness <= 1:
        raise ValueError("경계와 불규칙도는 0~1이어야 합니다")
    choices = [DefectType(defect_type)] if defect_type is not None else [DefectType(t) for t in params.types]
    if not choices:
        raise ValueError("불량 유형을 선택하세요")
    seed = int(params.seed) if params.seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    roi = np.ones((h, w), dtype=bool) if roi_mask is None else np.asarray(roi_mask) > 0
    if roi.shape != (h, w) or not roi.any():
        raise ValueError("ROI는 원본과 같은 크기의 비어 있지 않은 2차원 마스크여야 합니다")
    coordinates = np.argwhere(roi) if roi_mask is not None else None
    scale = np.iinfo(image.dtype).max
    original = image.astype(np.float32) / scale
    original = original[..., None] if original.ndim == 2 else original
    result = original.copy()
    donor = None
    if texture is not None:
        texture = _validate_image(texture)
        donor = texture.astype(np.float32) / np.iinfo(texture.dtype).max
        donor = donor[..., None] if donor.ndim == 2 else donor
        if donor.shape[2] != original.shape[2]:
            donor = np.repeat(donor, 3, axis=2) if donor.shape[2] == 1 else donor.mean(axis=2, keepdims=True)
    chosen = choices[int(rng.integers(len(choices)))]
    applied, regions = [], []
    for index in range(params.count):
        kind = choices[int(rng.integers(len(choices)))] if params.mix_types and index else chosen
        x, y, rw, rh = _random_region(h, w, params.size_ratio, rng)
        if coordinates is not None:
            cy, cx = coordinates[int(rng.integers(len(coordinates)))]
            x, y = int(np.clip(cx - rw // 2, 0, w - rw)), int(np.clip(cy - rh // 2, 0, h - rh))
        patch = result[y:y + rh, x:x + rw]
        noise = _fractal((rh, rw), rng)
        yy, xx = np.mgrid[:rh, :rw].astype(np.float32)
        xx = (xx - (rw - 1) / 2) / max(rw / 2, .5)
        yy = (yy - (rh - 1) / 2) / max(rh / 2, .5)
        field = 1 - (xx * xx + yy * yy) + noise * params.roughness * 1.8
        alpha = np.clip(field / (.03 + params.feather * .6), 0, 1)
        strength = params.intensity
        polarity = 1 if float(patch.mean()) < .5 else -1
        if kind == DefectType.SCRATCH:
            angle = rng.uniform(0, np.pi)
            along = xx * np.cos(angle) + yy * np.sin(angle)
            across = -xx * np.sin(angle) + yy * np.cos(angle)
            curve = across - rng.uniform(-.25, .25) * np.sin(along * 3)
            width = max(1.5 / max(min(rh, rw), 1), .025 + strength * .035)
            distance = abs(curve) / (width * (1 + .35 * np.sin(along * 5)))
            alpha = np.clip((1.7 - distance) * 2, 0, 1) * np.clip((1 - abs(along)) * 8, 0, 1)
            signed = np.where(curve < -width * .6, -polarity * .25, polarity)
            altered = patch + signed[..., None] * strength * .55
        elif kind == DefectType.STAIN:
            pigment = np.array([.35, .12, -.12], dtype=np.float32) if patch.shape[2] == 3 else np.array([0])
            altered = patch + strength * (polarity * (.15 + .2 * noise[..., None]) + pigment * .35)
        elif kind == DefectType.CUTOUT:
            rim = np.clip(1 - abs(field - .08) / .13, 0, 1)
            altered = patch * (1 - .7 * strength) + (noise[..., None] * .08 + rim[..., None] * .45) * strength
            if patch.mean() < .05:
                altered += .15 * strength
        elif kind == DefectType.NOISE:
            grain = rng.normal(0, .18, (*patch.shape[:2], 1))
            grain = grain + rng.normal(0, .015, patch.shape)
            altered = patch + strength * (grain + noise[..., None] * .3)
        elif kind == DefectType.TEXTURE:
            source = original if donor is None else donor
            sx, sy, sw, sh = _random_region(*source.shape[:2], min(params.size_ratio * 2, 1), rng)
            fragment = source[sy:sy + sh, sx:sx + sw]
            if rng.random() < .5:
                fragment = fragment[::-1]
            fragment = np.stack([_resize(fragment[..., c], (rh, rw)) for c in range(fragment.shape[2])], axis=2)
            # 주변 조명을 일부 맞추되 참조의 질감과 색상 차이는 남긴다.
            adapted = fragment + (patch.mean(axis=(0, 1)) - fragment.mean(axis=(0, 1))) * .6
            altered = patch * (1 - strength) + (adapted + noise[..., None] * .2) * strength
        elif kind == DefectType.ELASTIC:
            displacement = strength * max(1, min(rh, rw) * .12)
            warped = _sample_bilinear(patch, noise * displacement, _fractal((rh, rw), rng) * displacement)
            # 평탄한 표면에서도 휨의 명암을 표현한다.
            altered = warped + noise[..., None] * strength * .12
        else:
            tint = rng.uniform(-.35, .35, (1, 1, patch.shape[2]))
            tint += polarity * .12
            altered = patch * (1 + tint * strength) + tint * strength * .2
        alpha *= np.clip((1 - np.maximum(abs(xx), abs(yy))) * 12, 0, 1)
        alpha *= roi[y:y + rh, x:x + rw]
        result[y:y + rh, x:x + rw] = patch * (1 - alpha[..., None]) + np.clip(altered, 0, 1) * alpha[..., None]
        applied.append(kind)
        regions.append([x, y, rw, rh])
    output = np.rint(np.clip(result, 0, 1) * scale).astype(image.dtype)
    if image.ndim == 2:
        output = output[..., 0]
    changed = output != image
    changed = changed.any(axis=2) if changed.ndim == 3 else changed
    mask = changed.astype(np.uint8) * 255
    recipe = {"generator": "surface_synthesis_v2", "seed": seed,
              "types": [kind.value for kind in applied], "regions_xywh": regions,
              "intensity": params.intensity, "size_ratio": params.size_ratio, "count": params.count,
              "feather": params.feather, "roughness": params.roughness, "mix_types": params.mix_types,
              "dtype": str(image.dtype), "changed_fraction": float(changed.mean()),
              "has_roi": roi_mask is not None, "has_reference_texture": texture is not None}
    return DefectSample(output, mask, applied, seed, recipe)


def generate_defect(image, params, defect_type=None):
    sample = generate_sample(image, params, defect_type)
    return sample.image, sample.types[0]


def generate_batch(image, params, num_images=1):
    seeds = np.random.SeedSequence(params.seed).spawn(num_images)
    return [generate_defect(image, replace(params, seed=int(seed.generate_state(1)[0]))) for seed in seeds]


def generate_scratch(image, params):
    return generate_defect(image, params, DefectType.SCRATCH)[0]


def generate_stain(image, params):
    return generate_defect(image, params, DefectType.STAIN)[0]


def generate_cutout(image, params):
    return generate_defect(image, params, DefectType.CUTOUT)[0]


def generate_noise(image, params):
    return generate_defect(image, params, DefectType.NOISE)[0]


def generate_texture_swap(image, params):
    return generate_defect(image, params, DefectType.TEXTURE)[0]


def generate_elastic_deform(image, params):
    return generate_defect(image, params, DefectType.ELASTIC)[0]


def generate_color_shift(image, params):
    return generate_defect(image, params, DefectType.COLOR_SHIFT)[0]
