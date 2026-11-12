"""Seeded, reference-textured new defect geometry at native pixel resolution.

Cracks get new paths/branches; material gets new irregular silhouettes. For
structural defects the annotated object arrangement is retained and transformed
as a unit: inventing random blobs would change the meaning of shift/double/flip.
"""

import math

import cv2
import numpy as np

DEFAULT_VARIATION = {
    "mode": "new_shape",
    "size_min": 0.65,
    "size_max": 1.45,
    "strength_min": 0.6,
    "strength_max": 1.5,
    "shape_amount": 0.65,
}


def variation_settings(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_VARIATION):
        raise ValueError("잘못된 결함 다양화 설정입니다.")
    settings = {**DEFAULT_VARIATION, **value}
    if settings["mode"] not in {"new_shape", "deform", "reference"}:
        raise ValueError("결함 생성 방식을 확인하세요.")
    for key in DEFAULT_VARIATION.keys() - {"mode"}:
        number = settings[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise ValueError("크기·강도·형상 값은 유한한 숫자여야 합니다.")
    if not 0.35 <= settings["size_min"] <= settings["size_max"] <= 3:
        raise ValueError("크기는 35~300% 범위이며 최소 ≤ 최대여야 합니다.")
    if not 0.2 <= settings["strength_min"] <= settings["strength_max"] <= 3:
        raise ValueError("강도는 20~300% 범위이며 최소 ≤ 최대여야 합니다.")
    if not 0 <= settings["shape_amount"] <= 1:
        raise ValueError("형상 변화는 0~100% 범위여야 합니다.")
    return settings


def _canvas(width, height):
    width, height = max(12, math.ceil(width)), max(12, math.ceil(height))
    if max(width, height) > 8192 or width * height > 16_777_216:
        raise ValueError("생성 결함이 너무 큽니다. 크기 범위를 낮춰 주세요.")
    return np.zeros((height, width), np.uint8)


def _crack(binary, rng, size, amount):
    ys, xs = np.where(binary)
    points = np.column_stack((xs, ys)).astype(np.float64)
    centered = points - points.mean(0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    direction = axes[0]
    along = centered @ direction
    length = max(5.0, float(np.ptp(along) + 1)) * size
    width = max(
        1,
        round(
            binary.sum()
            / max(1, float(np.ptp(along) + 1))
            * rng.uniform(0.6, 1.6)
            * size
        ),
    )
    width = min(width, max(1, round(length * 0.15)))
    angle = math.atan2(direction[1], direction[0]) + rng.uniform(-0.65, 0.65) * amount
    amplitude = max(1.0, length * (0.025 + 0.16 * amount))
    t = np.linspace(0, 1, max(16, math.ceil(length * 2)))
    bends = rng.uniform(-amplitude, amplitude, 7)
    curve = np.column_stack(
        ((t - 0.5) * length, np.interp(t, np.linspace(0, 1, 7), bends))
    )
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    path = curve @ rotation.T
    paths = [(path, width)]
    branches = int(rng.integers(0, 1 + int(4 * amount)))
    for _ in range(branches):
        index = int(rng.integers(len(path) // 4, 3 * len(path) // 4))
        branch_angle = angle + float(rng.choice([-1, 1])) * rng.uniform(0.45, 1.15)
        branch_length = length * rng.uniform(0.12, 0.32)
        vector = (
            np.array([math.cos(branch_angle), math.sin(branch_angle)]) * branch_length
        )
        bend = np.array([-vector[1], vector[0]]) * rng.uniform(-0.25, 0.25)
        branch = np.array(
            [path[index], path[index] + vector * 0.5 + bend, path[index] + vector]
        )
        paths.append((branch, max(1, width // 2)))
    all_points = np.concatenate([line for line, _ in paths])
    margin = max(8, width * 3)
    lower, upper = all_points.min(0) - margin, all_points.max(0) + margin
    mask = _canvas(*(upper - lower + 1))
    for line, thickness in paths:
        cv2.polylines(
            mask,
            [np.rint(line - lower).astype(np.int32)],
            False,
            1,
            thickness,
            cv2.LINE_8,
        )
    return mask, {
        "generator": "branched_crack_v1",
        "branches": branches,
        "width_px": width,
        "length_px": length,
    }


def _material(binary, rng, size, amount):
    _, _, width, height = cv2.boundingRect(binary)
    rx = max(2.0, width * size * rng.uniform(0.8, 1.2) / 2)
    ry = max(2.0, height * size * rng.uniform(0.8, 1.2) / 2)
    radius = max(rx, ry)
    mask = _canvas(3.4 * radius + 12, 3.4 * radius + 12)
    theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    profile = np.ones(theta.shape)
    for frequency in (2, 3, 5, 7):
        profile += (
            rng.uniform(0.07, 0.18)
            * amount
            * np.sin(frequency * theta + rng.uniform(0, 2 * np.pi))
        )
    outline = np.column_stack(
        (rx * profile * np.cos(theta), ry * profile * np.sin(theta))
    )
    angle = rng.uniform(-np.pi, np.pi)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    outline = outline @ rotation.T + [(mask.shape[1] - 1) / 2, (mask.shape[0] - 1) / 2]
    cv2.fillPoly(mask, [np.rint(outline).astype(np.int32)], 1)
    return mask, {"generator": "irregular_material_v1", "radii_px": [rx, ry]}


def varied_patch(image, mask, policy, rng, settings):
    settings = variation_settings(settings)
    classes = np.unique(mask[mask != 0])
    if len(classes) != 1 or classes[0] == 255:
        raise ValueError("생성 참조에는 한 종류의 결함 정답이 필요합니다.")
    cid = int(classes[0])
    binary = (mask == cid).astype(np.uint8)
    size = float(rng.uniform(settings["size_min"], settings["size_max"]))
    strength = float(rng.uniform(settings["strength_min"], settings["strength_max"]))
    recipe = {"variation": settings, "scale": size, "strength_sample": strength}
    amount = settings["shape_amount"]
    if settings["mode"] == "reference":
        return (
            image.copy(),
            mask.copy(),
            {**recipe, "scale": 1.0, "generator": "reference_v1"},
        )
    if settings["mode"] == "new_shape" and policy in {"crack", "material"}:
        generated, geometry = (_crack if policy == "crack" else _material)(
            binary, rng, size, amount
        )
        # Recover the reference's local surface, then render its contrast and
        # texture on a newly generated silhouette. Never retain its old defect.
        filled = cv2.inpaint(image, binary * 255, 3, cv2.INPAINT_TELEA)
        residual = image.astype(np.float32) - filled.astype(np.float32)
        samples = residual[binary.astype(bool)]
        tone = np.median(samples, axis=0)
        spread = np.std(samples, axis=0)
        h, w = generated.shape
        base = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )
        noise = cv2.GaussianBlur(
            rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.8
        )
        noise /= max(0.1, float(noise.std()))
        if image.ndim == 3:
            noise = noise[..., None]
        texture = tone + noise * np.minimum(
            spread, np.maximum(2, np.abs(tone) * 0.3)
        ) * (0.2 + amount * 0.3)
        patch = np.rint(np.clip(base + texture, 0, 255)).astype(np.uint8)
        base = np.rint(base).astype(np.uint8)
        base[generated > 0] = patch[generated > 0]
        return base, generated * cid, {**recipe, **geometry}
    # Object arrangement (shift/double/flip) is preserved as one annotated unit.
    # A coverage mask retains weak edge samples better than nearest sampling.
    # Reject a disappeared defect; the final blend also checks visibility.
    h, w = mask.shape
    angle = float(rng.uniform(-1, 1) * amount * (10 if policy == "structure" else 30))
    radians = math.radians(angle)
    if policy == "structure":
        # Transform the annotated unit, not the surrounding tray lattice.
        # Warping a whole reference rectangle creates ghost pins/neighbours
        # that are not labelled as defects, and defeats context matching.
        bx, by, bw, bh = cv2.boundingRect(binary)
        margin = max(8, min(w, h) // 8)
        matrix = cv2.getRotationMatrix2D(
            (bx + (bw - 1) / 2, by + (bh - 1) / 2), angle, size
        )
        corners = (
            np.array(
                [[bx, by, 1], [bx + bw, by, 1], [bx, by + bh, 1], [bx + bw, by + bh, 1]]
            )
            @ matrix.T
        )
        lower, upper = corners.min(0) - margin, corners.max(0) + margin
        left, top = max(0, math.ceil(-lower[0])), max(0, math.ceil(-lower[1]))
        right, bottom = max(0, math.ceil(upper[0] - w)), max(0, math.ceil(upper[1] - h))
        canvas = _canvas(w + left + right, h + top + bottom)
        matrix[:, 2] += [left, top]
        clean = cv2.inpaint(image, binary * 255, 3, cv2.INPAINT_TELEA)
        patch = cv2.copyMakeBorder(
            clean,
            top,
            canvas.shape[0] - h - top,
            left,
            canvas.shape[1] - w - left,
            cv2.BORDER_REFLECT_101,
        )
        coverage = cv2.warpAffine(
            binary.astype(np.float32),
            matrix,
            canvas.shape[::-1],
            flags=cv2.INTER_LINEAR,
        )
        transformed = (coverage >= 0.1).astype(np.uint8) * cid
        warped = cv2.warpAffine(
            image,
            matrix,
            canvas.shape[::-1],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        patch[transformed > 0] = warped[transformed > 0]
        if not transformed.any():
            raise ValueError(
                "크기 변형 후 결함이 사라졌습니다. 크기 범위를 높여 주세요."
            )
        return (
            patch,
            transformed,
            {
                **recipe,
                "angle": angle,
                "generator": "structure_affine_v1",
                "source_box": [left, top, w, h],
            },
        )
    canvas = _canvas(
        size * (abs(w * math.cos(radians)) + abs(h * math.sin(radians))) + 8,
        size * (abs(h * math.cos(radians)) + abs(w * math.sin(radians))) + 8,
    )
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, size)
    matrix[:, 2] += [(canvas.shape[1] - w) / 2, (canvas.shape[0] - h) / 2]
    patch = cv2.warpAffine(
        image,
        matrix,
        canvas.shape[::-1],
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    coverage = cv2.warpAffine(
        binary.astype(np.float32), matrix, canvas.shape[::-1], flags=cv2.INTER_LINEAR
    )
    transformed = (coverage >= 0.1).astype(np.uint8) * cid
    if not transformed.any():
        raise ValueError("크기 변형 후 결함이 사라졌습니다. 크기 범위를 높여 주세요.")
    return (
        patch,
        transformed,
        {
            **recipe,
            "angle": angle,
            "generator": "reference_affine_v1",
        },
    )
