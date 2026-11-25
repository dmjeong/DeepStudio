"""Native-resolution, annotation-guided defect synthesis. No model/download/GPU.

Semantic labels describe defects; changed pixels additionally include blending
of normal context. They are deliberately separate outputs.
"""

import cv2
import numpy as np

POLICIES = {
    "crack": "균열 질감",
    "material": "이물 형상",
    "structure": "배치·형상 보존",
}


def default_policy(name):
    name = name.lower().replace("_", " ")
    if any(word in name for word in ("crack", "균열", "크랙")):
        return "crack"
    if name == "fm" or any(word in name for word in ("foreign", "material", "이물")):
        return "material"
    return "structure"


def validate_pair(image, mask, classes):
    if (
        image.dtype != np.uint8
        or image.ndim not in (2, 3)
        or (image.ndim == 3 and image.shape[2] != 3)
    ):
        raise ValueError("참조 합성은 8bit 흑백 또는 RGB 원본을 지원합니다.")
    if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
        raise ValueError("원본 크기의 uint8 클래스 마스크가 필요합니다.")
    if not 2 <= classes <= 255 or np.any((mask >= classes) & (mask != 255)):
        raise ValueError("마스크 클래스 번호가 프로젝트와 다릅니다.")


def extract_regions(mask, class_id, policy, *, diagnostics=None):
    """Merge nearby disconnected parts for structural defects (e.g. double)."""
    binary = (mask == class_id).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = [tuple(map(int, s[:4])) for s in stats[1:]]
    groups = [{i} for i in range(1, n)]
    if policy == "structure":
        # A semantic raster has no instance IDs. Conservatively group close
        # components; users inspect the context and can correct the output.
        changed = True
        while changed:
            changed = False
            for a in range(len(groups)):
                if changed:
                    break
                for b in range(a + 1, len(groups)):
                    close = False
                    for i in groups[a]:
                        x, y, w, h = boxes[i - 1]
                        for j in groups[b]:
                            xx, yy, ww, hh = boxes[j - 1]
                            gap = max(
                                0, x - xx - ww, xx - x - w, y - yy - hh, yy - y - h
                            )
                            if gap <= 0.35 * max(w, h, ww, hh):
                                close = True
                                break
                        if close:
                            break
                    if close:
                        groups[a] |= groups.pop(b)
                        changed = True
                        break
    if diagnostics is not None:
        diagnostics.update(
            components=n - 1,
            groups=len(groups),
            accepted=0,
            crowded=0,
            edge=0,
            context=0,
        )
    regions = []
    height, width = mask.shape
    for group in groups:
        selected = [boxes[i - 1] for i in group]
        x, y = min(b[0] for b in selected), min(b[1] for b in selected)
        right, bottom = (
            max(b[0] + b[2] for b in selected),
            max(b[1] + b[3] for b in selected),
        )
        margin = max(
            8,
            int(max(right - x, bottom - y) * (0.65 if policy == "structure" else 0.2)),
        )
        # Context can include another labelled defect even when the defect
        # itself is usable. Retry a smaller clean context rather than discarding
        # every example. Keep at least 20% for structural geometry, 4px for texture.
        minimum = (
            max(4, int(np.ceil(max(right - x, bottom - y) * 0.2)))
            if policy == "structure"
            else 4
        )
        failure = "crowded"
        for padding in sorted(
            {margin, max(minimum, margin // 2), minimum}, reverse=True
        ):
            x0, y0, x1, y1 = (
                max(0, x - padding),
                max(0, y - padding),
                min(width, right + padding),
                min(height, bottom + padding),
            )
            if policy == "structure" and (
                x0 == x or y0 == y or x1 == right or y1 == bottom
            ):
                failure = "edge"
                break
            local = mask[y0:y1, x0:x1]
            own = np.isin(labels[y0:y1, x0:x1], list(group))
            if np.any((local != 0) & ~own):
                failure = "crowded"
                continue
            if _context(own.astype(np.uint8) * class_id, policy).sum() < 12:
                failure = "context"
                continue
            break
        else:
            if diagnostics is not None:
                diagnostics[failure] += 1
            continue
        if failure == "edge":
            if diagnostics is not None:
                diagnostics["edge"] += 1
            continue
        if diagnostics is not None:
            diagnostics["accepted"] += 1
        regions.append(
            {
                "box": [x0, y0, x1 - x0, y1 - y0],
                "pixels": int(own.sum()),
                "mask": own.astype(np.uint8) * class_id,
            }
        )
    return regions


def transformed_patch(image, mask, policy, rng):
    """Only surface defects vary geometry; narrow lines retain every pixel."""
    if policy not in POLICIES:
        raise ValueError("알 수 없는 합성 방식")
    binary = (mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    ridges = (distance > 0) & (
        distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)) - 1e-6
    )
    # A wide junction must not hide a one-pixel branch elsewhere in a crack.
    thin = bool(np.any(distance[ridges] <= 1.95))
    angle, scale = 0.0, 1.0
    if policy != "structure" and not thin:
        angle, scale = float(rng.uniform(-10, 10)), float(rng.uniform(0.92, 1.08))
        h, w = mask.shape
        # A padded canvas avoids clipping rotated foreground. Context borders
        # use reflection but labels outside the original remain background.
        pad = max(4, int(0.12 * max(h, w)))
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        h, w = mask.shape
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)
        image = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)
    return image, mask, {"angle": angle, "scale": scale, "thin_preserved": thin}


def _context(mask, policy):
    h, w = mask.shape
    if policy == "structure":
        ring = np.ones((h, w), np.uint8)
        edge = max(2, min(h, w) // 8)
        ring[edge:-edge, edge:-edge] = 0
    else:
        ring = 1 - cv2.dilate((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    return ring.astype(bool) & (mask == 0)


def find_placement(
    target,
    labels,
    patch,
    mask,
    policy,
    rng,
    tolerance=0.12,
    allowed=None,
    cancel=lambda: False,
):
    """Coarse context matching, then native-pixel alignment; never shrink output."""
    h, w = mask.shape
    height, width = labels.shape
    if h > height or w > width or target.ndim != patch.ndim:
        raise ValueError("바탕과 결함의 크기 또는 채널이 맞지 않습니다.")
    ring = _context(mask, policy)
    if ring.sum() < 12:
        raise ValueError("결함 주변 정상 표면이 부족합니다.")
    factor = max(1.0, max(height, width) / 768.0, max(h, w) / 192.0)
    sw, sh = max(1, round(width / factor)), max(1, round(height / factor))
    pw, ph = max(3, round(w / factor)), max(3, round(h / factor))
    if pw > sw or ph > sh:
        raise ValueError("합성할 정상 영역이 없습니다.")
    source = (
        cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)
        / 255
    )
    background = (
        cv2.resize(target, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        / 255
    )
    search_mask = cv2.resize(
        ring.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST
    )
    if not search_mask.any():
        raise ValueError("표면 탐색에 필요한 주변 영역이 부족합니다.")
    scores = cv2.matchTemplate(background, source, cv2.TM_SQDIFF, mask=search_mask)
    scores = np.nan_to_num(scores, nan=np.inf, posinf=np.inf, neginf=np.inf)
    # Exclude occupied/forbidden windows before ranking. A flat surface has
    # thousands of identical matches; searching its first 48 would otherwise
    # miss a user ROI or empty region far from the top-left corner.
    forbidden = labels != 0
    if allowed is not None:
        forbidden = forbidden | ~allowed
    integral = cv2.integral(forbidden.astype(np.uint8), sdepth=cv2.CV_32S)
    ys = np.minimum(
        np.rint(np.arange(scores.shape[0]) * factor).astype(int), height - h
    )
    xs = np.minimum(np.rint(np.arange(scores.shape[1]) * factor).astype(int), width - w)
    occupied = (
        integral[ys[:, None] + h, xs[None, :] + w]
        - integral[ys[:, None], xs[None, :] + w]
        - integral[ys[:, None] + h, xs[None, :]]
        + integral[ys[:, None], xs[None, :]]
    )
    scores[occupied > 0] = np.inf
    # Break ties across repeated chip cells rather than filling only the first
    # few top-left matches. Seeded jitter is far below the context gate.
    scores += rng.random(scores.shape, dtype=np.float32) * 1e-5
    ring_y, ring_x = np.nonzero(ring)
    step = max(1, len(ring_y) // 1024)
    sample_y, sample_x = ring_y[::step], ring_x[::step]
    reference_samples = patch[sample_y, sample_x].astype(np.float32)
    candidates = []
    for _ in range(48):
        if cancel():
            raise InterruptedError("생성 중단")
        yy, xx = np.unravel_index(np.argmin(scores), scores.shape)
        if not np.isfinite(scores[yy, xx]):
            break
        # Native alignment near coarse match. Labels are checked at full size.
        cx, cy = round(xx * factor), round(yy * factor)
        best = None
        radius = min(8, max(1, int(np.ceil(factor))))
        for y in range(max(0, cy - radius), min(height - h, cy + radius) + 1):
            for x in range(max(0, cx - radius), min(width - w, cx + radius) + 1):
                occupied = (
                    integral[y + h, x + w]
                    - integral[y, x + w]
                    - integral[y + h, x]
                    + integral[y, x]
                )
                if occupied:
                    continue
                delta = (
                    target[y + sample_y, x + sample_x].astype(np.float32)
                    - reference_samples
                )
                offset = np.median(delta, axis=0)
                error = float(np.sqrt(np.mean((delta - offset) ** 2)) / 255.0)
                if np.max(np.abs(offset)) > 60 or error > tolerance:
                    continue
                if best is None or error < best[0]:
                    best = (error, x, y, offset)
        if best:
            _, bx, by, _ = best
            delta = target[by : by + h, bx : bx + w][ring].astype(np.float32) - patch[
                ring
            ].astype(np.float32)
            offset = np.median(delta, axis=0)
            error = float(np.sqrt(np.mean((delta - offset) ** 2)) / 255.0)
            if error <= tolerance and np.max(np.abs(offset)) <= 60:
                candidates.append((error, bx, by, offset))
                if len(candidates) >= 8:
                    break
        ry, rx = max(2, ph // 3), max(2, pw // 3)
        scores[max(0, yy - ry) : yy + ry + 1, max(0, xx - rx) : xx + rx + 1] = np.inf
    if not candidates:
        raise ValueError(
            "기존 불량/제외 영역을 피하면서 표면이 맞는 위치를 찾지 못했습니다."
        )
    candidates.sort(key=lambda row: row[0])
    shortlist = [v for v in candidates if v[0] <= candidates[0][0] + 0.025]
    return shortlist[int(rng.integers(len(shortlist)))]


def synthesize(
    target,
    labels,
    patch,
    mask,
    policy,
    rng,
    *,
    tolerance=0.12,
    allowed=None,
    cancel=lambda: False,
    variation=None,
):
    if variation is None:
        patch, mask, transform = transformed_patch(patch, mask, policy, rng)
    else:
        from core.synthesis_variation import varied_patch

        patch, mask, transform = varied_patch(patch, mask, policy, rng, variation)
    error, x, y, offset = find_placement(
        target, labels, patch, mask, policy, rng, tolerance, allowed, cancel
    )
    h, w = mask.shape
    original = target[y : y + h, x : x + w]
    foreground = mask > 0
    if policy == "crack":
        filled = cv2.inpaint(
            patch, foreground.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA
        )
        strength = transform.get("strength_sample")
        if strength is None:
            strength = float(rng.uniform(0.85, 1.15))
        proposed = (
            original.astype(np.float32)
            + (patch.astype(np.float32) - filled.astype(np.float32)) * strength
        )
    else:
        strength = transform.get("strength_sample", 1.0)
        proposed = (
            original.astype(np.float32)
            + (patch.astype(np.float32) + offset - original.astype(np.float32))
            * strength
        )
    if policy == "structure":
        edge_distance = np.minimum(
            np.minimum(np.arange(h)[:, None], np.arange(h - 1, -1, -1)[:, None]),
            np.minimum(np.arange(w)[None, :], np.arange(w - 1, -1, -1)[None, :]),
        )
        alpha = np.minimum(1.0, edge_distance / max(2, min(h, w) // 10)).astype(
            np.float32
        )
        alpha[foreground] = 1.0
    else:
        alpha = np.clip(
            cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 3),
            0.75,
            1.0,
        )
        alpha[~foreground] = 0.0
    if target.ndim == 3:
        alpha = alpha[..., None]
    composite = np.rint(
        np.clip(original * (1 - alpha) + proposed * alpha, 0, 255)
    ).astype(np.uint8)
    difference = np.abs(composite.astype(np.int16) - original.astype(np.int16))
    visible = (difference.max(axis=2) if target.ndim == 3 else difference) >= 2
    if np.count_nonzero(visible & foreground) < max(1, int(0.1 * foreground.sum())):
        raise ValueError("실제 결함 영역의 색상 변화가 너무 작아 후보를 제외했습니다.")
    output, output_mask = target.copy(), labels.copy()
    output[y : y + h, x : x + w] = composite
    output_mask[y : y + h, x : x + w][foreground] = mask[foreground]
    changed = np.any(output != target, axis=2) if target.ndim == 3 else output != target
    return (
        output,
        output_mask,
        changed.astype(np.uint8) * 255,
        {
            "box": [x, y, w, h],
            "context_rmse": error,
            "strength": strength,
            "brightness_offset": np.asarray(offset).tolist(),
            **transform,
        },
    )
