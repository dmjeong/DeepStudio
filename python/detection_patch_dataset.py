"""Split detection images and box labels together, retaining train/val boundaries."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from native_patches import origins, scaled_image, tiles, validate_patches
from PIL import Image
from spatial_data import read_spatial_labels


SMALL_OBJECT_SIDE = 16  # Pixels at the model input scale.


def _select_small_object_tiles(boxes, size, xs, ys, baseline):
    """Select complete views of small boxes within the original tile budget."""
    coverage = defaultdict(set)
    object_priorities = []
    for x1, y1, x2, y2 in boxes:
        side = min(x2 - x1, y2 - y1)
        if side >= SMALL_OBJECT_SIDE:
            continue
        box_index = len(object_priorities)
        first_x, last_x = bisect_left(xs, x2 - size), bisect_right(xs, x1)
        first_y, last_y = bisect_left(ys, y2 - size), bisect_right(ys, y1)
        object_priorities.append((last_x - first_x) * (last_y - first_y))
        for yi in range(first_y, last_y):
            for xi in range(first_x, last_x):
                coverage[yi * len(xs) + xi].add(box_index)

    def represented(indices):
        return len({box for index in indices for box in coverage.get(index, ())})

    uncovered = {i for i, n in enumerate(object_priorities) if n > 0}
    baseline_count = represented(baseline)
    if baseline_count == len(uncovered):
        return baseline, (len(object_priorities), baseline_count, baseline_count)
    selected = set()
    while uncovered and len(selected) < len(baseline):
        # Prioritize additional boxes, then those with fewer complete views.
        best = max(coverage, key=lambda index: (
            len(coverage[index] & uncovered),
            sum(1 / object_priorities[i] for i in coverage[index] & uncovered),
            index in baseline, -index,
        ))
        selected.add(best)
        uncovered.difference_update(coverage[best])
    remaining = len(baseline) - len(selected)
    selected.update(sorted(baseline - selected)[:remaining])
    selected_count = represented(selected)
    if selected_count <= baseline_count:
        return baseline, (len(object_priorities), baseline_count, baseline_count)
    return selected, (len(object_priorities), baseline_count, selected_count)


def prepare_detection_patches(root, output, contract, *, num_classes, patches_per_image=0, seed=0,
                              small_object_coverage=False):
    validate_patches(contract, "detect")
    if type(patches_per_image) is not int or not 0 <= patches_per_image <= 256:
        raise ValueError("이미지당 패치는 0~256 정수 필요")
    if type(small_object_coverage) is not bool or (small_object_coverage and not patches_per_image):
        raise ValueError("작은 객체 우선에는 이미지당 패치 수를 1 이상으로 설정해야 합니다")
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("패치 출력은 원본 데이터셋 밖에 있어야 합니다")
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    coverage_stats = {"small_targets": 0, "randomly_covered": 0, "selected_covered": 0}
    for split in ("train", "val", "test"):
        folder = root / "images" / split
        if not folder.is_dir():
            continue
        count = 0
        for source in sorted(folder.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                continue
            relative = source.relative_to(root / "images")
            label_file = root / "labels" / relative.with_suffix(".txt")
            rows = read_spatial_labels(label_file, "detect", num_classes)
            with Image.open(source) as opened:
                pixels = scaled_image(np.asarray(opened.convert("RGB")), contract)
            height, width = pixels.shape[:2]
            size, stride = contract["size"], contract["size"] - contract["overlap"]
            xs, ys = origins(width, size, stride), origins(height, size, stride)
            total = len(xs) * len(ys)
            selected = None
            if split == "train" and patches_per_image and total > patches_per_image:
                rng = random.Random(f"{seed}/{relative.as_posix()}")
                selected = set(rng.sample(range(total), patches_per_image))
                if small_object_coverage and rows:
                    image_boxes = [(max(0, (cx-bw/2)*width), max(0, (cy-bh/2)*height),
                                    min(width, (cx+bw/2)*width), min(height, (cy+bh/2)*height))
                                   for _label, cx, cy, bw, bh in rows]
                    selected, stats = _select_small_object_tiles(image_boxes, size, xs, ys, selected)
                    for key, value in zip(coverage_stats, stats):
                        coverage_stats[key] += value
            identifier = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:20]
            for tile_index, (tile, (x, y, pw, ph)) in enumerate(tiles(pixels, contract)):
                if selected is not None and tile_index not in selected:
                    continue
                cropped = []
                for label, cx, cy, bw, bh in rows:
                    x1, y1 = max(x, (cx-bw/2)*width), max(y, (cy-bh/2)*height)
                    x2, y2 = min(x+pw, (cx+bw/2)*width), min(y+ph, (cy+bh/2)*height)
                    if x2 > x1 and y2 > y1:
                        size = contract["size"]
                        cropped.append([label, ((x1+x2)/2-x)/size, ((y1+y2)/2-y)/size,
                                        (x2-x1)/size, (y2-y1)/size])
                name = f"{identifier}_{tile_index}"
                image_path = output / "images" / split / f"{name}.png"
                target = output / "labels" / split / f"{name}.txt"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(tile).save(image_path)
                target.write_text("".join(" ".join(format(v, ".12g") for v in row) + "\n" for row in cropped), encoding="utf-8")
                count += 1
        counts[split] = count
    if not counts.get("train") or not counts.get("val"):
        raise ValueError("패치 탐지 학습에는 train과 val 원본 이미지가 각각 필요합니다")
    info = {"contract": contract, "counts": counts, "source_root": str(root),
            "seed": seed, "train_patch_limit": patches_per_image}
    if small_object_coverage:
        info["small_object_coverage"] = coverage_stats
    (output / "patches.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return output
