"""Split detection images and box labels together, retaining train/val boundaries."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from native_patches import origins, scaled_image, tiles, validate_patches
from PIL import Image
from spatial_data import read_spatial_labels


def prepare_detection_patches(root, output, contract, *, num_classes, patches_per_image=0, seed=0):
    validate_patches(contract, "detect")
    if type(patches_per_image) is not int or not 0 <= patches_per_image <= 256:
        raise ValueError("이미지당 패치는 0~256 정수 필요")
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("패치 출력은 원본 데이터셋 밖에 있어야 합니다")
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
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
            total = len(origins(height, size, stride)) * len(origins(width, size, stride))
            selected = None
            if split == "train" and patches_per_image and total > patches_per_image:
                rng = random.Random(f"{seed}/{relative.as_posix()}")
                selected = set(rng.sample(range(total), patches_per_image))
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
    (output / "patches.json").write_text(json.dumps({"contract": contract, "counts": counts,
        "source_root": str(root), "seed": seed, "train_patch_limit": patches_per_image}, indent=2), encoding="utf-8")
    return output
