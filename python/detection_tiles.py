"""검출 패치 학습: 큰 영상을 줄이지 않고 타일로 잘라 학습용 데이터셋을 만든다.

    전체 축소 학습   3580×2100 ─▶ 640으로 축소 ─▶ 10px 결함이 약 2px       (작은 결함이 사라진다)
    패치 학습        3580×2100 ─▶ (배율) ─▶ 640 타일로 자름 ─▶ 결함 크기 그대로

        images/train/a.png + labels/train/a.txt
              │  배율 s로 줄임(INTER_AREA) → 겹치는 타일 격자
              ▼
        타일마다 상자를 타일 좌표로 옮기고 타일 밖은 잘라 낸다
              │  물체가 있는 타일 전부 + 배경 타일 일부(영상당 물체 타일 수만큼, 최소 1)
              ▼
        <출력>/images/train/a_y0000_x0512.png + labels/train/a_y0000_x0512.txt

추론은 같은 배율 · 같은 타일 크기로 한다 (데스크톱: 겹치는 타일 + 병합, C++: 판정 영역 한 번에).
원본 데이터는 바꾸지 않는다.
"""

from __future__ import annotations

import random
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SCALE_INTERPOLATION = "INTER_AREA"   # 줄일 때 모든 픽셀을 평균한다 (1px 결함이 점선으로 끊기지 않게)
MIN_VISIBLE = 0.3                    # 상자 넓이의 이 비율 이상이 타일 안에 있어야 그 타일의 물체로 센다
MIN_SIDE = 2.0                       # 잘린 상자의 최소 변 길이 (px)


def tile_origins(length, tile, overlap):
    """한 축을 덮는 타일 시작 위치. 마지막 타일은 끝에 맞춘다 (영상이 타일보다 짧으면 [0])."""
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    origins = list(range(0, length - tile, step))
    origins.append(length - tile)
    return origins


def tile_rows(rows, image_size, origin, tile_size):
    """영상 기준 YOLO 행(cls cx cy w h, 0~1) → 타일 기준 행. 타일 밖 부분은 잘라 낸다.

    넓이의 MIN_VISIBLE 미만만 걸친 상자는 버린다 (조각난 물체를 정답으로 가르치지 않는다).
    """
    width, height = image_size
    left, top = origin
    tile_w, tile_h = tile_size
    kept = []
    for row in rows:
        cls, cx, cy, w, h = (float(value) for value in row[:5])
        x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
        x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
        cx1, cy1 = max(x1, left), max(y1, top)
        cx2, cy2 = min(x2, left + tile_w), min(y2, top + tile_h)
        if cx2 - cx1 < MIN_SIDE or cy2 - cy1 < MIN_SIDE:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area <= 0 or (cx2 - cx1) * (cy2 - cy1) / area < MIN_VISIBLE:
            continue
        kept.append([int(cls), ((cx1 + cx2) / 2 - left) / tile_w, ((cy1 + cy2) / 2 - top) / tile_h,
                     (cx2 - cx1) / tile_w, (cy2 - cy1) / tile_h])
    return kept


def _read_rows(path):
    """YOLO 검출 라벨. 파일이 없으면 배경 영상(빈 목록)."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if len(values) >= 5:
            rows.append([float(value) for value in values[:5]])
    return rows


def _load(path, scale):
    """영상을 읽어 배율로 줄인다. (PIL로 읽어 한글 경로도 된다)"""
    import numpy as np
    from PIL import Image
    with Image.open(path) as source:
        pixels = np.asarray(source if source.mode in ("L", "RGB") else source.convert("RGB"))
    if scale != 1:
        import cv2
        height, width = pixels.shape[:2]
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        pixels = cv2.resize(pixels, size, interpolation=getattr(cv2, SCALE_INTERPOLATION))
    return pixels


def prepare_tiled_detection_dataset(root, output, *, tile, scale=1.0, overlap=None, max_tiles=0, seed=0, log=None):
    """images/{train,val} + labels/{train,val} → 같은 구조의 타일 데이터셋. 요약 dict를 돌려준다.

        tile       타일 한 변(px) = 모델 입력 크기
        scale      자르기 전에 영상을 줄이는 배율 (0 < scale <= 1)
        overlap    타일 겹침(px). 기본 = 타일의 20% (경계에 걸친 물체가 어느 한 타일에는 온전히 들어간다)
        max_tiles  영상당 타일 수 상한 (0 = 제한 없음). 물체가 있는 타일을 먼저 고른다
    """
    from PIL import Image
    root, output = Path(root).resolve(), Path(output)
    tile = int(tile)
    if tile < 32 or not 0 < float(scale) <= 1 or int(max_tiles) < 0:
        raise ValueError("타일 크기는 32 이상, 배율은 0보다 크고 1 이하, 타일 수 상한은 0 이상이어야 합니다")
    overlap = tile // 5 if overlap is None else int(overlap)
    if not 0 <= overlap < tile:
        raise ValueError("타일 겹침은 0 이상이고 타일 크기보다 작아야 합니다")
    if not (root / "images" / "train").is_dir():
        raise ValueError(f"검출 학습 이미지 폴더가 없습니다: {root / 'images' / 'train'}")
    summary = {"tile": tile, "scale": float(scale), "overlap": overlap, "interpolation": SCALE_INTERPOLATION,
               "splits": {}, "region": None}
    sizes = {}
    for split in ("train", "val"):
        image_dir = root / "images" / split
        if not image_dir.is_dir():
            continue
        files = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        target_images, target_labels = output / "images" / split, output / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        counts = {"images": len(files), "tiles": 0, "object_tiles": 0, "boxes": 0}
        for index, path in enumerate(files):
            relative = path.relative_to(image_dir)
            rows = _read_rows((root / "labels" / split / relative).with_suffix(".txt"))
            pixels = _load(path, float(scale))
            height, width = pixels.shape[:2]
            # 판정 영역 = 가장 많은 원본 크기 (내보내기의 기본 입력 크기가 된다)
            original = (int(round(width / scale)), int(round(height / scale))) if scale != 1 else (width, height)
            sizes[original] = sizes.get(original, 0) + 1
            tile_w, tile_h = min(tile, width), min(tile, height)
            objects, background = [], []
            for top in tile_origins(height, tile, overlap):
                for left in tile_origins(width, tile, overlap):
                    kept = tile_rows(rows, (width, height), (left, top), (tile_w, tile_h))
                    (objects if kept else background).append((top, left, kept))
            # 배경 타일은 영상마다 물체 타일 수만큼(최소 1): 배경만 잔뜩 학습해 검출이 죽지 않게
            chooser = random.Random(f"{seed}:{split}:{relative.as_posix()}")
            chooser.shuffle(background)
            chosen = objects + background[:max(1, len(objects))]
            if max_tiles and len(chosen) > max_tiles:
                chooser.shuffle(objects)
                chosen = (objects + background)[:max_tiles]
            stem = "_".join(relative.with_suffix("").parts)
            for top, left, kept in chosen:
                name = f"{stem}_y{top:05d}_x{left:05d}"
                Image.fromarray(pixels[top:top + tile_h, left:left + tile_w]).save(
                    target_images / f"{name}.png", compress_level=1)
                (target_labels / f"{name}.txt").write_text(
                    "".join("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*row) for row in kept), encoding="utf-8")
                counts["tiles"] += 1
                counts["object_tiles"] += bool(kept)
                counts["boxes"] += len(kept)
            if log and (index + 1) % 50 == 0:
                log(f"  타일 생성 {split} {index + 1}/{len(files)}장")
        summary["splits"][split] = counts
    if not summary["splits"].get("train", {}).get("tiles"):
        raise ValueError("타일로 자를 학습 영상이 없습니다")
    if sizes:
        summary["region"] = list(max(sizes, key=sizes.get))
    return summary
