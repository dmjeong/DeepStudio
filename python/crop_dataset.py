"""Create an isolated, lossless cropped training snapshot, including spatial labels.

Snapshots persist for interrupted native training. The source and split membership
are never changed. A partially written snapshot is never handed to a trainer.
"""
import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image
from center_crop import center_crop_box, load_crop_image, validate_center_crop
from spatial_data import image_files, read_spatial_labels, semantic_pairs


def crop_spatial_rows(rows, task, image_size, crop):
    width, height = image_size
    left, top, right, bottom = center_crop_box(image_size, crop)
    cw, ch = right-left, bottom-top
    output = []
    for row in rows:
        cls = int(row[0])
        if task == "obb":
            from obb import crop_obb_row
            converted = crop_obb_row(row, image_size, (left, top, right, bottom))
            if converted is not None:
                output.append(converted)
        elif task == "detect":
            cx, cy, w, h = row[1:]
            x1, y1 = max(left, (cx-w/2)*width), max(top, (cy-h/2)*height)
            x2, y2 = min(right, (cx+w/2)*width), min(bottom, (cy+h/2)*height)
            if x2 > x1 and y2 > y1:
                output.append([cls, ((x1+x2)/2-left)/cw, ((y1+y2)/2-top)/ch,
                               (x2-x1)/cw, (y2-y1)/ch])
        else:
            # Intersection can split concave objects. Preserve separate components;
            # never bridge disconnected regions into a false foreground polygon.
            from shapely.geometry import Polygon, box
            polygon = Polygon([(x*width, y*height) for x, y in zip(row[1::2], row[2::2])])
            if not polygon.is_valid:
                raise ValueError("자기 교차 등 유효하지 않은 분할 폴리곤입니다")
            intersection = polygon.intersection(box(left, top, right, bottom))
            def parts(geometry):
                if geometry.geom_type == "Polygon":
                    yield geometry
                elif hasattr(geometry, "geoms"):
                    for part in geometry.geoms:
                        yield from parts(part)
            for part in parts(intersection):
                if part.is_empty or part.area <= 1e-9:
                    continue
                if part.interiors:
                    raise ValueError("구멍을 포함한 분할 라벨은 정규화 폴리곤으로 변환할 수 없습니다")
                values = [cls]
                for x, y in list(part.exterior.coords)[:-1]:
                    values.extend([min(1., max(0., (x-left)/cw)), min(1., max(0., (y-top)/ch))])
                output.append(values)
    return output


def prepare_crop_dataset(root, output_parent, task, crop, *, native=False,
                         num_classes=1, should_stop=lambda: False, log=lambda message: None):
    crop = validate_center_crop(crop)
    if crop is None:
        return str(root)
    root, parent = Path(root).resolve(), Path(output_parent).resolve()
    if not root.is_dir():
        raise ValueError(f"데이터 폴더 없음: {root}")
    if parent == root or root in parent.parents:
        raise ValueError("크롭 작업 폴더는 원본 데이터 폴더 밖이어야 합니다")
    spatial = task in ("detect", "segment", "obb")
    semantic = task == "segment" and not native
    files = image_files(root / "images" if spatial else root)
    if not files:
        raise ValueError("중앙 크롭할 이미지가 없습니다")
    pairs = dict(semantic_pairs(root / "images", root / "masks")) if semantic else {}
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".center-crop-", dir=parent))
    final = parent / staging.name.lstrip(".")
    try:
        # Retain empty class directories so class indices do not change.
        for folder in root.rglob("*"):
            if folder.is_dir() and not any(p.startswith(".") for p in folder.relative_to(root).parts):
                (staging / folder.relative_to(root)).mkdir(parents=True, exist_ok=True)
        seen = set()
        for index, path in enumerate(files):
            if should_stop():
                raise InterruptedError("중앙 크롭 데이터 준비 취소")
            relative = path.relative_to(root).with_suffix(".png")
            # Case-insensitive uniqueness also protects Windows deployments.
            key = relative.as_posix().casefold()
            if key in seen:
                raise ValueError(f"크롭 출력 이름 중복: {relative}")
            seen.add(key)
            try:
                if task == "obb":
                    from obb import require_unrotated_image
                    require_unrotated_image(path)
                with load_crop_image(path, exif=native) as source:
                    size = source.size
                    box = center_crop_box(size, crop)
                    image = source.crop(box)
                    # Existing native/custom engines use 8-bit RGB or L. Preserve
                    # uint16 originals losslessly here; conversion stays in the loader.
                    if image.mode not in ("RGB", "RGBA", "L", "LA", "P", "I", "I;16", "1"):
                        image = image.convert("RGB")
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image.save(target, format="PNG")
                if semantic:
                    with Image.open(pairs[str(path)]) as mask:
                        if mask.size != size:
                            raise ValueError("이미지와 정답 마스크의 원본 크기가 다릅니다")
                        if mask.mode not in ("P", "L", "I", "I;16"):
                            raise ValueError("정답 마스크는 클래스 인덱스 이미지여야 합니다")
                        destination = staging / "masks" / relative.relative_to("images")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        mask.crop(box).save(destination, format="PNG")
                elif spatial:
                    label = root / "labels" / path.relative_to(root / "images").with_suffix(".txt")
                    rows = crop_spatial_rows(read_spatial_labels(label, task, num_classes), task, size, crop)
                    destination = staging / "labels" / relative.relative_to("images").with_suffix(".txt")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text("".join(" ".join(format(v, ".12g") for v in row)+"\n" for row in rows), encoding="utf-8")
            except Exception as exc:
                raise ValueError(f"중앙 크롭 실패: {path.name}: {exc}") from exc
            if index == 0 or (index+1) % 100 == 0 or index+1 == len(files):
                log(f"중앙 크롭 데이터 준비: {index+1}/{len(files)}")
        (staging / "center_crop.json").write_text(json.dumps({
            "source_root": str(root), "center_crop": crop, "task": task,
            "native": native, "images": len(files), "status": "completed",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if should_stop():
            raise InterruptedError("중앙 크롭 데이터 준비 취소")
        staging.rename(final)
        return str(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
