"""Editable semantic annotations stored inside the training PNG itself.

Pixels remain an ordinary uint8 class-index mask. PNG text retains the original
raster and ordered drawing operations so polygons remain editable after reopen.
External pixel/class edits invalidate the metadata and preserve the new raster.
"""

import base64
import copy
import hashlib
import io
import json
import uuid

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

METADATA_KEY = "deepvision_annotation_v1"


def _png(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def _validate_pixels(pixels, width, height, classes):
    if pixels.shape != (height, width) or pixels.dtype.kind not in "ui":
        raise ValueError("원본 이미지와 같은 크기의 정수 클래스 마스크가 필요합니다.")
    # Fast exact path: an unsigned mask whose maximum is already a valid class
    # needs no copy of its non-255 pixels (the common case without ignore pixels).
    if not (pixels.dtype.kind == "u" and pixels.size and pixels.max() < classes):
        valid = pixels[pixels != 255]
        if valid.size and (valid.min() < 0 or valid.max() >= classes):
            raise ValueError("마스크의 클래스 번호가 프로젝트와 맞지 않습니다.")
    return pixels.astype(np.uint8, copy=True)


class MaskDocument:
    def __init__(self, width, height, classes, base=None, shapes=(), objects=None):
        if type(width) is not int or type(height) is not int or min(width, height) < 1:
            raise ValueError("이미지 크기 오류")
        if type(classes) is not int or not 1 <= classes <= 255:
            raise ValueError("분할 클래스는 배경을 포함해 1~255개여야 합니다.")
        self.width, self.height, self.classes = width, height, classes
        self.persisted = False
        self.base = (
            np.zeros((height, width), dtype=np.uint8)
            if base is None
            else _validate_pixels(np.asarray(base), width, height, classes)
        )
        self.base.setflags(write=False)
        self.base_digest = hashlib.sha256(self.base.tobytes()).hexdigest()
        self.base_classes = tuple(int(v) for v in np.unique(self.base))
        self.shapes = [self.validate_shape(shape) for shape in shapes]
        self.objects = self._validate_objects(objects or {})
        self._cache_key = None
        self._pixels = self._owners = None
        self.managed_cache = False
        self._cache_valid = False
        self._rendering = False
        self._key_indices = {}
        self._bounds_cache = {}

    def ensure_object_ids(self, namespace):
        """Deterministic IDs let a v1 draft reopen before its first PNG save."""
        remap = {}
        for i, shape in enumerate(self.shapes):
            if shape.get("operation") == "erase":
                continue
            old = f"shape:{i}"
            if "object_id" not in shape:
                shape["object_id"] = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    namespace + ":" + str(i) + ":" + json.dumps(shape, sort_keys=True),
                ).hex
            remap[old] = shape["object_id"]
        for shape in self.shapes:
            if "scope_object" in shape:
                shape["scope_object"] = remap.get(
                    shape["scope_object"], shape["scope_object"]
                )
            if "protected_objects" in shape:
                shape["protected_objects"] = [
                    remap.get(k, k) for k in shape["protected_objects"]
                ]
        self.objects = {remap.get(k, k): v for k, v in self.objects.items()}
        self.invalidate()

    def invalidate(self):
        self._cache_valid = False
        self._cache_key = None
        self._bounds_cache = {}

    def append_shape(self, shape):
        """Commit a new drawing by changing only its covered source pixels."""
        item = self.validate_shape(shape)
        self._ensure_cache()
        index = len(self.shapes)
        self.shapes.append(item)
        self._bounds_cache = {}
        if item.get("operation") != "erase":
            self._key_indices.setdefault(self.key(index), index)
        self._apply(self._pixels, self._owners, item, index)
        self._cache_key = None
        self._cache_valid = bool(self.managed_cache)
        return self.shape_region(item)[0]

    @staticmethod
    def _validate_objects(objects):
        if not isinstance(objects, dict) or len(objects) > 20000:
            raise ValueError("영역 속성 형식 오류")
        result = {}
        for key, value in objects.items():
            if (
                not isinstance(key, str)
                or len(key) > 128
                or not isinstance(value, dict)
            ):
                raise ValueError("영역 식별자 형식 오류")
            row = {}
            for flag in ("locked", "hidden"):
                if flag in value:
                    if type(value[flag]) is not bool:
                        raise ValueError("영역 잠금·표시 속성 오류")
                    row[flag] = value[flag]
            for name in ("row", "column"):
                if name in value:
                    if type(value[name]) is not int or not 1 <= value[name] <= 4096:
                        raise ValueError("격자 행·열 속성 오류")
                    row[name] = value[name]
            result[key] = row
        return result

    def key(self, index):
        if index < -1:
            return f"base:{-index - 2}"
        return self.shapes[index].get("object_id", f"shape:{index}")

    def representative(self, key):
        if key.startswith("base:"):
            return -int(key.split(":")[1]) - 2
        if self._rendering or (self.managed_cache and self._cache_valid):
            return self._key_indices.get(key, -1)
        return next(
            (
                i
                for i, s in enumerate(self.shapes)
                if self.key(i) == key and s.get("operation") != "erase"
            ),
            -1,
        )

    def validate_shape(self, shape):
        if not isinstance(shape, dict) or shape.get("kind") not in {
            "polygon",
            "stroke",
            "raster",
        }:
            raise ValueError("다각형·브러시·마스크 정답이 필요합니다.")
        class_id = shape.get("class_id")
        if type(class_id) is not int or not (
            0 <= class_id < self.classes or class_id == 255
        ):
            raise ValueError("정답 클래스 번호 오류")
        kind = shape["kind"]
        result = {"kind": kind, "class_id": class_id}
        if kind == "raster":
            bounds = shape.get("bounds")
            if (
                not isinstance(bounds, list)
                or len(bounds) != 4
                or any(type(v) is not int for v in bounds)
                or min(bounds[:2]) < 0
                or min(bounds[2:]) < 1
                or bounds[0] + bounds[2] > self.width
                or bounds[1] + bounds[3] > self.height
            ):
                raise ValueError("래스터 영역 범위 오류")
            try:
                raw = base64.b64decode(shape["mask_png"], validate=True)
                with Image.open(io.BytesIO(raw)) as mask:
                    if mask.size != tuple(bounds[2:]) or mask.mode not in {"1", "L"}:
                        raise ValueError("래스터 영역 크기 오류")
                    mask.load()
            except (KeyError, TypeError, OSError, ValueError) as exc:
                raise ValueError("래스터 영역을 읽을 수 없습니다.") from exc
            result.update(bounds=list(bounds), mask_png=shape["mask_png"])
        else:
            points = shape.get("points")
            minimum = 6 if kind == "polygon" else 2
            if (
                not isinstance(points, list)
                or not minimum <= len(points) <= 65536
                or len(points) % 2
                or any(
                    type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1
                    for v in points
                )
            ):
                raise ValueError("정답 좌표는 이미지 안의 유한한 값이어야 합니다.")
            result["points"] = list(points)
            if kind == "stroke":
                size = shape.get("width")
                if type(size) is not int or not 1 <= size <= 1024:
                    raise ValueError("브러시 크기는 1~1024 px여야 합니다.")
                result["width"] = size
                if "tip" in shape:
                    if shape["tip"] not in {"circle", "square"}:
                        raise ValueError("브러시 모양 오류")
                    result["tip"] = shape["tip"]
            else:
                xy = np.asarray(points).reshape(-1, 2)
                area = (
                    abs(
                        np.dot(xy[:, 0], np.roll(xy[:, 1], 1))
                        - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))
                    )
                    / 2
                )
                if area * self.width * self.height < 0.5:
                    raise ValueError("면적이 없는 다각형은 저장할 수 없습니다.")
                if polygon_crosses(xy):
                    raise ValueError("다각형의 변이 교차합니다. 꼭짓점을 수정하세요.")
        operation = shape.get("operation", "paint")
        if operation not in {"paint", "erase"} or (
            operation == "erase" and (kind != "stroke" or class_id != 0)
        ):
            raise ValueError("지우개는 배경 클래스의 브러시 작업이어야 합니다.")
        if operation == "erase":
            result["operation"] = "erase"
        for field in ("object_id", "scope_object"):
            if field in shape:
                value = shape[field]
                if not isinstance(value, str) or not 1 <= len(value) <= 128:
                    raise ValueError("영역 식별자 오류")
                result[field] = value
        if "protected_objects" in shape:
            keys = shape["protected_objects"]
            if (
                not isinstance(keys, list)
                or len(keys) > 20000
                or any(not isinstance(k, str) or not 1 <= len(k) <= 128 for k in keys)
            ):
                raise ValueError("보호 영역 형식 오류")
            result["protected_objects"] = list(keys)
        if "scope_class" in shape:
            value = shape["scope_class"]
            if type(value) is not int or not (
                0 <= value < self.classes or value == 255
            ):
                raise ValueError("편집 대상 클래스 오류")
            result["scope_class"] = value
        for field in ("background_only",):
            if field in shape:
                if type(shape[field]) is not bool:
                    raise ValueError("편집 범위 오류")
                result[field] = shape[field]
        return result

    def shape_region(self, item):
        """Return a small coverage bitmap and its original-resolution bounds."""
        if item["kind"] == "raster":
            with Image.open(io.BytesIO(base64.b64decode(item["mask_png"]))) as image:
                return tuple(item["bounds"]), np.asarray(image).astype(bool)
        points = [
            (
                round(item["points"][i] * (self.width - 1)),
                round(item["points"][i + 1] * (self.height - 1)),
            )
            for i in range(0, len(item["points"]), 2)
        ]
        radius = max(0, (item.get("width", 1) - 1) / 2)
        pad = int(np.ceil(radius)) + 2
        x = max(0, min(p[0] for p in points) - pad)
        y = max(0, min(p[1] for p in points) - pad)
        right = min(self.width, max(p[0] for p in points) + pad + 1)
        bottom = min(self.height, max(p[1] for p in points) + pad + 1)
        image = Image.new("L", (right - x, bottom - y), 0)
        draw = ImageDraw.Draw(image)
        local = [(a - x, b - y) for a, b in points]
        if item["kind"] == "polygon":
            draw.polygon(local, fill=1)
        elif item.get("tip") == "square":
            for a, b in zip(local, local[1:] or local):
                corners = sorted(
                    {
                        (x + dx, y + dy)
                        for x, y in (a, b)
                        for dx, dy in [
                            (-radius, -radius),
                            (radius, -radius),
                            (radius, radius),
                            (-radius, radius),
                        ]
                    }
                )

                def cross(o, a, b):
                    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

                lower = []
                upper = []
                for p in corners:
                    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                        lower.pop()
                    lower.append(p)
                for p in reversed(corners):
                    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                        upper.pop()
                    upper.append(p)
                hull = lower[:-1] + upper[:-1]
                if len(hull) >= 2:
                    draw.polygon(hull, fill=1)
                elif corners:
                    draw.point(corners[0], fill=1)
        else:
            if len(local) > 1:
                draw.line(local, fill=1, width=item["width"], joint="curve")
            for a, b in local:
                draw.ellipse((a - radius, b - radius, a + radius, b + radius), fill=1)
        return (x, y, right - x, bottom - y), np.asarray(image).astype(bool)

    def _apply(self, pixels, owners, item, index, origin=(0, 0)):
        (x, y, w, h), coverage = self.shape_region(item)
        ox, oy = origin
        target = pixels[y - oy : y - oy + h, x - ox : x - ox + w]
        own = owners[y - oy : y - oy + h, x - ox : x - ox + w]
        key = item.get("object_id", f"shape:{index}")
        representative = self.representative(key) if index < len(self.shapes) else index
        if representative == -1:
            representative = index
        if "scope_class" in item:
            coverage &= target == item["scope_class"]
        if "scope_object" in item:
            coverage &= own == self.representative(item["scope_object"])
        if item.get("background_only"):
            coverage &= target == 0
        # Ignore pixels are explicit training exclusions, never ordinary erase targets.
        explicit_ignore = (
            item.get("scope_class") == 255 or item.get("scope_object") == "base:255"
        )
        scope_index = (
            self.representative(item["scope_object"]) if "scope_object" in item else -1
        )
        if scope_index >= 0 and self.shapes[scope_index]["class_id"] == 255:
            explicit_ignore = True
        if item.get("class_id") != 255 and not explicit_ignore:
            coverage &= target != 255
        # Protection belongs to the edit at the time it was made. Toggling a
        # lock later must never change already saved training pixels.
        protected = [
            self.representative(locked)
            for locked in item.get("protected_objects", [])
            if locked != key
        ]
        if protected:
            coverage &= ~np.isin(own, protected)
        target[coverage] = item["class_id"]
        own[coverage] = -2 if item["class_id"] == 0 else representative

    def _ensure_cache(self):
        if self.managed_cache and self._cache_valid:
            return
        key = (
            id(self.base),
            self.classes,
            json.dumps(self.shapes, sort_keys=True, separators=(",", ":")),
            json.dumps(self.objects, sort_keys=True, separators=(",", ":")),
        )
        if key != self._cache_key:
            pixels = self.base.copy()
            # Same values as -base - 2 in one int32 allocation (4096²: 64 MiB, not 3×).
            owners = np.subtract(-2, self.base, dtype=np.int32)
            self._key_indices = {}
            for index, item in enumerate(self.shapes):
                if item.get("operation") != "erase":
                    self._key_indices.setdefault(self.key(index), index)
            self._rendering = True
            try:
                for index, item in enumerate(self.shapes):
                    self._apply(pixels, owners, item, index)
            finally:
                self._rendering = False
            self._pixels, self._owners, self._cache_key = pixels, owners, key
        self._cache_valid = True

    def render(self, preview=None, *, copy_pixels=True):
        self._ensure_cache()
        if preview is None:
            return self._pixels.copy() if copy_pixels else self._pixels
        item = self.validate_shape(preview)
        pixels = self._pixels.copy()
        owners = self._owners.copy()
        self._apply(pixels, owners, item, len(self.shapes))
        return pixels

    def preview_region(self, shape):
        self._ensure_cache()
        item = self.validate_shape(shape)
        bounds, _ = self.shape_region(item)
        x, y, w, h = bounds
        pixels = self._pixels[y : y + h, x : x + w].copy()
        owners = self._owners[y : y + h, x : x + w].copy()
        self._apply(pixels, owners, item, len(self.shapes), (x, y))
        return bounds, pixels

    def owner_at(self, x, y):
        self._ensure_cache()
        if (
            not (0 <= x < self.width and 0 <= y < self.height)
            or self._pixels[y, x] == 0
        ):
            return -1
        index = int(self._owners[y, x])
        key = self.key(index)
        return -1 if self.objects.get(key, {}).get("hidden") else index

    def bounds(self, index):
        self._ensure_cache()
        if index in self._bounds_cache:
            return self._bounds_cache[index]
        if index < -1:
            yy, xx = np.where(self.base == (-index - 2))
            value = (
                None
                if not len(xx)
                else [
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max() - xx.min() + 1),
                    int(yy.max() - yy.min() + 1),
                ]
            )
        else:
            key = self.key(index)
            bounds = []
            for i, item in enumerate(self.shapes):
                if self.key(i) != key or item.get("operation") == "erase":
                    continue
                if item["kind"] == "raster":
                    bounds.append(item["bounds"])
                    continue
                p = np.asarray(item["points"]).reshape(-1, 2) * [
                    max(1, self.width - 1),
                    max(1, self.height - 1),
                ]
                radius = item.get("width", 1) / 2
                low = np.maximum(0, np.floor(p.min(axis=0) - radius))
                high = np.minimum(
                    [self.width, self.height], np.ceil(p.max(axis=0) + radius) + 1
                )
                bounds.append([*low, *(high - low)])
            if bounds:
                x = min(b[0] for b in bounds)
                y = min(b[1] for b in bounds)
                value = [
                    int(x),
                    int(y),
                    int(max(b[0] + b[2] for b in bounds) - x),
                    int(max(b[1] + b[3] for b in bounds) - y),
                ]
            else:
                value = None
        self._bounds_cache[index] = value
        return value

    def raster_shape(self, mask, class_id, **fields):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.height, self.width) or not mask.any():
            raise ValueError("적용할 마스크 영역이 없습니다.")
        yy, xx = np.where(mask)
        x, y = int(xx.min()), int(yy.min())
        w, h = int(xx.max() - x + 1), int(yy.max() - y + 1)
        return self.validate_shape(
            {
                "kind": "raster",
                "class_id": class_id,
                "bounds": [x, y, w, h],
                "mask_png": base64.b64encode(
                    _png(mask[y : y + h, x : x + w].astype(np.uint8))
                ).decode(),
                **fields,
            }
        )

    def connected_region(self, x, y):
        import cv2

        pixels = self.render(copy_pixels=False)
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("이미지 안을 선택하세요.")
        plane = (pixels == pixels[y, x]).astype(np.uint8)
        cv2.floodFill(plane, None, (x, y), 2, flags=4)
        return plane == 2

    def component_shapes(self, index):
        """Split visible owner pixels, retaining holes and the original class."""
        import cv2

        if (
            type(index) is not int
            or index == -1
            or index >= len(self.shapes)
            or (index < -1 and -index - 2 not in self.base_classes)
        ):
            raise ValueError("나눌 영역이 없습니다.")
        self._ensure_cache()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (self._owners == index).astype(np.uint8), connectivity=8
        )
        class_id = -index - 2 if index < -1 else self.shapes[index]["class_id"]
        if count > 20001:
            raise ValueError(
                "연결 부분이 20,000개를 넘습니다. 작은 노이즈를 먼저 정리하세요."
            )
        result = []
        for component in range(1, count):
            x, y, width, height, _ = (int(v) for v in stats[component])
            pixels = (labels[y : y + height, x : x + width] == component).astype(
                np.uint8
            )
            result.append(
                self.validate_shape(
                    {
                        "kind": "raster",
                        "class_id": class_id,
                        "bounds": [x, y, width, height],
                        "mask_png": base64.b64encode(_png(pixels)).decode(),
                    }
                )
            )
        return result

    def payload(self):
        version = (
            2
            if self.objects
            or any(
                s["kind"] == "raster"
                or any(
                    k in s
                    for k in (
                        "object_id",
                        "scope_object",
                        "scope_class",
                        "background_only",
                        "tip",
                    )
                )
                for s in self.shapes
            )
            else 1
        )
        value = {
            "schema_version": version,
            "width": self.width,
            "height": self.height,
            "base_png": base64.b64encode(_png(self.base)).decode("ascii"),
            "shapes": copy.deepcopy(self.shapes),
        }
        if self.objects:
            value["objects"] = copy.deepcopy(self.objects)
        return value

    @classmethod
    def from_payload(cls, value, width, height, classes):
        if (
            not isinstance(value, dict)
            or value.get("schema_version") not in (1, 2)
            or (value.get("width"), value.get("height")) != (width, height)
            or not isinstance(value.get("shapes"), list)
            or len(value["shapes"]) > 10000
        ):
            raise ValueError("분할 편집 문서의 형식 또는 이미지 크기 오류")
        try:
            raw = base64.b64decode(value["base_png"], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != (width, height):
                    raise ValueError("원본 마스크 크기 오류")
                base = np.array(image)
        except (KeyError, TypeError, OSError, ValueError) as exc:
            raise ValueError("편집 문서의 원본 마스크를 읽을 수 없습니다.") from exc
        return cls(
            width,
            height,
            classes,
            base=base,
            shapes=value["shapes"],
            objects=value.get("objects", {}),
        )

    def encode(self):
        pixels = self.render()
        metadata = self.payload()
        metadata["pixel_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
        info = PngImagePlugin.PngInfo()
        info.add_text(METADATA_KEY, json.dumps(metadata, separators=(",", ":")))
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG", pnginfo=info)
        return stream.getvalue()

    @classmethod
    def from_image(cls, image, width, height, classes):
        pixels = _validate_pixels(np.array(image), width, height, classes)
        encoded = image.info.get(METADATA_KEY)
        if encoded:
            try:
                value = json.loads(encoded)
                if (
                    value.get("pixel_sha256")
                    == hashlib.sha256(pixels.tobytes()).hexdigest()
                ):
                    document = cls.from_payload(value, width, height, classes)
                    if np.array_equal(document.render(), pixels):
                        document.persisted = True
                        return document
            except (ValueError, TypeError, AttributeError):
                pass
        document = cls(width, height, classes, base=pixels)
        document.persisted = True
        return document


def polygon_crosses(points):
    """Detect proper crossings and overlapping non-adjacent polygon edges."""
    p = np.asarray(points, dtype=float)
    p = p[np.r_[True, (np.diff(p, axis=0) != 0).any(axis=1)]]
    if len(p) > 1 and np.array_equal(p[0], p[-1]):
        p = p[:-1]
    n = len(p)
    if n < 4:
        return False
    q = np.roll(p, -1, axis=0)
    for i in range(n - 2):
        js = np.arange(i + 2, n if i else n - 1)
        if not len(js):
            continue
        a, b = p[i], q[i]
        c, d = p[js], q[js]
        overlaps = (
            np.maximum(np.minimum(a, b), np.minimum(c, d))
            <= np.minimum(np.maximum(a, b), np.maximum(c, d)) + 1e-12
        ).all(axis=1)
        if not overlaps.any():
            continue
        c, d = c[overlaps], d[overlaps]

        def cross(u, v):
            return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

        if (
            (cross(b - a, c - a) * cross(b - a, d - a) <= 1e-20)
            & (cross(d - c, a - c) * cross(d - c, b - c) <= 1e-20)
        ).any():
            return True
    return False


def load_mask_document(project, image_path, split):
    from core.class_management import _plain_path
    from core.dataset_editor import read_annotations, sidecars, split_root

    image_path = _plain_path(image_path)
    image_path.relative_to(split_root(project, split))
    with Image.open(image_path) as image:
        width, height = image.size
    related = sidecars(project, image_path, split)
    masks = [path for kind, path in related if kind == "masks"]
    if len(masks) > 1:
        raise ValueError(
            "같은 이미지에 마스크가 여러 개 있습니다. 사용할 마스크 한 개만 남겨 주세요."
        )
    classes = max(1, len(project.data.class_names))
    if masks:
        with Image.open(masks[0]) as mask:
            return MaskDocument.from_image(mask, width, height, classes)
    rows = [
        row
        for kind, path in related
        if kind == "labels"
        for row in read_annotations(path, "segment", classes)
    ]
    return MaskDocument(
        width,
        height,
        classes,
        shapes=[
            {
                "kind": "polygon",
                "class_id": row["class_id"],
                "points": row["coordinates"],
            }
            for row in rows
        ],
    )
