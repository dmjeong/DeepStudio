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
    valid = pixels[pixels != 255]
    if valid.size and (valid.min() < 0 or valid.max() >= classes):
        raise ValueError("마스크의 클래스 번호가 프로젝트와 맞지 않습니다.")
    return pixels.astype(np.uint8, copy=True)


class MaskDocument:
    def __init__(self, width, height, classes, base=None, shapes=()):
        if type(width) is not int or type(height) is not int or min(width, height) < 1:
            raise ValueError("이미지 크기 오류")
        if type(classes) is not int or not 1 <= classes <= 255:
            raise ValueError("분할 클래스는 배경을 포함해 1~255개여야 합니다.")
        self.width, self.height, self.classes = width, height, classes
        self.persisted = False
        self.base = (np.zeros((height, width), dtype=np.uint8) if base is None else
                     _validate_pixels(np.asarray(base), width, height, classes))
        self.shapes = [self.validate_shape(shape) for shape in shapes]

    def validate_shape(self, shape):
        if not isinstance(shape, dict) or shape.get("kind") not in {"polygon", "stroke"}:
            raise ValueError("다각형 또는 브러시 정답이 필요합니다.")
        class_id = shape.get("class_id")
        if type(class_id) is not int or not (0 <= class_id < self.classes or class_id == 255):
            raise ValueError("정답 클래스 번호 오류")
        points = shape.get("points")
        minimum = 6 if shape["kind"] == "polygon" else 2
        if (not isinstance(points, list) or not minimum <= len(points) <= 65536 or len(points) % 2 or
                any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in points)):
            raise ValueError("정답 좌표는 이미지 안의 유한한 값이어야 합니다.")
        result = {"kind": shape["kind"], "class_id": class_id, "points": list(points)}
        if shape["kind"] == "stroke":
            width = shape.get("width")
            if type(width) is not int or not 1 <= width <= 1024:
                raise ValueError("브러시 크기는 1~1024 px여야 합니다.")
            result["width"] = width
        else:
            xy = np.asarray(points).reshape(-1, 2)
            area = abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))) / 2
            if area * self.width * self.height < .5:
                raise ValueError("면적이 없는 다각형은 저장할 수 없습니다.")
        return result

    def render(self):
        image = Image.fromarray(self.base.copy())
        draw = ImageDraw.Draw(image)
        for item in self.shapes:
            points = [(round(item["points"][i] * (self.width - 1)), round(item["points"][i+1] * (self.height - 1)))
                      for i in range(0, len(item["points"]), 2)]
            if item["kind"] == "polygon":
                draw.polygon(points, fill=item["class_id"])
            else:
                size, fill = item["width"], item["class_id"]
                if len(points) > 1:
                    draw.line(points, fill=fill, width=size, joint="curve")
                radius = max(0, (size - 1) / 2)
                for x, y in points:
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill)
        return np.asarray(image).copy()

    def payload(self):
        return {"schema_version": 1, "width": self.width, "height": self.height,
                "base_png": base64.b64encode(_png(self.base)).decode("ascii"),
                "shapes": copy.deepcopy(self.shapes)}

    @classmethod
    def from_payload(cls, value, width, height, classes):
        if (not isinstance(value, dict) or value.get("schema_version") != 1 or
                (value.get("width"), value.get("height")) != (width, height) or
                not isinstance(value.get("shapes"), list) or len(value["shapes"]) > 10000):
            raise ValueError("분할 편집 문서의 형식 또는 이미지 크기 오류")
        try:
            raw = base64.b64decode(value["base_png"], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != (width, height):
                    raise ValueError("원본 마스크 크기 오류")
                base = np.array(image)
        except (KeyError, TypeError, OSError, ValueError) as exc:
            raise ValueError("편집 문서의 원본 마스크를 읽을 수 없습니다.") from exc
        return cls(width, height, classes, base=base, shapes=value["shapes"])

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
                if value.get("pixel_sha256") == hashlib.sha256(pixels.tobytes()).hexdigest():
                    document = cls.from_payload(value, width, height, classes)
                    if np.array_equal(document.render(), pixels):
                        document.persisted = True
                        return document
            except (ValueError, TypeError, AttributeError):
                pass
        # Class remapping/external editors may legitimately change the pixels.
        document = cls(width, height, classes, base=pixels)
        document.persisted = True
        return document


def load_mask_document(project, image_path, split):
    from core.dataset_editor import sidecars, split_root, read_annotations
    from core.class_management import _plain_path
    image_path = _plain_path(image_path)
    image_path.relative_to(split_root(project, split))
    with Image.open(image_path) as image:
        width, height = image.size
    related = sidecars(project, image_path, split)
    masks = [path for kind, path in related if kind == "masks"]
    if len(masks) > 1:
        raise ValueError("같은 이미지에 마스크가 여러 개 있습니다. 사용할 마스크 한 개만 남겨 주세요.")
    classes = max(1, len(project.data.class_names))
    if masks:
        with Image.open(masks[0]) as mask:
            return MaskDocument.from_image(mask, width, height, classes)
    rows = [row for kind, path in related if kind == "labels"
            for row in read_annotations(path, "segment", classes)]
    return MaskDocument(width, height, classes, shapes=[
        {"kind": "polygon", "class_id": row["class_id"], "points": row["coordinates"]} for row in rows])
