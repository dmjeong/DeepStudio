"""합성 이미지, 변경 마스크와 기록을 하나의 디렉터리로 원자적으로 공개한다."""
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import numpy as np
from PIL import Image


def read_image_snapshot(path):
    """같은 바이트에서 영상과 해시를 구한다."""
    import cv2
    content = Path(path).read_bytes()
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"이미지 읽기 실패: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return image, hashlib.sha256(content).hexdigest()


def read_image(path):
    return read_image_snapshot(path)[0]


def png_bytes(image):
    import cv2
    pixels = image
    if image.ndim == 3 and image.shape[2] == 3:
        pixels = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise OSError("PNG 인코딩 실패")
    return encoded.tobytes()


def read_roi(path, shape):
    if not path:
        return None
    with Image.open(path) as image:
        roi = np.asarray(image.convert("L")) > 0
    if roi.shape != tuple(shape):
        raise ValueError(f"ROI 크기 불일치: 마스크 {roi.shape}, 원본 {tuple(shape)}. 원본 좌표의 마스크 필요")
    if not roi.any():
        raise ValueError("비어 있는 ROI")
    return roi


def durable_bytes(path, content):
    """임시 작업 디렉터리 안의 파일을 완전히 기록한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def verify_batch(directory, manifest):
    """저장된 이미지, 원본, 기록과 ZIP이 모두 일치할 때만 완료로 인정한다."""
    root = Path(directory).resolve()
    try:
        records = manifest["samples"]
        if (manifest.get("schema_version") != 3 or not isinstance(records, list)
                or not records or manifest["count"] != len(records)):
            raise ValueError("배치 기록 형식 오류")
        for index, record in enumerate(records, 1):
            for key in ("image", "mask", "original"):
                if key == "original" and key not in record:
                    continue
                path = (root / record[key]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("배치 밖의 파일 경로")
                if hashlib.sha256(path.read_bytes()).hexdigest() != record[f"{key}_sha256"]:
                    raise ValueError("파일 해시 불일치")
            recipe = json.loads((root / "recipes" / f"{index:06d}.json").read_text(encoding="utf-8"))
            if recipe != record:
                raise ValueError("생성 기록 불일치")
        expected = {p.relative_to(root).as_posix(): p for folder in ("masks", "recipes", "references")
                    for p in (root / folder).glob("*") if p.is_file()}
        with zipfile.ZipFile(root / "annotations.zip") as archive:
            names = archive.namelist()
            if len(names) != len(expected) or set(names) != set(expected):
                raise ValueError("ZIP 구성 파일 불일치")
            for name, path in expected.items():
                if archive.read(name) != path.read_bytes():
                    raise ValueError("ZIP 내용 불일치")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"합성 배치 손상: {exc}") from exc


class DefectBatchWriter:
    """완료 전 파일은 임시 폴더에만 존재하며 예외 시 제거한다.

    공개 배치는 images, masks, recipes, manifest.json과 호환용 주석 ZIP을 포함한다.
    마스크는 실제 픽셀 변경 영역이며 검수된 불량 정답이라는 의미가 아니다.
    """
    def __init__(self, output_dir, *, batch_id=None):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = batch_id or datetime.now().strftime("%y%m%d_%H%M%S")
        if not stamp or any(c not in "0123456789_" for c in stamp):
            raise ValueError("합성 배치 ID 형식 오류")
        for suffix in range(10000):
            self.batch_id = stamp + (f"_{suffix}" if suffix else "")
            self.output_dir = self.root / self.batch_id
            lock = self.root / f".{self.batch_id}.lock"
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
            except FileExistsError:
                continue
            if self.output_dir.exists():
                lock.unlink()
                continue
            self._lock = lock
            break
        else:
            raise FileExistsError("합성 배치 이름 예약 실패")
        try:
            self._stage = Path(tempfile.mkdtemp(prefix=".defect-", dir=self.root))
        except BaseException:
            self._lock.unlink(missing_ok=True)
            raise
        self.archive_path = self.output_dir / "annotations.zip"
        self.count = 0
        self.records = []
        self._closed = False

    def add_reference(self, name, path):
        if path:
            if name not in {"roi", "texture"}:
                raise ValueError("참조 종류 오류")
            durable_bytes(self._stage / "references" / f"{name}{Path(path).suffix.lower()}", Path(path).read_bytes())

    def save(self, sample, source, *, original=None, source_sha256=None):
        if self._closed:
            raise RuntimeError("종료된 합성 배치")
        if not sample.mask.any():
            raise ValueError(f"실제 변경된 픽셀 없음. 강도 또는 ROI 조절 필요: {source}")
        if sample.mask.shape != sample.image.shape[:2]:
            raise ValueError("이미지와 변경 마스크 크기 불일치")
        if original is not None:
            if original.shape != sample.image.shape or original.dtype != sample.image.dtype:
                raise ValueError("원본과 합성 결과의 크기 또는 정밀도 불일치")
            changed = sample.image != original
            if changed.ndim == 3:
                changed = changed.any(axis=2)
            if not np.array_equal(changed, sample.mask > 0):
                raise ValueError("실제 변경 픽셀과 마스크 불일치")
        index = self.count + 1
        kind = "mixed" if len(set(sample.types)) > 1 else sample.types[0].value
        name = f"{index:06d}_{kind}.png"
        image_path, mask_path = f"images/{name}", f"masks/{name}"
        pixels = png_bytes(sample.image)
        mask = io.BytesIO()
        Image.fromarray(sample.mask).save(mask, format="PNG")
        recipe = {**sample.recipe, "schema_version": 3, "source": str(Path(source).resolve()),
                  "source_sha256": source_sha256 or hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                  "image": image_path, "mask": mask_path, "mask_kind": "changed_pixels",
                  "image_sha256": hashlib.sha256(pixels).hexdigest(),
                  "mask_sha256": hashlib.sha256(mask.getvalue()).hexdigest()}
        original_bytes = png_bytes(original) if original is not None else None
        if original_bytes is not None:
            recipe["original_sha256"] = hashlib.sha256(original_bytes).hexdigest()
            recipe["original"] = f"originals/{index:06d}.png"
        paths = []
        try:
            for relative, content in ((image_path, pixels), (mask_path, mask.getvalue())):
                path = self._stage / relative
                paths.append(path)
                durable_bytes(path, content)
            if original is not None:
                path = self._stage / recipe["original"]
                paths.append(path)
                durable_bytes(path, original_bytes)
            path = self._stage / "recipes" / f"{index:06d}.json"
            paths.append(path)
            durable_bytes(path, json.dumps(recipe, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8"))
        except BaseException:
            for path in paths:
                path.unlink(missing_ok=True)
            raise
        self.records.append(recipe)
        self.count += 1
        return str(self.output_dir / image_path)

    def close(self):
        if self._closed:
            return
        try:
            if self.count:
                with zipfile.ZipFile(self._stage / "annotations.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
                    for folder in ("masks", "recipes", "references"):
                        for path in sorted((self._stage / folder).glob("*")):
                            archive.write(path, path.relative_to(self._stage).as_posix())
                with (self._stage / "annotations.zip").open("rb+") as stream:
                    os.fsync(stream.fileno())
                manifest = {"schema_version": 3, "synthetic": True, "count": self.count, "samples": self.records}
                durable_bytes(self._stage / "manifest.json", json.dumps(
                    manifest, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8"))
                verify_batch(self._stage, manifest)
                if self.output_dir.exists():
                    raise FileExistsError("출력 배치가 이미 존재함")
                os.rename(self._stage, self.output_dir)
        finally:
            self.abort()

    def abort(self):
        if not self._closed:
            shutil.rmtree(self._stage, ignore_errors=True)
            self._lock.unlink(missing_ok=True)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_args):
        if exc_type is None:
            self.close()
        else:
            self.abort()
