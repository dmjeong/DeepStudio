"""추론 이미지와 반응 맵을 임시 디스크에 보관해 선택 시 모델 실행을 없앤다."""

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


class InferencePreviewCache:
    """전체 해상도는 디스크, 격자는 축소본만 읽는 세션 단위 캐시."""

    def __init__(self):
        self._directory = None
        self._entries = {}

    def clear(self):
        self._entries.clear()
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

    @staticmethod
    def _thumbnail(array, limit=512):
        image = Image.fromarray(array)
        resampling = Image.Resampling.NEAREST if array.dtype == np.bool_ else Image.Resampling.BILINEAR
        image.thumbnail((limit, limit), resampling)
        return np.asarray(image).copy()

    def put(self, image_path, preview, heatmap=None, info=""):
        """소유권을 가진 배열 복사본을 저장한다. 실패를 호출자에게 전달한다."""
        if self._directory is None:
            self._directory = TemporaryDirectory(prefix="deep-studio-inference-")
        key = sha256(os.fsencode(image_path)).hexdigest()
        arrays = {}
        if preview is not None:
            arrays["preview"] = np.asarray(preview, dtype=np.uint8)
        metadata = {"image_path": image_path, "info": info, "kind": "", "heatmap_info": ""}
        if heatmap is not None:
            for name in ("original", "activation", "base_image", "valid_mask"):
                value = heatmap.get(name)
                if value is not None:
                    dtype = np.bool_ if name == "valid_mask" else np.float32 if name == "activation" else np.uint8
                    arrays[name] = np.asarray(value, dtype=dtype)
            metadata.update(kind=heatmap["kind"], heatmap_info=heatmap["info"],
                            detections=deepcopy(heatmap.get("detections")))
        paths = []
        try:
            for suffix, values in (("full", arrays), ("thumb", {
                    name: self._thumbnail(value) for name, value in arrays.items()})):
                destination = Path(self._directory.name) / f"{key}-{suffix}.npz"
                staging = destination.with_suffix(".tmp")
                paths.append(staging)
                with staging.open("wb") as stream:
                    np.savez(stream, **values)
                os.replace(staging, destination)
                metadata[suffix] = destination
        except Exception:
            for path in paths:
                path.unlink(missing_ok=True)
            self._entries.pop(image_path, None)
            raise
        self._entries[image_path] = metadata

    def get(self, image_path, *, thumbnail=False):
        """한 이미지의 복사본을 반환한다. pickle이나 모델 객체는 읽지 않는다."""
        entry = self._entries.get(image_path)
        if entry is None:
            return None
        with np.load(entry["thumb" if thumbnail else "full"], allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        heatmap = None
        if entry["kind"]:
            heatmap = {"image_path": image_path, "kind": entry["kind"],
                       "info": entry["heatmap_info"], "original": arrays["original"],
                       "activation": arrays["activation"], "base_image": arrays.get("base_image"),
                       "valid_mask": arrays.get("valid_mask"), "detections": deepcopy(entry.get("detections"))}
        return {"preview": arrays.get("preview"), "heatmap": heatmap, "info": entry["info"]}
