"""Offline SAM2 image assistance. Models/embeddings are reused, never uploaded."""

from pathlib import Path

import numpy as np
from PIL import Image


class LocalSamAssist:
    def __init__(self, model_id="sam2_hiera_tiny", *, device="auto"):
        self.model_id, self.device = model_id, device
        self.predictor = None
        self.image_revision = None

    def _load(self):
        if self.predictor is not None:
            return
        from core.paths import ensure_python_path

        ensure_python_path()
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2_assets import load_sam2_pretrained
        except ImportError as exc:
            raise RuntimeError(
                "로컬 SAM2 런타임이 없습니다. 기본 모델이 포함된 build.bat 설치 환경을 확인하세요."
            ) from exc
        device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "auto"
            else self.device
        )
        # load_sam2_pretrained resolves bundled assets; it never downloads them.
        model = load_sam2_pretrained(self.model_id, device=device)
        self.predictor = SAM2ImagePredictor(model)

    def predict(self, image_path, *, points=(), box=None):
        self._load()
        path = Path(image_path)
        stat = path.stat()
        revision = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        import torch

        with torch.inference_mode():
            if self.image_revision != revision:
                with Image.open(path) as image:
                    pixels = np.array(image.convert("RGB"), copy=True)
                self.predictor.set_image(pixels)
                self.image_revision = revision
            kwargs = {"multimask_output": True}
            if points:
                kwargs["point_coords"] = np.asarray(
                    [p[0] for p in points], dtype=np.float32
                )
                kwargs["point_labels"] = np.asarray(
                    [p[1] for p in points], dtype=np.int32
                )
            if box is not None:
                box = np.asarray(box, dtype=np.float32)
                if box.shape != (4,) or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("AI 박스의 가로·세로 크기가 필요합니다.")
                kwargs["box"] = box
            if not points and box is None:
                raise ValueError("포함점 또는 박스를 지정하세요.")
            masks, scores, _ = self.predictor.predict(**kwargs)
        masks, scores = np.asarray(masks), np.asarray(scores)
        if (
            masks.ndim != 3
            or len(masks) != len(scores)
            or not np.isfinite(scores).all()
        ):
            raise ValueError("SAM2 제안의 출력 형식이 올바르지 않습니다.")
        result = masks[int(np.argmax(scores))].astype(bool)
        with Image.open(path) as image:
            if result.shape != (image.height, image.width):
                raise ValueError("SAM2 제안의 크기가 원본 이미지와 다릅니다.")
        return result
