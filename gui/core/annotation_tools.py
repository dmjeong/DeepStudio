"""Local annotation persistence and grid geometry, independent of Qt/models."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_write(path, data):
    """A reader sees either the previous complete file or the new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".dvs-" + hashlib.sha256(path.name.encode()).hexdigest()[:8] + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


class AnnotationDraftStore:
    """Drafts never share a directory/extension with training annotations."""

    def __init__(self, root=None):
        if root is None:
            folder = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
            folder = Path(folder) if folder else Path.home() / ".local" / "share"
            root = folder / "DeepVisionStudio" / "annotation-drafts"
        self.root = Path(root)

    def path(self, image, task):
        key = hashlib.sha256(
            (str(Path(image).resolve()) + "\0" + task).encode()
        ).hexdigest()
        return self.root / (key + ".json")

    @staticmethod
    def image_revision(image):
        stat = Path(image).stat()
        return [stat.st_size, stat.st_mtime_ns]

    def save(
        self,
        image,
        task,
        baseline,
        payload,
        *,
        names,
        pending=None,
        state="draft",
        reviewed=False,
        label_revision=None,
    ):
        value = {
            "schema_version": 1,
            "image": str(Path(image).resolve()),
            "task": task,
            "image_revision": self.image_revision(image),
            "baseline": baseline,
            "names": list(names),
            "payload": payload,
            "pending": pending or {},
            "state": state,
            "reviewed": bool(reviewed),
            "label_revision": label_revision,
        }
        atomic_write(
            self.path(image, task),
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            ),
        )

    def load(self, image, task, baseline, names):
        path = self.path(image, task)
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("image") != str(Path(image).resolve())
            or value.get("task") != task
            or value.get("image_revision") != self.image_revision(image)
            or value.get("baseline") != baseline
            or value.get("names") != list(names)
        ):
            raise ValueError(
                "초안 이후 원본 이미지·정답·클래스가 변경되어 자동 복원할 수 없습니다."
            )
        return value

    def remove(self, image, task):
        self.path(image, task).unlink(missing_ok=True)


def grid_boxes(
    seed, columns=8, rows=16, *, column_end=None, row_end=None, far_corner=None
):
    """Generate boxes/corners in normalized image space using 3/4 anchors.

    seed=(cx,cy,w,h). Anchor endpoints are *chip centers*. Affine grids use
    three centers; four centers fit a projective transform. Out-of-frame
    geometry is rejected rather than silently clipping teaching data.
    """
    if (
        type(columns) is not int
        or type(rows) is not int
        or not 1 <= columns <= 128
        or not 1 <= rows <= 128
        or columns * rows > 4096
    ):
        raise ValueError("격자는 축당 1~128개, 전체 4096개 이하여야 합니다.")
    values = np.asarray(seed, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or min(values[2:]) <= 0:
        raise ValueError("기준 박스의 위치와 크기를 확인하세요.")
    x, y, w, h = values
    origin = np.array([x, y])
    column_end = np.asarray(
        column_end if column_end is not None else [x + (columns - 1) * w * 1.15, y],
        dtype=float,
    )
    row_end = np.asarray(
        row_end if row_end is not None else [x, y + (rows - 1) * h * 1.15], dtype=float
    )
    if (
        column_end.shape != (2,)
        or row_end.shape != (2,)
        or not np.isfinite([*column_end, *row_end]).all()
    ):
        raise ValueError("격자 기준점이 유효하지 않습니다.")
    u = (column_end - origin) / max(1, columns - 1)
    v = (row_end - origin) / max(1, rows - 1)
    transform = None
    if far_corner is not None:
        if columns < 2 or rows < 2:
            raise ValueError("원근 격자는 가로·세로 2개 이상이어야 합니다.")
        far = np.asarray(far_corner, dtype=float)
        if far.shape != (2,) or not np.isfinite(far).all():
            raise ValueError("원근 기준점이 유효하지 않습니다.")
        src = [
            (0.0, 0.0),
            (columns - 1.0, 0.0),
            (0.0, rows - 1.0),
            (columns - 1.0, rows - 1.0),
        ]
        matrix, target = [], []
        for (a, b), (c, d) in zip(src, [origin, column_end, row_end, far]):
            matrix.extend(
                [[a, b, 1, 0, 0, 0, -a * c, -b * c], [0, 0, 0, a, b, 1, -a * d, -b * d]]
            )
            target.extend([c, d])
        try:
            transform = np.r_[np.linalg.solve(matrix, target), 1.0].reshape(3, 3)
        except np.linalg.LinAlgError as exc:
            raise ValueError("원근 기준점이 겹치거나 일직선입니다.") from exc
    if columns > 1 and rows > 1 and abs(np.linalg.det(np.stack([u, v]))) < 1e-10:
        raise ValueError("가로·세로 기준점이 같은 방향으로 겹칩니다.")
    # Preserve the seed's width/height while rotating/shearing with the axes.
    du = (
        u / max(np.linalg.norm(u), 1e-12) * w / 2
        if columns > 1
        else np.array([w / 2, 0])
    )
    dv = v / max(np.linalg.norm(v), 1e-12) * h / 2 if rows > 1 else np.array([0, h / 2])
    result = []
    for r in range(rows):
        for c in range(columns):
            center = origin + c * u + r * v
            if transform is not None:

                def project(a, b):
                    p = transform @ np.array([a, b, 1.0])
                    if abs(p[2]) < 1e-10:
                        raise ValueError("원근 격자가 무한대 경계를 통과합니다.")
                    return p[:2] / p[2]

                center = project(c, r)
                du = (
                    (project(c + 0.01, r) - project(c - 0.01, r))
                    / 0.02
                    * (w / max(np.linalg.norm(u), 1e-12))
                    / 2
                )
                dv = (
                    (project(c, r + 0.01) - project(c, r - 0.01))
                    / 0.02
                    * (h / max(np.linalg.norm(v), 1e-12))
                    / 2
                )
            points = np.array(
                [center - du - dv, center + du - dv, center + du + dv, center - du + dv]
            )
            low, high = points.min(axis=0), points.max(axis=0)
            valid = bool(
                np.isfinite(points).all()
                and (low >= -1e-8).all()
                and (high <= 1 + 1e-8).all()
            )
            if valid:
                points = np.clip(points, 0, 1)
                low, high = points.min(axis=0), points.max(axis=0)
            result.append(
                {
                    "row": r + 1,
                    "column": c + 1,
                    "points": points.reshape(-1).tolist(),
                    "coordinates": [*((low + high) / 2), *(high - low)],
                    "valid": valid,
                }
            )
    return result
