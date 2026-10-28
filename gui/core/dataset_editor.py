"""Qt와 웹이 공유하는 데이터 탐색 및 백업 가능한 이미지/정답 편집."""

import copy
import io
import shutil
import threading
import uuid
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
from core.annotation_journal import guarded_dataset, write_journal
from core.class_management import (
    IMAGE_EXTENSIONS,
    SPLITS,
    _plain_path,
    _walk_files,
    validate_class_name,
)
from core.project import ProjectManager
from PIL import Image


def split_root(project, split):
    if split not in SPLITS:
        raise ValueError("데이터 분할 오류")
    return _plain_path(getattr(project.data, f"{split}_dir"))


def sidecars(project, image, split):
    """하위 폴더를 유지해 이미지와 대응하는 모든 정답을 찾는다."""
    root = split_root(project, split)
    relative = Path(image).relative_to(root)
    base = (
        root.parent.parent if root.parent.name == "images" else Path(project.data.root)
    )
    result = []
    for kind in (
        ("labels", "masks")
        if project.task == "segment"
        else ("labels",)
        if project.task in ("detect", "obb")
        else ()
    ):
        folder = _plain_path(base / kind / split / relative.parent)
        if folder.is_dir():
            result.extend(
                (kind, path)
                for path in folder.iterdir()
                if path.stem == relative.stem
                and (
                    path.suffix.lower() == ".txt"
                    if kind == "labels"
                    else path.suffix.lower() in IMAGE_EXTENSIONS
                )
            )
    return result


def annotation_revision(project, image, split):
    """A semantic label/class revision, checked while holding the edit lock."""
    import hashlib

    from core.annotation_tools import fingerprint

    root = split_root(project, split)
    image = root / Path(image).resolve().relative_to(root.resolve())
    values = []
    for kind, path in sorted(
        sidecars(project, image, split), key=lambda row: str(row[1])
    ):
        values.append([kind, str(path), hashlib.sha256(path.read_bytes()).hexdigest()])
    stat=Path(image).stat()
    return fingerprint({"classes": project.data.class_names, "labels": values,'image':[stat.st_size,stat.st_mtime_ns]})


def read_annotations(path, task, count):
    result = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        values = line.split()
        try:
            class_id, coords = int(values[0]), [float(v) for v in values[1:]]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"정답 형식 오류: {path}") from exc
        valid = (
            len(coords) == 4
            if task == "detect"
            else len(coords) >= 6 and len(coords) % 2 == 0
        )
        if (
            not valid
            or not 0 <= class_id < count
            or any(not np.isfinite(v) or not 0 <= v <= 1 for v in coords)
        ):
            raise ValueError(f"정답 클래스 또는 좌표 범위 오류: {path}")
        if task == "detect" and (coords[2] <= 0 or coords[3] <= 0):
            raise ValueError(f"박스 너비와 높이는 0보다 커야 함: {path}")
        if task == "obb":
            from core.paths import ensure_python_path

            ensure_python_path()
            from obb import validate_corners

            validate_corners(coords)
        result.append({"class_id": class_id, "coordinates": coords})
    return result


# 마스크 영역 집계 캐시: (경로, 수정 시각, 크기, 클래스 수) → (픽셀 클래스 번호들, {번호: 영역 수}).
# 데이터셋 화면은 이미지 추가·이동·저장 때마다 전체를 다시 탐색하므로, 바뀌지 않은 마스크는
# 다시 디코딩하지 않는다. 항목이 작아(튜플 두 개) 2만 장까지 유지한다.
_REGION_CACHE = OrderedDict()
_REGION_CACHE_LIMIT = 20000
_REGION_LOCK = threading.Lock()


def mask_regions(path, class_count):
    """클래스 번호 마스크의 (픽셀이 있는 클래스 번호, 클래스별 영역 수).

    분할의 클래스는 이미지 전체가 아니라 이미지 안의 영역(blob)에 붙는다. 영역은 같은 클래스
    픽셀의 8-연결 성분이다. 배경(0)과 학습 제외(255)는 영역으로 세지 않는다.
    """
    import cv2

    stat = Path(path).stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, class_count)
    with _REGION_LOCK:
        cached = _REGION_CACHE.get(key)
        if cached is not None:
            _REGION_CACHE.move_to_end(key)
            return cached
    with Image.open(path) as mask:
        pixels = np.asarray(mask)
    if pixels.ndim != 2:
        raise ValueError("정수 클래스 마스크 필요")
    if pixels.dtype == np.uint8:  # 정렬하는 np.unique(4096²에서 약 0.1초) 대신 히스토그램
        ids = {int(i) for i in np.flatnonzero(cv2.calcHist([pixels], [0], None, [256], [0, 256]).ravel())}
    else:
        ids = {int(i) for i in np.unique(pixels)}
    ids.discard(255)
    if any(i < 0 or i >= class_count for i in ids):
        raise ValueError("마스크 클래스 범위 오류")
    regions = {}
    for class_id in sorted(ids - {0}):
        member = (pixels == class_id).astype(np.uint8)
        regions[class_id] = int(cv2.connectedComponents(member, connectivity=8)[0]) - 1
    result = (tuple(sorted(ids)), regions)
    with _REGION_LOCK:
        _REGION_CACHE[key] = result
        while len(_REGION_CACHE) > _REGION_CACHE_LIMIT:
            _REGION_CACHE.popitem(last=False)
    return result


@guarded_dataset
def scan_dataset(project, cancelled=lambda: False):
    """정답 클래스별 이미지 수를 계산. 손상 정답은 해당 이미지에 오류로 표시.

    detect/obb/segment는 이미지마다 클래스별 영역 수(regions)도 기록한다. 박스·폴리곤은 정답 한 줄,
    마스크는 같은 클래스 픽셀의 연결 성분 하나가 영역 하나다. 분할의 배경(0)은 영역이 아니다.
    """
    records, splits = [], []
    names = project.data.class_names
    for split in SPLITS:
        root = split_root(project, split)
        counts = Counter()
        region_totals = Counter()
        unlabeled = empty = 0
        images = [p for p in _walk_files(root) if p.suffix.lower() in IMAGE_EXTENSIONS]
        for path in images:
            if cancelled():
                raise InterruptedError("데이터 탐색 취소")
            classes, error, annotated = set(), "", False
            regions = Counter()
            try:
                if project.task in {"classify", "anomaly"}:
                    parts = path.relative_to(root).parts
                    classes = {parts[0]} if len(parts) > 1 else set()
                    annotated = bool(classes)
                else:
                    for kind, label in sidecars(project, path, split):
                        annotated = True
                        if kind == "labels":
                            rows = read_annotations(label, project.task, len(names))
                            classes.update(names[row["class_id"]] for row in rows)
                            regions.update(row["class_id"] for row in rows)
                        else:
                            ids, mask_counts = mask_regions(label, len(names))
                            classes.update(names[i] for i in ids)
                            regions.update(mask_counts)
            except (OSError, ValueError) as exc:
                error = str(exc)
            counts.update(classes)
            if project.task == "segment":
                regions.pop(0, None)  # 폴리곤 정답의 배경 번호도 영역이 아니다
            region_names = {names[i]: regions[i] for i in sorted(regions) if regions[i]}
            region_totals.update(region_names)
            if project.task in {"detect", "obb", "segment"} and not error:
                unlabeled += not annotated
                empty += annotated and not region_names
            stat = path.stat()
            records.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "split": split,
                    "classes": sorted(classes),
                    "regions": region_names,
                    "annotated": annotated,
                    "error": error,
                    "revision": f"{stat.st_mtime_ns}-{stat.st_size}",
                }
            )
        splits.append(
            {
                "split": split,
                "path": str(root),
                "count": len(images),
                "classes": {
                    name: counts[name] for name in dict.fromkeys([*names, *counts])
                },
                "regions": {name: region_totals[name] for name in names if region_totals[name]},
                "unlabeled": unlabeled,
                "empty": empty,
            }
        )
    return {
        "splits": splits,
        "class_names": names,
        "root": project.data.root,
        "images": records,
    }


class EditTransaction:
    """변경 전 파일을 모두 기록하고 실패 시 원상 복구. 성공 백업은 보존."""

    def __init__(self, project, action):
        self.project = project
        self.before = copy.deepcopy(project)
        self.backup = (
            Path(project.project_dir).parent
            / ".deep_studio_dataset_archive"
            / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        )
        data_roots = [
            Path(project.data.root).resolve(),
            *(split_root(project, sp) for sp in SPLITS),
        ]
        if any(self.backup.resolve().is_relative_to(root) for root in data_roots):
            raise ValueError(
                "백업 위치가 데이터 루트 안에 포함됨. 프로젝트와 데이터 루트를 별도로 지정 필요"
            )
        self.backup.mkdir(parents=True)
        self.paths = {}
        self.created_dirs = set()
        self.action = action
        self.save(Path(ProjectManager.get_active_filepath(project)))

    def save(self, path):
        path = _plain_path(path)
        if path not in self.paths:
            target = self.backup / str(len(self.paths)) if path.exists() else None
            if target is not None:
                shutil.copy2(path, target)
            self.paths[path] = target
            self._manifest("pending")

    def _manifest(self, status):
        write_journal(self, status)

    def mkdir(self, directory):
        missing = []
        parent = Path(directory)
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        directory.mkdir(parents=True, exist_ok=True)
        self.created_dirs.update(missing)

    def write(self, path, data):
        from core.annotation_tools import atomic_write

        path = _plain_path(path)
        self.save(path)
        self.mkdir(path.parent)
        atomic_write(path, data)

    def copy(self, source, target, move=False):
        source, target = _plain_path(source), _plain_path(target)
        if source == target:
            return
        if target.exists():
            raise FileExistsError(f"같은 이름의 파일 존재: {target}")
        self.save(target)
        if move:
            self.save(source)
        self.mkdir(target.parent)
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        if move:
            source.unlink()

    def remove(self, path):
        self.save(path)
        Path(path).unlink()

    def finish(self):
        # 학습 시 재생성되는 캐시에서 이전 파일/클래스가 남지 않도록 제거.
        for path in _walk_files(self.project.data.root):
            if path.suffix == ".cache" or path.name == "dataset.yaml":
                self.remove(path)
        ProjectManager.save(self.project)
        self._manifest("completed")

    def rollback(self):
        failures = []
        for path, backup in reversed(list(self.paths.items())):
            try:
                if backup:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, path)
                else:
                    path.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(str(exc))
        for directory in sorted(
            self.created_dirs, key=lambda p: len(p.parts), reverse=True
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
        self.project.__dict__.update(self.before.__dict__)
        self._manifest("recovery_required" if failures else "rolled_back")
        if failures:
            raise RuntimeError(
                f"일부 파일 복구 필요: {self.backup}: " + "; ".join(failures)
            )


def annotation_view(project, path, split):
    image = _plain_path(path)
    image.relative_to(split_root(project, split))
    if not image.is_file():
        raise FileNotFoundError("이미지 없음")
    with Image.open(image) as opened:
        width, height = opened.size
    labels = sidecars(project, image, split)
    return {
        "path": str(image),
        "width": width,
        "height": height,
        "has_mask": any(kind == "masks" for kind, _ in labels),
        "annotations": [
            row
            for kind, p in labels
            if kind == "labels"
            for row in read_annotations(p, project.task, len(project.data.class_names))
        ],
    }


@guarded_dataset
def edit_dataset(
    project,
    action,
    paths=(),
    split="train",
    target_split="train",
    class_name="",
    source_class="",
    new_name="",
    annotations=None,
    cancelled=lambda: False,
    expected_revision=None,
):
    """이미지 묶음 편집. 오류/취소는 전체 묶음 복구, 원본 삭제는 백업 후 수행."""
    if action not in {
        "import",
        "move",
        "reclass",
        "delete",
        "rename_class",
        "annotations",
        "mask_annotations",
    }:
        raise ValueError("데이터 편집 작업 오류")
    root, target_root = split_root(project, split), split_root(project, target_split)
    names = project.data.class_names
    sources = [_plain_path(path) for path in paths]
    if len(set(sources)) != len(sources):
        raise ValueError("중복 이미지 선택")
    for path in sources:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"이미지 파일 필요: {path}")
        if action != "import":
            path.relative_to(root)
    if action in {"reclass", "import"} and (
        project.task in {"classify", "anomaly"} or action == "reclass"
    ) and class_name not in names:
        raise ValueError("대상 클래스 선택 필요")
    if action == "rename_class":
        validate_class_name(new_name)
        if source_class not in names or any(
            n.casefold() == new_name.casefold() for n in names
        ):
            raise ValueError("기존 클래스 또는 중복 이름 확인 필요")
    if (
        action == "reclass"
        and project.task in {"detect", "segment", "obb"}
        and source_class not in names
    ):
        raise ValueError("변경할 원래 클래스 선택 필요")
    if (
        action == "reclass"
        and project.task in {"detect", "segment", "obb"}
        and source_class == class_name
    ):
        raise ValueError("원래 클래스와 대상 클래스가 같습니다.")
    if (
        action == "reclass"
        and project.task == "segment"
        and (names.index(source_class) == 0 or names.index(class_name) == 0)
    ):
        raise ValueError(
            "세그멘테이션 클래스 ID 0은 배경으로 예약되어 있습니다. 전경 클래스만 변경하세요."
        )
    if action == "annotations" and (
        len(sources) != 1 or project.task not in {"detect", "segment", "obb"}
    ):
        raise ValueError("탐지/분할 이미지 한 장 선택 필요")
    if expected_revision is not None and action in {"annotations", "mask_annotations"}:
        if (
            len(sources) != 1
            or annotation_revision(project, sources[0], split) != expected_revision
        ):
            raise ValueError(
                "편집을 시작한 뒤 정답 또는 클래스가 다른 곳에서 변경됐습니다. 초안을 보존하고 이미지를 다시 여세요."
            )
    mask_document = None
    if action == "mask_annotations":
        if len(sources) != 1 or project.task != "segment":
            raise ValueError("분할 이미지 한 장 선택 필요")
        from core.mask_annotations import MaskDocument

        with Image.open(sources[0]) as image:
            mask_document = MaskDocument.from_payload(
                annotations, *image.size, len(names)
            )
    tx = EditTransaction(project, action)
    changed = 0
    try:
        if action == "rename_class":
            if project.task in {"classify", "anomaly"}:
                seen = set()
                for sp in SPLITS:
                    directory = split_root(project, sp)
                    for path in _walk_files(directory / source_class):
                        if path not in seen:
                            seen.add(path)
                            tx.copy(
                                path,
                                directory
                                / new_name
                                / path.relative_to(directory / source_class),
                                move=True,
                            )
                            changed += 1
            project.data.class_names[names.index(source_class)] = new_name
        for path in sources:
            if cancelled():
                raise InterruptedError("데이터 변경 취소")
            related = sidecars(project, path, split) if action != "import" else []
            reclass_changed = False
            if action == "delete":
                for _, label in related:
                    tx.remove(label)
                tx.remove(path)
            elif (
                action in {"move", "import"}
                or action == "reclass"
                and project.task in {"classify", "anomaly"}
            ):
                if action == "import":
                    with Image.open(path) as check:
                        check.verify()
                    dest_dir = (
                        target_root / class_name
                        if project.task in {"classify", "anomaly"}
                        else target_root
                    )
                    if project.task in {"detect", "segment", "obb"}:
                        for kind in (
                            ("labels", "masks")
                            if project.task == "segment"
                            else ("labels",)
                        ):
                            candidates = (
                                [path.with_suffix(".txt")] if kind == "labels" else []
                            )
                            # 폴더 가져오기는 images/split 하위의 대응 labels/masks를 함께 복사.
                            for parent in path.parents:
                                if parent.name == "images":
                                    candidate = (
                                        parent.parent / kind / path.relative_to(parent)
                                    )
                                    candidates += (
                                        [candidate.with_suffix(".txt")]
                                        if kind == "labels"
                                        else [
                                            candidate.with_suffix(s)
                                            for s in (".png", ".tif", ".bmp")
                                        ]
                                    )
                                    break
                            related.extend(
                                (kind, p)
                                for p in dict.fromkeys(candidates)
                                if p.is_file()
                            )
                else:
                    relative = path.relative_to(root)
                    dest_dir = (
                        target_root / relative.parent
                        if action == "move"
                        else root / class_name / Path(*relative.parts[1:-1])
                    )
                targets = [dest_dir / path.name]
                destinations = []
                for kind, label in related:
                    base = (
                        target_root.parent.parent
                        if target_root.parent.name == "images"
                        else Path(project.data.root)
                    )
                    folder = (
                        base
                        / kind
                        / target_split
                        / targets[0].relative_to(target_root).parent
                    )
                    destinations.append(folder / (path.stem + label.suffix))
                # 같은 stem은 정답을 공유하므로 확장자가 달라도 충돌 거부.
                if project.task in {"detect", "segment", "obb"} and dest_dir.is_dir():
                    if any(
                        p.stem.casefold() == path.stem.casefold() and p != path
                        for p in dest_dir.iterdir()
                    ):
                        raise FileExistsError(f"같은 이름의 정답 충돌: {path.stem}")
                for (kind, source), target in zip(related, destinations):
                    if kind == "labels":
                        read_annotations(source, project.task, len(names))
                    tx.copy(source, target, move=action != "import")
                tx.copy(path, targets[0], move=action != "import")
            elif action == "reclass":
                old, new = names.index(source_class), names.index(class_name)
                for kind, label in related:
                    if kind == "labels":
                        rows = read_annotations(label, project.task, len(names))
                        if not any(row["class_id"] == old for row in rows):
                            continue
                        for row in rows:
                            if row["class_id"] == old:
                                row["class_id"] = new
                        tx.write(label, encode_annotations(rows))
                        reclass_changed = True
                    else:
                        with Image.open(label) as mask:
                            pixels = np.array(mask)
                            if pixels.ndim != 2 or pixels.dtype.kind not in "ui":
                                raise ValueError("정수 클래스 마스크 필요")
                            if not np.any(pixels == old):
                                continue
                            pixels[pixels == old] = new
                            out = Image.fromarray(pixels)
                            stream = io.BytesIO()
                            out.save(stream, format=mask.format)
                            tx.write(label, stream.getvalue())
                            reclass_changed = True
            elif action == "mask_annotations":
                base = (
                    root.parent.parent
                    if root.parent.name == "images"
                    else Path(project.data.root)
                )
                mask_path = (
                    base / "masks" / split / path.relative_to(root).with_suffix(".png")
                )
                tx.write(mask_path, mask_document.encode())
                # One authoritative target per image. The transaction retains
                # prior raster/vector files for recovery if any save step fails.
                for kind, label in related:
                    if label != mask_path:
                        tx.remove(label)
            elif action == "annotations":
                if any(kind == "masks" for kind, _ in related):
                    raise ValueError(
                        "픽셀 마스크는 클래스 변경으로 편집. 박스/폴리곤과 혼합 저장 불가"
                    )
                base = (
                    root.parent.parent
                    if root.parent.name == "images"
                    else Path(project.data.root)
                )
                label = (
                    base / "labels" / split / path.relative_to(root).with_suffix(".txt")
                )
                tx.write(label, encode_annotations(annotations or []))
                read_annotations(label, project.task, len(names))
            if action == "reclass" and project.task in {"detect", "segment", "obb"}:
                changed += int(reclass_changed)
            else:
                changed += 1
        if (
            action == "reclass"
            and project.task in {"detect", "segment", "obb"}
            and changed == 0
        ):
            raise ValueError(
                f"선택한 이미지에 '{source_class}' 라벨이 없습니다. 이미지 전체에 클래스를 붙이는 작업은 "
                "탐지/분할 학습 정답이 되지 않습니다. 이미지를 열고 객체 박스나 영역을 표시한 뒤 저장하세요."
            )
        tx.finish()
        if action == "rename_class" and project.task in {"classify", "anomaly"}:
            # ImageFolder가 이전 빈 클래스 폴더를 별도 클래스로 읽지 않도록 정리.
            for sp in SPLITS:
                old_root = split_root(project, sp) / source_class
                if old_root.is_dir():
                    for directory in sorted(
                        (p for p in old_root.rglob("*") if p.is_dir()),
                        key=lambda p: len(p.parts),
                        reverse=True,
                    ):
                        directory.rmdir()
                    old_root.rmdir()
    except BaseException:
        tx.rollback()
        raise
    result = {"changed": changed, "backup_path": str(tx.backup)}
    if action in {"annotations", "mask_annotations"}:
        result["annotation_revision"] = annotation_revision(project, sources[0], split)
    return result


def encode_annotations(rows):
    return "".join(
        f"{row['class_id']} "
        + " ".join(format(float(v), ".8g") for v in row["coordinates"])
        + "\n"
        for row in rows
    ).encode("utf-8")
