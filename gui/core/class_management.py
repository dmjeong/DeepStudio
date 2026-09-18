"""클래스 변경과 데이터 이동. Qt와 학습 엔진에 의존하지 않는 파일 작업 계층."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

from PIL import Image

from core.project import ProjectManager


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LOSSLESS_MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}


def validate_class_name(name):
    """Windows에서 경로 또는 예약 파일명으로 해석되는 클래스명 거부."""
    return ProjectManager.validate_name(name, kind="클래스")


def _inside(path, parent):
    return path == parent or parent in path.parents


def _plain_path(path):
    """링크가 가리키는 다른 데이터까지 수정하지 않도록 경로 구성요소 검사."""
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
           for p in (path, *path.parents)):
        raise ValueError(f"심볼릭 링크 경로는 클래스 변경을 지원하지 않습니다: {path}")
    return path


def _walk_files(directory):
    directory = _plain_path(directory)
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError(f"데이터 폴더 경로가 파일입니다: {directory}")
    result = []
    for root, dirs, files in os.walk(directory):
        for name in dirs + files:
            _plain_path(Path(root) / name)
        result.extend(Path(root) / name for name in sorted(files))
    return sorted(result)


def _digest(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_dirs(project, kind=None):
    """표준 폴더와 명시적 분할 경로를 중복 없이 포함."""
    if not project.data.root or not os.path.isabs(project.data.root):
        raise ValueError("데이터 루트의 절대 경로가 필요합니다.")
    root = _plain_path(project.data.root)
    if not root.is_dir():
        raise ValueError(f"데이터 루트 폴더가 없습니다: {root}")
    paths = set()
    for split in SPLITS:
        paths.add(root / kind / split if kind else root / split)
        configured = getattr(project.data, f"{split}_dir", "")
        if configured:
            if not os.path.isabs(configured):
                raise ValueError(f"{split} 데이터의 절대 경로가 필요합니다.")
            path = _plain_path(configured)
            if kind:
                # images/train -> labels/train 또는 masks/train
                if path.parent.name == "images":
                    paths.add(path.parent.parent / kind / path.name)
            else:
                paths.add(path)
    return sorted(paths)


def _remap_labels(path, deleted, count, task):
    """정규화 박스/폴리곤 라벨의 형식 검증 후 삭제와 ID 재번호화."""
    text = path.read_text(encoding="utf-8-sig")
    output, removed, remapped = [], 0, 0
    for line_no, line in enumerate(text.splitlines(keepends=True), 1):
        fields = line.split()
        if not fields:
            output.append(line)
            continue
        location = f"{path}:{line_no}"
        valid_size = len(fields) == 5 if task == "detect" else len(fields) >= 7 and len(fields) % 2 == 1
        if not valid_size or not re.fullmatch(r"[0-9]+", fields[0]):
            raise ValueError(f"잘못된 정규화 라벨 형식: {location}")
        class_id = int(fields[0])
        try:
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"잘못된 라벨 좌표: {location}") from exc
        if (class_id >= count or any(not math.isfinite(v) or v < 0 or v > 1 for v in coordinates)
                or (task == "detect" and any(v <= 0 for v in coordinates[2:]))):
            raise ValueError(f"클래스 번호 또는 좌표 범위 오류: {location}")
        if task == "obb":
            from core.paths import ensure_python_path
            ensure_python_path()
            from obb import validate_corners
            validate_corners(coordinates)
        if class_id == deleted:
            removed += 1
            continue
        if class_id > deleted:
            line = re.sub(r"^(\s*)[0-9]+", lambda m: m.group(1) + str(class_id - 1), line, count=1)
            remapped += 1
        output.append(line)
    return "".join(output).encode("utf-8"), removed, remapped


def _remap_mask(path, deleted, count):
    """팔레트 색상이 아닌 원시 픽셀 번호를 변경. RGB/손실 압축 마스크 거부."""
    if path.suffix.lower() not in LOSSLESS_MASK_EXTENSIONS:
        raise ValueError(f"손실 없는 정수 클래스 마스크만 변경 가능합니다: {path}")
    with Image.open(path) as source:
        if source.mode not in ("L", "P", "I", "I;16", "I;16L", "I;16B") or getattr(source, "n_frames", 1) != 1:
            raise ValueError(f"단일 프레임 정수/팔레트 마스크가 필요합니다: {path}")
        source.load()
        if source.mode in ("L", "P"):
            counts = source.histogram()
            if any(counts[count:255]):
                raise ValueError(f"클래스 범위를 벗어난 마스크 값: {path}")
            removed, remapped = counts[deleted], sum(counts[deleted + 1:255])
            result = source.point([255 if value == 255 else 0 if value == deleted else value - (value > deleted)
                                   for value in range(256)])
        else:
            values = list(source.getdata())
            if any(value != 255 and (value < 0 or value >= count) for value in values):
                raise ValueError(f"클래스 범위를 벗어난 마스크 값: {path}")
            removed = sum(value == deleted for value in values)
            remapped = sum(deleted < value < 255 for value in values)
            mapped = [255 if value == 255 else 0 if value == deleted else value - (value > deleted) for value in values]
            # I;16의 putdata는 Pillow 버전에 따라 바이트를 잘못 해석하므로 I 경유.
            result = Image.new("I", source.size)
            result.putdata(mapped)
            result = result.convert(source.mode)
        if source.mode == "P":
            palette = source.getpalette()
            if palette:
                palette = (palette + [0] * 768)[:768]
                palette[deleted * 3:254 * 3] = palette[(deleted + 1) * 3:255 * 3]
                palette[254 * 3:255 * 3] = [0, 0, 0]
                result.putpalette(palette)
            transparency = source.info.get("transparency")
            if isinstance(transparency, int) and transparency != 255:
                result.info["transparency"] = 0 if transparency == deleted else transparency - (transparency > deleted)
            elif isinstance(transparency, bytes):
                result.info["transparency"] = (transparency[:deleted] + transparency[deleted + 1:255]
                                               + (b"\xff" + transparency[255:] if len(transparency) > 255 else b""))
        return result, source.format, removed, remapped


@dataclass(frozen=True)
class DeletePreview:
    class_name: str
    class_index: int
    class_names: tuple
    task: str
    folders: tuple
    labels: tuple
    masks: tuple
    derived: tuple
    fingerprints: tuple
    image_count: int
    removed_annotations: int
    remapped_annotations: int
    removed_pixels: int
    remapped_pixels: int
    changed_files: int


class ClassManager:
    """백업 준비 -> 데이터 변경 -> 프로젝트 원자 저장. 실패 시 역순 복구."""

    @staticmethod
    def preview_delete(project, class_name):
        validate_class_name(class_name)
        names = tuple(project.data.class_names)
        for name in names:
            validate_class_name(name)
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("중복 클래스 이름을 먼저 수정해야 합니다.")
        if class_name not in names:
            raise ValueError("삭제할 클래스가 없습니다.")
        if len(names) <= 1:
            raise ValueError("최소 한 개 클래스가 필요합니다.")
        if project.task not in ("classify", "detect", "segment", "obb"):
            raise ValueError("이 태스크는 클래스 삭제를 지원하지 않습니다.")
        deleted = names.index(class_name)
        semantic = project.task == "segment" and (
            project.training.training_mode == "custom"
            or any(path.suffix.lower() in IMAGE_EXTENSIONS
                   for directory in _split_dirs(project, "masks") for path in _walk_files(directory))
        )
        if semantic and len(names) > 255:
            raise ValueError("255는 무시 픽셀 번호입니다. 255개 이하 클래스에서 삭제를 지원합니다.")
        if semantic and deleted == 0:
            raise ValueError("시맨틱 분할의 배경 클래스(번호 0)는 삭제할 수 없습니다.")
        folders, labels, masks, fingerprints = [], [], [], []
        image_count = removed = remapped = pixels = shifted_pixels = changed = 0
        if project.task == "classify":
            for split_dir in _split_dirs(project):
                path = _plain_path(split_dir / class_name)
                if path.exists():
                    folders.append(str(path))
                    files = _walk_files(path)
                    image_count += sum(p.suffix.lower() in IMAGE_EXTENSIONS for p in files)
                    fingerprints.append((str(path), "folder"))
                    fingerprints.extend((str(p), f"{p.stat().st_size}:{p.stat().st_mtime_ns}") for p in files)
        else:
            for directory in _split_dirs(project, "labels"):
                for path in _walk_files(directory):
                    if path.suffix.lower() != ".txt":
                        continue
                    _, nr, ns = _remap_labels(path, deleted, len(names), project.task)
                    labels.append(str(path))
                    removed += nr
                    remapped += ns
                    changed += bool(nr + ns)
                    fingerprints.append((str(path), _digest(path)))
            if project.task == "segment":
                for directory in _split_dirs(project, "masks"):
                    for path in _walk_files(directory):
                        if path.suffix.lower() not in IMAGE_EXTENSIONS:
                            continue
                        _, _, nr, ns = _remap_mask(path, deleted, len(names))
                        masks.append(str(path))
                        pixels += nr
                        shifted_pixels += ns
                        changed += bool(nr + ns)
                        fingerprints.append((str(path), _digest(path)))
        # 학습 시 다시 생성되는 설정과 라벨 캐시는 이전 번호를 담을 수 있음.
        derived = tuple(str(p) for p in _walk_files(project.data.root)
                        if (p.suffix.lower() == ".cache" or p == Path(project.data.root) / "dataset.yaml")
                        and not any(_inside(p, Path(folder)) for folder in folders))
        fingerprints.extend((p, _digest(Path(p))) for p in derived)
        return DeletePreview(class_name, deleted, names, project.task, tuple(folders), tuple(labels),
                             tuple(masks), derived, tuple(fingerprints), image_count, removed, remapped,
                             pixels, shifted_pixels, changed)

    @staticmethod
    def delete(project, class_name, preview=None):
        current = ClassManager.preview_delete(project, class_name)
        if preview is not None and preview != current:
            raise ValueError("미리보기 이후 데이터가 변경되었습니다. 삭제 내용을 다시 확인해 주세요.")
        preview = current
        root = _plain_path(project.data.root)
        archive_base = root.parent / ".deep_studio_class_archive"
        active_dirs = [root, *(_plain_path(getattr(project.data, f"{s}_dir")) for s in SPLITS
                              if getattr(project.data, f"{s}_dir", ""))]
        if any(_inside(archive_base, active) for active in active_dirs):
            raise ValueError("백업 위치가 활성 데이터 경로에 포함됩니다. 데이터 경로를 분리해 주세요.")
        archive = _plain_path(archive_base / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]))
        archive.mkdir(parents=True)
        old_names = list(project.data.class_names)
        old_count, old_images, old_modified = project.data.num_classes, project.data.image_count, project.modified
        staged, applied, moved, manifest = [], [], [], []
        try:
            # 모든 변환과 원본 백업을 성공시킨 뒤 활성 데이터 변경 시작.
            for path_string in (*preview.labels, *preview.masks):
                path = Path(path_string)
                if path_string in preview.labels:
                    content, removed, remapped = _remap_labels(path, preview.class_index, len(old_names), project.task)
                    if not removed + remapped:
                        continue
                    def writer(target, data=content):
                        Path(target).write_bytes(data)
                else:
                    mask, image_format, removed, remapped = _remap_mask(path, preview.class_index, len(old_names))
                    if not removed + remapped:
                        continue
                    def writer(target, image=mask, fmt=image_format):
                        image.save(target, format=fmt)
                backup = archive / f"original_{len(manifest):06d}{path.suffix}"
                shutil.copy2(path, backup)
                manifest.append({"source": str(path), "backup": backup.name, "kind": "file"})
                fd, temporary = tempfile.mkstemp(prefix=".class_edit_", suffix=".pending", dir=path.parent)
                os.close(fd)
                staged.append((path, Path(temporary), backup))
                writer(temporary)
            for path_string in (*preview.folders, *preview.derived):
                path = Path(path_string)
                backup = archive / f"removed_{len(manifest):06d}{path.suffix}"
                # 폴더 이동은 동일 파일시스템의 원자 rename만 사용. 부분 복사 방지.
                if path.stat().st_dev != archive.stat().st_dev:
                    raise ValueError(f"백업과 데이터가 다른 파일시스템입니다: {path}")
                manifest.append({"source": str(path), "backup": backup.name, "kind": "move"})
            active_path = getattr(project, "_active_filepath", "")
            if active_path and Path(active_path).is_file():
                shutil.copy2(active_path, archive / "project_before.dvproj")
            (archive / "manifest.json").write_text(json.dumps({
                "purpose": "originals_before_class_deletion", "task": project.task, "class_names_before": old_names,
                "deleted_class": class_name, "files": manifest,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            for path, temporary, backup in staged:
                os.replace(temporary, path)
                applied.append((path, backup))
            for record in manifest:
                if record["kind"] == "move":
                    source, backup = Path(record["source"]), archive / record["backup"]
                    os.replace(source, backup)
                    moved.append((source, backup))
            project.data.class_names = [name for name in old_names if name != class_name]
            project.data.num_classes = len(project.data.class_names)
            if project.task == "classify":
                project.data.image_count = max(0, old_images - preview.image_count)
            ProjectManager.save(project)
        except Exception as original:
            project.data.class_names = old_names
            project.data.num_classes, project.data.image_count, project.modified = old_count, old_images, old_modified
            failures = []
            for source, backup in reversed(moved):
                try:
                    os.replace(backup, source)
                except OSError as exc:
                    failures.append(f"{source}: {exc}")
            for source, backup in reversed(applied):
                try:
                    shutil.copy2(backup, source)
                except OSError as exc:
                    failures.append(f"{source}: {exc}")
            if failures:
                raise RuntimeError(f"클래스 삭제 및 일부 복구 실패. 원본 백업: {archive}\n" + "\n".join(failures)) from original
            shutil.rmtree(archive)
            raise
        finally:
            for _, temporary, _ in staged:
                temporary.unlink(missing_ok=True)
        # 저장 이후 실패 가능한 추가 파일 작업을 수행하지 않음. manifest는 복구용 원본 기록.
        return str(archive)

    @staticmethod
    def add(project, class_name):
        validate_class_name(class_name)
        if project.task not in ("classify", "detect", "segment", "obb"):
            raise ValueError("이 태스크는 클래스 추가를 지원하지 않습니다.")
        if class_name.casefold() in {name.casefold() for name in project.data.class_names}:
            raise ValueError("이미 존재하는 클래스 이름입니다.")
        previous = list(project.data.class_names)
        previous_count, previous_modified = project.data.num_classes, project.modified
        created = []
        try:
            if project.task == "classify":
                for split in _split_dirs(project):
                    target = _plain_path(split / class_name)
                    if target.exists():
                        raise ValueError(f"같은 이름의 데이터 폴더가 이미 존재합니다: {target}")
                    target.mkdir(parents=True)
                    created.append(target)
            project.data.class_names = previous + [class_name]
            project.data.num_classes = len(project.data.class_names)
            ProjectManager.save(project)
        except Exception:
            project.data.class_names = previous
            project.data.num_classes, project.modified = previous_count, previous_modified
            for path in reversed(created):
                path.rmdir()
            raise


def move_image_with_sidecars(image_path, destination_dir, sidecars, reserve_stem=False, reserve_dirs=()):
    """이미지와 정답의 모든 이름을 함께 확보하고 실패한 묶음은 원위치로 복구."""
    source = _plain_path(image_path)
    destination_dir = _plain_path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    sidecars = [(_plain_path(path), _plain_path(directory), suffix) for path, directory, suffix in sidecars]
    if not source.is_file() or any(not path.is_file() for path, _, _ in sidecars):
        raise ValueError("이동할 이미지 또는 정답 파일이 없습니다.")
    if reserve_stem and sidecars and any(p != source and p.stem.casefold() == source.stem.casefold() and p.suffix.lower() in IMAGE_EXTENSIONS
                                         for p in source.parent.iterdir()):
        raise ValueError("같은 이름의 여러 이미지가 하나의 정답을 공유합니다. 파일명을 먼저 구분해 주세요.")
    counter = 0
    while True:
        stem = source.stem + (f"_{counter}" if counter else "")
        destination = destination_dir / (stem + source.suffix)
        targets = [(path, directory / (stem + suffix)) for path, directory, suffix in sidecars]
        destinations = [destination, *(target for _, target in targets)]
        if len({str(path).casefold() for path in destinations}) != len(destinations):
            raise ValueError("동일한 경로를 사용하는 이미지/정답 파일이 중복됩니다.")
        occupied = any(path.exists() for path in destinations)
        # a.jpg와 a.png도 같은 정답 a.txt를 공유하므로 파일명이 아닌 stem 충돌 검사.
        if reserve_stem:
            directories = {destination_dir, *(directory for _, directory, _ in sidecars),
                           *(_plain_path(directory) for directory in reserve_dirs)}
            occupied = occupied or any(any(p.stem.casefold() == stem.casefold() for p in directory.iterdir())
                                       for directory in directories if directory.is_dir())
        if not occupied:
            break
        counter += 1
    completed = []
    try:
        for origin, target in [*targets, (source, destination)]:
            target.parent.mkdir(parents=True, exist_ok=True)
            # 'xb'는 이름 확인 후 외부 파일이 생성되어도 덮어쓰지 않음.
            with origin.open("rb") as reader, target.open("xb") as writer:
                completed.append((origin, target))
                shutil.copyfileobj(reader, writer)
            shutil.copystat(origin, target)
        for origin, _ in completed:
            origin.unlink()
    except Exception as original:
        failures = []
        for origin, target in reversed(completed):
            try:
                if not origin.exists():
                    shutil.copy2(target, origin)
                target.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError("이미지 이동 복구 실패: " + "; ".join(failures)) from original
        raise
    return str(destination)
