"""Recover explicit image/annotation links without guessing from size or file order."""

from pathlib import Path

from core.annotation_journal import guarded_dataset
from core.class_management import IMAGE_EXTENSIONS, SPLITS, _plain_path, _walk_files
from core.dataset_editor import EditTransaction, sidecars, split_root
from core.project import ProjectManager


def mask_root(project, split):
    root = split_root(project, split)
    base = root.parent.parent if root.parent.name == "images" else Path(project.data.root)
    return _plain_path(base / "masks" / split)


def orphan_masks(project, split=None):
    """Read-only inventory, also usable while scan_dataset already holds the lock."""
    if project.task != "segment":
        return []
    result = []
    for part in (SPLITS if split is None else (split,)):
        images, masks = split_root(project, part), mask_root(project, part)
        keys = {p.relative_to(images).with_suffix("") for p in _walk_files(images)
                if p.suffix.lower() in IMAGE_EXTENSIONS}
        for path in _walk_files(masks):
            relative = path.relative_to(masks)
            if (path.suffix.lower() in IMAGE_EXTENSIONS
                    and not any(p.startswith(".") for p in relative.parts)
                    and relative.with_suffix("") not in keys):
                result.append({"path": str(path), "name": str(relative), "split": part})
    return result


def _image(project, path, split):
    root, path = split_root(project, split), _plain_path(path)
    path.relative_to(root)
    path.resolve().relative_to(root.resolve())
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"이미지 파일 필요: {path}")
    matches = [p for p in path.parent.iterdir() if p.is_file()
               and p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.casefold() == path.stem.casefold()]
    if len(matches) != 1:
        raise ValueError(f"이미지 이름 중복: {path.stem}")
    return path


def _check_targets(moves):
    """Reject collisions across extensions/case, including unlinked annotations."""
    for source, target in moves:
        if not target.parent.exists():
            continue
        for other in target.parent.iterdir():
            if other.stem.casefold() == target.stem.casefold() and other != source:
                raise FileExistsError(f"같은 이름의 파일 존재: {other}")


def _apply_moves(project, action, moves):
    tx = EditTransaction(project, action)
    try:
        for source, target in moves:
            # Windows and default macOS volumes treat case-only names as one
            # path. A two-step rename lets the journal restore even that case.
            if str(source) == str(target):
                continue
            if target.exists() and source.samefile(target):
                temporary = source.with_name(".dvs-rename-" + tx.backup.name + source.suffix)
                tx.copy(source, temporary, move=True)
                tx.copy(temporary, target, move=True)
            else:
                tx.copy(source, target, move=True)
        tx.finish()
    except BaseException:
        tx.rollback()
        raise
    return {"changed": 1, "backup_path": str(tx.backup), "path": str(moves[-1][1])}


@guarded_dataset
def rename_image(project, path, new_stem, split="train"):
    """Rename the image and its raster/vector labels as a single transaction."""
    image = _image(project, path, split)
    ProjectManager.validate_name(new_stem, kind="이미지 이름")
    target = image.with_stem(new_stem)
    if str(target) == str(image):
        raise ValueError("새 이름이 현재 이름과 같습니다.")
    related = sidecars(project, image, split)
    # Reserve the destination even if this image currently has no annotation.
    occupied = sidecars(project, target, split)
    if any(path not in [p for _, p in related] for _, path in occupied):
        raise FileExistsError("새 이름에 이미 정답 파일이 있습니다.")
    moves = [(path, path.with_stem(new_stem)) for _, path in related]
    moves.append((image, target))
    _check_targets(moves)
    return _apply_moves(project, "rename_image", moves)


def validate_mask_link(project, image_path, mask_path, split):
    """Validate both for preview and again under the write lock before moving."""
    from core.paths import ensure_python_path

    ensure_python_path()
    from spatial_data import load_semantic_mask

    if project.task != "segment":
        raise ValueError("정답 마스크 연결은 segmentation에서만 사용합니다.")
    image, mask = _image(project, image_path, split), _plain_path(mask_path)
    root = mask_root(project, split)
    relative = mask.relative_to(root)
    mask.resolve().relative_to(root.resolve())
    if not any(Path(row["path"]) == mask for row in orphan_masks(project, split)):
        raise ValueError("선택한 마스크는 이미 다른 이미지에 연결되어 있거나 존재하지 않습니다.")
    if sidecars(project, image, split):
        raise FileExistsError("새 이미지에 이미 정답이 있습니다. 기존 정답은 덮어쓰지 않습니다.")
    previous = split_root(project, split) / relative
    related = sidecars(project, previous, split)
    if sum(kind == "masks" for kind, _ in related) != 1:
        raise ValueError("이전 이름에 마스크가 여러 개 있습니다. 사용할 정답을 먼저 확인하세요.")
    pixels, _ = load_semantic_mask(image, mask, len(project.data.class_names))
    return image, related, pixels


@guarded_dataset
def relink_mask(project, image_path, mask_path, split="train"):
    """User-chosen link only. Keep annotation bytes and the split unchanged."""
    image, related, _pixels = validate_mask_link(project, image_path, mask_path, split)
    root = split_root(project, split)
    base = mask_root(project, split).parent.parent
    relative = image.relative_to(root)
    moves = [(path, base / kind / split / relative.with_suffix(path.suffix))
             for kind, path in related]
    _check_targets(moves)
    return _apply_moves(project, "relink_mask", moves)
