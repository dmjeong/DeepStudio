"""Recover interrupted dataset writes before reading or editing that project."""

import hashlib
import json
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from core.annotation_tools import atomic_write
from core.class_management import SPLITS, _plain_path
from core.project import ProjectManager


def journal_paths(project):
    project_file = Path(ProjectManager.get_active_filepath(project)).resolve()
    root = _plain_path(
        Path(project.project_dir).parent / ".deep_studio_dataset_archive"
    )
    key = hashlib.sha256(os.path.normcase(str(project_file)).encode()).hexdigest()
    return root, root / ("pending-" + key + ".json"), root / ("lock-" + key)


@contextmanager
def project_edit_lock(project):
    root, _, lock = journal_paths(project)
    root.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        if not lock.stat().st_size:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "다른 창에서 같은 데이터셋을 읽거나 저장 중입니다. 잠시 후 다시 시도하세요."
            ) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def write_journal(transaction, status):
    _root, pointer, _ = journal_paths(transaction.project)
    value = {
        "schema_version": 1,
        "project_file": str(
            Path(ProjectManager.get_active_filepath(transaction.project)).resolve()
        ),
        "action": transaction.action,
        "status": status,
        "files": [
            {"path": str(p), "backup": b.name if b else None}
            for p, b in transaction.paths.items()
        ],
        "created_dirs": [str(p) for p in transaction.created_dirs],
    }
    atomic_write(
        transaction.backup / "manifest.json",
        json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    if status in {"pending", "recovery_required"}:
        atomic_write(
            pointer,
            json.dumps(
                {
                    "backup": transaction.backup.name,
                    "project_file": value["project_file"],
                }
            ).encode("utf-8"),
        )
    else:
        pointer.unlink(missing_ok=True)


def recover_pending(project):
    """Caller holds project_edit_lock. Repeating a recovery is harmless."""
    root, pointer, _ = journal_paths(project)
    if not pointer.exists():
        return False
    _plain_path(pointer)
    reference = json.loads(pointer.read_text(encoding="utf-8"))
    name = reference.get("backup")
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError("데이터 복구 저널 경로 오류")
    backup = _plain_path(root / name)
    manifest = _plain_path(backup / "manifest.json")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    expected = Path(ProjectManager.get_active_filepath(project)).resolve()
    if value.get("project_file") != str(expected) or reference.get(
        "project_file"
    ) != str(expected):
        raise ValueError("다른 프로젝트의 복구 저널입니다.")
    if value.get("status") in {"completed", "rolled_back"}:
        pointer.unlink(missing_ok=True)
        return False
    if value.get("status") not in {"pending", "recovery_required"} or not isinstance(
        value.get("files"), list
    ):
        raise ValueError("데이터 복구 저널 형식 오류")
    roots = [
        Path(project.data.root).resolve(),
        *(Path(getattr(project.data, f"{sp}_dir")).resolve() for sp in SPLITS),
    ]
    restores = []
    for row in value["files"]:
        path = _plain_path(row["path"])
        resolved = path.resolve()
        if resolved != expected and not any(
            resolved.is_relative_to(folder) for folder in roots
        ):
            raise ValueError("데이터 복구 대상이 프로젝트 밖에 있습니다.")
        old = row.get("backup")
        if old is not None and (not isinstance(old, str) or not old.isdigit()):
            raise ValueError("데이터 복구 백업 이름 오류")
        source = _plain_path(backup / old) if old is not None else None
        if source is not None and not source.is_file():
            raise FileNotFoundError(f"데이터 복구 백업이 없습니다: {source}")
        restores.append((path, source))
    # Validate every entry before changing any file.
    for path, source in reversed(restores):
        if source is not None:
            atomic_write(path, source.read_bytes())
        else:
            path.unlink(missing_ok=True)
    value["status"] = "rolled_back"
    atomic_write(
        manifest, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    )
    pointer.unlink(missing_ok=True)
    if expected.is_file():
        restored = ProjectManager.load(str(expected))
        project.__dict__.update(restored.__dict__)
    return True


def guarded_dataset(function):
    @wraps(function)
    def guarded(project, *args, **kwargs):
        with project_edit_lock(project):
            recover_pending(project)
            return function(project, *args, **kwargs)

    return guarded
