"""프로젝트와 작업의 원자적 JSON 기록. 브라우저 저장소에 판정을 보관하지 않는다."""

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from core.project import (ProjectData, DataConfig, ModelConfig, TrainingConfig,
                          AugmentationConfig, RunRecord, ProjectManager)


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_value(value), ensure_ascii=False, allow_nan=False, indent=2)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def restore_project(raw):
    """작업 스냅샷은 기존 절대 리소스 경로와 실제 저장 위치를 그대로 사용한다."""
    training = dict(raw["training"])
    training["augmentation"] = AugmentationConfig(**training.get("augmentation", {}))
    project = ProjectData(**{key: raw[key] for key in ("name", "task", "created", "modified", "project_dir")},
                          data=DataConfig(**raw["data"]), model=ModelConfig(**raw["model"]),
                          training=TrainingConfig(**training), runs=[RunRecord(**run) for run in raw["runs"]],
                          defect_generation=raw.get("defect_generation", {}))
    project._active_filepath = raw["filepath"]
    return project


def project_view(project):
    return json_value({**asdict(project), "filepath": ProjectManager.get_active_filepath(project)})


class ProjectStore:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "workspace.json"
        self.project = None
        self.recent = []
        self.error = ""
        self.revision = ""
        if self.state_file.exists():
            try:
                saved = read_json(self.state_file)
                self.recent = saved.get("recent", [])[:12]
                if saved.get("active"):
                    self.project = ProjectManager.load(saved["active"])
            except (OSError, ValueError, TypeError) as exc:
                self.error = f"이전 프로젝트 복원 실패: {exc}"

    def select(self, project):
        path = ProjectManager.get_active_filepath(project)
        recent = [path, *[item for item in self.recent if item != path]][:12]
        write_json(self.state_file, {"active": path, "recent": recent})
        self.project, self.recent, self.error = project, recent, ""
        self.revision = self._revision(path)

    @staticmethod
    def _revision(path):
        stat = Path(path).stat()
        return f"{stat.st_mtime_ns}:{stat.st_ctime_ns}:{stat.st_size}:{stat.st_ino}"

    def reload(self):
        if self.project is not None:
            try:
                path = ProjectManager.get_active_filepath(self.project)
                revision = self._revision(path)
                if revision == self.revision:
                    return
                self.project = ProjectManager.load(path)
                self.revision = revision
                self.error = ""
            except (OSError, ValueError, TypeError) as exc:
                # 사라지거나 손상된 프로젝트에 이전 메모리 상태를 덮어쓰지 않는다.
                self.project = None
                self.error = f"프로젝트 다시 열기 필요: {exc}"
                raise
