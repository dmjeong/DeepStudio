"""같은 PC의 브라우저만 사용하는 FastAPI 작업 API와 정적 React 화면."""

from contextlib import asynccontextmanager
import copy
from dataclasses import asdict, fields
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import string
import uuid
from functools import lru_cache
from urllib.parse import urlsplit
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from starlette.middleware.trustedhost import TrustedHostMiddleware

from webapp import ROOT
from webapp.jobs import JobManager
from webapp.storage import ProjectStore, digest, json_value, project_view, read_json
from core.project import ProjectManager
from core.model_selection import available_metrics, LABELS, selection_policy
from core.version import APP_VERSION
from core.inference_region import read_input_region, input_region_label

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathBody(Body):
    path: str


class CreateProject(Body):
    name: str
    task: str = "classify"
    parent: str
    class_names: list[str] = Field(default_factory=list)


class Settings(Body):
    training: dict = Field(default_factory=dict)
    model: dict = Field(default_factory=dict)
    data_root: str | None = None


class ClassName(Body):
    name: str
    preview_digest: str = ""


class InferenceRequest(Body):
    weights: str
    folder: str = ""
    images: list[str] = Field(default_factory=list)
    device: str = "cpu"
    gradcam: bool = True
    threads: int = Field(default=4, ge=1, le=64)
    crop_mode: Literal["model", "json", "full"] = "model"
    crop_json: str = ""


class ExportRequest(Body):
    weights: str
    output: str
    opset: int = Field(default=17, ge=17, le=20)
    dynamic_batch: bool = False


class DefectRequest(Body):
    folder: str
    output: str
    per_image: int = Field(default=1, ge=1, le=100)
    params: dict
    roi: str = ""
    texture: str = ""
    preview: bool = False
    project_path: str = ""


class DefectSettings(Body):
    project_path: str
    settings: dict


class DefectPublish(Body):
    project_path: str
    source_job: str
    sample_ids: list[str] = Field(min_length=1, max_length=10000)
    output: str


class DataGenAction(Body):
    project_path: str
    values: dict = Field(default_factory=dict)


class DatasetEdit(Body):
    action: str
    paths: list[str] = Field(default_factory=list, max_length=10000)
    folder: str = ""
    split: str = "train"
    target_split: str = "train"
    class_name: str = ""
    source_class: str = ""
    new_name: str = ""
    annotations: list[dict] | None = None


class JobSelection(Body):
    ids: list[str] = Field(min_length=1, max_length=1000)


def existing_path(value, directory=False):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("절대 경로 필요")
    path = path.resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(f"{'폴더' if directory else '파일'} 없음: {path}")
    return path


def image_paths(directory):
    root = existing_path(directory, directory=True)
    paths = []
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if not name.startswith(".") and not (Path(parent) / name).is_symlink())
        paths.extend(str(Path(parent) / name) for name in sorted(files)
                     if Path(name).suffix.lower() in IMAGE_EXTENSIONS and not (Path(parent) / name).is_symlink())
        if len(paths) > 100000:
            raise ValueError("한 작업은 최대 100,000장. 데이터 폴더 분할 필요")
    return paths


def update_dataclass(current, changes):
    allowed = {field.name for field in fields(current)}
    if set(changes) - allowed:
        raise ValueError("지원하지 않는 설정 키: " + ", ".join(set(changes) - allowed))
    merged = {**asdict(current), **changes}
    if "augmentation" in changes:
        if set(changes["augmentation"]) - {field.name for field in fields(current.augmentation)}:
            raise ValueError("지원하지 않는 증강 설정")
        merged["augmentation"] = {**asdict(current.augmentation), **changes["augmentation"]}
    return TypeAdapter(type(current)).validate_python(merged)


def create_app(state_dir=None):
    state_dir = Path(state_dir or os.environ.get("DEEP_STUDIO_STATE_DIR") or
                     Path.home() / ".deep-vision-studio-react")
    store = ProjectStore(state_dir)
    dataset_cache = {}
    result_indexes = {}
    def after_job():
        dataset_cache.clear()
        store.reload()
    jobs = JobManager(state_dir, on_finish=after_job)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        jobs.close()

    app = FastAPI(title="Deep Vision Studio", version=APP_VERSION, lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    app.state.projects, app.state.jobs = store, jobs
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "외부 페이지에서의 접근 불가"}, status_code=403)
        if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("x-studio-request") != "1":
            return JSONResponse({"detail": "스튜디오 요청 헤더 필요"}, status_code=403)
        response = await call_next(request)
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
        return response

    for exception, code in ((ValueError, 400), (FileNotFoundError, 404), (FileExistsError, 409), (PermissionError, 403), (RuntimeError, 409)):
        async def handler(_request, exc, status=code):
            return JSONResponse({"detail": str(exc)}, status_code=status)
        app.add_exception_handler(exception, handler)

    def project_required():
        with jobs.lock:
            if store.project is None:
                raise ValueError("프로젝트 생성 또는 열기 필요")
            store.reload()
            return store.project

    @app.get("/api/state")
    def state(summary: bool = False):
        with jobs.lock:
            if store.project is not None:
                try:
                    project_required()
                except (OSError, ValueError, TypeError):
                    # 오류와 최근 경로를 반환해야 브라우저에서 다른 파일을 열 수 있다.
                    pass
            project = store.project
            project_summary = {"filepath": ProjectManager.get_active_filepath(project),
                               "revision": store.revision} if project else None
            from core.model_registry import ModelRegistry
            return {"version": APP_VERSION, "project": project_view(project) if project and not summary else None,
                    "model_catalog": ModelRegistry.builtin().as_dict(),
                    "project_summary": project_summary, "warnings": jobs.warnings,
                    "recent": store.recent, "error": store.error, "active_job": jobs.active_id,
                    "jobs": jobs.list(project_path=ProjectManager.get_active_filepath(project) if project else None), "home": str(Path.home()),
                    "model_runtime_available": importlib.util.find_spec("torch") is not None}

    @app.get("/api/project/view")
    def view_project():
        return project_view(project_required())

    @app.post("/api/projects")
    def create_project(body: CreateProject):
        with jobs.lock:
            jobs.require_idle()
            if body.task != "anomaly" and not body.class_names:
                raise ValueError("클래스 이름을 한 개 이상 입력해 주세요")
            parent = existing_path(body.parent, directory=True)
            project = ProjectManager.create_new(body.name, body.task, str(parent / body.name), body.class_names)
            ProjectManager.save(project)
            store.select(project)
            return project_view(project)

    @app.post("/api/projects/open")
    def open_project(body: PathBody):
        with jobs.lock:
            jobs.require_idle()
            path = existing_path(body.path)
            if path.suffix.lower() != ".dvproj":
                raise ValueError(".dvproj 프로젝트 파일 필요")
            project = ProjectManager.load(str(path))
            store.select(project)
            return project_view(project)

    @app.put("/api/project")
    def save_project(body: Settings):
        with jobs.lock:
            jobs.require_idle()
            project = copy.deepcopy(project_required())
            project.training = update_dataclass(project.training, body.training)
            project.model = update_dataclass(project.model, body.model)
            cfg = project.training
            from center_crop import configured_center_crop
            configured_center_crop(cfg)
            for value in asdict(cfg).values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("학습 설정에 유한한 숫자 필요")
            if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.input_size < 32 or cfg.learning_rate <= 0:
                raise ValueError("에폭, 배치, 입력 크기 또는 학습률 범위 오류")
            if cfg.in_channels not in (1, 3) or cfg.early_stop_patience < 0:
                raise ValueError("채널 또는 조기 종료 설정 범위 오류")
            from core.training_modes import MODE_LABELS, training_engine_name, validate_training_options
            validate_training_options(project)
            if cfg.training_mode not in MODE_LABELS:
                raise ValueError("학습 모드 오류")
            if cfg.training_mode.startswith("efficientnet") and (
                    project.task != "classify" or cfg.efficientnet_model not in {"efficientnet_b0", "efficientnet_b1"}):
                raise ValueError("EfficientNet은 분류 태스크와 B0/B1 모델 선택 필요")
            if cfg.class_weights not in {"none", "balanced", "sqrt"} or cfg.optimizer not in {"adamw", "adam", "sgd"}:
                raise ValueError("클래스 가중치 또는 optimizer 설정 오류")
            if cfg.scheduler not in {"cosine", "step", "none"} or cfg.anomaly_method not in {"patchcore", "reconstruction"}:
                raise ValueError("scheduler 또는 이상 탐지 설정 오류")
            if not 0 < cfg.patchcore_sampling_ratio <= 1 or min(cfg.patchcore_n_neighbors, cfg.patchcore_max_candidates, cfg.patchcore_max_memory_bank) < 1:
                raise ValueError("PatchCore 비율 또는 패치 수 오류")
            if not math.isfinite(project.model.backbone_lr_mult) or project.model.backbone_lr_mult <= 0:
                raise ValueError("백본 학습률 배수 범위 오류")
            engine = training_engine_name(cfg.training_mode) if project.task != "anomaly" else "custom"
            selection_policy(cfg, engine, project.task)
            if body.data_root is not None:
                root = existing_path(body.data_root, directory=True)
                project.data.root = str(root)
                base = root / "images" if project.task in {"detect", "segment", "obb"} else root
                for split in ("train", "val", "test"):
                    setattr(project.data, f"{split}_dir", str(base / split))
                if project.task == "anomaly" and not (root / "val").is_dir():
                    project.data.val_dir = str(root / "train")
            ProjectManager.save(project)
            store.select(project)
            return project_view(project)

    @app.get("/api/options")
    def options(task: str = "classify", engine: str = "efficientnet", mode: str = "", anomaly_method: str = "patchcore"):
        from core.training_modes import training_capabilities
        capabilities = training_capabilities(task, mode or (engine + "_finetune" if engine != "custom" else "custom"), anomaly_method)
        if capabilities["engine"] != engine:
            raise ValueError("학습 모드와 엔진 불일치")
        if task not in {"classify", "detect", "segment", "anomaly", "obb"} or engine not in {"custom", "efficientnet"}:
            raise ValueError("태스크 또는 학습 엔진 오류")
        if task == "obb":
            raise ValueError("OBB 학습 엔진은 제공하지 않습니다")
        if engine == "efficientnet" and task != "classify":
            raise ValueError("EfficientNet은 분류 태스크만 지원")
        return {"metrics": [{"value": value, "label": LABELS[value]} for value in available_metrics(engine, task)],
                "models": capabilities["models"], "capabilities": capabilities}

    @app.get("/api/models")
    def models(task: str | None = None, release_ready: bool = False):
        """기본 모델/팩 카탈로그를 제공한다.

        ``release_ready``는 아직 런타임을 등록부에서 필터링하는 다음 단계의 호환 인자다.
        현재는 모든 항목의 검증 상태를 그대로 반환해 UI가 요청/검증 중임을 표시할 수 있다.
        """
        from core.model_registry import ModelRegistry
        if task is not None and task not in {"classify", "anomaly", "detect", "segment"}:
            raise ValueError("모델 카탈로그 태스크 오류")
        registry = ModelRegistry.builtin()
        selected = registry.list(task, release_ready=release_ready)
        return {"models": [spec.to_dict() for spec in selected], "release_ready": release_ready}

    @app.get("/api/files")
    def files(path: str = "", offset: int = Query(0, ge=0)):
        if not path:
            roots = [f"{letter}:\\" for letter in string.ascii_uppercase if Path(f"{letter}:\\").is_dir()] if os.name == "nt" else ["/"]
            return {"path": "", "parent": "", "entries": [{"name": value, "path": value, "directory": True} for value in roots], "total": len(roots)}
        root = existing_path(path, directory=True)
        entries = sorted((entry for entry in root.iterdir() if not entry.name.startswith(".")), key=lambda entry: (not entry.is_dir(), entry.name.casefold()))
        return {"path": str(root), "parent": str(root.parent), "total": len(entries),
                "entries": [{"name": entry.name, "path": str(entry), "directory": entry.is_dir()} for entry in entries[offset:offset + 150]]}

    @app.get("/api/images")
    def images(path: str, offset: int = Query(0, ge=0), limit: int = Query(60, ge=1, le=200)):
        paths = image_paths(path)
        return {"total": len(paths), "images": paths[offset:offset + limit]}

    @lru_cache(maxsize=128)
    def cached_image(path, revision, size):
        return render_image(path, size)

    @app.get("/api/image")
    def image(path: str, size: int = Query(512, ge=64, le=4096)):
        source = existing_path(path)
        revision = (source.stat().st_mtime_ns, source.stat().st_size)
        return cached_image(str(source), revision, size)

    def render_image(path, size):
        import numpy as np
        from PIL import Image
        from core.image_display import display_rgb
        source = existing_path(path)
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("이미지 파일 필요")
        with Image.open(source) as original:
            pixels = display_rgb(np.array(original.convert("RGB") if original.mode == "P" else original))
        preview = Image.fromarray(pixels)
        preview.thumbnail((size, size))
        stream = io.BytesIO()
        preview.save(stream, format="PNG")
        return Response(stream.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=0, must-revalidate"})

    def dataset_index(rescan=False):
        from core.dataset_editor import scan_dataset
        with jobs.lock:
            project = project_required()
            key = (ProjectManager.get_active_filepath(project), store.revision)
            cached = dataset_cache.get(key)
            if rescan or cached is None:
                value = scan_dataset(project)
                dataset_cache.clear()
                dataset_cache[key] = value
            return dataset_cache[key]

    @app.get("/api/dataset")
    def dataset(rescan: bool = False):
        return {key: value for key, value in dataset_index(rescan).items() if key != "images"}

    @app.get("/api/dataset/images")
    def dataset_images(split: str = "train", class_name: str = "", q: str = "", status: str = "all",
                       offset: int = Query(0, ge=0), limit: int = Query(36, ge=1, le=200)):
        if split not in {"all", "train", "val", "test"}:
            raise ValueError("데이터 분할 오류")
        records = [row for row in dataset_index()["images"]
                   if (split == "all" or row["split"] == split)
                   and (not class_name or class_name in row["classes"])
                   and q.casefold() in row["name"].casefold()
                   and (status == "all" or status == "unlabeled" and not row["annotated"]
                        or status == "error" and row["error"])]
        return {"total": len(records), "images": records[offset:offset + limit]}

    @app.get("/api/dataset/annotations")
    def dataset_annotations(path: str, split: str):
        from core.dataset_editor import annotation_view
        return annotation_view(project_required(), path, split)

    @app.post("/api/dataset/edit")
    def dataset_edit(body: DatasetEdit):
        with jobs.lock:
            jobs.require_idle()
            project = project_required()
            edit = body.model_dump(exclude={"folder"})
            if body.folder:
                if body.action != "import":
                    raise ValueError("폴더는 이미지 가져오기에서만 사용 가능")
                edit["paths"] = image_paths(body.folder)
            if not edit["paths"] and body.action != "rename_class":
                raise ValueError("이미지 선택 필요")
            return jobs.start("dataset_edit", {"project": project_view(project),
                "project_digest": digest(ProjectManager.get_active_filepath(project)), "edit": edit})

    @app.post("/api/dataset/upload")
    async def upload(request: Request, name: str, batch: str):
        from core.project import ProjectManager
        jobs.require_idle()
        if len(batch) != 32 or any(c not in "0123456789abcdef" for c in batch):
            raise ValueError("업로드 묶음 ID 오류")
        ProjectManager.validate_name(name, kind="파일")
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS | {".txt"}:
            raise ValueError("이미지 또는 정규화 정답 파일만 추가 가능")
        target = state_dir / "uploads" / batch / name
        target.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        created = False
        try:
            with target.open("xb") as stream:
                created = True
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > 128 * 1024 * 1024:
                        raise ValueError("파일 한 개는 최대 128 MB")
                    stream.write(chunk)
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
        return {"path": str(target)}

    @app.get("/api/storage")
    def storage_usage():
        result = jobs.storage()
        upload_root = state_dir / "uploads"
        result["upload_bytes"] = sum(p.stat().st_size for p in upload_root.rglob("*") if p.is_file())
        return result

    @app.post("/api/storage/delete")
    def delete_history(body: JobSelection):
        return jobs.delete(body.ids)

    @app.post("/api/storage/clear-uploads")
    def clear_uploads():
        import shutil
        with jobs.lock:
            jobs.require_idle()
            shutil.rmtree(state_dir / "uploads", ignore_errors=True)
            return {"cleared": True}

    @app.get("/api/history/{job_id}/archive")
    def history_archive(job_id: str):
        return FileResponse(jobs.export_archive(job_id), filename=f"studio-job-{job_id}.zip")

    @app.post("/api/classes")
    def add_class(body: ClassName):
        from core.class_management import ClassManager
        with jobs.lock:
            jobs.require_idle()
            ClassManager.add(project_required(), body.name)
            return project_view(store.project)

    @app.post("/api/classes/preview-delete")
    def preview_class(body: ClassName):
        from core.class_management import ClassManager
        with jobs.lock:
            jobs.require_idle()
            preview = asdict(ClassManager.preview_delete(project_required(), body.name))
            preview_hash = hashlib.sha256(json.dumps(json_value(preview), sort_keys=True).encode()).hexdigest()
            return {**{key: value for key, value in preview.items() if not isinstance(value, tuple)}, "preview_digest": preview_hash}

    @app.post("/api/classes/delete")
    def delete_class(body: ClassName):
        with jobs.lock:
            if not body.preview_digest:
                raise ValueError("삭제 미리보기 확인 필요")
            return jobs.start("delete_class", {"project": project_view(project_required()), "class_name": body.name,
                                                "project_digest": digest(ProjectManager.get_active_filepath(store.project)),
                                                "preview_digest": body.preview_digest})

    @app.post("/api/jobs/train")
    def train():
        with jobs.lock:
            project = project_required()
            jobs.require_idle()
            if project.task == "obb":
                raise ValueError("OBB 학습 엔진은 제공하지 않습니다")
            if project.training.training_mode in {"efficientnet_transfer", "efficientnet_resume"} and not project.model.pretrained_weights:
                raise ValueError("이어학습 체크포인트 경로 필요")
            if project.training.training_mode.startswith("efficientnet") and project.task != "classify":
                raise ValueError("EfficientNet은 분류 태스크만 지원")
            ProjectManager.save(project)
            return jobs.start("train", {"project": project_view(project), "project_digest": digest(ProjectManager.get_active_filepath(project))})

    @app.get("/api/inference/region")
    def inference_region(path: str):
        region = read_input_region("json", path)
        return {**region, "label": input_region_label(region)}

    @app.post("/api/jobs/infer")
    def infer(body: InferenceRequest):
        with jobs.lock:
            jobs.require_idle()
            existing_path(body.weights)
            paths = image_paths(body.folder) if body.folder else [str(existing_path(path)) for path in body.images]
            if not paths or len(paths) != len(set(paths)):
                raise ValueError("중복 없는 추론 이미지 필요")
            region = read_input_region(body.crop_mode, body.crop_json)
            return jobs.start("infer", {**body.model_dump(exclude={"folder", "crop_mode", "crop_json"}),
                "input_region": region, "images": paths,
                "project": project_view(store.project) if store.project else {}})

    @app.post("/api/jobs/export")
    def export(body: ExportRequest):
        with jobs.lock:
            jobs.require_idle()
            existing_path(body.weights)
            if not Path(body.output).is_absolute():
                raise ValueError("출력의 절대 경로 필요")
            return jobs.start("export", body.model_dump())

    def defect_project(path=""):
        project = project_required()
        if path and path != ProjectManager.get_active_filepath(project):
            raise ValueError("프로젝트가 변경됨. 합성 화면 다시 열기 필요")
        return project

    @app.get("/api/datagen")
    def datagen_state():
        from core.datagen_store import DataGenStore
        with jobs.lock:
            return DataGenStore(project_required()).state()

    @app.post("/api/datagen/{action}")
    def datagen_action(action: str, body: DataGenAction):
        from core.datagen_store import mutate
        with jobs.lock:
            jobs.require_idle()
            return mutate(defect_project(body.project_path), action, body.values)

    @app.get("/api/datagen/image/{collection}/{key}/{kind}")
    def datagen_image(collection: str, key: str, kind: str):
        from core.datagen_store import DataGenStore
        with jobs.lock:
            path = DataGenStore(project_required()).file(collection, key, kind)
        return FileResponse(path, media_type="image/png")

    @app.post("/api/jobs/datagen/{operation}")
    def datagen_job(operation: str, body: DataGenAction):
        if operation not in {"train", "generate"}:
            raise ValueError("Data Gen 계산 작업 오류")
        with jobs.lock:
            jobs.require_idle()
            project = defect_project(body.project_path)
            return jobs.start("datagen_" + operation, {**body.values, "project": project_view(project)})

    @app.get("/api/defects/settings")
    def defect_settings():
        from core.defect_workflow import settings_for
        with jobs.lock:
            return settings_for(project_required())

    @app.put("/api/defects/settings")
    def save_defect_settings(body: DefectSettings):
        from core.defect_workflow import validate_settings
        with jobs.lock:
            jobs.require_idle()
            project = copy.deepcopy(defect_project(body.project_path))
            project.defect_generation = validate_settings(body.settings)
            ProjectManager.save(project)
            store.select(project)
            return project.defect_generation

    @app.post("/api/jobs/defects")
    def defects(body: DefectRequest):
        from core.defect_workflow import validate_settings, validate_output, validate_source
        with jobs.lock:
            jobs.require_idle()
            project = defect_project(body.project_path)
            values = validate_settings(body.model_dump(exclude={"preview", "project_path"}))
            existing_path(values["folder"], directory=True)
            validate_source(project, Path(values["folder"]) / "__source__")
            validate_output(project, values["output"])
            for key in ("roi", "texture"):
                if values[key]:
                    existing_path(values[key])
            return jobs.start("defects", {**values, "preview": body.preview, "project": project_view(project)})

    def defect_job(job_id):
        directory = jobs.directory(job_id)
        job = read_json(directory / "job.json")
        project = project_required()
        if job.get("kind") != "defects" or job.get("project_path") != ProjectManager.get_active_filepath(project):
            raise ValueError("현재 프로젝트의 합성 작업 선택 필요")
        return directory

    @app.get("/api/defects/jobs")
    def defect_jobs():
        with jobs.lock:
            project = project_required()
            return [job for job in jobs.list(ProjectManager.get_active_filepath(project), limit=10000) if job["kind"] == "defects"]

    @app.get("/api/defects/{job_id}/candidates")
    def defect_candidates(job_id: str, offset: int = Query(0, ge=0), limit: int = Query(48, ge=1, le=200)):
        from core.defect_workflow import candidates, candidate_ids
        with jobs.lock:
            directory = defect_job(job_id)
        return {"total": len(candidate_ids(directory)), "samples": candidates(directory, offset, limit)}

    @app.get("/api/defects/{job_id}/image/{sample_id}")
    def defect_image(job_id: str, sample_id: str, kind: str = "image", size: int = Query(1024, ge=64, le=4096)):
        from core.defect_workflow import candidate
        if kind not in {"image", "mask", "original"}:
            raise ValueError("합성 이미지 종류 오류")
        with jobs.lock:
            directory = defect_job(job_id)
        row = candidate(directory, sample_id)
        return render_image(row[kind], size)

    @app.post("/api/jobs/defect-publish")
    def defect_publish(body: DefectPublish):
        from core.defect_workflow import validate_output
        with jobs.lock:
            jobs.require_idle()
            project = defect_project(body.project_path)
            directory = defect_job(body.source_job)
            validate_output(project, body.output)
            return jobs.start("defect_publish", {"project": project_view(project), "source_directory": str(directory),
                              "sample_ids": body.sample_ids, "output": body.output})

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, offset: int = Query(0, ge=0)):
        with jobs.lock:
            return jobs.read(job_id, offset)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return jobs.cancel(job_id)

    @app.get("/api/jobs/{job_id}/results")
    def results(job_id: str, offset: int = Query(0, ge=0), limit: int = Query(60, ge=1, le=200),
                threshold: float | None = None, class_name: str = "", decision: str = "", search: str = "",
                selected: str = "", selected_only: bool = False, sort: str = "index", descending: bool = False):
        from core.inference_review import review_page, source_class
        root = jobs.directory(job_id) / "results"
        revision = root.stat().st_mtime_ns if root.exists() else 0
        cached = result_indexes.get(job_id)
        if cached is None or cached[0] != revision:
            paths = sorted((p for p in root.glob("*.json") if p.stem.isdigit()), key=lambda path: int(path.stem))
            if len(result_indexes) >= 64:
                result_indexes.pop(next(iter(result_indexes)))
            result_indexes[job_id] = (revision, paths)
        records = [read_json(path) for path in result_indexes[job_id][1]]
        request_path = root.parent / "request.json"
        project = read_json(request_path).get("payload", {}).get("project", {}) if request_path.exists() else {}
        for row in records:
            if not row.get("source_class"):
                row["source_class"] = source_class(row.get("image_path", ""), project)
        return review_page(records, threshold=threshold, class_name=class_name, decision=decision,
                           search=search, selected=[int(i) for i in selected.split(",") if i],
                           selected_only=selected_only, sort=sort, descending=descending, offset=offset, limit=limit)

    @app.get("/api/jobs/{job_id}/download/{name}")
    def job_download(job_id: str, name: str):
        if name not in {"output_path", "config_path", "archive_path"}:
            raise ValueError("다운로드 종류 오류")
        job = jobs.read(job_id)["job"]
        path = job.get("output", {}).get(name)
        if not path:
            raise FileNotFoundError("작업의 저장 파일 없음")
        source = existing_path(path)
        return FileResponse(source, filename=source.name)

    @app.get("/api/jobs/{job_id}/image/{index}")
    def result_image(job_id: str, index: int, cam: bool = True, original: bool = False, lower: float = Query(0, ge=0, le=1),
                     upper: float = Query(1, ge=0, le=1), alpha: float = Query(.5, ge=0, le=1),
                     size: int = Query(2048, ge=64, le=4096)):
        import numpy as np
        from PIL import Image
        from core.heatmap import render_heatmap
        from core.spatial_preview import draw_detection_boxes
        if index < 0 or lower >= upper:
            raise ValueError("이미지 번호 또는 히트맵 범위 오류")
        root = jobs.directory(job_id) / "results"
        record = read_json(root / f"{index}.json")
        if not (root / f"{index}.npz").is_file():
            return image(record["image_path"], size)
        thumbnail = root / f"{index}.thumb.png"
        if size <= 256 and thumbnail.is_file() and not cam and not original:
            return FileResponse(thumbnail, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})
        preview_key = hashlib.sha256(f"{index}:{cam}:{original}:{lower}:{upper}:{alpha}:{size}".encode()).hexdigest()
        rendered = jobs.directory(job_id) / "previews" / (preview_key + ".png")
        if record.get("cache_ready") and rendered.is_file():
            return FileResponse(rendered, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})
        with np.load(root / f"{index}.npz", allow_pickle=False) as stored:
            keys = ("original",) if original else ("original", "activation", "valid_mask") if record.get("heatmap") and cam else ("preview",)
            arrays = {key: stored[key] for key in keys if key in stored.files}
        heatmap = record.get("heatmap")
        if original:
            pixels = arrays.get("original")
        elif heatmap and cam:
            _, pixels = render_heatmap(arrays["activation"], arrays["original"], alpha, lower, upper,
                                       valid_mask=arrays.get("valid_mask"))
            if heatmap.get("detections") is not None:
                pixels = draw_detection_boxes(pixels, heatmap["detections"])
        else:
            pixels = arrays.get("preview")
        if pixels is None:
            return image(record["image_path"], size)
        preview = Image.fromarray(pixels)
        preview.thumbnail((size, size))
        stream = io.BytesIO()
        preview.save(stream, format="PNG")
        if record.get("cache_ready"):
            rendered.parent.mkdir(exist_ok=True)
            # 256개를 넘는 표시 조합은 오래된 것부터 정리.
            cached = sorted(rendered.parent.glob("*.png"), key=lambda p: p.stat().st_mtime)
            for old in cached[:-255]:
                old.unlink(missing_ok=True)
            temporary = rendered.with_name(uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(stream.getvalue())
            os.replace(temporary, rendered)
        return Response(stream.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/download")
    def download(path: str):
        source = existing_path(path)
        project = project_required()
        if not source.is_relative_to(Path(project.project_dir).resolve()):
            raise ValueError("현재 프로젝트 폴더 안의 결과만 다운로드 가능")
        return FileResponse(source, filename=source.name)

    @app.get("/api/runs/{index}/csv")
    def run_csv(index: int):
        project = project_required()
        if not 0 <= index < len(project.runs):
            raise FileNotFoundError("학습 기록 없음")
        record = project.runs[index]
        checkpoint = Path(record.checkpoint_path)
        # 공식 엔진은 weights/ 아래, Custom과 PatchCore는 실행 폴더에 저장한다.
        for directory in (checkpoint.parent, checkpoint.parent.parent):
            target = directory / "results.csv"
            if target.is_file():
                return download(str(target))
        raise FileNotFoundError("이 학습 기록의 CSV 없음")

    frontend = ROOT / "web" / "dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="react")
    else:
        @app.get("/")
        def build_required():
            return JSONResponse({"detail": "웹 화면 빌드 없음. 릴리스 ZIP 사용 또는 web 폴더에서 npm ci와 npm run build 실행 필요"}, status_code=503)
    return app
