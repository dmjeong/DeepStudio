"""
Deep Vision Studio — 프로젝트 관리

프로젝트 파일 구조 (.dvproj):
┌────────────────────────────────────────┐
│ {                                      │
│   "name": "MyProject",                │
│   "task": "classify",                 │
│   "created": "2026-09-02T...",        │
│   "data": { ... },                    │
│   "model": { ... },                   │
│   "training": { ... },               │
│   "runs": [ ... ]                     │
│ }                                      │
└────────────────────────────────────────┘
"""

import json
import ntpath
import os
import re
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any


# ── 지원 태스크 ────────────────────────────────────
SUPPORTED_TASKS = {
    "classify": {
        "name": "Classification",
        "icon": "CLS",          # 텍스트 기반 식별자 (이모지 제거)
        "description": "이미지 분류 — 이미지 전체를 하나의 클래스로 분류",
        "default_input_size": 224,
        "default_batch_size": 16,
    },
    "segment": {
        "name": "Segmentation",
        "icon": "SEG",
        "description": "시맨틱 분할 — 픽셀 단위 클래스 레이블 예측",
        "default_input_size": 320,
        "default_batch_size": 8,
    },
    "detect": {
        "name": "Object Detection",
        "icon": "DET",
        "description": "객체 탐지 — 바운딩 박스 + 클래스 예측",
        "default_input_size": 416,
        "default_batch_size": 8,
    },
    "obb": {
        "name": "Oriented Object Detection",
        "icon": "OBB",
        "description": "회전 객체 탐지 - 네 꼭짓점과 클래스 예측",
        "default_input_size": 640,
        "default_batch_size": 8,
    },
    "anomaly": {
        "name": "Anomaly Detection",
        "icon": "ANO",
        "description": "이상 탐지 — 정상/비정상 이미지 판별 (정상 데이터만 학습)",
        "default_input_size": 224,
        "default_batch_size": 16,
    },
}


@dataclass
class DataConfig:
    """데이터셋 설정"""
    root: str = ""  # 데이터 루트 디렉토리
    train_dir: str = ""  # 학습 데이터 경로
    val_dir: str = ""  # 검증 데이터 경로
    test_dir: str = ""  # 테스트 데이터 경로
    class_names: List[str] = field(default_factory=list)
    num_classes: int = 0
    val_split: float = 0.2            # 검증 데이터 분할 비율
    image_count: int = 0              # 총 이미지 수


@dataclass
class ModelConfig:
    """모델 아키텍처 설정"""
    backbone_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )
    csp_depth: List[int] = field(
        default_factory=lambda: [1, 2, 3, 2]
    )
    dropout: float = 0.2
    # 트랜스퍼 러닝 설정
    pretrained_weights: str = ""  # 사전학습 가중치 경로
    freeze_backbone: bool = False     # 백본 동결 여부
    backbone_lr_mult: float = 0.1     # 백본 학습률 배수 (파인튜닝 시)


@dataclass
class AugmentationConfig:
    """데이터 증강 설정"""
    horizontal_flip: float = 0.5      # 수평 뒤집기 확률
    vertical_flip: float = 0.0        # 수직 뒤집기 확률
    rotation: float = 15.0            # 회전 각도 범위
    color_jitter: float = 0.2         # 색상 변화 범위
    scale_range: List[float] = field(
        default_factory=lambda: [0.8, 1.2]
    )
    mixup_alpha: float = 0.0          # Mixup 알파
    mosaic: bool = False              # 모자이크 증강 (detection용)


@dataclass
class TrainingConfig:
    """학습 하이퍼파라미터"""
    epochs: int = 100
    batch_size: int = 8
    input_size: int = 224
    in_channels: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: str = "adamw"  # adamw, sgd, adam
    scheduler: str = "cosine"  # cosine, step, none
    warmup_epochs: int = 3
    early_stop_patience: int = 15
    selection_metric: str = "engine_default"  # 검증 지표로 Best 저장과 조기 종료 판단
    label_smoothing: float = 0.05     # (classification용)
    # 클래스 불균형 보정: "none"| "balanced"(역빈도) | "sqrt"(완만한 역빈도)
    class_weights: str = "none"
    device: str = "auto"  # auto, cpu, cuda, cuda:0, cuda:1
    use_amp: bool = True              # Mixed Precision (GPU 시 자동 활성화)
    # ── 학습 모드 ──
    #  "custom"  : 커스텀 CustomCSP 모델 스크래치/트랜스퍼 학습
    training_mode: str = "efficientnet_finetune"
    efficientnet_model: str = "efficientnet_b0"
    efficientnet_no_decay: bool = True
    layer_debug_enabled: bool = False
    layer_debug_patterns: str = "features.0,features.1.*,classifier.1"
    layer_debug_batches: int = 1
    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig
    )
    # ── 이상 탐지 방법 ──
    #  "patchcore"      : PatchCore (메모리 뱅크 기반, SOTA)
    #  "reconstruction" : 재구성 기반 (기존 AnomalyHead 사용)
    anomaly_method: str = "patchcore"
    # PatchCore 코어셋 비율 (전체 패치 중 저장할 비율, 기본 1%)
    patchcore_sampling_ratio: float = 0.01
    # PatchCore kNN 이웃 수
    patchcore_n_neighbors: int = 9
    patchcore_backbone: str = "wide_resnet50_2"
    patchcore_weight_source: str = "imagenet"  # imagenet / backbone / patchcore
    patchcore_weights: str = ""
    patchcore_append: bool = False
    patchcore_max_candidates: int = 20000
    patchcore_max_memory_bank: int = 4096
    patchcore_seed: int = 0
    # Shared by every task; retain legacy key names for saved project compatibility.
    patchcore_crop_enabled: bool = False
    patchcore_crop_width: int = 1024
    patchcore_crop_height: int = 1024


@dataclass
class RunRecord:
    """
    학습 실행 기록

    ┌──────────────────────────────────────────────────┐
    │  RunRecord                                       │
    │  ├── 기본 정보: run_id, status, epochs_done      │
    │  ├── 최적 결과: best_metric, best_epoch          │
    │  ├── 학습 이력: metrics_history (에폭별 기록)     │
    │  ├── 평가 결과: eval_results (종합 평가 지표)     │
    │  └── 학습률 이력: lr_history                      │
    └──────────────────────────────────────────────────┘
    """
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    status: str = "pending"  # pending, running, completed, failed
    epochs_done: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    best_metric_name: str = ""  # 주요 메트릭 이름 (accuracy, mIoU 등)
    checkpoint_path: str = ""
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    lr_history: List[float] = field(default_factory=list)         # 에폭별 LR
    eval_results: Dict[str, Any] = field(default_factory=dict)    # 종합 평가 결과
    config_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectData:
    """
    CustomCSP 프로젝트 전체 데이터 모델

    저장/로드 가능 JSON 구조
    """
    name: str = "Untitled"
    task: str = "classify"
    created: str = ""
    modified: str = ""
    project_dir: str = ""  # 프로젝트 파일 위치
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runs: List[RunRecord] = field(default_factory=list)
    defect_generation: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        # 태스크별 기본값 적용
        if self.task in SUPPORTED_TASKS:
            info = SUPPORTED_TASKS[self.task]
            if not self.training.input_size:
                self.training.input_size = info["default_input_size"]
            if not self.training.batch_size:
                self.training.batch_size = info["default_batch_size"]


class ProjectManager:
    """
    프로젝트 생성/저장/로드 관리

    사용 흐름:
    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │ create_new │───►│   save     │───►│   load     │
    │ (새 프로젝트)│    │ (.dvproj) │    │ (기존 로드) │
    └────────────┘    └────────────┘    └────────────┘
    """

    FILE_EXTENSION = ".dvproj"

    @staticmethod
    def validate_name(name: str, kind: str = "프로젝트") -> str:
        """Windows와 POSIX에서 단일 파일명으로 사용 가능한 이름 검사."""
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError(f"{kind} 이름이 비어 있거나 앞뒤 공백 포함")
        if (name in {".", ".."} or name.endswith((".", " "))
                or any(ord(c) < 32 or c in '<>:"/\\|?*' for c in name)):
            raise ValueError(f"{kind} 이름에 경로 구분자 또는 금지 문자 포함")
        reserved = {"CON", "PRN", "AUX", "NUL"}
        reserved.update(f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10))
        if name.split(".")[0].upper() in reserved:
            raise ValueError(f"{kind} 이름에 Windows 예약 이름 사용 불가")
        return name

    @staticmethod
    def get_active_filepath(project: ProjectData) -> str:
        """열거나 저장한 실제 파일 경로. 실행 중 속성이므로 JSON에는 미포함."""
        active = getattr(project, "_active_filepath", "")
        if active:
            return active
        name = ProjectManager.validate_name(project.name)
        return os.path.abspath(os.path.join(project.project_dir, name + ProjectManager.FILE_EXTENSION))

    @staticmethod
    def _relative_inside(path: str, base: str):
        if not path or not base:
            return None
        path_module = ntpath if ntpath.splitdrive(path)[0] or ntpath.splitdrive(base)[0] else os.path
        try:
            target = path_module.abspath(path)
            root = path_module.abspath(base)
            if path_module.normcase(path_module.commonpath([target, root])) == path_module.normcase(root):
                return path_module.relpath(target, root).replace("\\", "/")
        except (ValueError, OSError):
            pass
        return None

    @staticmethod
    def _path_fields(raw: dict):
        """프로젝트 내부 이동에 따라 재배치할 현재 리소스 경로."""
        for key in ("root", "train_dir", "val_dir", "test_dir"):
            yield raw["data"], key
        yield raw["model"], "pretrained_weights"
        yield raw.setdefault("training", {}), "patchcore_weights"
        for key in ("folder", "output", "roi", "texture"):
            yield raw.setdefault("defect_generation", {}), key
        for run in raw.get("runs", []):
            yield run, "checkpoint_path"

    @staticmethod
    def create_new(name: str, task: str, project_dir: str,
                   class_names: List[str] = None) -> ProjectData:
        """
        새 프로젝트 생성 + 데이터 폴더 자동 구성

        Args:
            name: 프로젝트 이름
            task: 태스크 종류 (classify/segment/detect/anomaly)
            project_dir: 프로젝트 저장 디렉토리
            class_names: 클래스 이름 목록 (classify/segment/detect 용)

        자동 생성되는 폴더 구조:
        ┌── Classification ──────────────┐
        │  data/                         │
        │  ├── train/{클래스1, 클래스2}/  │
        │  ├── val/{클래스1, 클래스2}/    │
        │  └── test/{클래스1, 클래스2}/   │
        ├── Segmentation ────────────────┤
        │  data/                         │
        │  ├── images/{train,val,test}/  │
        │  └── masks/{train,val,test}/   │
        ├── Detection ───────────────────┤
        │  data/                         │
        │  ├── images/{train,val,test}/  │
        │  └── labels/{train,val,test}/  │
        ├── Anomaly ─────────────────────┤
        │  data/                         │
        │  ├── train/good/               │
        │  └── test/{good,defect}/       │
        └────────────────────────────────┘
        """
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"미지원 태스크: {task}")
        ProjectManager.validate_name(name)
        project_dir = os.path.abspath(project_dir)
        if os.path.exists(project_dir) and (not os.path.isdir(project_dir) or os.listdir(project_dir)):
            raise FileExistsError(f"기존 프로젝트 또는 파일이 있는 경로 사용 불가: {project_dir}")

        info = SUPPORTED_TASKS[task]
        if class_names is None:
            class_names = []
        class_names = list(class_names)
        for cls_name in class_names:
            ProjectManager.validate_name(cls_name, "클래스")
        if len({c.casefold() for c in class_names}) != len(class_names):
            raise ValueError("중복 클래스 이름 사용 불가")

        project = ProjectData(
            name=name,
            task=task,
            project_dir=project_dir,
            training=TrainingConfig(
                input_size=info["default_input_size"],
                batch_size=info["default_batch_size"],
                training_mode="efficientnet_finetune" if task == "classify" else "custom",
            ),
        )

        # ── 프로젝트 기본 폴더 생성 ──
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "runs"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "exports"), exist_ok=True)

        # ── 태스크별 데이터 폴더 자동 생성 ──
        data_root = os.path.join(project_dir, "data")
        os.makedirs(data_root, exist_ok=True)

        if task == "classify":
            # Classification: data/{train,val,test}/{클래스명}/
            for split in ["train", "val", "test"]:
                split_dir = os.path.join(data_root, split)
                os.makedirs(split_dir, exist_ok=True)
                for cls_name in class_names:
                    os.makedirs(os.path.join(split_dir, cls_name), exist_ok=True)
            project.data.train_dir = os.path.join(data_root, "train")
            project.data.val_dir = os.path.join(data_root, "val")
            project.data.test_dir = os.path.join(data_root, "test")

        elif task == "segment":
            # Segmentation: data/{images,masks}/{train,val,test}/
            for sub in ["images", "masks", "labels"]:
                for split in ["train", "val", "test"]:
                    os.makedirs(os.path.join(data_root, sub, split), exist_ok=True)
            project.data.train_dir = os.path.join(data_root, "images", "train")
            project.data.val_dir = os.path.join(data_root, "images", "val")
            project.data.test_dir = os.path.join(data_root, "images", "test")

        elif task in ("detect", "obb"):
            # Detection: data/{images,labels}/{train,val,test}/
            for sub in ["images", "labels"]:
                for split in ["train", "val", "test"]:
                    os.makedirs(os.path.join(data_root, sub, split), exist_ok=True)
            project.data.train_dir = os.path.join(data_root, "images", "train")
            project.data.val_dir = os.path.join(data_root, "images", "val")
            project.data.test_dir = os.path.join(data_root, "images", "test")

        elif task == "anomaly":
            # Anomaly: data/train/good/, data/test/{good,defect}/
            os.makedirs(os.path.join(data_root, "train", "good"), exist_ok=True)
            os.makedirs(os.path.join(data_root, "test", "good"), exist_ok=True)
            os.makedirs(os.path.join(data_root, "test", "defect"), exist_ok=True)
            project.data.train_dir = os.path.join(data_root, "train")
            project.data.val_dir = os.path.join(data_root, "train")  # anomaly: train=val
            project.data.test_dir = os.path.join(data_root, "test")

        # 데이터 설정 저장
        project.data.root = data_root
        project.data.class_names = class_names
        project.data.num_classes = len(class_names) if class_names else 0

        return project

    @staticmethod
    def save(project: ProjectData, filepath: str = None) -> str:
        """
        프로젝트를 JSON 파일로 저장

        Returns: 저장된 파일 경로
        """
        filepath = os.path.abspath(filepath or ProjectManager.get_active_filepath(project))
        base = os.path.dirname(filepath)
        modified = datetime.now().isoformat()
        data = asdict(project)
        data["schema_version"] = 2
        data["project_dir"] = "."
        data["modified"] = modified
        for container, key in ProjectManager._path_fields(data):
            value = container.get(key, "")
            relative = ProjectManager._relative_inside(value, base)
            if relative is not None:
                container[key] = relative
        # 원본을 열기 전에 직렬화 완료. 실패시 이전 파일과 메모리 상태 보존.
        payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=base,
                                             prefix=".project-", suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, filepath)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        project.modified = modified
        project.project_dir = base
        project._active_filepath = filepath
        return filepath

    @staticmethod
    def load(filepath: str) -> ProjectData:
        """
        JSON 파일에서 프로젝트 로드

        Returns: ProjectData 인스턴스
        """
        filepath = os.path.abspath(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict) or raw.get("task", "classify") not in SUPPORTED_TASKS:
            raise ValueError("지원하지 않는 프로젝트 형식 또는 태스크")
        version = raw.get("schema_version", 1)
        if version not in (1, 2):
            raise ValueError(f"지원하지 않는 프로젝트 버전: {version}")
        base = os.path.dirname(filepath)
        old_base = raw.get("project_dir", base)
        if not isinstance(raw.get("defect_generation", {}), dict):
            raise ValueError("합성 설정 형식 오류")
        raw.setdefault("data", {})
        raw.setdefault("model", {})
        for container, key in ProjectManager._path_fields(raw):
            value = container.get(key, "")
            if not value:
                continue
            if not isinstance(value, str):
                raise ValueError(f"프로젝트 경로 형식 오류: {key}")
            absolute = os.path.isabs(value) or ntpath.isabs(value)
            relative = ProjectManager._relative_inside(value, old_base) if version == 1 else None
            if relative is not None:
                container[key] = os.path.abspath(os.path.join(base, *relative.split("/")))
            elif not absolute:
                container[key] = os.path.abspath(os.path.join(base, *value.replace("\\", "/").split("/")))

        # 중첩 dataclass 복원
        data_cfg = DataConfig(**raw.get("data", {}))
        ProjectManager.validate_name(raw.get("name", "Untitled"))
        for class_name in data_cfg.class_names:
            ProjectManager.validate_name(class_name, "클래스")
        if len({name.casefold() for name in data_cfg.class_names}) != len(data_cfg.class_names):
            raise ValueError("중복 클래스 이름이 포함된 프로젝트")
        model_cfg = ModelConfig(**raw.get("model", {}))

        # augmentation 중첩 복원
        train_raw = raw.get("training", {})
        aug_raw = train_raw.pop("augmentation", {})
        aug_cfg = AugmentationConfig(**aug_raw)
        # 채널 필드가 없던 구형 프로젝트의 과거 기본값을 보존한다.
        train_raw.setdefault("in_channels", 3)
        train_cfg = TrainingConfig(**train_raw, augmentation=aug_cfg)

        # runs 복원
        runs = [RunRecord(**r) for r in raw.get("runs", [])]

        project = ProjectData(
            name=raw.get("name", "Untitled"),
            task=raw.get("task", "classify"),
            created=raw.get("created", ""),
            modified=raw.get("modified", ""),
            project_dir=base,
            data=data_cfg,
            model=model_cfg,
            training=train_cfg,
            runs=runs,
            defect_generation=raw.get("defect_generation", {}),
        )
        project._active_filepath = filepath
        return project

    @staticmethod
    def new_run_id(
        task: str = "",
        model_name: str = "",
        input_size: int = 0,
        *,
        project_dir: str = "",
    ) -> str:
        """초 단위 폴더명 생성. 실제 저장 시 폴더를 예약하고 중복에만 순번 추가.

        예: Classification_efficientnetb0_224_260907_10h47m05s
        project_dir 생략 시 폴더를 만들지 않고 기본 이름만 반환한다.
        """
        now = datetime.now()
        if not task:
            base_name = f"run_{now.strftime('%Y%m%d_%H%M%S')}"
        else:
            task_labels = {
                "classify": "Classification",
                "segment": "Segmentation",
                "detect": "Detection",
                "obb": "OBB",
                "anomaly": "Anomaly",
            }
            task_label = task_labels.get(task, task.capitalize())
            task_label = re.sub(r"[^\w-]", "", task_label) or "Run"
            clean_name = os.path.splitext(model_name)[0] if model_name else "custom"
            clean_name = clean_name.replace("-", "").replace("_", "")
            clean_name = re.sub(r"[^\w-]", "", clean_name) or "custom"
            size_str = str(int(input_size)) if input_size else "224"
            time_str = now.strftime("%y%m%d_%Hh%Mm%Ss")
            base_name = f"{task_label}_{clean_name}_{size_str}_{time_str}"

        if not project_dir:
            return base_name
        runs_dir = os.path.join(project_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        candidate = base_name
        sequence = 1
        while True:
            try:
                # mkdir의 원자성을 사용해 다른 프로세스의 동시 저장도 보호한다.
                os.mkdir(os.path.join(runs_dir, candidate))
                return candidate
            except FileExistsError:
                sequence += 1
                candidate = f"{base_name}_{sequence}"

    @staticmethod
    def list_recent(base_dir: str = ".", limit: int = 10) -> List[str]:
        """최근 프로젝트 파일 검색"""
        projects = []
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.endswith(ProjectManager.FILE_EXTENSION):
                    projects.append(os.path.join(root, f))
        # 수정일 기준 정렬
        projects.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return projects[:limit]
