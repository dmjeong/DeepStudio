"""모델 팩과 기본 모델을 하나의 선언형 등록부로 관리한다.

이 모듈은 Qt, PyTorch, Docker에 의존하지 않는다. 따라서 데스크톱 UI, 로컬 웹 API,
Windows worker와 나중에 추가할 Docker worker가 같은 모델 계약을 읽을 수 있다.

등록부에 모델을 추가하는 것과 해당 모델의 학습/ONNX/C#/C++ 검증을 완료하는 것은
별개의 상태다. 검증되지 않은 모델을 화면에서 사용할 수 있다고 표시하지 않도록
``release_status``를 명시적으로 보관한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import zipfile

from model_runtime.special_contracts import validate_special_assets, validate_special_manifest


MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,79}$")
PACK_SUFFIX = ".dvmodel"
SUPPORTED_TASKS = frozenset({"classify", "anomaly", "detect", "segment"})
SUPPORTED_RUNTIMES = frozenset({"windows_native", "onnx", "container"})
SUPPORTED_STATUSES = frozenset({
    "requested", "scoped", "runtime_verified", "trained_verified",
    "export_verified", "sdk_verified", "release_ready",
})


class ModelRegistryError(ValueError):
    """모델 정의나 모델 팩이 계약을 위반할 때 발생하는 오류."""


@dataclass(frozen=True)
class ModelSpec:
    """모델 하나의 UI/학습/배포 계약을 설명하는 불변 정의."""

    model_id: str
    family: str
    variant: str
    task: str
    runtimes: tuple[str, ...]
    capabilities: frozenset[str]
    input_size: tuple[int, int]
    input_channels: tuple[int, ...] = (3,)
    release_status: str = "requested"
    pretrained_asset: str = ""
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_model_spec(self)

    @property
    def display_name(self) -> str:
        return f"{self.family} {self.variant}".strip()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["runtimes"] = list(self.runtimes)
        value["capabilities"] = sorted(self.capabilities)
        value["input_size"] = list(self.input_size)
        value["input_channels"] = list(self.input_channels)
        return value


def _positive_pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ModelRegistryError(f"{name} must contain [height, width]")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ModelRegistryError(f"{name} must contain positive integers")
    return int(value[0]), int(value[1])


def validate_model_spec(spec: ModelSpec) -> None:
    if not isinstance(spec.model_id, str) or not MODEL_ID_RE.fullmatch(spec.model_id):
        raise ModelRegistryError(f"invalid model id: {spec.model_id!r}")
    if spec.task not in SUPPORTED_TASKS:
        raise ModelRegistryError(f"unsupported task: {spec.task}")
    if not spec.runtimes or any(runtime not in SUPPORTED_RUNTIMES for runtime in spec.runtimes):
        raise ModelRegistryError(f"unsupported runtime in {spec.model_id}")
    if not spec.capabilities or any(not isinstance(item, str) or not item for item in spec.capabilities):
        raise ModelRegistryError(f"capabilities required for {spec.model_id}")
    _positive_pair(spec.input_size, "input_size")
    if not spec.input_channels or any(channel not in (1, 3) for channel in spec.input_channels):
        raise ModelRegistryError(f"input_channels must contain 1 or 3: {spec.model_id}")
    if spec.release_status not in SUPPORTED_STATUSES:
        raise ModelRegistryError(f"unsupported release status: {spec.release_status}")
    if spec.family in {"Re-DETR v4", "SAM2"}:
        try:
            manifest = spec.to_dict()
            manifest.update(spec.metadata)
            validate_special_manifest(manifest)
        except ValueError as exc:
            raise ModelRegistryError(str(exc)) from exc


def builtin_model_specs() -> tuple[ModelSpec, ...]:
    """사용자가 요구한 기본 모델군의 등록 목록.

    ``requested`` 상태인 항목은 아직 실제 Windows 학습/내보내기 검증 전이다. 이 목록을
    등록해 두면 구현 중인 모델도 프로젝트/팩 참조로 안정적으로 식별할 수 있지만,
    ``ModelRegistry.available``는 release-ready 모델만 반환한다.
    """

    common = frozenset({"train", "infer", "export_onnx", "csharp", "cpp"})
    specs: list[ModelSpec] = [
        ModelSpec("efficientnet_b0", "EfficientNet", "B0", "classify", ("windows_native", "onnx"), common, (224, 224), (1, 3), "sdk_verified"),
        ModelSpec("efficientnet_b1", "EfficientNet", "B1", "classify", ("windows_native", "onnx"), common, (240, 240), (1, 3), "sdk_verified", notes="224x224는 별도 학습 프로필"),
        ModelSpec("resnet18", "ResNet", "18", "classify", ("windows_native", "onnx"), common, (224, 224), (1, 3), "export_verified", notes="weight-free native adapter"),
        ModelSpec("resnet50", "ResNet", "50", "classify", ("windows_native", "onnx"), common, (224, 224), (1, 3), "export_verified", notes="weight-free native adapter"),
        ModelSpec("convnext_v1_tiny", "ConvNeXt V1", "Tiny", "classify", ("windows_native", "onnx"), common, (224, 224), (3,), "export_verified", notes="weight-free native adapter"),
        ModelSpec("libreyolo_classify_mobilenetv4_small", "LibreYOLO", "MobileNetV4 Small", "classify", ("container", "onnx"), common, (224, 224), (3,)),
        ModelSpec("patchcore_wide_resnet50_2", "PatchCore", "Wide-ResNet50-2", "anomaly", ("windows_native", "onnx"), frozenset({"fit", "infer", "export_onnx", "csharp", "cpp"}), (224, 224), (3,), notes="memory bank와 kNN 포함"),
        ModelSpec("patchcore_resnet18", "PatchCore", "ResNet18", "anomaly", ("windows_native", "onnx"), frozenset({"fit", "infer", "export_onnx", "csharp", "cpp"}), (224, 224), (3,), notes="memory bank와 kNN 포함"),
    ]
    for variant, size in (("small", 640), ("medium", 800), ("large", 1024)):
        specs.append(ModelSpec(
            f"re_detr_v4_{variant}", "Re-DETR v4", variant.title(), "detect", ("container", "onnx"), common,
            (size, size), (3,), notes="requested family fixed by product requirement",
            metadata={"contracts": {"onnx": {"input_name": "input_image", "boxes_name": "pred_boxes",
                                                "logits_name": "pred_logits", "boxes_format": "normalized_cxcywh",
                                                "score_activation": "sigmoid"}}}))
    specs.extend([
        ModelSpec("libreyolo_detect_9t", "LibreYOLO", "9 Tiny", "detect", ("container", "onnx"), common, (640, 640), (3,)),
        ModelSpec("sam2_hiera_tiny", "SAM2", "Hiera Tiny", "segment", ("container", "onnx"), frozenset({"train", "infer", "export_onnx", "csharp", "cpp", "prompt", "video"}), (1024, 1024), (3,), notes="image/prompt/video 계약은 별도 검증", metadata={"contracts": {"graphs": {"encoder": {"file": "sam2_encoder.onnx", "outputs": ["image_embeddings"]}, "decoder": {"file": "sam2_decoder.onnx", "outputs": ["low_res_mask_logits", "iou_predictions"]}}, "prompt_types": ["point", "box", "mask"], "video_state": True}}),
        ModelSpec("sam2_hiera_small", "SAM2", "Hiera Small", "segment", ("container", "onnx"), frozenset({"train", "infer", "export_onnx", "csharp", "cpp", "prompt", "video"}), (1024, 1024), (3,), notes="image/prompt/video 계약은 별도 검증", metadata={"contracts": {"graphs": {"encoder": {"file": "sam2_encoder.onnx", "outputs": ["image_embeddings"]}, "decoder": {"file": "sam2_decoder.onnx", "outputs": ["low_res_mask_logits", "iou_predictions"]}}, "prompt_types": ["point", "box", "mask"], "video_state": True}}),
        ModelSpec("sam2_hiera_base_plus", "SAM2", "Hiera Base+", "segment", ("container", "onnx"), frozenset({"train", "infer", "export_onnx", "csharp", "cpp", "prompt", "video"}), (1024, 1024), (3,), notes="image/prompt/video 계약은 별도 검증", metadata={"contracts": {"graphs": {"encoder": {"file": "sam2_encoder.onnx", "outputs": ["image_embeddings"]}, "decoder": {"file": "sam2_decoder.onnx", "outputs": ["low_res_mask_logits", "iou_predictions"]}}, "prompt_types": ["point", "box", "mask"], "video_state": True}}),
        ModelSpec("sam2_hiera_large", "SAM2", "Hiera Large", "segment", ("container", "onnx"), frozenset({"train", "infer", "export_onnx", "csharp", "cpp", "prompt", "video"}), (1024, 1024), (3,), notes="image/prompt/video 계약은 별도 검증", metadata={"contracts": {"graphs": {"encoder": {"file": "sam2_encoder.onnx", "outputs": ["image_embeddings"]}, "decoder": {"file": "sam2_decoder.onnx", "outputs": ["low_res_mask_logits", "iou_predictions"]}}, "prompt_types": ["point", "box", "mask"], "video_state": True}}),
        ModelSpec("deeplabv3plus_resnet34", "DeepLab V3+", "ResNet34", "segment", ("windows_native", "onnx"), common, (512, 512), (3,), "export_verified", notes="weight-free native adapter"),
        ModelSpec("unet_resnet18", "U-Net", "ResNet18", "segment", ("windows_native", "onnx"), common, (512, 512), (3,), "export_verified", notes="weight-free native adapter"),
    ])
    return tuple(specs)


def _spec_from_mapping(data: Mapping[str, Any]) -> ModelSpec:
    if not isinstance(data, Mapping):
        raise ModelRegistryError("manifest must be a JSON object")
    if data.get("schema_version") != 1:
        raise ModelRegistryError("unsupported or missing manifest schema_version")
    required = {"model_id", "family", "variant", "task", "runtimes", "capabilities", "input_size"}
    missing = required - set(data)
    if missing:
        raise ModelRegistryError("manifest missing fields: " + ", ".join(sorted(missing)))
    metadata = dict(data.get("metadata", {}))
    for key in ("contracts", "runtime_requirements"):
        if key in data:
            metadata[key] = data[key]
    return ModelSpec(
        model_id=data["model_id"], family=data["family"], variant=data["variant"], task=data["task"],
        runtimes=tuple(data["runtimes"]), capabilities=frozenset(data["capabilities"]),
        input_size=_positive_pair(data["input_size"], "input_size"),
        input_channels=tuple(data.get("input_channels", (3,))),
        release_status=data.get("release_status", "requested"),
        pretrained_asset=data.get("pretrained_asset", ""), notes=data.get("notes", ""),
        metadata=metadata,
    )


class ModelRegistry:
    """기본 목록과 설치된 `.dvmodel` 팩을 조회하는 등록부."""

    def __init__(self, specs: Iterable[ModelSpec] = ()) -> None:
        self._models: dict[str, ModelSpec] = {}
        for spec in specs:
            self.register(spec)

    @classmethod
    def builtin(cls) -> "ModelRegistry":
        return cls(builtin_model_specs())

    def register(self, spec: ModelSpec) -> None:
        validate_model_spec(spec)
        if spec.model_id in self._models:
            raise ModelRegistryError(f"duplicate model id: {spec.model_id}")
        self._models[spec.model_id] = spec

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ModelRegistryError(f"unknown model: {model_id}") from exc

    def list(self, task: str | None = None, *, release_ready: bool = False) -> tuple[ModelSpec, ...]:
        values = tuple(self._models.values())
        if task is not None:
            values = tuple(spec for spec in values if spec.task == task)
        if release_ready:
            values = tuple(spec for spec in values if spec.release_status == "release_ready")
        return values

    def available(self, task: str | None = None) -> tuple[ModelSpec, ...]:
        """실제 출시 표시가 허용된 모델만 반환한다."""
        return self.list(task, release_ready=True)

    def as_dict(self, task: str | None = None) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self.list(task)]

    def load_pack(self, path: str | Path) -> ModelSpec:
        """팩을 staging 없이 실행하지 않고, manifest만 먼저 엄격히 읽는다."""
        pack = Path(path)
        if pack.suffix.lower() != PACK_SUFFIX or not pack.is_file():
            raise ModelRegistryError(f"model pack must be a .dvmodel file: {pack}")
        try:
            with zipfile.ZipFile(pack) as archive:
                names = archive.namelist()
                for name in names:
                    candidate = Path(name)
                    if candidate.is_absolute() or ".." in candidate.parts:
                        raise ModelRegistryError(f"unsafe model pack path: {name}")
                if "manifest.json" not in names:
                    raise ModelRegistryError("model pack manifest.json missing")
                manifest = json.loads(archive.read("manifest.json"))
                try:
                    validate_special_assets(manifest, set(names))
                except ValueError as exc:
                    raise ModelRegistryError(str(exc)) from exc
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelRegistryError(f"cannot read model pack: {pack}") from exc
        spec = _spec_from_mapping(manifest)
        self.register(spec)
        return spec
