"""학습 엔진과 태스크별 지원 옵션의 공통 계약."""

import json
from pathlib import Path
MODE_LABELS = {
    "efficientnet_finetune": "EfficientNet 사전학습 모델로 시작",
    "efficientnet_transfer": "EfficientNet 내 가중치로 추가 학습",
    "efficientnet_resume": "EfficientNet 중단한 학습 재개",
    "efficientnet_scratch": "EfficientNet 무작위 초기화로 시작",
    "builtin_finetune": "선택 모델의 ImageNet 가중치로 시작",
    "builtin_transfer": "선택 모델의 로컬 가중치로 시작",
    "builtin_scratch": "선택 모델을 무작위 초기화로 시작",
    "upstream_finetune": "선택 모델의 사전학습 가중치로 시작",
    "upstream_transfer": "선택 모델의 로컬 가중치로 시작",
    "upstream_resume": "선택 모델의 중단한 학습 재개",
    "upstream_scratch": "선택 모델을 무작위 초기화로 시작",
    "sam2_finetune": "SAM2 기본 제공 가중치로 프롬프트 마스크 미세조정",
    "sam2_transfer": "SAM2 로컬 미세조정 체크포인트로 시작",
    "custom": "Custom CSP",
}
MODE_HELP = {
    "efficientnet_finetune": "B0/B1에 ImageNet 가중치를 검증 후 로드하고 내 클래스 분류기를 학습합니다.",
    "efficientnet_transfer": "로컬 가중치에서 새 학습을 시작합니다. 클래스가 다르면 분류기를 교체합니다.",
    "efficientnet_resume": "중단한 학습의 모델, optimizer, 스케줄러와 난수 상태를 복원합니다.",
    "efficientnet_scratch": "선택한 EfficientNet B0/B1 구조를 무작위 초기화하고 처음부터 학습합니다.",
    "builtin_finetune": "선택한 모델의 설치본 ImageNet 가중치를 검증 후 로드합니다. 분할 모델은 백본만 사전학습되어 있고 분할 헤드는 새로 학습합니다.",
    "builtin_transfer": "같은 모델의 Deep Vision Studio 체크포인트 또는 해당 torchvision 백본의 .pth를 선택하세요. 새 데이터의 클래스로 학습을 시작합니다.",
    "builtin_scratch": "선택한 기본 모델 구조를 무작위 초기화하고 처음부터 학습합니다.",
    "upstream_finetune": "선택한 기본 제공 모델의 공식 사전학습 가중치로 시작합니다.",
    "upstream_transfer": "같은 모델군의 로컬 checkpoint를 사용해 새 데이터로 학습합니다.",
    "upstream_resume": "같은 모델군의 last.pt를 복원해 학습을 이어갑니다.",
    "upstream_scratch": "선택한 모델의 동일한 upstream 구조를 무작위 초기화하고 처음부터 학습합니다.",
    "sam2_finetune": "설치본에 포함된 SAM2.1 가중치에서 시작합니다. semantic mask의 각 전경 클래스를 양성 점과 이진 객체 마스크로 바꿔 prompt encoder와 mask decoder를 학습합니다.",
    "sam2_transfer": "같은 SAM2 Hiera 변형의 Deep Vision Studio SAM2 미세조정 체크포인트(.pt)를 선택해 새 데이터로 이어서 학습합니다.",
    "custom": "자체 CSP 구조. 초기 가중치가 없으면 무작위 초기화합니다.",
}

BUILTIN_ADAPTER_IDS = frozenset({
    "resnet18", "resnet50", "convnext_v1_tiny",
    "deeplabv3plus_resnet34", "unet_resnet18",
})

UPSTREAM_ADAPTER_IDS = frozenset({
    "libreyolo_classify_mobilenetv4_small", "libreyolo_detect_9t",
    "re_detr_v4_small", "re_detr_v4_medium", "re_detr_v4_large",
})

SAM2_ADAPTER_IDS = frozenset({
    "sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus",
    "sam2_hiera_large",
})


def training_engine_name(mode):
    if mode not in MODE_LABELS:
        raise ValueError("지원하지 않는 학습 모드입니다. 현재 엔진을 선택하세요.")
    if str(mode).startswith("efficientnet"):
        return "efficientnet"
    if str(mode).startswith("builtin"):
        return "builtin"
    if str(mode).startswith("sam2"):
        return "sam2"
    return "upstream" if str(mode).startswith("upstream") else "custom"


def training_capabilities(task, mode, anomaly_method="patchcore", *, model_id=""):
    if task == "obb":
        raise ValueError("회전 박스 데이터 편집은 지원하지만 OBB 학습 엔진은 제공하지 않습니다.")
    if task not in {"classify", "segment", "detect", "anomaly"}:
        raise ValueError("태스크 오류")
    engine = "custom" if task == "anomaly" else training_engine_name(mode)
    if engine == "builtin" and model_id not in BUILTIN_ADAPTER_IDS:
        raise ValueError("선택한 모델은 기본 모델의 사전학습 모드를 지원하지 않습니다.")
    if engine == "upstream" and model_id not in UPSTREAM_ADAPTER_IDS:
        raise ValueError("선택한 모델은 upstream native 학습 모드를 지원하지 않습니다.")
    if engine == "sam2" and model_id not in SAM2_ADAPTER_IDS:
        raise ValueError("선택한 모델은 SAM2 prompt 미세조정 모드를 지원하지 않습니다.")
    if engine == "sam2" and task != "segment":
        raise ValueError("SAM2는 분할 태스크만 지원합니다.")
    if engine == "efficientnet" and task != "classify":
        raise ValueError("EfficientNet은 분류 태스크만 지원")
    patchcore = task == "anomaly" and anomaly_method == "patchcore"
    # These adapters run train_builtin, which does not install observation hooks.
    separate_adapter = model_id in BUILTIN_ADAPTER_IDS | UPSTREAM_ADAPTER_IDS | SAM2_ADAPTER_IDS
    return {
        "engine": engine,
        "models": ["efficientnet_b0", "efficientnet_b1"] if engine == "efficientnet" else [],
        "resume": mode.endswith("_resume") and task != "anomaly",
        "input_channels": [3] if engine == "sam2" else [1, 3],
        "augmentation": ["horizontal_flip", "rotation", "color_jitter"] if not patchcore and task == "classify" else [],
        "layer_debug": not patchcore and not separate_adapter,
        "layer_debug_reason": "레이어 관찰은 현재 EfficientNet과 Custom CSP에서 지원합니다. 선택한 모델의 학습 경로에는 연결되어 있지 않습니다.",
        "default_layer_patterns": "features.0,features.1.*,classifier.1" if engine == "efficientnet" else "backbone.stem,backbone.stage1,backbone.stage4",
    }


def validate_training_options(project, *, require_runnable=True):
    cfg = project.training
    if project.task == "obb" and not require_runnable:
        capabilities = {
            "engine": "custom",
            "models": [],
            "resume": False,
            "input_channels": [1, 3],
            "augmentation": [],
            "layer_debug": False,
            "layer_debug_reason": "OBB 학습 엔진은 제공하지 않습니다.",
            "default_layer_patterns": "",
        }
    else:
        capabilities = training_capabilities(project.task, cfg.training_mode, cfg.anomaly_method,
                                             model_id=getattr(project.model, "model_id", ""))
    model_id = getattr(project.model, "model_id", "")
    if cfg.training_mode in {"builtin_transfer", "upstream_transfer", "upstream_resume", "sam2_transfer"} and require_runnable:
        source = getattr(project.model, "pretrained_weights", "")
        if not source or not Path(source).is_file():
            raise ValueError("선택 모델의 로컬 가중치 파일(.pt/.pth)이 필요합니다.")
    if project.task == "anomaly" and cfg.anomaly_method == "patchcore" and model_id:
        expected_backbones = {
            "patchcore_wide_resnet50_2": "wide_resnet50_2",
            "patchcore_resnet18": "resnet18",
        }
        expected = expected_backbones.get(model_id)
        if expected is not None and cfg.patchcore_backbone != expected:
            raise ValueError(
                f"선택한 PatchCore 모델 {model_id}의 백본은 {expected}이지만 "
                f"학습 설정은 {cfg.patchcore_backbone}입니다. 모델과 백본을 동일하게 선택하세요."
            )
    if model_id:
        # Built-in adapters keep their input/task contract
        # in one registry.  Container packs validate their own contract after
        # installation, so an unknown legacy/pack id remains loadable here.
        try:
            from core.model_registry import registry_with_installed_packs
            registry, _ = registry_with_installed_packs()
            spec = registry.get(model_id)
        except (ImportError, KeyError, ValueError):
            spec = None
        if spec is not None and not (project.task == "anomaly" and
                                     cfg.anomaly_method == "patchcore"):
            if spec.task != project.task:
                raise ValueError(f"선택한 모델 {model_id}은 {project.task} 태스크와 맞지 않습니다.")
            if cfg.in_channels not in spec.input_channels:
                raise ValueError(f"선택한 모델 {model_id}은 입력 채널 {cfg.in_channels}을 지원하지 않습니다.")
            # Container-only catalog entries must never fall through to the
            # legacy Custom CSP engine.  The installed pack is deliberately a
            # project setting so a saved project can be reproduced offline.
            if require_runnable and "container" in spec.runtimes and "windows_native" not in spec.runtimes:
                pack_path = getattr(project.model, "pack_path", "")
                if not isinstance(pack_path, str) or not pack_path.strip():
                    raise ValueError(
                        f"{spec.display_name}은 설치된 Docker 모델 팩에서만 학습할 수 있습니다. "
                        "모델 팩을 설치하고 pack_path를 지정하세요."
                    )
                pack_root = Path(pack_path).expanduser()
                if not pack_root.is_absolute() or pack_root.is_symlink() or not pack_root.is_dir():
                    raise ValueError("모델 팩 경로는 검증된 절대 디렉터리여야 합니다.")
                try:
                    pack_root = pack_root.resolve(strict=True)
                    manifest_path = pack_root / "manifest.json"
                    if manifest_path.is_symlink() or not manifest_path.is_file():
                        raise ValueError("설치된 모델 팩의 manifest.json이 없습니다.")
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError("설치된 모델 팩 manifest.json을 읽을 수 없습니다.") from exc
                if not isinstance(manifest, dict) or manifest.get("model_id") != model_id:
                    raise ValueError("프로젝트 모델 ID와 설치된 모델 팩이 일치하지 않습니다.")
    if type(cfg.efficientnet_no_decay) is not bool:
        raise ValueError("EfficientNet weight decay 제외 옵션은 boolean 필요")
    if capabilities["engine"] == "efficientnet" and cfg.efficientnet_model not in capabilities["models"]:
        raise ValueError("EfficientNet B0 또는 B1 선택 필요")
    if type(cfg.layer_debug_enabled) is not bool:
        raise ValueError("레이어 관찰 활성화는 boolean 필요")
    if type(cfg.layer_debug_batches) is not int or not 1 <= cfg.layer_debug_batches <= 10:
        raise ValueError("레이어 관찰 배치는 1~10 정수 필요")
    if not isinstance(cfg.layer_debug_patterns, str) or len(cfg.layer_debug_patterns) > 1024:
        raise ValueError("레이어 패턴은 1024자 이하 문자열 필요")
    patterns = [part.strip() for part in cfg.layer_debug_patterns.split(",") if part.strip()]
    if cfg.layer_debug_enabled:
        if not capabilities["layer_debug"]:
            raise ValueError(capabilities["layer_debug_reason"])
        if not patterns or len(patterns) > 16:
            raise ValueError("레이어 이름 패턴을 1~16개 입력하세요")
    if not capabilities["resume"] and not (project.task == "anomaly" and cfg.anomaly_method == "patchcore"):
        if cfg.augmentation.vertical_flip or cfg.augmentation.mixup_alpha:
            raise ValueError("현재 학습 엔진의 수직 반전과 Mixup은 미지원입니다. 저장된 두 값을 0으로 변경하세요.")
    return capabilities
