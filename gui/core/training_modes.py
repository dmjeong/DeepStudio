"""학습 엔진과 태스크별 지원 옵션의 공통 계약."""
MODE_LABELS = {
    "efficientnet_finetune": "EfficientNet 사전학습 모델로 시작",
    "efficientnet_transfer": "EfficientNet 내 가중치로 추가 학습",
    "efficientnet_resume": "EfficientNet 중단한 학습 재개",
    "custom": "Custom CSP",
}
MODE_HELP = {
    "efficientnet_finetune": "B0/B1에 ImageNet 가중치를 검증 후 로드하고 내 클래스 분류기를 학습합니다.",
    "efficientnet_transfer": "로컬 가중치에서 새 학습을 시작합니다. 클래스가 다르면 분류기를 교체합니다.",
    "efficientnet_resume": "중단한 학습의 모델, optimizer, 스케줄러와 난수 상태를 복원합니다.",
    "custom": "자체 CSP 구조. 초기 가중치가 없으면 무작위 초기화합니다.",
}


def training_engine_name(mode):
    if mode not in MODE_LABELS:
        raise ValueError("지원하지 않는 학습 모드입니다. 현재 엔진을 선택하세요.")
    return "efficientnet" if str(mode).startswith("efficientnet") else "custom"


def training_capabilities(task, mode, anomaly_method="patchcore"):
    if task == "obb":
        raise ValueError("회전 박스 데이터 편집은 지원하지만 OBB 학습 엔진은 제공하지 않습니다.")
    if task not in {"classify", "segment", "detect", "anomaly"}:
        raise ValueError("태스크 오류")
    engine = "custom" if task == "anomaly" else training_engine_name(mode)
    if engine == "efficientnet" and task != "classify":
        raise ValueError("EfficientNet은 분류 태스크만 지원")
    patchcore = task == "anomaly" and anomaly_method == "patchcore"
    return {
        "engine": engine,
        "models": ["efficientnet_b0", "efficientnet_b1"] if engine == "efficientnet" else [],
        "resume": mode.endswith("_resume") and task != "anomaly",
        "input_channels": [1, 3],
        "augmentation": ["horizontal_flip", "rotation", "color_jitter"] if not patchcore and task == "classify" else [],
        "layer_debug": not patchcore,
        "layer_debug_reason": "레이어 관찰은 EfficientNet과 Custom CSP에서 지원합니다. PatchCore는 미지원입니다.",
        "default_layer_patterns": "features.0,features.1.*,classifier.1" if engine == "efficientnet" else "backbone.stem,backbone.stage1,backbone.stage4",
    }


def validate_training_options(project):
    cfg = project.training
    capabilities = training_capabilities(project.task, cfg.training_mode, cfg.anomaly_method)
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
