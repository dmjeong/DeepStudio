"""PatchCore 백본의 사전학습 및 로컬 전이 가중치를 엄격하게 검증한다."""

import hashlib
from pathlib import Path


BACKBONES = {
    "resnet18": (128, 256, "ResNet18_Weights"),
    "resnet50": (512, 1024, "ResNet50_Weights"),
    "wide_resnet50_2": (512, 1024, "Wide_ResNet50_2_Weights"),
}


def build_backbone(name, pretrained):
    if name not in BACKBONES:
        raise ValueError(f"PatchCore 미지원 백본: {name}")
    import torch
    import torchvision.models as models
    from builtin_assets import builtin_asset_path
    weights = getattr(models, BACKBONES[name][2]).IMAGENET1K_V1 if pretrained else None
    try:
        if weights is not None:
            path = builtin_asset_path(name)
            model = getattr(models, name)(weights=None)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            return model
        return getattr(models, name)(weights=None)
    except Exception as exc:
        if not pretrained:
            raise
        raise RuntimeError(
            f"{name} 기본 제공 ImageNet 가중치 로드 실패: {exc}\n"
            "앱은 실행 중 가중치를 다운로드하지 않습니다. build.bat으로 설치본을 다시 만드세요."
        ) from exc


def load_backbone_weights(backbone, path, name):
    import torch
    from torch import nn
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"전이할 백본 가중치 없음: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, dict):
        declared = payload.get("backbone_name")
        if declared and declared != name:
            raise ValueError(f"백본 불일치: 선택 {name}, 가중치 {declared}")
        for key in ("backbone_state_dict", "state_dict", "model_state_dict", "model"):
            if key in payload:
                state = payload[key]
                break
    if isinstance(state, nn.Module):
        state = state.state_dict()
    if not isinstance(state, dict):
        raise ValueError("지원하는 형식: torchvision ResNet state_dict 또는 PatchCore 백본 가중치")
    expected = backbone.state_dict()
    loaded = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            continue
        normalized = key
        while normalized.startswith(("module.", "backbone.", "encoder.", "model.")):
            normalized = normalized.split(".", 1)[1]
        if normalized not in expected:
            continue  # 사용하지 않는 layer4와 분류 헤드는 전이하지 않는다.
        if normalized in loaded:
            raise ValueError(f"중복 백본 가중치 키: {normalized}")
        if tensor.shape != expected[normalized].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"백본 가중치 크기/값 불일치: {normalized}")
        loaded[normalized] = tensor
    missing = sorted(set(expected) - set(loaded))
    if missing:
        raise ValueError(f"{name} 백본 가중치 누락 {len(missing)}개: {', '.join(missing[:5])}\n"
                         "같은 torchvision ResNet 계열 가중치 필요. 다른 구조의 가중치는 이 백본과 호환되지 않습니다")
    backbone.load_state_dict(loaded, strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"kind": "local_backbone", "name": path.name, "sha256": digest.hexdigest(),
            "backbone": name, "loaded_tensors": len(loaded)}
