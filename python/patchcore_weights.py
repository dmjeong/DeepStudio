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
    from model_download import cached_imagenet_weights, certificate_failure
    weights = getattr(models, BACKBONES[name][2]).IMAGENET1K_V1 if pretrained else None
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    try:
        if weights is not None:
            cached_imagenet_weights(weights.url, cache)
        return getattr(models, name)(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        if certificate_failure(exc):
            raise RuntimeError(
                f"{name} ImageNet 다운로드 인증서 검증 실패: {exc}\n"
                f"다운로드 URL: {weights.url}\n가중치 캐시: {cache}\n"
                "Windows 인증서 저장소와 공개 CA 목록으로도 인증서를 확인하지 못했습니다. "
                "PC 날짜/시간과 회사 보안망의 루트 인증서 등록 상태를 확인하세요. "
                "관리자가 제공한 PEM 인증서는 SSL_CERT_FILE 환경 변수로 지정할 수 있습니다.\n"
                "브라우저에서 위 URL을 다운로드할 수 있다면 '로컬 ResNet 백본 가중치'에서 "
                "같은 모델의 .pth 파일을 선택하세요."
            ) from exc
        raise RuntimeError(
            f"{name} ImageNet 가중치 로드 실패: {exc}\n"
            f"가중치 캐시: {cache}\n인터넷/인증서 또는 캐시 파일 상태 확인 필요. "
            "다운로드가 불가능하면 '로컬 백본 가중치'에서 같은 모델의 .pth 파일을 선택하세요."
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
