"""Locations and integrity checks for the pretrained weights used by the app.

The desktop application never downloads in the background.  ``build.bat``
tries to bundle every weight; any file it could not fetch can be added later
without rebuilding:

    설치본 묶음 (build.bat, manifest 해시)      ─┐
    사용자 가중치 폴더 (직접 받은 파일, 가져오기) ─┼─▶ builtin_asset_path(model_id)
    torch hub 캐시 (torchvision이 받아 둔 파일)  ─┘

A manifest entry, when present, must match.  Without one, torchvision files are
checked against the SHA-256 prefix in their official name and other files
against the PyTorch archive format; loading checks the structure again.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


ASSET_FILES = {
    "efficientnet_b0": "torchvision/efficientnet_b0_rwightman-7f5810bc.pth",
    "efficientnet_b1": "torchvision/efficientnet_b1-c27df63c.pth",
    "resnet18": "torchvision/resnet18-f37072fd.pth",
    "resnet34": "torchvision/resnet34-b627a593.pth",
    "resnet50": "torchvision/resnet50-0676ba61.pth",
    "wide_resnet50_2": "torchvision/wide_resnet50_2-95faca4d.pth",
    "convnext_v1_tiny": "torchvision/convnext_tiny-983f1562.pth",
    "dinov2_vits14": "dinov2/dinov2_vits14_pretrain.pth",
    "dinov2_vitb14": "dinov2/dinov2_vitb14_pretrain.pth",
    "libreyolo_classify_mobilenetv4_small": "libreyolo/LibreMobileNetV4s-cls.pt",
    "libreyolo_detect_9t": "libreyolo/LibreYOLO9t.pt",
    "re_detr_v4_small": "libreyolo/LibreRTDETRv4s.pt",
    "re_detr_v4_medium": "libreyolo/LibreRTDETRv4m.pt",
    "re_detr_v4_large": "libreyolo/LibreRTDETRv4l.pt",
    "sam2_hiera_tiny": "sam2/sam2.1_hiera_tiny.pt",
    "sam2_hiera_small": "sam2/sam2.1_hiera_small.pt",
    "sam2_hiera_base_plus": "sam2/sam2.1_hiera_base_plus.pt",
    "sam2_hiera_large": "sam2/sam2.1_hiera_large.pt",
}


# 수동 다운로드 주소 (build.bat의 prepare_builtin_assets.py가 받는 것과 같은 공식 파일)
SOURCE_URLS = {
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "convnext_v1_tiny": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "libreyolo_classify_mobilenetv4_small":
        "https://huggingface.co/LibreYOLO/LibreMobileNetV4s-cls/resolve/main/LibreMobileNetV4s-cls.pt",
    "libreyolo_detect_9t": "https://huggingface.co/LibreYOLO/LibreYOLO9t/resolve/main/LibreYOLO9t.pt",
    "re_detr_v4_small": "https://huggingface.co/LibreYOLO/LibreRTDETRv4s/resolve/main/LibreRTDETRv4s.pt",
    "re_detr_v4_medium": "https://huggingface.co/LibreYOLO/LibreRTDETRv4m/resolve/main/LibreRTDETRv4m.pt",
    "re_detr_v4_large": "https://huggingface.co/LibreYOLO/LibreRTDETRv4l/resolve/main/LibreRTDETRv4l.pt",
    "sam2_hiera_tiny": "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt",
    "sam2_hiera_small": "https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt",
    "sam2_hiera_base_plus":
        "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt",
    "sam2_hiera_large": "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt",
}


# 공식 파일 이름에 해시가 없는 가중치의 전체 SHA-256 (공식 주소에서 받은 파일, 2026-10 확인).
# manifest가 없어도(직접 받은 파일) 이 값과 같아야 쓴다.
PINNED_SHA256 = {
    "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    "dinov2_vitb14_pretrain.pth": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
}


def builtin_asset_root() -> Path:
    """Return the immutable bundle path in both development and frozen mode."""
    override = os.environ.get("DVS_BUILTIN_ASSETS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "builtin_assets"
    return Path(__file__).resolve().parents[1] / "gui" / "builtin_assets"


def user_asset_root() -> Path:
    """사용자가 직접 받은 가중치를 두는 쓰기 가능한 폴더 (설치본 폴더는 바꾸지 않는다).

    Windows: %LOCALAPPDATA%\\DeepVisionStudio\\weights, 그 외: ~/.local/share/DeepVisionStudio/weights.
    DVS_USER_WEIGHTS_DIR로 바꿀 수 있다.
    """
    override = os.environ.get("DVS_USER_WEIGHTS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else ""
    base = Path(base) if base else Path.home() / ".local" / "share"
    return (base / "DeepVisionStudio" / "weights").resolve()


def _torch_hub_dir() -> Path:
    """torchvision이 받아 둔 파일(~/.cache/torch/hub/checkpoints)도 그대로 쓴다."""
    home = os.environ.get("TORCH_HOME", "").strip()
    return (Path(home).expanduser() if home else Path.home() / ".cache" / "torch") / "hub" / "checkpoints"


def _candidates(model_id: str):
    """찾는 순서: 설치본 묶음 → 사용자 가중치 폴더(하위 폴더 또는 파일만) → torch hub 캐시."""
    relative = ASSET_FILES[model_id]
    name = Path(relative).name
    user = user_asset_root()
    found = [(builtin_asset_root(), relative), (user, relative), (user, name)]
    if relative.startswith(("torchvision/", "dinov2/")):
        found.append((_torch_hub_dir(), name))  # torch.hub.load('facebookresearch/dinov2')도 같은 곳에 받는다
    return found


def _manifest_entry(root: Path, relative: str):
    """manifest 항목(없으면 None). manifest는 빌드·가져오기 때 기록한 해시이며 없어도 된다."""
    try:
        value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    files = value.get("files") if isinstance(value, dict) else None
    entry = files.get(relative) if isinstance(files, dict) else None
    return entry if isinstance(entry, dict) and isinstance(entry.get("sha256"), str) else None


def _sha256(path: Path) -> str:
    digestor = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digestor.update(block)
    return digestor.hexdigest()


_VERIFIED: dict = {}  # (경로, 크기, 수정 시각) → 이미 검증함 (큰 SAM2 파일을 매번 해시하지 않는다)


def _check_file(path: Path, expected_name: str) -> None:
    """manifest 없이 할 수 있는 확인.

    torchvision 파일 이름의 해시(resnet18-f37072fd.pth의 f37072fd)는 SHA-256 앞자리다 → 비교.
    이름에 해시가 없는 DINOv2는 고정한 전체 SHA-256(PINNED_SHA256)과 비교한다.
    그 밖(LibreYOLO·SAM2)은 PyTorch 저장 형식(zip)인지 본다. 구조는 불러올 때 strict로 다시 검사한다.
    """
    pinned = PINNED_SHA256.get(expected_name)
    if pinned is not None:
        if _sha256(path) != pinned:
            raise RuntimeError(f"{path.name}은(는) 공식 {expected_name}과 다른 파일입니다(해시 불일치). "
                               "다운로드가 끊겼거나 다른 모델 파일입니다. 다시 받아 주세요.")
        return
    match = re.search(r"-([0-9a-f]{8,})\.pth$", expected_name)
    if match:
        if not _sha256(path).startswith(match.group(1)):
            raise RuntimeError(f"{path.name}은(는) {expected_name}과 다른 파일입니다(해시 불일치). "
                               "다운로드가 끊겼거나 다른 모델 파일입니다. 다시 받아 주세요.")
        return
    with path.open("rb") as source:
        if source.read(4) != b"PK\x03\x04":
            raise RuntimeError(f"{path.name}은(는) PyTorch 가중치 파일 형식이 아닙니다. "
                               "다운로드 페이지(HTML)를 저장했거나 받다가 끊긴 파일입니다.")


def _verify(path: Path, root: Path, relative: str, expected_name: str) -> None:
    """manifest 항목이 있으면 SHA-256이 같아야 하고(빌드한 묶음·가져온 파일), 없으면 파일 자체를 확인한다."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key in _VERIFIED:
        return
    entry = _manifest_entry(root, relative)
    if entry is not None:
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"기본 모델 가중치 무결성 검증 실패: {relative}. 파일을 다시 받아 주세요.")
    else:
        _check_file(path, expected_name)
    _VERIFIED[key] = True


def builtin_asset_path(model_id: str, *, verify: bool = True) -> Path:
    """사전학습 가중치 파일 경로. 설치본 → 사용자 가중치 폴더 → torch 캐시 순서로 찾는다."""
    try:
        relative = ASSET_FILES[model_id]
    except KeyError as exc:
        raise ValueError(f"기본 모델 가중치 ID가 아닙니다: {model_id}") from exc
    name = Path(relative).name
    errors = []
    for root, inside in _candidates(model_id):
        path = root / inside
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        if not verify:
            return path.resolve()
        try:
            _verify(path, root, inside, name)
        except (OSError, RuntimeError) as exc:
            errors.append(str(exc))
            continue
        return path.resolve()
    if errors:
        raise RuntimeError(errors[0])
    raise FileNotFoundError(
        f"{model_id} 사전학습 가중치({name})가 없습니다.\n"
        f"Settings · 모델 관리 → 사전학습 가중치에서 받은 파일을 가져오거나, 이 폴더에 넣으세요: {user_asset_root()}"
    )


def asset_status(model_id: str) -> tuple[str, Path | None, str]:
    """('ready' | 'missing' | 'invalid', 경로, 설명) — 설정 화면 표시용."""
    try:
        return "ready", builtin_asset_path(model_id), ""
    except FileNotFoundError:
        return "missing", None, ""
    except (OSError, RuntimeError, ValueError) as exc:
        return "invalid", None, str(exc)


def import_asset(model_id: str, source: str | Path) -> Path:
    """사용자가 직접 받은 파일을 검증해 사용자 가중치 폴더에 넣고 해시를 기록한다.

    파일 이름은 달라도 된다(브라우저가 '(1)'을 붙이는 경우). torchvision 파일은 해시 앞자리를 확인한다.
    """
    relative = ASSET_FILES[model_id]
    source = Path(source).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"가져올 파일이 없습니다: {source}")
    root = user_asset_root()
    target = root / relative
    _check_file(source, Path(relative).name)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".importing")
    shutil.copyfile(source, temporary)
    digest = _sha256(temporary)
    os.replace(temporary, target)
    manifest = root / "manifest.json"
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        files = value.get("files") if isinstance(value, dict) else None
        files = files if isinstance(files, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        files = {}
    files[relative] = {"sha256": digest, "bytes": target.stat().st_size}
    manifest.write_text(json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n", encoding="utf-8")
    return target.resolve()


# 기본 모델 → ImageNet 백본 가중치 (builtin_models._torchvision_base 와 같은 대응)
BUILTIN_BACKBONE_ASSETS = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "convnext_v1_tiny": "convnext_v1_tiny",
    "unet_resnet18": "resnet18",
    "deeplabv3plus_resnet34": "resnet34",
    # 모델 ID(…_cls 분류, …_seg 분할)는 main의 DINOv2 분할(dinov2_vits14)과 겹치지 않게, 가중치 ID는 같은 공식 파일이라 공유
    "dinov2_vits14_cls": "dinov2_vits14", "dinov2_vits14_seg": "dinov2_vits14",
    "dinov2_vitb14_cls": "dinov2_vitb14", "dinov2_vitb14_seg": "dinov2_vitb14",
}


def required_pretrained_asset(task: str, mode: str, model_id: str = "", *,
                              efficientnet_model: str = "", anomaly_method: str = "",
                              patchcore_source: str = "", patchcore_backbone: str = "") -> str | None:
    """이 학습 설정이 시작 전에 꼭 필요로 하는 설치본 가중치 ID (필요 없으면 None)."""
    if task == "anomaly":
        if anomaly_method == "patchcore" and patchcore_source == "imagenet":
            return patchcore_backbone if patchcore_backbone in ASSET_FILES else None
        return None
    if str(model_id).startswith("sam2_hiera_"):
        return model_id  # SAM2는 모든 모드에서 기본 가중치 구조를 쓴다
    if mode == "builtin_finetune":
        return BUILTIN_BACKBONE_ASSETS.get(model_id)
    if mode == "efficientnet_finetune":
        return efficientnet_model if efficientnet_model in ASSET_FILES else None
    if mode == "upstream_finetune":
        return model_id if model_id in ASSET_FILES else None
    return None


def missing_asset_help(asset_id: str) -> str:
    """가중치가 없을 때 무엇을 하면 되는지 (사용자 안내 문구). 다시 빌드하지 않아도 되는 방법이 먼저다."""
    status, _path, problem = asset_status(asset_id)
    name = Path(ASSET_FILES[asset_id]).name
    return (
        f"이 학습에 필요한 사전학습 가중치({asset_id})를 쓸 수 없습니다"
        + (f": {problem}" if status == "invalid" else ".") + "\n\n"
        "1) 브라우저로 이 파일을 받으세요 (회사망에서 막히면 다른 PC에서 받아 옮겨도 됩니다):\n"
        f"   {SOURCE_URLS[asset_id]}\n"
        "2) '가중치 가져오기'로 받은 파일을 고르거나, 이 폴더에 그대로 넣으세요:\n"
        f"   {user_asset_root()}  (파일 이름: {name})\n\n"
        "manifest는 필요 없습니다. 앱이 확인한 뒤 바로 사용합니다.\n"
        "소스에서 실행 중이면 이 모델만 자동으로 받을 수도 있습니다:\n"
        f"   python python/prepare_builtin_assets.py --output gui/builtin_assets --only {asset_id}\n\n"
        "또는 학습 모드를 '무작위 초기화로 시작'으로 바꾸면 가중치 없이 학습합니다(보통 정확도가 낮고 더 오래 걸립니다)."
    )


def missing_assets(*, bundle_only: bool = False) -> list[str]:
    """지금 쓸 수 없는 가중치 ID (설치본·사용자 폴더·torch 캐시 어디에도 없거나 손상).

    bundle_only=True면 설치본 묶음만 본다 (빌드: 빌드 PC의 사용자 폴더는 설치본에 들어가지 않는다).
    """
    if not bundle_only:
        return [model_id for model_id in ASSET_FILES if asset_status(model_id)[0] != "ready"]
    root, missing = builtin_asset_root(), []
    for model_id, relative in ASSET_FILES.items():
        path = root / relative
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            _verify(path, root, relative, Path(relative).name)
        except (OSError, RuntimeError):
            missing.append(model_id)
    return missing


def builtin_assets_ready() -> bool:
    return not missing_assets()
