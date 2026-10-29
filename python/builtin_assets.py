"""Locations and integrity checks for the pretrained weights shipped with the app.

The desktop application is deliberately offline at runtime.  ``build.bat``
materialises this directory before PyInstaller runs, and the frozen app reads
only this bundle.  A missing file is an installation error, never a reason to
start a background download from a model screen.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


ASSET_FILES = {
    "efficientnet_b0": "torchvision/efficientnet_b0_rwightman-7f5810bc.pth",
    "efficientnet_b1": "torchvision/efficientnet_b1-c27df63c.pth",
    "resnet18": "torchvision/resnet18-f37072fd.pth",
    "resnet34": "torchvision/resnet34-b627a593.pth",
    "resnet50": "torchvision/resnet50-0676ba61.pth",
    "wide_resnet50_2": "torchvision/wide_resnet50_2-95faca4d.pth",
    "convnext_v1_tiny": "torchvision/convnext_tiny-983f1562.pth",
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


def builtin_asset_root() -> Path:
    """Return the immutable bundle path in both development and frozen mode."""
    override = os.environ.get("DVS_BUILTIN_ASSETS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "builtin_assets"
    return Path(__file__).resolve().parents[1] / "gui" / "builtin_assets"


def _manifest(root: Path) -> dict:
    path = root / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("기본 모델 가중치 묶음 manifest가 없습니다. build.bat을 다시 실행하세요.") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("files"), dict):
        raise RuntimeError("기본 모델 가중치 묶음 manifest 형식이 올바르지 않습니다. build.bat을 다시 실행하세요.")
    return value


def builtin_asset_path(model_id: str, *, verify: bool = True) -> Path:
    """Return one packaged file and reject substitutions/corruption early."""
    try:
        relative = ASSET_FILES[model_id]
    except KeyError as exc:
        raise ValueError(f"기본 모델 가중치 ID가 아닙니다: {model_id}") from exc
    root = builtin_asset_root()
    path = root / relative
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"기본 제공 {model_id} 가중치가 설치본에 없습니다: {path}\n"
            "앱에서 다운로드하지 않습니다. build.bat을 끝까지 다시 실행해 설치본을 만드세요."
        )
    if verify:
        entry = _manifest(root)["files"].get(relative)
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise RuntimeError(f"기본 모델 가중치 manifest 항목이 없습니다: {relative}")
        digestor = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digestor.update(block)
        digest = digestor.hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"기본 모델 가중치 무결성 검증 실패: {relative}. build.bat을 다시 실행하세요.")
    return path.resolve()


# 기본 모델 → ImageNet 백본 가중치 (builtin_models._torchvision_base 와 같은 대응)
BUILTIN_BACKBONE_ASSETS = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "convnext_v1_tiny": "convnext_v1_tiny",
    "unet_resnet18": "resnet18",
    "deeplabv3plus_resnet34": "resnet34",
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
    """설치본에 가중치가 없을 때 무엇을 하면 되는지 (사용자 안내 문구)."""
    return (
        f"이 학습에 필요한 사전학습 가중치({asset_id})가 설치본에 없습니다: {builtin_asset_root() / ASSET_FILES[asset_id]}\n\n"
        "설치본(EXE)이면 build.bat을 끝까지 다시 실행해 설치본을 만드세요.\n"
        "소스에서 실행 중이면 저장소 폴더에서 이 모델의 가중치만 받으세요 (인터넷 필요, 한 번만):\n"
        f"  python python/prepare_builtin_assets.py --output gui/builtin_assets --only {asset_id}\n\n"
        "또는 학습 모드를 '무작위 초기화로 시작'으로 바꾸면 가중치 없이 학습합니다(보통 정확도가 낮고 더 오래 걸립니다)."
    )


def builtin_assets_ready() -> bool:
    try:
        return all(builtin_asset_path(model_id).is_file() for model_id in ASSET_FILES)
    except (OSError, RuntimeError, ValueError):
        return False
