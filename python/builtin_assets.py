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


def builtin_assets_ready() -> bool:
    try:
        return all(builtin_asset_path(model_id).is_file() for model_id in ASSET_FILES)
    except (OSError, RuntimeError, ValueError):
        return False
