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
import re
import sys


ASSET_FILES = {
    "efficientnet_b0": "torchvision/efficientnet_b0_rwightman-7f5810bc.pth",
    "efficientnet_b1": "torchvision/efficientnet_b1-c27df63c.pth",
    "resnet18": "torchvision/resnet18-f37072fd.pth",
    "resnet34": "torchvision/resnet34-b627a593.pth",
    "resnet50": "torchvision/resnet50-11ad3fa6.pth",
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

SEGMENTATION_ASSET_IDS = (
    "resnet18", "resnet34", "sam2_hiera_tiny", "sam2_hiera_small",
    "sam2_hiera_base_plus", "sam2_hiera_large",
)


def builtin_asset_root() -> Path:
    """Return the immutable bundle path in both development and frozen mode."""
    override = os.environ.get("DVS_BUILTIN_ASSETS_DIR", "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            # Batch files cd to gui/, while python gui/main.py may run at the
            # repository root. An override must not change meaning with cwd.
            base = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
                    else Path(__file__).resolve().parents[1])
            path = base / path
        return path.resolve()
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "builtin_assets"
    return Path(__file__).resolve().parents[1] / "gui" / "builtin_assets"


def _manifest(root: Path) -> dict:
    path = root / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"기본 모델 가중치 manifest를 읽을 수 없습니다: {path}. "
                           "가중치 준비 도구로 확인하세요.") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("files"), dict):
        raise RuntimeError("기본 모델 가중치 묶음 manifest 형식이 올바르지 않습니다. build.bat을 다시 실행하세요.")
    return value


def official_torchvision_url(model_id: str) -> str | None:
    """Pinned official file, shared by the builder and offline file verification."""
    relative = ASSET_FILES[model_id]
    if relative.startswith("torchvision/"):
        return "https://download.pytorch.org/models/" + Path(relative).name
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_asset(model_id: str, root: Path, files: dict | None = None) -> dict:
    """Verify without network or writes; never bypass an existing manifest entry.

    Missing entries for pinned torchvision/SAM2 files use the publisher's SHA256
    (the same check as the official downloader). Arbitrary PT files and malformed
    manifests cannot use that fallback.
    """
    relative = ASSET_FILES[model_id]
    path = root / relative
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"기본 제공 {model_id} 가중치가 없습니다: {path}")
    if files is None:
        files = _manifest(root)["files"] if (root / "manifest.json").exists() else {}
    digest = file_sha256(path)
    size = path.stat().st_size
    if relative in files:
        entry = files[relative]
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise RuntimeError(f"기본 모델 가중치 manifest 항목 형식 오류: {relative}")
        if digest != entry["sha256"] or ("bytes" in entry and size != entry["bytes"]):
            raise RuntimeError(f"기본 모델 가중치 무결성 검증 실패: {relative}. 파일을 확인하세요.")
    else:
        match = re.search(r"-([a-f0-9]{8,64})\.pth$", relative)
        if model_id.startswith("sam2_"):
            from sam2_assets import get_sam2_asset
            expected_hash = get_sam2_asset(model_id).sha256
        elif official_torchvision_url(model_id) and match:
            expected_hash = match[1]
        else:
            raise RuntimeError(f"기본 모델 가중치 manifest 항목이 없습니다: {relative}. "
                               "이 모델은 공식 다운로드 도구로 준비하세요.")
        if not digest.startswith(expected_hash):
            raise RuntimeError(f"공식 가중치 SHA256 불일치: {path}. 파일 손상·버전을 확인하세요.")
    return {"sha256": digest, "bytes": size}


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
            "Python 실행: gui 폴더의 prepare_model_weights.bat을 실행하세요.\n"
            "EXE 설치본: build.bat으로 가중치를 포함해 다시 빌드하세요."
        )
    if verify:
        verify_local_asset(model_id, root)
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
        f"이 학습에 필요한 기본 가중치({asset_id}) 확인 위치: {builtin_asset_root() / ASSET_FILES[asset_id]}\n\n"
        "설치본(EXE)이면 build.bat을 끝까지 다시 실행해 설치본을 만드세요.\n"
        "Python 실행이면 gui 폴더의 prepare_model_weights.bat을 실행하세요 (전체 기본 모델, 한 번만).\n"
        "세그멘테이션만 준비: prepare_model_weights.bat segment\n"
        "EXE를 다시 만들 필요 없습니다. 준비 도구는 위 앱 검색 폴더에 저장합니다.\n"
        f"개별 준비: python python/prepare_builtin_assets.py --only {asset_id}\n\n"
        + ("ResNet 가중치만 준비: gui 폴더의 prepare_resnet_weights.bat을 실행하세요.\n"
           "공식 파일을 직접 넣었다면 SHA256 검증 후 manifest 없이도 인식합니다.\n\n"
           if asset_id in {"resnet18", "resnet34", "resnet50"} else "") +
        ("SAM2는 공식 기반 가중치가 반드시 필요합니다. 무작위 초기화 모드는 지원하지 않습니다."
         if asset_id.startswith("sam2_") else
         "또는 학습 모드를 '무작위 초기화로 시작'으로 바꾸면 가중치 없이 학습합니다(보통 정확도가 낮고 더 오래 걸립니다).")
    )


def builtin_assets_ready() -> bool:
    try:
        files = _manifest(builtin_asset_root())["files"]
        return all(relative in files and builtin_asset_path(model_id).is_file()
                   for model_id, relative in ASSET_FILES.items())
    except (OSError, RuntimeError, ValueError):
        return False
