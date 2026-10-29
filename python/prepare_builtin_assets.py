"""Fetch every basic-model pretrained checkpoint during the Windows build.

This command is intentionally a build-time operation.  It writes one checked
manifest which is included in the EXE; the GUI never calls these download APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from builtin_assets import ASSET_FILES
from efficientnet import VARIANTS
from model_download import cached_imagenet_weights, enable_requests_native_ca
from sam2_assets import download_sam2_pretrained


_TORCHVISION_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "convnext_v1_tiny": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_upstream(model_id: str, destination: Path) -> None:
    """Use the pinned LibreYOLO public downloader once, at build time."""
    from upstream_models import get_upstream_spec
    # LibreYOLO uses requests internally. Apply the Windows native CA store
    # before importing it so corporate TLS inspection certificates installed
    # in Windows are trusted without disabling certificate verification.
    enable_requests_native_ca()
    from libreyolo.utils.download import download_weights
    spec = get_upstream_spec(model_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    download_weights(str(destination), spec.size)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"{model_id} 기본 가중치 다운로드가 완료되지 않았습니다")


def prepare(output: str | Path, only=None) -> Path:
    """가중치를 받고 SHA-256 manifest를 쓴다. only가 있으면 그 모델들만 받고 기존 manifest에 합친다
    (소스에서 실행할 때 필요한 모델만 준비). 빌드(build.bat)는 전부 받는다."""
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = list(ASSET_FILES) if not only else list(only)
    unknown = [model_id for model_id in selected if model_id not in ASSET_FILES]
    if unknown:
        raise SystemExit(f"알 수 없는 가중치 ID: {unknown}. 가능한 값: {', '.join(ASSET_FILES)}")
    for model_id in selected:
        relative = ASSET_FILES[model_id]
        target = root / relative
        if model_id.startswith("sam2_"):
            downloaded = download_sam2_pretrained(model_id, target.parent)
            if downloaded.resolve() != target.resolve():
                raise RuntimeError(f"SAM2 가중치 경로가 예상과 다릅니다: {downloaded}")
        elif model_id.startswith("efficientnet_"):
            cached_imagenet_weights(VARIANTS[model_id].url, target.parent)
        elif model_id in _TORCHVISION_URLS:
            cached_imagenet_weights(_TORCHVISION_URLS[model_id], target.parent)
        elif model_id.startswith(("libreyolo_", "re_detr_")):
            _download_upstream(model_id, target)
        else:
            raise AssertionError(model_id)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"기본 가중치 다운로드가 완료되지 않았습니다: {model_id}")
        print(f"[ready] {model_id}: {target}")
    files = {}
    if only and (root / "manifest.json").is_file():
        try:  # 이미 받은 다른 모델의 항목은 유지한다
            files = json.loads((root / "manifest.json").read_text(encoding="utf-8")).get("files", {})
        except (OSError, ValueError):
            files = {}
    for model_id in selected:
        relative = ASSET_FILES[model_id]
        files[relative] = {"sha256": _sha256(root / relative), "bytes": (root / relative).stat().st_size}
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n",
                                           encoding="utf-8")
    print(f"[ready] manifest: {root / 'manifest.json'}")
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download packaged Deep Vision Studio basic-model weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", default="",
                        help="쉼표로 구분한 가중치 ID만 받는다 (예: resnet18,resnet34). 생략하면 전부")
    args = parser.parse_args(argv)
    prepare(args.output, [v.strip() for v in args.only.split(",") if v.strip()] or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
