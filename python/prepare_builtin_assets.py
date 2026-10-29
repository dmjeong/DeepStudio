"""Fetch every basic-model pretrained checkpoint during the Windows build.

This command is intentionally a build-time operation.  It writes one checked
manifest which is included in the EXE; the GUI never calls these download APIs.
A weight that cannot be fetched (blocked network, proxy) is reported with its
manual download URL and does not stop the build.
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


def _fetch(model_id: str, target: Path) -> None:
    """가중치 하나를 받는다. 실패하면 예외 (호출한 쪽이 모아서 알린다)."""
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


def prepare(output: str | Path, only=None, *, strict: bool = False) -> Path:
    """가중치를 받고 SHA-256 manifest를 쓴다.

    한 모델이 실패해도(회사망 차단, Hugging Face 접속 불가 등) 나머지는 계속 받는다.
    받은 파일만 manifest에 기록하고, 실패한 모델은 마지막에 수동으로 받는 방법을 알린다.
    빌드는 멈추지 않는다 — 빠진 가중치는 앱의 Settings · 모델 관리에서 나중에 가져온다.
    strict=True면 하나라도 실패할 때 예외(이전 동작). only가 있으면 그 모델들만 받는다.
    """
    from builtin_assets import SOURCE_URLS, user_asset_root
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = list(ASSET_FILES) if not only else list(only)
    unknown = [model_id for model_id in selected if model_id not in ASSET_FILES]
    if unknown:
        raise SystemExit(f"알 수 없는 가중치 ID: {unknown}. 가능한 값: {', '.join(ASSET_FILES)}")
    failed = {}
    for model_id in selected:
        target = root / ASSET_FILES[model_id]
        try:
            _fetch(model_id, target)
        except Exception as exc:  # 네트워크·인증서·저장소 오류 모두: 기록하고 다음 모델로
            failed[model_id] = f"{type(exc).__name__}: {exc}"
            print(f"[failed] {model_id}: {failed[model_id]}")
            continue
        print(f"[ready] {model_id}: {target}")
    # 기존 manifest의 항목 중 파일이 남아 있는 것은 유지하고, 이번에 받은 것을 덧붙인다.
    files = {}
    try:
        previous = json.loads((root / "manifest.json").read_text(encoding="utf-8")).get("files", {})
        files = {key: value for key, value in previous.items() if (root / key).is_file()}
    except (OSError, ValueError, AttributeError):
        files = {}
    for model_id in selected:
        relative = ASSET_FILES[model_id]
        if model_id not in failed:
            files[relative] = {"sha256": _sha256(root / relative), "bytes": (root / relative).stat().st_size}
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n",
                                           encoding="utf-8")
    print(f"[ready] manifest: {root / 'manifest.json'} ({len(files)}/{len(ASSET_FILES)}개)")
    if failed:
        print("")
        print(f"[warning] 사전학습 가중치 {len(failed)}개를 받지 못했습니다. 빌드는 계속합니다.")
        print("          그 모델을 쓸 때만 필요합니다. 브라우저로 받아 앱의 Settings · 모델 관리 →")
        print(f"          '가중치 가져오기'로 넣거나 이 폴더에 넣으세요: {user_asset_root()}")
        for model_id in failed:
            print(f"   - {model_id}: {SOURCE_URLS[model_id]}")
        if strict:
            raise RuntimeError(f"가중치 다운로드 실패: {', '.join(failed)}")
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download packaged Deep Vision Studio basic-model weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", default="",
                        help="쉼표로 구분한 가중치 ID만 받는다 (예: resnet18,resnet34). 생략하면 전부")
    parser.add_argument("--strict", action="store_true",
                        help="하나라도 받지 못하면 실패로 끝낸다 (기본: 경고만 하고 계속)")
    args = parser.parse_args(argv)
    prepare(args.output, [v.strip() for v in args.only.split(",") if v.strip()] or None, strict=args.strict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
