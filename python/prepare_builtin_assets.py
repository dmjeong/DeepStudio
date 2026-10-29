"""Fetch every basic-model pretrained checkpoint during the Windows build.

This command is intentionally a build-time operation.  Each verified file is
published in the manifest immediately; the GUI never calls these download APIs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from builtin_assets import (ASSET_FILES, SEGMENTATION_ASSET_IDS, _manifest, builtin_asset_root,
                            file_sha256, official_torchvision_url, verify_local_asset)
from model_download import cached_imagenet_weights, enable_requests_native_ca
from sam2_assets import download_sam2_pretrained


def _write_manifest(root: Path, files: dict) -> None:
    """Publish each completed entry atomically; an interrupted write keeps the old JSON."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix=".manifest-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"schema_version": 1, "files": files}, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "manifest.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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


def prepare(output: str | Path, only=None, *, register_existing: bool = False) -> Path:
    """선택 모델마다 검증한 SHA-256을 기존 manifest에 원자적으로 합친다.

    개별 다운로드 실패에도 다음 모델을 시도하며 마지막에 실패 목록을 보고한다.
    빌드(build.bat)는 전부, only는 필요한 모델만, register_existing은 오프라인 등록만 한다.
    """
    root = Path(output).expanduser().resolve()
    selected = list(ASSET_FILES) if not only else list(dict.fromkeys(only))
    unknown = [model_id for model_id in selected if model_id not in ASSET_FILES]
    if unknown:
        raise SystemExit(f"알 수 없는 가중치 ID: {unknown}. 가능한 값: {', '.join(ASSET_FILES)}")
    files = dict(_manifest(root)["files"]) if (root / "manifest.json").exists() else {}
    root.mkdir(parents=True, exist_ok=True)
    failures = []
    for index, model_id in enumerate(selected, 1):
        relative = ASSET_FILES[model_id]
        target = root / relative
        print(f"[prepare {index}/{len(selected)}] {model_id}: {target}", flush=True)
        try:
            if register_existing or (relative in files and target.is_file()):
                entry = verify_local_asset(model_id, root, files)
            else:
                url = official_torchvision_url(model_id)
                if url:
                    # This also verifies manual files without creating a network connection.
                    downloaded = cached_imagenet_weights(url, target.parent)
                    if downloaded.resolve() != target.resolve():
                        raise RuntimeError(f"공식 가중치 경로가 예상과 다릅니다: {downloaded}")
                elif model_id.startswith("sam2_"):
                    downloaded = download_sam2_pretrained(model_id, target.parent)
                    if downloaded.resolve() != target.resolve():
                        raise RuntimeError(f"SAM2 가중치 경로가 예상과 다릅니다: {downloaded}")
                elif model_id.startswith(("libreyolo_", "re_detr_")):
                    _download_upstream(model_id, target)
                else:
                    raise AssertionError(model_id)
                if not target.is_file() or target.stat().st_size == 0:
                    raise RuntimeError(f"기본 가중치 다운로드가 완료되지 않았습니다: {model_id}")
                entry = {"sha256": file_sha256(target), "bytes": target.stat().st_size}
        except Exception as exc:
            failures.append((model_id, str(exc)))
            print(f"[failed] {model_id}: {exc}", flush=True)
            continue
        files[relative] = entry
        _write_manifest(root, files)  # Storage failures stop immediately, before reporting success.
        print(f"[ready] {model_id}: {target}", flush=True)
    if failures:
        details = "\n".join(f"  {model_id}: {error}" for model_id, error in failures)
        retry = ",".join(model_id for model_id, _error in failures)
        raise RuntimeError(
            f"가중치 준비 실패 {len(failures)}개. 성공한 모델의 파일과 manifest 항목은 유지됩니다.\n"
            f"{details}\n원인 해결 후 --only {retry} 로 실패한 모델만 재시도하세요.")
    print(f"[ready] manifest: {root / 'manifest.json'}")
    return root


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Download packaged Deep Vision Studio basic-model weights")
    parser.add_argument("--output", type=Path, default=None,
                        help="생략하면 앱이 사용하는 기본 가중치 폴더 (DVS_BUILTIN_ASSETS_DIR 포함)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--only", default="",
                        help="쉼표로 구분한 가중치 ID만 받는다 (예: resnet18,resnet34). 생략하면 전부")
    selection.add_argument("--group", choices=["segment"], help="세그멘테이션 기본 가중치 6종")
    parser.add_argument("--check", action="store_true", help="다운로드/파일 변경 없이 가중치 상태 검사")
    parser.add_argument("--register-existing", action="store_true",
                        help="다운로드 없이 검증된 로컬 파일만 manifest에 등록")
    args = parser.parse_args(argv)
    root = (args.output or builtin_asset_root()).expanduser().resolve()
    selected = (list(SEGMENTATION_ASSET_IDS) if args.group == "segment" else
                [v.strip() for v in args.only.split(",") if v.strip()] or list(ASSET_FILES))
    unknown = [model_id for model_id in selected if model_id not in ASSET_FILES]
    if unknown:
        parser.error(f"알 수 없는 가중치 ID: {unknown}")
    print(f"[assets] {root}", flush=True)
    if args.check:
        failed = 0
        for model_id in selected:
            try:
                verify_local_asset(model_id, root)
                print(f"[ready] {model_id}: {root / ASSET_FILES[model_id]}", flush=True)
            except (OSError, ValueError, RuntimeError) as exc:
                failed += 1
                print(f"[failed] {model_id}: {exc}", flush=True)
        print(f"[check] {len(selected) - failed}/{len(selected)} ready", flush=True)
        return 1 if failed else 0
    try:
        prepare(root, selected,
                register_existing=args.register_existing)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
