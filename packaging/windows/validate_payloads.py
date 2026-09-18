"""Verify every byte in a staged offline Windows payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy

_payload_api = runpy.run_path(str(Path(__file__).with_name("payload_manifest.py")))
verify_payload = _payload_api["verify_payload"]
verify_offline_wsl_payload = _payload_api["verify_offline_wsl_payload"]
_catalog_api = runpy.run_path(str(Path(__file__).with_name("model_catalog_payload.py")))
validate_model_catalog_payload = _catalog_api["validate_model_catalog_payload"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify an offline Windows payload manifest")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require", action="append", default=[],
                        help="require an exact relative file in the staged payload")
    parser.add_argument("--require-offline-wsl", action="store_true",
                        help="verify WSL/Docker artifact hashes and license inventory")
    parser.add_argument("--model-catalog", default="models/default-model-catalog.json",
                        help="catalog path relative to the staged payload")
    parser.add_argument("--require-release-ready-models", action="store_true",
                        help="require every catalog entry and its staged model payload")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_payload(args.root, manifest, required_paths=args.require)
    validate_model_catalog_payload(args.root, args.model_catalog,
                                   require_release_ready=args.require_release_ready_models)
    if args.require_offline_wsl:
        verify_offline_wsl_payload(args.root, manifest)
    print("payload verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
