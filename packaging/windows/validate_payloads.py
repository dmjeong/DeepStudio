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
    parser.add_argument("--skip-model-catalog", action="store_true",
                        help="do not require a model catalog (minimal app/example installer only)")
    parser.add_argument("--require-release-ready-models", action="store_true",
                        help="require every catalog entry and its staged model payload")
    parser.add_argument("--model-pack-trust-store", type=Path,
                        help="absolute Ed25519 public-key trust store for release model packs")
    args = parser.parse_args(argv)
    if args.skip_model_catalog and args.require_release_ready_models:
        parser.error("--skip-model-catalog cannot be combined with --require-release-ready-models")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_payload(args.root, manifest, required_paths=args.require)
    if not args.skip_model_catalog:
        validate_model_catalog_payload(args.root, args.model_catalog,
                                       require_release_ready=args.require_release_ready_models,
                                       trust_store=args.model_pack_trust_store)
    if args.require_offline_wsl:
        verify_offline_wsl_payload(args.root, manifest)
    print("payload verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
