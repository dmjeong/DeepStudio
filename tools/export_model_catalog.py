"""Write the deterministic built-in model catalog for an offline installer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gui.core.model_registry import ModelRegistry  # noqa: E402


def build_catalog(*, release_ready_only: bool = False) -> dict:
    registry = ModelRegistry.builtin()
    models = registry.list(release_ready=release_ready_only)
    return {
        "schema_version": 1,
        "catalog_id": "deepvisionstudio-builtin-models",
        "offline": True,
        "release_ready_only": release_ready_only,
        "redistribution_policy": {
            "weights_included": True,
            "require_third_party_notices": True,
            "require_license_files_for_release_packs": True,
        },
        "models": [spec.to_dict() for spec in models],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export the offline built-in model catalog")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-ready-only", action="store_true")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_catalog(release_ready_only=args.release_ready_only),
                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
