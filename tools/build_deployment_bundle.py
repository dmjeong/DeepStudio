"""Build a verified ``.dvdeploy`` directory for the native SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_runtime.deployment_bundle import (  # noqa: E402
    DeploymentBundleError,
    build_deployment_bundle,
    verify_deployment_bundle,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build a Deep Vision ONNX deployment bundle")
    parser.add_argument("source", type=Path, help="exported ONNX file or SAM2 export directory")
    parser.add_argument("output", type=Path, help="output .dvdeploy directory")
    args = parser.parse_args(argv)
    try:
        result = build_deployment_bundle(args.source, args.output)
        manifest = verify_deployment_bundle(result)
    except DeploymentBundleError as exc:
        parser.error(str(exc))
    print(json.dumps({"path": str(result), "files": len(manifest["files"]),
                      "backend": manifest["backend"], "task": manifest["task"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
