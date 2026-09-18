"""모델 팩의 manifest와 안전한 ZIP 경로를 오프라인에서 검증한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))

from core.model_registry import ModelRegistry  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Deep Vision Studio .dvmodel pack")
    parser.add_argument("pack", type=Path)
    args = parser.parse_args(argv)
    registry = ModelRegistry.builtin()
    try:
        spec = registry.load_pack(args.pack)
    except Exception as exc:
        parser.error(str(exc))
    print(json.dumps({"valid": True, "model": spec.to_dict()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
