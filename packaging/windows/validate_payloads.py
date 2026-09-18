"""Verify every byte in a staged offline Windows payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy

verify_payload = runpy.run_path(str(Path(__file__).with_name("payload_manifest.py")))["verify_payload"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify an offline Windows payload manifest")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    verify_payload(args.root, json.loads(args.manifest.read_text(encoding="utf-8")))
    print("payload verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
