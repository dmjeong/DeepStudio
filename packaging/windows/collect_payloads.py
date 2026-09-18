"""Build the deterministic manifest consumed by the WiX release script.

This entry point intentionally performs no downloads or model imports.  The
Windows build pipeline must stage all runtime DLLs, worker executables, SDK
files, notices, and model packs before calling it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy

collect_payload = runpy.run_path(str(Path(__file__).with_name("payload_manifest.py")))["collect_payload"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect an offline Windows payload manifest")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", default="")
    parser.add_argument("--budget-bytes", type=int)
    args = parser.parse_args(argv)
    kwargs = {"version": args.version, "commit": args.commit}
    if args.budget_bytes is not None:
        kwargs["budget_bytes"] = args.budget_bytes
    manifest = collect_payload(args.root, **kwargs)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    import json
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
