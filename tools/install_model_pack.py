"""Offline model pack installer CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_runtime.pack_installer import PackInstallError, PackInstaller  # noqa: E402
from gui.core.model_registry import default_installed_model_root  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Install a verified Deep Vision Studio model pack")
    parser.add_argument("pack", type=Path)
    parser.add_argument("--root", type=Path,
                        help="installed model root (default: per-user DeepVisionStudio directory)")
    parser.add_argument("--allow-unsigned", action="store_true", help="development-only unsigned pack")
    args = parser.parse_args(argv)
    try:
        root = args.root or default_installed_model_root()
        installed = PackInstaller(root).install(args.pack, allow_unsigned=args.allow_unsigned)
    except PackInstallError as error:
        parser.error(str(error))
    print(json.dumps({"model_id": installed.model_id, "pack_version": installed.pack_version,
                      "content_hash": installed.content_hash, "path": str(installed.path),
                      "signed": installed.signed}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
