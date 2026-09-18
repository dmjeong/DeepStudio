#!/usr/bin/env python3
"""Build an offline .dvmodel archive without importing model code."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_runtime.pack_builder import PackBuildError, build_pack


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build a Deep Vision Studio model pack")
    parser.add_argument("source", type=Path, help="directory containing manifest.json and model assets")
    parser.add_argument("output", type=Path, help="output .dvmodel path")
    parser.add_argument("--manifest", type=Path, help="manifest path when it is outside source")
    parser.add_argument("--allow-unsigned", action="store_true",
                        help="allow an unsigned development pack; release packs need an external signature")
    args = parser.parse_args(argv)
    try:
        path = build_pack(args.source, args.output, manifest=args.manifest,
                          allow_unsigned=args.allow_unsigned)
    except PackBuildError as exc:
        parser.error(str(exc))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

