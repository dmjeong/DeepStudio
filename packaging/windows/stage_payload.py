"""Stage a complete offline Windows payload without following symlinks.

The command accepts already-built artifacts only.  It is deliberately boring:
the CI job is responsible for producing the frozen UI, native workers, model
packs, SDK files, runtime DLLs, notices, and SBOM before WiX is invoked.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import shutil


class PayloadStageError(ValueError):
    pass


def _destination(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (path.is_absolute() or not value or ":" in normalized or "//" in normalized or
            ".." in path.parts or any(part == "" for part in path.parts)):
        raise PayloadStageError(f"unsafe payload destination: {value}")
    return path


def _copy(source: Path, target: Path) -> None:
    if source.is_symlink():
        raise PayloadStageError(f"payload source symlink is not allowed: {source}")
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or target.read_bytes() != source.read_bytes():
                raise PayloadStageError(f"conflicting payload file: {target}")
            return
        shutil.copy2(source, target)
        return
    if not source.is_dir():
        raise PayloadStageError(f"payload source does not exist: {source}")
    for child in sorted(source.rglob("*")):
        if child.is_symlink():
            raise PayloadStageError(f"payload source symlink is not allowed: {child}")
        if child.is_file():
            _copy(child, target / child.relative_to(source))


def stage_payload(output: str | Path, sources: list[tuple[str, str | Path]], *,
                  notice: str | Path | None = None) -> Path:
    output_path = Path(output).expanduser()
    if output_path.exists() and output_path.is_symlink():
        raise PayloadStageError("payload output symlink is not allowed")
    root = output_path.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for destination, source in sources:
        target_root = root / _destination(destination)
        source_path = Path(source).expanduser()
        if source_path.is_symlink():
            raise PayloadStageError(f"payload source symlink is not allowed: {source_path}")
        source_path = source_path.resolve()
        _copy(source_path, target_root)
    if notice is not None:
        notice_path = Path(notice).expanduser()
        if notice_path.is_symlink():
            raise PayloadStageError(f"payload notice symlink is not allowed: {notice_path}")
        _copy(notice_path.resolve(), root / "THIRD_PARTY_NOTICES.md")
    if not (root / "THIRD_PARTY_NOTICES.md").is_file():
        raise PayloadStageError("payload requires THIRD_PARTY_NOTICES.md")
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage an offline Windows payload")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], metavar="DEST=PATH",
                        help="copy a file/tree to a relative payload destination")
    parser.add_argument("--notice", type=Path)
    args = parser.parse_args(argv)
    sources: list[tuple[str, str]] = []
    for item in args.source:
        if "=" not in item:
            parser.error("--source must use DEST=PATH")
        destination, source = item.split("=", 1)
        sources.append((destination, source))
    print(stage_payload(args.output, sources, notice=args.notice))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
