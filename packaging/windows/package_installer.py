"""Package large, fully offline MSI/cabinet layouts as one distributable ZIP."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import zipfile


def package_installer(root: Path, version: str, cabinet_prefix: str, notice: Path,
                      extras=()) -> Path:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Invalid installer version")
    if not re.fullmatch(r"d[0-9a-f]{4}", cabinet_prefix):
        raise ValueError("Invalid cabinet prefix")
    root = Path(root)
    setup_name = f"DeepVisionStudio-Setup-{version}-win-x64.exe"
    paths = [root / setup_name, root / f"DeepVisionStudio-{version}.msi",
             root / "release-manifest.json", Path(notice)]
    cabinets = sorted(root.glob(f"{cabinet_prefix}*.cab"))
    if not cabinets:
        raise ValueError("External installer has no cabinets")
    paths += cabinets + [Path(item) for item in extras]
    names = set()
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Installer file missing or symlink: {path}")
        if path.name.casefold() in names:
            raise ValueError(f"Duplicate installer filename: {path.name}")
        names.add(path.name.casefold())
    target = root / f"DeepVisionStudio-Offline-{version}-win-x64.zip"
    temporary = target.with_suffix(".zip.tmp")
    try:
        # CABs are already compressed. ZIP64 permits the entire pretrained
        # library without the 4 GiB limit of a signed Windows executable.
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in paths:
                archive.write(path, path.name)
            archive.writestr("INSTALL.ko.txt",
                "Deep Vision Studio 오프라인 설치\n\n"
                "1. 이 ZIP을 폴더에 전부 압축 해제하세요.\n"
                f"2. {setup_name}를 실행하세요.\n"
                "3. 같은 폴더의 MSI와 CAB 파일을 옮기거나 삭제하지 마세요.\n\n"
                "설치할 PC에는 Python이나 별도 패키지를 설치할 필요가 없습니다.\n"
                "설치 과정에 관리자 권한이 필요합니다. NVIDIA GPU 학습에는 호환 드라이버가 필요합니다.\n"
                "설치 후 앱 폴더의 README.ko.md에서 최소 사양과 사용법을 확인하세요.\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--cabinet-prefix", required=True)
    parser.add_argument("--notice", type=Path, required=True)
    parser.add_argument("--extra", action="append", default=[], type=Path)
    args = parser.parse_args()
    print(package_installer(args.root, args.version, args.cabinet_prefix, args.notice, args.extra))


if __name__ == "__main__":
    main()
