"""추적 중인 프로젝트 소스와 검증된 React 빌드로 실행용 ZIP을 생성한다."""
import argparse
import hashlib
from pathlib import Path
import runpy
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def packaged_bytes(path):
    content = Path(path).read_bytes()
    if Path(path).suffix.lower() == ".bat":
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n").replace(b"\n", b"\r\n")
    return content


def validate_windows_launcher(content):
    if not content.isascii() or b"\n" in content.replace(b"\r\n", b"") or not content.endswith(b"\r\n"):
        raise RuntimeError("Windows 시작 배치 파일은 BOM 없는 ASCII와 CRLF 줄바꿈이어야 합니다.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="release")
    args = parser.parse_args()
    version = runpy.run_path(str(ROOT / "gui" / "core" / "version.py"))["APP_VERSION"]
    frontend = ROOT / "web" / "dist"
    if not (frontend / "index.html").is_file():
        raise RuntimeError("검증된 React 빌드 필요")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"Deep-Vision-Studio-React-v{version}.zip"
    tracked = subprocess.check_output(["git", "ls-files", "-z", "."], cwd=ROOT).decode().split("\0")
    files = {ROOT / name for name in tracked if name and (ROOT / name).is_file()}
    files.update(path for path in frontend.rglob("*") if path.is_file())
    # 패키지 잠금 파일은 배포 ZIP에도 포함한다.
    files.add(ROOT / "web" / "package-lock.json")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for path in sorted(files):
            target.writestr("DeepVisionStudio/" + path.relative_to(ROOT).as_posix(), packaged_bytes(path))
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        names = target.namelist()
        for required in ("run_web.py", "start_web.py", "start_web.bat", "webapp/bootstrap.py", "web/dist/index.html", "webapp/requirements.txt"):
            assert f"DeepVisionStudio/{required}" in names
        assert not any("node_modules/" in name or "/.venv/" in name for name in names)
        validate_windows_launcher(target.read("DeepVisionStudio/start_web.bat"))
        (output / "start_web.bat").write_bytes(target.read("DeepVisionStudio/start_web.bat"))
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    print(archive)


if __name__ == "__main__":
    main()
