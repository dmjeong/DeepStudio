"""CUDA 배포를 자동 압축 해제 EXE로 묶고 릴리스 크기 제한을 검사한다."""
from pathlib import Path
import hashlib
import os
import runpy
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
version = runpy.run_path(str(ROOT / "gui/core/version.py"))["APP_VERSION"]
source = ROOT / "gui/dist/DeepVisionStudio"
if not (source / "DeepVisionStudio.exe").is_file():
    raise RuntimeError("EXE 빌드 필요")
output = ROOT.parent / "release"
output.mkdir(exist_ok=True)
seven_zip = shutil.which("7z")
if not seven_zip:
    raise RuntimeError("패키지 생성용 7-Zip 설치 필요")
candidates = [Path(seven_zip).parent / "7z.sfx",
              Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "7-Zip/7z.sfx"]
stub = next((path for path in candidates if path.is_file()), None)
if stub is None:
    raise RuntimeError("7-Zip 자동 압축 해제 모듈 누락")
license_path = stub.parent / "License.txt"
if license_path.is_file():
    (source / "licenses").mkdir(exist_ok=True)
    shutil.copyfile(license_path, source / "licenses/7zip.txt")
archive = output / f"Deep-Vision-Studio-Desktop-v{version}.exe"
subprocess.run([seven_zip, "a", "-t7z", "-sfx" + str(stub), "-mx=3", "-mmt=2",
                str(archive), source.name], cwd=source.parent, check=True)
if archive.stat().st_size >= 2 * 1024 ** 3:
    raise RuntimeError("GitHub 릴리스의 파일당 2 GiB 제한 초과")
print(f"DESKTOP_PACKAGE_BYTES: {archive.stat().st_size}", flush=True)
hasher = hashlib.sha256()
with archive.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        hasher.update(chunk)
checksum = hasher.hexdigest()
archive.with_suffix(".exe.sha256").write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
print(f"DESKTOP_PACKAGE_SHA256: {checksum}")
