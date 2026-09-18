"""개발 Python 경로를 제거한 상태에서 실제 자동 압축 해제 패키지와 앱 EXE를 실행한다."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
release = ROOT.parent / "release"
archive, = release.glob("*.exe")
hasher = hashlib.sha256()
with archive.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        hasher.update(chunk)
checksum = hasher.hexdigest()
assert archive.with_suffix(".exe.sha256").read_text().split()[0] == checksum
with tempfile.TemporaryDirectory(prefix="Deep Studio (한글) ") as folder:
    # 사용자가 받는 EXE 자체가 별도 압축 프로그램 없이 풀리는지 검사한다.
    subprocess.run([str(archive), "-y", "-o" + folder], check=True, timeout=600)
    executable = Path(folder) / "DeepVisionStudio/DeepVisionStudio.exe"
    output = release / "desktop-smoke.json"
    env = {key: value for key, value in os.environ.items() if not key.startswith(("PYTHON", "CONDA", "VIRTUAL_ENV"))}
    env["PATH"] = os.path.join(os.environ["SystemRoot"], "System32")
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MPLBACKEND"] = "Agg"
    process = subprocess.Popen([str(executable), "--smoke-test", str(output)], cwd=folder, env=env)
    try:
        code = process.wait(timeout=240)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False)
        process.wait(timeout=30)
        raise
    finally:
        log = Path.home() / "deep-studio-desktop.log"
        if log.exists():
            content = log.read_text(encoding="utf-8", errors="replace")
            (release / "desktop-runtime.log").write_text(content, encoding="utf-8")
            print(content[-16000:])
        for log in (release / "smoke-state").rglob("console.log"):
            print(str(log), log.read_text(encoding="utf-8", errors="replace")[-8000:])
    if code or not output.exists():
        raise RuntimeError(f"배포 EXE 실행 실패: {code}")
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["cuda_runtime"] == "13.0"
    assert result["frozen"] and result["persistent_results"] and result["gui_ticks"] >= 3
    print("DESKTOP_PACKAGE_OK:", checksum, result)
