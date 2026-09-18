"""배포할 ZIP의 바이트를 그대로 풀어 Windows 배치와 실제 서버를 검증한다."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import socket
import subprocess
import tempfile
import time
import urllib.request
import venv
import zipfile

COMMAND_ERRORS = re.compile(r"not recognized|cannot find the batch label|not executable", re.IGNORECASE)


def clean_environment(state):
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "DEEP_STUDIO_STATE_DIR": str(state)}
    for name in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(name, None)
    return env


def batch_command(arguments, codepage=949):
    # 모든 인자는 이 검사에서 정한 옵션과 정수다. 파일 경로는 cwd로만 전달한다.
    return ["cmd", "/d", "/e:on", "/v:off", "/c", f"chcp {codepage} >nul & start_web.bat {arguments}"]


def run_batch(app, arguments, env, codepage=949, timeout=180):
    result = subprocess.run(batch_command(arguments, codepage), cwd=app, env=env, input="",
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    output = result.stdout + result.stderr
    if result.returncode or COMMAND_ERRORS.search(output):
        raise RuntimeError(f"배포 배치 실행 실패 ({result.returncode}):\n{output}")
    return output


def diagnose_baseline(archive, directory):
    with zipfile.ZipFile(archive) as source:
        batch = source.read("DeepVisionStudio/start_web.bat")
    app = directory / "이전 배포 검사"
    app.mkdir()
    (app / "start_web.bat").write_bytes(batch)
    # 이전 배치의 명령 해석만 재현한다. 잘못된 분기가 실제 패키지를 설치하지 않게 한다.
    (app / "start_web.py").write_text("print('LEGACY_PYTHON_REACHED')\n", encoding="ascii")
    print(f"BASELINE_BYTES: CRLF={batch.count(bytes([13, 10]))}, LF={batch.count(bytes([10]))}, ASCII={batch.isascii()}", flush=True)
    crlf = batch.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    for label, content in (("ORIGINAL", batch), ("CRLF_ONLY", crlf)):
        (app / "start_web.bat").write_bytes(content)
        for codepage in (949, 65001):
            try:
                result = subprocess.run(batch_command("--help", codepage), cwd=app,
                                        env=clean_environment(directory / "baseline-state"), input="",
                                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
                output = result.stdout + result.stderr
                print(f"BASELINE_{label}_CP{codepage}: exit={result.returncode}, command_errors={bool(COMMAND_ERRORS.search(output))}\n{output}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"BASELINE_{label}_CP{codepage}: batch timed out", flush=True)


def verify_runtime(app, directory, env):
    environment = app / ".venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    run_batch(app, "--setup-only", env)
    marker = environment / ".deep-studio-web.json"
    log = environment / "deep-studio-setup.log"
    before = marker.read_bytes(), marker.stat().st_mtime_ns, log.read_bytes()
    run_batch(app, "--setup-only", env, codepage=65001)
    assert before == (marker.read_bytes(), marker.stat().st_mtime_ns, log.read_bytes())
    print("PACKAGE_SETUP_OK: shipped batch, actual imports, cached second start", flush=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server_log = directory / "server.log"
    with server_log.open("wb") as stream:
        process = subprocess.Popen(batch_command(f"--port {port} --no-browser"), cwd=app, env=env,
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(server_log.read_text(encoding="utf-8", errors="replace"))
                try:
                    with urllib.request.urlopen(url + "/api/state", timeout=1) as response:
                        state = json.load(response)
                    break
                except OSError:
                    time.sleep(.2)
            else:
                raise RuntimeError("배포 서버 응답 시간 초과\n" + server_log.read_text(encoding="utf-8", errors="replace"))
            version = runpy.run_path(str(app / "gui" / "core" / "version.py"))["APP_VERSION"]
            assert state["version"] == version
            with urllib.request.urlopen(url + "/", timeout=5) as response:
                html = response.read().decode("utf-8")
            assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', html)
            assert assets
            for asset in assets:
                with urllib.request.urlopen(url + asset, timeout=5) as response:
                    assert response.status == 200 and response.read()
            print(f"PACKAGE_SERVER_OK: version={version}, state=200, index=200, assets={len(assets)}", flush=True)
        finally:
            import psutil
            # taskkill 완료만으로는 자식의 파일 핸들 해제까지 보장되지 않는다.
            # 검사에서 시작한 프로세스만 추적하고 모두 종료된 뒤 임시 ZIP을 정리한다.
            try:
                parent = psutil.Process(process.pid)
                owned = parent.children(recursive=True) + [parent]
            except psutil.NoSuchProcess:
                owned = []
            if process.poll() is None:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
            process.wait(timeout=15)
            _, alive = psutil.wait_procs(owned, timeout=10)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=10)
            if alive:
                raise RuntimeError(f"배포 검사 자식 프로세스 종료 실패: {[child.pid for child in alive]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("Windows 실행 환경 필요")
    archives = list(args.release_dir.glob("*.zip"))
    assert len(archives) == 1
    archive = archives[0]
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive.with_suffix(".zip.sha256").read_text(encoding="ascii").split()[0] == checksum
    print(f"PACKAGE_SHA256: {checksum}", flush=True)
    with tempfile.TemporaryDirectory(prefix="studio-release-") as folder:
        directory = Path(folder)
        if args.baseline_dir:
            baseline = list(args.baseline_dir.glob("*.zip"))
            assert len(baseline) == 1
            diagnose_baseline(baseline[0], directory)
        extraction = directory / "실행 검사 (한글 공백)"
        with zipfile.ZipFile(archive) as source:
            batch = source.read("DeepVisionStudio/start_web.bat")
            assert batch.isascii() and batch.endswith(b"\r\n")
            assert b"\n" not in batch.replace(b"\r\n", b"")
            assert (args.release_dir / "start_web.bat").read_bytes() == batch
            source.extractall(extraction)
        app = extraction / "DeepVisionStudio"
        assert (app / "start_web.bat").read_bytes() == batch
        env = clean_environment(directory / "state")
        for codepage in (949, 65001):
            output = run_batch(app, "--help", env, codepage=codepage, timeout=30)
            assert "--setup-only" in output and "--port" in output
            print(f"PACKAGE_HELP_OK: codepage={codepage}, Unicode directory, Python launcher discovery", flush=True)
        if args.runtime:
            verify_runtime(app, directory, env)


if __name__ == "__main__":
    main()
