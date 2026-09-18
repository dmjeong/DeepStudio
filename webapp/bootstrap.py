"""전용 Python 환경을 준비하고 설치 완료 상태를 검증한다."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import venv

from webapp.locking import exclusive_file
from core.accelerator import cuda_index, nvidia_devices

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_VERSION = "2.14.0"
VISION_VERSION = "0.29.0"
RUNTIME_MODULES = ("fastapi", "uvicorn", "torch", "torchvision", "PIL", "numpy", "cv2",
                   "matplotlib", "onnx", "onnxruntime", "psutil", "truststore", "certifi", "shapely")


def environment_path(root):
    """활성 가상환경을 우선 재사용하고 기본 Python이면 프로젝트 전용 환경을 쓴다."""
    prefix = Path(sys.prefix).resolve()
    active = any(Path(os.environ[name]).resolve() == prefix
                 for name in ("VIRTUAL_ENV", "CONDA_PREFIX") if os.environ.get(name))
    return prefix if sys.prefix != sys.base_prefix or active else Path(root) / ".venv"


def python_path(environment):
    # Conda의 Windows Python은 Scripts가 아닌 환경 루트에 있다.
    if Path(environment).resolve() == Path(sys.prefix).resolve():
        return Path(sys.executable)
    return Path(environment) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run_logged(command, log):
    """인자를 셸 해석 없이 전달하고 설치 출력을 화면과 파일에 함께 남긴다."""
    log.write("\n> " + subprocess.list2cmdline([str(value) for value in command]) + "\n")
    log.flush()
    with subprocess.Popen([str(value) for value in command], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    log.flush()
    if code:
        raise RuntimeError(f"설치 또는 환경 검사 실패 (종료 코드 {code}). 로그: {log.name}")


def installed_state(python, packages):
    script = """
import importlib.metadata as metadata
import json, sys
versions = {}
for name in json.loads(sys.argv[1]):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"python": list(sys.version_info[:3]), "versions": versions}))
"""
    result = subprocess.run([str(python), "-c", script, json.dumps(packages)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"가상환경 Python 실행 실패: {python}\n{result.stderr.strip()}")
    return json.loads(result.stdout)


def inspect_runtime(python, root):
    script = ("import json,sys; sys.path.insert(0,sys.argv[1]); "
              "from core.accelerator import runtime_report; "
              "print(json.dumps(runtime_report(check=True)))")
    try:
        result = subprocess.run([str(python), "-c", script, str(Path(root) / "gui")],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if result.returncode:
            return {"cuda_available": False, "error": result.stderr.strip()}
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return {"cuda_available": False, "error": str(exc)}


def runtime_plan(versions, report, devices, accelerator):
    """정상 GPU 환경 보존, CPU 전용 환경의 GPU 복구, 명시 CPU 실행을 구분한다."""
    usable = report.get("cuda_available") and not report.get("error")
    require_cuda = accelerator == "cuda" or (accelerator == "auto" and bool(devices or usable))
    complete = bool(versions.get("torch") and versions.get("torchvision"))
    if require_cuda:
        return require_cuda, None if complete and usable else cuda_index(devices)
    if complete:
        return False, None
    # 부분 설치된 GPU 패키지도 CPU 패키지로 덮어쓰지 않는다.
    if report.get("cuda_runtime"):
        return False, "cu" + report["cuda_runtime"].replace(".", "")
    return False, "cpu"


def ensure_environment(root, state_dir, runtime_modules=RUNTIME_MODULES, accelerator="auto"):
    if accelerator not in ("auto", "cuda", "cpu"):
        raise ValueError("accelerator: auto, cuda, cpu 중 선택 필요")
    root = Path(root).resolve()
    requirements = root / "webapp" / "requirements.txt"
    content = requirements.read_bytes()
    packages = []
    for line in content.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*", line)
        if not match or line.startswith("-"):
            raise RuntimeError("지원하지 않는 requirements 형식")
        packages.append(match.group())
    environment = environment_path(root)
    python = python_path(environment)
    marker = environment / ".deep-studio-web.json"
    fingerprint = hashlib.sha256(content).hexdigest()
    with exclusive_file(environment / ".deep-studio-setup.lock", "패키지 설치가 이미 진행 중입니다. 해당 창에서 완료를 기다리세요."):
        if not python.is_file():
            print(f"[1/3] Python 가상환경 생성: {environment}", flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        installed = installed_state(python, packages)
        has_torch = "torch" in packages and "torchvision" in packages
        devices = nvidia_devices() if has_torch else []
        report = inspect_runtime(python, root) if has_torch and installed["versions"].get("torch") else {}
        require_cuda, backend = runtime_plan(installed["versions"], report, devices, accelerator) if has_torch else (False, None)
        expected = {"requirements": fingerprint, **installed, "accelerator": accelerator,
                    "require_cuda": require_cuda, "devices": devices}
        try:
            ready = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            ready = None
        if ready == expected and all(installed["versions"].values()) and backend is None and not report.get("error"):
            print(f"설치된 환경 확인 완료: {report.get('name', 'CPU')} / CUDA {report.get('cuda_runtime') or '없음'}", flush=True)
            return python

        # 실행 중인 스튜디오의 패키지를 설치 도중 변경하지 않는다.
        with exclusive_file(Path(state_dir) / "server.lock", "실행 중인 스튜디오를 종료한 후 패키지를 설치하세요."):
            marker.unlink(missing_ok=True)
            log_path = environment / "deep-studio-setup.log"
            with log_path.open("w", encoding="utf-8") as log:
                print(f"[2/3] 필요한 패키지 설치 중. 최초 설치에는 시간이 걸릴 수 있습니다.\n로그: {log_path}", flush=True)
                if subprocess.run([str(python), "-m", "pip", "--version"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                    run_logged([python, "-m", "ensurepip", "--upgrade"], log)
                pip = [python, "-m", "pip", "--disable-pip-version-check"]
                if backend is not None:
                    print(f"PyTorch 실행 환경 준비: {backend}", flush=True)
                    index = CPU_INDEX if backend == "cpu" else "https://download.pytorch.org/whl/" + backend
                    run_logged([*pip, "install", "--upgrade", "--progress-bar", "off",
                                f"torch=={TORCH_VERSION}+{backend}", f"torchvision=={VISION_VERSION}+{backend}",
                                "--index-url", index], log)
                run_logged([*pip, "install", "--progress-bar", "off", "-r", requirements], log)
                print("[3/3] 패키지 호환성과 실행 환경 검사 중", flush=True)
                run_logged([*pip, "check"], log)
                script = "import importlib, json, sys; [importlib.import_module(name) for name in json.loads(sys.argv[1])]"
                run_logged([python, "-c", script, json.dumps(list(runtime_modules))], log)
                if has_torch:
                    verified_runtime = inspect_runtime(python, root)
                    if verified_runtime.get("error") or (require_cuda and not verified_runtime.get("backward_checked")):
                        raise RuntimeError("GPU 실행 환경 검사 실패. CPU로 전환하지 않습니다. "
                                           + verified_runtime.get("error", "CUDA 장치와 드라이버 확인 필요"))
                    print(f"실행 환경: {verified_runtime.get('name', 'CPU')} / CUDA {verified_runtime.get('cuda_runtime') or '없음'}", flush=True)
            verified = installed_state(python, packages)
            if not all(verified["versions"].values()):
                raise RuntimeError(f"설치 후 누락된 패키지 확인 필요. 로그: {log_path}")
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps({**expected, **verified}), encoding="utf-8")
            os.replace(temporary, marker)
        return python
