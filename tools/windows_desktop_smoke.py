"""Install/run the shipped Windows desktop, never a Python source substitute."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def clean_app_environment(source):
    """Allow Windows system DLLs, never a build-time Python/native SDK PATH."""
    excluded = {"PATH", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "DVS_BUILTIN_ASSETS_DIR"}
    prefixes = ("PYTHON", "DEEP_STUDIO_", "DEEPVISION_", "DVS_", "VISION_", "ONNXRUNTIME_", "NATIVE_",
                "QT_", "QML", "CUDA_PATH", "CUDNN", "CONDA_")
    environment = {k: v for k, v in source.items()
                   if k.upper() not in excluded and not k.upper().startswith(prefixes)}
    from pathlib import PureWindowsPath
    windows = PureWindowsPath(next((v for k, v in source.items() if k.upper() == "SYSTEMROOT"), r"C:\Windows"))
    environment["PATH"] = ";".join(map(str, (windows / "System32", windows, windows / "System32/Wbem")))
    environment["QT_QPA_PLATFORM"] = "offscreen"
    return environment


def checked(command, *, timeout=900, env=None, cwd=None):
    print("EXECUTE:", command if isinstance(command, str) else subprocess.list2cmdline(command), flush=True)
    process = subprocess.Popen(command, env=env, cwd=cwd)
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        import psutil
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        process.wait()
        raise
    if code not in (0, 3010):
        raise RuntimeError(f"실행 실패 ({code}): {command}")


def run_msi(msi, log, *, install=None):
    """MSI properties need VALUE quotes, not CRT quotes around NAME=VALUE.

    Passing a list to Popen quotes the entire INSTALLFOLDER=... argument when
    it contains spaces. msiexec parses properties itself and then displays its
    usage dialog instead of installing, even with /qn. Pass the documented
    PROPERTY="value" syntax directly, without cmd.exe or another shell.
    https://learn.microsoft.com/en-us/windows/win32/msi/command-line-options
    """
    def quote(path):
        value = str(Path(path).resolve())
        if any(char in value for char in ('"', '\r', '\n', '\x00')):
            raise ValueError("설치 경로에 사용할 수 없는 문자가 있습니다.")
        return '"' + value + '"'
    command = f'msiexec.exe /{"i" if install is not None else "x"} {quote(msi)} /qn /norestart /l*v {quote(log)}'
    if install is not None:
        command += " INSTALLFOLDER=" + quote(install)
    checked(command)


def run(output, *, app=None, msi=None):
    if os.name != "nt":
        raise RuntimeError("실제 EXE/설치 검증은 Windows에서 실행하세요.")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Native exceptions must produce a failed job, not an invisible error box.
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    with tempfile.TemporaryDirectory(prefix="dvs-installed-") as directory:
        folder = Path(directory)
        install = folder / "한글 설치 폴더"
        installed = False
        try:
            if msi:
                run_msi(msi, output.with_suffix(".install.log"), install=install)
                installed = True
                executable = install / "app" / "DeepVisionStudio.exe"
                if not (install / "Examples" / "cpp").is_dir():
                    raise RuntimeError("설치본 C++ 예제 누락")
                if not (install / "Examples" / "csharp").is_dir():
                    raise RuntimeError("설치본 C# 예제 누락")
            else:
                executable = Path(app).resolve()
            if not executable.is_file():
                raise FileNotFoundError(executable)
            environment = clean_app_environment(os.environ)
            checked([str(executable), "--smoke-test", str(output)], env=environment, cwd=folder)
            report = json.loads(output.read_text(encoding="utf-8"))
            for field in ("frozen", "persistent_results", "defect_review_saved", "project_class_generation_saved"):
                if report.get(field) is not True:
                    raise RuntimeError(f"EXE 실행 검증 누락: {field}")
            if report["inference_status"] == "error":
                raise RuntimeError("EXE 추론 검증 실패")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            if installed:
                run_msi(msi, output.with_suffix(".uninstall.log"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--app")
    source.add_argument("--msi")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        run(args.output, app=args.app, msi=args.msi)
    except Exception:
        log = Path(args.output).with_suffix(".log")
        if log.exists():
            print(log.read_text(encoding="utf-8", errors="replace")[-24000:])
        raise
