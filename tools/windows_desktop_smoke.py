"""Install/run the shipped Windows desktop, never a Python source substitute."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def checked(command, *, timeout=900, env=None, cwd=None):
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
        raise RuntimeError(f"실행 실패 ({code}): {command[0]}")


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
                checked(["msiexec.exe", "/i", str(Path(msi).resolve()), "/qn", "/norestart",
                         "INSTALLFOLDER=" + str(install), "/l*v", str(output.with_suffix(".install.log"))])
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
            environment = {k: v for k, v in os.environ.items()
                           if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "DVS_BUILTIN_ASSETS_DIR"}
                           and not k.startswith(("DEEP_STUDIO_", "DEEPVISION_"))}
            environment["QT_QPA_PLATFORM"] = "offscreen"
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
                checked(["msiexec.exe", "/x", str(Path(msi).resolve()), "/qn", "/norestart",
                         "/l*v", str(output.with_suffix(".uninstall.log"))])


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
