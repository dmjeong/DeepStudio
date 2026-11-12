"""Install missing model packages using THIS Python, preserving CUDA torch builds.

No model weights, no EXE build, no PowerShell. --check performs imports only.
"""

import argparse
import importlib.metadata
import subprocess
import sys
import tempfile
from pathlib import Path

GUI = Path(__file__).resolve().parent


def protected_versions():
    result = {}
    for package in ("torch", "torchvision", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package != "numpy":
                raise RuntimeError("PyTorch가 설치된 앱 실행 Python으로 실행하세요. "
                                   "새 PC 설치는 build.bat setup gpu를 사용하세요.") from None
    return result


def check_runtime():
    # New interpreter: never reuse an import cached before pip repaired files.
    code = (
        "import sys, torch, torchvision, timm, scipy.optimize; "
        "from timm.layers import DropPath, LayerNorm2d; "
        "from timm.models.efficientvit_mit import MBConv; "
        "print('Python:', sys.executable); "
        "print('torch:', torch.__version__, 'CUDA:', torch.version.cuda); "
        "print('torchvision:', torchvision.__version__, 'timm:', timm.__version__); "
        "print('Model runtime imports OK')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def repair_runtime():
    before = protected_versions()
    with tempfile.TemporaryDirectory(prefix="dvs-model-runtime-") as directory:
        constraints = Path(directory) / "keep-torch.txt"
        constraints.write_text("".join(f"{name}=={version}\n" for name, version in before.items()), encoding="utf-8")
        print("Preserving installed packages:", before, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                        str(GUI / "requirements-model-runtime.txt"), "--constraint", str(constraints)], check=True)
    if protected_versions() != before:
        raise RuntimeError("보존해야 할 PyTorch/torchvision/numpy 버전이 변경됐습니다. 실행 환경을 확인하세요.")
    check_runtime()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    print("Using Python:", sys.executable, flush=True)
    try:
        check_runtime() if args.check else repair_runtime()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Model runtime preparation failed: {exc}", file=sys.stderr)
        return 1
    print("Ready. Restart Deep Vision Studio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
