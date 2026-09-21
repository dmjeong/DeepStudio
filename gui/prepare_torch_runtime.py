"""Prepare the PyTorch runtime used by the frozen Windows application."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = ROOT / "gui"
for entry in (str(ROOT), str(GUI_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from core.accelerator import nvidia_devices  # noqa: E402
from webapp.bootstrap import (  # noqa: E402
    TORCH_VERSION,
    VISION_VERSION,
    inspect_runtime,
    installed_state,
    runtime_plan,
)


def prepare_runtime(accelerator: str = "auto", *, python: str | Path = sys.executable) -> dict:
    """Install and verify CPU/CUDA wheels for the current build interpreter."""
    if accelerator not in {"auto", "cuda", "cpu"}:
        raise ValueError("accelerator: auto, cuda, cpu 중 선택 필요")
    python = Path(python).resolve()
    packages = ("torch", "torchvision")
    installed = installed_state(python, packages)
    devices = nvidia_devices()
    report = inspect_runtime(python, ROOT) if installed["versions"].get("torch") else {}
    require_cuda, backend = runtime_plan(
        installed["versions"], report, devices, accelerator,
    )
    if backend is not None:
        index = "https://download.pytorch.org/whl/" + backend
        command = [
            str(python), "-m", "pip", "--disable-pip-version-check", "install",
            "--upgrade", "--progress-bar", "off",
            f"torch=={TORCH_VERSION}+{backend}",
            f"torchvision=={VISION_VERSION}+{backend}",
            "--index-url", index,
        ]
        print(f"PyTorch 빌드 런타임 준비: {backend}", flush=True)
        subprocess.run(command, check=True)

    verified_versions = installed_state(python, packages)["versions"]
    if not all(verified_versions.values()):
        raise RuntimeError("PyTorch 또는 torchvision 설치 확인 실패")
    verified = inspect_runtime(python, ROOT)
    if verified.get("error"):
        raise RuntimeError("PyTorch 실행 환경 검사 실패: " + verified["error"])
    if require_cuda and not verified.get("backward_checked"):
        raise RuntimeError("GPU 합성곱·역전파 검사 실패. CPU로 전환하지 않습니다.")
    print(
        f"PyTorch 빌드 런타임 확인: {verified.get('name', 'CPU')} / "
        f"CUDA {verified.get('cuda_runtime') or '없음'}",
        flush=True,
    )
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep Vision Studio 빌드용 PyTorch 준비")
    parser.add_argument("--accelerator", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    prepare_runtime(args.accelerator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
