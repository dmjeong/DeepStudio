"""설치 전 NVIDIA 감지와 실제 학습 프로세스의 CUDA 연산 검사."""

import csv
import os
from pathlib import Path
import shutil
import subprocess


def nvidia_devices():
    """PyTorch 설치 종류와 무관하게 드라이버가 인식한 GPU를 조회한다."""
    executable = shutil.which("nvidia-smi")
    if executable is None and os.name == "nt":
        candidates = [Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/nvidia-smi.exe",
                      Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation/NVSMI/nvidia-smi.exe"]
        executable = next((str(path) for path in candidates if path.is_file()), None)
    if executable is None:
        return []
    try:
        result = subprocess.run([executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode:
        return []
    return [{"name": row[0].strip(), "driver": row[1].strip()}
            for row in csv.reader(result.stdout.splitlines()) if len(row) == 2]


def cuda_index(devices):
    """드라이버 세대에 맞는 공식 PyTorch 배포 인덱스를 선택한다."""
    if not devices:
        raise RuntimeError("NVIDIA GPU 감지 실패. NVIDIA 드라이버 설치와 GPU 연결 확인 필요")
    try:
        major = min(int(item["driver"].split(".")[0]) for item in devices)
    except (KeyError, ValueError):
        raise RuntimeError("NVIDIA 드라이버 버전 확인 불가") from None
    if major >= 580:
        return "cu130"
    if major >= (528 if os.name == "nt" else 525):
        return "cu126"
    raise RuntimeError("CUDA 학습을 위한 NVIDIA 드라이버 업데이트 필요. CPU로 전환하지 않습니다.")


def runtime_report(device="auto", check=False):
    """GPU 이름만 조회하지 않고 선택한 GPU에서 합성곱과 역전파까지 검사한다."""
    import torch
    report = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "device": device, "error": ""}
    if device == "auto":
        device = "cuda:0" if report["cuda_available"] else "cpu"
    report["device"] = device
    try:
        selected = torch.device(device)
        if selected.type == "cuda":
            if not report["cuda_available"]:
                raise RuntimeError("CUDA 사용 불가. CUDA용 PyTorch와 NVIDIA 드라이버 확인 필요")
            report["name"] = torch.cuda.get_device_name(selected)
            if check:
                with torch.enable_grad():
                    model = torch.nn.Conv2d(3, 4, 3).to(selected)
                    sample = torch.ones(2, 3, 16, 16, device=selected)
                    loss = model(sample).square().mean()
                    loss.backward()
                    if not torch.isfinite(model.weight.grad).all().item():
                        raise RuntimeError("CUDA 역전파 검사 결과 오류")
                    torch.cuda.synchronize(selected)
                report["backward_checked"] = True
        else:
            report["name"] = "CPU"
    except Exception as exc:
        report["error"] = str(exc)
    return report
