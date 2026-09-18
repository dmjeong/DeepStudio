"""
Deep Vision Studio — GPU/CPU 디바이스 관리

디바이스 선택 흐름:
┌─────────────────────────────────────────────────────────┐
│  DeviceManager — 전역 디바이스 감지 & 관리               │
│                                                          │
│  ┌────────────┐                                          │
│  │ detect()   │  시스템 GPU 자동 감지                     │
│  │            │  → CUDA 사용 가능 여부 확인                │
│  │            │  → GPU 이름/메모리 정보 수집               │
│  └────┬───────┘                                          │
│       │                                                  │
│  ┌────▼───────┐  ┌───────────────────────────────────┐  │
│  │ Training   │  │ Inference                          │  │
│  │ "auto"  │  │ "cpu" / "cuda"  │  │
│  │ → GPU 우선 │  │ → 사용자 선택 가능                   │  │
│  └────────────┘  └───────────────────────────────────┘  │
│                                                          │
│  Mixed Precision (AMP):                                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ GPU 사용 시 FP16 자동 활성화 → 메모리 절약 + 속도↑  │  │
│  │ CPU 사용 시 FP32 유지                               │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class GPUInfo:
    """GPU 정보 데이터 클래스"""
    index: int                    # GPU 인덱스 (0, 1, ...)
    name: str                     # GPU 이름 (e.g., "NVIDIA RTX 4090")
    total_memory_mb: float        # 전체 메모리 (MB)
    free_memory_mb: float         # 사용 가능 메모리 (MB)
    compute_capability: str       # 연산 능력 (e.g., "8.9")


class DeviceManager:
    """
    시스템 디바이스 감지 및 관리

    사용 예시:
    ┌────────────────────────────────────────────────────┐
    │  dm = DeviceManager()                              │
    │  dm.detect()                                       │
    │                                                    │
    │  # 학습용 (GPU 우선 자동 선택)                       │
    │  train_device = dm.get_device("auto")              │
    │                                                    │
    │  # 추론용 (사용자가 CPU 선택)                        │
    │  infer_device = dm.get_device("cpu")               │
    │                                                    │
    │  # Mixed Precision 사용 여부                        │
    │  use_amp = dm.supports_amp(train_device)            │
    └────────────────────────────────────────────────────┘
    """

    def __init__(self):
        self._cuda_available = False
        self._gpu_list: List[GPUInfo] = []
        self._detected = False
        # 초기 감지 수행
        self.detect()

    def detect(self):
        """
        시스템 GPU 감지

        Returns:
            bool: CUDA 사용 가능 여부
        """
        self._cuda_available = torch.cuda.is_available()
        self._gpu_list = []

        if self._cuda_available:
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                # 메모리 정보
                total_mem = props.total_memory / (1024 ** 2)  # → MB
                try:
                    # free_memory는 런타임에만 정확
                    free_mem = torch.cuda.mem_get_info(i)[0] / (1024 ** 2)
                except Exception:
                    free_mem = total_mem  # 할당 전이면 전체와 동일로 추정

                gpu_info = GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory_mb=total_mem,
                    free_memory_mb=free_mem,
                    compute_capability=f"{props.major}.{props.minor}",
                )
                self._gpu_list.append(gpu_info)

        self._detected = True
        return self._cuda_available

    @property
    def cuda_available(self) -> bool:
        """CUDA 사용 가능 여부"""
        return self._cuda_available

    @property
    def gpu_count(self) -> int:
        """감지된 GPU 수"""
        return len(self._gpu_list)

    @property
    def gpus(self) -> List[GPUInfo]:
        """감지된 GPU 목록"""
        return self._gpu_list

    def get_device(self, mode: str = "auto") -> torch.device:
        """명시한 GPU는 반드시 사용하며 사용 불가 시 오류를 반환한다."""
        from core.accelerator import nvidia_devices
        if mode == "cpu":
            return torch.device("cpu")
        if mode == "auto":
            if self._cuda_available:
                return torch.device("cuda:0")
            if nvidia_devices():
                raise RuntimeError(
                    "NVIDIA GPU는 감지됐지만 CUDA 실행 불가. CUDA용 패키지와 NVIDIA 드라이버 확인 필요. "
                    "웹은 start_web.bat --accelerator cuda로 실행하고, EXE는 최신 GPU 지원 릴리스를 사용하세요.")
            return torch.device("cpu")
        try:
            device = torch.device(mode)
        except (RuntimeError, ValueError):
            raise ValueError(f"지원하지 않는 장치: {mode}") from None
        if device.type != "cuda":
            raise ValueError(f"지원하지 않는 장치: {mode}")
        if not self._cuda_available:
            raise RuntimeError(
                "GPU 학습/추론을 요청했지만 CUDA 사용 불가. CPU로 전환하지 않습니다. "
                "CUDA용 PyTorch와 NVIDIA 드라이버 확인 필요")
        index = device.index if device.index is not None else 0
        if index >= self.gpu_count:
            raise ValueError(f"GPU {index} 없음. 사용 가능한 GPU 수: {self.gpu_count}")
        return torch.device(f"cuda:{index}")

    def get_device_label(self, device: torch.device) -> str:
        """
        디바이스의 사람이 읽을 수 있는 레이블

        예: "CPU" / "NVIDIA RTX 4090 (24.0 GB)"
        """
        if device.type == "cpu":
            return "CPU"

        idx = device.index if device.index is not None else 0
        if idx < len(self._gpu_list):
            gpu = self._gpu_list[idx]
            mem_gb = gpu.total_memory_mb / 1024
            return f"{gpu.name} ({mem_gb:.1f} GB)"

        return f"GPU:{idx}"

    def get_combo_items(self) -> list:
        """
        UI 콤보박스용 디바이스 목록 생성

        Returns:
            list of (label, mode_value) 튜플

        예시 반환:
        ┌────────────────────────────────────────────────┐
        │ ("자동 (GPU 우선)", "auto")                     │
        │ ("NVIDIA RTX 4090 (24.0 GB)", "cuda:0")        │
        │ ("CPU", "cpu")                                  │
        └────────────────────────────────────────────────┘
        """
        items = []

        if self._cuda_available:
            # GPU 있을 때: Auto → GPU 목록 → CPU
            items.append(("자동 (GPU 우선)", "auto"))
            for gpu in self._gpu_list:
                mem_gb = gpu.total_memory_mb / 1024
                label = f"{gpu.name} ({mem_gb:.1f} GB)"
                items.append((label, f"cuda:{gpu.index}"))
            items.append(("CPU", "cpu"))
        else:
            from core.accelerator import nvidia_devices
            if nvidia_devices():
                # CUDA 설치 오류가 저장된 GPU 선택을 CPU로 바꾸지 않도록 유지한다.
                items.extend([("자동 (GPU 우선)", "auto"),
                              ("GPU (CUDA 실행 환경 확인 필요)", "cuda:0"), ("CPU", "cpu")])
            else:
                items.append(("CPU (GPU 미감지)", "cpu"))

        return items

    def supports_amp(self, device: torch.device) -> bool:
        """
        Mixed Precision (AMP) 지원 여부

        AMP 조건:
        - CUDA 디바이스여야 함
        - Compute Capability ≥ 7.0 (Volta 이상)
        """
        if device.type != "cuda":
            return False

        idx = device.index if device.index is not None else 0
        if idx < len(self._gpu_list):
            major = int(self._gpu_list[idx].compute_capability.split(".")[0])
            return major >= 7  # Volta(7.0) 이상
        return False

    def get_status_text(self) -> str:
        """
        상태 바에 표시할 디바이스 정보 텍스트

        예: "GPU: NVIDIA RTX 4090 (24.0 GB) | CUDA 12.1"
            "CPU 전용 모드 (GPU 미감지)"
        """
        if self._cuda_available and self._gpu_list:
            gpu = self._gpu_list[0]
            mem_gb = gpu.total_memory_mb / 1024
            cuda_ver = torch.version.cuda or "N/A"
            return (
                f"GPU: {gpu.name} ({mem_gb:.1f} GB) | "
                f"CUDA {cuda_ver}"
            )
        from core.accelerator import nvidia_devices
        if nvidia_devices():
            return "NVIDIA GPU 감지됨 / CUDA 실행 불가: 패키지와 드라이버 확인 필요"
        return "CPU 전용 모드 (GPU 미감지)"

    def get_optimal_num_workers(self, device: torch.device) -> int:
        """
        DataLoader의 최적 num_workers 추천

        가이드라인:
        - GPU 학습: CPU 코어 수의 50% (최소 2, 최대 8)
        - CPU 학습: 0 (메인 프로세스에서 로딩)
        """
        if device.type == "cuda":
            import multiprocessing
            cores = multiprocessing.cpu_count()
            workers = max(2, min(cores // 2, 8))
            return workers
        return 0


# ── 전역 싱글턴 ─────────────────────────────────────
_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """전역 DeviceManager 싱글턴"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager
