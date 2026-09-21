"""추론과 Grad-CAM 계산 시간을 장치 동기화로 분리 측정한다."""

import math
import time


class StageTimer:
    """단계 완료 시간만 제공하며 실패나 미실행을 0초로 바꾸지 않는다."""

    def __init__(self, device, torch_module=None, clock=None):
        self.device = device
        self._torch = torch_module
        self._clock = clock or time.perf_counter
        self.elapsed_sec = None
        self.failed = False
        self._started = None

    @property
    def status(self):
        if self.failed:
            return "error"
        return "completed" if self.elapsed_sec is not None else "not_run"

    def _synchronize(self):
        device_type = getattr(self.device, "type", str(self.device).split(":")[0])
        if device_type not in ("cuda", "mps"):
            return
        backend = self._torch
        if backend is None:
            import torch
            backend = torch
        if device_type == "cuda":
            backend.cuda.synchronize(self.device)
        else:
            backend.mps.synchronize()

    def __enter__(self):
        self.elapsed_sec = None
        self.failed = False
        try:
            self._synchronize()
            self._started = self._clock()
        except Exception:
            self.failed = True
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.failed = True
            return False
        try:
            self._synchronize()
            elapsed = self._clock() - self._started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("단계 시간 측정값 오류")
            self.elapsed_sec = elapsed
        except Exception:
            self.failed = True
            raise
        return False


def format_duration(value, status="not_run"):
    """시간 없음과 유효한 0초를 구분해 표시한다."""
    if value is None:
        return {"unsupported": "해당 없음", "error": "실패"}.get(status, "미실행")
    if not math.isfinite(value) or value < 0:
        return "측정 불가"
    return f"{value * 1000:.1f} ms" if value < 1 else f"{value:.2f} 초"


def format_result_timing(result, include_total=False):
    inference = format_duration(result.inference_sec, result.inference_status)
    gradcam = format_duration(result.gradcam_sec, result.gradcam_status)
    text = f"추론 {inference} | Grad-CAM {gradcam}"
    details = getattr(result, "details", None) or {}
    if details.get("runtime"):
        runtime = {"onnxruntime": "ONNX Runtime", "pytorch": "PyTorch"}.get(details["runtime"], details["runtime"])
        text += f" | {runtime} {details.get('device', '')}"
        if details.get("runtime_threads"):
            text += f" ({details['runtime_threads']} threads)"
        level = details.get("runtime_optimization")
        if level in ("basic", "disabled"):
            text += " [기본 최적화]" if level == "basic" else " [최적화 꺼짐]"
        if details.get("runtime_warning"):
            if details["runtime"] == "pytorch":
                text += " [ONNX 준비 실패 → PyTorch]"
            elif details.get("runtime_compute_precision") == "float64":
                text += " [고정밀 FP64 · 속도 저하 가능]"
            else:
                text += " [실행 경고]"
    if include_total:
        text += f" | 전체 처리 {format_duration(result.elapsed_sec)}"
    return text


def format_runtime_stages(result):
    """Only display measured CPU stages; never replace the total with model-only time."""
    values = (getattr(result, "details", None) or {}).get("timing_ms", {})
    labels = (("decode", "파일 읽기"), ("preprocess", "전처리"), ("model", "모델"), ("postprocess", "후처리"))
    parts = [f"{label} {values[key]:.2f} ms" for key, label in labels if key in values]
    return " | ".join(parts) + "\n" if parts else ""


def batch_stage_summary(results, stage):
    """완료한 이미지의 시간만 평균에 넣고 분모와 실패 수를 드러낸다."""
    if stage not in ("inference", "gradcam"):
        raise ValueError("알 수 없는 추론 시간 단계")
    results = list(results)
    label = "추론" if stage == "inference" else "Grad-CAM"
    values = [getattr(result, f"{stage}_sec") for result in results
              if getattr(result, f"{stage}_status") == "completed"
              and getattr(result, f"{stage}_sec") is not None
              and math.isfinite(getattr(result, f"{stage}_sec"))
              and getattr(result, f"{stage}_sec") >= 0]
    if values:
        return f"{label} 평균 {format_duration(sum(values) / len(values))} (완료 {len(values)}/{len(results)}장)"
    statuses = [getattr(result, f"{stage}_status") for result in results]
    failures = statuses.count("error")
    if failures:
        return f"{label} 완료 없음 (실패 {failures}장)"
    if statuses and all(status == "unsupported" for status in statuses):
        return f"{label} 해당 없음"
    return f"{label} 미실행"
