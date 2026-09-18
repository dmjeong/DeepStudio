"""CPU 측정 집계. 준비 시간과 첫 실행은 반복 추론 시간에서 분리한다."""

import math
import platform
import time
import numpy as np


def latency_summary(seconds):
    values = np.asarray(seconds, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("측정 시간은 비어 있지 않은 유한한 양수 또는 0 필요")
    return {"samples": int(values.size), "mean_ms": float(values.mean() * 1000),
            "p50_ms": float(np.percentile(values, 50) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
            "min_ms": float(values.min() * 1000), "max_ms": float(values.max() * 1000)}


def measure_inference(engine, paths, *, warmup=3, repeats=20, prepare_measurement=None):
    if not paths or warmup < 0 or repeats < 1:
        raise ValueError("이미지와 유효한 워밍업/반복 횟수 필요")
    records = []
    for index in range(warmup + repeats):
        if index == warmup and prepare_measurement is not None:
            prepare_measurement()
        path = str(paths[index % len(paths)])
        started = time.perf_counter()
        result = engine.infer(path)
        total = time.perf_counter() - started
        if result.status == "error":
            raise RuntimeError(f"측정 이미지 추론 실패: {path}: {result.error}")
        if result.inference_sec is None or not math.isfinite(result.inference_sec):
            raise ValueError("모델 추론 단계 시간 없음")
        if index >= warmup:
            records.append({"image": path, "iteration": index - warmup + 1,
                            "total_sec": total, "inference_sec": result.inference_sec,
                            "gradcam_sec": result.gradcam_sec, "gradcam_status": result.gradcam_status})
    summaries = {name: latency_summary([row[name + "_sec"] for row in records])
                 for name in ("total", "inference")}
    cam = [row["gradcam_sec"] for row in records if row["gradcam_status"] == "completed" and row["gradcam_sec"] is not None]
    summaries["gradcam"] = latency_summary(cam) if cam else None
    return {"environment": {"platform": platform.platform(), "machine": platform.machine(),
                            "processor": platform.processor(), "python": platform.python_version()},
            "warmup": warmup, "repeats": repeats, "batch_size": 1,
            "scope": "image read, preprocessing, inference, postprocessing and preview preparation; excludes UI and disk cache",
            "summary": summaries, "records": records}
