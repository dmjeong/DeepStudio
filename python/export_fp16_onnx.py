"""Create and verify a deployable, single-graph FP16 ONNX with FP32 public tensors."""

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np


MIN_PIXEL_AGREEMENT = .95
MIN_REAL_FOREGROUND_IOU = .50


def _probes(cfg):
    shape = (1, cfg["input_channels"], cfg["input_height"], cfg["input_width"])
    rng = np.random.default_rng(42)
    yield "seeded", rng.standard_normal(shape, dtype=np.float32)
    yield "zero", np.zeros(shape, dtype=np.float32)
    if cfg.get("export", {}).get("dynamic_batch"):
        yield "batch_2", rng.standard_normal((2, *shape[1:]), dtype=np.float32)


def _real_segment_maps(cfg, image, compare):
    whole = cfg["preprocessing"].get("whole_image")
    if not whole:
        from dense_deployment import predict_dense_maps, region_pixels

        if cfg["preprocessing"].get("tiling"):
            return predict_dense_maps(cfg, image, compare)
        tensor = region_pixels(cfg, image)[None]
        return [outputs[0][0] for outputs in compare(tensor)]

    from center_crop import center_crop_box
    from quantize_onnx import preprocess

    x, y, right, bottom = center_crop_box((image.shape[1], image.shape[0]), cfg["preprocessing"].get("center_crop"))
    region = image[y:bottom, x:right]
    if region.dtype != np.uint8 or list(region.shape[1::-1]) != whole["region"]:
        raise ValueError(f"FP16 검증에는 C++ 판정 영역 {whole['region']} 크기의 8비트 영상이 필요합니다.")
    w, h = whole["scaled"]
    return [outputs[0][0, :, :h, :w] for outputs in compare(preprocess(cfg, region))]


def _max_error(left, right, atol, rtol, label):
    if left.shape != right.shape or left.dtype != np.float32 or right.dtype != np.float32:
        raise ValueError(f"FP16 검증 실패 ({label}): 출력 크기 또는 타입이 다릅니다.")
    # Compare a native-resolution output without allocating another full FP64 map.
    block = 262144
    left, right = left.reshape(-1), right.reshape(-1)
    maximum = 0.0
    for start in range(0, left.size, block):
        a, b = left[start:start + block], right[start:start + block]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"FP16 검증 실패 ({label}): 출력에 NaN 또는 Inf가 있습니다.")
        diff = np.abs(a.astype(np.float64) - b)
        limit = atol + rtol * np.abs(a)
        if np.any(diff > limit):
            i = int(np.argmax(diff / limit))
            raise ValueError(f"FP16 검증 실패 ({label}): 원소 {start + i} FP32={a[i]:.7g}, "
                             f"FP16={b[i]:.7g}, 차이 {diff[i]:.6g}, 허용 {limit[i]:.6g}.")
        maximum = max(maximum, float(diff.max()))
    return maximum


def _compare_masks(left, right, *, label, real_image):
    # Chunk label reduction so large segmentation tensors do not create full-size int64 masks.
    batch, classes, height, width = left.shape
    bins = np.zeros((3, classes), np.int64)
    step = max(1, 262144 // (batch * classes))
    rows = max(1, step // width)
    for start in range(0, height, rows):
        stop = start + rows
        a = left[:, :, start:stop].argmax(axis=1).ravel()
        b = right[:, :, start:stop].argmax(axis=1).ravel()
        bins[0] += np.bincount(a, minlength=classes)
        bins[1] += np.bincount(b, minlength=classes)
        bins[2] += np.bincount(a[a == b], minlength=classes)
    reference, converted, intersection = bins
    agreement = int(intersection.sum()) / int(reference.sum())
    if agreement < MIN_PIXEL_AGREEMENT:
        raise ValueError(f"FP16 검증 실패 ({label}): 분할 픽셀 일치율 {agreement:.4f} < {MIN_PIXEL_AGREEMENT}.")
    union = reference + converted - intersection
    present = np.flatnonzero(reference[1:]) + 1
    class_iou = {str(c): float(intersection[c] / union[c]) for c in present}
    if real_image:
        for c, score in class_iou.items():
            if score < MIN_REAL_FOREGROUND_IOU:
                raise ValueError(f"FP16 검증 실패 ({label}): 전경 클래스 {c} IoU {score:.4f} < "
                                 f"{MIN_REAL_FOREGROUND_IOU} (FP32 기준).")
    return {"pixel_agreement": agreement, "pixels": int(reference.sum()), "matched": int(intersection.sum()),
            "foreground_iou_by_class_id": class_iou}


def _verify(reference_config, reference, model_path, validation_dir):
    import onnxruntime as ort
    from onnx_session import cpu_session_options, deployment_session_settings

    names = reference.get("output_names") or [reference["output_name"]]
    settings = deployment_session_settings(reference)
    options = cpu_session_options(**settings)
    sessions = [ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                for path in (reference_config.parent / reference["model_path"], model_path)]
    for session in sessions:
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != reference["input_name"] or inputs[0].type != "tensor(float)"
                or [out.name for out in outputs] != names or any(out.type != "tensor(float)" for out in outputs)):
            raise ValueError("FP16 배포 그래프의 FP32 입력·출력 계약이 다릅니다.")

    atol, rtol = .01, .01
    report = {
        "reference": "verified_fp32_onnx", "onnxruntime_version": ort.__version__,
        "runtime_settings": settings, "atol": atol, "rtol": rtol,
        "probes": [], "validation_images": 0, "validation_windows": 0,
        "max_abs_error": {name: 0.0 for name in names},
    }
    segmentation = reference["task"] == "segment"
    if segmentation:
        report["segmentation"] = {
            "minimum_pixel_agreement": MIN_PIXEL_AGREEMENT,
            "minimum_real_foreground_iou": MIN_REAL_FOREGROUND_IOU,
            "scope": "model masks before defect filtering; merged valid pixels for real images",
            "synthetic": {}, "real_images": [],
        }

    def compare(tensor, label):
        feed = {reference["input_name"]: tensor}
        expected, actual = (session.run(names, feed) for session in sessions)
        for name, left, right in zip(names, expected, actual):
            error = _max_error(left, right, atol, rtol, f"{label}, {name}")
            report["max_abs_error"][name] = max(report["max_abs_error"][name], error)
        if reference["task"] == "classify":
            if not np.array_equal(expected[0].argmax(axis=1), actual[0].argmax(axis=1)):
                raise ValueError(f"FP16 검증 실패 ({label}): FP32와 분류 top-1이 다릅니다.")
        return expected, actual

    for probe, tensor in _probes(reference):
        expected, actual = compare(tensor, probe)
        report["probes"].append(probe)
        if segmentation:
            report["segmentation"]["synthetic"][probe] = _compare_masks(
                expected[0], actual[0], label=probe, real_image=False)
        del expected, actual

    if validation_dir:
        from classification_export_validation import collect_images
        from precision_benchmark import _classification_tensors, _read_image

        if reference["task"] not in {"classify", "segment"}:
            raise ValueError("FP16 실제 이미지 검증은 분류·세그멘테이션 모델에서 지원합니다.")
        for path in collect_images(validation_dir, reference.get("class_names")):
            label = str(path.relative_to(validation_dir))
            image = _read_image(path)
            def compare_window(tensor):
                report["validation_windows"] += 1
                return compare(tensor, label)
            if segmentation:
                a, b = _real_segment_maps(reference, image, compare_window)
                stats = _compare_masks(a[None], b[None], label=label, real_image=True)
                report["segmentation"]["real_images"].append({"image": label, **stats})
                del a, b
            else:
                for tensor in _classification_tensors(reference, image):
                    compare_window(tensor)
            report["validation_images"] += 1
    if segmentation:
        rows = [*report["segmentation"]["synthetic"].values(), *report["segmentation"]["real_images"]]
        report["pixel_agreement"] = sum(row["matched"] for row in rows) / sum(row["pixels"] for row in rows)
    if reference["task"] == "classify":
        report["top1_equal"] = True
    return report


def _publish(staged_model, staged_config, output):
    """Roll back the previous model if replacing its JSON fails (e.g. a Windows file lock)."""
    config_path = output.with_suffix(".json")
    backup = staged_model.parent / ".previous-model"
    if output.is_file():
        try:
            os.link(output, backup)
        except OSError:
            shutil.copyfile(output, backup)
    os.replace(staged_model, output)
    try:
        os.replace(staged_config, config_path)
    except OSError:
        if backup.exists():
            os.replace(backup, output)
        else:
            output.unlink()
        raise


def export_fp16(export_reference, destination, *, validation_dir, log):
    from benchmark_fp16 import convert_benchmark_fp16

    output = Path(destination)
    if output.suffix.lower() != ".onnx":
        raise ValueError("FP16 배포 모델의 출력 파일 확장자는 .onnx여야 합니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fp16-export-", dir=output.parent) as directory:
        stage = Path(directory)
        result = export_reference(stage / "reference.onnx")
        if result.get("output_dir") or result.get("backend") == "sam2":
            raise ValueError("SAM2 인코더·디코더 두 그래프의 FP16 내보내기는 지원하지 않습니다.")
        reference_config = Path(result["config_path"])
        reference = json.loads(reference_config.read_text(encoding="utf-8"))
        if reference.get("verification") != "passed":
            raise ValueError("FP16 배포에는 출력 검증을 통과한 FP32 기준 모델이 필요합니다.")
        log("FP16 배포 모델 변환 중 · ONNX 입출력은 FP32")
        converted = convert_benchmark_fp16(reference_config, stage / "converted")
        cfg = json.loads(converted.read_text(encoding="utf-8"))
        model = converted.parent / cfg["model_path"]
        log("FP16 CPU 검증 중 · 검증된 FP32 모델의 모든 출력과 비교")
        comparison = _verify(reference_config, reference, model, validation_dir)
        for key in ("spatial_execution", "feature_aggregation", "output_names"):
            if key in reference["export"]:
                cfg["export"][key] = reference["export"][key]
        cfg["export"].update(verification_policy="fp16_cpu_comparison_v1", fp16_comparison=comparison)
        cfg.update(model_path=output.name, verification="fp16_compared")
        converted.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        _publish(model, converted, output)
    log(f"FP16 배포 모델 저장 완료: {output}")
    return {"output_path": str(output), "config_path": str(output.with_suffix(".json")),
            "backend": cfg["backend"], "task": cfg["task"], "cpp_supported": cfg["cpp_supported"],
            "precision": "float16", "verification": cfg["verification"], "fp16_comparison": comparison,
            "file_size_mb": output.stat().st_size / (1024 * 1024)}
