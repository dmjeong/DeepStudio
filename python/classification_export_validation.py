"""Classification-specific FP32 acceptance using a local image corpus.

This is an explicitly named deployment policy, not a relaxation of the strict
logit comparator. It certifies the inspected images, not every possible input.
"""
import hashlib
from pathlib import Path
import shutil

import numpy as np
import torch

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PROBABILITY_ATOL = .001


def compare_classification(reference, actual, probability_atol=PROBABILITY_ATOL):
    ref, out = np.asarray(reference, np.float64), np.asarray(actual, np.float64)
    if ref.shape != out.shape or ref.ndim != 2 or not ref.size:
        raise ValueError("분류 출력 형태 불일치")
    if not np.isfinite(ref).all() or not np.isfinite(out).all():
        raise ValueError("분류 출력에 NaN 또는 무한대")
    def probabilities(logits):
        value = np.exp(logits - logits.max(axis=1, keepdims=True))
        return value / value.sum(axis=1, keepdims=True)
    rp, ap = probabilities(ref), probabilities(out)
    difference = np.abs(rp - ap).max(axis=1)
    if not np.array_equal(ref.argmax(axis=1), out.argmax(axis=1)):
        raise ValueError(f"분류 판정 불일치: PyTorch={ref.argmax(axis=1).tolist()}, ONNX={out.argmax(axis=1).tolist()}")
    if np.any(difference > probability_atol):
        raise ValueError(f"분류 확률 차이 초과: {difference.max():.8g} > {probability_atol}")
    if ref.shape[1] > 1:
        ordered = np.sort(rp, axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        if np.any(margin < 2 * difference):
            raise ValueError(f"분류 경계 근처 입력: 최소 1·2등 간격={margin.min():.8g}, 최대 확률 오차={difference.max():.8g}")
    return {"max_probability_error": float(difference.max()),
            "max_logit_error": float(np.abs(ref - out).max())}


def collect_images(directory, class_names):
    root = Path(directory)
    if not root.is_dir():
        raise ValueError("실제 이미지 검증 폴더가 없습니다")
    records = []
    for name in class_names:
        # Require the entire set of model classes; never silently certify one class.
        folder = root / name
        if Path(name).name != name or not folder.is_dir():
            raise ValueError(f"검증 폴더에 클래스 하위 폴더가 필요합니다: {name}")
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"검증 이미지가 없는 클래스: {name}")
        records.extend(files)
    if not records:
        raise ValueError("검증 이미지가 없습니다")
    return records


def export_validated_fp32(reference, spec, path, dummy, *, validation_dir,
                          opset, dynamic_batch, log):
    import onnxruntime as ort
    from efficientnet import prepare_for_inference
    from export_onnx import export_to_onnx, create_inference_config
    from onnx_classifier import ImageClassifier
    from onnx_session import cpu_session_options
    from opencv_preprocess import read_image

    files = collect_images(validation_dir, spec["class_names"])
    config = create_inference_config(Path(path).parent, "classify", spec["num_classes"],
        (spec["input_height"], spec["input_width"]), spec["in_channels"], Path(path).name,
        class_names=spec["class_names"], preprocessing=spec["preprocessing"], backend="efficientnet",
        config_filename=".validation-config.json", architecture=reference.architecture,
        model_config=reference.checkpoint_config())
    processor = ImageClassifier(config)
    expected, hashes = [], []
    log(f"실제 이미지 분류 검증: {len(files)}장, 모든 클래스 포함. 판정 일치·확률 오차 0.1%p 이하 검사")
    with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
        for file in files:
            pixels = read_image(file, spec["in_channels"])
            hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
            expected.append(reference(torch.from_numpy(processor.preprocess(pixels))).numpy().copy())
        probes = [dummy, torch.zeros_like(dummy)]
        if dynamic_batch:
            probes.append(dummy.repeat(2, 1, 1, 1))
        probe_reference = [reference(probe).numpy().copy() for probe in probes]

    class Runtime(ImageClassifier):
        def logits(self, tensor):
            self.last_logits = self.session.run(None, {self.input_name: tensor})[0]
            return self.last_logits

    candidate_models = [("fused_fp32", prepare_for_inference(reference)), ("unfused_fp32", reference)]
    available_threads = max(1, torch.get_num_threads())
    profiles = [(level, threads) for level in ("all", "basic", "disabled")
                for threads in dict.fromkeys((min(4, available_threads), min(2, available_threads), 1))]
    attempts, winner, best_key = [], None, (True, float("inf"))
    best_path = Path(path).with_name(".validated-fp32.onnx")
    for graph, candidate in candidate_models:
        export_to_onnx(candidate, dummy, path, opset, dynamic_batch, constant_folding=graph == "fused_fp32")
        for level, threads in profiles:
            settings = {"graph_optimization_level": level, "num_threads": threads}
            row = {"graph": graph, **settings, "passed": False}
            runtime = None
            try:
                runtime = Runtime(config)
                runtime.session = ort.InferenceSession(str(path), cpu_session_options(**settings), providers=["CPUExecutionProvider"])
                runtime.input_name = runtime.session.get_inputs()[0].name
                probability_error, logit_error = 0., 0.
                for probe_index, (probe, ref) in enumerate(zip(probes, probe_reference)):
                    row["checking"] = ["seeded", "zero", "batch2"][probe_index]
                    detail = compare_classification(ref, runtime.logits(probe.numpy()))
                    probability_error = max(probability_error, detail["max_probability_error"])
                    logit_error = max(logit_error, detail["max_logit_error"])
                for _ in range(3):
                    runtime.logits(dummy.numpy())
                times = []
                for index, file in enumerate(files):
                    row["checking"] = f"image_{index + 1}"
                    pixels = read_image(file, spec["in_channels"])
                    if hashlib.sha256(pixels.tobytes()).hexdigest() != hashes[index]:
                        raise ValueError("검증 중 이미지가 변경되었습니다")
                    result = runtime.predict_rgb(pixels)
                    detail = compare_classification(expected[index], runtime.last_logits)
                    probability_error = max(probability_error, detail["max_probability_error"])
                    logit_error = max(logit_error, detail["max_logit_error"])
                    times.append(result["total_ms"])
                median, p95 = float(np.median(times)), float(np.percentile(times, 95))
                row.update(passed=True, image_count=len(files), max_probability_error=probability_error,
                           max_logit_error=logit_error, pipeline_median_ms=median, pipeline_p95_ms=p95,
                           pipeline_max_ms=float(max(times)))
                row.pop("checking", None)
                # Prefer meeting the observed per-image target, then lower p95.
                key = (row["pipeline_max_ms"] > 8., p95)
                if key < best_key:
                    shutil.copyfile(path, best_path)
                    winner, best_key = (graph, candidate, settings, dict(row)), key
                log(f"FP32 후보 {graph}/{level}/{threads}T: 모든 이미지 판정 일치, 전체 처리 중앙값 {median:.2f}ms / p95 {p95:.2f}ms")
            except Exception as exc:
                row["reason"] = str(exc)
                log(f"FP32 후보 제외 {graph}/{level}/{threads}T: {exc}")
            finally:
                runtime = None  # Do not keep idle ORT thread pools alive while profiling.
                attempts.append(row)
    if winner is None:
        raise ValueError("실제 이미지 분류 검증을 통과한 FP32 후보가 없습니다. 전체 FP64로 자동 전환하지 않습니다. "
                         + " | ".join(row.get("reason", "") for row in attempts))
    graph, model, settings, result = winner
    shutil.copyfile(best_path, path)
    best_path.unlink()
    # Reopen the exact chosen artifact and verify every image once more.
    runtime = Runtime(config)
    runtime.session = ort.InferenceSession(str(path), cpu_session_options(**settings), providers=["CPUExecutionProvider"])
    runtime.input_name = runtime.session.get_inputs()[0].name
    for probe, ref in zip(probes, probe_reference):
        compare_classification(ref, runtime.logits(probe.numpy()))
    for index, file in enumerate(files):
        pixels = read_image(file, spec["in_channels"])
        if hashlib.sha256(pixels.tobytes()).hexdigest() != hashes[index]:
            raise ValueError("최종 검증 중 이미지가 변경되었습니다")
        compare_classification(expected[index], runtime.logits(processor.preprocess(pixels)))
    report = {"policy": "classification_dataset_v1", "image_count": len(files),
              "criteria": {"top1_equal": True, "probability_atol": PROBABILITY_ATOL,
                           "minimum_margin_error_ratio": 2},
              "dataset_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
              "scope": "provided_images_and_synthetic_probes", "custom_thresholds_validated": False,
              "selected": result, "attempts": attempts,
              "latency_scope": "preprocess_inference_postprocess_excluding_file_decode",
              "target_ms": 8., "target_metric": "max_observed",
              "target_met": result["pipeline_max_ms"] <= 8.}
    # Reference is never mutated, including its metadata.
    import copy
    model = copy.deepcopy(model)
    model.inference_optimization = {**getattr(model, "inference_optimization", {}),
        "fallback": graph, "compute_precision": "float32", "io_precision": "float32"}
    return model, settings, report
