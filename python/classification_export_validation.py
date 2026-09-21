"""Classification-specific FP32 acceptance using images or deterministic probes.

These are explicitly named deployment policies, not a silent relaxation of the
strict logit comparator. Each report states the exact inputs it certifies.
"""
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
import torch

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PROBABILITY_ATOL = .001
MINIMUM_PROBABILITY_MARGIN = 1e-6


def compare_classification(reference, actual, probability_atol=PROBABILITY_ATOL,
                           minimum_margin=MINIMUM_PROBABILITY_MARGIN):
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
        reference_ordered = np.sort(rp, axis=1)
        actual_ordered = np.sort(ap, axis=1)
        margin = reference_ordered[:, -1] - reference_ordered[:, -2]
        actual_margin = actual_ordered[:, -1] - actual_ordered[:, -2]
        required = np.maximum(2 * difference, minimum_margin)
        if np.any(margin <= required) or np.any(actual_margin <= minimum_margin):
            raise ValueError(f"분류 경계 근처 입력: 최소 PyTorch 1·2등 간격={margin.min():.8g}, "
                             f"최소 ONNX 간격={actual_margin.min():.8g}, 최대 확률 오차={difference.max():.8g}")
    else:
        margin = actual_margin = np.ones(ref.shape[0], dtype=np.float64)
    return {"max_probability_error": float(difference.max()),
            "max_logit_error": float(np.abs(ref - out).max()),
            "min_reference_margin": float(margin.min()),
            "min_actual_margin": float(actual_margin.min())}


def synthetic_classification_probes(dummy, preprocessing):
    """Build deterministic probes constrained to the declared image domain."""
    channels, height, width = dummy.shape[1:]
    mean = np.asarray(preprocessing.get("normalize_mean", preprocessing.get("mean")), np.float32)
    std = np.asarray(preprocessing.get("normalize_std", preprocessing.get("std")), np.float32)
    if mean.shape != (channels,) or std.shape != (channels,) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("분류 합성 검증용 정규화 메타데이터 오류")

    def normalize(raw):
        raw = np.asarray(raw, np.float32)
        if raw.shape == (height, width):
            raw = np.broadcast_to(raw, (channels, height, width)).copy()
        return torch.from_numpy((raw - mean[:, None, None]) / std[:, None, None]).unsqueeze(0)

    probes = [("seeded_normal", dummy.detach().cpu().float()),
              ("normalized_zero", torch.zeros_like(dummy).cpu())]
    for name, value in (("black", 0.), ("midgray", .5), ("white", 1.)):
        probes.append((name, normalize(np.full((channels, height, width), value, np.float32))))
    horizontal = np.linspace(0., 1., width, dtype=np.float32)[None, :]
    vertical = np.linspace(0., 1., height, dtype=np.float32)[:, None]
    probes.extend((
        ("horizontal_gradient", normalize(np.broadcast_to(horizontal, (height, width)))),
        ("vertical_gradient", normalize(np.broadcast_to(vertical, (height, width)))),
        ("checkerboard", normalize((np.indices((height, width)).sum(axis=0) % 2).astype(np.float32))),
    ))
    for seed in (7, 29, 101, 997):
        raw = np.random.default_rng(seed).random((channels, height, width), dtype=np.float32)
        probes.append((f"valid_random_{seed}", normalize(raw)))
    return probes


def export_validated_synthetic_fp32(reference, spec, path, dummy, *, opset,
                                    dynamic_batch, log):
    """Select a fast FP32 graph using classification decisions on robust probes.

    This fallback certifies only the deterministic probe suite. A supplied image
    corpus continues to use ``classification_dataset_v1`` and is stronger.
    """
    import copy
    import onnxruntime as ort
    from efficientnet import prepare_for_inference, prepare_native_bn_export
    from export_onnx import export_to_onnx
    from onnx_session import cpu_session_options

    probes = synthetic_classification_probes(dummy, spec["preprocessing"])
    if dynamic_batch:
        probes.append(("dynamic_batch_2", torch.cat((probes[2][1], probes[4][1]), dim=0)))
    with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
        expected = [(name, reference(probe).detach().cpu().numpy()) for name, probe in probes]
    threads = max(1, min(torch.get_num_threads(), 4))
    thread_counts = list(dict.fromkeys((threads, 1)))
    candidates = [
        ("fused_fp32", prepare_for_inference(reference), ("all", "basic", "disabled")),
        ("unfused_fp32", copy.deepcopy(reference).cpu().float().eval(), ("all", "basic", "disabled")),
        ("native_batch_norm_affine", prepare_native_bn_export(reference, emulate_fma=False), ("disabled",)),
        ("native_batch_norm_affine_fma", prepare_native_bn_export(reference, emulate_fma=True), ("disabled",)),
    ]
    attempts, winner = [], None
    best_key = (float("inf"), float("inf"))
    best_path = Path(path).with_name(".synthetic-validated-fp32.onnx")
    log(f"분류 판정 기반 FP32 검증: 유효 픽셀 범위의 결정론적 입력 {len(probes)}개, "
        "top-1·softmax 확률·판정 마진 검사")
    try:
        for graph, candidate, levels in candidates:
            constant_folding = graph == "fused_fp32"
            try:
                export_to_onnx(candidate, dummy, path, opset, dynamic_batch,
                               constant_folding=constant_folding)
            except Exception as exc:
                attempts.append({"graph": graph, "passed": False, "reason": str(exc)})
                continue
            for level in levels:
                for count in thread_counts:
                    settings = {"graph_optimization_level": level, "num_threads": count}
                    row = {"graph": graph, **settings, "passed": False,
                           "policy": "classification_synthetic_probe_v1"}
                    session = None
                    try:
                        session = ort.InferenceSession(str(path), cpu_session_options(**settings),
                                                       providers=["CPUExecutionProvider"])
                        input_name = session.get_inputs()[0].name
                        probability_error = logit_error = 0.
                        reference_margin = actual_margin = float("inf")
                        for (name, probe), (_, ref) in zip(probes, expected):
                            row["checking"] = name
                            out = session.run(None, {input_name: probe.numpy()})[0]
                            detail = compare_classification(ref, out)
                            probability_error = max(probability_error, detail["max_probability_error"])
                            logit_error = max(logit_error, detail["max_logit_error"])
                            reference_margin = min(reference_margin, detail["min_reference_margin"])
                            actual_margin = min(actual_margin, detail["min_actual_margin"])
                        feed = {input_name: dummy.detach().cpu().numpy()}
                        for _ in range(2):
                            session.run(None, feed)
                        timings = []
                        for _ in range(7):
                            started = time.perf_counter()
                            session.run(None, feed)
                            timings.append((time.perf_counter() - started) * 1000)
                        latency = float(np.median(timings))
                        row.update(passed=True, probe_count=len(probes),
                                   max_probability_error=probability_error,
                                   max_logit_error=logit_error,
                                   min_reference_margin=reference_margin,
                                   min_actual_margin=actual_margin,
                                   model_median_ms=latency)
                        row.pop("checking", None)
                        key = (latency, probability_error)
                        if key < best_key:
                            shutil.copyfile(path, best_path)
                            winner, best_key = (graph, candidate, dict(settings), dict(row)), key
                        log(f"판정 검증 통과 {graph}/{level}/{count}T: 모델 추론 중앙값 {latency:.2f}ms, "
                            f"최대 확률 차이 {probability_error:.8g}")
                    except Exception as exc:
                        row["reason"] = str(exc)
                    finally:
                        session = None
                        attempts.append(row)
        if winner is None:
            raise ValueError("결정론적 분류 판정 검증을 통과한 FP32 후보가 없습니다: " +
                             " | ".join(row.get("reason", "") for row in attempts if row.get("reason")))
        graph, model, settings, selected = winner
        shutil.copyfile(best_path, path)
        # Reload the exact artifact that will be shipped and repeat all probes.
        session = ort.InferenceSession(str(path), cpu_session_options(**settings),
                                       providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        for (name, probe), (_, ref) in zip(probes, expected):
            try:
                compare_classification(ref, session.run(None, {input_name: probe.numpy()})[0])
            except ValueError as exc:
                raise ValueError(f"최종 FP32 후보 재검증 실패 ({name}): {exc}") from exc
        model = copy.deepcopy(model)
        model.inference_optimization = {**getattr(model, "inference_optimization", {}),
            "fallback": graph, "compute_precision": "float32", "io_precision": "float32",
            "classification_policy": "classification_synthetic_probe_v1"}
        report = {"policy": "classification_synthetic_probe_v1", "image_count": 0,
                  "probe_count": len(probes),
                  "criteria": {"top1_equal": True, "probability_atol": PROBABILITY_ATOL,
                               "minimum_margin_error_ratio": 2,
                               "minimum_probability_margin": MINIMUM_PROBABILITY_MARGIN},
                  "scope": "deterministic_synthetic_probes_only",
                  "custom_thresholds_validated": False,
                  "selected": selected, "attempts": attempts}
        return model, settings, report
    finally:
        if best_path.exists():
            best_path.unlink()


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
