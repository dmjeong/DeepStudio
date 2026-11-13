"""Compare exported ONNX precisions on the same images and CPU thread counts."""

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
from center_crop import center_crop_box
from onnx_session import cpu_session_options, deployment_session_settings
from opencv_preprocess import convert_channels, resize


def parse_thread_counts(text):
    fields = [part.strip() for part in text.split(",")]
    if not all(re.fullmatch(r"[0-9]{1,3}", part) for part in fields):
        raise ValueError("CPU 스레드 수를 쉼표로 구분해 입력하세요. 예) 1, 2, 4, 8")
    return _thread_counts(tuple(map(int, fields)))


def _thread_counts(counts):
    if not counts or any(type(n) is not int or not 0 <= n <= 256 for n in counts):
        raise ValueError("CPU 스레드 수는 0(자동)~256 사이여야 합니다.")
    return tuple(dict.fromkeys(counts))


def _manifest(path, precision, reference=None):
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not cfg.get("cpp_supported") or cfg.get("encryption"):
        raise ValueError("암호화하지 않은 C++ 배포용 ONNX 설정 JSON을 선택하세요.")
    export = cfg.get("export", {})
    declared = export.get("precision", "float32")
    if declared != precision:
        raise ValueError(f"설정 파일의 정밀도가 {precision}과 다릅니다 ({declared}).")
    if precision == "float32" and export.get("compute_precision", "float32") != "float32":
        raise ValueError("FP32 기준에는 FP64 보정 없이 내보낸 모델이 필요합니다.")
    fields = ("task", "backend", "input_channels", "input_height", "input_width", "num_classes",
              "class_names", "normalize_mean", "normalize_std", "preprocessing", "postprocessing", "patches")
    if reference and any(cfg.get(key) != reference.get(key) for key in fields):
        raise ValueError("FP32와 모델 입출력·전처리·판정 설정이 다릅니다. 같은 모델의 내보내기 파일을 선택하세요.")
    return cfg


class _Runner:
    """Time ORT calls inside one image, including every tile."""

    def __init__(self, config_path, cfg, threads):
        import onnxruntime as ort

        self.cfg = cfg
        self.names = cfg.get("output_names") or [cfg["output_name"]]
        settings = deployment_session_settings(cfg)
        settings["num_threads"] = threads
        model = Path(config_path).parent / cfg["model_path"]
        self.session = ort.InferenceSession(str(model), cpu_session_options(**settings),
                                            providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        shape = [1, cfg["input_channels"], cfg["input_height"], cfg["input_width"]]
        if (len(inputs) != 1 or inputs[0].name != cfg["input_name"] or inputs[0].type != "tensor(float)"
                or len(inputs[0].shape) != 4
                or any(isinstance(v, int) and v != size for v, size in zip(inputs[0].shape, shape))
                or [o.name for o in outputs] != self.names or any(o.type != "tensor(float)" for o in outputs)):
            raise ValueError("ONNX와 JSON의 텐서 이름, 입력 크기 또는 FP32 입출력 계약이 다릅니다.")
        self.model_seconds = 0.0

    def run(self, names, feed):
        start = perf_counter()
        outputs = self.session.run(names, feed)
        self.model_seconds += perf_counter() - start
        return outputs

    def _infer(self, tensor):
        outputs = self.run(self.names, {self.cfg["input_name"]: tensor})
        if not all(np.isfinite(value).all() for value in outputs):
            raise ValueError("ONNX 출력에 NaN/Inf가 있습니다.")
        return outputs

    def predict(self, path):
        cfg = self.cfg
        task = cfg["task"]
        if task == "detect":
            from detection_quantization import DetectionInputs, _predict
            return _predict(cfg, self, DetectionInputs(cfg, path))

        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or image.size == 0:
            raise ValueError(f"이미지 읽기 실패: {path.name}")
        prep = cfg["preprocessing"]
        if task == "segment" and prep.get("whole_image"):
            from quantize_onnx import judgement_region, preprocess
            region = judgement_region(cfg, image)
            if region is None:
                raise ValueError(f"이미지가 판정 영역보다 작습니다: {path.name}")
            width, height = prep["whole_image"]["scaled"]
            output = self._infer(preprocess(cfg, region))[0]
            if output.shape != (1, cfg["num_classes"], cfg["input_height"], cfg["input_width"]):
                raise ValueError("분할 ONNX 출력 크기가 설정과 다릅니다.")
            return _mask(output[0, :, :height, :width], cfg, (region.shape[1], region.shape[0]))

        if task == "segment" or (task == "anomaly" and cfg["backend"] == "dinomaly" and prep.get("tiling")):
            from dense_deployment import predict_dense
            result = predict_dense(cfg, image, self._infer)
            if task == "segment":
                crop = prep.get("center_crop")
                size = (crop["width"], crop["height"]) if crop else (image.shape[1], image.shape[0])
                return _mask(result, cfg, size, probabilities=bool(prep.get("tiling")))
            return float(result.max())

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if image.shape[2] == 3 else cv2.COLOR_BGRA2RGBA)
        x, y, right, bottom = center_crop_box((image.shape[1], image.shape[0]), prep.get("center_crop"))
        image = image[y:bottom, x:right]
        if image.dtype == np.uint16 and cfg["backend"] == "patchcore" and cfg["input_channels"] == 3:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = convert_channels(image, cfg["input_channels"])
        if task == "classify":
            from native_patches import scaled_image, tiles, validate_patches
            patches = validate_patches(cfg.get("patches"), "classify")
            if patches:
                pixels = scaled_image(image, patches)
                logits = None
                for tile, _ in tiles(pixels, patches):
                    part = self._classification_logits(tile)
                    logits = part if logits is None else np.maximum(logits, part)
            else:
                logits = self._classification_logits(image)
            probabilities = np.exp(logits.astype(np.float64) - np.max(logits))
            return int(np.argmax(logits)), probabilities / probabilities.sum()

        tensor = self._tensor(image)
        outputs = self._infer(tensor)
        if cfg["backend"] in {"patchcore", "dinomaly"}:
            if outputs[0].shape != (1,) or len(outputs) != 2 or outputs[1].ndim != 4:
                raise ValueError("이상 탐지 ONNX 출력 크기가 설정과 다릅니다.")
            return float(outputs[0][0])
        if cfg["backend"] == "custom" and outputs[0].shape == tensor.shape:
            return float(np.abs(tensor - outputs[0]).mean())
        raise ValueError("지원하지 않는 이상 탐지 출력 형식입니다.")

    def _tensor(self, image):
        cfg = self.cfg
        pixels = resize(image, (cfg["input_height"], cfg["input_width"]))
        scale = 65535 if pixels.dtype == np.uint16 else 255
        pixels = pixels.reshape(*pixels.shape[:2], cfg["input_channels"]).astype(np.float32) / scale
        pixels = (pixels - np.asarray(cfg["normalize_mean"], np.float32)) / np.asarray(cfg["normalize_std"], np.float32)
        return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])

    def _classification_logits(self, image):
        value = self._infer(self._tensor(image))[0]
        if value.shape != (1, self.cfg["num_classes"]):
            raise ValueError("분류 ONNX 출력 클래스 수가 설정과 다릅니다.")
        return value[0]


def _mask(values, cfg, size, *, probabilities=False):
    mask = values.argmax(0).astype(np.uint8 if cfg["num_classes"] <= 256 else np.int32)
    rules = cfg.get("postprocessing", {}).get("defect_rules")
    if not rules:
        return mask, None
    from core.segment_defects import analyze_segmentation, judge

    if not probabilities:
        values = np.exp(values - values.max(0, keepdims=True))
        values /= values.sum(0, keepdims=True)
    findings = analyze_segmentation(mask, values, size=size, class_names=cfg["class_names"],
                                    num_classes=cfg["num_classes"])
    thresholds = {key: rules[key] for key in ("min_area", "min_score")}
    thresholds["per_class"] = {rule["class_name"]: rule for rule in rules.get("classes", [])}
    return mask, bool(judge(findings, thresholds)["kept"])


def _comparison(cfg, expected, actual):
    task = cfg["task"]
    if task == "detect":
        from detection_quantization import _pair_detections
        ious, _ = _pair_detections(expected, actual)
        return len(expected) - len(ious), len(actual) - len(ious), int(bool(expected) != bool(actual))
    if task == "segment":
        (a, left), (b, right) = expected, actual
        if a.shape != b.shape:
            raise ValueError("비교할 분할 마스크의 크기가 다릅니다.")
        fg = (a != 0) | (b != 0)
        return int((a == b).sum()), a.size, int(((a == b) & fg).sum()), int(fg.sum()), int(left != right)
    if task == "classify":
        return int(expected[0] != actual[0]), float(np.abs(expected[1] - actual[1]).max())
    threshold = cfg["postprocessing"]["threshold"]
    return int((expected >= threshold) != (actual >= threshold)), abs(expected - actual)


def _summarize(cfg, comparisons):
    if not comparisons:
        return "기준"
    task = cfg["task"]
    totals = np.sum(comparisons, axis=0)
    if task == "detect":
        lost, gained, changed = map(int, totals)
        return f"박스 누락 {lost} · 추가 {gained} · 판정 변경 {changed}장"
    if task == "segment":
        same, count, fg_same, fg_count, changed = totals
        foreground = f"{fg_same / fg_count:.2%}" if fg_count else "전경 없음"
        verdict = f" · 판정 변경 {int(changed)}장" if cfg.get("postprocessing", {}).get("defect_rules") else ""
        return f"픽셀 일치 {same / count:.2%} · 전경 일치/합집합 {foreground}{verdict}"
    changed = int(totals[0])
    error = max(row[1] for row in comparisons)
    if task == "classify":
        return f"Top-1 변경 {changed}장 · 확률차 최대 {100 * error:.3f}%p"
    return f"판정 변경 {changed}장 · 점수차 최대 {error:.5g}"


def _measure(path, cfg, threads, paths, warmup, repeats, baseline, cancelled, cache):
    runner = _Runner(path, cfg, threads)
    total_ms, model_ms, comparisons = [], [], []
    for index in range(warmup):
        if cancelled():
            return None
        runner.predict(paths[index % len(paths)])
    reference = baseline is None
    outputs = [] if reference else None
    for repeat in range(repeats):
        for index, source in enumerate(paths):
            if cancelled():
                return None
            runner.model_seconds = 0.0
            started = perf_counter()
            prediction = runner.predict(source)
            total_ms.append((perf_counter() - started) * 1000)
            model_ms.append(runner.model_seconds * 1000)
            if repeat == 0:
                if reference:
                    if cfg["task"] == "segment":
                        mask, verdict = prediction
                        file = cache / f"fp32-{threads}-{index}.npy"
                        np.save(file, mask, allow_pickle=False)
                        outputs.append((file, verdict))
                    else:
                        outputs.append(prediction)
                else:
                    expected = baseline[index]
                    if cfg["task"] == "segment":
                        file, verdict = expected
                        expected = (np.load(file, mmap_mode="r", allow_pickle=False), verdict)
                    comparisons.append(_comparison(cfg, expected, prediction))
    return {
        "p50_ms": float(np.median(total_ms)),
        "p95_ms": float(np.percentile(total_ms, 95)),
        "model_p50_ms": float(np.median(model_ms)),
        "comparison": _summarize(cfg, comparisons),
        "samples": len(total_ms),
        "detail": f"CPUExecutionProvider · 그래프 최적화 {deployment_session_settings(cfg)['graph_optimization_level']}",
    }, outputs


def benchmark_precisions(fp32_path, image_dir, *, thread_counts=(1, 2, 4, 8), fp16_path=None, int8_path=None,
                         calibration_dir=None, run_fp16=True, run_int8=True, image_count=5, repeats=2, warmup=2,
                         log=print, cancelled=lambda: False):
    """Benchmark one image at a time with a separate CPU session per thread setting."""
    from quantize_onnx import list_images, quantize_segment

    counts = _thread_counts(thread_counts)
    if (type(image_count) is not int or image_count < 1 or type(warmup) is not int or warmup < 0
            or type(repeats) is not int or repeats < 1):
        raise ValueError("비교 영상과 반복 횟수는 1 이상, 워밍업 횟수는 0 이상이어야 합니다.")
    paths = [path for path in list_images(image_dir, image_count) if path.is_file()]
    if not paths:
        raise ValueError("선택한 폴더에 비교할 영상이 없습니다.")
    reference = _manifest(fp32_path, "float32")
    rows, baselines = [], {}
    variants = [("FP32", "float32", fp32_path)]
    if run_fp16:
        variants.append(("FP16", "float16", fp16_path))
    if run_int8:
        variants.append(("INT8", "int8", int8_path))

    with TemporaryDirectory(prefix="dvs-precision-benchmark-") as temporary:
        for label, precision, path in variants:
            if cancelled():
                break
            try:
                if precision == "float16" and not path:
                    from benchmark_fp16 import convert_benchmark_fp16
                    log("FP16 비교용 그래프 변환 중")
                    path = convert_benchmark_fp16(fp32_path, temporary)
                elif precision == "int8" and not path:
                    if not calibration_dir:
                        raise ValueError("INT8 JSON 또는 양자화 보정 영상 폴더를 선택하세요.")
                    log("INT8 비교용 그래프 보정 및 검증 중")
                    path = quantize_segment(fp32_path, calibration_dir, output_json=Path(temporary) / "int8.json",
                                            log=log, benchmark_only=True)
                cfg = reference if precision == "float32" else _manifest(path, precision, reference)
            except Exception as error:
                log(f"{label} 준비 실패: {error}")
                rows.append({"precision": label, "threads": None, "status": "준비 실패", "detail": str(error)})
                continue
            for threads in counts:
                if cancelled():
                    return rows
                log(f"{label} · CPU 스레드 {threads or '자동'} 측정 중")
                row = {"precision": label, "threads": threads}
                try:
                    baseline = baselines.get(threads)
                    if precision != "float32" and baseline is None:
                        raise ValueError("이 스레드 수의 FP32 기준 측정이 실패했습니다.")
                    measured = _measure(path, cfg, threads, paths, warmup, repeats, baseline, cancelled, Path(temporary))
                    if measured is None:
                        return rows
                    result, outputs = measured
                    if precision == "float32":
                        baselines[threads] = outputs
                    if cfg.get("export", {}).get("quantization", {}).get("comparison", {}).get("deployment_check_error"):
                        result["detail"] += " · INT8 배포 검증 미통과 (임시 비교)"
                    row.update(result, status="완료")
                except Exception as error:
                    log(f"{label} 측정 실패: {error}")
                    row.update(status="실행 실패", detail=str(error))
                rows.append(row)
    return rows
