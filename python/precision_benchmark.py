"""Compare ONNX graphs across CPU runtimes, precisions, and thread counts."""

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


_BACKENDS = {"onnxruntime": "ONNX Runtime", "openvino": "OpenVINO"}


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


def _read_image(path):
    flags = cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None or image.size == 0:
        raise ValueError(f"이미지 읽기 실패: {path.name}")
    return image


def _rgb_region(image, preprocessing):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if image.shape[2] == 3 else cv2.COLOR_BGRA2RGBA)
    x, y, right, bottom = center_crop_box((image.shape[1], image.shape[0]), preprocessing.get("center_crop"))
    return image[y:bottom, x:right]


def _tensor(cfg, image):
    pixels = resize(image, (cfg["input_height"], cfg["input_width"]))
    scale = 65535 if pixels.dtype == np.uint16 else 255
    pixels = pixels.reshape(*pixels.shape[:2], cfg["input_channels"]).astype(np.float32) / scale
    pixels = (pixels - np.asarray(cfg["normalize_mean"], np.float32)) / np.asarray(cfg["normalize_std"], np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def _classification_tensors(cfg, image):
    from native_patches import scaled_image, tiles, validate_patches

    pixels = convert_channels(_rgb_region(image, cfg["preprocessing"]), cfg["input_channels"])
    patches = validate_patches(cfg.get("patches"), "classify")
    if patches:
        for tile, _ in tiles(scaled_image(pixels, patches), patches):
            yield _tensor(cfg, tile)
    else:
        yield _tensor(cfg, pixels)


def _efficientnet_int8(fp32_path, cfg, calibration_dir, workspace, cancelled):
    """Build one temporary QDQ graph used unchanged by both CPU runtimes."""
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from quantize_onnx import list_images

    paths = [path for path in list_images(calibration_dir, 16) if path.is_file()]
    if not paths:
        raise ValueError("EfficientNet INT8 보정 이미지가 없습니다.")
    source = Path(fp32_path).parent / cfg["model_path"]
    target = workspace / "benchmark-efficientnet-int8.onnx"
    model = onnx.load(source)
    if (any(node.op_type in {"QuantizeLinear", "DequantizeLinear"} for node in model.graph.node)
            or any(initializer.data_type in {onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE}
                   for initializer in model.graph.initializer)):
        raise ValueError("EfficientNet INT8 변환에는 FP32 원본 그래프가 필요합니다.")
    if len(model.graph.input) != 1 or model.graph.input[0].name != cfg["input_name"]:
        raise ValueError("EfficientNet ONNX와 JSON의 입력 이름이 다릅니다.")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rewind()

        def _inputs(self):
            for path in paths:
                if cancelled():
                    raise InterruptedError("INT8 보정을 취소했습니다.")
                for tensor in _classification_tensors(cfg, _read_image(path)):
                    yield {cfg["input_name"]: tensor}

        def rewind(self):
            self.inputs = self._inputs()

        def get_next(self):
            return next(self.inputs, None)

    del model
    quantize_static(str(source), str(target), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=["Conv", "Gemm", "MatMul"])
    quantized = onnx.load(target)
    onnx.checker.check_model(quantized)
    if not any(node.op_type == "QuantizeLinear" for node in quantized.graph.node):
        raise ValueError("EfficientNet에서 INT8로 변환된 연산이 없습니다.")
    result = {**cfg, "model_path": target.name, "verification": "benchmark_only",
              "export": {**cfg.get("export", {}), "precision": "int8"}}
    manifest = target.with_suffix(".json")
    manifest.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return manifest


class _OpenVINOSession:
    def __init__(self, model_path, input_name, output_names, shape, threads):
        try:
            import openvino as ov
        except ImportError as error:
            raise RuntimeError("OpenVINO를 불러올 수 없습니다. 앱 빌드 또는 Python 환경의 OpenVINO 설치를 확인하세요.") from error
        from openvino_session import compile_cpu_model

        core = ov.Core()
        if "CPU" not in core.available_devices:
            raise RuntimeError("OpenVINO CPU 실행기를 찾을 수 없습니다.")
        model = core.read_model(str(model_path))
        if (len(model.inputs) != 1 or input_name not in model.input().get_names()
                or len(model.outputs) != len(output_names)):
            raise ValueError("OpenVINO 모델의 입력·출력 텐서 이름 또는 개수가 설정 JSON과 다릅니다.")
        inp = model.input()
        input_shape = inp.get_partial_shape()
        if (inp.get_element_type() != ov.Type.f32 or input_shape.rank.is_dynamic
                or input_shape.rank.get_length() != 4 or not input_shape.compatible(ov.PartialShape(shape))):
            raise ValueError("OpenVINO 모델의 NCHW 입력 크기 또는 FP32 타입이 설정 JSON과 다릅니다.")
        # Read the original names before compilation, then fetch by output index.
        indices = {name: [index for index, port in enumerate(model.outputs) if name in port.get_names()]
                   for name in output_names}
        if (any(len(candidates) != 1 for candidates in indices.values())
                or len({candidates[0] for candidates in indices.values()}) != len(output_names)
                or any(port.get_element_type() != ov.Type.f32 for port in model.outputs)):
            raise ValueError("OpenVINO 모델의 출력 이름 또는 FP32 타입이 설정 JSON과 다릅니다.")
        if input_shape.is_dynamic:
            model.reshape({inp: shape})
        self.compiled_model, settings = compile_cpu_model(core, model, threads)
        self.request = self.compiled_model.create_infer_request()
        self.input_name = input_name
        self.indices = {name: candidates[0] for name, candidates in indices.items()}
        self.detail = (f"OpenVINO {ov.__version__.split('-')[0]} · {settings['device']} · "
                       f"실제 스레드 {settings['INFERENCE_NUM_THREADS']} · 스트림 {settings['NUM_STREAMS']} · FP32 힌트")

    def run(self, names, feed):
        # The inference request reuses output buffers between calls and tiles.
        self.request.infer({0: feed[self.input_name]}, share_inputs=True, share_outputs=True)
        return [self.request.get_output_tensor(self.indices[name]).data.copy() for name in names]


class _Runner:
    """Time CPU runtime calls inside one image, including every tile."""

    def __init__(self, config_path, cfg, threads, backend):
        self.cfg = cfg
        self.names = cfg.get("output_names") or [cfg["output_name"]]
        model = Path(config_path).parent / cfg["model_path"]
        shape = [1, cfg["input_channels"], cfg["input_height"], cfg["input_width"]]
        self.model_seconds = 0.0
        if backend == "openvino":
            self.session = _OpenVINOSession(model, cfg["input_name"], self.names, shape, threads)
            self.detail = self.session.detail
            return

        import onnxruntime as ort

        settings = deployment_session_settings(cfg)
        settings["num_threads"] = threads
        self.session = ort.InferenceSession(str(model), cpu_session_options(**settings),
                                            providers=["CPUExecutionProvider"])
        self.detail = f"ONNX Runtime {ort.__version__} · CPUExecutionProvider · 그래프 최적화 {settings['graph_optimization_level']}"
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != cfg["input_name"] or inputs[0].type != "tensor(float)"
                or len(inputs[0].shape) != 4
                or any(isinstance(v, int) and v != size for v, size in zip(inputs[0].shape, shape))
                or [o.name for o in outputs] != self.names or any(o.type != "tensor(float)" for o in outputs)):
            raise ValueError("ONNX와 JSON의 텐서 이름, 입력 크기 또는 FP32 입출력 계약이 다릅니다.")

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

        image = _read_image(path)
        prep = cfg["preprocessing"]
        if task == "classify":
            logits = None
            for tensor in _classification_tensors(cfg, image):
                part = self._classification_logits(tensor)
                logits = part if logits is None else np.maximum(logits, part)
            probabilities = np.exp(logits.astype(np.float64) - np.max(logits))
            return int(np.argmax(logits)), probabilities / probabilities.sum()
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

        image = _rgb_region(image, prep)
        if image.dtype == np.uint16 and cfg["backend"] == "patchcore" and cfg["input_channels"] == 3:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = convert_channels(image, cfg["input_channels"])
        tensor = _tensor(cfg, image)
        outputs = self._infer(tensor)
        if cfg["backend"] in {"patchcore", "dinomaly"}:
            if outputs[0].shape != (1,) or len(outputs) != 2 or outputs[1].ndim != 4:
                raise ValueError("이상 탐지 ONNX 출력 크기가 설정과 다릅니다.")
            return float(outputs[0][0])
        if cfg["backend"] == "custom" and outputs[0].shape == tensor.shape:
            return float(np.abs(tensor - outputs[0]).mean())
        raise ValueError("지원하지 않는 이상 탐지 출력 형식입니다.")

    def _classification_logits(self, tensor):
        value = self._infer(tensor)[0]
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


def _measure(path, cfg, threads, backend, paths, warmup, repeats, *, reference, baseline, cancelled, cache):
    runner = _Runner(path, cfg, threads, backend)
    total_ms, model_ms, comparisons = [], [], []
    for index in range(warmup):
        if cancelled():
            return None
        runner.predict(paths[index % len(paths)])
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
                elif baseline is not None:
                    expected = baseline[index]
                    if cfg["task"] == "segment":
                        file, verdict = expected
                        expected = (np.load(file, mmap_mode="r", allow_pickle=False), verdict)
                    comparisons.append(_comparison(cfg, expected, prediction))
    return {
        "p50_ms": float(np.median(total_ms)),
        "p95_ms": float(np.percentile(total_ms, 95)),
        "model_p50_ms": float(np.median(model_ms)),
        "model_p95_ms": float(np.percentile(model_ms, 95)),
        "comparison": (_summarize(cfg, comparisons) if reference or baseline is not None
                       else "같은 스레드의 기준 FP32 결과 없음"),
        "samples": len(total_ms),
        "detail": runner.detail,
    }, outputs


def benchmark_precisions(fp32_path, image_dir, *, thread_counts=(1, 2, 4, 8), fp16_path=None, int8_path=None,
                         calibration_dir=None, run_fp16=True, run_int8=True, image_count=5, repeats=2, warmup=2,
                         backends=("onnxruntime",), log=print, cancelled=lambda: False):
    """Benchmark one image at a time with a separate CPU session per row."""
    from quantize_onnx import list_images, quantize_segment

    counts = _thread_counts(thread_counts)
    if not isinstance(backends, (tuple, list)) or not backends or any(name not in _BACKENDS for name in backends):
        raise ValueError("ONNX Runtime 또는 OpenVINO CPU 실행기를 선택하세요.")
    # ORT FP32 is the common reference when both runtimes are selected.
    backends = tuple(name for name in _BACKENDS if name in backends)
    reference_backend = backends[0]
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
                    if reference["task"] == "classify" and reference["backend"] == "efficientnet":
                        path = _efficientnet_int8(fp32_path, reference, calibration_dir, Path(temporary), cancelled)
                    else:
                        path = quantize_segment(fp32_path, calibration_dir, output_json=Path(temporary) / "int8.json",
                                                log=log, benchmark_only=True)
                cfg = reference if precision == "float32" else _manifest(path, precision, reference)
            except Exception as error:
                log(f"{label} 준비 실패: {error}")
                rows.extend({"backend": backend, "precision": label, "threads": None,
                             "status": "준비 실패", "detail": str(error)} for backend in backends)
                continue
            for backend in backends:
                for threads in counts:
                    if cancelled():
                        return rows
                    title = f"{_BACKENDS[backend]} · {label} · CPU 스레드 {threads or '자동'}"
                    log(f"{title} 측정 중")
                    row = {"backend": backend, "precision": label, "threads": threads}
                    try:
                        is_reference = precision == "float32" and backend == reference_backend
                        measured = _measure(path, cfg, threads, backend, paths, warmup, repeats,
                                            reference=is_reference, baseline=baselines.get(threads),
                                            cancelled=cancelled, cache=Path(temporary))
                        if measured is None:
                            return rows
                        result, outputs = measured
                        if is_reference:
                            baselines[threads] = outputs
                        if cfg.get("export", {}).get("quantization", {}).get("comparison", {}).get("deployment_check_error"):
                            result["detail"] += " · INT8 배포 검증 미통과 (임시 비교)"
                        row.update(result, status="완료")
                    except Exception as error:
                        log(f"{title} 측정 실패: {error}")
                        row.update(status="실행 실패", detail=str(error))
                    rows.append(row)
    return rows
