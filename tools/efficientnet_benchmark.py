"""동일 FP32 가중치로 EfficientNet CPU 지연시간과 8 ms 목표를 검증한다."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]


def measure(call, warmup, runs, target_ms=8.0):
    import numpy as np
    if warmup < 1 or runs < 1 or not math.isfinite(target_ms) or target_ms <= 0:
        raise ValueError("워밍업, 측정 횟수, 목표 시간은 양수 필요")
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"p50_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99)), "mean_ms": float(np.mean(samples)),
            "max_ms": max(samples), "within_target_fraction": float(np.mean(np.asarray(samples) <= target_ms)),
            "runs": runs, "samples_ms": samples}


def target_verdict(rows, scope, target_ms):
    candidates = [row for row in rows if row["scope"] == scope]
    best = min(candidates, key=lambda row: row["p95_ms"])
    return {"scope": scope, "metric": "p95_ms", "threshold_ms": target_ms,
            "met_on_this_machine": best["p95_ms"] <= target_ms,
            "runtime": best["runtime"], "threads": best["threads"], "p95_ms": best["p95_ms"],
            "all_measured_requests_within_target": best["max_ms"] <= target_ms,
            "acceptance_note": "Confirm on the target i7 with production weights/images and load; not a hard real-time guarantee."}


def environment():
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    elif Path("/proc/cpuinfo").is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    elif platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    packages = {}
    for name in ("torch", "torchvision", "numpy", "onnx", "onnxruntime", "openvino", "opencv-python", "opencv-python-headless"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"cpu": cpu, "os": platform.platform(), "machine": platform.machine(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version(), "packages": packages}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--architecture", choices=["efficientnet_b0", "efficientnet_b1"],
                        help="학습 가중치 없는 구조 속도 검사. 정확도 측정 아님")
    parser.add_argument("--input-size", type=int, help="구조 검사 기본 224; 체크포인트의 크기는 변경 불가")
    parser.add_argument("--in-channels", type=int, choices=[1, 3], help="구조 검사 기본 1")
    parser.add_argument("--num-classes", type=int, help="구조 검사 기본 2")
    parser.add_argument("--output", type=Path, default=Path("efficientnet-benchmark"))
    parser.add_argument("--validation", type=Path, help="체크포인트 클래스별 하위 폴더의 실제 검증 이미지")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--openvino", action="store_true", help="선택적 OpenVINO FP32 CPU도 비교")
    parser.add_argument("--target-ms", type=float, default=8.0)
    parser.add_argument("--target-scope", choices=["model", "pipeline"], default="pipeline")
    parser.add_argument("--require-target", action="store_true", help="p95 목표 미달이면 결과 저장 후 exit 2")
    args = parser.parse_args(argv)
    if min(args.runs, args.warmup, *args.threads) < 1 or not math.isfinite(args.target_ms) or args.target_ms <= 0:
        parser.error("측정 횟수, 워밍업, 스레드와 목표 시간은 양수 필요")
    if any(value is not None and value < 1 for value in (args.input_size, args.num_classes)):
        parser.error("입력 크기와 클래스 수는 양수 필요")
    if args.architecture and args.validation:
        parser.error("정확도 검증에는 학습 체크포인트가 필요합니다")
    args.threads = list(dict.fromkeys(args.threads))
    return args


def checkpoint_source(args):
    import torch
    from checkpoint import make_checkpoint_metadata
    from efficientnet import EfficientNet
    from export_onnx import resolve_checkpoint_spec
    if args.checkpoint:
        path = args.checkpoint
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("engine") != "efficientnet":
            raise ValueError("EfficientNet 체크포인트 필요")
        spec = resolve_checkpoint_spec(checkpoint)
        for option, actual in ((args.input_size, spec["input_height"]),
                               (args.input_size, spec["input_width"]),
                               (args.in_channels, spec["in_channels"]),
                               (args.num_classes, spec["num_classes"])):
            if option is not None and option != actual:
                raise ValueError("체크포인트 설정과 명령행 설정 불일치")
        origin = {"kind": "checkpoint", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        torch.manual_seed(42)
        model = EfficientNet(args.architecture, args.num_classes or 2, args.in_channels or 1)
        checkpoint = make_checkpoint_metadata("classify", model.num_classes,
            [str(i) for i in range(model.num_classes)], args.input_size or 224, model.in_channels)
        checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
        path = args.output / "synthetic-checkpoint.pt"
        torch.save(checkpoint, path)
        origin = {"kind": "random_weights", "seed": 42, "accuracy_evaluated": False}
    return path, checkpoint, origin


def accuracy_metrics(matrix):
    import numpy as np
    tp = matrix.diagonal()
    precision = tp / np.maximum(matrix.sum(0), 1)
    recall = tp / np.maximum(matrix.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"accuracy": float(tp.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
            "recall_per_class": recall.tolist(), "confusion_matrix": matrix.tolist(), "samples": int(matrix.sum())}


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    import cv2
    from efficientnet import prepare_for_inference
    from export_onnx import export_checkpoint, load_custom_model, resolve_checkpoint_spec, validate_outputs
    from onnx_classifier import ImageClassifier, OnnxClassifier
    from opencv_preprocess import read_image
    # Configure pools before export or inference. Only one backend session lives at a time.
    torch.set_num_threads(args.threads[0])
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    if args.openvino:
        import openvino  # Fail explicitly when requested but not installed.
        from openvino_classifier import OpenVINOClassifier
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path, checkpoint, origin = checkpoint_source(args)
    spec = resolve_checkpoint_spec(checkpoint)
    reference = load_custom_model(checkpoint, spec)
    models = {"pytorch": reference, "pytorch_fused": prepare_for_inference(reference),
              "pytorch_fused_channels_last": prepare_for_inference(reference, channels_last=True)}
    exported = export_checkpoint(checkpoint_path, args.output / "model.onnx", log=lambda line: print(line, file=sys.stderr))
    preprocessing = ImageClassifier(exported["config_path"])
    files = []
    if args.validation:
        files = sorted(p for p in args.validation.glob("*/*") if p.is_file() and
                       p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        if not files:
            raise ValueError("검증 이미지 없음")
        unknown = {p.parent.name for p in files} - set(spec["class_names"])
        if unknown:
            raise ValueError(f"모델 클래스에 없는 검증 폴더: {sorted(unknown)}")
    h, w = spec["input_height"], spec["input_width"]
    crop = spec["preprocessing"].get("center_crop") or {}
    image_shape = (max(h, crop.get("height", 0)), max(w, crop.get("width", 0)))
    if spec["in_channels"] == 3:
        image_shape += (3,)
    pixels = read_image(files[0], spec["in_channels"]) if files else np.random.default_rng(42).integers(
        0, 256, image_shape, dtype=np.uint8)
    array = preprocessing.preprocess(pixels)
    report = {"schema_version": 2, "environment": environment(), "weights": origin,
              "architecture": reference.architecture, "input_size": [h, w], "batch_size": 1,
              "num_classes": spec["num_classes"], "model_config": reference.checkpoint_config(),
              "synthetic_input": not bool(files), "source_image_shape": list(pixels.shape),
              "timing_image": str(files[0].relative_to(args.validation)) if files else None,
              "precision": "float32", "warmup": args.warmup, "measurements": [],
              "scope_definitions": {"model": "prebuilt tensor -> CPU logits, including runtime output checks",
                                    "pipeline": "uint8 memory image -> preprocessing -> model -> probabilities; no decode, IO, UI or Grad-CAM"},
              "thread_policy": "one backend session at a time; torch interop=1, OpenCV=1; continuous batch-one requests",
              "optimization": models["pytorch_fused"].inference_optimization, "accuracy": None,
              "openvino_requested": args.openvino}

    class TorchClassifier(ImageClassifier):
        def __init__(self, model, channels_last):
            super().__init__(exported["config_path"])
            self.model, self.channels_last = model, channels_last

        def logits(self, x):
            tensor = torch.from_numpy(x)
            if self.channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            result = self.model(tensor).numpy()
            if not np.isfinite(result).all():
                raise ValueError("PyTorch 비유한 출력")
            return result

    with torch.inference_mode():
        expected = reference(torch.from_numpy(array)).numpy()
        # Validation is performed outside timed regions; cache only labels/logits.
        validation = []
        baseline_matrix = np.zeros((spec["num_classes"], spec["num_classes"]), dtype=int)
        for path in files:
            x = preprocessing.preprocess(read_image(path, spec["in_channels"]))
            logits = reference(torch.from_numpy(x)).numpy()
            label = spec["class_names"].index(path.parent.name)
            baseline_matrix[label, int(logits.argmax(1)[0])] += 1
            validation.append((path, label, logits))
        if files:
            report["accuracy"] = {"pytorch": accuracy_metrics(baseline_matrix)}
        backends = list(models) + ["onnx"] + (["openvino"] if args.openvino else [])
        for name in backends:
            for threads in args.threads:
                torch.set_num_threads(threads)
                if name in models:
                    runtime = TorchClassifier(models[name], name.endswith("channels_last"))
                    settings = {"intra_op_threads": torch.get_num_threads(), "inter_op_threads": torch.get_num_interop_threads()}
                elif name == "onnx":
                    runtime = OnnxClassifier(exported["config_path"], num_threads=threads)
                    settings = {"intra_op_threads": runtime.session.get_session_options().intra_op_num_threads,
                                "provider": runtime.session.get_providers()}
                else:
                    runtime = OpenVINOClassifier(exported["config_path"], num_threads=threads)
                    settings = runtime.runtime_settings
                # Check the exact deployed path, including each chosen thread configuration.
                actual = runtime.logits(array)
                validate_outputs(expected, actual)
                parity = {"max_abs_error": float(np.abs(expected - actual).max()),
                          "top1_equal": bool(expected.argmax(1)[0] == actual.argmax(1)[0])}
                if not parity["top1_equal"]:
                    raise ValueError(f"{name}/{threads}: top1 판정 불일치")
                for scope, call in (("model", lambda: runtime.logits(array)),
                                    ("pipeline", lambda: runtime.predict_rgb(pixels))):
                    row = {"runtime": name, "threads": threads, "scope": scope,
                           "runtime_settings": settings, "parity": parity,
                           **measure(call, args.warmup, args.runs, args.target_ms)}
                    report["measurements"].append(row)
                    print(f'{name:28s} threads={threads} {scope:8s} p50={row["p50_ms"]:.3f} p95={row["p95_ms"]:.3f} ms', file=sys.stderr)
                if files:
                    matrix = np.zeros_like(baseline_matrix)
                    matches = 0
                    for path, label, original in validation:
                        x = preprocessing.preprocess(read_image(path, spec["in_channels"]))
                        result = runtime.logits(x)
                        validate_outputs(original, result)
                        predicted = int(result.argmax(1)[0])
                        matches += predicted == int(original.argmax(1)[0])
                        matrix[label, predicted] += 1
                    report["accuracy"][f"{name}/{threads}"] = {**accuracy_metrics(matrix), "reference_top1_agreement": matches / len(files)}
                    if matches != len(files):
                        raise ValueError(f"{name}/{threads}: 검증 이미지 top1 판정 불일치")
                del call, runtime
                gc.collect()
    report["target"] = target_verdict(report["measurements"], args.target_scope, args.target_ms)
    candidates = [row for row in report["measurements"] if row["runtime"] == "onnx" and row["scope"] == "pipeline"]
    report["recommended_onnx_threads"] = min(candidates, key=lambda row: row["p95_ms"])["threads"]
    report["thread_recommendation_scope"] = "this machine and workload only; confirm under production concurrency"
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"environment": report["environment"], "target": report["target"],
                      "report": str(args.output / "report.json")}, ensure_ascii=False, indent=2))
    return 2 if args.require_target and not report["target"]["met_on_this_machine"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
