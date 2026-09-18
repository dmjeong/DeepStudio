"""사용자 PC에서 동일 조건으로 반복 가능한 CPU 추론 측정."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--images", type=Path)
    data.add_argument("--synthetic", action="store_true", help="고정 난수 이미지로 시간만 측정. 정확도 평가 아님")
    parser.add_argument("--runtime", choices=["auto", "pytorch", "onnx"], default="pytorch",
                        help="auto는 ONNX를 검증해 사용하고 준비 실패 시 PyTorch로 실행")
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--source-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("CPU 성능 비교는 런타임 초기화 제외를 위해 워밍업 1회 이상 필요")
    if min(args.source_size, args.threads) < 1 or (args.input_size is not None and args.input_size < 1):
        parser.error("입력 크기와 스레드 수는 양수 필요")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from core.benchmark import measure_inference
    from core.inference_loading import load_cpu_engine
    import numpy as np
    from PIL import Image
    source = str(args.weights)
    started = time.perf_counter()
    engine = load_cpu_engine(source, gradcam=args.gradcam, input_size=args.input_size,
                             runtime=args.runtime, threads=args.threads)
    engine.prepare()
    setup_sec = time.perf_counter() - started
    try:
        with tempfile.TemporaryDirectory() as temporary:
            if args.synthetic:
                path = Path(temporary) / "synthetic.png"
                pixels = np.random.default_rng(42).integers(0, 256, (args.source_size, args.source_size, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(path)
                paths = [path]
            else:
                paths = sorted(path for path in args.images.rglob("*")
                               if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
            report = measure_inference(engine, paths, warmup=args.warmup, repeats=args.runs,
                                       prepare_measurement=lambda: torch.set_num_threads(args.threads))
            manifest = [{"name": path.name if args.synthetic else str(path.relative_to(args.images)),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
            if args.synthetic:
                for row in report["records"]:
                    row["image"] = "synthetic.png"
        packages = {}
        for name in ["torch", "torchvision", "onnxruntime", "openvino"]:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        report.update(runtime=args.runtime, actual_runtime=engine.runtime, checkpoint=source, setup_sec=setup_sec,
                      runtime_warning=engine.runtime_warning,
                      checkpoint_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest() if Path(source).is_file() else None,
                      synthetic=args.synthetic, accuracy_evaluated=False, images=manifest,
                      requested_input_size=args.input_size, model_input_size=list(engine._input_size),
                      source_size=args.source_size if args.synthetic else None,
                      gradcam_requested=args.gradcam, packages=packages)
        report["environment"].update(logical_cpus=os.cpu_count(), requested_threads=args.threads,
                                     torch_threads=torch.get_num_threads(), runner=os.getenv("RUNNER_NAME", "local"))
        session = getattr(getattr(engine, "_onnx_runtime", None), "session", None)
        if callable(getattr(session, "get_session_options", None)):
            report["environment"]["onnx_intra_op_threads"] = session.get_session_options().intra_op_num_threads
            report["environment"]["onnx_inter_op_threads"] = session.get_session_options().inter_op_num_threads
        if platform.system() == "Linux" and Path("/proc/cpuinfo").exists():
            report["environment"]["processor"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                                        if line.startswith("model name")), platform.processor())
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        with (args.output / "benchmark.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report["records"][0]))
            writer.writeheader()
            writer.writerows(report["records"])
        print(json.dumps({"environment": report["environment"], "summary": report["summary"], "synthetic": args.synthetic,
                          "actual_runtime": engine.runtime, "runtime_warning": engine.runtime_warning}, ensure_ascii=False))
    finally:
        if engine._gradcam is not None:
            engine._gradcam.release()


if __name__ == "__main__":
    main()
