"""설비 PC에서 분할 모델 후보의 CPU 추론 시간을 잰다 (속도만 — 가중치가 무작위라 정확도는 알 수 없다).

    "어떤 모델 · 배율 · 정밀도 · 런타임이면 목표 시간 안에 드는가"를 학습 전에 고른다.

        ① 준비 (torch 필요, 한 번)                      ② 측정 (numpy + onnxruntime, openvino는 있으면)
        모델 × 출력 방식 ─▶ FP32.onnx ─▶ INT8.onnx  ──▶  배율 · 슬롯 크기별로 같은 파일을 크기만 바꿔 실행
        (입력 크기와 무관한 그래프라 파일은 조합마다 하나)      └▶ 표 + JSON

    출력 방식
        full : 원본 해상도 로짓 (지금 내보내기와 같은 모양)
        half : 1/2 해상도 로짓 (head를 확대보다 먼저, 마지막 확대 없음 → argmax는 C++에서 1/2 해상도로)
    배율
        판정 영역을 줄여 넣는 비율 (패치 학습 배율과 같은 뜻). 줄인 뒤 32의 배수로 채운다 = C++ 계약과 같은 크기.
    슬롯
        --slot-size 를 주면 그 크기 조각 --slots 개를 스레드마다 하나씩 끝까지 계산한 시간도 잰다
        (고정 위치 슬롯만 잘라 넣는 방식. 칸 전체가 아니라 칩 부분만 자르면 픽셀이 줄어든 만큼 빨라진다).

사용:
    python tools/segment_cpu_benchmark.py --region 3100x1500 --threads 8 --output result.json
    python tools/segment_cpu_benchmark.py --region 3100x1500 --slot-size 192x160 --slots 96

    설비 PC에 torch가 없으면: 개발 PC에서 --export-only 로 만든 --work 폴더를 그대로 복사해 온다.
    OpenVINO 열은 `pip install openvino` 뒤에 나온다. --threads 는 물리 코어 수를 준다.

주의: 모델 시간만 잰다. 전처리(축소·정규화)와 후처리(argmax·결함 판정)는 포함하지 않는다.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

MODELS = ("unet_resnet18", "unet_mobilenetv2", "unet_lite")
HEADS = ("full", "half")
LAST_DECODER = {"unet_resnet18": "dec1", "unet_mobilenetv2": "dec1", "unet_lite": "d1"}  # 1/2 해상도 특징을 내는 블록
MULTIPLE = 32                 # 분할 백본의 최대 보폭 (export_onnx.WHOLE_IMAGE_MULTIPLE 과 같은 값)
EXPORT_SIZE = (64, 96)        # 내보낼 때 쓰는 작은 입력 (세로, 가로). 그래프는 크기와 무관하게 바뀐다
CALIBRATION_TILE = (256, 384)
INPUT_NAME = "input_image"


def parse_size(text):
    """'3100x1500' → (가로, 세로)."""
    try:
        width, height = (int(value) for value in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"크기는 가로x세로 로 씁니다 (예: 3100x1500): {text}") from None
    if min(width, height) < 1:
        raise argparse.ArgumentTypeError(f"크기는 양수여야 합니다: {text}")
    return width, height


def model_input(width, height, scale=1.0):
    """판정 영역 → 모델 입력 (세로, 가로): 배율로 줄이고 32의 배수로 올린다 (whole_image 계약과 같은 계산)."""
    if scale != 1:
        width, height = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    return -(-height // MULTIPLE) * MULTIPLE, -(-width // MULTIPLE) * MULTIPLE


def model_path(work, model, head, precision):
    return Path(work) / f"{model}_{head}_{precision}.onnx"


# ─────────────────────────────────────────────────────────────
#  ① 준비: 무작위 가중치 모델 → 크기와 무관한 FP32 / INT8 ONNX
# ─────────────────────────────────────────────────────────────
def build_export_model(model, head):
    """내보낼 torch 모델. half 는 마지막 디코더 블록의 1/2 해상도 특징에 head 를 바로 적용한다."""
    import torch
    from builtin_models import build_builtin_model

    torch.manual_seed(0)
    net = build_builtin_model(model, 3, 3).eval()
    # 갓 만든 BatchNorm은 층마다 값이 같아(평균 0 · 분산 1) 합성곱에 합치면 편향이 모두 0이 된다. 내보내기가
    # 같은 값을 하나로 묶어(Identity) 양자화가 그 편향을 건너뛰고, ONNX Runtime이 정수 합성곱으로 합치지 못해
    # INT8이 실제보다 느리게 나온다 → 학습된 모델처럼 층마다 다른 통계를 넣는다.
    for module in net.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_(0, 0.1)
            module.running_var.uniform_(0.5, 1.5)
            module.weight.data.uniform_(0.5, 1.5)
            module.bias.data.normal_(0, 0.1)
    if head == "full":
        return net

    class HalfHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net, self.feature = net, None
            getattr(net, LAST_DECODER[model]).register_forward_hook(self._keep)

        def _keep(self, module, inputs, output):
            self.feature = output

        def forward(self, x):
            self.net(x)                          # 훅이 1/2 해상도 특징을 잡는다 (그 뒤 확대는 그래프에서 빠진다)
            return self.net.head(self.feature)   # 1×1 합성곱과 선형 확대는 순서를 바꿔도 값이 같다

    return HalfHead().eval()


def export_fp32(model, head, path):
    """작은 고정 입력으로 내보낸 뒤 업샘플을 배율 방식으로 바꿔, 어떤 입력 크기에도 쓰는 그래프로 만든다."""
    import inspect

    import onnx
    import torch
    from quantize_onnx import _set_hw, size_free_graph

    exporter_options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    if "external_data" in inspect.signature(torch.onnx.export).parameters:
        exporter_options["external_data"] = False
    fixed = path.with_suffix(".fixed.onnx")
    with torch.no_grad():
        torch.onnx.export(build_export_model(model, head), torch.zeros(1, 3, *EXPORT_SIZE), str(fixed),
                          input_names=[INPUT_NAME], output_names=["logits"], opset_version=17, **exporter_options)
    onnx.save(_set_hw(size_free_graph(onnx.load(str(fixed))), "height", "width"), str(path))
    fixed.unlink()


def export_int8(source, path):
    """quantize_onnx 와 같은 설정(QDQ · 가중치 int8 채널별 · 활성값 uint8)으로 양자화한다.

    보정 입력은 무작위 타일이다 — 정수 연산의 속도는 값 범위와 무관하므로 시간 측정에는 충분하다.
    """
    import tempfile

    import numpy as np
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            rng = np.random.default_rng(0)
            self.items = iter([{INPUT_NAME: rng.standard_normal((1, 3, *CALIBRATION_TILE), dtype=np.float32)}
                               for _ in range(4)])

        def get_next(self):
            return next(self.items, None)

    with tempfile.TemporaryDirectory(prefix="dvs-bench-") as work:
        prepared = Path(work) / "prepared.onnx"
        quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
        quantize_static(str(prepared), str(path), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                        calibrate_method=CalibrationMethod.MinMax)


def prepare(work, models, heads, int8, log=print):
    """없는 ONNX 파일만 만든다. 이미 있으면 torch 없이도 측정할 수 있다."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for model in models:
        for head in heads:
            fp32 = model_path(work, model, head, "fp32")
            quantized = model_path(work, model, head, "int8")
            if fp32.exists() and (quantized.exists() or not int8):
                continue
            try:
                if not fp32.exists():
                    log(f"준비: {fp32.name}")
                    export_fp32(model, head, fp32)
                if int8 and not quantized.exists():
                    log(f"준비: {quantized.name}")
                    export_int8(fp32, quantized)
            except ImportError as error:
                raise SystemExit(f"{fp32.name} 을(를) 만들 수 없습니다 ({error}). torch · onnx · opencv 가 있는 PC에서 "
                                 f"--export-only 로 만든 뒤 '{work}' 폴더를 복사해 오세요.") from error


# ─────────────────────────────────────────────────────────────
#  ② 측정
# ─────────────────────────────────────────────────────────────
def _timed(run, runs):
    run()                                             # 예열: 메모리 할당 · 커널 선택을 측정에서 뺀다
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        run()
        times.append((time.perf_counter() - started) * 1e3)
    return times


def _inputs(shape, count):
    import numpy as np
    rng = np.random.default_rng(1)
    return [rng.standard_normal(shape, dtype=np.float32) for _ in range(count)]


def time_onnxruntime(path, shape, threads, runs, count=1):
    """count=1: 한 장을 스레드 전부로.  count>1: 조각 count개를 스레드마다 하나씩(조각 안은 1스레드)."""
    import onnxruntime as ort
    from onnx_session import cpu_session_options

    session = ort.InferenceSession(str(path), cpu_session_options(num_threads=threads if count == 1 else 1),
                                   providers=["CPUExecutionProvider"])
    feeds = [{INPUT_NAME: data} for data in _inputs(shape, count)]
    if count == 1:
        return _timed(lambda: session.run(None, feeds[0]), runs)
    with ThreadPoolExecutor(threads) as pool:        # Run은 GIL을 놓고, 한 세션을 여러 스레드가 함께 쓸 수 있다
        return _timed(lambda: list(pool.map(lambda feed: session.run(None, feed), feeds)), runs)


def time_openvino(path, shape, threads, runs, count=1):
    """같은 뜻의 측정. FP32 모델이 bf16(AMX)으로 바뀌지 않게 정밀도를 f32로 고정한다."""
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(path))
    model.reshape({model.inputs[0].get_any_name(): list(shape)})        # 고정 크기로 컴파일
    streams = 1 if count == 1 else threads
    compiled = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": threads, "NUM_STREAMS": streams,
                                                 "INFERENCE_PRECISION_HINT": "f32"})
    data = _inputs(shape, count)
    if count == 1:
        request = compiled.create_infer_request()
        return _timed(lambda: request.infer({0: data[0]}), runs)
    queue = ov.AsyncInferQueue(compiled, streams)

    def run_all():
        for item in data:
            queue.start_async({0: item})
        queue.wait_all()

    return _timed(run_all, runs)


TIMERS = {"onnxruntime": time_onnxruntime, "openvino": time_openvino}


def available_runtimes(requested, log=print):
    found = []
    for name in requested:
        try:
            __import__(name)
            found.append(name)
        except ImportError:
            log(f"{name} 이(가) 설치되어 있지 않아 건너뜁니다 (pip install {name}).")
    if not found:
        raise SystemExit("측정할 런타임이 없습니다. onnxruntime 또는 openvino 를 설치하세요.")
    return found


def measure(work, models, heads, cases, precisions, runtimes, threads, runs, log=print):
    """cases: [(이름, (세로, 가로), 조각 수)].  기록 목록을 돌려준다."""
    records = []
    for model in models:
        for head in heads:
            for label, (height, width), count in cases:
                for precision in precisions:
                    for runtime in runtimes:
                        times = TIMERS[runtime](model_path(work, model, head, precision),
                                                (1, 3, height, width), threads, runs, count)
                        record = {"model": model, "head": head, "case": label, "input": [width, height],
                                  "count": count, "precision": precision, "runtime": runtime,
                                  "median_ms": round(statistics.median(times), 2),
                                  "runs_ms": [round(value, 2) for value in times]}
                        records.append(record)
                        log(f"  {model} {head} {label} {precision} {runtime}: {record['median_ms']:.1f} ms")
    return records


def table(records, runtimes):
    """행 = 모델 · 출력 · 조건 · 정밀도,  열 = 런타임 (중앙값 ms)."""
    rows = {}
    for record in records:
        key = (record["model"], record["head"], record["case"],
               f"{record['input'][0]}x{record['input'][1]}" + (f" x{record['count']}" if record["count"] > 1 else ""),
               record["precision"])
        rows.setdefault(key, {})[record["runtime"]] = record["median_ms"]
    header = ["모델", "출력", "조건", "입력", "정밀도", *runtimes]
    lines = [header] + [[*key, *(f"{values[name]:.1f}" if name in values else "-" for name in runtimes)]
                        for key, values in rows.items()]
    widths = [max(_display_width(line[index]) for line in lines) for index in range(len(header))]
    return "\n".join("  ".join(cell + " " * (width - _display_width(cell)) for cell, width in zip(line, widths)).rstrip()
                     for line in lines)


def _display_width(text):
    """콘솔에서 차지하는 칸 수 (한글은 두 칸)."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def cpu_name():
    """CPU 이름 (세대에 따라 INT8 배속이 달라 결과와 함께 남긴다)."""
    try:
        if platform.system() == "Windows":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def environment(threads, max_isa):
    packages = {}
    for name in ("onnxruntime", "openvino", "numpy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"processor": cpu_name(), "logical_cpus": os.cpu_count(), "threads": threads,
            "platform": platform.platform(), "python": platform.python_version(), "packages": packages,
            "openvino_max_isa": max_isa or "native"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="분할 모델 후보의 CPU 추론 시간 측정 (속도만, 정확도 아님)")
    parser.add_argument("--region", type=parse_size, default=(3100, 1500), help="판정 영역 가로x세로 (기본 3100x1500)")
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5], help="입력 배율 (기본 1.0 0.5)")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=list(HEADS), help="full=원본 해상도 출력, half=1/2 해상도 출력")
    parser.add_argument("--runtimes", nargs="+", choices=list(TIMERS), default=list(TIMERS))
    parser.add_argument("--no-int8", action="store_true", help="INT8 을 만들지도 재지도 않는다")
    parser.add_argument("--slot-size", type=parse_size, help="슬롯 조각 가로x세로. 주면 조각 --slots 개의 시간도 잰다")
    parser.add_argument("--slots", type=int, default=96, help="슬롯 수 (기본 96 = 8x12)")
    parser.add_argument("--threads", type=int, default=4, help="스레드 수. 물리 코어 수를 준다 (기본 4)")
    parser.add_argument("--runs", type=int, default=5, help="반복 횟수. 중앙값을 쓴다 (기본 5)")
    parser.add_argument("--work", type=Path, default=Path("segment_cpu_benchmark"), help="ONNX 파일을 두는 폴더")
    parser.add_argument("--export-only", action="store_true", help="ONNX 파일만 만들고 끝낸다")
    parser.add_argument("--max-isa", help="OpenVINO 명령어 세트 상한 (예: AVX2, AVX2_VNNI). 다른 PC 수준을 흉내 낼 때만 쓴다")
    parser.add_argument("--output", type=Path, help="결과 JSON 경로")
    args = parser.parse_args(argv)
    if min(args.threads, args.runs, args.slots) < 1 or any(not 0 < scale <= 1 for scale in args.scales):
        parser.error("스레드 · 반복 · 슬롯 수는 양수, 배율은 0보다 크고 1 이하여야 합니다")
    if args.max_isa:
        os.environ["ONEDNN_MAX_CPU_ISA"] = args.max_isa      # openvino 를 불러오기 전에 정해야 적용된다

    prepare(args.work, args.models, args.heads, not args.no_int8)
    if args.export_only:
        print(f"ONNX 파일을 만들었습니다: {args.work}")
        return 0
    runtimes = available_runtimes(args.runtimes)
    cases = [(f"{scale:g}x", model_input(*args.region, scale), 1) for scale in args.scales]
    if args.slot_size:
        cases.append(("slots", model_input(*args.slot_size), args.slots))
    info = environment(args.threads, args.max_isa)
    print(f"CPU: {info['processor']} · 논리 코어 {info['logical_cpus']} · 스레드 {args.threads} · "
          f"판정 영역 {args.region[0]}x{args.region[1]}")
    records = measure(args.work, args.models, args.heads, cases, ["fp32"] + ([] if args.no_int8 else ["int8"]),
                      runtimes, args.threads, args.runs)
    print("\n모델 시간 중앙값 (ms) — 전처리 · 후처리 제외, 가중치 무작위(정확도 아님)")
    print(table(records, runtimes))
    if args.max_isa and "onnxruntime" in runtimes:
        print(f"※ --max-isa {args.max_isa} 는 openvino 열에만 적용됩니다. onnxruntime 열은 이 PC의 명령어 세트 그대로입니다.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"environment": info, "region": list(args.region), "accuracy_evaluated": False,
                                           "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"결과 저장: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
