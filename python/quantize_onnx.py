"""분할 ONNX(FP32) → INT8 ONNX 변환 + 판정 비교 (데스크톱·C++ 테스트용, 실험 기능).

    내보낸 모델.json + 모델.onnx (FP32)
            │
            ├─ ① 보정: 실제 영상 N장을 C++ Segment와 같은 전처리로 넣어 각 층의 값 범위(min/max)를 잰다
            ├─ ② 양자화: 가중치 int8(채널별) · 활성값 uint8, QDQ 형식 → 모델_int8.onnx
            │      (QDQ = 그래프에 Quantize/Dequantize 노드를 끼운 표준 형식: ONNX Runtime CPU가
            │       Conv를 정수 연산(AVX2 / AVX-VNNI)으로 합쳐 실행한다. C++ SDK 코드 수정 없이 같은 Segment로 돈다)
            └─ ③ 비교: 같은 영상에서 FP32 vs INT8 클래스 맵 일치율 · 클래스별 IoU → 리포트
                   (결함 목록 비교·속도는 C++ 테스트 솔루션 cpp/int8_test 가 한다)

    모델_int8.json = 원래 JSON 복사 + model_path만 바꾸고 export.quantization 기록
    → C++: engine.InitializeFromJson("모델_int8.json", "", 8스레드) 그대로.

사용:
    python quantize_onnx.py 모델.json 보정영상폴더 [--count 16]
                            [--reduce-range] [--compare 비교영상폴더]

주의: INT8은 값 범위를 256단계로 자르는 근사다. 얇은 크랙처럼 확률이 경계에 걸린 결함은
      판정이 바뀔 수 있으므로, 반드시 비교 리포트와 C++ 결함 목록 비교를 보고 쓴다.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
INTERPOLATIONS = {"INTER_AREA": cv2.INTER_AREA, "INTER_LINEAR": cv2.INTER_LINEAR,
                  "INTER_LINEAR_EXACT": cv2.INTER_LINEAR_EXACT, "INTER_NEAREST": cv2.INTER_NEAREST}


# ─────────────────────────────────────────────────────────────
#  전처리: C++ VisionInference::PreprocessWholeRegion 과 같은 계산
# ─────────────────────────────────────────────────────────────
def _whole(config):
    whole = (config.get("preprocessing") or {}).get("whole_image")
    if config.get("task") != "segment" or not whole:
        raise ValueError("판정 영역을 한 번에 추론하는 분할 모델(whole_image 계약)만 INT8 변환을 지원합니다.")
    return whole


def judgement_region(config, image):
    """원본 → C++가 모델에 넣는 판정 영역 (중앙 크롭이 있으면 그 크기, 없으면 원본 = 판정 영역)."""
    width, height = _whole(config)["region"]
    crop = (config.get("preprocessing") or {}).get("center_crop")
    rows, cols = image.shape[:2]
    if crop or (cols, rows) != (width, height):
        # 크롭 모델(또는 크기가 다른 보정 영상): 가운데에서 판정 영역 크기만큼 잘라 쓴다.
        if cols < width or rows < height:
            return None
        x, y = (cols - width) // 2, (rows - height) // 2
        return image[y:y + height, x:x + width]
    return image


def preprocess(config, image):
    """BGR/GRAY 8비트 판정 영역 → (1, C, H, W) float32. 정규화 뒤 0으로 오른쪽·아래를 채운다."""
    whole = _whole(config)
    channels = int(config["input_channels"])
    if channels == 1:
        pixels = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3
                                                            else cv2.COLOR_BGRA2GRAY)
    else:
        pixels = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if image.ndim == 2 else
                              cv2.COLOR_BGR2RGB if image.shape[2] == 3 else cv2.COLOR_BGRA2RGB)
    scaled_w, scaled_h = whole["scaled"]
    if (pixels.shape[1], pixels.shape[0]) != (scaled_w, scaled_h):
        pixels = cv2.resize(pixels, (scaled_w, scaled_h),
                            interpolation=INTERPOLATIONS.get(whole.get("interpolation"), cv2.INTER_AREA))
    pixels = pixels.reshape(scaled_h, scaled_w, channels).astype(np.float32) / 255.0
    mean = np.asarray(config["normalize_mean"], np.float32)
    std = np.asarray(config["normalize_std"], np.float32)
    pixels = (pixels - mean) / std
    tensor = np.zeros((1, channels, int(config["input_height"]), int(config["input_width"])), np.float32)
    tensor[0, :, :scaled_h, :scaled_w] = pixels.transpose(2, 0, 1)
    return tensor


def list_images(folder, count=None):
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if count:
        # 폴더 전체에서 고르게 뽑는다 (앞쪽 한 종류만 보정되지 않게).
        step = max(1, len(paths) // count)
        paths = paths[::step][:count]
    return paths


def _tensors(config, paths):
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        region = None if image is None else judgement_region(config, image)
        if region is not None:
            yield path, preprocess(config, region)


def _tiles(tensor, tile):
    """(1, C, H, W) 입력 → tile×tile 조각들 (마지막 줄·칸은 끝에 맞춰 겹친다). 전체 영상을 빠짐없이 덮는다."""
    height, width = tensor.shape[2:]
    rows = sorted({min(y, height - tile) for y in range(0, height, tile)})
    cols = sorted({min(x, width - tile) for x in range(0, width, tile)})
    for y in rows:
        for x in cols:
            yield np.ascontiguousarray(tensor[:, :, y:y + tile, x:x + tile])


# ─────────────────────────────────────────────────────────────
#  그래프 손질: 업샘플을 '크기 고정' → '배율' 로 (입력 크기와 무관한 같은 그래프)
# ─────────────────────────────────────────────────────────────
def size_free_graph(model):
    """내보낸 U-Net은 업샘플 목표 크기(sizes)가 상수로 접혀 있어 입력 크기를 바꿀 수 없다.

        Resize(x, sizes=[1, C, 94, 220])   ──▶   Resize(x, scales=[1, 1, 2, 2])

    전체 크기 그래프에서 형 추론으로 (출력/입력) 배율을 구해 바꾼다. 배율이 정수 배면 half_pixel
    선형 보간 결과가 똑같다. 이렇게 하면 같은 그래프(같은 층 이름)를 작은 타일로 보정할 수 있다.
    정수 배가 아닌 Resize가 있으면 ValueError (이 도구가 지원하지 않는 구조).
    """
    shapes = _resize_shapes(model)
    from onnx import numpy_helper
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        source, target = shapes[node.input[0]], shapes[node.output[0]]
        ratios = [target[2] / source[2], target[3] / source[3]]
        if any(abs(r - round(r)) > 1e-9 for r in ratios):
            raise ValueError(f"{node.name}: 정수 배가 아닌 업샘플 {ratios} (지원하지 않는 구조)")
        name = f"{node.name}_scales"
        model.graph.initializer.append(numpy_helper.from_array(np.array([1, 1, *ratios], np.float32), name))
        node.input[:] = [node.input[0], "", name]          # roi 없음 · scales 사용 · sizes 제거
    _prune(model)
    del model.graph.value_info[:]
    return model


def _resize_shapes(model):
    """업샘플 크기는 Shape→Concat 계산이라 정적 형 추론으로 안 나온다 → 0 입력으로 한 번 실행해 잰다.
    (Resize의 입력·출력만 추가 출력으로 꺼낸다)"""
    import onnx
    import onnxruntime as ort
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    names = [n for node in probe.graph.node if node.op_type == "Resize" for n in (node.input[0], node.output[0])]
    existing = {o.name for o in probe.graph.output}
    probe.graph.output.extend(onnx.ValueInfoProto(name=n) for n in dict.fromkeys(names) if n not in existing)
    session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    feed = {}
    for value in session.get_inputs():
        feed[value.name] = np.zeros(value.shape, np.float32)
    outputs = session.run(None, feed)
    return {meta.name: list(array.shape) for meta, array in zip(session.get_outputs(), outputs)}


def _prune(model):
    """출력에 닿지 않는 노드(이제 쓰지 않는 Shape→Concat 크기 계산)와 초기값을 지운다."""
    needed = {o.name for o in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(o in needed for o in node.output):
            kept.append(node)
            needed.update(i for i in node.input if i)
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    initializers = [i for i in model.graph.initializer if i.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)


def _set_hw(model, height, width):
    """입력·출력의 세로·가로를 고정 값(숫자) 또는 이름(가변)으로."""
    for value in list(model.graph.input) + list(model.graph.output):
        dims = value.type.tensor_type.shape.dim
        for dim, size in ((dims[2], height), (dims[3], width)):
            if isinstance(size, str):
                dim.dim_param = size
            else:
                dim.dim_value = size
    del model.graph.value_info[:]
    return model


# ─────────────────────────────────────────────────────────────
#  ① + ② 보정 · 양자화
# ─────────────────────────────────────────────────────────────
def quantize_segment(config_path, calibration_dir, *, count=16, reduce_range=False,
                     tile=768, output_json=None, log=print):
    """FP32 분할 모델 → INT8(QDQ) 모델 + JSON. 만든 JSON 경로를 돌려준다.

        전체 입력(예: 1504×3520)으로 보정하면 층마다 중간값을 모두 잡아 15GB 넘게 든다
        → tile×tile 조각으로 보정한다. 경계/전역 문맥의 차이가 있으므로 실제 ROI에서
          FP32/INT8 비교가 필요하며 정확도가 보존된 것으로 자동 승인하지 않는다.

        FP32 ─size_free_graph─▶ 가변 입력 그래프 ─타일 보정·양자화─▶ INT8(가변) ─입력 고정─▶ 모델_int8.onnx
    """
    import tempfile

    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not (config.get("preprocessing") or {}).get("whole_image"):
        from quantize_dense import quantize_dense
        return quantize_dense(config_path, calibration_dir, count=count, reduce_range=reduce_range,
                              output_json=output_json, log=log)
    _whole(config)
    height, width = int(config["input_height"]), int(config["input_width"])
    tile = min(int(tile), height, width) // 32 * 32       # 백본 보폭 32의 배수
    source = config_path.parent / config["model_path"]
    output_json = Path(output_json) if output_json else config_path.with_name(config_path.stem + "_int8.json")
    output_onnx = output_json.with_suffix(".onnx")
    from export_onnx import _paths_alias
    if any(_paths_alias(target, original) for target in (output_json,output_onnx)
           for original in (config_path,source)):
        raise ValueError("INT8 출력으로 FP32 원본을 덮어쓸 수 없습니다")
    output_json.parent.mkdir(parents=True,exist_ok=True)
    paths = list_images(calibration_dir, count)
    if not paths:
        raise ValueError(f"보정 영상이 없습니다: {calibration_dir}")

    # 판정 영역에 맞는 영상만 (작은 영상은 C++도 거절한다)
    usable = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None and judgement_region(config, image) is not None:
            usable.append(path)
    if not usable:
        raise ValueError("판정 영역 크기에 맞는 보정 영상이 없습니다 (영상이 판정 영역보다 작습니다).")
    per_image = sum(1 for _ in _tiles(np.zeros((1, 1, height, width), np.float32), tile))

    class Reader(CalibrationDataReader):
        """(영상, 타일) 목록. ORT가 set_range로 한 타일씩 요청 → 범위 계산 → 누적 (CalibStridedMinMax).

            ORT 1.29의 CalibMaxIntermediateOutputs 는 범위를 계산하기 전에 중간값을 버려 쓰지 않는다.
        """

        def __init__(self):
            self.items = [(path, index) for path in usable for index in range(per_image)]
            self._range = (0, len(self.items))
            self._cursor = 0
            self._cached = (None, None)        # 마지막 영상의 타일들 (같은 영상 타일이 이어서 온다)
            self.tiles = 0

        def __len__(self):
            return len(self.items)

        def set_range(self, start_index, end_index):
            self._range, self._cursor = (start_index, end_index), start_index

        def get_next(self):
            if self._cursor >= self._range[1]:
                return None
            path, index = self.items[self._cursor]
            self._cursor += 1
            if self._cached[0] != path:
                region = judgement_region(config, cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
                self._cached = (path, list(_tiles(preprocess(config, region), tile)))
                log(f"  보정 {usable.index(path) + 1}/{len(usable)}: {path.name} ({per_image}타일)")
            self.tiles += 1
            return {config["input_name"]: self._cached[1][index]}

        @property
        def used(self):
            return len(usable)

    started = time.perf_counter()
    reader = Reader()
    with tempfile.TemporaryDirectory(prefix=".dvs-int8-", dir=output_json.parent) as work:
        work = Path(work)
        # ① 크기와 무관한 그래프 (FP32 결과는 원본과 같다; compare()가 확인한다)
        flexible = _set_hw(size_free_graph(onnx.load(str(source))), "height", "width")
        onnx.save(flexible, str(work / "flexible.onnx"))
        # 형 추론·그래프 정리를 먼저 하면 양자화 노드가 Conv+ReLU 묶음에 정확히 붙는다.
        quant_pre_process(str(work / "flexible.onnx"), str(work / "prepared.onnx"), skip_symbolic_shape=True)
        # ② 타일 보정 + 양자화
        quantize_static(
            str(work / "prepared.onnx"), str(work / "int8.onnx"), reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,      # 활성값 u8 · 가중치 s8 = x86 정수 커널(U8S8)이 쓰는 조합
            weight_type=QuantType.QInt8,
            per_channel=True,                      # 출력 채널마다 따로 범위 → 정확도 손실이 가장 작다
            reduce_range=reduce_range,             # VNNI 없는 AVX2 CPU에서 u8×s8 합산 포화 방지 (7비트 가중치)
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"CalibStridedMinMax": 1},   # 타일 1장마다 범위 계산 → 누적 (메모리 = 타일 1장 분)
        )
        # ③ 입력을 C++ 계약의 고정 크기로 되돌린다
        staged_model, staged_json = work / output_onnx.name, work / output_json.name
        fixed = _set_hw(onnx.load(str(work / "int8.onnx")), height, width)
        onnx.checker.check_model(fixed)
        onnx.save(fixed, str(staged_model))
        result = copy.deepcopy(config)
        result["model_path"] = output_onnx.name
        result.setdefault("export", {})["quantization"] = {
            "format": "QDQ", "weights": "int8 per-channel", "activations": "uint8",
            "calibration": "minmax (tiles)", "calibration_images": reader.used, "calibration_tiles": reader.tiles,
            "calibration_tile": tile, "reduce_range": bool(reduce_range),
            "source_model": config["model_path"], "experimental": True}
        result["export"]["precision"] = "int8"
        result["export"]["fp32_verification"] = result.get("verification")
        result["export"]["verification_policy"] = "int8_finite_and_calibration_comparison_v1"
        result["verification"] = "int8_compared"
        staged_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        comparison = compare(config_path, staged_json, calibration_dir, count=count, threads=2, log=log)
        comparison["scope"] = "calibration inputs; not independent accuracy evaluation"
        result["export"]["quantization"]["comparison"] = comparison
        staged_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        import os
        os.replace(staged_model, output_onnx)
        os.replace(staged_json, output_json)
    log(f"INT8 모델: {output_onnx.name} ({output_onnx.stat().st_size / 1e6:.1f} MB, 원본 "
        f"{source.stat().st_size / 1e6:.1f} MB) · 보정 {reader.used}장({reader.tiles}타일) "
        f"{time.perf_counter() - started:.0f}초")
    return output_json


# ─────────────────────────────────────────────────────────────
#  ③ 비교: 같은 입력에서 FP32 vs INT8 클래스 맵
# ─────────────────────────────────────────────────────────────
def _session(config_path, threads):
    import onnxruntime as ort
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(Path(config_path).parent / config["model_path"]), options,
                                   providers=["CPUExecutionProvider"])
    return config, session


def compare(fp32_json, int8_json, image_dir, *, count=None, threads=0, log=print):
    """클래스 맵 일치율과 결함 클래스별 IoU(FP32를 기준으로). 리포트 dict를 돌려준다."""
    config, fp32 = _session(fp32_json, threads)
    _, int8 = _session(int8_json, threads)
    if config["task"] == "detect":
        from detection_quantization import compare_detection_sessions
        return compare_detection_sessions(config, fp32, int8, list_images(image_dir, count))
    if not config.get("preprocessing", {}).get("whole_image"):
        from quantize_dense import compare_sessions
        return compare_sessions(config, fp32, int8, list_images(image_dir, count))
    scaled_w, scaled_h = config["preprocessing"]["whole_image"]["scaled"]
    classes = int(config["num_classes"])
    names = config.get("class_names") or [str(i) for i in range(classes)]
    inter, union = np.zeros(classes), np.zeros(classes)
    agree = total = 0
    times = {"fp32": [], "int8": []}
    rows = []
    for path, tensor in _tensors(config, list_images(image_dir, count)):
        maps = {}
        for key, session in (("fp32", fp32), ("int8", int8)):
            started = time.perf_counter()
            logits = session.run(None, {config["input_name"]: tensor})[0]
            if logits.shape != (1,classes,config["input_height"],config["input_width"]) or not np.isfinite(logits).all():
                raise ValueError("INT8 비교 출력 형상 또는 NaN/Inf 오류")
            times[key].append(time.perf_counter() - started)
            maps[key] = logits[0, :, :scaled_h, :scaled_w].argmax(0)
        same = int((maps["fp32"] == maps["int8"]).sum())
        agree, total = agree + same, total + maps["fp32"].size
        for c in range(1, classes):
            a, b = maps["fp32"] == c, maps["int8"] == c
            inter[c] += np.logical_and(a, b).sum()
            union[c] += np.logical_or(a, b).sum()
        rows.append({"image": path.name, "agreement": same / maps["fp32"].size})
        log(f"  {path.name}: 픽셀 일치 {same / maps['fp32'].size:.5f}")
    if not rows:
        raise ValueError(f"비교할 영상이 없습니다: {image_dir}")
    report = {
        "images": len(rows), "pixel_agreement": agree / total,
        "class_iou": {names[c]: (float(inter[c] / union[c]) if union[c] else None) for c in range(1, classes)},
        "median_seconds": {k: float(np.median(v)) for k, v in times.items()},
        "threads": threads or "auto", "per_image": rows}
    log(f"픽셀 일치 {report['pixel_agreement']:.5f} · 클래스 IoU(FP32 기준) "
        + ", ".join(f"{k} {v:.3f}" if v is not None else f"{k} -" for k, v in report["class_iou"].items())
        + f" · 중앙값 FP32 {report['median_seconds']['fp32']:.2f}s / INT8 {report['median_seconds']['int8']:.2f}s")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="분할 ONNX(FP32) → INT8 변환과 비교 (실험)")
    parser.add_argument("config", help="내보낸 모델 JSON (FP32)")
    parser.add_argument("calibration", help="보정 영상 폴더 (학습 영상 중 정상·불량 섞어 16장 정도)")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--reduce-range", action="store_true", help="VNNI 없는 구형 i7(AVX2)에서 정확도가 떨어질 때")
    parser.add_argument("--compare", help="비교 영상 폴더 (검증 영상). 리포트를 모델_int8.report.json 으로 저장")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)
    output = quantize_segment(args.config, args.calibration, count=args.count,
                              reduce_range=args.reduce_range)
    if args.compare:
        report = compare(args.config, output, args.compare, threads=args.threads)
        output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()
