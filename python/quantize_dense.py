"""Static QDQ INT8 for fixed-size segmentation and Dinomaly graphs."""

import copy
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from dense_deployment import PreparedWindows, predict_dense, region_pixels


def quantize_dense(
    config_path,
    calibration_dir,
    *,
    count=16,
    reduce_range=False,
    output_json=None,
    log=print,
):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from quantize_onnx import list_images

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["task"] not in {"segment", "anomaly"} or (
        config["task"] == "anomaly" and config["backend"] != "dinomaly"
    ):
        raise ValueError("고정 입력 INT8은 분할·Dinomaly 모델을 지원합니다")
    if config.get("encryption"):
        raise ValueError("INT8 보정을 먼저 하고 암호화하세요")
    source = config_path.parent / config["model_path"]
    output = (
        Path(output_json)
        if output_json
        else config_path.with_name(config_path.stem + "_int8.json")
    )
    from export_onnx import _paths_alias

    if any(
        _paths_alias(target, original)
        for target in (
            output,
            output.with_suffix(".onnx"),
            output.with_suffix(".report.json"),
        )
        for original in (config_path, source)
    ):
        raise ValueError("INT8 출력으로 FP32 원본을 덮어쓸 수 없습니다")
    paths = list_images(calibration_dir, count)
    if not paths:
        raise ValueError("보정 이미지가 없습니다")
    output.parent.mkdir(parents=True, exist_ok=True)
    graph = onnx.load(str(source))
    input_name = graph.graph.input[0].name
    if input_name != config["input_name"]:
        raise ValueError("ONNX/JSON 입력 이름 불일치")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = []
            for path in paths:
                image = cv2.imdecode(
                    np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
                pixels = region_pixels(config, image)
                self.items.extend(
                    (path, index)
                    for index in range(len(PreparedWindows(config, pixels)))
                )
            self.start, self.end = 0, len(self.items)
            self.cached_path, self.cached_pixels = None, None
            self.rewind()

        def __len__(self):
            return len(self.items)

        def set_range(self, start_index, end_index):
            self.start, self.end = start_index, end_index
            self.rewind()

        def rewind(self):
            self.cursor = self.start

        def get_next(self):
            if self.cursor >= self.end:
                return None
            path, index = self.items[self.cursor]
            self.cursor += 1
            if path != self.cached_path:
                self.cached_pixels = PreparedWindows(
                    config,
                    region_pixels(
                        config,
                        cv2.imdecode(
                            np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                        ),
                    ),
                )
                self.cached_path = path
            tensor = self.cached_pixels[index][2]
            return {input_name: tensor}

    reader = Reader()
    with tempfile.TemporaryDirectory(
        prefix=".dense-int8-", dir=output.parent
    ) as directory:
        staged = Path(directory) / output.with_suffix(".onnx").name
        # Fixed shapes preserve query masks, positions and attention domains.
        quantize_static(
            str(source),
            str(staged),
            reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=reduce_range,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            extra_options={"CalibStridedMinMax": 1},
        )
        quantized = onnx.load(str(staged))
        onnx.checker.check_model(quantized)
        if not any(n.op_type == "QuantizeLinear" for n in quantized.graph.node):
            raise ValueError("INT8로 변환된 연산이 없습니다")
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 2
        fp = ort.InferenceSession(
            str(source), settings, providers=["CPUExecutionProvider"]
        )
        quant = ort.InferenceSession(
            str(staged), settings, providers=["CPUExecutionProvider"]
        )
        report = compare_sessions(config, fp, quant, paths)
        result = copy.deepcopy(config)
        result["model_path"] = staged.name
        result.setdefault("export", {}).update(
            precision="int8",
            quantization={
                "format": "QDQ",
                "weights": "int8 per-channel",
                "activations": "uint8",
                "calibration": "minmax (fixed training tiles)",
                "calibration_images": len(paths),
                "calibration_tiles": len(reader),
                "reduce_range": bool(reduce_range),
                "source_model": source.name,
                "experimental": True,
                "comparison": report,
            },
        )
        # Verification applies to executable shape/finite outputs. Accuracy is
        # reported separately, never inherited as FP32 numerical equivalence.
        result["export"]["fp32_verification"] = result.get("verification")
        result["verification"] = "int8_compared"
        result["export"]["verification_policy"] = (
            "int8_finite_and_calibration_comparison_v1"
        )
        staged_json = Path(directory) / output.name
        staged_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(staged, output.with_suffix(".onnx"))
        os.replace(staged_json, output)
    output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(
        f"INT8 완료: {output.name} · 실제 보정 타일 {len(reader)}개 · 출력 차이: {report}"
    )
    return output


def compare_sessions(config, fp32, int8, paths):
    rows = []
    intersections = np.zeros(config["num_classes"], np.int64)
    unions = intersections.copy()
    for path in paths:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        values = [
            predict_dense(
                config,
                image,
                lambda t, s=session: s.run(None, {config["input_name"]: t}),
            )
            for session in (fp32, int8)
        ]
        row = {
            "image": Path(path).name,
            "max_map_error": float(np.max(np.abs(values[0] - values[1]))),
        }
        if config["task"] == "segment":
            a, b = [v.argmax(0) for v in values]
            row["pixel_agreement"] = float((a == b).mean())
            for c in range(config["num_classes"]):
                intersections[c] += ((a == c) & (b == c)).sum()
                unions[c] += ((a == c) | (b == c)).sum()
        else:
            scores = [float(v.max()) for v in values]
            threshold = config["postprocessing"]["threshold"]
            row.update(
                fp32_score=scores[0],
                int8_score=scores[1],
                decision_equal=(scores[0] >= threshold) == (scores[1] >= threshold),
            )
        rows.append(row)
    if not rows:
        raise ValueError("비교 이미지가 없습니다")
    return {
        "scope": "calibration inputs; not independent accuracy evaluation",
        "images": len(rows),
        "class_iou_vs_fp32": [
            float(i / u) if u else None for i, u in zip(intersections, unions)
        ],
        "per_image": rows,
    }
