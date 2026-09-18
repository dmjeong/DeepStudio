"""고정된 검증/테스트 manifest로 여러 체크포인트를 동일하게 평가한다."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from core.inference_loading import load_cpu_engine
    from core.comparison_evaluation import evaluate_predictions
    import numpy as np
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    if data.get("split") not in {"val", "test"}:
        raise ValueError("manifest에 split=val 또는 test 명시 필요")
    if data.get("task") not in {"classify", "detect"}:
        raise ValueError("공통 비교는 classify 또는 detect 지원")
    engine = load_cpu_engine(args.weights)
    engine.detection_confidence = 0.001
    engine.detection_iou = 0.5
    names = data["class_names"]
    records = []
    try:
        for item in data["images"]:
            path = args.manifest.parent / item["path"]
            result = engine.infer(str(path))
            if result.status == "error" or result.task != data["task"]:
                raise ValueError(f"평가 추론 실패 또는 태스크 불일치: {path}: {result.error}")
            predicted_names = result.details["class_names"]
            if len(predicted_names) != len(names) or set(predicted_names) != set(names):
                raise ValueError("체크포인트와 manifest 클래스 이름 불일치")
            remap = [names.index(name) for name in predicted_names]
            if result.task == "classify":
                prediction = remap[int(np.argmax(result.details["probabilities"]))]
            else:
                prediction = [{**box, "class_id": remap[box["class_id"]]} for box in result.details["detections"]]
            records.append({"image_id": item["path"], "prediction": prediction, "target": item["target"]})
        report = evaluate_predictions(data["task"], names, records)
        report.update(checkpoint=str(args.weights), split=data["split"], model_input_size=list(engine._input_size),
                      confidence_threshold=0.001, nms_iou=0.5, max_detections=300,
                      manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                      checkpoint_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False,
                                         default=lambda value: value.tolist() if isinstance(value, np.ndarray) else float(value)), encoding="utf-8")
        print(f"공통 평가 저장: {args.output}")
    finally:
        if engine._gradcam is not None:
            engine._gradcam.release()


if __name__ == "__main__":
    main()
