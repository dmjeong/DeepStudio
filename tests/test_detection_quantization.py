"""Exercise object decisions during INT8 export and SDK loading."""
import json
import os
import subprocess

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime.quantization")


def test_detection_int8_compares_real_decisions_and_preserves_output_on_failure(tmp_path, monkeypatch):
    from export_onnx import create_inference_config, export_to_onnx
    from model_crypto import create_key, encrypt_config, read_key
    from native_detection import preprocessing_contract
    from native_patches import patch_contract
    from quantize_onnx import quantize_segment

    class DetectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.score = torch.nn.Conv2d(3, 2, 1)
            with torch.no_grad():
                self.score.weight.zero_()
                self.score.weight[0, 0] = 12
                self.score.bias[:] = torch.tensor([-8., -8.])
            self.register_buffer("box", torch.tensor([[[.5, .5, .2, .2]]]))

        def forward(self, x):
            scores = self.score(x).mean((2, 3))[:, None]
            return scores, self.box.expand(x.shape[0], -1, -1)

    source = tmp_path / "model.onnx"
    names = ["detection_scores", "detection_boxes"]
    export_to_onnx(DetectionModel().eval(), torch.zeros(1, 3, 32, 32), source,
                   task="detect", output_names=names)
    create_inference_config(tmp_path, "detect", 2, (32, 32), 3, source.name,
        ["crack", "foreign"], preprocessing=preprocessing_contract("dfine", 32),
        backend="libreyolo_dfine", output_names=names, detection_box_encoding="normalized_cxcywh")
    config = tmp_path / "inference_config.json"
    manifest = json.loads(config.read_text(encoding="utf-8"))
    manifest.update(schema_version=7, patches=patch_contract("detect", 32))
    manifest["postprocessing"].update(selection="query_class_topk", class_aware_nms=False)
    manifest["preprocessing"]["center_crop"] = {"width": 50, "height": 50}
    config.write_text(json.dumps(manifest), encoding="utf-8")
    images = tmp_path / "calibration"
    images.mkdir()
    for name, value in (("normal", 20), ("positive", 255)):
        rgb = np.zeros((56, 56, 3), dtype=np.uint8)
        rgb[..., 0] = value
        Image.fromarray(rgb).save(images / f"{name}.png")

    logs = []
    output = quantize_segment(config, images, log=logs.append)
    config_before = output.read_bytes()
    report = json.loads(config_before)["export"]["quantization"]["comparison"]
    assert report["fp32_positive_images"] == 1
    assert report["matched_boxes"] == report["fp32_boxes"] == report["int8_boxes"] == 4
    assert report["lost_boxes"] == report["gained_boxes"] == report["changed_image_decisions"] == 0
    assert report["mean_matched_iou"] == pytest.approx(1.)
    assert [v.name for v in onnx.load(str(output.with_suffix(".onnx"))).graph.output] == names

    if probe := os.environ.get("DETECT_PROBE"):
        image = str(images / "positive.png")
        key_path = create_key(tmp_path / "deploy.key")
        sealed = tmp_path / "sealed.dvsenc"
        encrypt_config(output, sealed, read_key(key_path))

        def run(args):
            result = subprocess.run([probe, *map(str, args)], capture_output=True, text=True, check=True)
            return json.loads(result.stdout.strip().splitlines()[-1])["detections"]

        expected = run([output, image])
        assert len(expected) == 4
        assert run([sealed, image, key_path]) == expected

    from quantize_dense import compare_detection_sessions
    import quantize_dense

    def changed(*args):
        report = compare_detection_sessions(*args)
        report["changed_image_decisions"] = 1
        return report

    onnx_before = output.with_suffix(".onnx").read_bytes()
    monkeypatch.setattr(quantize_dense, "compare_detection_sessions", changed)
    with pytest.raises(ValueError, match="탐지 판정 차이가 기준"):
        quantize_segment(config, images, log=logs.append)
    assert output.read_bytes() == config_before
    assert output.with_suffix(".onnx").read_bytes() == onnx_before

    from precision_benchmark import benchmark_precisions, parse_thread_counts

    source_before = source.read_bytes()
    counts = parse_thread_counts("1, 2, 1")
    rows = benchmark_precisions(config, images, thread_counts=counts, calibration_dir=images,
                                image_count=2, repeats=1, warmup=1, log=logs.append)
    assert [(row["precision"], row["threads"]) for row in rows] == [
        (precision, threads) for precision in ("FP32", "FP16", "INT8") for threads in (1, 2)
    ]
    assert all(row["status"] == "완료" and row["samples"] == 2 for row in rows)
    assert all(row["p95_ms"] >= row["p50_ms"] >= row["model_p50_ms"] > 0 for row in rows)
    assert all("판정 변경 0장" in row["comparison"] for row in rows[2:])
    assert all("INT8 배포 검증 미통과" in row["detail"] for row in rows[-2:])
    assert source.read_bytes() == source_before
    assert output.read_bytes() == config_before
