"""Actual PatchCore export -> C++ loader, preprocessing, score/map and decision.

Set ANOMALY_PROBE to the cpp/ build's anomaly_probe executable. Random backbone
weights check the deployment contract only; this is not an accuracy benchmark.
The native map must match Python ONNX Runtime, including training-image queries
whose near-zero kNN distances are sensitive to FP32 cancellation across engines.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
PROBE = os.environ.get("ANOMALY_PROBE")


@pytest.mark.skipif(not PROBE, reason="ANOMALY_PROBE native executable required")
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("aggregation", ["align_then_pool_v1", "pool_then_align_v2"])
@pytest.mark.parametrize("crop", [None, {"width": 48, "height": 40}])
@pytest.mark.parametrize("training_query", [False, True])
def test_actual_export_loads_and_matches_python(tmp_path, legacy, aggregation, crop, training_query):
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    ort = pytest.importorskip("onnxruntime")
    from export_patchcore_onnx import export_patchcore_checkpoint
    from patchcore import LEGACY_SCORE_DEFINITION, STANDARD_SCORE_DEFINITION, PatchCore
    from patchcore_data import input_tensor
    from torchvision.models import resnet18

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(81)
            weights = tmp_path / "weights.pth"
            torch.save(resnet18(weights=None).state_dict(), weights)
            pc = PatchCore(backbone_name="resnet18", pretrained=False, backbone_weights=str(weights),
                           input_size=64, max_candidates=128, max_memory_bank=32,
                           sampling_ratio=.5, n_neighbors=3, center_crop=crop,
                           feature_aggregation=aggregation,
                           score_definition=LEGACY_SCORE_DEFINITION if legacy else STANDARD_SCORE_DEFINITION)
        pixels = np.random.default_rng(71).integers(0, 256, (80, 96, 3), dtype=np.uint8)
        image = tmp_path / "sample.png"
        assert cv2.imwrite(str(image), pixels)
        x = input_tensor(image, 64, center_crop=crop).unsqueeze(0)
        pc.fit([x])
        if not training_query:
            pixels = np.random.default_rng(72).integers(0, 256, (80, 96, 3), dtype=np.uint8)
            assert cv2.imwrite(str(image), pixels)
            x = input_tensor(image, 64, center_crop=crop).unsqueeze(0)
        pc.anomaly_threshold = .5
        checkpoint = tmp_path / "best.pt"
        pc.save(checkpoint)
        exported = export_patchcore_checkpoint(checkpoint, tmp_path / "model.onnx", log=lambda _: None)
        config = json.loads(Path(exported["config_path"]).read_text())
        assert config["export"]["feature_aggregation"] == aggregation
        scores, maps = pc.predict(x)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(exported["output_path"], options, providers=["CPUExecutionProvider"])
        ort_score, ort_map = session.run(None, {session.get_inputs()[0].name: x.numpy()})
        np.testing.assert_allclose(ort_score, scores, atol=1e-3, rtol=1e-3)
        if not training_query:
            np.testing.assert_allclose(ort_map[0, 0], maps[0], atol=1e-3, rtol=1e-3)
        output = subprocess.run([PROBE, exported["config_path"], str(image), str(tmp_path / "map.f32")],
                                capture_output=True, text=True, timeout=90, check=False)
        assert output.returncode == 0, output.stderr
        result = json.loads(output.stdout)
        np.testing.assert_allclose(result["score"], scores[0], atol=1e-3, rtol=1e-3)
        assert result["threshold"] == pc.anomaly_threshold
        assert result["anomalous"] == bool(scores[0] >= pc.anomaly_threshold)
        width, height = (crop["width"], crop["height"]) if crop else (96, 80)
        expected = np.zeros((80, 96), np.float32)
        left, top = (96 - width) // 2, (80 - height) // 2
        expected[top:top+height, left:left+width] = cv2.resize(ort_map[0, 0], (width, height))
        actual = np.fromfile(tmp_path / "map.f32", dtype=np.float32).reshape(80, 96)
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
    finally:
        torch.set_num_threads(previous_threads)
