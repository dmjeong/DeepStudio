"""분할 ONNX(FP32) → INT8 변환 도구 (python/quantize_onnx.py) 와 C++ 실행 호환.

    best.pt ─export─▶ m.json/m.onnx (FP32, 업샘플 크기 고정) ─quantize─▶ m_int8.json/m_int8.onnx
        · 그래프를 배율 업샘플로 바꿔도 FP32 결과가 같다 (타일 보정의 전제)
        · INT8 클래스 맵이 FP32와 거의 같다
        · (SEGMENT_PROBE 가 있으면) C++ SDK가 INT8 JSON을 그대로 읽어 같은 결함을 낸다
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxruntime.quantization")
onnx = pytest.importorskip("onnx")
REGION = (96, 64)


def _images(folder, count, seed):
    """회색 바탕 + 밝은 점 몇 개 (점 = 결함 클래스 1)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        pixels = np.full((REGION[1], REGION[0], 3), 110, np.uint8) + rng.integers(0, 15, (REGION[1], REGION[0], 3),
                                                                                   dtype=np.uint8)
        for _ in range(3):
            x, y = int(rng.integers(4, REGION[0] - 12)), int(rng.integers(4, REGION[1] - 12))
            pixels[y:y + 7, x:x + 7] = 225
        Image.fromarray(pixels).save(folder / f"{index}.png")
    return folder


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """조금 학습한 U-Net Lite를 판정 영역 96×64 계약으로 내보낸다 (학습 안 한 모델은 확률이 경계에 몰려 비교가 무의미)."""
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from checkpoint import tiling_contract
    from export_onnx import export_checkpoint
    root = tmp_path_factory.mktemp("q8")
    train = _images(root / "calib", 4, 1)
    torch.manual_seed(0)
    model = build_builtin_model("unet_lite", 2, 3).eval()
    batch = torch.stack([torch.from_numpy(np.asarray(Image.open(p))).permute(2, 0, 1).float() / 255
                         for p in sorted(train.glob("*.png"))])
    mean, std = torch.tensor([.485, .456, .406])[:, None, None], torch.tensor([.229, .224, .225])[:, None, None]
    target = (batch[:, 0] > .8).long()
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    model.train()
    for _ in range(120):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model((batch - mean) / std), target, torch.tensor([.05, 1.])).backward()
        optimizer.step()
    model.eval()
    checkpoint = make_builtin_checkpoint("unet_lite", model, num_classes=2, input_size=64, in_channels=3,
                                         class_names=["background", "spot"], tiling=tiling_contract(64, 1.0, REGION))
    torch.save(checkpoint, root / "best.pt")
    export_checkpoint(root / "best.pt", root / "m.onnx", log=lambda _m: None)
    return root


def test_tiles_cover_the_whole_input():
    from quantize_onnx import _tiles
    tensor = np.arange(1 * 1 * 96 * 160, dtype=np.float32).reshape(1, 1, 96, 160)
    seen = np.zeros((96, 160), bool)
    pieces = list(_tiles(tensor, 64))
    assert all(p.shape == (1, 1, 64, 64) for p in pieces)
    for piece in pieces:   # 각 조각이 원본의 어느 위치인지 값으로 되짚는다
        first = int(piece[0, 0, 0, 0])
        y, x = divmod(first, 160)
        seen[y:y + 64, x:x + 64] = True
    assert seen.all()


def test_size_free_graph_keeps_the_fp32_result(exported):
    """업샘플을 '고정 크기 → 배율'로 바꾼 그래프가 원본과 같은 값을 내고, 다른 크기 입력도 받는다."""
    from quantize_onnx import _set_hw, size_free_graph
    original = onnx.load(str(exported / "m.onnx"))
    config = json.loads((exported / "m.json").read_text(encoding="utf-8"))
    shape = (1, 3, config["input_height"], config["input_width"])
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    flexible = _set_hw(size_free_graph(onnx.load(str(exported / "m.onnx"))), "height", "width")
    a = ort.InferenceSession(original.SerializeToString(), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(flexible.SerializeToString(), providers=["CPUExecutionProvider"])
    name = a.get_inputs()[0].name
    np.testing.assert_allclose(a.run(None, {name: x})[0], b.run(None, {name: x})[0], atol=1e-5)
    assert b.run(None, {name: np.zeros((1, 3, 32, 32), np.float32)})[0].shape[2:] == (32, 32)


def test_quantize_writes_an_int8_model_that_matches_fp32(exported):
    from quantize_onnx import compare, quantize_segment
    output = quantize_segment(exported / "m.json", exported / "calib", count=4, tile=32, log=lambda _m: None)
    config = json.loads(output.read_text(encoding="utf-8"))
    assert config["model_path"] == "m_int8.onnx" and config["export"]["precision"] == "int8"
    assert config["export"]["quantization"]["calibration_images"] == 4
    assert config["export"]["quantization"]["calibration_tiles"] == 4 * 6   # 96×64 입력을 32 타일로 = 3×2
    model = onnx.load(str(exported / "m_int8.onnx"))
    assert any(node.op_type == "QuantizeLinear" for node in model.graph.node)
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert dims[2:] == [config["input_height"], config["input_width"]]           # C++ 계약의 고정 입력
    report = compare(exported / "m.json", output, _images(exported / "check", 3, 7), log=lambda _m: None)
    assert report["images"] == 3 and report["pixel_agreement"] > 0.995
    assert report["class_iou"]["spot"] > 0.9


def test_quantize_rejects_non_dense_tasks(tmp_path):
    from quantize_onnx import quantize_segment
    (tmp_path / "c.json").write_text(json.dumps({"task": "classify", "model_path": "c.onnx"}), encoding="utf-8")
    with pytest.raises(ValueError, match="분할·Dinomaly"):
        quantize_segment(tmp_path / "c.json", tmp_path, log=lambda _m: None)


@pytest.mark.skipif(not os.environ.get("SEGMENT_PROBE"), reason="SEGMENT_PROBE(C++ segment_probe 경로)가 없습니다")
def test_cpp_sdk_runs_the_int8_model_unchanged(exported):
    """SDK 수정 없이 INT8 JSON을 읽고, 판정 기준을 넣으면 FP32와 같은 결함 목록을 낸다."""
    from quantize_onnx import quantize_segment
    output = exported / "m_int8.json"
    if not output.exists():
        quantize_segment(exported / "m.json", exported / "calib", count=4, tile=32, log=lambda _m: None)
    image = _images(exported / "probe", 1, 11) / "0.png"
    rows = {}
    for name in ("m", "m_int8"):
        config = json.loads((exported / f"{name}.json").read_text(encoding="utf-8"))
        config.setdefault("postprocessing", {})["defect_rules"] = {"min_area": 10, "min_score": 0.0}
        path = exported / f"{name}_rules.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        result = subprocess.run([os.environ["SEGMENT_PROBE"], str(path), str(image), str(exported / f"{name}.png")],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        rows[name] = sorted((int(v[0]), int(v[1]), int(v[2])) for v in
                            (line.split() for line in result.stdout.split("\n")[1:] if line.strip()))
    assert rows["m"] and len(rows["m"]) == len(rows["m_int8"])
    for a, b in zip(rows["m"], rows["m_int8"]):    # 같은 클래스, 위치 ±2 px
        assert a[0] == b[0] and abs(a[1] - b[1]) <= 2 and abs(a[2] - b[2]) <= 2
