"""두 분할 모델을 하나로: 표면 결함(B)이 있으면 B, 없으면 칩 상태(A). 데스크톱에서 두 모델을 따로 돌린 결과와 같다."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python"), str(ROOT / "tests")]
torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
PROBE = os.environ.get("SEGMENT_PROBE")
REGION = (96, 64)


def _model(tmp_path, name, classes, scale, bright_class, dark_class):
    """밝은 점·어두운 점을 몇 번 학습한 초경량 U-Net (test_cpp_segment_rules와 같은 방식)."""
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from checkpoint import tiling_contract
    from dataset import scale_image, scaled_size
    from test_cpp_segment_rules import _pixels
    torch.manual_seed(len(name))
    model = build_builtin_model("unet_lite", len(classes), 3).eval()
    pixels = _pixels(REGION)
    if scale != 1:
        width, height = scaled_size(*REGION, scale)
        pixels = scale_image(pixels, (height, width), "INTER_AREA")
    x = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1)[None].float() / 255
    mean, std = torch.tensor([.485, .456, .406])[:, None, None], torch.tensor([.229, .224, .225])[:, None, None]
    target = torch.zeros(x.shape[-2:], dtype=torch.long)
    if bright_class:
        target[x[0, 0] > .7] = bright_class
    if dark_class:
        target[x[0, 0] < .3] = dark_class
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    weight = torch.tensor([.05] + [1.] * (len(classes) - 1))
    for _ in range(200):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model((x - mean) / std), target[None], weight).backward()
        optimizer.step()
    path = tmp_path / f"{name}.pt"
    torch.save(make_builtin_checkpoint("unet_lite", model, num_classes=len(classes), input_size=64, in_channels=3,
                                       class_names=classes, tiling=tiling_contract(64, scale, REGION)), path)
    return path


@pytest.fixture
def models(tmp_path):
    a = _model(tmp_path, "state", ["background", "shift", "double"], 0.5, 1, 2)   # 칩 상태: 50%
    b = _model(tmp_path, "surface", ["background", "crack"], 1.0, 1, 0)          # 표면: 원본, 밝은 점만
    return a, b


def test_merged_model_keeps_surface_over_state_and_carries_both_rule_sets(tmp_path, models):
    from merge_segmenters import export_merged
    from core.segment_defects import save_model_rules
    a, b = models
    save_model_rules(a, {"min_area": 4, "min_score": 0.5})
    save_model_rules(b, {"min_area": 1, "min_score": 0.0, "per_class": {"crack": {"min_area": 2, "min_score": 0.9}}})
    result = export_merged(a, b, tmp_path / "merged.onnx", log=lambda _m: None)
    config = json.loads((tmp_path / "merged.json").read_text(encoding="utf-8"))
    assert config["class_names"] == ["background", "shift", "double", "crack"]
    assert config["preprocessing"]["whole_image"]["region"] == list(REGION)
    assert (config["input_height"], config["input_width"]) == (64, 96)  # 입력 = 판정 영역 (채움은 그래프 안)
    assert config["postprocessing"]["defect_rules"]["classes"] == [
        {"class_id": 1, "class_name": "shift", "min_area": 4, "min_score": 0.5},
        {"class_id": 2, "class_name": "double", "min_area": 4, "min_score": 0.5},
        {"class_id": 3, "class_name": "crack", "min_area": 2, "min_score": 0.9}]
    assert result["class_names"] == config["class_names"]


def test_merging_refuses_overlapping_class_names_and_non_patch_models(tmp_path, models):
    from merge_segmenters import export_merged
    a, _ = models
    with pytest.raises(ValueError, match="겹칩니다"):
        export_merged(a, a, tmp_path / "m.onnx", log=lambda _m: None)


@pytest.mark.skipif(not PROBE, reason="SEGMENT_PROBE(C++ segment_probe 경로)가 없습니다")
def test_cpp_merged_mask_equals_running_both_models_on_the_desktop(tmp_path, models):
    from merge_segmenters import export_merged
    from core.segment_defects import upsample_labels
    from test_cpp_segment_rules import _image
    a, b = models
    image = _image(tmp_path / "part.png", REGION)
    export_merged(a, b, tmp_path / "merged.onnx", log=lambda _m: None)
    output = subprocess.run([PROBE, str(tmp_path / "merged.json"), str(image), str(tmp_path / "mask.png")],
                            capture_output=True, text=True, timeout=120)
    assert output.returncode == 0, output.stderr
    merged = np.asarray(Image.open(tmp_path / "mask.png"))

    def labels(checkpoint):
        """데스크톱 엔진의 '판정 영역 한 번에'와 같은 계산 (축소 → 정규화 → 32 배수 채움 → 자르기 → nearest 확대)."""
        from builtin_models import load_builtin_checkpoint
        from dataset import scale_image, scaled_size
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model, scale = load_builtin_checkpoint(payload), payload["tiling"]["scale"]
        pixels = np.asarray(Image.open(image).convert("RGB"))
        width, height = scaled_size(*REGION, scale)
        if scale != 1:
            pixels = scale_image(pixels, (height, width), "INTER_AREA")
        x = (torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).float() / 255
             - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        padded = torch.zeros(3, -(-height // 32) * 32, -(-width // 32) * 32)
        padded[:, :height, :width] = x
        with torch.no_grad():
            label = model(padded[None])[0, :, :height, :width].argmax(0).numpy()
        return upsample_labels(label, *REGION)

    state, surface = labels(a), labels(b)
    expected = np.where(surface > 0, surface + 2, state)  # crack = 3, shift/double = 1/2
    assert (surface > 0).any() and (state > 0).any()
    assert np.array_equal(merged[surface > 0], expected[surface > 0])  # 표면 결함이 우선
    # 칩 상태는 정수배 평균(풀링)과 8비트 INTER_AREA 반올림 차이로 경계 픽셀이 드물게 다를 수 있다
    assert (merged != expected).mean() < 0.002


def test_export_page_merges_two_models_in_one_run(tmp_path, models):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from widgets.export_widget import ExportWorker
    QApplication.instance() or QApplication([])
    a, b = models
    worker = ExportWorker(str(a), str(tmp_path / "line.onnx"), 17, False, False, True,
                          overrides={"region_size": list(REGION)}, merge_with=str(b))
    done, errors = [], []
    worker.completed.connect(done.append)
    worker.error.connect(errors.append)
    worker.run()  # 스레드 없이 같은 작업
    assert not errors and done[0]["merged"] and done[0]["class_names"] == ["background", "shift", "double", "crack"]
    assert (tmp_path / "line.json").is_file()
