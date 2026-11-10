"""데스크톱 판정 = C++ SDK 판정: 패치 학습 분할 모델을 내보내 C++ Segment 결과와 데스크톱 결과를 비교한다.

    체크포인트 + best.rules.json ─▶ ONNX 내보내기 (판정 영역 고정 입력 + 판정 기준)
         │                                   │
    데스크톱 엔진 → judge(기준) 결함 목록    C++ segment_probe → 결함 목록 · 마스크
         └──────────────── 같아야 한다 ─────────┘

C++ 프로그램은 SEGMENT_PROBE 환경 변수로 넘긴다 (cpp/ 빌드의 segment_probe). 없으면 건너뛴다.
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
pytest.importorskip("onnxruntime")
PROBE = os.environ.get("SEGMENT_PROBE")


def _pixels(size):
    """회색 바탕에 밝은 점(crack)·어두운 점(fm) 여러 크기: 덩어리 크기·확신이 제각각이 된다."""
    rng = np.random.default_rng(5)
    pixels = np.full((size[1], size[0], 3), 128, np.uint8) + rng.integers(0, 12, (size[1], size[0], 3), dtype=np.uint8)
    for index in range(14):
        x, y = int(rng.integers(0, size[0] - 8)), int(rng.integers(0, size[1] - 8))
        w, h = int(rng.integers(2, 9)), int(rng.integers(2, 9))  # 50% 축소 뒤에도 남는 크기
        pixels[y:y + h, x:x + w] = (235 - index * 4) if index % 2 else (20 + index * 4)
    return pixels


def _checkpoint(tmp_path, scale, region):
    """작은 영상의 점무늬를 몇 번 학습한 모델: 크기·확신이 제각각인 덩어리가 여러 개 생긴다."""
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from checkpoint import tiling_contract
    torch.manual_seed(3)
    model = build_builtin_model("unet_lite", 3, 3)
    pixels = _pixels(region or (70, 45))
    if scale != 1:  # 추론과 같은 배율로 줄인 영상에서 학습 (패치 학습과 같은 순서)
        from dataset import scale_image, scaled_size
        width, height = scaled_size(pixels.shape[1], pixels.shape[0], scale)
        pixels = scale_image(pixels, (height, width), "INTER_AREA")
    x = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1)[None].float() / 255
    mean, std = torch.tensor([.485, .456, .406])[:, None, None], torch.tensor([.229, .224, .225])[:, None, None]
    target = torch.zeros(x.shape[-2:], dtype=torch.long)
    target[x[0, 0] > .7], target[x[0, 0] < .3] = 1, 2  # 밝은 점 = crack, 어두운 점 = fm
    model.eval()  # BN은 고정 통계 그대로: 몇 번의 학습으로도 추론 출력이 학습 출력과 같다
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    weight = torch.tensor([.05, 1., 1.])  # 결함 픽셀이 적어도 배경으로만 쏠리지 않게
    for _ in range(200):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model((x - mean) / std), target[None], weight).backward()  # 추론과 같은 정규화
        optimizer.step()
    checkpoint = make_builtin_checkpoint("unet_lite", model, num_classes=3, input_size=64, in_channels=3,
                                         class_names=["background", "crack", "fm"],
                                         tiling=tiling_contract(64, scale, region))
    path = tmp_path / f"best_{int(scale * 100)}.pt"
    torch.save(checkpoint, path)
    return path


def _image(path, size):
    Image.fromarray(_pixels(size)).save(path)
    return path


def test_export_records_the_whole_region_input_and_rules(tmp_path):
    from export_onnx import export_checkpoint
    from core.segment_defects import save_model_rules
    checkpoint = _checkpoint(tmp_path, 0.5, (90, 61))
    save_model_rules(checkpoint, {"min_area": 2, "min_score": 0.1, "per_class": {"fm": {"min_area": 9, "min_score": 0.6}}})
    export_checkpoint(checkpoint, tmp_path / "m.onnx", log=lambda _m: None)
    config = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    whole = config["preprocessing"]["whole_image"]
    # 90×61을 50%로 줄이면 45×30 (학습 데이터와 같은 scaled_size 반올림) → 32의 배수로 채운 고정 입력 64×32
    assert whole["region"] == [90, 61] and whole["scaled"] == [45, 30] and whole["interpolation"] == "INTER_AREA"
    assert (config["input_height"], config["input_width"]) == (32, 64) and config["cpp_supported"]
    assert config["postprocessing"]["defect_rules"] == {
        "min_area": 2, "min_score": 0.1, "classes": [{"class_id": 2, "class_name": "fm", "min_area": 9, "min_score": 0.6}]}


def test_patch_model_without_a_region_asks_for_it(tmp_path):
    from export_onnx import export_checkpoint
    checkpoint = _checkpoint(tmp_path, 1.0, None)
    with pytest.raises(Exception, match="판정 영역"):
        export_checkpoint(checkpoint, tmp_path / "m.onnx", log=lambda _m: None)
    export_checkpoint(checkpoint, tmp_path / "m.onnx", overrides={"region_size": [70, 45]}, log=lambda _m: None)
    assert json.loads((tmp_path / "m.json").read_text())["preprocessing"]["whole_image"]["region"] == [70, 45]


def test_export_command_line_takes_the_region():
    from export_onnx import parse_args
    assert parse_args(["--checkpoint", "best.pt", "--region", "3100x1500"]).region == [3100, 1500]
    assert parse_args(["--checkpoint", "best.pt"]).region is None       # 없으면 학습 때 기록한 크기를 쓴다
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "best.pt", "--region", "3100"])


@pytest.mark.skipif(not PROBE, reason="SEGMENT_PROBE(C++ segment_probe 경로)가 없습니다")
# 50%는 정수배(90×62 → 45×31)로 둔다: 정수배가 아닌 INTER_AREA는 OpenCV 판마다 반올림이 1px 다를 수 있다
# (Python opencv 5.0 ↔ C++ 4.6에서 61→30 축소의 한 픽셀). 3504×1504 → 1752×752 같은 실제 판정 영역은 정수배다.
@pytest.mark.parametrize("scale,region", [(1.0, (70, 45)), (0.5, (90, 62))])
def test_cpp_segment_matches_the_desktop_judgement(tmp_path, scale, region):
    from export_onnx import export_checkpoint
    from core.inference_loading import load_inference_engine
    from core.segment_defects import judge, save_model_rules
    checkpoint = _checkpoint(tmp_path, scale, region)
    image = _image(tmp_path / "part.png", region)
    engine = load_inference_engine(checkpoint, segment_review={"min_area": 1, "min_score": 0, "split_dirs": []})
    details = engine.infer(str(image)).details
    blobs = details["defects"]
    assert len(blobs) >= 4, "무작위 모델이 덩어리를 충분히 만들어야 비교가 의미 있다"
    # 결함 일부만 남는 기준: 면적·점수 중앙값 근처
    areas = sorted(b["area"] for b in blobs)
    scores = sorted(b["score"] for b in blobs)
    rules = {"min_area": areas[len(areas) // 2], "min_score": 0.0,
             "per_class": {"fm": {"min_area": 1, "min_score": scores[len(scores) // 2]}}}
    save_model_rules(checkpoint, rules)
    kept = judge(details, rules)["kept"]
    assert 0 < len(kept) < len(blobs)
    export_checkpoint(checkpoint, tmp_path / "m.onnx", log=lambda _m: None)
    output = subprocess.run([PROBE, str(tmp_path / "m.json"), str(image), str(tmp_path / "mask.png")],
                            capture_output=True, text=True, timeout=120)
    assert output.returncode == 0, output.stderr
    lines = output.stdout.split("\n")
    assert int(lines[0].split()[1]) == len(blobs) - len(kept)
    cpp = sorted(tuple(int(v) for v in line.split()[:6]) + (round(float(line.split()[6]), 3),)
                 for line in lines[1:] if line.strip())
    desktop = sorted((b["class_id"], b["x"], b["y"], b["w"], b["h"], b["area"], round(b["score"], 3)) for b in kept)
    assert cpp == desktop
    mask = np.asarray(Image.open(tmp_path / "mask.png"))
    assert mask.shape == (region[1], region[0])
    assert int((mask > 0).sum()) == sum(b["area"] for b in kept)  # 지운 덩어리는 배경


def test_export_page_shows_the_region_and_the_rules_that_go_into_the_model(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from core.segment_defects import save_model_rules
    from widgets.export_widget import ExportWidget
    QApplication.instance() or QApplication([])
    checkpoint = _checkpoint(tmp_path, 1.0, (70, 45))
    widget = ExportWidget()
    widget.ckpt_edit.setText(str(checkpoint))
    widget._refresh_deploy_preview()
    assert not widget.region_row.isHidden() and (widget.region_w.value(), widget.region_h.value()) == (70, 45)
    assert "판정 기준 없음" in widget.deploy_info.text()
    save_model_rules(checkpoint, {"min_area": 3, "min_score": 0.0, "per_class": {"crack": {"min_score": 0.85}}})
    widget._refresh_deploy_preview()
    assert "crack 면적 ≥ 3px · 점수 ≥ 0.85" in widget.deploy_info.text()
    widget.deleteLater()
