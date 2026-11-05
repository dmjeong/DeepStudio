"""학습 시 중앙 크롭 + 패치 학습 분할 모델: 학습·추론·내보내기·C++가 같은 영역(크롭)을 본다.

    원본 ──중앙 크롭──▶ 크롭 영역 ──패치 학습──▶ 체크포인트 (center_crop + 판정 영역 = 크롭 크기)
                                                     │
     데스크톱 추론: 원본 → 같은 크롭 → 판정            C++ Segment: 원본 → 같은 크롭 → 판정 영역 한 번에
     (이전: 기본 모델 체크포인트에 크롭이 없어 원본 전체를 추론 → 학습에서 본 적 없는 바깥이 불량으로 나왔다)
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
torch = pytest.importorskip("torch")
CROP = {"width": 70, "height": 45}


def _dataset(root, size=(110, 75)):
    """원본 110×75, 가운데 70×45만 쓰는 데이터 (크롭 밖은 학습에 들어가지 않는다)."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        for index in range(2):
            pixels = np.full((size[1], size[0], 3), 120, np.uint8)
            mask = np.zeros((size[1], size[0]), np.uint8)
            pixels[35:40, 50:58], mask[35:40, 50:58] = 230, 1
            Image.fromarray(pixels).save(root / "images" / split / f"{index}.png")
            Image.fromarray(mask).save(root / "masks" / split / f"{index}.png")
    return root


def test_builtin_training_records_the_crop_and_inference_uses_it(tmp_path):
    from crop_dataset import prepare_crop_dataset
    from train_builtin import train_builtin
    from core.inference_loading import load_inference_engine
    root = _dataset(tmp_path / "data")
    cropped = prepare_crop_dataset(root, tmp_path / "generated", "segment", CROP, num_classes=2)
    best = train_builtin("unet_lite", cropped, num_classes=2, input_size=32, in_channels=3, epochs=1,
                         batch_size=2, output_dir=tmp_path / "run", device="cpu", num_workers=0,
                         class_names=["background", "spot"], patch_training=True, patches_per_image=2,
                         center_crop=CROP, log=lambda _m: None)
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert checkpoint["center_crop"] == CROP and checkpoint["preprocessing"]["center_crop"] == CROP
    assert checkpoint["preprocessing"]["tiling"]["region"] == [70, 45]  # 판정 영역 = 크롭 크기
    engine = load_inference_engine(best, segment_review={"min_area": 1, "min_score": 0, "split_dirs": []})
    details = engine.infer(str(root / "images" / "val" / "0.png")).details
    assert details["crop_box"] == [20, 15, 90, 60]                      # 원본 110×75의 가운데
    assert sum(details["pixel_counts"].values()) == 70 * 45            # 크롭 밖은 추론하지 않는다


def test_builtin_checkpoint_without_crop_is_unchanged(tmp_path):
    from train_builtin import train_builtin
    root = _dataset(tmp_path / "data", size=(64, 48))
    best = train_builtin("unet_lite", root, num_classes=2, input_size=32, in_channels=3, epochs=1, batch_size=2,
                         output_dir=tmp_path / "run", device="cpu", num_workers=0,
                         class_names=["background", "spot"], log=lambda _m: None)
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert "center_crop" not in checkpoint and "center_crop" not in checkpoint["preprocessing"]


@pytest.mark.parametrize("desktop,expected", [(True, CROP), (False, None)])
def test_only_desktop_training_requests_record_the_crop(tmp_path, monkeypatch, desktop, expected):
    """웹 학습 요청은 그대로 둔다 (데스크톱 요청만 class_report와 함께 크롭을 남긴다)."""
    import webapp.worker as worker
    from core.project import ProjectManager
    project = ProjectManager.create_new("crop", "segment", str(tmp_path / "p"), ["background", "spot"])
    _dataset(Path(project.data.root))
    project.model.model_id = "unet_lite"
    cfg = project.training
    cfg.patchcore_crop_enabled, cfg.patchcore_crop_width, cfg.patchcore_crop_height = True, 70, 45
    seen = {}

    def capture(model_id, data_root, **kwargs):
        seen.update(kwargs)
        return tmp_path / "missing.pt"   # 학습하지 않고 인자만 확인 (없는 파일 → 취소 기록)

    monkeypatch.setattr(worker, "train_builtin", capture, raising=False)
    import train_builtin as module
    monkeypatch.setattr(module, "train_builtin", capture)
    context = SimpleNamespace(emit=lambda *a: None, cancelled=lambda: False)
    worker._train_builtin_project(context, project, "cpu", class_report=desktop)
    assert seen["center_crop"] == expected


def test_custom_training_records_the_crop_size_as_the_region(tmp_path):
    from core.project import ProjectManager
    from core.trainer import TrainWorker
    project = ProjectManager.create_new("crop", "segment", str(tmp_path / "p"), ["background", "spot"])
    _dataset(Path(project.data.root))
    cfg = project.training
    cfg.patch_training, cfg.input_size = True, 32
    cfg.patchcore_crop_enabled, cfg.patchcore_crop_width, cfg.patchcore_crop_height = True, 70, 45
    metadata = TrainWorker(project)._checkpoint_metadata()
    assert metadata["center_crop"] == CROP and metadata["tiling"]["region"] == [70, 45]  # 이전: 원본 [110, 75]


@pytest.mark.skipif(not os.environ.get("SEGMENT_PROBE"), reason="SEGMENT_PROBE(C++ segment_probe 경로)가 없습니다")
def test_cpp_segment_crops_like_the_desktop(tmp_path):
    """크롭 모델을 내보내 C++가 데스크톱과 같은 결함 목록(원본 좌표)을 낸다."""
    pytest.importorskip("onnxruntime")
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from checkpoint import tiling_contract
    from core.inference_loading import load_inference_engine
    from core.segment_defects import judge, save_model_rules
    from export_onnx import export_checkpoint
    rng = np.random.default_rng(4)
    original = np.full((75, 110, 3), 128, np.uint8) + rng.integers(0, 12, (75, 110, 3), dtype=np.uint8)
    for index in range(10):          # 크롭 안팎에 점: 크롭 밖 점은 결과에 나오면 안 된다
        x, y = int(rng.integers(0, 104)), int(rng.integers(0, 69))
        original[y:y + 5, x:x + 5] = 235 - index * 6
    image = tmp_path / "part.png"
    Image.fromarray(original).save(image)
    torch.manual_seed(3)
    model = build_builtin_model("unet_lite", 2, 3).eval()
    x = torch.from_numpy(original[15:60, 20:90].copy()).permute(2, 0, 1)[None].float() / 255
    mean, std = torch.tensor([.485, .456, .406])[:, None, None], torch.tensor([.229, .224, .225])[:, None, None]
    target = (x[0, 0] > .8).long()
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(150):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model((x - mean) / std), target[None], torch.tensor([.05, 1.])).backward()
        optimizer.step()
    checkpoint = make_builtin_checkpoint("unet_lite", model, num_classes=2, input_size=64, in_channels=3,
                                         class_names=["background", "spot"], tiling=tiling_contract(64, 1.0, (70, 45)))
    checkpoint["center_crop"] = checkpoint["preprocessing"]["center_crop"] = CROP
    path = tmp_path / "best.pt"
    torch.save(checkpoint, path)
    engine = load_inference_engine(path, segment_review={"min_area": 1, "min_score": 0, "split_dirs": []})
    blobs = engine.infer(str(image)).details["defects"]
    assert blobs and all(20 <= b["x"] and b["x"] + b["w"] <= 90 and 15 <= b["y"] and b["y"] + b["h"] <= 60
                         for b in blobs)
    save_model_rules(path, {"min_area": 1, "min_score": 0.0})
    kept = judge({"defects": blobs}, {"min_area": 1, "min_score": 0.0})["kept"]
    export_checkpoint(path, tmp_path / "m.onnx", log=lambda _m: None)
    config = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert config["preprocessing"]["center_crop"] == CROP and config["preprocessing"]["whole_image"]["region"] == [70, 45]
    output = subprocess.run([os.environ["SEGMENT_PROBE"], str(tmp_path / "m.json"), str(image), str(tmp_path / "mask.png")],
                            capture_output=True, text=True, timeout=120)
    assert output.returncode == 0, output.stderr
    cpp = sorted(tuple(int(v) for v in line.split()[:6]) for line in output.stdout.split("\n")[1:] if line.strip())
    desktop = sorted((b["class_id"], b["x"], b["y"], b["w"], b["h"], b["area"]) for b in kept)
    assert cpp == desktop
    mask = np.asarray(Image.open(tmp_path / "mask.png"))
    assert mask.shape == (75, 110) and not mask[:15].any() and not mask[:, :20].any()  # 크롭 밖은 배경
