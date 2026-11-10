"""경량 사전학습 백본 U-Net (MobileNetV2 인코더 + Lite 폭 디코더): 데스크톱 학습·내보내기·C++ 판정까지.

    ResNet18 U-Net처럼 1/32까지 넓게 보면서(조각 판정 방지) CPU에서 약 4배 빠르다.
    전역 연산(SE·전역 평균)이 없어야 패치 학습 = 판정 영역 전체 추론 (데스크톱·C++ 모두 전체 한 번).
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
pytest.importorskip("torchvision")


def _asset_available():
    from builtin_assets import builtin_asset_path
    try:
        builtin_asset_path("mobilenet_v2")
        return True
    except Exception:
        return False


def test_structure_has_no_global_operation_and_keeps_any_size():
    from builtin_models import build_builtin_model
    model = build_builtin_model("unet_mobilenetv2", 4, 3).eval()
    banned = (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d, torch.nn.Linear)
    assert not any(isinstance(m, banned) for m in model.modules())       # 패치 학습 = 전체 추론
    with torch.no_grad():
        assert model(torch.randn(1, 3, 75, 110)).shape == (1, 4, 75, 110)  # 32의 배수가 아니어도
    gray = build_builtin_model("unet_mobilenetv2", 2, 1).eval()
    assert gray(torch.randn(1, 1, 64, 64)).shape == (1, 2, 64, 64)
    assert model.gradcam_target_layer == "encoder.17" and not model.global_attention
    assert sum(p.numel() for p in model.parameters()) < 3e6


def test_pretrained_encoder_is_the_official_imagenet_weights():
    if not _asset_available():   # 실행 시점에 확인 (다른 테스트가 가중치 폴더 설정을 바꿀 수 있다)
        pytest.skip("MobileNetV2 ImageNet 가중치가 없습니다")
    import torchvision
    from builtin_assets import builtin_asset_path
    from builtin_models import build_builtin_model
    model = build_builtin_model("unet_mobilenetv2", 3, 3, pretrained=True)
    assert model.weight_provenance["source"] == "imagenet"
    reference = torchvision.models.mobilenet_v2(weights=None)
    reference.load_state_dict(torch.load(builtin_asset_path("mobilenet_v2"), map_location="cpu", weights_only=True))
    assert torch.equal(model.encoder[0][0].weight, reference.features[0][0].weight)
    assert torch.equal(model.encoder[17].conv[0][0].weight, reference.features[17].conv[0][0].weight)


def test_app_offers_it_on_the_desktop_with_imagenet_finetuning():
    from builtin_assets import required_pretrained_asset
    from core.model_registry import builtin_model_specs
    from core.training_modes import BUILTIN_ADAPTER_IDS, BUILTIN_SCRATCH_ONLY_IDS
    spec = next(s for s in builtin_model_specs() if s.model_id == "unet_mobilenetv2")
    assert spec.task == "segment" and spec.metadata.get("surfaces") == ["desktop"]
    assert "unet_mobilenetv2" in BUILTIN_ADAPTER_IDS and "unet_mobilenetv2" not in BUILTIN_SCRATCH_ONLY_IDS
    assert required_pretrained_asset("segment", "builtin_finetune", "unet_mobilenetv2") == "mobilenet_v2"


def _dataset(root, size=(96, 64)):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)
        for index in range(2):
            pixels = np.full((size[1], size[0], 3), 110, np.uint8)
            mask = np.zeros((size[1], size[0]), np.uint8)
            pixels[20:28, 30:44], mask[20:28, 30:44] = 220, 1
            Image.fromarray(pixels).save(root / "images" / split / f"{index}.png")
            Image.fromarray(mask).save(root / "masks" / split / f"{index}.png")
    return root


def test_patch_training_records_backbone_learning_rate_and_tiling(tmp_path):
    from train_builtin import train_builtin
    logs = []
    best = train_builtin("unet_mobilenetv2", _dataset(tmp_path / "data"), num_classes=2, input_size=64,
                         in_channels=3, epochs=1, batch_size=2, output_dir=tmp_path / "run", device="cpu",
                         num_workers=0, class_names=["background", "spot"], patch_training=True,
                         patches_per_image=2, backbone_lr_mult=0.1, log=lambda m: logs.append(str(m)))
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert checkpoint["model_id"] == "unet_mobilenetv2"
    assert checkpoint["preprocessing"]["tiling"]["region"] == [96, 64]


@pytest.mark.skipif(not os.environ.get("SEGMENT_PROBE"), reason="SEGMENT_PROBE(C++ segment_probe 경로)가 없습니다")
def test_cpp_segment_matches_the_desktop_judgement(tmp_path):
    pytest.importorskip("onnxruntime")
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from checkpoint import tiling_contract
    from core.inference_loading import load_inference_engine
    from core.segment_defects import judge, save_model_rules
    from export_onnx import export_checkpoint
    torch.manual_seed(1)
    model = build_builtin_model("unet_mobilenetv2", 3, 3).eval()
    with torch.no_grad():                               # 덩어리가 여럿 생기게 마지막 층 편향을 키운다
        model.head.bias.copy_(torch.tensor([0.0, 0.2, 0.1]))
    checkpoint = make_builtin_checkpoint("unet_mobilenetv2", model, num_classes=3, input_size=64, in_channels=3,
                                         class_names=["background", "crack", "fm"],
                                         tiling=tiling_contract(64, 1.0, (96, 64)))
    path = tmp_path / "best.pt"
    torch.save(checkpoint, path)
    rng = np.random.default_rng(2)
    image = tmp_path / "part.png"
    Image.fromarray(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)).save(image)
    engine = load_inference_engine(path, segment_review={"min_area": 1, "min_score": 0, "split_dirs": []})
    blobs = engine.infer(str(image)).details["defects"]
    assert blobs
    rules = {"min_area": sorted(b["area"] for b in blobs)[len(blobs) // 2], "min_score": 0.0}
    save_model_rules(path, rules)
    kept = judge({"defects": blobs}, rules)["kept"]
    export_checkpoint(path, tmp_path / "m.onnx", log=lambda _m: None)
    config = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert config["cpp_supported"] and config["preprocessing"]["whole_image"]["region"] == [96, 64]
    output = subprocess.run([os.environ["SEGMENT_PROBE"], str(tmp_path / "m.json"), str(image), str(tmp_path / "k.png")],
                            capture_output=True, text=True, timeout=120)
    assert output.returncode == 0, output.stderr
    cpp = sorted(tuple(int(v) for v in line.split()[:6]) for line in output.stdout.split("\n")[1:] if line.strip())
    assert cpp == sorted((b["class_id"], b["x"], b["y"], b["w"], b["h"], b["area"]) for b in kept)
