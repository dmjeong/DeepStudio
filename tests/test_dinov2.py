"""DINOv2 기본 모델 (ViT-S/14·ViT-B/14 분류·분할): 구조, 가중치 검증, 데스크톱 전용 목록, 추론·내보내기·학습 연결."""

from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
torch = pytest.importorskip("torch")

DINO_IDS = ("dinov2_vits14_cls", "dinov2_vitb14_cls", "dinov2_vits14_seg", "dinov2_vitb14_seg")


def test_registry_lists_dinov2_on_the_desktop_only():
    """웹·API 목록(기본)은 그대로, 데스크톱 화면과 설치본 카탈로그에만 보인다. get()은 어디서나 찾는다."""
    from core.model_registry import ModelRegistry
    from core.training_modes import BUILTIN_ADAPTER_IDS, training_capabilities
    registry = ModelRegistry.builtin()
    shared = {spec.model_id for spec in registry.list()}
    desktop = {spec.model_id for spec in registry.list(surface="desktop")}
    assert not shared & set(DINO_IDS) and set(DINO_IDS) <= desktop
    assert shared <= desktop  # 다른 모델은 그대로
    assert {item["model_id"] for item in registry.as_dict()} == shared  # 웹 화면 카탈로그
    assert [s.model_id for s in registry.list("segment", surface="desktop") if s.family == "DINOv2"] == \
        ["dinov2_vits14_seg", "dinov2_vitb14_seg"]
    assert registry.get("dinov2_vits14_cls").display_name == "DINOv2 ViT-S/14"
    assert set(DINO_IDS) <= BUILTIN_ADAPTER_IDS
    # main의 DINOv2 분할 모델 ID(dinov2_vits14)는 이 브랜치에서 쓰지 않는다: 체크포인트가 서로 섞이지 않게
    from builtin_models import BUILTIN_MODEL_SPECS
    assert "dinov2_vits14" not in BUILTIN_MODEL_SPECS and "dinov2_vits14" not in desktop
    assert training_capabilities("segment", "builtin_finetune", model_id="dinov2_vits14_seg")["engine"] == "builtin"


@pytest.mark.parametrize("model_id", DINO_IDS)
@pytest.mark.parametrize("channels", [1, 3])
def test_models_accept_any_size_and_keep_the_output_size(model_id, channels):
    from builtin_models import build_builtin_model
    torch.manual_seed(0)
    model = build_builtin_model(model_id, 3, channels).eval()  # 가중치 없이 (구조만)
    assert model.global_attention and model.backbone_lr_scale == .1
    x = torch.randn(1, channels, 100, 130)  # 14의 배수가 아님 → 안에서 채운다
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == ((1, 3) if model.task == "classify" else (1, 3, 100, 130))
    assert torch.isfinite(y).all()


def test_classifier_exposes_a_patch_map_for_gradcam():
    from builtin_models import build_builtin_model
    model = build_builtin_model("dinov2_vits14_cls", 2, 3).eval()
    seen = []
    model.get_submodule(model.gradcam_target_layer).register_forward_hook(lambda _m, _i, out: seen.append(out.shape))
    model(torch.randn(1, 3, 224, 224))
    assert seen == [torch.Size([1, 384, 16, 16])]


def test_backbone_keeps_the_official_layout_and_position_rule(tmp_path):
    """공식 키 이름 그대로 strict 로드. 학습 격자(37×37)는 보간하지 않고, 다른 격자만 보간한다."""
    from dinov2 import PRETRAIN_GRID, VARIANTS, DinoV2ViT, load_pretrained
    small = DinoV2ViT(**VARIANTS["vits14"])
    names = set(small.state_dict())
    assert {"cls_token", "pos_embed", "mask_token", "patch_embed.proj.weight", "norm.weight",
            "blocks.11.ls2.gamma", "blocks.0.attn.qkv.weight"} <= names and len(names) == 175
    torch.save(small.state_dict(), tmp_path / "s.pth")
    other = DinoV2ViT(**VARIANTS["vits14"])
    load_pretrained(other, tmp_path / "s.pth")
    assert torch.equal(other.pos_embed, small.pos_embed)
    with pytest.raises(ValueError, match="구조가 일치하지 않습니다"):
        load_pretrained(DinoV2ViT(**VARIANTS["vitb14"]), tmp_path / "s.pth")
    assert other._position(PRETRAIN_GRID, PRETRAIN_GRID, torch.float32) is other.pos_embed
    assert tuple(other._position(16, 22, torch.float32).shape) == (1, 1 + 16 * 22, 384)


def test_weights_without_a_hash_in_the_name_are_checked_against_the_pinned_sha(tmp_path, monkeypatch):
    import builtin_assets
    monkeypatch.setenv("DVS_BUILTIN_ASSETS_DIR", str(tmp_path / "bundle"))
    monkeypatch.setenv("DVS_USER_WEIGHTS_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    builtin_assets._VERIFIED.clear()
    fake = tmp_path / "dinov2_vits14_pretrain.pth"
    torch.save({"pos_embed": torch.zeros(1)}, fake)  # 올바른 zip 형식이지만 공식 파일이 아님
    with pytest.raises(RuntimeError, match="해시 불일치"):
        builtin_assets.import_asset("dinov2_vits14", fake)
    assert builtin_assets.asset_status("dinov2_vits14")[0] == "missing"
    assert builtin_assets.required_pretrained_asset("segment", "builtin_finetune", "dinov2_vits14_seg") == "dinov2_vits14"
    assert builtin_assets.required_pretrained_asset("classify", "builtin_scratch", "dinov2_vitb14_cls") is None
    assert "dl.fbaipublicfiles.com/dinov2" in builtin_assets.missing_asset_help("dinov2_vitb14")


def test_patch_trained_dinov2_runs_in_tiles_not_in_one_pass(tmp_path):
    """ViT 주의는 입력 전체를 본다: 패치 학습 모델은 큰 영역을 한 번에 넣지 않고 학습 크기의 타일로 추론한다."""
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from checkpoint import tiling_contract
    from core.inference_loading import load_inference_engine
    model = build_builtin_model("dinov2_vits14_seg", 2, 1).eval()
    checkpoint = make_builtin_checkpoint("dinov2_vits14_seg", model, num_classes=2, input_size=112, in_channels=1,
                                         class_names=["background", "crack"], tiling=tiling_contract(112))
    torch.save(checkpoint, tmp_path / "dino.pt")
    image = tmp_path / "part.png"
    Image.fromarray(np.full((150, 260), 120, np.uint8), "L").save(image)
    engine = load_inference_engine(tmp_path / "dino.pt", segment_review={"min_area": 1, "min_score": 0, "split_dirs": []})
    result = engine.infer(str(image))
    assert result.status == "ok", result.error
    assert result.details["tiling"]["run"] == "tiles" and result.details["tiling"]["tiles"] > 1
    assert sum(result.details["pixel_counts"].values()) == 150 * 260


def test_onnx_export_matches_pytorch(tmp_path):
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from export_onnx import export_checkpoint
    torch.manual_seed(0)
    model = build_builtin_model("dinov2_vits14_cls", 2, 3).eval()
    torch.save(make_builtin_checkpoint("dinov2_vits14_cls", model, num_classes=2, input_size=112, in_channels=3,
                                       class_names=["ok", "ng"]), tmp_path / "c.pt")
    export_checkpoint(tmp_path / "c.pt", tmp_path / "c.onnx", log=lambda _m: None)  # 내보내기 자체 검증 포함
    session = ort.InferenceSession(str(tmp_path / "c.onnx"), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((1, 3, 112, 112)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    assert np.abs(session.run(None, {session.get_inputs()[0].name: x})[0] - expected).max() < 1e-4


def test_training_uses_a_smaller_backbone_learning_rate(tmp_path):
    """ViT 백본 lr = 학습률 × 백본 배수 × 0.1 (기본 1e-3 × 0.1 × 0.1 = 1e-5). 로그에 남긴다."""
    from train_builtin import train_builtin
    for split in ("train", "val"):
        for cls, value in (("ok", 60), ("ng", 200)):
            folder = tmp_path / "data" / split / cls
            folder.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (56, 56), (value, value, value)).save(folder / f"{index}.png")
    logs = []
    train_builtin("dinov2_vits14_cls", tmp_path / "data", num_classes=2, input_size=56, in_channels=3, epochs=1,
                  batch_size=2, learning_rate=1e-3, output_dir=tmp_path / "run", device="cpu", pretrained=False,
                  backbone_lr_mult=.1, num_workers=0, log=logs.append)
    assert any("백본 학습률 1.0e-05" in str(line) for line in logs)
    checkpoint = torch.load(tmp_path / "run" / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_id"] == "dinov2_vits14_cls" and checkpoint["input_size"] == [56, 56]


def test_training_page_offers_dinov2_for_segmentation(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from core.project import ProjectManager
    from widgets.training_widget import TrainingWidget
    QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("seg", "segment", str(tmp_path / "p"), ["background", "crack"])
    widget = TrainingWidget()
    widget.set_project(project)
    combo = widget.model_id_combo
    index = combo.findData("dinov2_vits14_seg")
    assert index >= 0 and combo.findData("dinov2_vits14_cls") < 0  # 분할 프로젝트에는 분할 변형만
    combo.setCurrentIndex(index)
    assert widget.input_size_spin.value() == 448 and project.model.model_id == "dinov2_vits14_seg"
    assert widget.mode_combo.findData("builtin_finetune") >= 0
    widget.deleteLater()


def test_desktop_segmentation_log_reports_defect_iou_and_all_background(tmp_path):
    """mIoU는 배경을 포함해(전부 배경이어도 3클래스 0.33) 결함을 못 찾는 상태가 안 보였다.
    데스크톱 학습(class_report)만 결함 클래스별 IoU와 '전부 배경' 경고를 로그에 남긴다. 기본(웹)은 그대로."""
    from train_builtin import _class_report, train_builtin
    line, collapsed = _class_report({"iou": [.99, 0.0, 0.0], "predicted": [100, 0, 0], "truth": [90, 5, 5]},
                                    ["background", "scratch", "dot"])
    assert collapsed and "scratch 0.000 · dot 0.000" in line and "전부 배경" in line
    line, collapsed = _class_report({"iou": [.99, .5, 0.0], "predicted": [90, 6, 0], "truth": [90, 5, 0]},
                                    ["background", "scratch", "dot"])
    assert not collapsed and "scratch 0.500" in line and "dot" not in line  # 정답 없는 클래스는 빼고 평균
    for split in ("train", "val"):
        (tmp_path / "data" / "images" / split).mkdir(parents=True)
        (tmp_path / "data" / "masks" / split).mkdir(parents=True)
        for index in range(2):
            Image.new("L", (64, 64), 100).save(tmp_path / "data" / "images" / split / f"{index}.png")
            mask = np.zeros((64, 64), np.uint8)
            mask[20:30, 20:30] = 1
            Image.fromarray(mask, "L").save(tmp_path / "data" / "masks" / split / f"{index}.png")
    for report in (False, True):
        logs = []
        train_builtin("unet_resnet18", tmp_path / "data", num_classes=2, input_size=64, in_channels=1, epochs=1,
                      batch_size=2, output_dir=tmp_path / f"run{report}", device="cpu", num_workers=0,
                      class_names=["background", "crack"], class_report=report,
                      log=lambda m: logs.append(m) if isinstance(m, str) else None)
        assert any("결함 클래스 IoU" in line for line in logs) is report


def test_only_the_desktop_training_request_asks_for_the_class_report():
    import inspect
    from webapp import worker
    source = inspect.getsource(worker.train)
    assert 'class_report=bool(payload.get("class_report"))' in source  # 웹 요청에는 이 키가 없다
    qt_source = (ROOT / "gui" / "core" / "qt_training.py").read_text(encoding="utf-8")
    assert '"class_report": True' in qt_source
