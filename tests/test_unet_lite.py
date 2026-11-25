"""초경량 U-Net (사전학습 없음): 구조, 데스크톱 전용 목록, 처음부터 학습 전용 모드, 학습·내보내기·추론 연결."""

from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
torch = pytest.importorskip("torch")


def test_registry_lists_the_tiny_unet_on_the_desktop_only():
    from core.model_registry import ModelRegistry
    from core.training_modes import BUILTIN_ADAPTER_IDS, BUILTIN_SCRATCH_ONLY_IDS
    from builtin_models import SCRATCH_ONLY_MODELS
    registry = ModelRegistry.builtin()
    assert "unet_lite" not in {spec.model_id for spec in registry.list()}  # 웹·API 목록은 그대로
    assert "unet_lite" in {spec.model_id for spec in registry.list("segment", surface="desktop")}
    assert not registry.get("unet_lite").pretrained_asset  # 받아야 할 가중치가 없다
    assert "unet_lite" in BUILTIN_ADAPTER_IDS and BUILTIN_SCRATCH_ONLY_IDS == SCRATCH_ONLY_MODELS


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("size", [(64, 64), (75, 130)])
def test_output_keeps_the_input_size_and_a_1px_line_reaches_the_output(channels, size):
    from builtin_models import build_builtin_model
    model = build_builtin_model("unet_lite", 3, channels).eval()
    with torch.no_grad():
        y = model(torch.zeros(1, channels, *size))
        assert y.shape == (1, 3, *size)
        line = torch.zeros(1, channels, *size)
        line[..., size[0] // 2, :] = 1.0  # 1px 가로선: 첫 stride-2 합성곱이 모든 픽셀을 본다
        assert not torch.equal(model(line), y)
    assert sum(p.numel() for p in model.parameters()) < 400_000  # ResNet18 U-Net의 약 1/40


def test_pretrained_start_is_refused_with_a_clear_message():
    from builtin_models import build_builtin_model
    from core.training_modes import training_capabilities
    with pytest.raises(ValueError, match="처음부터 학습"):
        build_builtin_model("unet_lite", 2, 3, pretrained=True)
    with pytest.raises(ValueError, match="처음부터 학습"):
        training_capabilities("segment", "builtin_finetune", model_id="unet_lite")
    assert training_capabilities("segment", "builtin_scratch", model_id="unet_lite")["engine"] == "builtin"


def _segmentation_data(root, size=(48, 96)):
    for split in ("train", "val"):
        for kind in ("images", "masks"):
            (root / kind / split).mkdir(parents=True)
        for index in range(2):
            image = np.full((*size, 3), 90, np.uint8)
            mask = np.zeros(size, np.uint8)
            image[10 + index * 5, 8:80] = 200  # 1px 밝은 선 = crack
            mask[9 + index * 5:12 + index * 5, 8:80] = 1
            Image.fromarray(image).save(root / "images" / split / f"{index}.png")
            Image.fromarray(mask).save(root / "masks" / split / f"{index}.png")


def test_train_export_and_infer_end_to_end(tmp_path):
    """처음부터 학습 → 체크포인트 → ONNX(PyTorch와 같은 값) → 데스크톱 추론 엔진이 원본 크기 그대로 분할."""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort
    from train_builtin import train_builtin
    from export_onnx import export_checkpoint
    from core.inference_loading import load_inference_engine
    from builtin_models import load_builtin_checkpoint
    _segmentation_data(tmp_path / "data")
    best = train_builtin("unet_lite", tmp_path / "data", num_classes=2, input_size=48, in_channels=3, epochs=1,
                         batch_size=2, learning_rate=1e-3, output_dir=tmp_path / "run", device="cpu",
                         pretrained=False, class_names=["background", "crack"], num_workers=0, log=lambda _m: None)
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert checkpoint["model_id"] == "unet_lite" and checkpoint["weight_provenance"] == {"source": "random_init"}
    export_checkpoint(best, tmp_path / "m.onnx", log=lambda _m: None)
    session = ort.InferenceSession(str(tmp_path / "m.onnx"), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    x = np.random.default_rng(0).standard_normal((1, 3, shape[2], shape[3])).astype(np.float32)
    with torch.no_grad():
        expected = load_builtin_checkpoint(checkpoint)(torch.from_numpy(x)).numpy()
    assert np.abs(session.run(None, {session.get_inputs()[0].name: x})[0] - expected).max() < 1e-4
    engine = load_inference_engine(best, segment_review={"min_area": 1, "min_score": 0, "split_dirs": []})
    result = engine.infer(str(tmp_path / "data" / "images" / "val" / "0.png"))
    assert result.status == "ok", result.error
    assert sum(result.details["pixel_counts"].values()) == 48 * 96


def test_training_page_offers_only_scratch_and_local_start(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from core.project import ProjectManager
    from widgets.training_widget import TrainingWidget
    QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("seg", "segment", str(tmp_path / "p"), ["background", "crack"])
    widget = TrainingWidget()
    widget.set_project(project)
    combo = widget.model_id_combo
    combo.setCurrentIndex(combo.findData("unet_lite"))
    modes = [widget.mode_combo.itemData(i) for i in range(widget.mode_combo.count())]
    assert modes == ["builtin_scratch", "builtin_transfer"]
    assert widget.mode_combo.currentData() == "builtin_scratch" and "사전학습" in widget.mode_desc.text()
    widget.deleteLater()
