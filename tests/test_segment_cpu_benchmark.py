"""분할 모델 CPU 속도 측정 도구 (tools/segment_cpu_benchmark.py).

    무작위 가중치 모델 ─▶ 크기와 무관한 FP32/INT8 ONNX ─▶ 배율 · 슬롯 크기별 시간
        · 모델 입력 크기가 C++ 계약(32의 배수로 채움)과 같다
        · half 출력(1/2 해상도 로짓)을 확대하면 full 출력과 같다 = 같은 모델의 다른 출력 방식이다
        · 한 번 만든 ONNX는 torch 없이 다시 쓴다 (설비 PC로 폴더만 복사)
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("segment_cpu_benchmark",
                                              Path(__file__).parents[1] / "tools/segment_cpu_benchmark.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def test_model_input_is_padded_like_the_whole_image_contract():
    assert tool.model_input(3100, 1500) == (1504, 3104)            # (세로, 가로), 32의 배수로 올림
    assert tool.model_input(3100, 1500, 0.5) == (768, 1568)        # 1550×750 으로 줄인 뒤 올림
    assert tool.model_input(192, 160) == (160, 192)                # 이미 배수면 그대로
    assert tool.parse_size("3100x1500") == (3100, 1500)


def test_candidates_cover_the_unets_and_the_realtime_models():
    # U-Net 계열은 1/2 해상도 출력(half)이 있고, 1/8 해상도에서 판정하는 실시간 분할망에는 없다
    assert set(tool.MODELS) == {"unet_resnet18", "unet_mobilenetv2", "unet_lite", "unet_mobilenetv4_small",
                                "unet_shufflenetv2", "unet_efficientnet_b1", "bisenetv2", "stdc1"}
    assert set(tool.MODELS) - set(tool.HALF_OUTPUT) == {"bisenetv2", "stdc1"}


def test_realtime_models_reject_the_half_output():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="half"):
        tool.build_export_model("bisenetv2", "half")


@pytest.mark.parametrize("model", sorted(tool.HALF_OUTPUT))
def test_half_output_upsampled_equals_full_output(model):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    if model in ("unet_mobilenetv4_small", "unet_efficientnet_b1"):
        pytest.importorskip("timm")
    x = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        full = tool.build_export_model(model, "full")(x)
        half = tool.build_export_model(model, "half")(x)
        upsampled = torch.nn.functional.interpolate(half, size=x.shape[-2:], mode="bilinear", align_corners=False)
    assert half.shape == (1, 3, 32, 48)
    # ResNet18 U-Net은 확대 뒤 head 를 적용한다: 1×1 합성곱과 선형 확대의 순서만 다르므로 반올림 오차만 남는다
    assert torch.allclose(upsampled, full, atol=1e-4)


def test_benchmark_writes_records_and_reuses_models(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("cv2")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxruntime.quantization")
    work, output = tmp_path / "models", tmp_path / "result.json"
    arguments = ["--region", "96x64", "--models", "unet_lite", "--threads", "1", "--runs", "1",
                 "--slot-size", "40x40", "--slots", "3", "--runtimes", "onnxruntime",
                 "--work", str(work), "--output", str(output)]
    assert tool.main(arguments) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["accuracy_evaluated"] is False
    records = report["records"]
    assert len(records) == 2 * 3 * 2                                # 출력 방식 2 × 조건 3 × 정밀도 2
    assert {(r["case"], tuple(r["input"]), r["count"]) for r in records} == {
        ("1x", (96, 64), 1), ("0.5x", (64, 32), 1), ("slots", (64, 64), 3)}
    assert all(r["median_ms"] > 0 for r in records)

    # 합성곱 편향이 모두 초기값이어야 양자화가 편향까지 정수로 바꾼다 (Identity 로 묶이면 INT8이 느리게 나온다)
    onnx = pytest.importorskip("onnx")
    graph = onnx.load(str(tool.model_path(work, "unet_lite", "full", "fp32"))).graph
    initializers = {value.name for value in graph.initializer}
    convolutions = [node for node in graph.node if node.op_type == "Conv"]
    assert convolutions and all(node.input[2] in initializers for node in convolutions)

    # 같은 파일이 다른 입력 크기에서도 돈다 (업샘플이 배율 방식) · half 는 1/2 해상도
    feed = {tool.INPUT_NAME: np.zeros((1, 3, 96, 160), np.float32)}
    for head, shape in (("full", (1, 3, 96, 160)), ("half", (1, 3, 48, 80))):
        for precision in ("fp32", "int8"):
            session = ort.InferenceSession(str(tool.model_path(work, "unet_lite", head, precision)),
                                           providers=["CPUExecutionProvider"])
            assert session.run(None, feed)[0].shape == shape

    # 두 번째 실행은 내보내기를 부르지 않는다 = torch 없는 PC에서도 측정된다
    def fail(*args, **kwargs):
        raise AssertionError("이미 있는 ONNX를 다시 만들면 안 됩니다")

    monkeypatch.setattr(tool, "export_fp32", fail)
    monkeypatch.setattr(tool, "export_int8", fail)
    assert tool.main(arguments) == 0


@pytest.mark.parametrize("model,head", [("unet_shufflenetv2", "full"), ("unet_shufflenetv2", "half"),
                                         ("bisenetv2", "full"), ("stdc1", "full")])
def test_new_models_export_a_graph_that_runs_at_other_sizes(tmp_path, model, head):
    """새 모델도 파일 하나로 배율 · 슬롯 크기를 바꿔 잰다. ShuffleNetV2의 채널 섞기가 크기를 굳히면 여기서 실패한다."""
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("cv2")
    ort = pytest.importorskip("onnxruntime")
    path = tmp_path / f"{model}_{head}.onnx"
    tool.export_fp32(model, head, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(None, {tool.INPUT_NAME: np.zeros((1, 3, 128, 192), np.float32)})[0]
    assert output.shape == ((1, 3, 128, 192) if head == "full" else (1, 3, 64, 96))
