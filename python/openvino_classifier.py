"""EfficientNet용 선택적 OpenVINO CPU 배포 경로. PyTorch 불필요."""
from pathlib import Path

import numpy as np

from onnx_classifier import ImageClassifier


class OpenVINOClassifier(ImageClassifier):
    """Batch 1, FP32, 단일 stream. 동시 호출에는 인스턴스를 분리해야 한다."""

    def __init__(self, config_path, *, num_threads=0):
        import openvino as ov

        super().__init__(config_path)
        if type(num_threads) is not int or num_threads < 0:
            raise ValueError("스레드 수는 0 이상의 정수 필요")
        cfg = self.config
        self.input_shape = (1, cfg["input_channels"], cfg["input_height"], cfg["input_width"])
        self.core = ov.Core()
        model = self.core.read_model(Path(config_path).parent / cfg["model_path"])
        if len(model.inputs) != 1 or len(model.outputs) != 1:
            raise ValueError("EfficientNet 입력/출력 텐서 하나 필요")
        inp, out = model.input(), model.output()
        if cfg["input_name"] not in inp.get_names() or cfg["output_name"] not in out.get_names():
            raise ValueError("ONNX와 JSON 텐서 이름 불일치")
        if inp.get_element_type() != ov.Type.f32 or out.get_element_type() != ov.Type.f32:
            raise ValueError("FP32 입력/출력 필요")
        shape = inp.get_partial_shape()
        if shape.rank.is_dynamic or shape.rank.get_length() != 4:
            raise ValueError("NCHW 입력 필요")
        if not shape.compatible(ov.PartialShape(self.input_shape)):
            raise ValueError("ONNX와 JSON 입력 크기 불일치")
        # Only the batch dimension may be dynamic in the existing export contract.
        if any(d.is_dynamic or d.get_length() != size for d, size in zip(list(shape)[1:], self.input_shape[1:])):
            raise ValueError("ONNX와 JSON 입력 채널/크기 불일치")
        model.reshape({inp: self.input_shape})
        if list(model.output().get_shape()) != [1, cfg["num_classes"]]:
            raise ValueError("ONNX 클래스 개수 불일치")
        options = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
                   "INFERENCE_PRECISION_HINT": "f32"}
        if num_threads:
            options["INFERENCE_NUM_THREADS"] = num_threads
        self.compiled_model = self.core.compile_model(model, "CPU", options)
        self.request = self.compiled_model.create_infer_request()
        self.runtime_settings = {key: str(self.compiled_model.get_property(key)) for key in options}
        self.runtime_settings["INFERENCE_NUM_THREADS"] = int(
            self.compiled_model.get_property("INFERENCE_NUM_THREADS"))
        self.runtime_settings["device"] = self.core.get_property("CPU", "FULL_DEVICE_NAME")

    def logits(self, tensor):
        tensor = np.asarray(tensor)
        if tensor.dtype != np.float32 or tensor.shape != self.input_shape:
            raise ValueError(f"float32 입력 {self.input_shape} 필요")
        self.request.infer({self.config["input_name"]: np.ascontiguousarray(tensor)}, share_inputs=True)
        # InferRequest reuses its buffer. Never expose a view that the next call mutates.
        result = self.request.get_output_tensor().data.copy()
        if result.shape != (1, self.config["num_classes"]) or not np.isfinite(result).all():
            raise ValueError("OpenVINO 출력 형태 또는 유한성 오류")
        return result
