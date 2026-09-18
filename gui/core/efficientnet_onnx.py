"""Prepare the loaded CPU EfficientNet for the actual Studio inference button."""
from pathlib import Path
import tempfile
import time

import numpy as np
import torch


class EfficientNetOnnx:
    def __init__(self, model, input_size, *, threads=4, warmup=10):
        started = time.perf_counter()
        # Decoder module loading belongs to model setup, not the first image's decode timing.
        import cv2  # noqa: F401
        from efficientnet import EfficientNet, prepare_for_inference
        from export_onnx import (export_to_onnx, validate_classification_outputs,
                                 verification_tolerances)
        import onnxruntime as ort
        if not isinstance(model, EfficientNet) or next(model.parameters()).device.type != "cpu":
            raise ValueError("CPU EfficientNet 모델이 필요합니다")
        if type(threads) is not int or threads < 1:
            raise ValueError("ONNX 스레드 수는 양의 정수여야 합니다")
        self.model_id = id(model)
        self.input_shape = (1, model.in_channels, *input_size)
        self.threads = threads
        self.version = ort.__version__
        self.verification_tolerance = verification_tolerances("classify")
        # Export and validate the same FP32 graph that is deployed.  Conv/BN
        # fusion is numerically equivalent, but comparing its reassociated
        # logits with the unfused training graph rejects valid ONNX models at
        # near-zero values (for example the reported 0.00052 delta).
        export_model = prepare_for_inference(model)
        self.export_optimization = dict(export_model.inference_optimization)
        generator = torch.Generator().manual_seed(42)
        dummy = torch.randn(self.input_shape, generator=generator)
        self.num_classes = model.num_classes
        # Export the loaded model, never a checkpoint that may have changed on disk.
        with tempfile.TemporaryDirectory(prefix="studio-efficientnet-") as folder:
            path = Path(folder) / "model.onnx"
            export_to_onnx(export_model, dummy, path, dynamic_batch=False, task="classify")
            model_bytes = path.read_bytes()
        probes = []
        with torch.inference_mode():
            for name, tensor in (("seeded", dummy), ("zero", torch.zeros_like(dummy))):
                expected = export_model(tensor).detach().numpy().copy()
                # A non-finite reference cannot be repaired by a runtime option.
                if not np.isfinite(expected).all():
                    raise ValueError(f"ONNX 검증 실패: 기준 PyTorch 출력에 NaN 또는 무한대 "
                                     f"{np.count_nonzero(~np.isfinite(expected))}개 (입력 {name}); "
                                     "ONNX 출력 비교 전 중단")
                probes.append((name, tensor.numpy(), expected))
        self.validation_attempts = []
        self.session = None
        levels = (("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                  ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                  ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL))
        for name, level in levels:
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = level
            self.session = ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            try:
                for probe_name, tensor, expected in probes:
                    actual = self.session.run([self.output_name], {self.input_name: tensor})[0]
                    validate_classification_outputs(expected, actual, **self.verification_tolerance)
            except ValueError as exc:
                self.validation_attempts.append({"optimization": name, "passed": False,
                                                 "probe": probe_name, "error": str(exc)})
                # Release this pool before trying another one; no concurrent idle pools.
                self.session = None
            else:
                self.optimization_level = name
                self.validation_attempts.append({"optimization": name, "passed": True})
                break
        if self.session is None:
            failures = "\n".join(f"{item['optimization']}/{item['probe']}: {item['error']}"
                                 for item in self.validation_attempts)
            raise ValueError(f"ONNX 검증 실패: 모든 최적화 설정의 출력 비교 실패 "
                             f"(PyTorch {torch.__version__}, ONNX Runtime {self.version}, "
                             f"입력 {self.input_shape}, CPU {threads} threads).\n{failures}")
        for _ in range(warmup):
            self.logits(dummy.numpy())
        self.setup_sec = time.perf_counter() - started

    def matches(self, model, input_size, threads):
        return (self.model_id == id(model) and self.input_shape == (1, model.in_channels, *input_size)
                and self.threads == threads)

    def logits(self, tensor):
        if tensor.dtype != np.float32 or tensor.shape != self.input_shape:
            raise ValueError(f"ONNX 입력은 float32 {self.input_shape}이어야 합니다")
        output = self.session.run([self.output_name], {self.input_name: np.ascontiguousarray(tensor)})[0]
        if output.shape != (1, self.num_classes) or not np.isfinite(output).all():
            raise ValueError("ONNX 출력 형태/유한성 오류")
        return output
