"""Prepare the loaded CPU EfficientNet for the actual Studio inference button."""
import copy
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
        from efficientnet import EfficientNet, prepare_for_inference, prepare_native_bn_export
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
        generator = torch.Generator().manual_seed(42)
        dummy = torch.randn(self.input_shape, generator=generator)
        self.num_classes = model.num_classes
        self.validation_attempts = []
        self.session = None
        levels = (("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                  ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                  ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL))
        reference = copy.deepcopy(model).cpu().float().eval()
        # Fix the reference before any export transformation, including fusion.
        probes = []
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            for probe_name, tensor in (("seeded", dummy), ("zero", torch.zeros_like(dummy))):
                expected = reference(tensor).detach().numpy().copy()
                if not np.isfinite(expected).all():
                    raise ValueError(f"ONNX 검증 실패: 기준 PyTorch 출력에 NaN 또는 무한대 "
                                     f"(입력 {probe_name}); ONNX 출력 비교 전 중단")
                probes.append((probe_name, tensor.numpy(), expected))
        candidates = (("fused", lambda: prepare_for_inference(reference)),
                      ("unfused", lambda: copy.deepcopy(reference)),
                      ("native_batch_norm_affine", lambda: prepare_native_bn_export(reference)))
        for graph_name, factory in candidates:
            export_model = factory()
            optimization = getattr(export_model, "inference_optimization", {
                "conv_bn_fused": 0, "channels_last": False,
            })
            # Export the loaded in-memory model, never a checkpoint that may
            # have changed on disk. If fusion/export interaction is defective,
            # retry the original evaluation graph before falling back to PyTorch.
            with tempfile.TemporaryDirectory(prefix="studio-efficientnet-") as folder:
                path = Path(folder) / "model.onnx"
                export_to_onnx(export_model, dummy, path, dynamic_batch=False, task="classify",
                               constant_folding=graph_name == "fused")
                model_bytes = path.read_bytes()
            profiles = [(name, level, threads) for name, level in levels]
            if graph_name == "native_batch_norm_affine":
                profiles = [("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, count)
                            for count in dict.fromkeys((threads, 1))]
            for name, level, count in profiles:
                options = ort.SessionOptions()
                options.intra_op_num_threads = count
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
                    self.validation_attempts.append({"graph": graph_name, "optimization": name, "num_threads": count,
                                                     "passed": False, "probe": probe_name,
                                                     "error": str(exc)})
                    self.session = None
                else:
                    self.export_graph = graph_name
                    self.export_optimization = dict(optimization)
                    self.optimization_level = name
                    self.runtime_threads = count
                    self.validation_attempts.append({"graph": graph_name,
                                                     "optimization": name, "num_threads": count, "passed": True})
                    break
            if self.session is not None:
                break
        if self.session is None:
            failures = "\n".join(f"{item['graph']}/{item['optimization']}/{item['probe']}: {item['error']}"
                                 for item in self.validation_attempts)
            raise ValueError(f"ONNX 검증 실패: fused/unfused/native-BN 모든 최적화 설정의 출력 비교 실패 "
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
