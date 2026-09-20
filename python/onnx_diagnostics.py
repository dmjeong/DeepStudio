"""Local numerical diagnostics for failed EfficientNet exports.

Only aggregate statistics are returned. Temporary instrumented graphs are never
published. Diagnostic comparisons cannot approve an export or modify weights.
"""

import copy
from collections import Counter
from pathlib import Path
import tempfile

import numpy as np
import torch


def comparison(reference, actual, *, atol=1e-3, rtol=5e-4, classification=False):
    reference, actual = np.asarray(reference), np.asarray(actual)
    result = {"passed": False, "shape_matches": reference.shape == actual.shape}
    if not result["shape_matches"]:
        return result
    result["finite"] = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    if not result["finite"]:
        return result
    ref, out = reference.astype(np.float64), actual.astype(np.float64)
    difference = np.abs(ref - out)
    limit = atol + rtol * np.abs(out)
    ratio = difference / np.maximum(limit, np.finfo(np.float64).tiny)
    result.update(max_abs_error=float(difference.max(initial=0)),
                  max_tolerance_ratio=float(ratio.max(initial=0)),
                  reference_abs_max=float(np.abs(ref).max(initial=0)),
                  actual_abs_max=float(np.abs(out).max(initial=0)),
                  failed_elements=int(np.count_nonzero(difference > limit)))
    result["passed"] = result["failed_elements"] == 0
    if classification:
        result["top1_equal"] = bool(ref.ndim == 2 and
                                   np.array_equal(ref.argmax(1), out.argmax(1)))
        result["passed"] = result["passed"] and result["top1_equal"]
    return result


def state_summary(model):
    state = model.state_dict()
    nonfinite = [name for name, value in state.items()
                 if value.is_floating_point() and not torch.isfinite(value).all().item()]
    batch_norm = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d) and module.running_var is not None:
            values = module.running_var.detach().cpu()
            minimum = float(values.min()) if torch.isfinite(values).all().item() else None
            batch_norm.append({"layer": name, "min_running_variance": minimum,
                               "negative_variances": int((values < 0).sum()),
                               "epsilon": module.eps})
    return {"dtypes": dict(Counter(str(value.dtype) for value in state.values())),
            "nonfinite_tensors": nonfinite, "batch_norm": batch_norm}


def conclusion(report):
    state = report.get("state", {})
    if state.get("nonfinite_tensors") or any(row["negative_variances"] for row in state.get("batch_norm", [])):
        return "invalid_model_state"
    repeated = report.get("pytorch_repeat", {})
    if repeated.get("finite") is False:
        return "pytorch_nonfinite_output"
    if repeated and not repeated.get("passed"):
        return "pytorch_repeat_difference"
    sessions = report.get("onnx_sessions", {})
    baseline = sessions.get("all_default", sessions.get("all", {}))
    if baseline.get("passed"):
        return "not_reproduced_in_diagnostic"
    if baseline.get("passed") is False and sessions.get("all", {}).get("passed"):
        return "ort_thread_difference"
    if sessions.get("all", {}).get("passed") is False and sessions.get("disabled", {}).get("passed"):
        return "ort_optimization_difference"
    precision = report.get("pytorch_fp64_vs_fp32", {})
    if "passed" in precision and not precision["passed"]:
        return "pytorch_precision_sensitive"
    return "unresolved_graph_or_runtime_difference"


EXPLANATIONS = {
    "invalid_model_state": "가중치·정규화 통계에서 비정상 값 확인. 정상 체크포인트와 비교 필요",
    "pytorch_nonfinite_output": "PyTorch 출력에 NaN 또는 무한대 확인. 모델 상태와 입력의 수치 범위 조사 필요",
    "pytorch_repeat_difference": "동일 입력의 PyTorch 반복 실행 결과가 달라짐",
    "not_reproduced_in_diagnostic": "진단 재실행에서는 원본 그래프 오차가 재현되지 않음",
    "ort_thread_difference": "ONNX Runtime 스레드 수를 제한하면 통과함. 실행 스레드에 따른 차이 확인",
    "ort_optimization_difference": "ONNX Runtime 최적화를 끄면 통과함. 실행 최적화에 따른 차이 확인",
    "pytorch_precision_sensitive": "PyTorch 자체도 FP32·FP64 출력이 다름. 수치 민감도 확인, 재학습 필요 여부는 미확정",
    "unresolved_graph_or_runtime_difference": "그래프 변환 또는 런타임 연산 차이 추가 조사 필요",
}


class _StageOutputs(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        values, handles = [], []
        try:
            for stage in self.model.features:
                handles.append(stage.register_forward_hook(lambda _m, _a, out: values.append(out)))
            logits = self.model(images)
        finally:
            for handle in handles:
                handle.remove()
        return (logits, *values)


def diagnose_efficientnet(model, onnx_path, probe, *, probe_name, opset=17):
    """Compare a failing probe without changing deployment acceptance settings."""
    import onnxruntime as ort
    from export_onnx import export_to_onnx

    report = {"schema_version": 1, "probe": probe_name, "input_shape": list(probe.shape),
              "state": state_summary(model), "onnx_sessions": {},
              "requires_retraining": "not_determined"}
    reference_model = copy.deepcopy(model).cpu().float().eval()
    sample = probe.detach().cpu().float().contiguous()
    threads = max(1, min(torch.get_num_threads(), 4))
    report["threads"] = {"pytorch": torch.get_num_threads(), "onnx_default": 0,
                         "onnx_controlled": threads}
    report["mkldnn_available"] = torch.backends.mkldnn.is_available()
    report["mkldnn_enabled"] = torch.backends.mkldnn.enabled
    try:
        report["cpu_autocast_enabled_before_diagnostic"] = torch.is_autocast_enabled("cpu")
    except TypeError:  # PyTorch versions before the device_type argument.
        report["cpu_autocast_enabled_before_diagnostic"] = torch.is_autocast_cpu_enabled()
    report["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    try:
        report["mkldnn_conv_fp32_precision"] = torch.backends.mkldnn.conv.fp32_precision
    except (AttributeError, RuntimeError):
        report["mkldnn_conv_fp32_precision"] = "unavailable"
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        expected = reference_model(sample).numpy().copy()
        report["pytorch_repeat"] = comparison(expected, reference_model(sample).numpy(), classification=True)
        try:
            double = copy.deepcopy(reference_model).double()(sample.double()).numpy()
            report["pytorch_fp64_vs_fp32"] = comparison(double, expected, classification=True)
        except Exception as exc:
            report["fp64_error"] = str(exc)

    disabled_output = None
    for name, level, session_threads in (
            ("all_default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 0),
            ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, threads),
            ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, threads),
            ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, threads)):
        options = ort.SessionOptions()
        options.intra_op_num_threads = session_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = level
        try:
            session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
            actual = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
            report["onnx_sessions"][name] = comparison(expected, actual, classification=True)
            if name == "disabled":
                disabled_output = actual
            del session
        except Exception as exc:
            report["onnx_sessions"][name] = {"error": str(exc)}

    # Exposing intermediate outputs can inhibit fusion, so report the traced
    # final output separately before interpreting the first differing stage.
    try:
        with tempfile.TemporaryDirectory(prefix=".onnx-diagnostic-", dir=Path(onnx_path).parent) as temp:
            wrapper = _StageOutputs(reference_model).eval()
            labels = [f"features.{index}" for index in range(len(reference_model.features))]
            names = ["logits", *(name.replace(".", "_") for name in labels)]
            with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
                expected_stages = [out.numpy().copy() for out in wrapper(sample)]
            traced = Path(temp) / "stages.onnx"
            with torch.autocast(device_type="cpu", enabled=False):
                export_to_onnx(wrapper, sample, traced, opset, output_names=names, constant_folding=False)
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(str(traced), options, providers=["CPUExecutionProvider"])
            actual_stages = session.run(names, {session.get_inputs()[0].name: sample.numpy()})
            del session
            report["instrumented_final"] = comparison(expected_stages[0], actual_stages[0], classification=True)
            if disabled_output is not None:
                report["instrumentation_vs_original_onnx"] = comparison(disabled_output, actual_stages[0], classification=True)
            report["stages"] = [{"layer": label, **comparison(ref, out)}
                                for label, ref, out in zip(labels, expected_stages[1:], actual_stages[1:])]
            report["first_divergent_stage"] = next((row["layer"] for row in report["stages"] if not row["passed"]), None)
    except Exception as exc:
        report["stage_diagnostic_error"] = str(exc)
    report["finding"] = conclusion(report)
    report["summary"] = EXPLANATIONS[report["finding"]]
    return report
