"""Reduce precision only after parity and measured latency both improve.

This search runs during explicit export, never on the inference button. The
validated artifact is kept separately so rejected candidates cannot replace it.
"""
import copy
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
from torch import nn


class _MixedPrecisionExport(nn.Module):
    def __init__(self, reference, fp64_stages):
        super().__init__()
        from efficientnet_precision import prepare_precision_export
        self.model = prepare_precision_export(reference).model
        self.fp64_stages = frozenset(fp64_stages)
        for index in range(len(self.model.features)):
            if index not in self.fp64_stages:
                self.model.features[index] = copy.deepcopy(reference.features[index]).cpu().float().eval()
        self.inference_optimization = {
            "fallback": "mixed_precision", "compute_precision": "mixed_float32_float64",
            "io_precision": "float32", "fp64_stages": sorted(self.fp64_stages),
            "head_precision": "float64", "conv_bn_fused": 0,
            "constant_folding": False, "simplified": False, "latency_optimized": False,
        }

    def checkpoint_config(self):
        return self.model.checkpoint_config()

    def forward(self, images):
        from efficientnet_contract import LEGACY_GRAY_INPUT
        x = images.double() if 0 in self.fp64_stages else images.float()
        if self.model.input_adapter == LEGACY_GRAY_INPUT:
            x = (x * .226 + .449 - self.model.rgb_mean.to(x.dtype)) / self.model.rgb_std.to(x.dtype)
        for index, stage in enumerate(self.model.features):
            x = stage(x.double() if index in self.fp64_stages else x.float())
        x = self.model.avgpool(x.double()).flatten(1)
        return self.model.classifier(x).float()


def prepare_mixed_precision_export(reference, fp64_stages):
    stages = frozenset(fp64_stages)
    if not stages <= set(range(len(reference.features))):
        raise ValueError("잘못된 FP64 stage 인덱스")
    return _MixedPrecisionExport(reference, stages).eval()


def _session(path, settings):
    import onnxruntime as ort
    from onnx_session import cpu_session_options
    return ort.InferenceSession(str(path), cpu_session_options(**settings), providers=["CPUExecutionProvider"])


def _median_ms(session, sample):
    feeds = {session.get_inputs()[0].name: sample.detach().cpu().numpy()}
    for _ in range(2):
        session.run(None, feeds)
    elapsed = []
    for _ in range(7):
        started = time.perf_counter()
        session.run(None, feeds)
        elapsed.append((time.perf_counter() - started) * 1000)
    return float(np.median(elapsed))


def _measure(path, settings, sample):
    # No other ORT thread pool is kept spinning during the next measurement.
    session = _session(path, settings)
    try:
        return _median_ms(session, sample)
    finally:
        del session


def tune_precision_export(reference, model, path, sample, settings, verify, *,
                          opset, dynamic_batch, attempts, log, time_budget=30.):
    """Greedy bounded search; parity alone never authorizes a slower artifact.

    ``verify`` validates every original-reference probe against ``path``. The
    caller has already verified ``model`` with ``settings`` at that path.
    """
    from export_onnx import export_to_onnx
    best_model, best_settings = model, dict(settings)
    best_stages = set(range(len(reference.features)))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".precision-tuning-", dir=Path(path).parent) as folder:
        best_path = Path(folder) / "verified.onnx"
        shutil.copyfile(path, best_path)
        try:
            baseline_ms = _measure(best_path, best_settings, sample)
            best_ms = baseline_ms
            # Small depthwise kernels can be faster without thread fan-out.
            trials = [("threads", set())] if settings["num_threads"] != 1 else []
            # Try broad regions first so a short search can remove many costly
            # FP64 operations. Failure of a region does not skip its single stages.
            if len(reference.features) == 9:
                trials += [("stage", set(range(4))), ("stage", set(range(5, 9)))]
            trials += [("stage", {index}) for index in range(len(reference.features))]
            for kind, indices in trials:
                if time.monotonic() - started >= time_budget:
                    log("정밀도·속도 탐색 시간 한도 도달: 검증된 최선 후보 유지")
                    break
                candidate_settings = dict(best_settings)
                candidate_stages = best_stages - indices if kind == "stage" else best_stages
                if kind == "stage" and candidate_stages == best_stages:
                    continue
                candidate = (prepare_mixed_precision_export(reference, candidate_stages)
                             if kind == "stage" else best_model)
                row = {**candidate_settings, "graph": "mixed_precision" if kind == "stage" else "portable_fp64",
                       "fp64_stages": sorted(candidate_stages), "passed": False}
                try:
                    if kind == "threads":
                        shutil.copyfile(best_path, path)
                        candidate_settings["num_threads"] = 1
                        row["num_threads"] = 1
                    else:
                        export_to_onnx(candidate, sample, path, opset, dynamic_batch, constant_folding=False)
                    verify(candidate, candidate_settings)
                    row["passed"] = True
                    elapsed = _measure(path, candidate_settings, sample)
                    current_ms = _measure(best_path, best_settings, sample)
                    row.update(model_median_ms=elapsed, compared_model_median_ms=current_ms,
                               selected=elapsed < current_ms * .95)
                    if row["selected"]:
                        best_model, best_settings = candidate, candidate_settings
                        best_stages, best_ms = candidate_stages, elapsed
                        shutil.copyfile(path, best_path)
                        log(f"검증·속도 개선: FP64 stage {sorted(best_stages)}, CPU {best_settings['num_threads']} threads, "
                            f"모델 추론 {current_ms:.2f} → {elapsed:.2f} ms (현재 PC 측정)")
                except Exception as exc:
                    row["error"] = str(exc)
                finally:
                    attempts.append(row)
            best_model.inference_optimization = dict(best_model.inference_optimization)
            best_model.inference_optimization["latency_optimized"] = any(row.get("selected") for row in attempts)
            best_model.inference_optimization["latency_measurement"] = {
                "scope": "model_only", "baseline_median_ms": baseline_ms,
                "selected_median_ms": best_ms, "runs": 7,
                "selection": "verified_candidate_at_least_5_percent_faster", "host_specific": True,
            }
        except Exception as exc:
            log(f"속도 탐색 중단: 검증된 모델 유지 ({exc})")
        finally:
            shutil.copyfile(best_path, path)
    verify(best_model, best_settings)  # Reload the restored winning artifact.
    return best_model, best_settings
