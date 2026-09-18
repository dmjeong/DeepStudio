"""Latency acceptance decisions and checkpoint contracts."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("efficientnet_benchmark", Path(__file__).parents[1] / "tools/efficientnet_benchmark.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_target_uses_pipeline_p95_not_fast_model_or_median():
    rows = [
        {"runtime": "fast_model", "scope": "model", "threads": 4, "p50_ms": 2, "p95_ms": 3, "max_ms": 4},
        {"runtime": "jittery", "scope": "pipeline", "threads": 4, "p50_ms": 4, "p95_ms": 9, "max_ms": 15},
        {"runtime": "steady", "scope": "pipeline", "threads": 2, "p50_ms": 6, "p95_ms": 7, "max_ms": 10},
    ]
    result = bench.target_verdict(rows, "pipeline", 8)
    assert result["runtime"] == "steady"
    assert result["met_on_this_machine"] is True
    assert result["all_measured_requests_within_target"] is False
    assert bench.target_verdict(rows[:2], "pipeline", 8)["met_on_this_machine"] is False


def test_measure_excludes_warmup_and_keeps_raw_samples(monkeypatch):
    calls = []
    times = iter([0, 1_000_000, 10_000_000, 19_000_000])
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: next(times))
    result = bench.measure(lambda: calls.append(1), 3, 2, 8)
    assert len(calls) == 5
    assert result["samples_ms"] == [1.0, 9.0]
    assert result["within_target_fraction"] == .5
    assert result["p95_ms"] > 8


def test_postprocessing_preserves_logit_ranking_when_softmax_rounds_to_tie():
    import numpy as np
    from onnx_classifier import ImageClassifier
    model = object.__new__(ImageClassifier)
    model.config = {"class_names": ["first", "second"]}
    model.preprocess = lambda x: x
    model.logits = lambda x: np.array([[-1e-9, 1e-9]], dtype=np.float32)
    result = model.predict_rgb(np.zeros((1, 1), dtype=np.uint8))
    assert result["probabilities"] == [.5, .5]
    assert result["class_id"] == 1
    assert result["class_name"] == "second"


@pytest.mark.parametrize("extra", [["--runs", "0"], ["--warmup", "0"], ["--threads", "0"],
                                  ["--target-ms", "nan"], ["--target-ms", "inf"], ["--input-size", "0"],
                                  ["--num-classes", "0"], ["--validation", "somewhere"]])
def test_invalid_arguments(extra):
    with pytest.raises(SystemExit):
        bench.parse_args(["--architecture", "efficientnet_b0", *extra])


def test_synthetic_checkpoint_is_explicit_and_cannot_override_stored_size(tmp_path):
    args = bench.parse_args(["--architecture", "efficientnet_b0", "--output", str(tmp_path)])
    path, checkpoint, origin = bench.checkpoint_source(args)
    assert checkpoint["input_size"] == 224
    assert checkpoint["in_channels"] == 1
    assert origin["kind"] == "random_weights"
    assert origin["accuracy_evaluated"] is False
    args = bench.parse_args(["--checkpoint", str(path), "--input-size", "240"])
    with pytest.raises(ValueError, match="설정 불일치"):
        bench.checkpoint_source(args)


def test_cli_runs_without_pretrained_downloads_and_saves_failed_target(tmp_path):
    import json
    import subprocess
    import sys
    command = [sys.executable, str(Path(__file__).parents[1] / "tools/efficientnet_benchmark.py"),
               "--architecture", "efficientnet_b0", "--input-size", "32", "--threads", "1",
               "--warmup", "1", "--runs", "1", "--target-ms", "0.000001", "--require-target",
               "--output", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 2, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["weights"]["kind"] == "random_weights"
    assert report["accuracy"] is None
    assert report["target"]["met_on_this_machine"] is False
    assert len(report["measurements"]) == 8
    assert report["input_size"] == [32, 32]
    # Exercise real-image validation routing using explicitly synthetic test fixtures.
    import numpy as np
    from PIL import Image
    for name in ("0", "1"):
        folder = tmp_path / "validation" / name
        folder.mkdir(parents=True)
        Image.fromarray(np.full((40, 40), int(name) * 255, np.uint8)).save(folder / "sample.png")
    command = [sys.executable, str(Path(__file__).parents[1] / "tools/efficientnet_benchmark.py"),
               "--checkpoint", str(tmp_path / "synthetic-checkpoint.pt"), "--input-size", "32",
               "--validation", str(tmp_path / "validation"), "--threads", "1", "--warmup", "1",
               "--runs", "1", "--output", str(tmp_path / "validated")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "validated/report.json").read_text())
    assert report["accuracy"]["onnx/1"]["samples"] == 2
    assert report["accuracy"]["onnx/1"]["reference_top1_agreement"] == 1
    assert report["synthetic_input"] is False
    assert len(report["weights"]["sha256"]) == 64
