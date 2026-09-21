"""Precision search keeps only accurate, faster artifacts and matching settings."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import efficientnet_precision_tuning as tuning
import export_onnx


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def search(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"anchor")
    reference = SimpleNamespace(features=[None] * 3)
    anchor = SimpleNamespace(label="anchor", inference_optimization={
        "fallback": "portable_fp64", "compute_precision": "float64"})
    attempts, logs = [], []

    def factory(_reference, stages):
        assert _reference is reference
        label = ",".join(str(index) for index in sorted(stages))
        return SimpleNamespace(label=label, inference_optimization={
            "fallback": "mixed_precision", "fp64_stages": sorted(stages)})

    def export(candidate, _sample, destination, *_args, **_kwargs):
        Path(destination).write_text(candidate.label)

    monkeypatch.setattr(tuning, "prepare_mixed_precision_export", factory)
    monkeypatch.setattr(export_onnx, "export_to_onnx", export)

    def run(verify, *, threads=1, time_budget=30.):
        settings = {"graph_optimization_level": "disabled", "num_threads": threads}
        result = tuning.tune_precision_export(
            reference, anchor, path, torch.zeros(1, 1, 2, 2), settings, verify,
            opset=17, dynamic_batch=False, attempts=attempts, log=logs.append,
            time_budget=time_budget)
        assert settings == {"graph_optimization_level": "disabled", "num_threads": threads}
        return result

    return SimpleNamespace(path=path, reference=reference, anchor=anchor, attempts=attempts,
                           logs=logs, run=run, factory=factory)


def test_slower_valid_and_faster_invalid_candidates_cannot_replace_winner(search, monkeypatch):
    measured = []
    times = {"anchor": 100., "1,2": 110., "0,2": 1., "0,1": 50.}

    def measure(path, _settings, _sample):
        label = Path(path).read_text()
        measured.append(label)
        return times[label]

    def verify(candidate, settings):
        assert search.path.read_text() == candidate.label
        assert settings["num_threads"] == 1
        if candidate.label == "0,2":
            raise ValueError("original checkpoint comparison failed")

    monkeypatch.setattr(tuning, "_measure", measure, raising=False)
    winner, settings = search.run(verify)
    assert winner.label == search.path.read_text() == "0,1"
    assert settings == {"graph_optimization_level": "disabled", "num_threads": 1}
    assert "0,2" not in measured, "Incorrect candidates must never qualify on speed"
    assert search.attempts[0]["passed"] and not search.attempts[0]["selected"]
    assert not search.attempts[1]["passed"]
    assert search.attempts[2]["passed"] and search.attempts[2]["selected"]


def test_selected_thread_profile_belongs_to_preserved_artifact_after_later_failure(search, monkeypatch):
    def measure(path, settings, _sample):
        label = Path(path).read_text()
        if label == "anchor":
            return 100. if settings["num_threads"] == 4 else 50.
        return 30.

    def verify(candidate, settings):
        assert search.path.read_text() == candidate.label
        assert settings["num_threads"] == 1

    def factory(reference, stages):
        if stages == {2}:
            # A factory error is outside the per-candidate export/verify block.
            search.path.write_bytes(b"partial-invalid-artifact")
            raise RuntimeError("candidate construction interrupted")
        return search.factory(reference, stages)

    monkeypatch.setattr(tuning, "_measure", measure, raising=False)
    monkeypatch.setattr(tuning, "prepare_mixed_precision_export", factory)
    winner, settings = search.run(verify, threads=4)
    assert winner.label == search.path.read_text() == "1,2"
    assert settings == {"graph_optimization_level": "disabled", "num_threads": 1}
    assert search.attempts[0]["selected"] and search.attempts[0]["num_threads"] == 1
    assert search.attempts[1]["selected"]
    assert any("candidate construction interrupted" in message for message in search.logs)


def test_time_budget_preserves_last_verified_winner(search, monkeypatch):
    clock = iter((0., 0., 31.))
    monkeypatch.setattr(tuning.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(tuning, "_measure", lambda path, _settings, _sample:
                        100. if Path(path).read_text() == "anchor" else 20., raising=False)

    def verify(candidate, _settings):
        assert search.path.read_text() == candidate.label

    winner, settings = search.run(verify)
    assert winner.label == search.path.read_text() == "1,2"
    assert settings["num_threads"] == 1
    assert len(search.attempts) == 1
    assert any("시간 한도" in message for message in search.logs)


def test_failed_baseline_measurement_keeps_original_verified_artifact(search, monkeypatch):
    def fail_measure(*_args):
        raise RuntimeError("measurement unavailable")

    monkeypatch.setattr(tuning, "_measure", fail_measure, raising=False)
    verified = []

    def verify(candidate, _settings):
        assert candidate is search.anchor, "Only the preserved artifact may be revalidated"
        assert search.path.read_bytes() == b"anchor"
        verified.append(candidate)

    winner, settings = search.run(verify)
    assert winner is search.anchor
    assert search.path.read_bytes() == b"anchor"
    assert settings["num_threads"] == 1
    assert search.attempts == []
    assert verified == [search.anchor]
    assert any("measurement unavailable" in message for message in search.logs)


def test_marginal_timing_change_does_not_replace_verified_anchor(search, monkeypatch):
    monkeypatch.setattr(tuning, "_measure", lambda path, _settings, _sample:
                        100. if Path(path).read_text() == "anchor" else 96., raising=False)

    def verify(candidate, _settings):
        assert search.path.read_text() == candidate.label

    winner, _settings = search.run(verify)
    assert winner is search.anchor
    assert search.path.read_bytes() == b"anchor"
    assert all(row["passed"] and not row["selected"] for row in search.attempts)
    assert not winner.inference_optimization["latency_optimized"]


@pytest.mark.parametrize("input_adapter", ["native", "rgb_repeat_inside_model"])
def test_mixed_b1_100px_reload_preserves_original_and_dynamic_batch(tmp_path, input_adapter):
    """Real ORT validates both grayscale contracts across FP32/FP64 boundaries."""
    from efficientnet import EfficientNet

    torch.manual_seed(42)
    reference = EfficientNet("efficientnet_b1", 4, 1, input_adapter=input_adapter).eval()
    with torch.no_grad():
        for layer in reference.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.running_mean.uniform_(-.1, .1)
                layer.running_var.uniform_(.7, 1.3)
                layer.bias.uniform_(-.1, .1)
        reference.features[4][0].block[1][1].running_var[0] = 4.328019258537097e-6
    before = {name: value.clone() for name, value in reference.state_dict().items()}
    candidate = tuning.prepare_mixed_precision_export(reference, {4, 5})
    assert candidate.checkpoint_config() == reference.checkpoint_config()
    assert candidate.inference_optimization["fp64_stages"] == [4, 5]
    assert next(candidate.model.features[0].parameters()).dtype == torch.float32
    assert next(candidate.model.features[4].parameters()).dtype == torch.float64
    path = tmp_path / "mixed.onnx"
    sample = torch.randn(1, 1, 100, 100)
    export_onnx.export_to_onnx(candidate, sample, path, dynamic_batch=True, constant_folding=False)
    settings = {"graph_optimization_level": "disabled", "num_threads": 2}
    for probe in (sample, torch.zeros_like(sample), torch.randn(2, 1, 100, 100)):
        # verify_onnx reloads the actual artifact with the deployment settings.
        assert export_onnx.verify_onnx(path, probe, candidate, reference_model=reference,
                                      runtime_settings=settings)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
