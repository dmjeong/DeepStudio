"""Export timestamps survive packaging; they never alter the executable model."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest

import model_version
from export_onnx import create_inference_config
from model_crypto import encrypt_config, load_package


ROOT = Path(__file__).resolve().parents[1]
KEY = bytes(range(32))  # Public test key only.


def test_timestamp_uses_one_clock_read_and_records_timezone(monkeypatch):
    fixed = datetime(2026, 11, 26, 14, 30, 52, 123456, timezone(timedelta(hours=9)))
    monkeypatch.setattr(model_version, "datetime", SimpleNamespace(
        now=lambda: SimpleNamespace(astimezone=lambda: fixed)))
    config = {"schema_version": 6, "model_version": "old", "model_exported_at": "old", "num_threads": 4}
    model_version.stamp_model_version(config)
    assert config == {"schema_version": 6, "num_threads": 4,
                      "model_version": "20261126_143052_123456",
                      "model_exported_at": "2026-11-26T14:30:52.123456+09:00"}


@pytest.mark.parametrize("task", ["classify", "detect", "segment", "anomaly"])
def test_all_task_configs_get_an_export_time(tmp_path, task):
    path = create_inference_config(tmp_path, task, 2, 224, 3, "model.onnx")
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    instant = datetime.fromisoformat(config["model_exported_at"])
    assert instant.utcoffset() is not None
    assert config["model_version"] == instant.strftime("%Y%m%d_%H%M%S_%f")


def test_encryption_and_reencryption_preserve_version_and_predictions(tmp_path):
    config = json.loads((ROOT / "example/assets/test.json").read_text())
    model_version.stamp_model_version(config)
    graph = tmp_path / config["model_path"]
    original_bytes = (ROOT / "example/assets/test.onnx").read_bytes()
    graph.write_bytes(original_bytes)
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps(config), encoding="utf-8")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    plain = ort.InferenceSession(original_bytes, options, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(19).normal(size=(1, 1, 224, 224)).astype(np.float32)
    expected = plain.run(None, {plain.get_inputs()[0].name: x})
    for name in ("first.dvsenc", "second.dvsenc"):
        encrypted = encrypt_config(manifest, tmp_path / name, KEY)
        restored, graphs = load_package(encrypted, KEY)
        assert restored == config  # Encryption does not create another model version.
        assert graphs[config["model_path"]] == original_bytes
        session = ort.InferenceSession(graphs[config["model_path"]], options, providers=["CPUExecutionProvider"])
        for a, b in zip(expected, session.run(None, {session.get_inputs()[0].name: x})):
            np.testing.assert_array_equal(a, b)
    assert graph.read_bytes() == original_bytes
    assert json.loads(manifest.read_text()) == config
