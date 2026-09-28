"""Protect the only copy of a checkpoint/key, including diagnostic collisions."""
import json
import os

import onnx
from onnx import TensorProto, helper
import pytest

import export_onnx
from model_crypto import create_key, load_package, read_key


def fake_export(checkpoint, output, *args, **kwargs):
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])])
    onnx.save(helper.make_model(graph), str(output))
    config = output.with_suffix(".json")
    config.write_text(json.dumps({"model_path": output.name}))
    return {"config_path": str(config), "output_path": str(output)}


@pytest.mark.parametrize("source", ["checkpoint", "key"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
@pytest.mark.parametrize("fail", [False, True])
def test_diagnostic_never_deletes_or_overwrites_sources(tmp_path, monkeypatch, source, alias, fail):
    output = tmp_path / "model.dvsenc"
    diagnostic = output.with_suffix(".export-error.json")
    checkpoint = tmp_path / "source.pt"
    key = tmp_path / "secret.key"
    if alias == "direct":
        if source == "key":
            key = diagnostic
        else:
            checkpoint = diagnostic
    checkpoint.write_bytes(b"checkpoint must survive")
    create_key(key)
    if alias != "direct":
        origin = key if source == "key" else checkpoint
        try:
            diagnostic.symlink_to(origin) if alias == "symlink" else os.link(origin, diagnostic)
        except OSError:
            pytest.skip("This OS/account does not allow the requested filesystem link")
    before = checkpoint.read_bytes(), key.read_bytes()
    def exporter(*args, **kwargs):
        if fail:
            raise RuntimeError("intentional export failure")
        return fake_export(*args, **kwargs)
    monkeypatch.setattr(export_onnx, "_export_checkpoint", exporter)
    if fail:
        with pytest.raises(ValueError, match="intentional export failure"):
            export_onnx.export_checkpoint(checkpoint, output, encryption_key_path=key, log=lambda _: None)
        assert not output.exists()
    else:
        export_onnx.export_checkpoint(checkpoint, output, encryption_key_path=key, log=lambda _: None)
        assert load_package(output, read_key(key))[0]["model_path"] == "model.onnx"
    assert (checkpoint.read_bytes(), key.read_bytes()) == before
    assert diagnostic.exists()


@pytest.mark.parametrize("source", ["checkpoint", "key"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_export_cannot_overwrite_its_source(tmp_path, monkeypatch, source, alias):
    checkpoint, key, output = tmp_path / "source.pt", tmp_path / "secret.key", tmp_path / "result.dvsenc"
    checkpoint.write_bytes(b"checkpoint must survive")
    create_key(key)
    origin = key if source == "key" else checkpoint
    if alias == "direct":
        output = origin
    else:
        try:
            output.symlink_to(origin) if alias == "symlink" else os.link(origin, output)
        except OSError:
            pytest.skip("This OS/account does not allow the requested filesystem link")
    before = checkpoint.read_bytes(), key.read_bytes()
    def should_not_run(*args, **kwargs):
        pytest.fail("Source collision must be rejected before export starts")
    monkeypatch.setattr(export_onnx, "_export_checkpoint", should_not_run)
    with pytest.raises(ValueError, match="다른 경로"):
        export_onnx.export_checkpoint(checkpoint, output, encryption_key_path=key, log=lambda _: None)
    assert (checkpoint.read_bytes(), key.read_bytes()) == before
