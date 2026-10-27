"""Authenticated format, atomic publication, external tensors and all-task parity."""
import json
from pathlib import Path
import struct
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from model_crypto import MAGIC, create_key, decode_package, encrypt_config, encrypted_export, load_package, read_key

KEY = bytes(range(32))  # Public test vector, never a production key.


def fixture(tmp_path, **updates):
    config = json.loads((ROOT / "example/assets/test.json").read_text())
    config.update(updates)
    path = tmp_path / "model.json"
    graph = tmp_path / config["model_path"]
    graph.parent.mkdir(parents=True, exist_ok=True)
    graph.write_bytes((ROOT / "example/assets/test.onnx").read_bytes())
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path, graph


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])


def test_roundtrip_preserves_settings_unicode_and_model_bytes(tmp_path):
    config, graph = fixture(tmp_path, schema_version=6, num_threads=4,
                            onnxruntime={"graph_optimization_level": "disabled", "allow_spinning": True},
                            class_names=["정상", "불량"])
    target = encrypt_config(config, tmp_path / "검사.dvsenc", KEY)
    doc, models = load_package(target, KEY)
    assert doc == json.loads(config.read_text())
    assert models[doc["model_path"]] == graph.read_bytes()
    plain, encrypted = session(str(graph)), session(models[doc["model_path"]])
    for seed in range(5):
        x = np.random.default_rng(seed).normal(size=(1, 1, 224, 224)).astype(np.float32)
        a = plain.run(None, {plain.get_inputs()[0].name: x})
        b = encrypted.run(None, {encrypted.get_inputs()[0].name: x})
        for one, two in zip(a, b):
            np.testing.assert_array_equal(one, two)


def test_nonce_is_fresh_and_wrong_key_tampering_and_truncation_rejected(tmp_path):
    config, _ = fixture(tmp_path)
    a = encrypt_config(config, tmp_path / "a.dvsenc", KEY).read_bytes()
    b = encrypt_config(config, tmp_path / "b.dvsenc", KEY).read_bytes()
    assert a[8:20] != b[8:20] and a != b
    for offset in (0, 7, 8, 19, 20, len(a)//2, len(a)-1):
        changed = bytearray(a)
        changed[offset] ^= 1
        with pytest.raises(ValueError):
            decode_package(bytes(changed), KEY)
    for length in (0, 8, 20, 39, len(a)-1):
        with pytest.raises(ValueError):
            decode_package(a[:length], KEY)
    with pytest.raises(ValueError, match="암호키"):
        decode_package(a, b"x"*32)


@pytest.mark.parametrize("name", ["../a.onnx", "/a.onnx", "C:/a.onnx", "a\\b", "a//b", "./a", "a/../b", "x\x00y"])
def test_bad_paths_rejected(tmp_path, name):
    config, _ = fixture(tmp_path)
    doc = json.loads(config.read_text()); doc["model_path"] = name
    config.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        encrypt_config(config, tmp_path / "a.dvsenc", KEY)


def test_external_weights_are_packaged_in_memory(tmp_path):
    from onnx import helper, numpy_helper, TensorProto
    x = helper.make_tensor_value_info("input_image", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("class_logits", TensorProto.FLOAT, [1, 2])
    weight = numpy_helper.from_array(np.array([[2., -3.]], np.float32), "w")
    model = helper.make_model(helper.make_graph([helper.make_node("Add", ["input_image", "w"], ["class_logits"])],
                             "external", [x], [y], [weight]), opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    config, graph = fixture(tmp_path)
    onnx.save_model(model, graph, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=0)
    encrypt_config(config, tmp_path / "model.dvsenc", KEY)
    doc, models = load_package(tmp_path / "model.dvsenc", KEY)
    a, b = session(str(graph)), session(models[doc["model_path"]])
    sample = {"input_image": np.ones((1, 2), np.float32)}
    np.testing.assert_array_equal(a.run(None, sample)[0], b.run(None, sample)[0])
    (tmp_path / "weights.bin").unlink()
    np.testing.assert_array_equal(session(models[doc["model_path"]]).run(None, sample)[0], [[3., -2.]])


@pytest.mark.parametrize("kind", ["initializer", "attribute", "subgraph", "function"])
def test_sparse_external_tensors_are_self_contained(tmp_path, kind):
    from onnx import helper, numpy_helper, TensorProto, external_data_helper

    def external(array, name):
        tensor = numpy_helper.from_array(array, name)
        location = name + ".bin"
        (tmp_path / location).write_bytes(tensor.raw_data)
        external_data_helper.set_external_data(tensor, location, offset=0, length=len(tensor.raw_data))
        tensor.ClearField("raw_data")
        return tensor

    values = external(np.array([2., -3.], np.float32), "w")
    indices = numpy_helper.from_array(np.array([0, 1], np.int64), "indices")
    # ORT shape inference needs embedded indices for sparse subgraph constants;
    # the other three source graph forms support both tensors being external.
    if kind != "subgraph":
        indices = external(np.array([0, 1], np.int64), "indices")
    sparse = helper.make_sparse_tensor(values, indices, [1, 2])
    x = helper.make_tensor_value_info("input_image", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("class_logits", TensorProto.FLOAT, [1, 2])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 2])
    nodes = [helper.make_node("Add", ["input_image", "w"], ["class_logits"])]
    initializers, sparse_initializers, functions = [], [], []
    opsets = [helper.make_opsetid("", 13)]
    constant = helper.make_node("Constant", [], ["w"], sparse_value=sparse)
    if kind == "initializer":
        sparse_initializers = [sparse]
    elif kind == "attribute":
        nodes.insert(0, constant)
    elif kind == "subgraph":
        then_graph = helper.make_graph([constant], "then", [], [w])
        otherwise = helper.make_node("Constant", [], ["w"],
                                     value=numpy_helper.from_array(np.array([[2., -3.]], np.float32)))
        else_graph = helper.make_graph([otherwise], "else", [], [w])
        initializers = [numpy_helper.from_array(np.array(True), "condition")]
        nodes.insert(0, helper.make_node("If", ["condition"], ["w"],
                                        then_branch=then_graph, else_branch=else_graph))
    else:
        functions = [helper.make_function("crypto.test", "Weights", [], ["w"], [constant],
                                           opset_imports=opsets)]
        opsets.append(helper.make_opsetid("crypto.test", 1))
        nodes.insert(0, helper.make_node("Weights", [], ["w"], domain="crypto.test"))
    model = helper.make_model(helper.make_graph(nodes, "external-sparse", [x], [y], initializers,
                                                sparse_initializer=sparse_initializers),
                              opset_imports=opsets, ir_version=8, functions=functions)
    config, graph = fixture(tmp_path)
    graph.write_bytes(model.SerializeToString())
    sample = {"input_image": np.ones((1, 2), np.float32)}
    expected = session(str(graph)).run(None, sample)[0]
    target = encrypt_config(config, tmp_path / "sparse.dvsenc", KEY)
    doc, graphs = load_package(target, KEY)
    for path in tmp_path.glob("*.bin"):
        path.unlink()
    np.testing.assert_array_equal(session(graphs[doc["model_path"]]).run(None, sample)[0], expected)


def test_external_weight_output_collision_preserves_source(tmp_path):
    from onnx import helper, numpy_helper, TensorProto
    x = helper.make_tensor_value_info("input_image", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("class_logits", TensorProto.FLOAT, [1, 2])
    weight = numpy_helper.from_array(np.array([[2., -3.]], np.float32), "w")
    model = helper.make_model(helper.make_graph([helper.make_node("Add", ["input_image", "w"], ["class_logits"])],
                             "external", [x], [y], [weight]), opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    config, graph = fixture(tmp_path)
    onnx.save_model(model, graph, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.dvsenc", size_threshold=0)
    weights = tmp_path / "weights.dvsenc"
    original = weights.read_bytes()
    with pytest.raises(ValueError, match="외부 가중치"):
        encrypt_config(config, weights, KEY)
    assert weights.read_bytes() == original
    sample = {"input_image": np.ones((1, 2), np.float32)}
    np.testing.assert_array_equal(session(str(graph)).run(None, sample)[0], [[3., -2.]])


def test_oversize_external_weights_fail_before_loading_or_publishing(tmp_path, monkeypatch):
    from onnx import helper, numpy_helper, TensorProto
    import model_crypto
    value = helper.make_tensor_value_info("w", TensorProto.FLOAT, [2048])
    weight = numpy_helper.from_array(np.zeros(2048, np.float32), "w")
    model = helper.make_model(helper.make_graph([], "large-external", [], [value], [weight]),
                              opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    config, graph = fixture(tmp_path)
    onnx.save_model(model, graph, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=0)
    target = tmp_path / "model.dvsenc"
    target.write_bytes(b"previous")
    monkeypatch.setattr(model_crypto, "MAX_BYTES", 2048)
    original_open = Path.open
    def checked_open(path, mode="r", *args, **kwargs):
        if path.name == "weights.bin" and mode == "rb":
            pytest.fail("oversize weights must not be loaded")
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked_open)
    with pytest.raises(ValueError, match="2 GiB"):
        encrypt_config(config, target, KEY)
    assert target.read_bytes() == b"previous"


def test_zero_length_external_tensor_is_bounded_on_older_onnx(tmp_path, monkeypatch):
    from onnx import helper, numpy_helper, TensorProto, external_data_helper
    import model_crypto
    weight = numpy_helper.from_array(np.zeros(0, np.float32), "w")
    external_data_helper.set_external_data(weight, "weights.bin", offset=0, length=0)
    weight.ClearField("raw_data")
    value = helper.make_tensor_value_info("w", TensorProto.FLOAT, [0])
    model = helper.make_model(helper.make_graph([], "zero-length", [], [value], [weight]),
                              opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    graph = tmp_path / "model.onnx"
    graph.write_bytes(model.SerializeToString())
    config = tmp_path / "model.json"
    config.write_text(json.dumps({"model_path": graph.name}))
    (tmp_path / "weights.bin").write_bytes(b"x" * 8192)
    monkeypatch.setattr(model_crypto, "MAX_BYTES", 512)
    legacy_calls = []
    def legacy_loader(tensor, base_dir):
        # ONNX 1.16 treats an explicit zero exactly like an omitted length.
        legacy_calls.append(tensor.name)
        info = external_data_helper.ExternalDataInfo(tensor)
        with (Path(base_dir) / info.location).open("rb") as stream:
            if info.offset:
                stream.seek(info.offset)
            tensor.raw_data = stream.read(info.length) if info.length else stream.read()
    monkeypatch.setattr(external_data_helper, "load_external_data_for_tensor", legacy_loader)
    target = encrypt_config(config, tmp_path / "model.dvsenc", KEY)
    _, graphs = load_package(target, KEY)
    assert legacy_calls == []
    assert target.stat().st_size <= 512
    assert len(onnx.load_from_string(graphs[graph.name]).graph.initializer[0].raw_data) == 0
    (tmp_path / "weights.bin").unlink()
    np.testing.assert_array_equal(session(graphs[graph.name]).run(None, {})[0], np.zeros(0, np.float32))


def test_external_file_truncated_after_preflight_preserves_package(tmp_path, monkeypatch):
    from onnx import helper, numpy_helper, TensorProto
    weight = numpy_helper.from_array(np.array([2., -3.], np.float32), "w")
    value = helper.make_tensor_value_info("w", TensorProto.FLOAT, [2])
    model = helper.make_model(helper.make_graph([], "external", [], [value], [weight]),
                              opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    config, graph = fixture(tmp_path)
    onnx.save_model(model, graph, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=0)
    target = tmp_path / "model.dvsenc"
    target.write_bytes(b"previous")
    original_open = Path.open
    def truncate_before_read(path, mode="r", *args, **kwargs):
        if path.name == "weights.bin" and mode == "rb":
            with original_open(path, "wb") as stream:
                stream.write(b"\0" * 4)
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", truncate_before_read)
    with pytest.raises(ValueError, match="잘렸"):
        encrypt_config(config, target, KEY)
    assert target.read_bytes() == b"previous"


def test_key_is_never_overwritten_and_failed_export_preserves_package(tmp_path):
    key = create_key(tmp_path / "model.key")
    original_key = read_key(key)
    with pytest.raises(FileExistsError):
        create_key(key)
    assert read_key(key) == original_key
    target = tmp_path / "model.dvsenc"
    target.write_bytes(b"previous")
    def failing(path):
        path.write_bytes(b"staging plaintext")
        raise ValueError("test export failed")
    with pytest.raises(ValueError, match="test export"):
        encrypted_export(failing, target, key)
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".encrypted-export-*"))
    key.write_bytes(b"short")
    with pytest.raises(ValueError, match="32"):
        encrypted_export(lambda p: pytest.fail("must fail before exporter"), target, key)


@pytest.mark.parametrize("change", ["length", "duplicate", "missing", "trailing", "version", "unsafe", "null_row", "null_contract"])
def test_authenticated_but_malformed_payload_rejected(tmp_path, change):
    config, graph = fixture(tmp_path)
    c = json.loads(config.read_text())
    desc = {"name": c["model_path"], "size": graph.stat().st_size}
    doc = {"format": "dvs-model-v1", "config": c, "models": [desc]}
    if change == "length": desc["size"] = 2**63
    if change == "duplicate": doc["models"].append(desc.copy())
    if change == "missing": c["model_path"] = "missing.onnx"
    if change == "version": doc["format"] = "dvs-model-v99"
    if change == "unsafe": desc["name"] = c["model_path"] = "../model.onnx"
    if change == "null_row": doc["models"] = [None]
    if change == "null_contract": c["contracts"] = None
    header = json.dumps(doc).encode()
    payload = struct.pack("<I", len(header)) + header + graph.read_bytes()
    if change == "trailing": payload += b"trailing"
    # Unique nonce per test payload, no nonce/key reuse even in test fixtures.
    import os
    nonce = os.urandom(12)
    data = MAGIC + nonce + AESGCM(KEY).encrypt(nonce, payload, MAGIC)
    with pytest.raises(ValueError): decode_package(data, KEY)


@pytest.mark.parametrize("encoding,constant", [("utf-16", 0), ("utf-32", 0),
                                               ("utf-8", float("nan")), ("utf-8", float("inf")),
                                               ("utf-8", "overflow")])
def test_authenticated_header_requires_utf8_and_finite_json(tmp_path, encoding, constant):
    config, graph = fixture(tmp_path)
    settings = json.loads(config.read_text())
    settings["nonstandard_number"] = constant
    doc = {"format": "dvs-model-v1", "config": settings,
           "models": [{"name": settings["model_path"], "size": graph.stat().st_size}]}
    header = json.dumps(doc).replace('"overflow"', "1e999").encode(encoding)
    payload = struct.pack("<I", len(header)) + header + graph.read_bytes()
    import os
    nonce = os.urandom(12)
    encrypted = MAGIC + nonce + AESGCM(KEY).encrypt(nonce, payload, MAGIC)
    with pytest.raises(ValueError):
        decode_package(encrypted, KEY)


def test_export_option_routes_official_sam2_through_same_encryption(tmp_path, monkeypatch):
    import export_onnx
    import export_sam2_onnx
    seen = []
    def sam(model_id, checkpoint, output, **kwargs):
        seen.append(model_id)
        config, graph = fixture(output)
        doc = json.loads(config.read_text())
        decoder = output / "decoder.onnx"; decoder.write_bytes(graph.read_bytes())
        doc.update(backend="sam2", task="segment", contracts={"graphs": {
            "encoder": {"file": graph.name}, "decoder": {"file": decoder.name}}})
        config.write_text(json.dumps(doc))
        return {"config_path": str(config), "encoder_path": str(graph), "decoder_path": str(decoder), "verification": "passed"}
    monkeypatch.setattr(export_sam2_onnx, "export_official_sam2_checkpoint", sam)
    key = create_key(tmp_path / "secret.key")
    for variant in ("tiny", "small", "base_plus", "large"):
        result = export_onnx.export_checkpoint("unused", tmp_path / f"{variant}.dvsenc",
                    encryption_key_path=key, sam2_model_id=f"sam2_hiera_{variant}", log=lambda _: None)
        doc, graphs = load_package(result["output_path"], read_key(key))
        assert len(graphs) == 2 and doc["backend"] == "sam2"
        assert result["config_path"] is None and "encoder_path" not in result
    assert len(seen) == 4 and not list(tmp_path.glob("*.onnx")) and not list(tmp_path.glob("*.json"))


def test_standalone_header_matches_sdk():
    assert (ROOT / "example/cpp/model_crypto.h").read_bytes() == (ROOT / "cpp/include/model_crypto.h").read_bytes()
