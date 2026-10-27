"""Task-independent authenticated ONNX + deployment JSON container (DVSENC01).

Encryption changes storage only. No graph optimization, quantization or runtime
tuning is performed here. See docs/ENCRYPTED_ONNX.md for the wire format.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import tempfile

MAGIC = b"DVSENC01"
MAX_BYTES = 2**31 - 1
MAX_JSON = 16 * 1024 * 1024


def read_key(path) -> bytes:
    with Path(path).open("rb") as stream:
        key = stream.read(33)
    if len(key) != 32:
        raise ValueError("암호키 파일은 정확히 32바이트여야 합니다.")
    return key


def create_key(path):
    """Create a new key exclusively; never overwrite an existing key."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(os.urandom(32))
    return target


def safe_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or
            "\x00" in name or name.startswith("/") or
            any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("암호화 패키지의 모델 경로가 올바르지 않습니다.")
    return name


def model_names(config):
    names = [safe_name(config.get("model_path"))]
    contracts = config.get("contracts", {})
    if not isinstance(contracts, dict) or not isinstance(contracts.get("graphs", {}), dict):
        raise ValueError("암호화 모델의 그래프 계약이 올바르지 않습니다.")
    for graph in contracts.get("graphs", {}).values():
        if not isinstance(graph, dict):
            raise ValueError("암호화 모델의 그래프 항목이 올바르지 않습니다.")
        name = safe_name(graph.get("file"))
        if name not in names:
            names.append(name)
    return names


def _inside_file(root, name):
    path = root / safe_name(name)
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("모델 또는 외부 가중치 파일이 없거나 배포 폴더 밖에 있습니다.")
    if path.is_symlink() or any((root / Path(*Path(name).parts[:i])).is_symlink() for i in range(1, len(Path(name).parts))):
        raise ValueError("모델 패키지에는 심볼릭 링크를 사용할 수 없습니다.")
    return path


def _all_tensors(message):
    """Visit dense/sparse tensors everywhere, including functions and subgraphs.

    ONNX's external-data model helpers omit SparseTensorProto values/indices.
    Walk protobuf messages so every tensor stored in the model is self-contained.
    """
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        yield message
        return
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            repeated = (field.is_repeated if hasattr(field, "is_repeated")
                        else field.label == field.LABEL_REPEATED)
            if repeated:
                for item in value:
                    yield from _all_tensors(item)
            else:
                yield from _all_tensors(value)


def _graph_bytes(root, name, *, output, max_bytes):
    import onnx
    from onnx import external_data_helper
    path = _inside_file(root, name)
    if path.stat().st_size > max_bytes:
        raise ValueError("암호화 포맷 v1은 2 GiB 미만을 지원합니다.")
    model = onnx.load(path, load_external_data=False)
    external = [t for t in _all_tensors(model)
                if external_data_helper.uses_external_data(t)]
    if not external:
        return path.read_bytes()
    external_bytes, reads = 0, []
    for tensor in external:
        info = external_data_helper.ExternalDataInfo(tensor)
        dependency = _inside_file(path.parent, info.location)
        if output.resolve() == dependency.resolve():
            raise ValueError("암호화 출력으로 원본 외부 가중치 파일을 덮어쓸 수 없습니다.")
        available = dependency.stat().st_size
        offset = info.offset if info.offset is not None else 0
        length = info.length if info.length is not None else available - offset
        if offset < 0 or length < 0 or offset > available or length > available - offset:
            raise ValueError("외부 가중치의 offset/length가 파일 크기를 벗어납니다.")
        external_bytes += length
        if external_bytes > max_bytes:
            raise ValueError("암호화 포맷 v1은 2 GiB 미만을 지원합니다.")
        reads.append((tensor, dependency, offset, length))
    for tensor, dependency, offset, length in reads:
        # Older supported ONNX helpers interpret length=0 as read-to-EOF.
        # Read exactly the validated range regardless of the installed version.
        with dependency.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(length)
        if len(data) != length:
            raise ValueError("외부 가중치 파일이 변경되었거나 데이터가 잘렸습니다.")
        tensor.raw_data = data
        del tensor.external_data[:]
        tensor.data_location = onnx.TensorProto.DEFAULT
    if model.ByteSize() > max_bytes:
        raise ValueError("암호화 포맷 v1은 2 GiB 미만을 지원합니다.")
    return model.SerializeToString()


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("암호화 패키지 JSON에 중복 필드가 있습니다.")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ValueError(f"JSON에는 유한한 숫자만 사용할 수 있습니다: {value}")


def _finite_json_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        _invalid_json_constant(value)
    return parsed


def decode_package(data: bytes, key: bytes):
    """Authenticate before parsing. Return original config and in-memory graphs."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.exceptions import InvalidTag
    if len(key) != 32:
        raise ValueError("AES-256 키는 32바이트여야 합니다.")
    if not 40 <= len(data) <= MAX_BYTES or data[:8] != MAGIC:
        raise ValueError("지원하지 않거나 손상된 암호화 모델 파일입니다.")
    try:
        plain = AESGCM(key).decrypt(data[8:20], data[20:], MAGIC)
    except InvalidTag as exc:
        raise ValueError("암호키가 맞지 않거나 모델 파일이 변조되었습니다.") from exc
    if len(plain) < 4:
        raise ValueError("암호화 모델 payload가 잘렸습니다.")
    size = struct.unpack_from("<I", plain)[0]
    if not 0 < size <= min(MAX_JSON, len(plain) - 4):
        raise ValueError("암호화 모델 JSON 크기가 올바르지 않습니다.")
    doc = json.loads(plain[4:4+size].decode("utf-8"), object_pairs_hook=_no_duplicates,
                     parse_constant=_invalid_json_constant, parse_float=_finite_json_float)
    if not isinstance(doc, dict) or doc.get("format") != "dvs-model-v1" or not isinstance(doc.get("config"), dict):
        raise ValueError("암호화 모델 계약이 올바르지 않습니다.")
    rows = doc.get("models")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
        raise ValueError("암호화 모델 그래프 목록이 올바르지 않습니다.")
    offset, models = 4 + size, {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("암호화 모델 그래프 항목이 올바르지 않습니다.")
        name = safe_name(row.get("name"))
        length = row.get("size")
        if name in models or type(length) is not int or not 0 < length <= len(plain) - offset:
            raise ValueError("암호화 모델 그래프 크기 또는 이름이 올바르지 않습니다.")
        models[name] = plain[offset:offset+length]
        offset += length
    if offset != len(plain) or set(model_names(doc["config"])) != set(models):
        raise ValueError("암호화 모델과 배포 설정이 일치하지 않습니다.")
    return doc["config"], models


def load_package(path, key):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("암호화 포맷 v1은 2 GiB 미만을 지원합니다.")
    return decode_package(path.read_bytes(), key)


def encrypt_config(config_path, output_path, key: bytes):
    """Pack any exported task, including SAM2. Publish only after round-trip QA."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    source, output = Path(config_path), Path(output_path)
    if output.suffix.lower() != ".dvsenc":
        raise ValueError("암호화 출력 확장자는 .dvsenc여야 합니다.")
    if len(key) != 32:
        raise ValueError("AES-256 키는 32바이트여야 합니다.")
    config = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates,
                        parse_constant=_invalid_json_constant, parse_float=_finite_json_float)
    if not isinstance(config, dict):
        raise ValueError("배포 설정은 JSON object여야 합니다.")
    names = model_names(config)
    if len(names) > 64:
        raise ValueError("암호화 모델 그래프는 최대 64개까지 지원합니다.")
    inputs = [source, *(source.parent / name for name in names)]
    if any(output.resolve() == path.resolve() for path in inputs):
        raise ValueError("암호화 출력으로 원본 모델 또는 설정 파일을 덮어쓸 수 없습니다.")
    graphs, graph_bytes = {}, 0
    for name in names:
        graphs[name] = _graph_bytes(source.parent, name, output=output,
                                    max_bytes=MAX_BYTES - 40 - graph_bytes)
        graph_bytes += len(graphs[name])
    header = json.dumps({"format": "dvs-model-v1", "config": config,
                         "models": [{"name": n, "size": len(v)} for n, v in graphs.items()]},
                        ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(header) > MAX_JSON or 40 + len(header) + sum(map(len, graphs.values())) > MAX_BYTES:
        raise ValueError("암호화 포맷 v1 크기 제한을 초과했습니다 (전체 2 GiB / JSON 16 MiB).")
    payload = struct.pack("<I", len(header)) + header + b"".join(graphs.values())
    nonce = os.urandom(12)
    encrypted = MAGIC + nonce + AESGCM(key).encrypt(nonce, payload, MAGIC)
    # Every exported byte and all settings must survive encryption unchanged.
    restored_config, restored_models = decode_package(encrypted, key)
    if restored_config != config or restored_models != graphs:
        raise RuntimeError("암호화 모델 복원 검증 실패")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".encrypted-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encrypted)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


def encrypted_export(exporter, output_path, key_path, log=print):
    """Run an existing exporter in private staging, then publish one ciphertext."""
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".dvsenc":
        raise ValueError("암호화 출력 확장자는 .dvsenc여야 합니다.")
    if output == Path(key_path).resolve():
        raise ValueError("암호키와 모델의 저장 경로가 같을 수 없습니다.")
    key = read_key(key_path)  # fail before expensive export
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".encrypted-export-", dir=output.parent) as temp:
        result = exporter(Path(temp) / "model.onnx")
        encrypt_config(result["config_path"], output, key)
    # Do not leak dead staging paths or create a plaintext sidecar.
    for name in ("config_path", "output_dir", "encoder_path", "decoder_path", "bundle_path"):
        result.pop(name, None)
    result.update(output_path=str(output), config_path=None, encrypted=True,
                  encryption="AES-256-GCM", file_size_mb=output.stat().st_size / 1024**2,
                  encryption_verification="byte_exact_roundtrip")
    log(f"암호화 내보내기 완료: {output} (모델·설정 복원 일치 확인)")
    return result
