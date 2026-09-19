"""Ed25519 authentication for offline model packs."""

from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path
from typing import Any, Mapping


MAX_TRUST_STORE_BYTES = 1024 * 1024


def load_trusted_keys(path: str | Path) -> dict[str, bytes]:
    """Load a strict versioned Ed25519 public-key trust store.

    The file contains public keys only and is safe to distribute with a
    release. Private signing keys must never be placed in this file.
    """
    source = Path(path).expanduser()
    if not source.is_absolute():
        raise ValueError("model pack trust store path must be absolute")
    if source.is_symlink() or not source.is_file():
        raise ValueError("model pack trust store must be a regular file")
    try:
        if source.stat().st_size > MAX_TRUST_STORE_BYTES:
            raise ValueError("model pack trust store is too large")
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("model pack trust store is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("model pack trust store must use schema_version 1")
    entries = value.get("keys")
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("model pack trust store requires at least one key")
    trusted: dict[str, bytes] = {}
    for raw_key_id, entry in entries.items():
        if not isinstance(raw_key_id, str) or not raw_key_id.strip() or raw_key_id != raw_key_id.strip():
            raise ValueError("model pack trust store contains an invalid key id")
        if not isinstance(entry, Mapping) or entry.get("algorithm") != "ed25519":
            raise ValueError(f"trusted model pack key must use Ed25519: {raw_key_id}")
        encoded = entry.get("public_key")
        if not isinstance(encoded, str) or not encoded.strip():
            raise ValueError(f"trusted model pack key is missing public_key: {raw_key_id}")
        try:
            key_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"trusted model pack key is not valid base64: {raw_key_id}") from exc
        if len(key_bytes) != 32:
            raise ValueError(f"trusted model pack Ed25519 key must be 32 bytes: {raw_key_id}")
        trusted[raw_key_id] = key_bytes
    return trusted


def signature_payload(manifest: Mapping[str, Any], file_hashes: Mapping[str, Any]) -> bytes:
    """Return the versioned canonical bytes authenticated by a pack signature.

    ``manifest.json`` is excluded from the signed hash map because that file
    contains the signature itself. Its remaining fields are included directly.
    Every other archive file is bound through its SHA-256 value.
    """
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("signature", None)
    authenticated_files = {
        str(name): value for name, value in file_hashes.items()
        if str(name) != "manifest.json"
    }
    value = {
        "schema": "deepvision-model-pack-signature-v1",
        "manifest": unsigned_manifest,
        "files": authenticated_files,
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def verify_pack_signature(manifest: Mapping[str, Any], file_hashes: Mapping[str, Any],
                          trusted_keys: Mapping[str, bytes | str]) -> str:
    """Verify a manifest signature and return its trusted key id."""
    signature = manifest.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError("model pack signature object is required")
    if signature.get("algorithm") != "ed25519":
        raise ValueError("model pack signature algorithm must be ed25519")
    key_id = signature.get("key_id")
    encoded = signature.get("value")
    if not isinstance(key_id, str) or not key_id.strip():
        raise ValueError("model pack signature key_id is required")
    key_id = key_id.strip()
    if key_id not in trusted_keys:
        raise ValueError(f"model pack signature uses an unknown key: {key_id}")
    if not isinstance(encoded, str) or not encoded.strip():
        raise ValueError("model pack signature value is required")
    try:
        signature_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("model pack signature is not valid base64") from exc
    if len(signature_bytes) != 64:
        raise ValueError("model pack Ed25519 signature must be 64 bytes")

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise ValueError("cryptography is required to verify signed model packs") from exc

    key_value = trusted_keys[key_id]
    if isinstance(key_value, str):
        key_value = key_value.encode("utf-8")
    if not isinstance(key_value, bytes):
        raise ValueError(f"trusted model pack key has an invalid type: {key_id}")
    try:
        if key_value.startswith(b"-----BEGIN"):
            public_key = serialization.load_pem_public_key(key_value)
            if not isinstance(public_key, Ed25519PublicKey):
                raise ValueError(f"trusted model pack key is not Ed25519: {key_id}")
        else:
            public_key = Ed25519PublicKey.from_public_bytes(key_value)
        public_key.verify(signature_bytes, signature_payload(manifest, file_hashes))
    except InvalidSignature as exc:
        raise ValueError("model pack signature verification failed") from exc
    except (TypeError, ValueError) as exc:
        if str(exc).startswith("trusted model pack key"):
            raise
        raise ValueError(f"trusted model pack key is invalid: {key_id}") from exc
    return key_id


__all__ = ["load_trusted_keys", "signature_payload", "verify_pack_signature"]
