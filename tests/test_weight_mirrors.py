"""Retry publisher TLS outages without bypassing trust or checkpoint identity."""

import hashlib
import io
import ssl
from urllib.error import URLError
from unittest.mock import patch

import pytest

from model_download import cached_https_weights_from_sources


def test_certificate_outage_retries_pinned_https_bytes_and_keeps_offline_cache(tmp_path):
    payload = b"original publisher checkpoint bytes"
    expected = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "weight.pth"
    urls = ("https://publisher.example/model", "https://mirror.example/model")
    with patch("model_download.download_context"), patch("model_download.build_opener") as opener:
        opener.return_value.open.side_effect = [
            URLError(ssl.SSLCertVerificationError(1, "certificate has expired")), io.BytesIO(payload),
        ]
        assert cached_https_weights_from_sources(urls, destination, expected) == destination
        assert destination.read_bytes() == payload
        assert [call.args[0].full_url for call in opener.return_value.open.call_args_list] == list(urls)
    with patch("model_download.download_context", side_effect=AssertionError("offline cache")):
        assert cached_https_weights_from_sources(urls, destination, expected) == destination
    assert list(tmp_path.iterdir()) == [destination]


def test_mirrors_never_hide_corrupt_data_or_allow_short_hashes(tmp_path):
    expected = hashlib.sha256(b"original").hexdigest()
    urls = ("https://publisher.example/model", "https://mirror.example/model")
    with patch("model_download.download_context"), patch("model_download.build_opener") as opener:
        opener.return_value.open.return_value = io.BytesIO(b"tampered")
        with pytest.raises(ValueError, match="SHA256"):
            cached_https_weights_from_sources(urls, tmp_path / "weight.pth", expected)
        assert opener.return_value.open.call_count == 1
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="SHA256"):
        cached_https_weights_from_sources(urls, tmp_path / "weight.pth", expected[:8])
