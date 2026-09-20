"""Verified HTTPS model downloads using native CA stores, without global SSL patches."""

import hashlib
import os
from pathlib import Path
import re
import ssl
import tempfile
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import build_opener, HTTPRedirectHandler, HTTPSHandler, Request


_REQUESTS_NATIVE_CA_ENABLED = False


def certificate_failure(exc):
    """urllib wraps TLS failures in URLError.reason; retain the original cause."""
    seen = set()
    while isinstance(exc, BaseException) and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ssl.SSLCertVerificationError):
            return True
        if isinstance(exc, ssl.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(exc).upper():
            return True
        exc = exc.reason if isinstance(exc, URLError) else exc.__cause__ or exc.__context__
    return False


def download_context():
    try:
        import certifi
        import truststore
    except ImportError as exc:
        raise RuntimeError(
            "인증서 지원 패키지 설치 필요. 스튜디오를 실행하는 Python 환경에서 "
            "python -m pip install truststore==0.10.4 certifi 실행 후 다시 시작하세요."
        ) from exc
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_default_certs()
    context.load_verify_locations(cafile=certifi.where())
    # Explicit CA bundles supplied by the operator also apply to urllib downloads.
    for path in dict.fromkeys(os.environ.get(key) for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")):
        if path:
            context.load_verify_locations(cafile=path)
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    return context


def enable_requests_native_ca() -> None:
    """Make third-party ``requests`` downloaders trust the Windows CA store.

    LibreYOLO downloads its pretrained files through ``requests`` rather than
    this module's urllib opener.  ``truststore.SSLContext`` alone therefore
    does not help that path.  Injecting truststore before importing that
    downloader keeps certificate verification enabled while including an
    organization's Windows-installed TLS inspection root certificate.
    """
    global _REQUESTS_NATIVE_CA_ENABLED
    if _REQUESTS_NATIVE_CA_ENABLED:
        return
    try:
        import truststore
    except ImportError as exc:
        raise RuntimeError(
            "Windows 인증서 저장소를 사용하려면 truststore==0.10.4가 필요합니다. "
            "gui 폴더에서 python -m pip install -r requirements.txt를 실행하세요."
        ) from exc
    try:
        truststore.inject_into_ssl()
    except Exception as exc:
        raise RuntimeError(
            "Windows 시스템 인증서를 LibreYOLO 다운로드기에 적용하지 못했습니다. "
            "HTTPS 검증은 유지됩니다."
        ) from exc
    _REQUESTS_NATIVE_CA_ENABLED = True


class _HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise URLError("가중치 다운로드의 HTTPS가 아닌 리디렉션 거부")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _verify_hash(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if not digest.hexdigest().startswith(expected):
        raise ValueError(f"가중치 SHA256 불일치: {path}. 손상된 파일을 확인한 후 다시 다운로드하세요.")


def cached_imagenet_weights(url, cache):
    """Publish only complete, hash-checked files into torchvision's normal cache."""
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    match = re.search(r"-([a-f0-9]{8,64})\.[^.]+$", filename)
    if parsed.scheme != "https" or not match:
        raise ValueError("HTTPS와 SHA256 접미사가 있는 공식 가중치 URL 필요")
    cache = Path(cache)
    destination = cache / filename
    if destination.is_file():
        _verify_hash(destination, match[1])
        return destination  # Cached/offline use does not need an SSL context.
    context = download_context()
    opener = build_opener(HTTPSHandler(context=context), _HTTPSRedirect())
    cache.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(url, headers={"User-Agent": "Deep-Vision-Studio"})
        with opener.open(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=cache, prefix=".weights-", suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                while block := response.read(1024 * 1024):
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
        _verify_hash(temporary, match[1])
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
