"""Real local HTTPS: untrusted roots, hostname checks, trusted CA recovery and cache integrity."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from model_download import cached_imagenet_weights, certificate_failure, download_context, _HTTPSRedirect

PAYLOAD = b"model download fixture\n" * 256
FILENAME = "model-" + hashlib.sha256(PAYLOAD).hexdigest()[:8] + ".pth"
HAS_NATIVE = all(importlib.util.find_spec(name) for name in ("truststore", "certifi"))


class DownloadTests(unittest.TestCase):
    def test_incomplete_download_does_not_replace_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch("model_download.download_context"), patch("model_download.build_opener") as opener:
                opener.return_value.open.return_value = io.BytesIO(PAYLOAD[:-1])
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    cached_imagenet_weights("https://example.com/" + FILENAME, cache)
            self.assertEqual(list(cache.iterdir()), [])

    def test_valid_cache_is_offline_and_corrupt_cache_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / FILENAME
            path.write_bytes(PAYLOAD)
            with patch("model_download.download_context", side_effect=AssertionError("no network")):
                self.assertEqual(cached_imagenet_weights("https://example.com/" + FILENAME, directory), path)
                path.write_bytes(b"broken")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    cached_imagenet_weights("https://example.com/" + FILENAME, directory)
            self.assertEqual(path.read_bytes(), b"broken")

    def test_http_and_downgrade_redirect_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            cached_imagenet_weights("http://example.com/" + FILENAME, ".")
        with self.assertRaises(URLError):
            _HTTPSRedirect().redirect_request(None, None, 302, "", {}, "http://example.com/" + FILENAME)

    def test_only_certificate_errors_are_classified_as_tls_trust_failures(self):
        self.assertTrue(certificate_failure(URLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))))
        self.assertFalse(certificate_failure(URLError("connection timed out")))


@unittest.skipUnless(shutil.which("openssl"), "OpenSSL fixture generator required")
class HTTPSDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        def openssl(*args):
            subprocess.run([shutil.which("openssl"), *args], cwd=root, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
        openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                "-keyout", "ca.key", "-out", "ca.pem", "-subj", "/CN=Studio TLS Test CA",
                "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign")
        openssl("req", "-newkey", "rsa:2048", "-nodes", "-keyout", "server.key", "-out", "server.csr",
                "-subj", "/CN=127.0.0.1")
        (root / "server.ext").write_text(
            "subjectAltName=IP:127.0.0.1\nbasicConstraints=critical,CA:FALSE\n"
            "keyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n"
            "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n", encoding="ascii")
        openssl("x509", "-req", "-in", "server.csr", "-CA", "ca.pem", "-CAkey", "ca.key",
                "-CAcreateserial", "-out", "server.pem", "-days", "1", "-extfile", "server.ext")
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.end_headers()
                self.wfile.write(PAYLOAD)

            def log_message(self, *args):
                pass
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(root / "server.pem", root / "server.key")
        cls.server.socket = context.wrap_socket(cls.server.socket, server_side=True)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"https://127.0.0.1:{cls.server.server_port}/{FILENAME}"
        cls.ca = root / "ca.pem"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.temp.cleanup()

    def check_tls_recovery(self, context_factory):
        original_context = ssl._create_default_https_context
        with tempfile.TemporaryDirectory() as directory:
            # No test root is trusted initially. A failed connection leaves no final file.
            with patch("model_download.download_context", side_effect=context_factory), \
                    patch.dict(os.environ, {"SSL_CERT_FILE": "", "REQUESTS_CA_BUNDLE": "", "NO_PROXY": "127.0.0.1,localhost"}):
                with self.assertRaises(URLError) as error:
                    cached_imagenet_weights(self.url, directory)
                self.assertTrue(certificate_failure(error.exception))
                self.assertEqual(list(Path(directory).iterdir()), [])
            with patch("model_download.download_context", side_effect=context_factory), \
                    patch.dict(os.environ, {"SSL_CERT_FILE": str(self.ca), "REQUESTS_CA_BUNDLE": "", "NO_PROXY": "127.0.0.1,localhost"}):
                # Correct CA still must not permit a hostname mismatch.
                with self.assertRaises(URLError):
                    cached_imagenet_weights(self.url.replace("127.0.0.1", "localhost"), directory)
                result = cached_imagenet_weights(self.url, directory)
                self.assertEqual(result.read_bytes(), PAYLOAD)
                self.assertEqual([path.name for path in Path(directory).iterdir()], [FILENAME])
            with patch("model_download.download_context", side_effect=AssertionError("offline")):
                self.assertEqual(cached_imagenet_weights(self.url, directory), result)
        self.assertIs(ssl._create_default_https_context, original_context)

    def test_https_rejects_untrusted_root_then_recovers_with_trusted_ca(self):
        def context_factory():
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            if os.environ.get("SSL_CERT_FILE"):
                context.load_verify_locations(cafile=os.environ["SSL_CERT_FILE"])
            return context
        self.check_tls_recovery(context_factory)

    @unittest.skipUnless(HAS_NATIVE, "Native truststore dependencies required")
    def test_native_truststore_uses_operator_ca_and_keeps_hostname_verification(self):
        self.check_tls_recovery(download_context)


if __name__ == "__main__":
    unittest.main()
