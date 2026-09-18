"""호스트 OS와 Git 체크아웃 설정에 관계없이 배포 배치 파일의 바이트를 검증한다."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.package_web import packaged_bytes, validate_windows_launcher


class WebPackageTests(unittest.TestCase):
    def test_linux_and_windows_sources_produce_the_same_crlf_launcher(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "start_web.bat"
            expected = b"@echo off\r\ngoto finish\r\n:finish\r\nexit /b 0\r\n"
            for source in (expected, expected.replace(b"\r\n", b"\n")):
                path.write_bytes(source)
                result = packaged_bytes(path)
                self.assertEqual(result, expected)
                validate_windows_launcher(result)

    def test_invalid_launcher_is_rejected_before_publication(self):
        for content in (b"@echo off\n", b"\xef\xbb\xbf@echo off\r\n", "echo 오류\r\n".encode("utf-8")):
            with self.assertRaises(RuntimeError):
                validate_windows_launcher(content)


if __name__ == "__main__":
    unittest.main()
