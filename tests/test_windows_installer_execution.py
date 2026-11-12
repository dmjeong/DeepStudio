"""Execute a real, small MSI with Unicode/spaces before freezing the full app."""
import hashlib
import os
from pathlib import Path
import shutil

import pytest

from tools.windows_desktop_smoke import run_msi


@pytest.mark.skipif(os.name != "nt" or not os.environ.get("DVS_TEST_INSTALLER_MSI"),
                    reason="Requires the Windows CI's real WiX test MSI")
def test_msi_installs_and_removes_from_paths_with_spaces(tmp_path):
    source = Path(os.environ["DVS_TEST_INSTALLER_MSI"])
    media = tmp_path / "한글 설치 미디어"
    media.mkdir()
    for path in (source, *source.parent.glob("*.cab")):
        shutil.copyfile(path, media / path.name)
    msi = media / source.name
    destination = tmp_path / "한글 앱 설치 폴더"
    log = tmp_path / "한글 설치 로그.txt"
    executable = destination / "app" / "DeepVisionStudio.exe"
    try:
        run_msi(msi, log, install=destination)
        assert log.is_file()
        expected = Path(os.environ["SystemRoot"]) / "System32" / "where.exe"
        assert hashlib.sha256(executable.read_bytes()).digest() == hashlib.sha256(expected.read_bytes()).digest()
    finally:
        if executable.is_file():
            run_msi(msi, tmp_path / "한글 제거 로그.txt")
    assert not executable.exists()
