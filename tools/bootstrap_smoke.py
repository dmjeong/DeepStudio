"""CI의 실제 모델 패키지를 재사용하는 가상환경에서 자동 설치 경로를 검증한다."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="studio-bootstrap-smoke-") as folder:
        root = Path(folder)
        environment = root / "environment"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command = [str(python), str(ROOT / "start_web.py"), "--setup-only"]
        env = {**os.environ, "DEEP_STUDIO_STATE_DIR": str(root / "state"), "PYTHONUTF8": "1"}
        subprocess.run(command, env=env, check=True, timeout=180)
        marker = environment / ".deep-studio-web.json"
        log = environment / "deep-studio-setup.log"
        before = marker.read_bytes(), marker.stat().st_mtime_ns, log.read_bytes()
        subprocess.run(command, env=env, check=True, timeout=30)
        assert before == (marker.read_bytes(), marker.stat().st_mtime_ns, log.read_bytes())
        print("BOOTSTRAP_SMOKE_OK: actual runtime imports, pip check, cached second launch")


if __name__ == "__main__":
    try:
        main()
    except subprocess.SubprocessError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
