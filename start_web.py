"""최초 설치부터 브라우저 실행까지 처리하는 로컬 웹 시작점."""

import argparse
import os
from pathlib import Path
import subprocess
import struct
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Deep Vision Studio 자동 설치 및 실행")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--setup-only", action="store_true", help="환경 설치와 검사 후 종료")
    parser.add_argument("--accelerator", choices=("auto", "cuda", "cpu"), default="auto",
                        help="자동 GPU 감지 또는 CUDA 필수 설치")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 이상 필요. Python 3.11 64비트 설치 권장")
    if struct.calcsize("P") != 8:
        raise RuntimeError("64비트 Python 설치 필요. Python 3.11 64비트 설치 권장")
    if not 1024 <= args.port <= 65535:
        parser.error("port 범위: 1024–65535")
    if not args.setup_only and not (ROOT / "web" / "dist" / "index.html").is_file():
        raise RuntimeError("React 화면 빌드 없음. GitHub Releases의 실행용 Deep-Vision-Studio-React ZIP을 사용하세요.")
    from webapp.bootstrap import ensure_environment
    state_dir = Path(os.environ.get("DEEP_STUDIO_STATE_DIR") or Path.home() / ".deep-vision-studio-react").resolve()
    python = ensure_environment(ROOT, state_dir, accelerator=args.accelerator)
    if args.setup_only:
        print("설치와 검사 완료. start_web.bat 또는 python start_web.py로 실행하세요.", flush=True)
        return 0
    command = [str(python), str(ROOT / "run_web.py"), "--port", str(args.port)]
    if args.no_browser:
        command.append("--no-browser")
    return subprocess.call(command, cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        print(f"시작 실패: {exc}\n문제를 해결한 뒤 다시 실행하면 설치를 재시도합니다.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
