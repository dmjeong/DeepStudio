"""로컬 전용 웹 실행기. 프로젝트 폴더와 무관하게 같은 사용자 상태를 복원한다."""

import argparse
import importlib.util
import os
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent


def open_when_ready(url):
    for _ in range(100):
        try:
            with urllib.request.urlopen(url + "api/state", timeout=.5) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(.2)


def main():
    parser = argparse.ArgumentParser(description="Deep Vision Studio - local React edition")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port 범위: 1024–65535")
    missing = [name for name in ("fastapi", "uvicorn", "torch", "PIL", "numpy") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("필요한 패키지 없음: " + ", ".join(missing) + "\nstart_web.bat 또는 python start_web.py로 자동 설치 후 실행하세요.")
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        raise RuntimeError("React 화면 빌드 없음. 릴리스의 Deep-Vision-Studio-React ZIP을 사용하거나 web 폴더에서 npm ci && npm run build 실행 필요")
    state_dir = Path(os.environ.get("DEEP_STUDIO_STATE_DIR") or Path.home() / ".deep-vision-studio-react").resolve()
    from webapp.locking import exclusive_file
    with exclusive_file(state_dir / "server.lock", "이미 실행 중인 웹 서버를 먼저 종료하세요."):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", args.port))
            except OSError as exc:
                raise RuntimeError(f"포트 {args.port} 사용 중. --port 옵션으로 변경하세요.") from exc
        from webapp.server import create_app
        import uvicorn
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Deep Vision Studio: {url}\n종료: Ctrl+C. 학습 중에는 이 창을 유지하세요.", flush=True)
        if not args.no_browser:
            threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
        uvicorn.run(create_app(state_dir), host="127.0.0.1", port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
