"""
Deep Vision Studio — 메인 엔트리포인트

실행 방법:
    cd gui
    python main.py

또는:
    python -m gui.main

시스템 요구사항:
    • Windows 11 (권장)
    • Python 3.9+
    • PySide6 6.5+
    • PyTorch 2.0+
    • matplotlib (차트용)

아키텍처:
┌──────────────────────────────────────────────────┐
│  main.py                                          │
│  └─► QApplication                                 │
│      └─► MainWindow                               │
│          ├── Sidebar (네비게이션)                   │
│          ├── ProjectWidget                        │
│          ├── DatasetWidget                        │
│          ├── TrainingWidget                       │
│          ├── InferenceWidget                      │
│          ├── ExportWidget                         │
│          └── DefectGenWidget                      │
└──────────────────────────────────────────────────┘
"""

import sys
import os
import multiprocessing

# PyInstaller worker가 GUI 진입점을 다시 실행하지 않도록 먼저 분기한다.
if __name__ == "__main__":
    multiprocessing.freeze_support()

# ── 경로 부트스트랩 ─────────────────────────────────
# PyInstaller frozen 환경에서 __file__이 올바르지 않으므로
# sys._MEIPASS 기반으로 경로를 설정해야 한다.
# core/paths.py를 import하기 전에 최소한의 경로 설정이 필요.
#
# 흐름:
# ┌─────────────────────────────────────────────────────────┐
# │  frozen?  ──► sys._MEIPASS → sys.path에 추가            │
# │  아니면   ──► __file__ 기준 gui/ → sys.path에 추가      │
# │  → core.paths import 가능해짐                           │
# │  → ensure_python_path()로 python/ 모듈도 사용 가능      │
# └─────────────────────────────────────────────────────────┘
if getattr(sys, "frozen", False):
    # PyInstaller EXE: 번들 리소스 폴더를 path에 추가
    _bundle = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    if _bundle not in sys.path:
        sys.path.insert(0, _bundle)
else:
    # 개발 환경: gui/ 폴더를 path에 추가
    _gui = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_gui)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    if _gui not in sys.path:
        sys.path.insert(0, _gui)

# windowed EXE에서도 모델 라이브러리가 stdout/stderr를 사용할 수 있도록 연결.
if sys.stdout is None or sys.stderr is None:
    _log = os.path.join(os.path.expanduser("~"), "deep-studio-desktop.log")
    if len(sys.argv) > 2 and sys.argv[1] == "--studio-worker":
        _log = os.path.join(os.path.dirname(sys.argv[2]), "console.log")
    _stream = open(_log, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _stream
    sys.stderr = sys.stderr or _stream

from core.console_encoding import configure_console_output
configure_console_output()

# 이제 core.paths를 안전하게 import 가능
from core.paths import GUI_DIR, PYTHON_DIR, ensure_python_path
from core.version import APP_NAME, APP_VERSION

# python/ 모듈(model.py, dataset.py 등) 경로 등록
ensure_python_path()

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--studio-worker":
    from webapp.worker import run_job
    run_job(sys.argv[2])
    sys.exit(0)

# 자동 배포 검증의 예외가 windowed 오류 창에 갇히지 않도록 종료 코드를 남긴다.
if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--smoke-test":
    try:
        from tools.desktop_package_smoke import run
        run(sys.argv[2])
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from app.main_window import MainWindow


def load_stylesheet() -> str:
    """
    다크 테마 QSS 로드

    QSS 의 url() 은 실행 시 작업 디렉토리를 기준으로 해석되어
    실행 위치에 따라 아이콘이 깨진다. 이를 막기 위해 QSS 안의
    {ICON_DIR} 자리표시자를 아이콘 폴더의 절대경로로 치환한다.
    (Qt 는 경로 구분자로 '/' 를 쓰므로 Windows 에서도 슬래시로 변환)
    """
    qss_path = os.path.join(GUI_DIR, "resources", "styles", "dark_theme.qss")
    if not os.path.isfile(qss_path):
        return ""

    with open(qss_path, "r", encoding="utf-8") as f:
        qss = f.read()

    icon_dir = os.path.join(GUI_DIR, "resources", "icons").replace("\\", "/")
    return qss.replace("{ICON_DIR}", icon_dir)


def main():
    """애플리케이션 메인"""
    # High DPI 지원 (Windows 11)
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    app = QApplication(sys.argv)

    # ── 폰트 설정 ──
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    # ── 다크 테마 적용 ──
    stylesheet = load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    # ── 메인 윈도우 ──
    window = MainWindow()
    window.show()

    print("=" * 50)
    print(f"  {APP_NAME} v{APP_VERSION}")
    print("=" * 50)
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    print(f"  GUI Dir:  {GUI_DIR}")
    print(f"  Model Dir: {PYTHON_DIR}")
    print("=" * 50)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
