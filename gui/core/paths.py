"""
Deep Vision Studio — 경로 유틸리티

PyInstaller EXE와 개발 환경 모두에서
올바른 경로를 반환하는 중앙 모듈.

경로 해결 전략:
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  개발 환경 (python main.py):                                 │
│  ┌──────────────────────────────────┐                        │
│  │ __file__  = gui/core/paths.py    │                        │
│  │ GUI_DIR   = gui/                 │                        │
│  │ PYTHON_DIR= python/              │                        │
│  └──────────────────────────────────┘                        │
│                                                              │
│  PyInstaller EXE (frozen):                                   │
│  ┌──────────────────────────────────┐                        │
│  │ sys._MEIPASS = _internal/        │ ← 번들 리소스 폴더     │
│  │ GUI_DIR   = _internal/           │ ← app/, widgets/ 여기  │
│  │ PYTHON_DIR= _internal/python/    │ ← model.py 여기       │
│  └──────────────────────────────────┘                        │
│                                                              │
│  사용법:                                                     │
│    from core.paths import PYTHON_DIR, GUI_DIR                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
"""

import os
import sys


def _is_frozen() -> bool:
    """PyInstaller 번들 환경인지 확인"""
    return getattr(sys, "frozen", False)


def _get_bundle_dir() -> str:
    """
    PyInstaller 번들 리소스 폴더 반환

    frozen 환경에서 sys._MEIPASS 가 실제 리소스 경로.
    (--onedir 모드: _internal/ 폴더)
    """
    return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))


def _get_gui_dir() -> str:
    """GUI 루트 디렉토리 (app/, widgets/, core/ 상위)"""
    if _is_frozen():
        return _get_bundle_dir()
    # 개발 환경: core/paths.py → core/ → gui/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_python_dir() -> str:
    """
    python/ 모듈 디렉토리 (model.py, dataset.py 등)

    탐색 순서:
    1. 번들 내 python/ (PyInstaller --add-data로 포함)
    2. 프로젝트 구조 기준 ../python/ (개발 환경)
    """
    if _is_frozen():
        # 번들 내 python/ 폴더
        bundle_python = os.path.join(_get_bundle_dir(), "python")
        if os.path.isdir(bundle_python):
            return bundle_python
        # EXE 옆 python/ 폴더 (외부 배치 시)
        exe_python = os.path.join(
            os.path.dirname(sys.executable), "python"
        )
        if os.path.isdir(exe_python):
            return exe_python
        return bundle_python  # 없어도 경로 반환 (나중에 에러 발생)

    # 개발 환경: gui/ → DeepVisionStudio/ → python/
    gui_dir = _get_gui_dir()
    return os.path.join(os.path.dirname(gui_dir), "python")


def _get_exe_dir() -> str:
    """EXE 파일이 있는 디렉토리 (frozen 시) 또는 GUI 디렉토리"""
    if _is_frozen():
        return os.path.dirname(sys.executable)
    return _get_gui_dir()


# ── 공개 상수 ──────────────────────────────────────
# 모든 모듈에서 이 값을 import하여 사용
IS_FROZEN = _is_frozen()
GUI_DIR = _get_gui_dir()
PYTHON_DIR = _get_python_dir()
EXE_DIR = _get_exe_dir()
PROJECT_ROOT = os.path.dirname(GUI_DIR) if not IS_FROZEN else EXE_DIR


def ensure_python_path():
    """
    python/ 디렉토리를 sys.path에 추가 (중복 방지)

    main.py 에서 한 번만 호출하면 되지만,
    개별 모듈에서도 안전하게 호출 가능.
    """
    if PYTHON_DIR and PYTHON_DIR not in sys.path:
        sys.path.insert(0, PYTHON_DIR)
    if GUI_DIR and GUI_DIR not in sys.path:
        sys.path.insert(0, GUI_DIR)
    if PROJECT_ROOT and PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
