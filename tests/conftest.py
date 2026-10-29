"""
Deep Vision Studio 테스트 — 공통 Fixture

테스트 환경:
┌──────────────────────────────────────────────────┐
│  conftest.py                                     │
│  ├── sys.path에 gui/ 추가 (임포트 경로)           │
│  ├── sample_image: 테스트용 RGB 이미지 (100x80)   │
│  ├── gray_image : 테스트용 그레이스케일 이미지     │
│  └── tmp_project: 임시 ProjectData               │
└──────────────────────────────────────────────────┘
"""

import os
import sys
import pytest
import numpy as np

# gui/ 경로를 sys.path에 추가하여 core/, widgets/ 임포트 가능
GUI_DIR = os.path.join(os.path.dirname(__file__), "..", "gui")
sys.path.insert(0, os.path.abspath(GUI_DIR))

# python/ 경로 추가 (model.py 등)
PYTHON_DIR = os.path.join(os.path.dirname(__file__), "..", "python")
sys.path.insert(0, os.path.abspath(PYTHON_DIR))


@pytest.fixture
def sample_image():
    """테스트용 RGB 이미지 (100×80×3, uint8)"""
    np.random.seed(42)
    return np.random.randint(0, 256, (80, 100, 3), dtype=np.uint8)


@pytest.fixture
def gray_image():
    """테스트용 그레이스케일 이미지 (80×100, uint8)"""
    np.random.seed(42)
    return np.random.randint(0, 256, (80, 100), dtype=np.uint8)


@pytest.fixture
def tmp_project(tmp_path):
    """임시 프로젝트 (classify 태스크)"""
    from core.project import ProjectManager
    project = ProjectManager.create_new(
        name="TestProject",
        task="classify",
        project_dir=str(tmp_path / "test_proj"),
        class_names=["OK", "NG"],
    )
    return project


@pytest.fixture(autouse=True)
def _release_qt_keyboard_modifiers():
    """QTest 클릭에 준 Shift·Ctrl은 Qt의 '현재 수정키' 상태로 남아 다음 테스트로 새어 나간다.

    (예: QTableView.selectRow가 Shift를 보고 범위 선택을 한다.) 실제 키보드에서는 놓을 때
    해제되므로, 테스트가 끝날 때마다 수정키를 놓은 상태로 되돌린다. Qt를 쓰지 않은 테스트는 건너뛴다.
    """
    yield
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None or app.keyboardModifiers() == Qt.KeyboardModifier.NoModifier:
        return
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QWidget
    target = QWidget()
    for key in (Qt.Key.Key_Shift, Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
        QTest.keyRelease(target, key)
    target.deleteLater()


@pytest.fixture(autouse=True)
def _isolated_weight_folders(tmp_path_factory, monkeypatch):
    """사용자 가중치 폴더와 torch 캐시를 테스트마다 비운다 (개발 PC에 받아 둔 파일이 결과를 바꾸지 않게)."""
    if "DVS_USER_WEIGHTS_DIR" not in os.environ:
        monkeypatch.setenv("DVS_USER_WEIGHTS_DIR", str(tmp_path_factory.mktemp("user_weights")))
    yield
