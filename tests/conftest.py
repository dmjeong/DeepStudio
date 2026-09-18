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
