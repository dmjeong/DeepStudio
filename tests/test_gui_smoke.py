"""Qt가 설치된 CI에서 실제 페이지 생성과 저장/로드 연결을 검사한다."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "python"))
GUI_AVAILABLE = all(importlib.util.find_spec(name) is not None
                    for name in ("PySide6", "torch", "torchvision", "cv2", "matplotlib"))


@unittest.skipUnless(GUI_AVAILABLE, "Qt와 모델 실행 의존성 필요")
class GuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from app.main_window import MainWindow
        self.temp = tempfile.TemporaryDirectory()
        self.window = MainWindow()

    def tearDown(self):
        self.window.deleteLater()
        self.application.processEvents()
        self.temp.cleanup()

    def test_real_gui_save_and_reload_preserves_form_values(self):
        from core.project import ProjectManager
        project = ProjectManager.create_new(
            "검사 프로젝트", "classify", str(Path(self.temp.name) / "project"), ["NG", "OK"])
        path = ProjectManager.save(project)
        self.assertTrue(self.window.set_project(project))
        self.window.training_page.input_size_spin.setValue(640)
        self.window._save_current_project()
        restored = ProjectManager.load(path)
        self.assertEqual(restored.training.input_size, 640)
        self.assertTrue(self.window.set_project(restored))
        self.assertEqual(self.window.training_page.input_size_spin.value(), 640)

    def test_anomaly_page_uses_custom_engine_without_external_engine_selection(self):
        from core.project import ProjectManager
        project = ProjectManager.create_new(
            "이상 검사", "anomaly", str(Path(self.temp.name) / "anomaly"))
        self.assertTrue(self.window.set_project(project))
        page = self.window.training_page
        self.assertEqual(page.mode_combo.currentData(), "custom")
        self.assertFalse(page.mode_combo.isEnabled())
        self.window._save_current_project()
        self.assertEqual(project.training.training_mode, "custom")

    def test_each_training_mode_restores_by_value_and_transfer_keeps_weights(self):
        from core.project import ProjectManager
        project = ProjectManager.create_new("모드 복원", "classify", str(Path(self.temp.name) / "modes"), ["OK", "NG"])
        page = self.window.training_page
        for mode in ("custom", "efficientnet_resume", "efficientnet_transfer", "efficientnet_finetune"):
            project.training.training_mode = mode
            project.model.pretrained_weights = "D:/models/my-best.pt" if mode == "efficientnet_transfer" else ""
            page.set_project(project)
            self.assertEqual(page.mode_combo.currentData(), mode)
            self.assertEqual(page.epochs_spin.isEnabled(), mode != "efficientnet_resume")
            self.assertTrue(page.device_combo.isEnabled())
            if mode == "efficientnet_transfer":
                self.assertEqual(page.resume_edit.text(), "D:/models/my-best.pt")
                page.collect_config()
                self.assertEqual(project.model.pretrained_weights, "D:/models/my-best.pt")

    def test_class_change_signal_refreshes_class_definition(self):
        from core.project import ProjectManager
        project = ProjectManager.create_new(
            "분류 검사", "classify", str(Path(self.temp.name) / "classes"), ["A", "B", "C"])
        ProjectManager.save(project)
        self.assertTrue(self.window.set_project(project))
        page = self.window.dataset_page
        if not hasattr(page, "project_changed"):
            self.skipTest("클래스 변경 기능 이전 버전")
        project.data.class_names = ["A", "C"]
        project.data.num_classes = 2
        ProjectManager.save(project)
        page.project_changed.emit(project)
        self.assertEqual(self.window.training_page.project.data.class_names, ["A", "C"])
        self.assertEqual(self.window.inference_page.project.data.num_classes, 2)

    def test_obb_project_selection_shows_class_guidance(self):
        page = self.window.project_page
        page._on_task_selected("obb")
        self.assertTrue(page.class_edit.isEnabled())
        self.assertIn("회전 박스", page.class_hint.text())
        self.assertIn("scratch", page.class_hint.text())


if __name__ == "__main__":
    unittest.main()
