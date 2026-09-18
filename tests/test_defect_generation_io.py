"""실제 PNG와 별도 프로세스, Qt 검수 저장 및 캐시 복원 검증."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'gui'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
RUNTIME = all(importlib.util.find_spec(name) for name in ('cv2', 'PySide6', 'psutil'))


@unittest.skipUnless(RUNTIME, 'OpenCV와 Qt 실행 환경 필요')
class DefectGenerationIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from core.project import ProjectManager
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.environment = patch.dict(os.environ, {'DEEP_STUDIO_DESKTOP_STATE_DIR': str(self.root / 'state')})
        self.manager_patch = patch('core.job_manager._manager', None)
        self.environment.start()
        self.manager_patch.start()
        self.project = ProjectManager.create_new('한글 프로젝트', 'anomaly', str(self.root / 'project'), ['good', 'defect'])
        ProjectManager.save(self.project)
        self.source = Path(self.project.data.train_dir) / 'good' / '정상_16비트.png'
        self.original = (np.arange(64 * 80).reshape(64, 80) * 10).astype(np.uint16)
        Image.fromarray(self.original).save(self.source)

    def tearDown(self):
        from core.job_manager import desktop_manager
        desktop_manager().close()
        self.manager_patch.stop()
        self.environment.stop()
        self.app.processEvents()
        self.temp.cleanup()

    def wait_widget(self, widget, timeout=30):
        deadline = time.monotonic() + timeout
        while widget._worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNone(widget._worker, '공통 프로세스 작업 완료')
        self.assertEqual(widget._job_result['status'], 'completed', widget.progress_label.text())

    def test_preview_saved_without_regeneration_and_recovered_after_reopen(self):
        from PySide6.QtCore import Qt
        from core.defect_io import read_image
        from core.job_manager import desktop_manager
        from core.defect_workflow import candidate
        from widgets.legacy_defect_gen_widget import DefectGenWidget
        widget = DefectGenWidget()
        self.addCleanup(widget.deleteLater)
        widget.set_project(self.project)
        widget.seed_spin.setValue(4294967295)
        self.assertEqual(widget._get_params().seed, 4294967295)
        widget._start_generation(preview=True)
        self.wait_widget(widget)
        job_id = widget._job_id
        row = candidate(desktop_manager().directory(job_id), '000001', verify=True)
        preview = read_image(row['image'])
        self.assertEqual(preview.dtype, np.uint16)
        self.assertEqual(widget.result_list.count(), 1)
        self.source.unlink()  # 원본이 사라져도 선택한 캐시 자체를 저장해야 한다.
        widget.result_list.item(0).setCheckState(Qt.CheckState.Checked)
        widget._publish()
        self.wait_widget(widget)
        folder = Path(widget._job_result['output']['output_dir'])
        manifest = json.loads((folder / 'manifest.json').read_text(encoding="utf-8"))
        stored = manifest['samples'][0]
        np.testing.assert_array_equal(read_image(folder / stored['image']), preview)
        np.testing.assert_array_equal(read_image(folder / stored['original']), self.original)
        self.assertFalse(any(Path(self.project.data.test_dir).rglob('*.png')))
        reopened = DefectGenWidget()
        self.addCleanup(reopened.deleteLater)
        reopened.set_project(self.project)
        self.assertEqual(reopened.result_list.count(), 1)
        self.assertEqual(reopened._job_id, job_id)

    def test_qt_preview_preserves_uint16_brightness(self):
        from PySide6.QtWidgets import QLabel
        from PySide6.QtGui import QImage
        from widgets.legacy_defect_gen_widget import DefectGenWidget
        widget = DefectGenWidget()
        self.addCleanup(widget.deleteLater)
        label = QLabel()
        label.resize(80, 64)
        widget._display_image(label, self.original)
        displayed = label.pixmap().toImage().convertToFormat(QImage.Format.Format_RGB888)
        self.assertLessEqual(abs(displayed.pixelColor(79, 63).red() - round(int(self.original[-1, -1]) / 257)), 1)


if __name__ == '__main__':
    unittest.main()
