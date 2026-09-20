"""Detection's desktop entry point creates usable, per-object normalized box labels."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'gui'), str(ROOT / 'python')]
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Qt runtime required')
class DetectionAnnotationTests(unittest.TestCase):
    def test_draw_change_class_save_reload_and_failure(self):
        from PySide6.QtWidgets import QApplication
        from core.project import ProjectManager
        from core.class_management import ClassManager
        from core.dataset_editor import edit_dataset, read_annotations
        from widgets.obb_annotation import OBBAnnotationDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            project = ProjectManager.create_new('detect', 'detect', directory, ['part'])
            path = Path(project.data.train_dir) / 'image.png'
            Image.new('RGB', (800, 400)).save(path)
            attempts = []

            def save(rows):
                attempts.append(rows)
                if len(attempts) == 1:
                    raise OSError('save failed')
                edit_dataset(project, 'annotations', [str(path)], annotations=rows)

            def add_class(name):
                ClassManager.add(project, name)
                return project.data.class_names

            dialog = OBBAnnotationDialog(str(path), ['part'], [], save,
                                         task='detect', add_class=add_class)
            dialog.show()
            app.processEvents()
            with patch('widgets.obb_annotation.QInputDialog.getText', return_value=('dent', True)):
                dialog.add_class_button.click()
            self.assertEqual(dialog.names, ['part', 'dent'])
            self.assertEqual(project.data.class_names, ['part', 'dent'])
            for x, y in [(600, 300), (200, 100)]:
                dialog._add_point(x, y)
            self.assertEqual(dialog.rows, [{'class_id': 1, 'coordinates': [.5, .5, .5, .5]}])
            # A zero-area box cannot become a training label.
            dialog._add_point(20, 20)
            self.assertFalse(dialog.save_button.isEnabled())
            dialog._add_point(20, 100)
            self.assertEqual(len(dialog.rows), 1)
            dialog.table.cellWidget(0, 1).setCurrentIndex(0)
            dialog.save_button.click()
            dialog.worker.wait(10000)
            app.processEvents()
            self.assertIn('save failed', dialog.status.text())
            self.assertTrue(dialog.isVisible())
            dialog.save_button.click()
            dialog.worker.wait(10000)
            app.processEvents()
            label = Path(project.data.root) / 'labels/train/image.txt'
            rows = read_annotations(label, 'detect', 2)
            self.assertEqual(rows, [{'class_id': 0, 'coordinates': [.5, .5, .5, .5]}])
            reopened = OBBAnnotationDialog(str(path), project.data.class_names, rows, lambda rows: None, task='detect')
            self.assertEqual(reopened.table.rowCount(), 1)
            reopened.table.cellWidget(0, 2).click()
            self.assertEqual(reopened.rows, [])
            reopened.dirty = False
            reopened.close()
            dialog.deleteLater()
            reopened.deleteLater()
            app.processEvents()

    def test_dataset_import_visible_and_edit_button_opens_detection_editor(self):
        from PySide6.QtWidgets import QApplication, QDialog
        from core.project import ProjectManager
        from widgets.dataset_widget import DatasetWidget
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            project = ProjectManager.create_new('detect', 'detect', str(Path(directory) / 'project'), ['part'])
            source = Path(directory) / 'new.png'
            Image.new('RGB', (100, 60)).save(source)
            with patch.object(DatasetWidget, '_scan_all'):
                page = DatasetWidget()
                page.set_project(project)
                self.assertFalse(page.annotation_row.isHidden())
                page._set_shared_class_filter('part')
                page._copy_images_to_project([str(source)])
                self.assertEqual(page._class_filter, '전체')
                image = Path(project.data.train_dir) / source.name
                self.assertTrue(image.is_file())
                page._selected_paths = {str(image)}
                with patch('widgets.obb_annotation.OBBAnnotationDialog') as dialog:
                    dialog.return_value.exec.return_value = QDialog.DialogCode.Rejected
                    page.annotate_btn.click()
                    self.assertEqual(dialog.call_args.kwargs['task'], 'detect')
                    self.assertEqual(dialog.call_args.args[0], str(image))
                page.close()
                page.deleteLater()
                app.processEvents()

    def test_mouse_coordinates_after_zoom_create_correct_box(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication
        from widgets.obb_annotation import OBBAnnotationDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (800, 400)).save(path)
            dialog = OBBAnnotationDialog(str(path), ['part'], [], lambda rows: None, task='detect')
            dialog.show()
            app.processEvents()
            dialog.zoom.setValue(150)
            app.processEvents()
            start = dialog.view.mapFromScene(QPointF(350, 170))
            end = dialog.view.mapFromScene(QPointF(450, 230))
            QTest.mousePress(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseMove(dialog.view.viewport(), end)
            QTest.mouseRelease(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=end)
            app.processEvents()
            self.assertEqual(len(dialog.rows), 1)
            for actual, expected in zip(dialog.rows[0]['coordinates'], [.5, .5, .125, .15]):
                self.assertAlmostEqual(actual, expected, delta=.005)
            dialog.dirty = False
            dialog.close()
            dialog.deleteLater()
            app.processEvents()

    def test_drag_resize_move_bounds_undo_redo_and_selection(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication
        from widgets.obb_annotation import OBBAnnotationDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (800, 400)).save(path)
            dialog = OBBAnnotationDialog(str(path), ['part', 'dent'], [], lambda rows: None, task='detect')
            dialog.show()
            app.processEvents()

            def drag(a, b):
                view = dialog.view
                start = view.mapFromScene(QPointF(*a))
                end = view.mapFromScene(QPointF(*b))
                QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
                QTest.mouseMove(view.viewport(), end)
                QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
                app.processEvents()

            drag((200, 100), (400, 250))
            self.assertEqual(len(dialog.rows), 1)
            self.assertEqual(dialog.selected, 0)
            original = list(dialog.rows[0]['coordinates'])
            # Bottom-right handle resizes in place, rather than creating another box.
            drag((400, 250), (500, 300))
            self.assertEqual(len(dialog.rows), 1)
            self.assertGreater(dialog.rows[0]['coordinates'][2], original[2])
            resized = list(dialog.rows[0]['coordinates'])
            QTest.keyClick(dialog.view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
            self.assertEqual(dialog.rows[0]['coordinates'], original)
            dialog.redo_button.click()
            self.assertEqual(dialog.rows[0]['coordinates'], resized)
            dialog.mode_buttons['select'].click()
            drag((330, 180), (0, 0))
            x, y, w, h = dialog.rows[0]['coordinates']
            self.assertGreaterEqual(x-w/2, -1e-8)
            self.assertGreaterEqual(y-h/2, -1e-8)
            self.assertAlmostEqual(w, resized[2])
            self.assertAlmostEqual(h, resized[3])
            dialog.table.cellWidget(0, 1).setCurrentIndex(1)
            self.assertEqual(dialog.rows[0]['class_id'], 1)
            # Erase mode deletes an object even when the click lands on a resize handle.
            dialog.mode_buttons['erase'].click()
            rect = dialog.view._rect(dialog.rows[0])
            handle = dialog.view.mapFromScene(rect.bottomRight())
            QTest.mouseClick(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=handle)
            self.assertFalse(dialog.rows)
            dialog.undo_button.click()
            self.assertEqual(dialog.rows[0]['class_id'], 1)
            dialog._select_box(0)
            dialog.view.setFocus()
            QTest.keyClick(dialog.view, Qt.Key.Key_Delete)
            self.assertFalse(dialog.rows)
            dialog.undo_button.click()
            self.assertEqual(dialog.rows[0]['class_id'], 1)
            # Escape during creation discards only the live preview.
            dialog.mode_buttons['draw'].click()
            start = dialog.view.mapFromScene(QPointF(600, 200))
            end = dialog.view.mapFromScene(QPointF(700, 300))
            QTest.mousePress(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseMove(dialog.view.viewport(), end)
            QTest.keyClick(dialog.view, Qt.Key.Key_Escape)
            QTest.mouseRelease(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=end)
            self.assertEqual(len(dialog.rows), 1)
            dialog.dirty = False
            dialog.close()
            dialog.deleteLater()
            app.processEvents()
