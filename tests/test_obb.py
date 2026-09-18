"""Rotated labels remain aligned through editing, cropping and CPU inference."""
import importlib.util
import math
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from obb import validate_corners, rectangle_from_three_points, crop_obb_row
from core.project import ProjectManager
from core.dataset_editor import edit_dataset, read_annotations, scan_dataset
from core.model_selection import selection_policy
from crop_dataset import prepare_crop_dataset
from spatial_data import read_spatial_labels, validate_spatial_dataset




class OBBGeometryTests(unittest.TestCase):
    def test_three_click_rectangle_uses_pixels_on_non_square_image(self):
        coords = rectangle_from_three_points([.2, .2, .5, .4, .3, .65], (200, 100))
        points = np.array(coords).reshape(4, 2) * [200, 100]
        a, b = points[1] - points[0], points[2] - points[1]
        self.assertAlmostEqual(float(a @ b), 0, places=8)
        np.testing.assert_allclose(points[0] + points[2], points[1] + points[3])
        with self.assertRaises(ValueError):
            rectangle_from_three_points([.1, .1, .9, .9, .9, .1], (100, 100))
        with self.assertRaises(ValueError):
            rectangle_from_three_points([.1, .1, .5, .5, .2, .2], (100, 100))

    def test_crossed_repeated_concave_outside_and_axis_box_labels_rejected(self):
        for coords in ([.2,.2,.8,.8,.8,.2,.2,.8], [.2,.2,.8,.2,.8,.2,.2,.8],
                       [.1,.1,.9,.1,.2,.2,.1,.9], [.1,.1,1.1,.1,.8,.8,.1,.8],
                       [.5,.5,.2,.2], [math.nan]*8):
            with self.subTest(coords=coords), self.assertRaises(ValueError):
                validate_corners(coords)
        validate_corners([.2,.2,.8,.2,.8,.8,.2,.8])
        validate_corners([.2,.8,.8,.8,.8,.2,.2,.2])

    def test_crop_preserves_corners_rejects_partial_and_drops_disjoint(self):
        row = [2,.4,.3,.6,.4,.55,.6,.35,.5]
        result = crop_obb_row(row, (200,100), (50,20,150,80))
        np.testing.assert_allclose(np.array(result[1:]).reshape(4,2) * [100,60] + [50,20],
                                   np.array(row[1:]).reshape(4,2) * [200,100])
        with self.assertRaisesRegex(ValueError, "크롭 경계"):
            crop_obb_row(row, (200,100), (90,20,150,80))
        self.assertIsNone(crop_obb_row(row, (200,100), (0,0,30,20)))
        # Envelopes overlap, but a thin diagonal misses the ROI.
        diagonal = [0,0,.7,.7,0,.8,.1,.1,.8]
        self.assertIsNone(crop_obb_row(diagonal, (100,100), (0,0,20,20)))



class OBBDatasetTests(unittest.TestCase):
    def test_roundtrip_edit_move_class_remap_and_atomic_invalid_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = ProjectManager.create_new("obb", "obb", str(root/"project"), ["part", "mark"])
            ProjectManager.save(project)
            from core.training_modes import validate_training_options
            with self.assertRaisesRegex(ValueError, "학습 엔진"):
                validate_training_options(project)
            with self.assertRaisesRegex(ValueError, "학습 엔진"):
                selection_policy(project.training, "custom", "obb")
            source = root/"input.png"
            Image.new("RGB", (200,100)).save(source)
            corners = [.2,.2,.6,.3,.55,.7,.15,.6]
            source.with_suffix(".txt").write_text("1 " + " ".join(map(str,corners)) + "\n")
            edit_dataset(project, "import", [str(source)])
            image = Path(project.data.train_dir)/source.name
            self.assertEqual(scan_dataset(project)["images"][0]["classes"], ["mark"])
            label = Path(project.data.root)/"labels/train/input.txt"
            original = label.read_bytes()
            with self.assertRaises(ValueError):
                edit_dataset(project, "annotations", [str(image)], annotations=[{"class_id":1,"coordinates":[.5]*4}])
            self.assertEqual(label.read_bytes(), original)
            edit_dataset(project, "annotations", [str(image)], annotations=[{"class_id":0,"coordinates":corners}])
            np.testing.assert_allclose(read_annotations(label,"obb",2)[0]["coordinates"], corners)
            edit_dataset(project, "move", [str(image)], target_split="val")
            moved = Path(project.data.root)/"labels/val/input.txt"
            self.assertFalse(label.exists())
            np.testing.assert_allclose(read_spatial_labels(moved,"obb",2)[0], [0,*corners])
            from core.class_management import ClassManager
            ClassManager.delete(project, "mark")
            self.assertEqual(project.data.class_names, ["part"])
            np.testing.assert_allclose(read_spatial_labels(moved,"obb",1)[0], [0,*corners])
            loaded = ProjectManager.load(ProjectManager.get_active_filepath(project))
            self.assertEqual(loaded.task, "obb")

    def test_snapshot_corners_match_pixels_and_failed_crop_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = ProjectManager.create_new("obb", "obb", str(root/"project"), ["part"])
            pixels = np.random.default_rng(5).integers(0,255,(100,200,3),dtype=np.uint8)
            for split in ("train","val"):
                Image.fromarray(pixels).save(Path(project.data.root)/f"images/{split}/a.png")
                (Path(project.data.root)/f"labels/{split}/a.txt").write_text("0 .4 .3 .6 .4 .55 .6 .35 .5\n")
            self.assertEqual(validate_spatial_dataset(project.data.root,"obb",1)["train"]["objects"],1)
            crop = {"width":100,"height":60}
            snapshot = Path(prepare_crop_dataset(project.data.root,root/"generated","obb",crop,native=True))
            with Image.open(snapshot/"images/train/a.png") as image:
                np.testing.assert_array_equal(np.asarray(image),pixels[20:80,50:150])
            coords = np.array(read_spatial_labels(snapshot/"labels/train/a.txt","obb",1)[0][1:]).reshape(4,2)
            np.testing.assert_allclose(coords*[100,60]+[50,20], [[80,30],[120,40],[110,60],[70,50]], atol=1e-5)
            before = set((root/"generated").iterdir())
            with self.assertRaisesRegex(ValueError,"크롭 경계"):
                prepare_crop_dataset(project.data.root,root/"generated","obb",{"width":20,"height":20},native=True)
            self.assertEqual(set((root/"generated").iterdir()),before)


RUNTIME = all(importlib.util.find_spec(name) for name in ("torch","PySide6","onnx","onnxruntime"))


@unittest.skipUnless(RUNTIME, "Actual training, Qt and ONNX dependencies required")
class OBBRuntimeTests(unittest.TestCase):

    def test_qt_annotation_draw_class_change_and_save(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication
        from widgets.obb_annotation import OBBAnnotationDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"image.png"
            Image.new("RGB",(800,600)).save(path)
            saved = []
            attempts = []
            def save(rows):
                attempts.append(rows)
                if len(attempts) == 1:
                    raise OSError("강제 저장 실패")
                saved.extend(rows)
            dialog = OBBAnnotationDialog(str(path),["part","mark"],[],save)
            dialog.resize(760,600)
            dialog.show()
            app.processEvents()
            def click(x, y):
                QTest.mouseClick(dialog.view.viewport(), Qt.MouseButton.LeftButton,
                                 pos=dialog.view.mapFromScene(QPointF(x, y)))
                app.processEvents()
            for x,y in [(160,120),(400,200),(240,390),(480,90),(680,180),(530,380)]:
                click(x,y)
            self.assertEqual(len(dialog.rows),2)
            dialog.table.cellWidget(0,2).click()
            self.assertEqual(len(dialog.rows),1)
            dialog.table.cellWidget(0,1).setCurrentIndex(1)
            click(80,80)
            self.assertFalse(dialog.save_button.isEnabled())
            dialog.cancel_draw.click()
            dialog.zoom.setValue(300)
            app.processEvents()
            self.assertGreater(dialog.view.horizontalScrollBar().maximum(),0)
            self.assertGreater(dialog.view.verticalScrollBar().maximum(),0)
            dialog.save_button.click()
            dialog.worker.wait(10000)
            app.processEvents()
            self.assertTrue(dialog.isVisible())
            self.assertTrue(dialog.isEnabled())
            self.assertIn("강제 저장 실패",dialog.status.text())
            self.assertEqual(len(dialog.rows),1)
            dialog.save_button.click()
            dialog.worker.wait(10000)
            app.processEvents()
            self.assertEqual(saved[0]["class_id"],1)
            self.assertEqual(len(saved[0]["coordinates"]),8)
            self.assertEqual(len(attempts),2)
            self.assertFalse(dialog.isVisible())
            dialog.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
