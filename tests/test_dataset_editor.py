"""데이터셋 편집에서 이미지-정답 연결과 실패 복구를 실제 파일로 검증."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from core.dataset_editor import edit_dataset, scan_dataset
from core.project import ProjectManager


class DatasetEditorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def project(self, task="classify"):
        project = ProjectManager.create_new("sample", task, str(self.root / task), ["good", "scratch", "dent"])
        ProjectManager.save(project)
        return project

    def image(self, folder, name="a.png"):
        path = Path(folder) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 14), "red").save(path)
        return path

    def test_import_reclass_move_delete_preserves_archive(self):
        p = self.project()
        source = self.image(self.root / "source")
        edit_dataset(p, "import", [str(source)], class_name="scratch")
        added = Path(p.data.train_dir) / "scratch" / source.name
        self.assertTrue(source.exists())
        self.assertEqual(scan_dataset(p)["splits"][0]["classes"]["scratch"], 1)
        edit_dataset(p, "reclass", [str(added)], class_name="dent")
        added = Path(p.data.train_dir) / "dent" / source.name
        edit_dataset(p, "move", [str(added)], target_split="val")
        moved = Path(p.data.val_dir) / "dent" / source.name
        result = edit_dataset(p, "delete", [str(moved)], split="val")
        self.assertFalse(moved.exists())
        self.assertTrue((Path(result["backup_path"]) / "manifest.json").is_file())
        self.assertTrue(source.exists())

    def test_nested_detection_reclass_only_changes_selected_class_and_moves_sidecar(self):
        p = self.project("detect")
        path = self.image(Path(p.data.train_dir) / "batch")
        label = Path(p.data.root) / "labels/train/batch/a.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("1 .5 .5 .2 .2\n0 .2 .2 .1 .1\n")
        edit_dataset(p, "reclass", [str(path)], class_name="dent", source_class="scratch")
        self.assertTrue(label.read_text().startswith("2 "))
        self.assertIn("\n0 ", label.read_text())
        edit_dataset(p, "move", [str(path)], target_split="test")
        self.assertTrue((Path(p.data.root) / "labels/test/batch/a.txt").exists())
        self.assertFalse(label.exists())
        self.assertEqual(scan_dataset(p)["splits"][2]["classes"]["dent"], 1)

    def test_rename_removes_old_class_directory_and_does_not_change_class_index(self):
        p = self.project()
        image = self.image(Path(p.data.train_dir) / "scratch" / "nested")
        edit_dataset(p, "rename_class", source_class="scratch", new_name="mark")
        self.assertFalse((Path(p.data.train_dir) / "scratch").exists())
        self.assertTrue((Path(p.data.train_dir) / "mark/nested" / image.name).is_file())
        self.assertEqual(p.data.class_names, ["good", "mark", "dent"])

    def test_collision_rolls_back_entire_batch(self):
        p = self.project()
        one = self.image(self.root / "input", "one.png")
        two = self.image(self.root / "input", "two.png")
        occupied = self.image(Path(p.data.train_dir) / "good", "two.png")
        before = Path(ProjectManager.get_active_filepath(p)).read_bytes()
        with self.assertRaises(FileExistsError):
            edit_dataset(p, "import", [str(one), str(two)], class_name="good")
        self.assertFalse((occupied.parent / "one.png").exists())
        self.assertTrue(occupied.exists())
        self.assertEqual(Path(ProjectManager.get_active_filepath(p)).read_bytes(), before)

    def test_invalid_annotation_restores_original_and_project(self):
        p = self.project("detect")
        path = self.image(p.data.train_dir)
        label = Path(p.data.root) / "labels/train/a.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 .5 .5 .2 .2\n")
        before = label.read_bytes()
        with self.assertRaises(ValueError):
            edit_dataset(p, "annotations", [str(path)], annotations=[{"class_id": 99, "coordinates": [.5, .5, .2, .2]}])
        self.assertEqual(label.read_bytes(), before)

    def test_mask_mapping_preserves_unselected_and_ignore_pixels(self):
        p = self.project("segment")
        path = self.image(p.data.train_dir)
        label = Path(p.data.root) / "masks/train/a.png"
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.array([[0, 1, 2, 255]], dtype=np.uint8)).save(label)
        edit_dataset(p, "reclass", [str(path)], source_class="scratch", class_name="dent")
        with Image.open(label) as mask:
            np.testing.assert_array_equal(np.array(mask), [[0, 2, 2, 255]])

    def test_project_save_failure_restores_files_and_class_names(self):
        p = self.project()
        image = self.image(Path(p.data.train_dir) / "scratch")
        with patch.object(ProjectManager, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                edit_dataset(p, "rename_class", source_class="scratch", new_name="mark")
        self.assertTrue(image.exists())
        self.assertEqual(p.data.class_names, ["good", "scratch", "dent"])
        self.assertFalse((Path(p.data.train_dir) / "mark").exists())

    def test_outside_dataset_and_shared_stem_rejected(self):
        p = self.project("detect")
        outside = self.image(self.root / "outside")
        with self.assertRaises(ValueError):
            edit_dataset(p, "delete", [str(outside)])
        self.image(p.data.train_dir, "a.jpg")
        with self.assertRaises(FileExistsError):
            edit_dataset(p, "import", [str(outside)])


if __name__ == "__main__":
    unittest.main()
