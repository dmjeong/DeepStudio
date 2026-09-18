"""PatchCore 폴더 선택 및 프로젝트 가중치 경로 회귀 검사."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "gui"))
from patchcore_data import discover_data
from core.project import ProjectManager


class PatchCoreDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def file(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"discovery only")
        return str(path)

    def test_empty_validation_does_not_block_normal_only_training(self):
        normal = self.file("train/good/part.PNG")
        self.file("train/defect/excluded.png")
        (self.root / "val/good").mkdir(parents=True)
        (self.root / "test").mkdir()
        self.assertEqual(discover_data(self.root), ([normal], [], [], ""))

    def test_empty_val_uses_test_with_top_level_labels(self):
        self.file("train/OK/a.png")
        (self.root / "val/good").mkdir(parents=True)
        normal = self.file("test/normal/defect_in_filename.png")
        defect = self.file("test/scratch/normal_in_filename.png")
        _, files, labels, split = discover_data(self.root)
        self.assertEqual(dict(zip(files, labels)), {normal: 0, defect: 1})
        self.assertEqual(split, "test")

    def test_nonempty_val_is_preferred_to_test(self):
        self.file("train/good/a.png")
        val = self.file("val/good/a.png")
        self.file("test/good/b.png")
        self.assertEqual(discover_data(self.root)[1:], ([val], [0], "val"))

    def test_flat_normal_folder_and_known_normal_folder(self):
        image = self.file("direct/a.png")
        self.assertEqual(discover_data(self.root / "direct")[0], [image])
        image = self.file("normal/nested/a.png")
        self.assertEqual(discover_data(self.root / "normal")[0], [image])

    def test_unknown_training_classes_are_not_silently_normal(self):
        self.file("train/scratch/a.png")
        with self.assertRaisesRegex(ValueError, "정상 학습 폴더"):
            discover_data(self.root)

    def test_unlabeled_validation_reports_file(self):
        self.file("train/good/a.png")
        self.file("val/a.png")
        with self.assertRaisesRegex(ValueError, "a.png"):
            discover_data(self.root)

    def test_hidden_files_are_excluded(self):
        self.file("train/good/.cache/a.png")
        self.file("train/good/.hidden.png")
        with self.assertRaisesRegex(ValueError, "정상 학습 이미지 없음"):
            discover_data(self.root)

    def test_transfer_settings_survive_project_move(self):
        original = self.root / "original"
        project = ProjectManager.create_new("Anomaly", "anomaly", str(original), ["good", "defect"])
        path = original / "weights/resnet.pth"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"weights")
        project.training.patchcore_weights = str(path)
        project.training.patchcore_weight_source = "backbone"
        project.training.patchcore_backbone = "resnet18"
        project.training.patchcore_max_candidates = 321
        project.training.patchcore_crop_enabled = True
        project.training.patchcore_crop_width, project.training.patchcore_crop_height = 1024, 768
        filename = Path(ProjectManager.save(project))
        raw = json.loads(filename.read_text(encoding="utf-8"))
        self.assertFalse(Path(raw["training"]["patchcore_weights"]).is_absolute())
        moved = self.root / "moved"
        shutil.move(str(original), str(moved))
        loaded = ProjectManager.load(str(moved / filename.name))
        self.assertEqual(Path(loaded.training.patchcore_weights), moved / "weights/resnet.pth")
        self.assertEqual(loaded.training.patchcore_backbone, "resnet18")
        self.assertEqual(loaded.training.patchcore_max_candidates, 321)
        self.assertTrue(loaded.training.patchcore_crop_enabled)
        self.assertEqual(loaded.training.patchcore_crop_width, 1024)
        self.assertEqual(loaded.training.patchcore_crop_height, 768)


if __name__ == "__main__":
    unittest.main()
