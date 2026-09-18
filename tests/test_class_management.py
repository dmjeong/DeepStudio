"""Qt/torch 없이 실제 임시 파일로 삭제, 재번호화, 저장 실패 복구 검증."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
from core.class_management import ClassManager, move_image_with_sidecars
from core.project import ProjectManager, RunRecord


class _FilesystemTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def project(self, task="classify", names=None):
        project = ProjectManager.create_new("example", task, str(self.base / task),
                                            names or ["background", "scratch", "dent"])
        ProjectManager.save(project)
        return project

    def write(self, path, content):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path


class ClassManagementTests(_FilesystemTests):
    def test_classify_archives_all_splits_and_persists_reopen(self):
        project = self.project()
        root = Path(project.data.root)
        for split in ("train", "val", "test"):
            self.write(root / split / "scratch" / "image.png", b"original image")
            self.write(root / split / "dent" / "retained.png", b"retained image")
        project.runs.append(RunRecord(run_id="old", eval_results={"class_names": list(project.data.class_names)}))
        ProjectManager.save(project)
        previous_runs = copy.deepcopy(project.runs)
        preview = ClassManager.preview_delete(project, "scratch")
        self.assertEqual(preview.image_count, 3)
        self.assertEqual(len(preview.folders), 3)
        archive = Path(ClassManager.delete(project, "scratch", preview))
        self.assertNotIn(root, archive.parents)
        self.assertEqual(project.runs, previous_runs)
        loaded = ProjectManager.load(ProjectManager.get_active_filepath(project))
        self.assertEqual(loaded.data.class_names, ["background", "dent"])
        self.assertEqual(loaded.data.num_classes, 2)
        for split in ("train", "val", "test"):
            self.assertFalse((root / split / "scratch").exists())
            self.assertEqual((root / split / "dent" / "retained.png").read_bytes(), b"retained image")
        manifest = json.loads((archive / "manifest.json").read_text())
        self.assertEqual(sum(record["kind"] == "move" for record in manifest["files"]), 3)
        self.assertEqual(len(list(archive.rglob("image.png"))), 3)

    def test_classify_save_failure_restores_folders_and_project(self):
        project = self.project()
        paths = [self.write(Path(project.data.root) / split / "scratch" / "a.png", split)
                 for split in ("train", "val", "test")]
        saved = Path(ProjectManager.get_active_filepath(project)).read_bytes()
        old_modified = project.modified
        with patch.object(ProjectManager, "save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                ClassManager.delete(project, "scratch")
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertEqual(project.data.class_names, ["background", "scratch", "dent"])
        self.assertEqual(project.modified, old_modified)
        self.assertEqual(Path(ProjectManager.get_active_filepath(project)).read_bytes(), saved)

    def test_detection_removes_and_remaps_all_splits_keeps_images(self):
        project = self.project("detect")
        root = Path(project.data.root)
        before = b"0 0.5 0.5 0.1 0.1\n1 0.4 0.4 0.2 0.2\n2 0.3 0.3 0.1 0.1\n"
        labels = [self.write(root / "labels" / split / "a.txt", before) for split in ("train", "val", "test")]
        images = [self.write(root / "images" / split / "a.png", b"pixels") for split in ("train", "val", "test")]
        preview = ClassManager.preview_delete(project, "scratch")
        self.assertEqual((preview.removed_annotations, preview.remapped_annotations, preview.changed_files), (3, 3, 3))
        archive = Path(ClassManager.delete(project, "scratch", preview))
        for path in labels:
            self.assertEqual(path.read_text(), "0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.1 0.1\n")
        self.assertTrue(all(path.read_bytes() == b"pixels" for path in images))
        self.assertEqual(len(list(archive.glob("original_*.txt"))), 3)
        self.assertTrue(all(path.read_bytes() == before for path in archive.glob("original_*.txt")))

    def test_empty_detection_annotation_becomes_background_file(self):
        project = self.project("detect")
        label = self.write(Path(project.data.root) / "labels/train/only.txt", "1 0.5 0.5 0.1 0.1\n")
        ClassManager.delete(project, "scratch")
        self.assertEqual(label.read_bytes(), b"")

    def test_malformed_detection_blocks_every_change(self):
        malformed = ["1.0 0.5 0.5 0.1 0.1", "3 0.5 0.5 0.1 0.1", "1 NaN 0.5 0.1 0.1",
                     "1 0.5 0.5 0.0 0.1", "1 0.5 0.5 0.1", "-1 0.5 0.5 0.1 0.1"]
        project = self.project("detect")
        for text in malformed:
            with self.subTest(label=text):
                root = Path(project.data.root)
                valid = self.write(root / "labels/train/a.txt", "2 0.5 0.5 0.1 0.1\n")
                self.write(root / "labels/test/b.txt", text)
                with self.assertRaises(ValueError):
                    ClassManager.delete(project, "scratch")
                self.assertEqual(valid.read_text(), "2 0.5 0.5 0.1 0.1\n")
                self.assertEqual(project.data.num_classes, 3)

    def test_detection_save_failure_restores_original_bytes(self):
        project = self.project("detect")
        label = self.write(Path(project.data.root) / "labels/train/a.txt", b"1 0.5 0.5 0.1 0.1\r\n2 0.4 0.4 0.2 0.2\r\n")
        before = label.read_bytes()
        with patch.object(ProjectManager, "save", side_effect=OSError("save error")):
            with self.assertRaises(OSError):
                ClassManager.delete(project, "scratch")
        self.assertEqual(label.read_bytes(), before)
        self.assertEqual(project.data.num_classes, 3)
        self.assertFalse(list(label.parent.glob("*.pending")))

    def test_label_replace_failure_restores_previously_applied_label(self):
        project = self.project("detect")
        root = Path(project.data.root)
        labels = [self.write(root / "labels" / split / "a.txt", "2 0.5 0.5 0.1 0.1\n")
                  for split in ("train", "val")]
        original_replace = os.replace
        edits = []
        def fail_second(source, target):
            if str(source).endswith(".pending"):
                edits.append(str(target))
                if len(edits) == 2:
                    raise OSError("label locked")
            return original_replace(source, target)
        with patch("core.class_management.os.replace", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "label locked"):
                ClassManager.delete(project, "scratch")
        self.assertTrue(all(path.read_text().startswith("2 ") for path in labels))

    def test_stale_preview_requires_confirmation_again(self):
        project = self.project("detect")
        label = self.write(Path(project.data.root) / "labels/train/a.txt", "2 0.5 0.5 0.1 0.1\n")
        preview = ClassManager.preview_delete(project, "scratch")
        label.write_text("1 0.5 0.5 0.1 0.1\n")
        with self.assertRaisesRegex(ValueError, "미리보기"):
            ClassManager.delete(project, "scratch", preview)
        self.assertTrue(label.read_text().startswith("1 "))

    def test_generated_yaml_and_cache_archived_until_regenerated(self):
        project = self.project("detect")
        root = Path(project.data.root)
        yaml = self.write(root / "dataset.yaml", "names: [background, scratch, dent]")
        cache = self.write(root / "labels/train.cache", b"old label cache")
        archive = Path(ClassManager.delete(project, "scratch"))
        self.assertFalse(yaml.exists())
        self.assertFalse(cache.exists())
        contents = [path.read_bytes() for path in archive.glob("removed_*")]
        self.assertIn(b"old label cache", contents)

    def test_explicit_classification_split_paths_are_included(self):
        project = self.project()
        project.data.test_dir = str(self.base / "external_test")
        image = self.write(Path(project.data.test_dir) / "scratch/a.png", b"external")
        ClassManager.delete(project, "scratch")
        self.assertFalse(image.exists())

    def test_segmentation_preserves_palette_indices_and_ignore255(self):
        project = self.project("segment")
        root = Path(project.data.root)
        palette = [0] * 768
        palette[3:6], palette[6:9], palette[765:768] = [255, 0, 0], [0, 255, 0], [255, 255, 255]
        paths = []
        for split in ("train", "val", "test"):
            path = root / "masks" / split / "indexed.png"
            mask = Image.new("P", (4, 1))
            mask.putdata([0, 1, 2, 255])
            mask.putpalette(palette)
            mask.save(path)
            paths.append(path)
        preview = ClassManager.preview_delete(project, "scratch")
        self.assertEqual((preview.removed_pixels, preview.remapped_pixels), (3, 3))
        ClassManager.delete(project, "scratch", preview)
        for path in paths:
            with Image.open(path) as mask:
                self.assertEqual(mask.mode, "P")
                self.assertEqual(list(mask.getdata()), [0, 0, 1, 255])
                self.assertEqual(mask.getpalette()[3:6], [0, 255, 0])
                self.assertEqual(mask.getpalette()[765:768], [255, 255, 255])

    def test_segmentation_16bit_masks_preserve_integers(self):
        project = self.project("segment")
        path = Path(project.data.root) / "masks/train/a.tiff"
        mask = Image.new("I", (4, 1))
        mask.putdata([0, 1, 2, 255])
        mask.convert("I;16").save(path)
        ClassManager.delete(project, "scratch")
        with Image.open(path) as output:
            self.assertEqual(output.mode, "I;16")
            self.assertEqual(list(output.getdata()), [0, 0, 1, 255])

    def test_segmentation_rgb_mask_rejected_without_changes(self):
        project = self.project("segment")
        path = Path(project.data.root) / "masks/train/a.png"
        Image.new("RGB", (2, 2), "red").save(path)
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            ClassManager.delete(project, "scratch")
        self.assertEqual(path.read_bytes(), before)

    def test_segment_polygon_labels_remapped_together_with_masks(self):
        project = self.project("segment")
        label = self.write(Path(project.data.root) / "labels/test/a.txt", "1 0.1 0.1 0.2 0.2 0.3 0.1\n2 0.1 0.1 0.2 0.2 0.3 0.1\n")
        ClassManager.delete(project, "scratch")
        self.assertEqual(label.read_text(), "1 0.1 0.1 0.2 0.2 0.3 0.1\n")

    def test_segment_background_and_last_class_cannot_be_deleted(self):
        project = self.project("segment")
        project.training.training_mode = "custom"
        with self.assertRaises(ValueError):
            ClassManager.delete(project, "background")
        project.data.class_names = ["background"]
        with self.assertRaises(ValueError):
            ClassManager.delete(project, "background")


    def test_invalid_names_and_case_duplicate_add_are_rejected(self):
        project = self.project()
        for name in ("../outside", "a/b", "a\\b", "CON", "LPT1.txt", "bad.", "", " background"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ClassManager.add(project, name)
        with self.assertRaises(ValueError):
            ClassManager.add(project, "SCRATCH")
        project.data.class_names.append("../outside")
        with self.assertRaises(ValueError):
            ClassManager.delete(project, "scratch")

    def test_symlink_class_and_label_directory_rejected(self):
        project = self.project()
        target = Path(project.data.root) / "train/scratch"
        target.rmdir()
        outside = self.base / "outside"
        outside.mkdir()
        self.write(outside / "a.png", b"safe")
        try:
            target.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symlink not supported")
        with self.assertRaises(ValueError):
            ClassManager.delete(project, "scratch")
        self.assertEqual((outside / "a.png").read_bytes(), b"safe")

    def test_add_persists_and_rolls_back_save_failure(self):
        project = self.project()
        ClassManager.add(project, "new defect")
        loaded = ProjectManager.load(ProjectManager.get_active_filepath(project))
        self.assertEqual(loaded.data.class_names[-1], "new defect")
        with patch.object(ProjectManager, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                ClassManager.add(project, "failed")
        self.assertNotIn("failed", project.data.class_names)
        self.assertFalse(any((Path(project.data.root) / split / "failed").exists() for split in ("train", "val", "test")))

    def test_missing_or_relative_data_root_rejected(self):
        project = self.project()
        for root in ("", ".", str(self.base / "does_not_exist")):
            with self.subTest(root=root):
                project.data.root = root
                with self.assertRaises(ValueError):
                    ClassManager.delete(project, "scratch")

    def test_symlink_label_directory_does_not_change_external_data(self):
        project = self.project("detect")
        path = Path(project.data.root) / "labels/train"
        path.rmdir()
        outside = self.base / "outside"
        label = self.write(outside / "a.txt", "2 0.5 0.5 0.1 0.1\n")
        try:
            path.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symlink not supported")
        with self.assertRaises(ValueError):
            ClassManager.delete(project, "scratch")
        self.assertTrue(label.read_text().startswith("2 "))

    def test_mask_save_failure_restores_palette_file_exactly(self):
        project = self.project("segment")
        path = Path(project.data.root) / "masks/train/a.png"
        image = Image.new("P", (3, 1))
        image.putdata([0, 1, 2])
        image.putpalette([0, 0, 0, 0, 255, 0, 255, 0, 0])
        image.save(path)
        before = path.read_bytes()
        with patch.object(ProjectManager, "save", side_effect=OSError("full")):
            with self.assertRaises(OSError):
                ClassManager.delete(project, "scratch")
        self.assertEqual(path.read_bytes(), before)


class SidecarMoveTests(_FilesystemTests):
    def test_orphan_label_collision_reserves_pair_name(self):
        image = self.write(self.base / "images/train/a.png", b"image")
        label = self.write(self.base / "labels/train/a.txt", b"correct")
        existing = self.write(self.base / "labels/val/a.txt", b"must not overwrite")
        destination = move_image_with_sidecars(image, self.base / "images/val", [(label, existing.parent, ".txt")], True)
        self.assertEqual(Path(destination).name, "a_1.png")
        self.assertEqual(existing.read_bytes(), b"must not overwrite")
        self.assertEqual((existing.parent / "a_1.txt").read_bytes(), b"correct")

    def test_different_image_extensions_still_share_label_stem(self):
        image = self.write(self.base / "images/train/a.png", b"new")
        label = self.write(self.base / "labels/train/a.txt", b"label")
        existing = self.write(self.base / "images/val/a.jpg", b"old")
        destination = move_image_with_sidecars(image, existing.parent, [(label, self.base / "labels/val", ".txt")], True)
        self.assertEqual(Path(destination).name, "a_1.png")
        self.assertEqual(existing.read_bytes(), b"old")

    def test_unlabelled_image_does_not_acquire_orphan_label(self):
        image = self.write(self.base / "images/train/a.png", b"new")
        existing = self.write(self.base / "labels/val/a.txt", b"wrong")
        destination = move_image_with_sidecars(image, self.base / "images/val", [], True, [existing.parent])
        self.assertEqual(Path(destination).name, "a_1.png")
        self.assertEqual(existing.read_bytes(), b"wrong")

    def test_partial_copy_failure_restores_all_source_files(self):
        image = self.write(self.base / "images/train/a.png", b"image")
        label = self.write(self.base / "labels/train/a.txt", b"label")
        import shutil
        original_copy = shutil.copyfileobj
        calls = []
        def fail_image(reader, writer, *args, **kwargs):
            calls.append(reader.name)
            if len(calls) == 2:
                writer.write(b"partial")
                raise OSError("disk full")
            return original_copy(reader, writer, *args, **kwargs)
        with patch("core.class_management.shutil.copyfileobj", side_effect=fail_image):
            with self.assertRaises(OSError):
                move_image_with_sidecars(image, self.base / "images/val", [(label, self.base / "labels/val", ".txt")], True)
        self.assertEqual(image.read_bytes(), b"image")
        self.assertEqual(label.read_bytes(), b"label")
        self.assertFalse(list((self.base / "images/val").glob("*")))
        self.assertFalse(list((self.base / "labels/val").glob("*")))

    def test_shared_source_label_is_rejected_without_stealing(self):
        image = self.write(self.base / "images/train/a.png", b"image")
        self.write(image.with_suffix(".jpg"), b"second")
        label = self.write(self.base / "labels/train/a.txt", b"shared")
        with self.assertRaises(ValueError):
            move_image_with_sidecars(image, self.base / "images/val", [(label, self.base / "labels/val", ".txt")], True)
        self.assertTrue(image.exists())
        self.assertEqual(label.read_bytes(), b"shared")


if __name__ == "__main__":
    unittest.main()
