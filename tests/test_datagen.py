"""Real data lineage, true minimum checkpointing, review and crop integrity."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
from core.project import ProjectManager
from core.datagen_store import DataGenStore, checkpoint_decision, save_png, uid
from core.datagen_learning import crop_pair, compose, config
from webapp.storage import write_json, digest


class DataGenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = ProjectManager.create_new("gen", "classify", str(self.root / "project"), ["OK", "NG"])
        self.store = DataGenStore(self.project)
        self.item = self.store.update_item("사용자 표면 패턴", class_name="NG")
        self.mask = self.root / "mask.png"
        pixels = Image.new("L", (96, 64))
        ImageDraw.Draw(pixels).rectangle((3, 4, 12, 20), fill=255)
        pixels.save(self.mask)

    def source(self, value, suffix="png"):
        path = self.root / f"source-{value}.{suffix}"
        Image.new("RGB", (96, 64), (value, 100, 120)).save(path)
        return path

    def ingest(self, value, split="train", group=None):
        self.store.import_images(self.item["id"], paths=[str(self.source(value))], split=split, group=group or split)
        row = self.store.state()["images"][-1]
        self.store.annotate(row["id"], str(self.mask))
        return row

    def sample(self):
        key = uid()
        root = self.store.root / "samples" / key
        for name in ("image", "original", "allowed", "requested", "changed"):
            pixels = Image.open(self.mask) if name not in {"image", "original"} else Image.open(self.source(21))
            save_png(root / f"{name}.png", pixels)
        row = {"id": key, "synthetic": True, "item_id": self.item["id"], "class_name": "NG",
               "review": {"status": "pending", "mask": "", "reason": ""},
               "hashes": {p.name: digest(p) for p in root.glob("*.png")}}
        write_json(root / "manifest.json", row)
        return key

    def test_custom_item_identity_survives_rename_and_archive(self):
        key = self.item["id"]
        changed = self.store.update_item("내가 추가한 항목", key, "NG")
        self.assertEqual(key, changed["id"])
        self.store.update_item("내가 추가한 항목", key, "NG", True)
        with self.assertRaises(ValueError):
            self.store.freeze(key)
        self.assertTrue(self.store.state()["items"][0]["archived"])

    def test_duplicate_reencoded_pixels_and_cross_split_group_rejected(self):
        self.ingest(20, group="lot-A")
        with self.assertRaisesRegex(ValueError, "그룹"):
            self.store.import_images(self.item["id"], paths=[str(self.source(22))], split="val", group="lot-A")
        with self.assertRaisesRegex(ValueError, "중복"):
            self.store.import_images(self.item["id"], paths=[str(self.source(20, "bmp"))], split="val", group="lot-B")
        self.assertEqual(len(self.store.state()["images"]), 1)

    def test_group_whitespace_cannot_bypass_split_isolation(self):
        self.ingest(20, group=" lot-A ")
        with self.assertRaisesRegex(ValueError, "그룹"):
            self.store.import_images(self.item["id"], paths=[str(self.source(22))], split="val", group="lot-A ")
        self.assertEqual(self.store.state()["images"][0]["group"], "lot-A")

    def test_other_ui_cannot_modify_dataset_during_compute(self):
        from core.datagen_store import mutate
        from webapp.locking import exclusive_file
        with exclusive_file(self.store.root / "compute.lock"):
            with self.assertRaises(RuntimeError):
                mutate(self.project, "item", {"name": "동시 변경"})
        self.assertEqual(len(self.store.state()["items"]), 1)
        mutate(self.project, "item", {"name": "작업 종료 후 추가"})
        self.assertEqual(len(self.store.state()["items"]), 2)

    def test_snapshot_is_immutable_after_annotation_change(self):
        first = self.ingest(20)
        self.ingest(22, "val")
        frozen = self.store.freeze(self.item["id"])
        original = self.store.root / "datasets" / frozen["id"] / first["id"] / "mask.png"
        before = digest(original)
        replacement = self.root / "replacement.png"
        Image.new("L", (96, 64), 255).save(replacement)
        self.store.annotate(first["id"], str(replacement))
        self.assertEqual(digest(original), before)

    def test_dataset_requires_masks_and_real_validation(self):
        self.ingest(20)
        with self.assertRaisesRegex(ValueError, "val"):
            self.store.freeze(self.item["id"])
        self.store.import_images(self.item["id"], paths=[str(self.source(22))], split="val", group="val")
        with self.assertRaisesRegex(ValueError, "마스크"):
            self.store.freeze(self.item["id"])

    def test_eval_folder_and_16bit_rejected(self):
        path = Path(self.project.data.test_dir) / "NG" / "bad.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (96, 64)).save(path)
        with self.assertRaises(ValueError):
            self.store.import_images(self.item["id"], paths=[str(path)], group="test")
        path = self.root / "height.png"
        Image.fromarray(np.ones((64, 96), dtype="uint16")).save(path)
        with self.assertRaisesRegex(ValueError, "8비트"):
            self.store.import_images(self.item["id"], paths=[str(path)], group="height")

    def test_tampered_training_snapshot_rejected(self):
        row = self.ingest(20)
        self.ingest(22, "val")
        Image.new("RGB", (96, 64)).save(self.store.file("images", row["id"]))
        with self.assertRaisesRegex(ValueError, "외부"):
            self.store.freeze(self.item["id"])
        self.assertFalse(self.store.records("datasets"))

    def test_true_minimum_independent_from_earlystop_delta(self):
        state, stop = checkpoint_decision({}, 1.0, "first", .1, 2)
        state, stop = checkpoint_decision(state, .99, "second", .1, 2)
        self.assertEqual(state["best_val_loss"], "second")
        self.assertFalse(stop)
        state, stop = checkpoint_decision(state, .99, "tie", .1, 2)
        self.assertEqual(state["best_val_loss"], "second")
        self.assertTrue(stop)
        for value in (None, math.nan, math.inf):
            previous = dict(state)
            state, stop = checkpoint_decision(state, value, "invalid")
            self.assertEqual(state, previous)
            self.assertFalse(stop)

    def test_crop_non_square_boundary_and_grayscale_preserves_outside(self):
        for mode in ("RGB", "L"):
            original = Image.new(mode, (96, 64), 100)
            mask = Image.open(self.mask)
            patch, scaled, box = crop_pair(original, mask, 256)
            self.assertEqual(patch.size, (256, 256))
            self.assertEqual(scaled.size, patch.size)
            self.assertLess(box[0], 0)
            output, changed = compose(original, Image.new("RGB", patch.size, 220), mask, mask, box)
            permit = np.array(mask) > 0
            self.assertEqual(output.mode, mode)
            self.assertTrue(np.array_equal(np.array(output)[~permit], np.array(original)[~permit]))
            self.assertTrue((np.array(changed)[permit] > 0).all())

    def test_mask_hole_and_disallowed_changes(self):
        original = Image.new("RGB", (96, 64), 100)
        mask = Image.open(self.mask).copy()
        ImageDraw.Draw(mask).rectangle((6, 8, 9, 12), fill=0)
        _, _, box = crop_pair(original, mask, 256)
        output, _ = compose(original, Image.new("RGB", (256, 256), 200), mask, mask, box)
        self.assertTrue(np.array_equal(np.array(output)[8:13, 6:10], np.array(original)[8:13, 6:10]))
        with self.assertRaises(ValueError):
            compose(original, output, mask, Image.new("L", original.size), box)

    def test_approval_requires_explicit_label_and_persists(self):
        key = self.sample()
        with self.assertRaises(ValueError):
            self.store.review(key, "approved")
        self.store.review(key, "approved", str(self.mask), "실제 영역 확인")
        restored = DataGenStore(self.project).record("samples", key)
        self.assertEqual(restored["review"]["status"], "approved")
        self.assertNotEqual(restored["review"]["mask"], "changed.png")
        self.store.review(key, "pending", reason="재검수")
        self.assertEqual(len(self.store.record("samples", key)["review_history"]), 2)

    def test_unreviewed_and_corrupted_label_cannot_publish(self):
        key = self.sample()
        output = str(self.root / "published")
        with self.assertRaises(ValueError):
            self.store.publish([key], output)
        self.store.review(key, "approved", str(self.mask))
        reviewed = self.store.file("samples", key, "reviewed")
        Image.new("L", (96, 64), 255).save(reviewed)
        with self.assertRaisesRegex(ValueError, "무결성"):
            self.store.publish([key], output)
        self.assertFalse(list(Path(output).glob("datagen-train-*")))

    def test_publication_recovers_after_manifest_index_failure(self):
        key = self.sample()
        self.store.review(key, "approved", str(self.mask))
        output = str(self.root / "published")
        from core import datagen_store
        write = datagen_store.write_json
        def fail_index(path, value):
            if "publications" in Path(path).parts:
                raise OSError("index disk failure")
            write(path, value)
        with patch.object(datagen_store, "write_json", side_effect=fail_index):
            with self.assertRaises(OSError):
                self.store.publish([key], output)
        result = self.store.publish([key], output)
        repeated = self.store.publish([key], output)
        self.assertEqual(result["id"], repeated["id"])
        self.assertEqual(len(list(Path(output).glob("datagen-train-*"))), 1)
        self.assertFalse(result["normal_training"])

    def test_invalid_config_never_silently_defaults(self):
        with self.assertRaises(ValueError):
            config({"prompt": "custom", "steps": -1})
        with self.assertRaises(ValueError):
            config({"prompt": "custom", "resolution": 513})
        with self.assertRaises(ValueError):
            config({"prompt": "custom", "learning_rate": math.nan})


if __name__ == "__main__":
    unittest.main()
