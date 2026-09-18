"""JSON compatibility, rejected ambiguity, and per-job region snapshots."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
from core.inference_region import crop_from_json, read_input_region, resolve_input_region, input_region_label


class InferenceRegionTests(unittest.TestCase):
    def test_studio_json_formats_and_explicit_full_image(self):
        crop = {"width": 1024, "height": 768}
        for data in ({"center_crop": crop}, {"preprocessing": {"center_crop": crop}},
                     {"preprocessing": "full_range_v1", "center_crop": crop},
                     {"training": {"patchcore_crop_enabled": True,
                                   "patchcore_crop_width": 1024, "patchcore_crop_height": 768}}):
            with self.subTest(data=data):
                self.assertEqual(crop_from_json(data), crop)
        for data in ({"center_crop": None}, {"training": {"patchcore_crop_enabled": False}},
                     {"schema_version": 1, "model_path": "old.onnx", "input_height": 224, "input_width": 224}):
            self.assertIsNone(crop_from_json(data))

    def test_invalid_or_conflicting_settings_never_silently_disable_crop(self):
        for data in ([], {}, {"preprocessing": "full_range_v1"},
                     {"center_crop": {"width": True, "height": 32}},
                     {"center_crop": {"width": 0, "height": 32}},
                     {"center_crop": {"width": 32.5, "height": 32}},
                     {"training": {"patchcore_crop_enabled": "false"}},
                     {"center_crop": None, "preprocessing": {"center_crop": {"width": 32, "height": 32}}}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                crop_from_json(data)
        with self.assertRaises(ValueError):
            resolve_input_region({"mode": "json"}, None)
        with self.assertRaises(ValueError):
            read_input_region("unknown")

    def test_job_snapshot_survives_file_changes_and_keeps_checkpoint_settings(self):
        saved = {"width": 80, "height": 60}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.json"
            path.write_text(json.dumps({"center_crop": {"width": 40, "height": 24}}), encoding="utf-8-sig")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            request = read_input_region("json", path)
            path.unlink()
            region = resolve_input_region(request, saved, (32, 64))
            self.assertEqual(region["center_crop"], {"width": 40, "height": 24})
            self.assertEqual(region["model_input_size"], [64, 32])
            self.assertEqual(region["json_sha256"], digest)
            self.assertIn("40×24", input_region_label(region))
            region["center_crop"]["width"] = 1
            self.assertEqual(request["center_crop"]["width"], 40)
            full = resolve_input_region(read_input_region("full", path), saved)
            self.assertIsNone(full["center_crop"])
            self.assertEqual(resolve_input_region(None, saved)["center_crop"], saved)
            self.assertEqual(saved, {"width": 80, "height": 60})
            with self.assertRaisesRegex(ValueError, "읽기 실패"):
                read_input_region("json", path)
            path.write_bytes(b"{bad JSON")
            with self.assertRaisesRegex(ValueError, "읽기 실패"):
                read_input_region("json", path)
