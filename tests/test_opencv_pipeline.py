"""OpenCV migration, image geometry and real classification transforms."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from opencv_preprocess import resize_contract, require_resume_contract, resize, center_crop, convert_channels, read_rgb
from checkpoint import make_checkpoint_metadata

CV_AVAILABLE = bool(importlib.util.find_spec("cv2"))


class MigrationTests(unittest.TestCase):
    def test_legacy_checkpoint_migration_is_recorded_without_mutating_source(self):
        for previous in (None, "pillow_u8_bilinear"):
            before = {"resize": "bilinear", "antialias": True, "mean": [.1, .2, .3]}
            if previous:
                before["resize_implementation"] = previous
            snapshot = dict(before)
            with self.assertWarnsRegex(UserWarning, "OpenCV"):
                current = resize_contract(before)
            self.assertEqual(before, snapshot)
            self.assertEqual(current["migrated_from"], previous or "legacy_unspecified")
            self.assertFalse(current["antialias"])
            self.assertEqual(current["mean"], before["mean"])
            self.assertEqual(current["interpolation"], "INTER_LINEAR_EXACT")

    def test_new_checkpoint_can_resume_but_old_pipeline_cannot(self):
        checkpoint = make_checkpoint_metadata("classify", 2, ["NG", "OK"], 224, 3)
        require_resume_contract(checkpoint["preprocessing"])
        for prep in (None, {}, {"resize_implementation": "pillow_u8_bilinear"}):
            with self.assertRaisesRegex(ValueError, "추가 학습"):
                require_resume_contract(prep)

    def test_unknown_resize_and_false_claims_are_rejected(self):
        for prep in ({"resize_implementation": "unknown"},
                     {"resize_implementation": "opencv_linear_exact_v1", "antialias": True},
                     {"resize_implementation": "opencv_linear_exact_v1", "interpolation": "INTER_CUBIC"}):
            with self.assertRaises(ValueError):
                resize_contract(prep)


@unittest.skipUnless(CV_AVAILABLE, "OpenCV unavailable")
class PixelTests(unittest.TestCase):
    def test_resize_and_channel_order_against_opencv(self):
        import cv2
        rgb = np.random.default_rng(101).integers(0, 256, (33, 87, 3), dtype=np.uint8)
        for shape in ((224, 224), (240, 240), (13, 41), (1, 1)):
            expected = cv2.resize(rgb, shape[::-1], interpolation=cv2.INTER_LINEAR_EXACT)
            np.testing.assert_array_equal(resize(rgb, shape), expected)
        np.testing.assert_array_equal(convert_channels(rgb, 1), cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        np.testing.assert_array_equal(convert_channels(rgb[..., 0], 3), cv2.cvtColor(rgb[..., 0], cv2.COLOR_GRAY2RGB))

    def test_short_edge_and_odd_center_offsets(self):
        image = np.arange(5 * 9, dtype=np.uint8).reshape(5, 9)
        np.testing.assert_array_equal(center_crop(image, (2, 2)), image[2:4, 4:6])
        self.assertEqual(resize(image, 10).shape, (10, 18))
        self.assertEqual(resize(image.T, 10).shape, (18, 10))
        with self.assertRaises(ValueError):
            center_crop(image, (6, 6))

    def test_unicode_image_roundtrip(self):
        import cv2
        rgb = np.random.default_rng(8).integers(0, 256, (7, 9, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "검사.png"
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.assertTrue(ok)
            encoded.tofile(path)
            np.testing.assert_array_equal(read_rgb(path), rgb)




if __name__ == "__main__":
    unittest.main()
