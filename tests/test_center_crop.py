"""Centered source ROI, orientation, height precision and displayed inspection area."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]
from center_crop import center_crop_box, configured_center_crop, validate_center_crop
from patchcore_data import input_image, validate_crop_images
from core.heatmap import reproject_center_crop, render_heatmap


class CenterCropTests(unittest.TestCase):
    def test_exact_center_of_rectangular_and_odd_images(self):
        self.assertEqual(center_crop_box((2048, 2048), {"width": 1024, "height": 512}), (512, 768, 1536, 1280))
        self.assertEqual(center_crop_box((11, 9), {"width": 4, "height": 2}), (3, 3, 7, 5))
        self.assertEqual(center_crop_box((7, 5), None), (0, 0, 7, 5))
        self.assertEqual(center_crop_box((7, 5), {"width": 7, "height": 5}), (0, 0, 7, 5))
        with self.assertRaisesRegex(ValueError, "원본"):
            center_crop_box((7, 5), {"width": 8, "height": 4})
        for bad in (0, -1, True, 2.5, float("nan"), 65537, "32"):
            with self.assertRaises(ValueError):
                validate_center_crop({"width": bad, "height": 32})
        self.assertIsNone(configured_center_crop(SimpleNamespace()))

    def test_height_pixels_and_exif_orientation_match_source_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.arange(9 * 11, dtype=np.uint16).reshape(9, 11) * 600
            path = root / "height.png"
            Image.fromarray(values).save(path)
            crop = {"width": 4, "height": 2}
            validate_crop_images([path], crop)
            with input_image(path) as image:
                np.testing.assert_array_equal(np.asarray(image.crop(center_crop_box(image.size, crop))), values[3:5, 3:7])
            validate_crop_images([path], None)
            with self.assertRaisesRegex(ValueError, "height.png"):
                validate_crop_images([path], {"width": 12, "height": 2})
            rgb = Image.new("RGB", (12, 6), (10, 20, 30))
            exif = Image.Exif()
            exif[274] = 6
            path = root / "oriented.jpg"
            rgb.save(path, exif=exif)
            with input_image(path) as image:
                self.assertEqual(image.size, (6, 12))
            validate_crop_images([path], {"width": 6, "height": 10})
            with self.assertRaisesRegex(ValueError, "원본"):
                validate_crop_images([path], {"width": 6, "height": 10}, "legacy_pil_rgb")

    def test_heatmap_does_not_paint_outside_inspected_crop(self):
        crop = {"width": 4, "height": 2}
        activation = np.array([[0, 1], [1, 0]], dtype=np.float32)
        restored, valid = reproject_center_crop(activation, (9, 11), crop)
        self.assertEqual(int(valid.sum()), 8)
        self.assertTrue(valid[3:5, 3:7].all())
        self.assertEqual(float(restored[~valid].sum()), 0)
        original = np.full((9, 11, 3), 110, dtype=np.uint8)
        for lower, upper in ((0, 1), (.2, .8)):
            _, overlay = render_heatmap(restored, original, 1, lower, upper, valid_mask=valid)
            np.testing.assert_array_equal(overlay[~valid], original[~valid])
        with self.assertRaisesRegex(ValueError, "원본"):
            reproject_center_crop(activation, (9, 11), {"width": 12, "height": 2})


if __name__ == "__main__":
    unittest.main()
