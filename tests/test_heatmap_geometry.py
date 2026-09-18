"""분류 전처리의 잘림과 패딩을 원본 CAM 좌표로 정확히 되돌리는지 검증한다."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("geometry_heatmap", ROOT / "gui/core/heatmap.py")
HEATMAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HEATMAP)
reproject = HEATMAP.reproject_classification_cam


class ClassificationCamGeometry(unittest.TestCase):
    def test_landscape_crop_preserves_unseen_left_and_right_margins(self):
        cam = np.ones((2, 2), dtype=np.float32)
        result, mask = reproject(cam, (4, 8), (4, 8), (4, 4), return_valid_mask=True)
        expected = np.zeros((4, 8), dtype=np.float32)
        expected[:, 2:6] = 1.0
        np.testing.assert_array_equal(result, expected)
        original = np.full((4, 8, 3), 157, dtype=np.uint8)
        _, overlay = HEATMAP.render_heatmap(result, original, alpha=1.0, valid_mask=mask)
        np.testing.assert_array_equal(overlay[:, :2], original[:, :2])
        np.testing.assert_array_equal(overlay[:, 6:], original[:, 6:])

    def test_zero_response_coverage_is_independent_of_cam_values(self):
        result, mask = reproject(np.zeros((2, 2), np.float32), (4, 8), (4, 8), (4, 4),
                                 return_valid_mask=True)
        np.testing.assert_array_equal(result, np.zeros((4, 8), np.float32))
        expected = np.zeros((4, 8), bool)
        expected[:, 2:6] = True
        np.testing.assert_array_equal(mask, expected)
        original = np.full((4, 8, 3), 157, np.uint8)
        _, overlay = HEATMAP.render_heatmap(result, original, alpha=1.0, valid_mask=mask)
        np.testing.assert_array_equal(overlay[mask], np.tile([0, 0, 127], (16, 1)))
        np.testing.assert_array_equal(overlay[~mask], original[~mask])

    def test_padding_and_direct_resize_cover_all_original_pixels(self):
        for shapes in (((8, 8), (4, 4), (7, 9)), ((7, 13), (5, 8), (5, 8))):
            with self.subTest(shapes=shapes):
                _, mask = reproject(np.zeros((2, 2), np.float32), *shapes, return_valid_mask=True)
                self.assertEqual(mask.shape, shapes[0])
                self.assertTrue(mask.all())

    def test_portrait_crop_uses_top_and_bottom_margins(self):
        result = reproject(np.ones((3, 3), np.float32), (16, 8), (8, 4), (4, 4))
        expected = np.zeros((16, 8), np.float32)
        expected[4:12, :] = 1.0
        np.testing.assert_array_equal(result, expected)

    def test_center_rounding_matches_even_rounding_for_odd_differences(self):
        for resized, first in ((5, 0), (7, 2), (9, 2)):
            with self.subTest(resized=resized):
                result = reproject(np.ones((4, 4), np.float32),
                                   (4, resized), (4, resized), (4, 4))
                expected = np.zeros((4, resized), np.float32)
                expected[:, first:first + 4] = 1.0
                np.testing.assert_array_equal(result, expected)

    def test_padding_activation_is_discarded_before_original_upsampling(self):
        cam = np.ones((7, 9), np.float32)
        # 4x4를 7x9로 패딩하면 위 1, 아래 2, 왼쪽 2, 오른쪽 3이다.
        cam[1:5, 2:6] = 0.0
        result = reproject(cam, (8, 8), (4, 4), (7, 9))
        np.testing.assert_array_equal(result, np.zeros((8, 8), np.float32))

    def test_padding_in_one_axis_and_crop_in_other_axis(self):
        cam = np.ones((7, 4), np.float32)
        result = reproject(cam, (4, 8), (4, 8), (7, 4))
        expected = np.zeros((4, 8), np.float32)
        expected[:, 2:6] = 1.0
        np.testing.assert_array_equal(result, expected)

    def test_known_peak_returns_to_original_location_instead_of_whole_image(self):
        cam = np.zeros((4, 4), np.float32)
        cam[1, 3] = 1.0
        result = reproject(cam, (4, 10), (4, 10), (4, 4))
        # (10 - 4) / 2 = 3: CAM의 오른쪽 끝은 원본 x=6이다.
        expected = np.zeros((4, 10), np.float32)
        expected[1, 6] = 1.0
        np.testing.assert_array_equal(result, expected)

    def test_pixel_centers_map_bilinear_ramp_back_through_resize(self):
        cam = np.tile(np.array([0.0, 1.0], np.float32), (2, 1))
        result = reproject(cam, (4, 8), (2, 4), (2, 2))
        expected_row = np.array([0.0, 0.0, 0.0, .25, .75, 1.0, 0.0, 0.0], np.float32)
        np.testing.assert_allclose(result, np.tile(expected_row, (4, 1)))

    def test_fractional_crop_boundary_has_no_leak_outside_viewed_pixels(self):
        result = reproject(np.ones((4, 4), np.float32), (7, 11), (4, 7), (4, 4))
        # Resize 좌표의 [2, 6) 구간에 중심이 들어오는 원본 픽셀만 포함.
        expected_row = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], np.float32)
        np.testing.assert_array_equal(result, np.tile(expected_row, (7, 1)))

    def test_no_crop_identity_preserves_values_and_does_not_alias_input(self):
        cam = np.arange(12, dtype=np.float32).reshape(3, 4) / 12.0
        before = cam.copy()
        result = reproject(cam, (3, 4), (3, 4), (3, 4))
        np.testing.assert_array_equal(result, before)
        self.assertFalse(np.shares_memory(result, cam))
        self.assertTrue(result.flags.c_contiguous)
        self.assertEqual(result.dtype, np.float32)

    def test_invalid_shapes_and_activation_values_rejected(self):
        for shape in ((0, 4), (-1, 4), (3.5, 4), (4,), (4, 4, 3), (True, False)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                reproject(np.ones((2, 2), np.float32), shape, (4, 4), (4, 4))
        for cam in (np.array([[np.nan]]), np.array([[1.1]]), np.array([[-.1]]),
                    np.zeros((0, 0)), np.ones((2, 2, 1))):
            with self.subTest(cam=cam), self.assertRaises(ValueError):
                reproject(cam, (4, 4), (4, 4), (4, 4))


if __name__ == "__main__":
    unittest.main()
