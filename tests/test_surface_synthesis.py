"""합성의 재현성, ROI 경계, 변경 마스크, 원본 정밀도 검증."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
from core.defect_generator import DefectParams, DefectType, generate_sample, generate_batch
from core.image_display import display_rgb


class SurfaceSynthesisTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:90, :130]
        self.image = np.stack([80 + xx // 2, 80 + yy, 100 + (xx + yy) // 4], axis=2).astype(np.uint8)
        self.roi = np.zeros((90, 130), dtype=np.uint8)
        self.roi[12:75, 20:110] = 1
        self.roi[30:45, 45:85] = 0

    def test_every_pattern_preserves_16bit_and_changes_only_masked_pixels(self):
        for kind in DefectType:
            for gray in (False, True):
                for dtype in (np.uint8, np.uint16):
                    with self.subTest(kind=kind, gray=gray, dtype=dtype):
                        image = self.image[..., 0] if gray else self.image
                        image = image.astype(dtype) * (257 if dtype == np.uint16 else 1)
                        before = image.copy()
                        sample = generate_sample(image, DefectParams(seed=12, types=[kind], count=3), roi_mask=self.roi)
                        self.assertEqual(sample.image.dtype, dtype)
                        self.assertEqual(sample.image.shape, image.shape)
                        self.assertGreater(np.count_nonzero(sample.mask), 0)
                        changed = (sample.image != image)
                        if not gray:
                            changed = changed.any(axis=2)
                        np.testing.assert_array_equal(sample.mask > 0, changed)
                        np.testing.assert_array_equal(sample.image[self.roi == 0], image[self.roi == 0])
                        np.testing.assert_array_equal(image, before)

    def test_seed_replays_exact_pixels_without_mutating_global_rng(self):
        np.random.seed(123)
        state = np.random.get_state()
        params = DefectParams(seed=789, count=5, types=list(DefectType), mix_types=True)
        a, b = generate_sample(self.image, params), generate_sample(self.image, params)
        np.testing.assert_array_equal(a.image, b.image)
        np.testing.assert_array_equal(a.mask, b.mask)
        self.assertEqual(a.recipe, b.recipe)
        self.assertGreater(len(set(a.types)), 1)
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])
        self.assertEqual(state[2:], np.random.get_state()[2:])
        self.assertFalse(np.array_equal(a.image, generate_sample(self.image, replace(params, seed=790)).image))

    def test_batch_seed_replays_but_samples_are_distinct(self):
        params = DefectParams(seed=3)
        first, second = generate_batch(self.image, params, 3), generate_batch(self.image, params, 3)
        for (a, _), (b, _) in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(first[0][0], first[1][0]))

    def test_reference_texture_is_used_and_not_modified(self):
        donor = np.zeros((31, 27, 3), dtype=np.uint16)
        donor[::2, :, 0] = 65535
        before = donor.copy()
        params = DefectParams(seed=9, types=[DefectType.TEXTURE])
        a = generate_sample(self.image, params, texture=donor)
        b = generate_sample(self.image, params)
        self.assertFalse(np.array_equal(a.image, b.image))
        np.testing.assert_array_equal(donor, before)

    def test_bad_input_and_empty_roi_fail_before_synthesis(self):
        for image, params, roi in ((self.image.astype(float), DefectParams(), None),
                                  (self.image, DefectParams(count=0), None),
                                  (self.image, DefectParams(types=[]), None),
                                  (self.image, DefectParams(), np.zeros((90, 130))),
                                  (self.image, DefectParams(), np.ones((10, 10)))):
            with self.subTest(params=params), self.assertRaises(ValueError):
                generate_sample(image, params, roi_mask=roi)

    def test_qt_display_converts_uint16_without_byte_wrap(self):
        image = np.array([[0, 32768, 65535]], dtype=np.uint16)
        pixels = display_rgb(image)
        np.testing.assert_array_equal(pixels[0, :, 0], [0, 128, 255])
        self.assertEqual(pixels.shape, (1, 3, 3))
        self.assertTrue(pixels.flags.c_contiguous)
        np.testing.assert_array_equal(image, [[0, 32768, 65535]])

    def test_display_accepts_strided_float_rgb_and_rejects_nan(self):
        pixels = display_rgb(np.ones((9, 7, 3), dtype=np.float32)[:, ::-1])
        self.assertTrue(pixels.flags.c_contiguous)
        self.assertTrue((pixels == 255).all())
        with self.assertRaises(ValueError):
            display_rgb(np.full((2, 2, 3), np.nan))


if __name__ == "__main__":
    unittest.main()
