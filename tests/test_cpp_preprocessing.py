"""Compare the compiled production C++ preprocessing with OpenCV and NumPy."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
import os
import shlex
import importlib.util
ROOT = Path(__file__).resolve().parents[1]


class CppPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not importlib.util.find_spec("cv2"):
            raise unittest.SkipTest("Python OpenCV unavailable")
        existing = os.environ.get("PREPROCESS_PROBE")
        if existing:
            cls.program = existing
            return
        if not shutil.which("pkg-config") or subprocess.run(["pkg-config", "--exists", "opencv4"]).returncode:
            raise unittest.SkipTest("OpenCV C++ development package unavailable")
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise unittest.SkipTest("C++17 compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.program = str(Path(cls.directory.name) / "preprocess_probe.exe")
        flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "opencv4"], text=True))
        subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "cpp/include"),
                        str(ROOT / "cpp/tests/preprocess_probe.cpp"), "-o", cls.program, *flags], check=True, capture_output=True)

    def check_pixels(self, pixels, output_size, grayscale=False, normalized=False):
        import cv2
        image = pixels
        channels = 1 if pixels.ndim == 2 else 3
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        expected = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR_EXACT)
        output = subprocess.check_output([self.program, str(pixels.shape[1]), str(pixels.shape[0]), str(channels),
                                          *map(str, output_size), str(int(grayscale)), str(int(normalized))], input=pixels.tobytes())
        if normalized:
            chw = expected[None] if expected.ndim == 2 else expected.transpose(2, 0, 1)
            mean, std = ([.449], [.226]) if chw.shape[0] == 1 else ([.485, .456, .406], [.229, .224, .225])
            expected = (chw.astype(np.float32) / np.float32(255) - np.array(mean, np.float32)[:, None, None]) / np.array(std, np.float32)[:, None, None]
            np.testing.assert_allclose(np.frombuffer(output, np.float32).reshape(expected.shape), expected, rtol=0, atol=1e-6)
        else:
            np.testing.assert_array_equal(np.frombuffer(output, np.uint8).reshape(expected.shape), expected)

    def test_random_up_down_and_non_square_resize(self):
        random = np.random.default_rng(21)
        for index in range(100):
            width, height, out_width, out_height = map(int, random.integers(1, 301, 4))
            shape = (height, width) if index % 2 else (height, width, 3)
            with self.subTest(index=index, shape=shape, output=(out_width, out_height)):
                self.check_pixels(random.integers(0, 256, shape, dtype=np.uint8), (out_width, out_height))

    def test_gray_conversion_and_normalized_tensor(self):
        pixels = np.random.default_rng(55).integers(0, 256, (251, 257, 3), dtype=np.uint8)
        self.check_pixels(pixels, (257, 251), grayscale=True)
        for shape in ((224, 224), (240, 240), (37, 43), (300, 320)):
            for grayscale in (False, True):
                self.check_pixels(pixels, shape, grayscale=grayscale, normalized=True)


    def test_single_pixel_identity_and_extremes(self):
        for value in (0, 255):
            for shape in ((1, 1), (1, 15, 3), (17, 1, 3)):
                pixels = np.full(shape, value, np.uint8)
                self.check_pixels(pixels, (shape[1], shape[0]))
                self.check_pixels(pixels, (7, 9), normalized=True)


if __name__ == "__main__":
    unittest.main()
