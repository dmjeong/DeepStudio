"""체크포인트 입력 크기의 복원, 검증과 명시적 변경 규칙."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))
from core.inference_settings import input_shape, require_saved_input_shape


class InferenceSettingsTests(unittest.TestCase):
    def test_invalid_sizes_are_rejected_instead_of_rounded_or_defaulted(self):
        for value in (0, -1, True, 12.5, "640", [], [1, 2, 3], [64, 0], [64, False], [float("nan")]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                input_shape(value)

    def test_fixed_checkpoint_size_override_is_never_silently_ignored(self):
        require_saved_input_shape(None, (96, 64))
        require_saved_input_shape(224, (224, 224))
        with self.assertRaisesRegex(ValueError, "체크포인트"):
            require_saved_input_shape(224, (512, 512))
