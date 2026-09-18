"""표시 범위는 원본 픽셀과 추론 결과를 보존하면서 색상만 바꿔야 한다."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))

from core.heatmap import normalize_activation_map, render_heatmap


class HeatmapDisplayTests(unittest.TestCase):
    def setUp(self):
        self.activation = np.array([[0.0, 0.2, 0.5, 0.8, 1.0]], dtype=np.float32)
        self.original = np.array([[[12, 34, 56], [71, 82, 93], [101, 123, 145],
                                   [172, 193, 214], [225, 236, 247]]], dtype=np.uint8)

    def test_below_lower_preserves_exact_original_pixels(self):
        heatmap, overlay = render_heatmap(self.activation, self.original, lower=0.5)
        np.testing.assert_array_equal(overlay[0, :2], self.original[0, :2])
        np.testing.assert_array_equal(heatmap[0, :2], np.zeros((2, 3), dtype=np.uint8))
        self.assertFalse(np.array_equal(overlay[0, 2:], self.original[0, 2:]))
        self.assertTrue(heatmap[0, 2].any())  # 최소값 자체는 표시 범위에 포함

    def test_zero_map_uses_uniform_cold_color_at_full_range(self):
        heatmap, overlay = render_heatmap(np.zeros((2, 2)), self.original, alpha=1.0)
        expected = np.zeros_like(self.original)
        expected[..., 2] = 127
        np.testing.assert_array_equal(heatmap, expected)
        np.testing.assert_array_equal(overlay, expected)

    def test_zero_activation_is_included_at_default_lower(self):
        heatmap, overlay = render_heatmap(self.activation, self.original, alpha=1.0)
        np.testing.assert_array_equal(overlay[0, 0], [0, 0, 127])
        np.testing.assert_array_equal(heatmap[0, 0], [0, 0, 127])

    def test_valid_zero_is_colored_but_unseen_positive_pixel_is_not(self):
        mask = np.array([[True, True, True, False, False]])
        heatmap, overlay = render_heatmap(self.activation, self.original, alpha=1.0, valid_mask=mask)
        np.testing.assert_array_equal(overlay[0, 0], [0, 0, 127])
        np.testing.assert_array_equal(overlay[~mask], self.original[~mask])
        np.testing.assert_array_equal(heatmap[~mask], np.zeros((2, 3), np.uint8))
        np.testing.assert_array_equal(mask, [[True, True, True, False, False]])

    def test_valid_mask_requires_boolean_original_resolution(self):
        for mask in (np.ones((1, 5), np.uint8), np.ones((2, 5), bool), np.ones(5, bool)):
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                render_heatmap(self.activation, self.original, valid_mask=mask)

    def test_upper_saturates_and_remaps_midpoint(self):
        heatmap, _ = render_heatmap(self.activation, self.original,
                                    alpha=1.0, lower=0.2, upper=0.8)
        np.testing.assert_array_equal(heatmap[0, 3], heatmap[0, 4])
        np.testing.assert_array_equal(heatmap[0, 3], [127, 0, 0])
        np.testing.assert_allclose(heatmap[0, 2], [127, 255, 127], atol=1)

    def test_zero_alpha_is_exact_original_and_one_alpha_is_heatmap(self):
        _, transparent = render_heatmap(self.activation, self.original, alpha=0.0)
        np.testing.assert_array_equal(transparent, self.original)
        heatmap, opaque = render_heatmap(self.activation, self.original, alpha=1.0)
        np.testing.assert_array_equal(opaque, heatmap)

    def test_float_map_is_resized_before_range_mask(self):
        activation = np.array([[0.0, 1.0]], dtype=np.float32)
        original = np.full((1, 4, 3), 50, dtype=np.uint8)
        heatmap, overlay = render_heatmap(activation, original, alpha=1.0, lower=0.5)
        np.testing.assert_array_equal(overlay[0, :2], original[0, :2])
        self.assertTrue(heatmap[0, 2].any())  # 선형 보간된 0.75 값 표시
        np.testing.assert_array_equal(heatmap[0, 2], [127, 255, 127])

    def test_render_does_not_mutate_cached_map_or_source(self):
        activation_before, original_before = self.activation.copy(), self.original.copy()
        first = render_heatmap(self.activation, self.original, lower=0.1, upper=0.9)
        render_heatmap(self.activation, self.original, lower=0.7, upper=1.0)
        repeated = render_heatmap(self.activation, self.original, lower=0.1, upper=0.9)
        np.testing.assert_array_equal(self.activation, activation_before)
        np.testing.assert_array_equal(self.original, original_before)
        for expected, actual in zip(first, repeated):
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.uint8)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertFalse(np.shares_memory(actual, self.original))

    def test_noncontiguous_rgb_and_one_pixel_activation_are_supported(self):
        original = self.original[:, ::-1]
        heatmap, overlay = render_heatmap(np.ones((1, 1)), original, alpha=1.0)
        self.assertEqual(heatmap.shape, original.shape)
        self.assertTrue(overlay.flags.c_contiguous)
        np.testing.assert_array_equal(heatmap, overlay)

    def test_invalid_display_settings_fail_before_render(self):
        settings = [dict(lower=0.5, upper=0.5), dict(lower=0.8, upper=0.2),
                    dict(lower=-0.1), dict(upper=1.1), dict(lower=np.nan),
                    dict(upper=np.inf), dict(alpha=np.nan), dict(alpha=-0.1),
                    dict(alpha=1.1), dict(alpha="invalid")]
        for kwargs in settings:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                render_heatmap(self.activation, self.original, **kwargs)

    def test_invalid_activation_maps_are_rejected(self):
        maps = [np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-0.1]]),
                np.array([[1.1]]), np.empty((0, 1)), np.zeros((1, 1, 1)),
                np.array([0.5]), np.array([[1j]]), np.array([["0.5"]])]
        for values in maps:
            with self.subTest(values=values), self.assertRaises(ValueError):
                render_heatmap(values, self.original)

    def test_non_rgb_or_non_uint8_sources_are_rejected(self):
        images = [np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2, 4), dtype=np.uint8),
                  np.zeros((2, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.uint8)]
        for image in images:
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                render_heatmap(self.activation, image)


class ActivationNormalizationTests(unittest.TestCase):
    def test_raw_score_normalization_preserves_relative_strength_and_shape(self):
        raw = np.array([[-4.0, 0.0, 4.0]])
        normalized = normalize_activation_map(raw)
        np.testing.assert_array_equal(normalized, [[0.0, 0.5, 1.0]])
        np.testing.assert_array_equal(raw, [[-4.0, 0.0, 4.0]])
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(normalized.shape, raw.shape)

    def test_constant_scores_produce_zero_even_if_large_and_positive(self):
        for value in (0, 255, -10, 1e30):
            with self.subTest(value=value):
                normalized = normalize_activation_map(np.full((1, 3), value))
                np.testing.assert_array_equal(normalized, np.zeros((1, 3)))

    def test_finite_large_scores_do_not_overflow_during_normalization(self):
        normalized = normalize_activation_map(np.array([[-1e308, 0.0, 1e308]]))
        np.testing.assert_array_equal(normalized, [[0.0, 0.5, 1.0]])

    def test_invalid_raw_scores_fail_instead_of_hiding_missing_data(self):
        for values in (np.array([[np.nan]]), np.array([[np.inf]]), np.ones(3)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_activation_map(values)


@unittest.skipUnless(importlib.util.find_spec("torch"), "실제 역전파 테스트에는 PyTorch 필요")
class GradCAMActivationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from core.gradcam import GradCAM

        class TinyClassifier(torch.nn.Module):
            task = "classify"

            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Module()
                self.backbone.stage4 = torch.nn.Conv2d(1, 1, 1, bias=False)
                torch.nn.init.ones_(self.backbone.stage4.weight)
                self.forward_calls = 0

            def forward(self, value):
                self.forward_calls += 1
                mean = self.backbone.stage4(value).mean(dim=(1, 2, 3))
                return torch.stack([mean, -mean], dim=1)

        self.torch = torch
        self.model = TinyClassifier()
        self.cam = GradCAM(self.model)
        self.addCleanup(self.cam.release)
        self.input = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])

    def test_cached_activation_repaints_without_another_model_forward(self):
        with self.torch.inference_mode():
            activation = self.cam.generate_activation(self.input)
        np.testing.assert_allclose(activation, [[0.0, 1 / 3], [2 / 3, 1.0]])
        self.assertEqual(activation.dtype, np.float32)
        original = np.full((4, 4, 3), 100, dtype=np.uint8)
        _, first = render_heatmap(activation, original)
        _, second = render_heatmap(activation, original, lower=0.6)
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(self.model.forward_calls, 1)
        self.assertTrue(self.model.training)
        self.assertIsNone(self.cam._activations)
        self.assertIsNone(self.cam._gradients)

    def test_compatibility_generate_uses_same_range_rendering(self):
        original = np.full((4, 6, 3), 50, dtype=np.uint8)
        activation = self.cam.generate_activation(self.input)
        expected = render_heatmap(activation, original, alpha=0.7, lower=0.2, upper=0.8)
        actual = self.cam.generate(self.input, original, alpha=0.7, lower=0.2, upper=0.8)
        for expected_image, actual_image in zip(expected, actual):
            np.testing.assert_array_equal(actual_image, expected_image)

    def test_single_pixel_positive_layer_preserves_uniform_contribution(self):
        activation = self.cam.generate_activation(self.torch.ones((1, 1, 1, 1)))
        self.assertEqual(activation.shape, (1, 1))
        np.testing.assert_array_equal(activation, [[1.0]])

    def test_constant_zero_layer_stays_zero_without_fabricated_hotspots(self):
        activation = self.cam.generate_activation(self.torch.zeros((1, 1, 2, 2)))
        np.testing.assert_array_equal(activation, np.zeros((2, 2)))

    def test_non_tensor_model_output_is_rejected(self):
        forward = self.model.forward

        def saturated(value):
            logits = forward(value) + value.new_tensor([[100.0, -100.0]])
            return logits.softmax(dim=1), logits

        self.model.forward = saturated
        probability_input = self.input.clone().requires_grad_(True)
        probability = self.model(probability_input)[0][0, 0]
        self.assertEqual(probability.item(), 1.0)
        # 이 입력에서 확률의 미분은 실제로 0이다. logits의 미분은 살아 있다.
        probability_gradient = self.torch.autograd.grad(probability, probability_input)[0]
        self.assertEqual(self.torch.count_nonzero(probability_gradient).item(), 0)
        with self.assertRaisesRegex(ValueError, "텐서"):
            self.cam.generate_activation(self.input, target_class=0)

    def test_frozen_model_still_uses_input_gradient(self):
        self.model.requires_grad_(False)
        with self.torch.inference_mode():
            inferred_input = self.input.clone()
        activation = self.cam.generate_activation(inferred_input)
        np.testing.assert_allclose(activation, [[0.0, 1 / 3], [2 / 3, 1.0]])
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_inference_mode_parameters_and_buffers_are_safe_and_restored(self):
        with self.torch.inference_mode():
            parameter = self.torch.nn.Parameter(
                self.torch.ones_like(self.model.backbone.stage4.weight), requires_grad=False)
            gain = self.torch.tensor(2.0)
        self.model.backbone.stage4.weight = parameter
        self.model.register_buffer("gain", gain)
        forward = self.model.forward
        self.model.forward = lambda value: forward(value) * self.model.gain
        # predict에서 생성된 inference 텐서가 입력 미분을 막는 조건을 재현한다.
        with self.assertRaises(RuntimeError):
            self.model(self.input.clone().requires_grad_(True))
        activation = self.cam.generate_activation(self.input)
        np.testing.assert_allclose(activation, [[0.0, 1 / 3], [2 / 3, 1.0]])
        self.assertIs(self.model.backbone.stage4.weight, parameter)
        self.assertIs(self.model.gain, gain)
        self.assertTrue(self.torch.is_inference(parameter))
        self.assertTrue(self.torch.is_inference(gain))
        with self.torch.inference_mode():
            np.testing.assert_array_equal(self.model(self.input).numpy(), [[3.0, -3.0]])

    def test_failure_restores_inference_parameter_identity(self):
        with self.torch.inference_mode():
            parameter = self.torch.nn.Parameter(
                self.torch.ones_like(self.model.backbone.stage4.weight), requires_grad=False)
        self.model.backbone.stage4.weight = parameter
        with self.assertRaises(IndexError):
            self.cam.generate_activation(self.input, target_class=99)
        self.assertIs(self.model.backbone.stage4.weight, parameter)
        self.assertFalse(self.model.backbone.stage4._forward_hooks)

    def test_existing_gradients_and_mixed_module_modes_are_preserved(self):
        self.model.train()
        self.model.backbone.stage4.eval()
        parameter = self.model.backbone.stage4.weight
        original_gradient = self.torch.full_like(parameter, 7.0)
        parameter.grad = original_gradient
        self.cam.generate_activation(self.input)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.backbone.stage4.training)
        self.assertIs(parameter.grad, original_gradient)
        self.assertTrue(self.torch.equal(parameter.grad, self.torch.full_like(parameter, 7.0)))

    def test_downstream_inplace_activation_does_not_conflict_with_capture(self):
        def inplace_forward(value):
            features = self.model.backbone.stage4(value)
            mean = features.relu_().mean(dim=(1, 2, 3))
            return self.torch.stack([mean, -mean], dim=1)

        self.model.forward = inplace_forward
        activation = self.cam.generate_activation(self.input - 1)
        np.testing.assert_allclose(activation, [[0, 0], [0.5, 1.0]])

    def test_ordinary_inference_has_no_cam_hooks_or_retained_tensors(self):
        for _ in range(2):
            with self.torch.inference_mode():
                self.model(self.input)
            self.assertIsNone(self.cam._activations)
            self.assertIsNone(self.cam._gradients)
            self.assertFalse(self.model.backbone.stage4._forward_hooks)
            self.assertFalse(self.model.backbone.stage4._backward_hooks)
            self.cam.generate_activation(self.input)
            self.assertIsNone(self.cam._tensor_hook)

    def test_failure_restores_training_mode_and_clears_hook_buffers(self):
        with self.assertRaises(IndexError):
            self.cam.generate_activation(self.input, target_class=5)
        self.assertTrue(self.model.training)
        self.assertIsNone(self.cam._activations)
        self.assertIsNone(self.cam._gradients)
        self.cam.release()
        self.assertFalse(self.model.backbone.stage4._forward_hooks)
        self.assertFalse(self.model.backbone.stage4._backward_hooks)

    def test_multiple_images_are_rejected_before_model_execution(self):
        with self.assertRaises(ValueError):
            self.cam.generate_activation(self.torch.zeros((2, 1, 4, 4)))
        self.assertEqual(self.model.forward_calls, 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "실제 CustomCSP 역전파에는 PyTorch 필요")
class CustomModelGradCAMTests(unittest.TestCase):
    def test_all_four_custom_tasks_produce_spatial_maps_and_visible_overlays(self):
        import torch
        from core.gradcam import GradCAM

        sys.path.insert(0, str(ROOT / "python"))
        from model import CustomCSP

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        image = torch.linspace(0, 1, 64 * 96).reshape(1, 1, 64, 96).repeat(1, 3, 1, 1)
        original = np.full((64, 96, 3), 90, dtype=np.uint8)
        for task in ("classify", "segment", "detect", "anomaly"):
            with self.subTest(task=task):
                model = CustomCSP(task=task, num_classes=2,
                               backbone_channels=[8, 16, 24, 32, 48],
                               csp_depth=[1, 1, 1, 1], dropout=0).eval()
                # 양의 경로와 공간 대비가 있는 실제 모델을 구성한다.
                with torch.no_grad():
                    for module in model.modules():
                        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                            module.weight.fill_(1 / module.weight[0].numel())
                            if module.bias is not None:
                                module.bias.zero_()
                model.requires_grad_(False)
                cam = GradCAM(model)
                try:
                    target = torch.zeros_like(image) if task == "anomaly" else None
                    activation = cam.generate_activation(image, target_tensor=target)
                    self.assertEqual(activation.shape, (2, 3))
                    self.assertTrue(np.isfinite(activation).all())
                    self.assertEqual(float(activation.max()), 1.0)
                    self.assertEqual(float(activation.min()), 0.0)
                    _, overlay = render_heatmap(activation, original)
                    self.assertFalse(np.array_equal(overlay, original))
                    self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                finally:
                    cam.release()


if __name__ == "__main__":
    unittest.main()
