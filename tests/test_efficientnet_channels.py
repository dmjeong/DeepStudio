"""1채널 stem, 공식 초기 가중치 변환, 구형 모델 복원과 흑백 입출력 검증."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]
from checkpoint import make_checkpoint_metadata
from efficientnet_contract import (adapt_rgb_stem, checkpoint_input_contract, model_input_contract,
                                   NATIVE_INPUT, LEGACY_GRAY_INPUT)
from export_onnx import resolve_checkpoint_spec
from core.project import ProjectManager, TrainingConfig


def checkpoint(version=2, channels=1):
    saved = make_checkpoint_metadata("classify", 2, ["OK", "NG"], (32, 40), channels)
    stem = 3 if version == 1 else channels
    saved.update(engine="efficientnet",
                 model_config={"architecture": "efficientnet_b0", "implementation_version": version},
                 model_state_dict={"features.0.0.weight": np.zeros((32, stem, 3, 3), np.float32)})
    return saved


class ChannelContractTests(unittest.TestCase):
    def test_native_and_legacy_contracts_include_actual_stem(self):
        for version in (1, 2):
            for channels in (1, 3):
                saved = checkpoint(version, channels)
                actual = checkpoint_input_contract(saved, channels)
                expected = 3 if version == 1 else channels
                self.assertEqual(actual["stem_in_channels"], expected)
                self.assertEqual(actual["input_adapter"], LEGACY_GRAY_INPUT if version == 1 and channels == 1 else NATIVE_INPUT)
                self.assertEqual(resolve_checkpoint_spec(saved)["model_config"]["stem_in_channels"], expected)

    def test_conflicting_channels_are_rejected_before_normalization(self):
        for section in (None, "preprocessing", "model_config", "training_config"):
            saved = checkpoint()
            target = saved if section is None else saved.setdefault(section, {})
            target["in_channels"] = 3
            with self.assertRaisesRegex(ValueError, "불일치|충돌"):
                resolve_checkpoint_spec(saved)

    def test_wrong_stem_and_adapter_are_rejected(self):
        for version, wrong_channels in ((1, 1), (2, 3)):
            saved = checkpoint(version)
            saved["model_state_dict"]["features.0.0.weight"] = np.zeros((32, wrong_channels, 3, 3))
            with self.assertRaisesRegex(ValueError, "첫 Conv"):
                checkpoint_input_contract(saved, 1)
        for section, key, wrong in (("model_config", "input_adapter", LEGACY_GRAY_INPUT),
                                    ("preprocessing", "grayscale_adapter", LEGACY_GRAY_INPUT),
                                    ("preprocessing", "color_order", "RGB")):
            saved = checkpoint()
            saved[section][key] = wrong
            with self.assertRaises(ValueError):
                checkpoint_input_contract(saved, 1)
        for version in (True, 0, 3, "2", None):
            with self.assertRaises(ValueError):
                model_input_contract({"implementation_version": version}, 1)

    def test_weight_adaptation_matches_repeated_gray_convolution_at_borders(self):
        rng = np.random.default_rng(42)
        weights = rng.normal(size=(32, 3, 3, 3))
        before = weights.copy()
        adapted = adapt_rgb_stem(weights)
        for height, width in ((8, 10), (9, 11), (1, 1)):
            gray = rng.normal(size=(2, 1, height, width))
            padded = np.pad(gray, ((0, 0), (0, 0), (1, 1), (1, 1)))
            windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))[:, :, ::2, ::2]
            expected = np.einsum("nchwkl,ockl->nohw", np.repeat(windows, 3, axis=1), weights)
            actual = np.einsum("nchwkl,ockl->nohw", windows, adapted)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(before, weights)
        self.assertFalse(np.shares_memory(weights, adapted))
        with self.assertRaises(ValueError):
            adapt_rgb_stem(np.zeros((32, 1, 3, 3)))

    def test_new_project_is_gray_and_old_saved_channel_is_preserved(self):
        self.assertEqual(TrainingConfig().in_channels, 1)
        with tempfile.TemporaryDirectory() as directory:
            project = ProjectManager.create_new("gray", "classify", str(Path(directory) / "new"), ["OK", "NG"])
            self.assertEqual(project.training.in_channels, 1)
            path = Path(ProjectManager.save(project))
            for channels in (1, 3):
                project.training.in_channels = channels
                ProjectManager.save(project)
                self.assertEqual(ProjectManager.load(str(path)).training.in_channels, channels)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["training"].pop("in_channels")
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(ProjectManager.load(str(path)).training.in_channels, 3)

    def test_gray_api_does_not_accept_rgb_models_or_pixels(self):
        from onnx_classifier import OnnxClassifier
        model = object.__new__(OnnxClassifier)
        model.config = {"input_channels": 1}
        gray = np.zeros((5, 7), np.uint8)
        with patch.object(model, "predict_rgb", return_value={"class_id": 0}) as call:
            self.assertEqual(model.predict_gray(gray), {"class_id": 0})
            self.assertIs(call.call_args.args[0], gray)
            with self.assertRaises(ValueError):
                model.predict_gray(np.zeros((5, 7, 3), np.uint8))
            model.config["input_channels"] = 3
            with self.assertRaises(ValueError):
                model.predict_gray(gray)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required")
class GrayPixelTests(unittest.TestCase):
    def test_gray_files_and_arrays_never_expand_to_rgb(self):
        import cv2
        from opencv_preprocess import read_image, convert_channels
        pixels = np.random.default_rng(2).integers(0, 256, (19, 31), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".png", ".bmp", ".tiff", ".jpg"):
                path = Path(directory) / ("흑백" + suffix)
                ok, encoded = cv2.imencode(suffix, pixels)
                self.assertTrue(ok)
                encoded.tofile(path)
                expected = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
                with patch.object(cv2, "cvtColor", side_effect=AssertionError("gray expanded")):
                    actual = read_image(path, 1)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(convert_channels(actual[..., None], 1), expected)

    def test_color_to_gray_keeps_previous_pixel_values(self):
        import cv2
        from opencv_preprocess import read_image, read_rgb, convert_channels
        pixels = np.random.default_rng(7).integers(0, 256, (23, 41, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".png", ".jpg"):
                path = Path(directory) / ("color" + suffix)
                ok, encoded = cv2.imencode(suffix, pixels)
                self.assertTrue(ok)
                encoded.tofile(path)
                expected = convert_channels(read_rgb(path), 1)
                np.testing.assert_array_equal(read_image(path, 1), expected)


TORCH_AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "torchvision"))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch and torchvision required")
class GrayModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous)

    def test_b0_b1_gray_matches_official_model_with_the_same_one_channel_stem(self):
        import torch
        from torchvision import models
        from efficientnet import EfficientNet, load_imagenet
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "imagenet.pth"
            for architecture in ("efficientnet_b0", "efficientnet_b1"):
                official = getattr(models, architecture)(weights=None).eval()
                source = copy.deepcopy(official.state_dict())
                torch.save(source, path)
                model = EfficientNet(architecture, 3, 1).eval()
                head = model.classifier[1].weight.detach().clone()
                provenance = load_imagenet(model, weights_path=path)
                self.assertEqual(model.features[0][0].in_channels, 1)
                self.assertFalse(hasattr(model, "rgb_mean"))
                self.assertEqual(provenance["stem_adaptation"], "rgb_weight_sum")
                torch.testing.assert_close(model.features[0][0].weight, adapt_rgb_stem(source["features.0.0.weight"]), rtol=0, atol=0)
                torch.testing.assert_close(model.classifier[1].weight, head, rtol=0, atol=0)
                official.features[0][0] = torch.nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
                official.classifier[1] = torch.nn.Linear(1280, 3)
                official.load_state_dict(model.state_dict(), strict=True)
                expected_stages, actual_stages, handles = {}, {}, []
                for network, values in ((official, expected_stages), (model, actual_stages)):
                    for index, stage in enumerate(network.features):
                        handles.append(stage.register_forward_hook(
                            lambda _m, _a, output, i=index, v=values: v.__setitem__(i, output.detach().clone())))
                try:
                    with torch.no_grad():
                        for shape in ((1, 1, 64, 80), (2, 1, 65, 79)):
                            inputs = torch.randn(shape)
                            torch.testing.assert_close(model(inputs), official(inputs), rtol=1e-5, atol=1e-6)
                            for index in range(9):
                                torch.testing.assert_close(actual_stages[index], expected_stages[index], rtol=1e-5, atol=1e-6)
                finally:
                    for handle in handles:
                        handle.remove()
                model.train()
                before = model.features[0][0].weight.detach().clone()
                optimizer = torch.optim.SGD(model.parameters(), lr=.001)
                loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 1, 32, 40)), torch.tensor([0, 2]))
                loss.backward()
                self.assertTrue(torch.isfinite(model.features[0][0].weight.grad).all())
                optimizer.step()
                self.assertFalse(torch.equal(before, model.features[0][0].weight))

    def test_native_and_legacy_checkpoint_roundtrip_preserves_predictions(self):
        import torch
        from efficientnet import EfficientNet
        from export_onnx import load_custom_model
        for architecture in ("efficientnet_b0", "efficientnet_b1"):
            for adapter in (NATIVE_INPUT, LEGACY_GRAY_INPUT):
                model = EfficientNet(architecture, 2, 1, input_adapter=adapter).eval()
                saved = make_checkpoint_metadata("classify", 2, ["OK", "NG"], (32, 40), 1)
                saved.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
                if adapter == LEGACY_GRAY_INPUT:
                    with self.assertWarnsRegex(UserWarning, "구형 RGB 확장"):
                        restored = load_custom_model(saved)
                else:
                    restored = load_custom_model(saved)
                inputs = torch.randn(2, 1, 32, 40)
                with torch.no_grad():
                    torch.testing.assert_close(model(inputs), restored(inputs), atol=0, rtol=0)
                self.assertEqual(restored.features[0][0].in_channels, 3 if adapter == LEGACY_GRAY_INPUT else 1)
                saved["model_config"]["implementation_version"] = 2 if adapter == LEGACY_GRAY_INPUT else 1
                with self.assertRaises(ValueError):
                    load_custom_model(saved)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required")
    def test_dataset_and_gui_keep_gray_input_through_first_conv(self):
        import cv2
        import torch
        from dataset import ClassificationDataset, get_classification_transforms
        from efficientnet import EfficientNet
        from core.inference_loading import load_cpu_engine
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images/OK").mkdir(parents=True)
            path = root / "images/OK/gray.png"
            cv2.imwrite(str(path), np.arange(40*48, dtype=np.uint8).reshape(40, 48))
            dataset = ClassificationDataset(str(root / "images"),
                                            get_classification_transforms((32, 40), False, 1), 1)
            with patch.object(cv2, "cvtColor", side_effect=AssertionError("gray expanded")):
                tensor, _ = dataset[0]
            self.assertEqual(tuple(tensor.shape), (1, 32, 40))
            model = EfficientNet(num_classes=2, in_channels=1).eval()
            saved = make_checkpoint_metadata("classify", 2, ["OK", "NG"], (32, 40), 1)
            saved.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
            source = root / "gray.pt"
            torch.save(saved, source)
            engine = load_cpu_engine(source)
            observed = []
            handle = engine.model.features[0][0].register_forward_pre_hook(lambda _m, args: observed.append(tuple(args[0].shape)))
            try:
                result = engine.infer(str(path))
            finally:
                handle.remove()
            self.assertEqual(result.status, "ok", result.error)
            self.assertEqual(observed, [(1, 1, 32, 40)])
            self.assertEqual(result.details["model_config"]["stem_in_channels"], 1)


if __name__ == "__main__":
    unittest.main()
