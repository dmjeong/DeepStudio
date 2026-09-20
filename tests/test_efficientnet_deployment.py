"""EfficientNet 전용 배포, 실패 진단, 융합 수치와 optimizer 회귀 검사."""
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
from export_onnx import create_inference_config, export_checkpoint
from core.project import ProjectData
from core.training_modes import validate_training_options
from onnx_classifier import validate_config


class DeploymentPolicyTests(unittest.TestCase):
    def test_new_classification_defaults_and_retired_modes(self):
        project = ProjectData()
        self.assertEqual(project.training.training_mode, "efficientnet_finetune")
        self.assertEqual(project.training.in_channels, 1)
        self.assertTrue(project.training.efficientnet_no_decay)
        validate_training_options(project)
        for mode in ("unsupported_finetune", "unsupported_transfer", "unsupported_resume"):
            project.training.training_mode = mode
            with self.assertRaisesRegex(ValueError, "지원하지 않는"):
                validate_training_options(project)

    def test_retired_export_does_not_write_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "지원하지 않는"):
                create_inference_config(directory, "classify", 2, 224, 3, "old.onnx", backend="unsupported")
            self.assertFalse(list(Path(directory).iterdir()))

    def test_failure_writes_diagnostic_and_preserves_release(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            output.write_bytes(b"previous")
            with patch("export_onnx._export_checkpoint", side_effect=RuntimeError("synthetic exporter failure")):
                with self.assertRaisesRegex(ValueError, "synthetic exporter failure"):
                    export_checkpoint("weights.pt", output, log=lambda _: None)
            diagnostic = json.loads(output.with_suffix(".export-error.json").read_text(encoding="utf-8"))
            self.assertIn("python", diagnostic["versions"])
            self.assertIn("synthetic exporter failure", diagnostic["traceback"])
            self.assertEqual(diagnostic["opset"], 17)
            self.assertEqual(output.read_bytes(), b"previous")

    def test_runtime_manifest_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = make_checkpoint_metadata("classify", 2, ["OK", "NG"], 32, 3)
            path = create_inference_config(directory, "classify", 2, 32, 3, "model.onnx", ["OK", "NG"],
                                           metadata["preprocessing"], backend="efficientnet")
            config = json.loads(Path(path).read_text(encoding="utf-8"))
            validate_config(config)
            for change in ({"input_height": True}, {"normalize_std": [0, 0, 0]},
                           {"backend": "unsupported"}, {"class_names": ["wrong"]}):
                with self.assertRaises(ValueError):
                    validate_config({**config, **change})


AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "torchvision", "onnx", "onnxruntime", "cv2"))


@unittest.skipUnless(AVAILABLE, "PyTorch, torchvision, ONNX, ONNX Runtime and OpenCV required")
class EfficientNetDeploymentRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous)

    def test_fusion_preserves_source_parameters_outputs_and_gradcam(self):
        import torch
        from efficientnet import EfficientNet, prepare_for_inference
        from core.gradcam import GradCAM
        for architecture in ("efficientnet_b0", "efficientnet_b1"):
            torch.manual_seed(73)
            model = EfficientNet(architecture, num_classes=3, in_channels=3).eval()
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.running_mean.uniform_(-.1, .1)
                    module.running_var.uniform_(.8, 1.2)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            optimized = prepare_for_inference(model)
            sample = torch.randn(1, 3, 64, 80)
            with torch.no_grad():
                torch.testing.assert_close(model(sample), optimized(sample), atol=1e-4, rtol=1e-4)
            self.assertGreater(optimized.inference_optimization["conv_bn_fused"], 0)
            self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in optimized.modules()))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)
            cam = GradCAM(optimized)
            try:
                self.assertTrue(np.isfinite(cam.generate_activation(sample)).all())
            finally:
                cam.release()

    def test_real_b0_b1_rgb_gray_export_dynamic_batch_and_opencv_input(self):
        import torch
        from efficientnet import EfficientNet
        from export_onnx import validate_outputs
        from onnx_classifier import OnnxClassifier
        from dataset import get_classification_transforms
        from opencv_preprocess import convert_channels
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for architecture in ("efficientnet_b0", "efficientnet_b1"):
                for channels in (1, 3):
                    model = EfficientNet(architecture, 3, channels).eval()
                    ckpt = make_checkpoint_metadata("classify", 3, ["a", "b", "c"], (32, 40), channels)
                    ckpt.update(engine="efficientnet", model_config=model.checkpoint_config(),
                                model_state_dict=model.state_dict())
                    source = root / "source.pt"
                    torch.save(ckpt, source)
                    result = export_checkpoint(source, root / "model.onnx", dynamic_batch=True, log=lambda _: None)
                    import onnx
                    graph = onnx.load(str(root / "model.onnx")).graph
                    self.assertEqual(graph.input[0].type.tensor_type.shape.dim[1].dim_value, channels)
                    first_conv = next(node for node in graph.node if node.op_type == "Conv")
                    initializers = {value.name: value for value in graph.initializer}
                    self.assertEqual(initializers[first_conv.input[1]].dims[1], channels)
                    runtime = OnnxClassifier(result["config_path"], num_threads=2)
                    self.assertEqual(runtime.config["model_config"]["stem_in_channels"], channels)
                    rgb = np.random.default_rng(7).integers(0, 256, (43, 55, 3), dtype=np.uint8)
                    tensor = runtime.preprocess(rgb)
                    transform = get_classification_transforms((32, 40), is_train=False, in_channels=channels)
                    expected = transform(convert_channels(rgb, channels)).numpy()[None]
                    np.testing.assert_allclose(tensor, expected, rtol=0, atol=1e-6)
                    batch = np.repeat(tensor, 2, axis=0)
                    with torch.no_grad():
                        logits = model(torch.from_numpy(batch)).numpy()
                    validate_outputs(logits, runtime.logits(batch))
                    self.assertEqual(result["verification"], "passed")
                    self.assertEqual(runtime.predict_rgb(rgb)["class_id"], int(logits[0].argmax()))
                    if channels == 1:
                        import cv2
                        gray = convert_channels(rgb, 1)
                        expected_gray = runtime.predict_rgb(gray)
                        with patch.object(cv2, "cvtColor", side_effect=AssertionError("gray expanded")):
                            actual_gray = runtime.predict_gray(gray[..., None])
                        self.assertEqual(actual_gray["class_id"], expected_gray["class_id"])
                        np.testing.assert_array_equal(actual_gray["probabilities"], expected_gray["probabilities"])
                        file = root / "gray.png"
                        cv2.imwrite(str(file), gray)
                        with patch.object(cv2, "cvtColor", side_effect=AssertionError("gray expanded")):
                            self.assertEqual(runtime.predict_file(file)["class_id"], actual_gray["class_id"])

    def test_optimizer_excludes_only_bias_and_norm_from_decay(self):
        from efficientnet import EfficientNet
        from core.efficientnet_trainer import EfficientNetTrainWorker
        project = ProjectData()
        worker = EfficientNetTrainWorker(project)
        model = EfficientNet(num_classes=3)
        optimizer = worker._create_optimizer(model, project.training, project.model)
        groups = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(len(groups), len(list(model.parameters())))
        for name, parameter in model.named_parameters():
            group = groups[id(parameter)]
            expected = 0.0 if parameter.ndim <= 1 or name.endswith(".bias") else project.training.weight_decay
            self.assertEqual(group["weight_decay"], expected)
            multiplier = project.model.backbone_lr_mult if name.startswith("features.") else 1
            self.assertEqual(group["lr"], project.training.learning_rate * multiplier)


if __name__ == "__main__":
    unittest.main()
