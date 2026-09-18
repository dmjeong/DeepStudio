"""배포 메타데이터와 검증 실패 시 파일 보존 규약. GPU/모델 다운로드 불필요."""

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "python"),
                str(Path(__file__).resolve().parents[1] / "gui")]
import export_onnx
from checkpoint import make_checkpoint_metadata


class TinyReDetr(torch.nn.Module):
    """Two-output fixture used only when torch is available in the test env."""

    def forward(self, images):
        value = images.mean(dim=(1, 2, 3))
        boxes = torch.stack((value, value, value, value,
                             value, value, value, value), dim=1).reshape(images.shape[0], 2, 4)
        logits = torch.stack((value, value, value,
                              value, value, value), dim=1).reshape(images.shape[0], 2, 3)
        return boxes, logits


class ExportContractTests(unittest.TestCase):
    def checkpoint(self):
        checkpoint = make_checkpoint_metadata(
            "classify", 2, ["정상", "불량"], (96, 128), 1,
            {"backbone_channels": [16, 32, 64, 128, 256], "csp_depth": [1, 1, 1, 1], "dropout": 0.1})
        checkpoint["model_state_dict"] = {}
        return checkpoint

    def test_training_metadata_restores_nondefault_architecture_and_grayscale(self):
        spec = export_onnx.resolve_checkpoint_spec(self.checkpoint())
        self.assertEqual((spec["input_height"], spec["input_width"]), (96, 128))
        self.assertEqual(spec["preprocessing"]["normalize_mean"], [0.449])
        self.assertEqual(spec["model_config"]["backbone_channels"], [16, 32, 64, 128, 256])
        self.assertEqual(spec["model_config"]["csp_depth"], [1, 1, 1, 1])

    def test_overrides_cannot_silently_change_saved_preprocessing(self):
        with self.assertRaisesRegex(ValueError, "불일치"):
            export_onnx.resolve_checkpoint_spec(self.checkpoint(), {"in_channels": 3})

    def test_legacy_checkpoint_requires_explicit_missing_settings(self):
        legacy = {"model_state_dict": {}, "class_names": ["OK", "NG"]}
        with self.assertRaisesRegex(ValueError, "메타데이터 누락"):
            export_onnx.resolve_checkpoint_spec(legacy)
        spec = export_onnx.resolve_checkpoint_spec(legacy, {"task": "classify", "in_channels": 3, "input_size": 224})
        self.assertEqual(spec["num_classes"], 2)

    def test_invalid_normalization_is_rejected(self):
        checkpoint = self.checkpoint()
        checkpoint["preprocessing"]["std"] = [0.0]
        with self.assertRaisesRegex(ValueError, "std"):
            export_onnx.resolve_checkpoint_spec(checkpoint)

    def test_numeric_validation_rejects_nan_shape_mismatch_and_wrong_values(self):
        for actual in (np.array([np.nan, 2.0]), np.array([[1.0, 2.0]]), np.array([99.0, 2.0])):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                export_onnx.validate_outputs(np.array([1.0, 2.0]), actual)
        self.assertTrue(export_onnx.validate_outputs([1., 2.], [1.000001, 2.]))

    def test_numeric_failure_reports_scale_and_the_actual_failing_element(self):
        # The largest absolute error is tolerated on the large logit, but the
        # smaller error on the near-zero logit must still fail the unchanged gate.
        with self.assertRaises(ValueError) as caught:
            export_onnx.validate_outputs([[0., 100000.]], [[2.8339844, 100010.]])
        message = str(caught.exception)
        self.assertIn("최대 오차 10", message)
        self.assertIn("실패 원소 (0, 0): PyTorch=0, ONNX=2.8339844", message)
        self.assertIn("허용 오차의", message)
        self.assertIn("출력 범위", message)
        with self.assertRaisesRegex(ValueError, "PyTorch=0, ONNX=1"):
            export_onnx.validate_outputs([1.], [np.nan])

    def test_classification_profile_allows_small_fp32_reassociation_only(self):
        tolerances = export_onnx.verification_tolerances("classify")
        self.assertTrue(export_onnx.validate_outputs([1.0], [1.00052], **tolerances))
        self.assertTrue(export_onnx.validate_outputs([0.0], [0.00052], **tolerances))
        with self.assertRaises(ValueError):
            export_onnx.validate_outputs([0.0], [0.002], **tolerances)
        with self.assertRaises(ValueError):
            export_onnx.validate_outputs([1.0], [1.01], **tolerances)

    def test_classification_profile_keeps_top1_contract(self):
        self.assertTrue(export_onnx.validate_classification_outputs(
            [[1.0, 0.0]], [[1.00052, 0.00001]]))
        with self.assertRaisesRegex(ValueError, "top-1"):
            export_onnx.validate_classification_outputs([[1.0, 0.9999]], [[0.9998, 1.0000]])

    def test_manifest_preserves_unicode_normalization_and_support_status(self):
        spec = export_onnx.resolve_checkpoint_spec(self.checkpoint())
        with tempfile.TemporaryDirectory() as directory:
            filename = export_onnx.create_inference_config(
                directory, "classify", 2, [96, 128], 1, "모델.onnx", spec["class_names"],
                preprocessing=spec["preprocessing"], config_filename="모델.json")
            manifest = json.loads(Path(filename).read_text(encoding="utf-8"))
        self.assertEqual(manifest["normalize_mean"], [0.449])
        self.assertEqual(manifest["normalize_std"], [0.226])
        self.assertEqual(manifest["class_names"], ["정상", "불량"])
        self.assertEqual(manifest["input_height"], 96)
        self.assertEqual(manifest["input_width"], 128)
        self.assertTrue(manifest["cpp_supported"])
        self.assertEqual(manifest["schema_version"], 5)
        self.assertEqual(manifest["preprocessing"]["resize_implementation"], "opencv_linear_exact_v1")
        self.assertFalse(manifest["preprocessing"]["antialias"])
        from core.inference_region import crop_from_json
        self.assertIsNone(crop_from_json(manifest))

    def test_redetr_manifest_uses_two_outputs_and_native_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = export_onnx.create_inference_config(
                directory, "detect", 2, [640, 640], 3, "redetr.onnx", ["box", "other"],
                preprocessing={"normalize_mean": [0.485, 0.456, 0.406],
                               "normalize_std": [0.229, 0.224, 0.225]},
                backend="redetr_v4", output_names=["pred_boxes", "pred_logits"],
                detection_box_encoding="normalized_cxcywh", config_filename="redetr.json")
            manifest = json.loads(Path(filename).read_text(encoding="utf-8"))
        self.assertTrue(manifest["cpp_supported"])
        self.assertEqual(manifest["schema_version"], 5)
        self.assertEqual(manifest["output_names"], ["pred_boxes", "pred_logits"])
        self.assertEqual(manifest["postprocessing"]["objectness"], "none")

    def test_redetr_manifest_rejects_single_output(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "두 출력"):
            export_onnx.create_inference_config(
                directory, "detect", 2, 640, 3, "redetr.onnx", backend="redetr_v4",
                output_names=["pred_boxes"])

    def test_redetr_module_checkpoint_exports_and_verifies_both_outputs(self):
        checkpoint = {
            "type": "redetr_v4", "backend": "redetr_v4", "task": "detect",
            "num_classes": 3, "in_channels": 3, "input_size": [8, 8],
            "class_names": ["a", "b", "c"],
            "model_config": {"boxes_format": "normalized_cxcywh", "score_activation": "softmax"},
            "model": TinyReDetr().eval(),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "redetr.pt"
            output = Path(directory) / "redetr.onnx"
            torch.save(checkpoint, checkpoint_path)
            result = export_onnx.export_checkpoint(checkpoint_path, output, verify=True, log=lambda _: None)
            manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(result["backend"], "redetr_v4")
        self.assertEqual(manifest["output_names"], ["pred_boxes", "pred_logits"])
        self.assertEqual(manifest["postprocessing"]["class_scores"], "softmax")
        self.assertEqual(manifest["export"]["verification_tolerance"], {"atol": 1e-3, "rtol": 1e-3})
        self.assertTrue(manifest["cpp_supported"])

    def test_sam2_checkpoint_routes_to_multi_graph_exporter(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "sam2.pt"
            output = Path(directory) / "sam2.onnx"
            torch.save({"type": "sam2", "backend": "sam2", "task": "segment"}, checkpoint_path)
            with patch("export_sam2_onnx.export_sam2_checkpoint",
                       return_value={"backend": "sam2", "verification": "passed"}) as exporter:
                result = export_onnx.export_checkpoint(checkpoint_path, output, log=lambda _: None)
        self.assertEqual(result["backend"], "sam2")
        exporter.assert_called_once_with(checkpoint_path, output.parent.resolve(), verify=True,
                                         opset=17, log=unittest.mock.ANY)

    def test_incompatible_resize_metadata_is_rejected(self):
        for key, value in (("resize", "bicubic"), ("resize_implementation", "opencv"),
                           ("antialias", True), ("antialias", 1), ("layout", "NHWC"), ("value_scale", 1)):
            checkpoint = self.checkpoint()
            checkpoint["preprocessing"][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "전처리"):
                export_onnx.resolve_checkpoint_spec(checkpoint)

    def test_backend_detection_does_not_treat_patchcore_as_custom_model(self):
        self.assertEqual(export_onnx.checkpoint_backend({"type": "patchcore"}), "patchcore")
        with self.assertRaises(ValueError):
            export_onnx.checkpoint_backend({"train_args": {}})
        with self.assertRaises(ValueError):
            export_onnx.resolve_checkpoint_spec({"type": "patchcore"})

    def test_failed_verification_preserves_existing_release_files(self):
        class Generator:
            def manual_seed(self, seed):
                return self
        fake_torch = types.SimpleNamespace(load=lambda *a, **k: self.checkpoint(),
                                          Generator=Generator, randn=lambda *a, **k: np.zeros((1, 1, 96, 128)),
                                          zeros_like=lambda x: object())
        fake_onnx = types.SimpleNamespace(checker=types.SimpleNamespace(check_model=lambda path: None))
        def export_candidate(model, dummy, path, *args):
            Path(path).write_bytes(b"candidate")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            checkpoint.write_bytes(b"weights")
            output = Path(directory) / "model.onnx"
            output.write_bytes(b"previous model")
            config = output.with_suffix(".json")
            config.write_bytes(b"previous config")
            with patch.dict(sys.modules, {"torch": fake_torch, "onnx": fake_onnx}), \
                 patch.object(export_onnx, "load_custom_model", return_value=object()), \
                 patch.object(export_onnx, "export_to_onnx", side_effect=export_candidate), \
                 patch.object(export_onnx, "verify_onnx", side_effect=ValueError("verification failed")):
                with self.assertRaisesRegex(ValueError, "verification failed"):
                    export_onnx.export_checkpoint(checkpoint, output, log=lambda _: None)
            self.assertEqual(output.read_bytes(), b"previous model")
            self.assertEqual(config.read_bytes(), b"previous config")
            self.assertFalse(list(Path(directory).glob(".onnx-export-*")))


if __name__ == "__main__":
    unittest.main()
