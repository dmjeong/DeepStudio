"""Offline SAM2 two-graph exporter contract tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from export_sam2_onnx import export_sam2_model, export_official_sam2_checkpoint


class TinySamEncoder(torch.nn.Module):
    def forward(self, input_image):
        return input_image.mean(dim=1, keepdim=True)


class TinySamDecoder(torch.nn.Module):
    def forward(self, image_embeddings, point_coords, point_labels, mask_input,
                has_mask_input, orig_im_size):
        # Keep every prompt input in the graph while retaining a deterministic
        # mask fixture for the exporter and ONNX Runtime parity check.
        prompt_bias = (point_coords.mean() + point_labels.float().mean() * 1e-3
                       + mask_input.mean() * 1e-3 + has_mask_input.mean() * 1e-3
                       + orig_im_size.mean() * 1e-6)
        logits = image_embeddings[:, :, :2, :2] + prompt_bias
        scores = logits.mean(dim=(1, 2, 3), keepdim=False).reshape(-1, 1)
        return logits, scores


class MultiFeatureSamEncoder(torch.nn.Module):
    def forward(self, input_image):
        return {
            "image_embeddings": input_image.mean(dim=1, keepdim=True),
            "image_features_0": input_image[:, :1],
            "image_features_1": input_image[:, 1:2],
        }


class MultiFeatureSamDecoder(torch.nn.Module):
    def forward(self, image_embeddings, image_features_0, image_features_1,
                point_coords, point_labels, mask_input, has_mask_input, orig_im_size):
        prompt_bias = (point_coords.mean() + point_labels.float().mean() * 1e-3
                       + mask_input.mean() * 1e-3 + has_mask_input.mean() * 1e-3
                       + orig_im_size.mean() * 1e-6)
        logits = (image_embeddings[:, :, :2, :2] + image_features_0[:, :, :2, :2]
                  + image_features_1[:, :, :2, :2] + prompt_bias)
        scores = logits.mean(dim=(1, 2, 3)).reshape(-1, 1)
        return logits, scores


class NamedSamDecoder(MultiFeatureSamDecoder):
    def forward(self, image_embeddings, image_features_0, image_features_1,
                point_coords, point_labels, mask_input, has_mask_input, orig_im_size):
        logits, scores = super().forward(
            image_embeddings, image_features_0, image_features_1, point_coords,
            point_labels, mask_input, has_mask_input, orig_im_size,
        )
        return {"scores": scores, "mask": logits}


class Sam2ExportTests(unittest.TestCase):
    def test_official_checkpoint_path_uses_the_actual_sam2_adapter_contract(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        fake_model = SimpleNamespace(
            image_size=1024,
            sam_prompt_encoder=SimpleNamespace(mask_input_size=(256, 256)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "sam2.1_hiera_tiny.pt"
            checkpoint.write_bytes(b"official")

            def export_adapter(value, output, **kwargs):
                self.assertEqual(value["variant"], "Hiera Tiny")
                self.assertEqual(value["input_size"], [1024, 1024])
                self.assertEqual(value["mask_size"], [256, 256])
                self.assertEqual(value["encoder_output_names"], [
                    "image_embeddings", "image_features_0", "image_features_1",
                ])
                Path(output).mkdir(parents=True, exist_ok=True)
                (Path(output) / "sam2_encoder.onnx").write_bytes(b"encoder")
                (Path(output) / "sam2_decoder.onnx").write_bytes(b"decoder")
                return {"backend": "sam2", "verification": "passed", "config_path": str(Path(output) / "sam2.json")}

            with patch("sam2_assets.load_sam2_pretrained", return_value=fake_model), \
                 patch("export_sam2_onnx.export_sam2_model", side_effect=export_adapter):
                result = export_official_sam2_checkpoint("sam2_hiera_tiny", checkpoint, root, log=lambda _: None)
        self.assertEqual(result["model_id"], "sam2_hiera_tiny")
        self.assertEqual(result["verification_reference"], "official_sam2.1_pytorch_image_predictor_path")
        self.assertEqual(result["file_size_mb"], 14 / (1024 * 1024))

    def test_encoder_decoder_graphs_and_schema_manifest(self):
        checkpoint = {
            "type": "sam2", "backend": "sam2", "task": "segment",
            "variant": "Hiera Tiny",
            "input_size": [2, 3], "mask_size": [2, 2],
            "encoder": TinySamEncoder().eval(), "decoder": TinySamDecoder().eval(),
        }
        with tempfile.TemporaryDirectory() as directory:
            result = export_sam2_model(checkpoint, directory, verify=True, opset=17,
                                       log=lambda _: None)
            manifest = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 5)
            self.assertEqual(manifest["backend"], "sam2")
            self.assertEqual(manifest["variant"], "Hiera Tiny")
            self.assertEqual(manifest["export"]["opset"], 17)
            self.assertEqual(manifest["export"]["verification_tolerance"],
                             {"atol": 1e-3, "rtol": 1e-3})
            self.assertTrue(manifest["export"]["dynamic_prompt_points"])
            self.assertEqual(manifest["output_names"], ["low_res_mask_logits", "iou_predictions"])
            self.assertEqual(manifest["contracts"]["prompt_types"], ["point", "box", "mask"])
            self.assertEqual(manifest["contracts"]["prompt_coordinate_space"], "resized_input")
            self.assertEqual(manifest["contracts"]["automatic_mask"]["max_grid"], 32)
            self.assertEqual(manifest["contracts"]["graphs"]["decoder"]["inputs"]["point_labels"],
                             "point_labels")
            self.assertTrue(Path(result["encoder_path"]).is_file())
            self.assertTrue(Path(result["decoder_path"]).is_file())
            self.assertEqual(result["verification"], "passed")

            import onnxruntime as ort
            session = ort.InferenceSession(result["decoder_path"], providers=["CPUExecutionProvider"])
            input_shapes = {item.name: item.shape for item in session.get_inputs()}
            self.assertEqual(input_shapes["point_coords"][1], "num_points")
            self.assertEqual(input_shapes["point_labels"][1], "num_points")

    def test_multi_feature_encoder_exports_contract_consumed_by_cpp_runtime(self):
        checkpoint = {
            "backend": "sam2", "task": "segment", "variant": "Hiera Base+",
            "input_size": [2, 3], "mask_size": [2, 2],
            "encoder": MultiFeatureSamEncoder().eval(),
            "decoder": MultiFeatureSamDecoder().eval(),
            "decoder_input_names": {"point_coords": "click_coordinates"},
        }
        with tempfile.TemporaryDirectory() as directory:
            result = export_sam2_model(checkpoint, directory, verify=True, opset=17,
                                       log=lambda _: None)
            manifest = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
            graphs = manifest["contracts"]["graphs"]
            self.assertEqual(graphs["encoder"]["outputs"], [
                "image_embeddings", "image_features_0", "image_features_1",
            ])
            self.assertEqual(graphs["decoder"]["inputs"], {
                "image_embeddings": "image_embeddings",
                "image_features_0": "image_features_0",
                "image_features_1": "image_features_1",
                "point_coords": "click_coordinates", "point_labels": "point_labels",
                "mask_input": "mask_input", "has_mask_input": "has_mask_input",
                "orig_im_size": "orig_im_size",
            })
            self.assertFalse(manifest["contracts"]["video_state"])
            self.assertEqual(result["verification"], "passed")

    def test_multi_output_tuple_requires_explicit_semantic_names(self):
        class TupleEncoder(torch.nn.Module):
            def forward(self, input_image):
                first = input_image.mean(dim=1, keepdim=True)
                return first, first

        checkpoint = {
            "backend": "sam2", "task": "segment", "variant": "Hiera Tiny",
            "input_size": [2, 3], "mask_size": [2, 2],
            "encoder": TupleEncoder().eval(), "decoder": TinySamDecoder().eval(),
        }
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
                ValueError, "encoder_output_names"):
            export_sam2_model(checkpoint, directory, verify=False, log=lambda _: None)

    def test_named_decoder_outputs_are_exported_in_runtime_order(self):
        checkpoint = {
            "backend": "sam2", "task": "segment", "variant": "Hiera Small",
            "input_size": [2, 3], "mask_size": [2, 2],
            "encoder": MultiFeatureSamEncoder().eval(),
            "decoder": NamedSamDecoder().eval(),
            "decoder_output_names": {
                "low_res_mask_logits": "mask", "iou_predictions": "scores",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            result = export_sam2_model(checkpoint, directory, verify=True, log=lambda _: None)
            manifest = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["output_names"], ["low_res_mask_logits", "iou_predictions"])
            self.assertEqual(result["verification"], "passed")

    def test_export_rejects_unregistered_variant(self):
        checkpoint = {
            "type": "sam2", "backend": "sam2", "task": "segment", "variant": "Hiera XL",
            "input_size": [2, 3], "mask_size": [2, 2],
            "encoder": TinySamEncoder().eval(), "decoder": TinySamDecoder().eval(),
        }
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "variant"):
            export_sam2_model(checkpoint, directory, verify=False, log=lambda _: None)


if __name__ == "__main__":
    unittest.main()
