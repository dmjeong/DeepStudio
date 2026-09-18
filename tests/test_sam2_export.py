"""Offline SAM2 two-graph exporter contract tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from export_sam2_onnx import export_sam2_model


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


class Sam2ExportTests(unittest.TestCase):
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
            self.assertEqual(manifest["output_names"], ["low_res_mask_logits", "iou_predictions"])
            self.assertEqual(manifest["contracts"]["prompt_types"], ["point", "box", "mask"])
            self.assertEqual(manifest["contracts"]["prompt_coordinate_space"], "resized_input")
            self.assertEqual(manifest["contracts"]["automatic_mask"]["max_grid"], 32)
            self.assertEqual(manifest["contracts"]["graphs"]["decoder"]["inputs"]["point_labels"],
                             "point_labels")
            self.assertTrue(Path(result["encoder_path"]).is_file())
            self.assertTrue(Path(result["decoder_path"]).is_file())
            self.assertEqual(result["verification"], "passed")

    def test_export_rejects_unregistered_variant(self):
        checkpoint = {
            "type": "sam2", "backend": "sam2", "task": "segment", "variant": "Hiera XL",
            "input_size": [2, 3], "mask_size": [2, 2],
            "encoder": TinySamEncoder().eval(), "decoder": TinySamDecoder().eval(),
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "variant"):
                export_sam2_model(checkpoint, directory, verify=False, log=lambda _: None)


if __name__ == "__main__":
    unittest.main()
