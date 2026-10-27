"""Generate deployment-loader regression fixtures through the real Studio exporter.

Only the optional LibreYOLO checkpoint loader is replaced with a tiny public-API
stand-in. Export verification, ONNX/JSON creation, and the C++ consumer remain
production code. Requires torch, onnx, and onnxruntime; downloads no weights.
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from export_onnx import export_checkpoint


class TinyClassifier(torch.nn.Module):
    def forward(self, images):
        value = images.mean(dim=(1, 2, 3))
        return torch.stack((value, -value), dim=1)


class LibreYOLOExportFixture:
    FAMILY = "mobilenetv4"

    def __init__(self):
        self.model = TinyClassifier().eval()

    def export(self, format, *, output_path, imgsz, opset, dynamic, simplify, device):
        assert format == "onnx" and tuple(imgsz) == (8, 8)
        assert device == "cpu" and not simplify
        torch.onnx.export(
            self.model, torch.zeros(1, 3, *imgsz), output_path,
            input_names=["images"], output_names=["output"], opset_version=opset,
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}} if dynamic else None,
            dynamo=False,
        )
        return output_path


def generate(output_dir):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    upstream = LibreYOLOExportFixture()
    checkpoint = {"model_family": "mobilenetv4", "task": "classify", "nc": 2,
                  "names": {0: "bright", 1: "dark"}, "imgsz": 8}
    # The checkpoint is temporary; only the actual exporter artifacts and the
    # independently computed expected probabilities reach the native test.
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "libre.pt"
        torch.save(checkpoint, source)
        with patch("upstream_models.load_upstream_checkpoint", return_value=upstream):
            result = export_checkpoint(source, root / "libreyolo.onnx", verify=True)
    if result["verification"] != "passed" or not result["cpp_supported"]:
        raise RuntimeError("Export did not produce a verified C++ classification artifact")

    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    samples = []
    with torch.inference_mode():
        for pixel in (0, 127, 255):
            image = torch.full((1, 3, 8, 8), pixel / 255.)
            probabilities = upstream.model((image - mean) / std).softmax(dim=1)[0]
            class_id = int(probabilities.argmax())
            samples.append({"pixel": pixel, "class_id": class_id,
                            "class_name": checkpoint["names"][class_id],
                            "probabilities": probabilities.tolist()})
    (root / "expected.json").write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    generate(parser.parse_args().output_dir)
