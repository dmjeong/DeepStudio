"""Regenerate the tiny PUBLIC test models for the C++ CSV reload example.

Not trained inspection models. Uses the already shipped example-only.key.
Run with the repository Python environment; native examples need no Python.
"""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from model_crypto import encrypt_config


def main():
    assets = ROOT / "example/assets"
    output = assets / "reload"
    output.mkdir(exist_ok=True)
    key = (assets / "example-only.key").read_bytes()
    with tempfile.TemporaryDirectory() as directory:
        stage = Path(directory)
        (stage / "test.onnx").write_bytes((assets / "test.onnx").read_bytes())
        for index in range(1, 5):
            config = json.loads((assets / "test.json").read_text())
            config["model_version"] = f"20261126_010000_{index:06d}"
            config["class_names"] = [f"white_v{index}", f"other_v{index}"]
            manifest = stage / "test.json"
            manifest.write_text(json.dumps(config), encoding="utf-8")
            name = "model.dvsenc" if index == 1 else f"v{index}.dvsenc"
            encrypt_config(manifest, output / name, key)
            if index == 4:
                # Authenticated but incompatible I/O: construction must fail.
                config["input_width"] = 100
                manifest.write_text(json.dumps(config), encoding="utf-8")
                encrypt_config(manifest, output / "bad-shape-v4.dvsenc", key)
    (output / "models.csv").write_text(
        "slot,task,model_file,model_version\n"
        "chip,classify,model.dvsenc,20261126_010000_000001\n", encoding="utf-8")


if __name__ == "__main__":
    main()
