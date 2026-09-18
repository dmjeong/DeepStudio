"""실제 ONNX/JSON을 C++17 실행 파일로 읽고 Python과 판정을 비교한다."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtimes", nargs="+", choices=["onnxruntime", "openvino"], default=["onnxruntime", "openvino"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from onnx_classifier import OnnxClassifier
    runtime = OnnxClassifier(args.config, num_threads=1)
    cfg = runtime.config
    crop = cfg["preprocessing"].get("center_crop") or {}
    h = max(cfg["input_height"], crop.get("height", 0)) + 7
    w = max(cfg["input_width"], crop.get("width", 0)) + 13
    rng = np.random.default_rng(42)
    cases = {"gray": rng.integers(0, 256, (h, w), dtype=np.uint8),
             "rgb": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
             "rgba": rng.integers(0, 256, (h, w, 4), dtype=np.uint8)}
    expected = {name: runtime.predict_rgb(pixels) for name, pixels in cases.items()}
    del runtime  # No idle ORT pool while running the C++ executable.
    report = {"input_channels": cfg["input_channels"], "input_size": [cfg["input_height"], cfg["input_width"]],
              "scope": "correctness smoke, not a latency benchmark", "comparisons": []}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, pixels in cases.items():
            image = root / f"{name}.png"
            encoded = pixels if name == "gray" else cv2.cvtColor(pixels,
                cv2.COLOR_RGB2BGR if name == "rgb" else cv2.COLOR_RGBA2BGRA)
            if not cv2.imwrite(str(image), encoded):
                raise ValueError("Cannot write test image")
            for backend in args.runtimes:
                for background in (False, True):
                    output = root / "cpp.json"
                    command = [str(args.executable.resolve()), "--config", str(args.config.resolve()),
                               "--image", str(image), "--runtime", backend, "--threads", "1",
                               "--warmup", "1", "--runs", "2", "--output", str(output)]
                    if background:
                        command.append("--background")
                    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
                    if process.returncode:
                        raise RuntimeError(f"C++ failed: {process.stderr}")
                    result = json.loads(output.read_text(encoding="utf-8"))
                    if result["cpp_standard"] != 201703:
                        raise ValueError("Expected a C++17 build")
                    if result["last_class_id"] != expected[name]["class_id"]:
                        raise ValueError(f"{backend}/{name}: class mismatch")
                    np.testing.assert_allclose(result["last_probabilities"], expected[name]["probabilities"], atol=1e-5, rtol=1e-5)
                    report["comparisons"].append({"image": name, "runtime": backend, "background": background,
                        "class_id": result["last_class_id"], "max_probability_error": float(np.max(np.abs(
                            np.asarray(result["last_probabilities"]) - expected[name]["probabilities"])))})
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
