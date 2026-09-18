"""C++ tuning must preserve source metadata and select using repeated pipeline measurements."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("tune_cpp", ROOT / "tools/tune_efficientnet_cpp.py")
tuner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tuner)


@pytest.mark.parametrize("extra", [["--runs", "0"], ["--rounds", "0"], ["--threads", "0"],
                                  ["--block-bases", "-1"], ["--spinning", "maybe"]])
def test_reject_invalid_measurement_parameters(tmp_path, extra):
    with pytest.raises(SystemExit):
        tuner.main(["--executable", "missing", "--config", "missing", "--synthetic",
                    "--output", str(tmp_path / "out"), *extra])


def test_actual_cpp_tuning_saves_reusable_config_and_repeated_measurements(tmp_path):
    executable = os.environ.get("CPP_BENCHMARK")
    if not executable:
        pytest.skip("Set CPP_BENCHMARK to the compiled C++17 executable")
    models = tmp_path / "models"
    subprocess.run([sys.executable, str(ROOT / "cpp/tests/make_models.py"), str(models)], check=True)
    source = models / "classify.json"
    original = source.read_bytes()
    output = tmp_path / "measurements"
    args = ["--executable", executable, "--config", str(source), "--synthetic", "--output", str(output),
            "--threads", "1", "2", "--block-bases", "0", "4", "--spinning", "off", "--rounds", "2",
            "--runs", "3", "--warmup", "1"]
    report = tuner.main(args)
    assert source.read_bytes() == original
    assert report["background"] is True
    assert len(report["candidates"]) == 4
    assert all(len(row["trials"]) == 2 for row in report["candidates"])
    scores = [row["worst_round_p95_ms"] for row in report["candidates"]]
    assert scores == sorted(scores)
    for candidate in report["candidates"]:
        assert candidate["worst_round_p95_ms"] == max(t["result"]["wall"]["p95_ms"] for t in candidate["trials"])
        assert candidate["options"]["allow_spinning"] == 0
        for trial in candidate["trials"]:
            raw = json.loads((output / trial["file"]).read_text())
            assert len(raw["wall"]["samples_ms"]) == 3
            assert raw["cpp_standard"] == 201703
    config = json.loads((output / "recommended.json").read_text())
    assert (output / config["model_path"]).resolve() == (models / "classify.onnx").resolve()
    assert config["num_threads"] == report["candidates"][0]["threads"]
    assert config["runtime"] == "onnxruntime"
    check = output / "recommended-check.json"
    subprocess.run([executable, "--config", str(output / "recommended.json"), "--synthetic",
                    "--warmup", "1", "--runs", "1", "--background", "--output", str(check)],
                   capture_output=True, text=True, check=True)
    assert json.loads(check.read_text())["threads"] == config["num_threads"]
    # Refuse destructive reruns into an existing result directory.
    with pytest.raises(SystemExit):
        tuner.main(args)
