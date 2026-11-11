"""Actual checkpoint export -> C++ evaluation, plus malformed/absent GT rejection.

Set SEGMENT_EVALUATOR to the built example executable. No private data/weights.
"""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

EXE = os.environ.get("SEGMENT_EVALUATOR")
pytestmark = pytest.mark.skipif(not EXE, reason="Build example/cpp/segmentation_eval and set SEGMENT_EVALUATOR")


def _run(job, cwd):
    return subprocess.run([EXE], cwd=cwd, env={**os.environ, "DVS_EVAL_CONFIG": str(job)},
                          text=True, encoding="utf-8", capture_output=True, timeout=120)


@pytest.mark.parametrize("model_id", ["unet_lite", "unet_resnet18", "unet_mobilenetv2", "deeplabv3plus_resnet34",
                                     "dinov2_vits14", "dinov2_vits14_seg", "dinov2_vitb14_seg"])
def test_export_load_and_evaluate_real_architectures(tmp_path, model_id):
    import torch
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from export_onnx import export_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = build_builtin_model(model_id, 3, 3).eval()
    checkpoint = make_builtin_checkpoint(model_id, model, num_classes=3, input_size=64, in_channels=3,
        class_names=["background", "크랙", "FM"])
    checkpoint["center_crop"] = {"width": 64, "height": 64}
    checkpoint["preprocessing"]["center_crop"] = checkpoint["center_crop"]
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    export_checkpoint(path, tmp_path/"model.onnx", log=lambda _: None)
    (tmp_path/"images").mkdir(); (tmp_path/"masks").mkdir()
    pixels = np.random.default_rng(17).integers(0, 256, (80, 96, 3), dtype=np.uint8)
    truth = np.zeros((80, 96), dtype=np.uint8)
    truth[20:30, 25:40] = 1; truth[40:50, 45:55] = 2; truth[30:40, 35:45] = 255
    Image.fromarray(pixels).save(tmp_path/"images/영상.png")
    Image.fromarray(truth).save(tmp_path/"masks/영상.png")
    if os.name != "nt":
        # Symlinked inputs keep their dataset-relative names, not their target location.
        original = tmp_path/"images/영상.png"
        external = tmp_path/"external.png"
        original.rename(external)
        original.symlink_to(external)

    job = {"model_json": "model.json", "images": "images", "masks": "masks", "output": "results",
           "threads": 2, "warmup": 1, "repeat": 2}
    jobfile = tmp_path/"evaluation.json"; jobfile.write_text(json.dumps(job), encoding="utf-8")
    result = _run(jobfile, tmp_path.parent)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path/"results/report.json").read_text(encoding="utf-8"))
    prediction = np.array(Image.open(tmp_path/"results/predictions/영상.png"))
    valid = np.zeros(truth.shape, bool); valid[8:72, 16:80] = True; valid &= truth != 255
    confusion = np.bincount((truth[valid].astype(int)*3+prediction[valid]), minlength=9).reshape(3, 3)
    assert report["confusion_matrix"] == confusion.tolist()
    assert report["evaluated_pixels"] == int(valid.sum())
    assert 0 < report["p50_ms"] <= report["p95_ms"]
    assert len(report["images"][0]["times_ms"]) == 2
    row = report["images"][0]
    assert len(row["timing_samples_ms"]) == 2  # warmup excluded
    assert row["timing_sample_ms"] == row["timing_samples_ms"][row["timing_sample_index"]]
    stages = ("preprocess_ms", "model_ms", "validation_ms", "decode_ms", "rules_ms", "render_ms")
    for index, timing in enumerate(row["timing_samples_ms"]):
        assert all(np.isfinite(timing[key]) and timing[key] >= 0 for key in stages)
        assert sum(timing[key] for key in stages) == pytest.approx(timing["sdk_total_ms"], abs=1e-6)
        assert sum(timing[key] for key in stages[2:]) == pytest.approx(timing["postprocess_ms"], abs=1e-6)
        assert timing["call_total_ms"] == row["times_ms"][index]
        assert timing["call_total_ms"] >= timing["sdk_total_ms"]
    assert (tmp_path/"results/classes.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    overlay = np.array(Image.open(tmp_path/"results/overlays/영상.png"))
    assert np.array_equal(overlay[prediction == 0], pixels[prediction == 0])

    # A missing or invalid mask must never turn into an all-normal/100% score.
    if model_id == "unet_lite":
        for index, bad in enumerate([None, np.full((80, 96), 4, np.uint8),
                                     np.zeros((10, 10), np.uint8), np.full((80, 96), 255, np.uint8)]):
            mask_path = tmp_path/"masks/영상.png"
            mask_path.unlink(missing_ok=True)
            if bad is not None:
                Image.fromarray(bad).save(mask_path)
            job["output"] = f"invalid{index}"
            jobfile.write_text(json.dumps(job), encoding="utf-8")
            failed = _run(jobfile, tmp_path.parent)
            assert failed.returncode == 2, failed.stdout + failed.stderr
            assert not (tmp_path/job["output"]/"report.json").exists()
