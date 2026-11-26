import copy
import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

import export_onnx
from classification_export_validation import (
    SYNTHETIC_POLICY, classification_image_probe, synthetic_classification_probes,
    compare_classification, collect_image_corpus,
)
from checkpoint import make_checkpoint_metadata
from efficientnet import EfficientNet


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_classification_policy_is_distinct_from_strict_logits():
    ref = np.array([[2., -2.]])
    actual = ref + .02
    with pytest.raises(ValueError):
        export_onnx.validate_classification_outputs(ref, actual)
    report = compare_classification(ref, actual)
    assert report["max_logit_error"] > .019
    assert report["max_probability_error"] < 1e-12


def test_reported_large_logit_model_has_stable_decision_and_probability():
    reference = np.array([[103.32628, -179.42242, 0., 24.536543]])
    actual = np.array([[103.40451, -179.4594, .02, 24.495605]])
    with pytest.raises(ValueError):
        export_onnx.validate_classification_outputs(reference, actual)
    report = compare_classification(reference, actual)
    assert report["max_logit_error"] == pytest.approx(.07823, abs=1e-5)
    assert report["max_probability_error"] < 1e-20
    assert report["min_reference_margin"] > .99


@pytest.mark.parametrize("ref,out", [([[1., 0.]], [[0., 1.]]),
                                      ([[1., 0.]], [[3., 0.]]),
                                      ([[.0001, 0.]], [[.0001, -.0002]]),
                                      ([[1., 0.]], [[float('nan'), 0.]])])
def test_wrong_rank_confidence_boundary_or_nonfinite_never_pass(ref, out):
    with pytest.raises(ValueError):
        compare_classification(ref, out)


@pytest.mark.parametrize("corrupt", [False, True, "actual_image"])
def test_real_images_choose_fp32_and_reload_or_preserve_existing_file(tmp_path, corrupt):
    import onnxruntime as ort
    from onnx_classifier import OnnxClassifier
    names = ["a", "b"]
    data = tmp_path / "images"
    for index, name in enumerate(names):
        (data / name).mkdir(parents=True)
        cv2.imwrite(str(data / name / "sample.png"), np.full((32, 32), index * 255, np.uint8))
    model = EfficientNet(num_classes=2).eval()
    with torch.no_grad():
        model.classifier[1].bias.copy_(torch.tensor([2., -2.]))
    checkpoint = make_checkpoint_metadata("classify", 2, names, (32, 32), 1)
    checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
    source, output = tmp_path / "model.pt", tmp_path / "model.onnx"
    torch.save(checkpoint, source)
    output.write_bytes(b"previous")
    real_export = export_onnx.export_to_onnx
    real_session = ort.InferenceSession

    class ImageMismatch:
        def __init__(self, session):
            self.session = session

        def __getattr__(self, name):
            return getattr(self.session, name)

        def run(self, names, feed, *args, **kwargs):
            values = self.session.run(names, feed, *args, **kwargs)
            # Only the actual black image changes class. The random and
            # normalized-zero probes still pass, which cannot certify images.
            if np.allclose(next(iter(feed.values())), -.449 / .226):
                values[0] = values[0][:, ::-1].copy()
            return values

    def session_factory(*args, **kwargs):
        session = real_session(*args, **kwargs)
        return ImageMismatch(session) if corrupt == "actual_image" else session

    def altered_export(candidate, dummy, path, *args, **kwargs):
        candidate = copy.deepcopy(candidate)
        with torch.no_grad():
            candidate.classifier[1].bias.add_(torch.tensor([-5., 5.]) if corrupt is True else .02)
        return real_export(candidate, dummy, path, *args, **kwargs)

    with patch.object(export_onnx, "export_to_onnx", side_effect=altered_export), \
            patch.object(ort, "InferenceSession", side_effect=session_factory), \
            patch("efficientnet_precision.prepare_precision_export", side_effect=AssertionError("FP64 must not run")):
        if corrupt:
            with pytest.raises(ValueError, match="통과한 FP32 후보가 없습니다"):
                export_onnx.export_checkpoint(source, output, validation_dir=data, log=lambda _: None)
            assert output.read_bytes() == b"previous"
            diagnostic = json.loads(output.with_suffix(".export-error.json").read_text(encoding="utf-8"))
            validation = diagnostic["numerical_diagnostic"]["classification_validation"]
            assert validation["image_count"] == 2
            assert validation["attempts"]
            assert all(not row["passed"] for row in validation["attempts"])
            failed_input = "image_1" if corrupt == "actual_image" else "seeded"
            assert all(row["checking"] == failed_input for row in validation["attempts"])
            assert all("판정 불일치" in row["reason"] for row in validation["attempts"])
            return
        result = export_onnx.export_checkpoint(source, output, validation_dir=data, dynamic_batch=True, log=lambda _: None)
    report = result["classification_validation"]
    assert report["image_count"] == 2 and report["selected"]["passed"]
    assert report["corpus_coverage"] == {
        "class_image_counts": {"a": 1, "b": 1}, "unrepresented_classes": [],
        "unlabeled_image_count": 0, "layout": "class_folders",
    }
    assert report["selected"]["max_logit_error"] > .019
    assert report["selected"]["max_probability_error"] < .001
    config = json.loads(output.with_suffix(".json").read_text())
    assert config["export"]["verification_policy"] == "classification_dataset_v1"
    assert config["export"]["compute_precision"] == "float32"
    assert "verification_tolerance" not in config["export"]
    classifier = OnnxClassifier(result["config_path"])
    assert classifier.predict_gray(np.zeros((32, 32), np.uint8))["class_id"] == 0
    assert str(data) not in json.dumps(report)


def test_empty_class_is_reported_and_excluded_from_image_corpus(tmp_path):
    (tmp_path / "a").mkdir()
    cv2.imwrite(str(tmp_path / "a" / "sample.png"), np.zeros((8, 8), np.uint8))
    files, coverage = collect_image_corpus(tmp_path, ["a", "b"])
    assert files == [tmp_path / "a" / "sample.png"]
    assert coverage["class_image_counts"] == {"a": 1, "b": 0}
    assert coverage["unrepresented_classes"] == ["b"]


def test_flat_image_folder_is_valid_for_output_parity(tmp_path):
    image = tmp_path / "sample.png"
    cv2.imwrite(str(image), np.zeros((8, 8), np.uint8))
    files, coverage = collect_image_corpus(tmp_path, ["a", "b"])
    assert files == [image]
    assert coverage["unrepresented_classes"] == ["a", "b"]
    assert coverage["unlabeled_image_count"] == 1
    assert coverage["layout"] == "flat_or_unlabeled"


def test_no_automatic_whole_fp64_export(tmp_path):
    model = EfficientNet(num_classes=2).eval()
    checkpoint = make_checkpoint_metadata("classify", 2, ["a", "b"], (32, 32), 1)
    checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
    source = tmp_path / "model.pt"
    torch.save(checkpoint, source)
    with patch.object(export_onnx, "verify_onnx", side_effect=ValueError("mismatch")), \
            patch("classification_export_validation.export_validated_synthetic_fp32",
                  side_effect=ValueError("decision mismatch")), \
            patch("efficientnet_precision.prepare_precision_export", side_effect=AssertionError("must not run")) as precision:
        with pytest.raises(ValueError, match="모두 ONNX 검증 실패"):
            export_onnx.export_checkpoint(source, tmp_path / "model.onnx", log=lambda _: None)
        precision.assert_not_called()


@pytest.mark.parametrize("corrupt", [False, True])
def test_synthetic_policy_exports_only_if_decisions_match(tmp_path, corrupt):
    from onnx_classifier import OnnxClassifier
    names = ["a", "b"]
    model = EfficientNet(num_classes=2).eval()
    with torch.no_grad():
        model.classifier[1].weight.zero_()
        model.classifier[1].bias.copy_(torch.tensor([4., -4.]))
    checkpoint = make_checkpoint_metadata("classify", 2, names, (32, 32), 1)
    checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(),
                      model_state_dict=model.state_dict())
    source, output = tmp_path / "model.pt", tmp_path / "model.onnx"
    torch.save(checkpoint, source)
    output.write_bytes(b"previous onnx")
    output.with_suffix(".json").write_bytes(b"previous config")
    real_export = export_onnx.export_to_onnx

    def shifted_export(candidate, dummy, path, *args, **kwargs):
        candidate = copy.deepcopy(candidate)
        target = candidate.model if hasattr(candidate, "model") else candidate
        with torch.no_grad():
            target.classifier[1].bias.add_(torch.tensor([-8., 8.]) if corrupt else .02)
        return real_export(candidate, dummy, path, *args, **kwargs)

    with patch.object(export_onnx, "export_to_onnx", side_effect=shifted_export):
        if corrupt:
            with pytest.raises(ValueError, match="판정 불일치|top-1 불일치") as caught:
                export_onnx.export_checkpoint(source, output, dynamic_batch=True, log=lambda _: None)
            assert output.read_bytes() == b"previous onnx"
            assert output.with_suffix(".json").read_bytes() == b"previous config"
            assert len(str(caught.value)) < 1600
            diagnostic = json.loads(output.with_suffix(".export-error.json").read_text(encoding="utf-8"))
            numeric = diagnostic["numerical_diagnostic"]
            rows = [row for row in numeric["runtime_attempts"] if row.get("policy") == SYNTHETIC_POLICY]
            assert len(rows) >= 8
            assert all(row["checking"] == "seeded_image" and not row["passed"] for row in rows)
            assert all("판정 불일치" in row["reason"] for row in rows)
            assert set(numeric["initial_failures"]) == {"optimized", "unfused_fp32"}
            return
        result = export_onnx.export_checkpoint(source, output, dynamic_batch=True, log=lambda _: None)
    report = result["classification_validation"]
    assert report["policy"] == SYNTHETIC_POLICY
    assert report["probe_count"] == 13
    assert report["selected"]["max_logit_error"] > .019
    assert report["selected"]["max_probability_error"] < 1e-6
    assert report["selected"]["model_median_ms"] > 0
    manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["export"]["verification_policy"] == SYNTHETIC_POLICY
    assert manifest["export"]["compute_precision"] == "float32"
    assert manifest["schema_version"] == 6
    classifier = OnnxClassifier(result["config_path"])
    sample = torch.zeros(1, 1, 32, 32)
    assert classifier.logits(sample.numpy()).argmax(axis=1).tolist() == [0]


def test_exact_probability_tie_never_passes_relaxed_policy():
    with pytest.raises(ValueError, match="경계 근처"):
        compare_classification([[0., 0.]], [[0., 0.]])


@pytest.mark.parametrize("mean,std", [
    ([.485, .456, .406], [.229, .224, .225]),
    ([.449], [.226]),
    ([-.1, .5, 1.1], [.2, .3, .4]),
])
def test_every_synthetic_probe_is_within_checkpoint_pixel_domain(mean, std):
    shape = (1, len(mean), 100, 100)
    metadata = {"normalize_mean": mean, "normalize_std": std}
    # Include an explicitly invalid caller tensor; it must never leak into
    # tracing, warmup or validation in the image-domain policy.
    dummy = torch.full(shape, 1000.)
    probes = synthetic_classification_probes(dummy, metadata)
    assert len(probes) == 12
    assert len({name for name, _ in probes}) == 12
    for name, probe in probes:
        assert tuple(probe.shape) == shape, name
        assert probe.dtype == torch.float32 and probe.is_contiguous(), name
        raw = probe.numpy() * np.array(std, np.float32)[None, :, None, None]
        raw += np.array(mean, np.float32)[None, :, None, None]
        assert np.isfinite(raw).all(), name
        assert raw.min() >= -2e-7 and raw.max() <= 1 + 2e-7, name
    # Caller and both global RNGs stay untouched. Export checks must be repeatable.
    torch_rng, numpy_rng = torch.get_rng_state().clone(), np.random.get_state()
    repeat = classification_image_probe(shape, metadata)
    torch.testing.assert_close(repeat, probes[0][1], rtol=0, atol=0)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert numpy_rng[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
    assert numpy_rng[2:] == np.random.get_state()[2:]
    assert torch.all(dummy == 1000.)


@pytest.mark.parametrize("mean,std", [
    ([float("nan")], [.2]), ([.5], [float("inf")]),
    ([.5], [0.]), ([.5], [-.2]), ([.5, .5], [.2]),
])
def test_invalid_normalization_cannot_create_verification_probe(mean, std):
    with pytest.raises(ValueError, match="정규화 메타데이터"):
        classification_image_probe((1, 1, 100, 100), {"mean": mean, "std": std})


def test_old_rgb_gaussian_probe_contains_impossible_pixels():
    sample = torch.randn(1, 3, 100, 100, generator=torch.Generator().manual_seed(42))
    raw = sample * torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    raw += torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    # Reproduce the invalid-domain defect without attributing the customer's
    # private checkpoint's class disagreement to it.
    assert raw.min() < 0 and raw.max() > 1
    assert ((raw < 0) | (raw > 1)).sum() > 0


def test_rgb_100_export_checks_real_onnx_with_bounded_inputs(tmp_path):
    from onnx_classifier import OnnxClassifier
    model = EfficientNet("efficientnet_b0", 4, 3).eval()
    checkpoint = make_checkpoint_metadata("classify", 4, ["a", "b", "c", "d"], (100, 100), 3)
    checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(),
                      model_state_dict=model.state_dict())
    source, output = tmp_path / "rgb.pt", tmp_path / "rgb.onnx"
    torch.save(checkpoint, source)
    verify = export_onnx.verify_onnx
    checked = []

    def check_inputs(path, probe, candidate, **kwargs):
        raw = probe.numpy() * np.array([.229, .224, .225], np.float32)[None, :, None, None]
        raw += np.array([.485, .456, .406], np.float32)[None, :, None, None]
        assert raw.min() >= -2e-7 and raw.max() <= 1 + 2e-7
        checked.append(probe.shape[0])
        return verify(path, probe, candidate, **kwargs)

    with patch.object(export_onnx, "verify_onnx", side_effect=check_inputs):
        result = export_onnx.export_checkpoint(source, output, dynamic_batch=True, log=lambda _: None)
    assert result["verification"] == "passed" and {1, 2}.issubset(checked)
    manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["export"]["verification_input_domain"] == "normalized_pixels_0_to_1"
    runtime = OnnxClassifier(result["config_path"])
    probe = classification_image_probe((1, 3, 100, 100),
        {"mean": [.485, .456, .406], "std": [.229, .224, .225]}, seed=19)
    with torch.no_grad():
        export_onnx.validate_classification_outputs(model(probe).numpy(), runtime.logits(probe.numpy()))


def test_web_export_worker_uses_complete_project_corpus_and_explicit_options(tmp_path):
    from types import SimpleNamespace
    from webapp.worker import export
    data = {"class_names": ["a", "b"], "val_dir": str(tmp_path / "val"),
            "test_dir": str(tmp_path / "test"), "train_dir": str(tmp_path / "train")}
    for name in data["class_names"]:
        folder = tmp_path / "train" / name
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "image.png"), np.zeros((8, 8), np.uint8))
    context = SimpleNamespace(cancelled=lambda: False, emit=lambda *args: None)
    payload = {"weights": "model.pt", "output": "model.onnx", "dataset_validation": True,
               "project": {"data": data}}
    with patch.object(export_onnx, "export_checkpoint", return_value={}) as exporter:
        assert export(context, payload)["status"] == "completed"
        assert exporter.call_args.kwargs["validation_dir"] == data["train_dir"]
        assert exporter.call_args.kwargs["allow_precision_fallback"] is False
        payload.update(dataset_validation=False, allow_precision_fallback=True)
        export(context, payload)
        assert exporter.call_args.kwargs["validation_dir"] is None
        assert exporter.call_args.kwargs["allow_precision_fallback"] is True


def test_web_export_worker_rejects_missing_validation_corpus(tmp_path):
    from types import SimpleNamespace
    from webapp.worker import export
    context = SimpleNamespace(cancelled=lambda: False, emit=lambda *args: None)
    with patch.object(export_onnx, "export_checkpoint") as exporter:
        with pytest.raises(ValueError, match="비교 경로"):
            export(context, {"weights": "model.pt", "output": "model.onnx", "dataset_validation": True})
        exporter.assert_not_called()
