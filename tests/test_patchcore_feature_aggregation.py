"""Feature-space changes require a new bank; saved models must keep their scores."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch.nn import functional as F
from patchcore import (PatchCore, LEGACY_FEATURE_AGGREGATION as OLD,
                       STANDARD_FEATURE_AGGREGATION as NEW, STANDARD_SCORE_DEFINITION)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(aggregation=NEW):
    with torch.random.fork_rng():
        torch.manual_seed(91)
        return PatchCore(backbone_name="resnet18", pretrained=False, input_size=64,
                         max_candidates=128, max_memory_bank=16, sampling_ratio=.5,
                         feature_aggregation=aggregation)


def inputs():
    return torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(18))


def mean_neighbourhood(x):
    # Independent explicit 3x3 patch reduction, including zero-padded borders.
    b, c, h, w = x.shape
    return F.unfold(x, 3, padding=1).reshape(b, c, 9, h, w).mean(2)


def test_each_layer_is_aggregated_before_alignment():
    pc = model()
    b = pc.backbone
    x = inputs()
    with torch.inference_mode():
        low = b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x))))))
        high = b.layer3(low)
        expected = torch.cat([mean_neighbourhood(low), F.interpolate(mean_neighbourhood(high),
            size=low.shape[-2:], mode="bilinear", align_corners=False)], 1)
        actual = pc._avg_pool(b(x))
        torch.testing.assert_close(actual, expected)
        old = mean_neighbourhood(torch.cat([low, F.interpolate(high,
            size=low.shape[-2:], mode="bilinear", align_corners=False)], 1))
        assert (actual-old).abs().max().item() > .01
        pc.set_feature_aggregation(OLD)
        torch.testing.assert_close(pc._avg_pool(b(x)), old)


@pytest.mark.parametrize("aggregation", [OLD, NEW])
def test_checkpoint_roundtrip_preserves_feature_space_and_outputs(tmp_path, aggregation):
    pc = model(aggregation)
    pc.fit([inputs()])
    pc.anomaly_threshold = .2
    x = inputs() * .9
    expected = pc.predict(x)
    checkpoint = tmp_path / "pc.pt"
    pc.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["schema_version"] == 6
    assert saved["feature_aggregation"] == aggregation
    loaded = PatchCore.load(checkpoint)
    assert loaded.feature_aggregation == aggregation
    assert loaded.training_metadata["feature_aggregation"] == aggregation
    assert loaded.anomaly_threshold == pc.anomaly_threshold
    for a, b in zip(expected, loaded.predict(x)):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="새로 구축"):
        loaded.set_feature_aggregation(NEW if aggregation == OLD else OLD)


def test_schema5_stays_legacy_without_changing_predictions(tmp_path):
    pc = model(OLD)
    pc.fit([inputs()])
    pc.anomaly_threshold = .3
    path = tmp_path / "old.pt"
    pc.save(path)
    data = torch.load(path, weights_only=False)
    data["schema_version"] = 5
    del data["feature_aggregation"]
    torch.save(data, path)
    loaded = PatchCore.load(path)
    assert loaded.feature_aggregation == OLD
    for a, b in zip(pc.predict(inputs()), loaded.predict(inputs())):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind", ["missing", "unknown", "wrong_version"])
def test_bad_feature_contract_rejected(tmp_path, kind):
    pc = model()
    pc.fit([inputs()])
    path = tmp_path / "bad.pt"
    pc.save(path)
    data = torch.load(path, weights_only=False)
    if kind == "missing":
        del data["feature_aggregation"]
    elif kind == "unknown":
        data["feature_aggregation"] = "unknown"
    else:
        data["schema_version"] = 5
    torch.save(data, path)
    with pytest.raises(ValueError, match="특징 집계"):
        PatchCore.load(path)


@pytest.mark.parametrize("append", [False, True])
def test_worker_rebuild_upgrades_but_append_keeps_old_space(tmp_path, append):
    pytest.importorskip("PySide6")
    from core.patchcore_trainer import PatchCoreWorker
    from core.project import ProjectManager
    from patchcore import LEGACY_SCORE_DEFINITION
    pc = model(OLD)
    pc.score_definition = LEGACY_SCORE_DEFINITION
    pc.fit([inputs()])
    path = tmp_path / "old.pt"
    pc.save(path)
    project = ProjectManager.create_new("aggregation", "anomaly", str(tmp_path / "project"), ["good", "bad"])
    cfg = project.training
    cfg.patchcore_weight_source = "patchcore"
    cfg.patchcore_weights = str(path)
    cfg.patchcore_append = append
    cfg.patchcore_max_memory_bank = 32
    cfg.input_size = 64
    worker = PatchCoreWorker(project)
    loaded = worker._prepare_model(cfg, torch.device("cpu"))
    assert loaded.feature_aggregation == (OLD if append else NEW)
    if append:
        torch.testing.assert_close(loaded.memory_bank, pc.memory_bank, atol=0, rtol=0)
        assert loaded.score_definition == LEGACY_SCORE_DEFINITION
    else:
        assert loaded.memory_bank is None
        assert loaded.score_definition == STANDARD_SCORE_DEFINITION


def test_calibration_midpoint_keeps_f1_decisions_and_avoids_equality():
    from patchcore_calibration import stable_threshold
    values = np.array([1., 2., 4., 4., 5.], dtype=np.float32)
    result = stable_threshold(values, 4.)
    assert result["optimal_threshold"] == 3.
    np.testing.assert_array_equal(values >= 4., values >= result["optimal_threshold"])
    # A small export rounding difference no longer changes the boundary sample.
    assert float(np.nextafter(np.float32(4), np.float32(0))) >= result["optimal_threshold"]
    assert result["threshold_margin"] == 1.


def test_calibration_retains_boundary_when_no_float32_gap_exists():
    from patchcore_calibration import stable_threshold
    a = np.float32(2)
    b = np.nextafter(a, np.float32(3))
    result = stable_threshold([a, b], b)
    assert result["optimal_threshold"] == float(b)
    assert result["threshold_margin"] == 0


def test_calibration_does_not_invent_margin_for_all_positive_partition():
    from patchcore_calibration import stable_threshold
    assert stable_threshold([2., 3.], 2.)["threshold_margin"] == 0
