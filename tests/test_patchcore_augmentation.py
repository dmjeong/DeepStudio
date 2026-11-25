"""Normal-view augmentation, actual coreset counts, and worker/CSV integration."""
import csv
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def pixels(tmp_path):
    path = tmp_path / "normal.png"
    Image.fromarray(np.random.default_rng(7).integers(30, 220, (40, 48, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    from torchvision.models import resnet18
    path = tmp_path_factory.mktemp("patchcore-weights") / "resnet18.pth"
    with torch.random.fork_rng():
        torch.manual_seed(42)
        torch.save(resnet18(weights=None).state_dict(), path)
    return str(path)


def test_extra_views_keep_original_and_apply_flip(pixels):
    from patchcore_data import PatchCoreDataset, input_tensor
    dataset = PatchCoreDataset([pixels], 32, augmentation_views=2, seed=7,
                              augmentation={"horizontal_flip": 1})
    assert len(dataset) == 3
    original = input_tensor(pixels, 32)
    torch.testing.assert_close(dataset[0][0], original, atol=0, rtol=0)
    for index in (1, 2):
        torch.testing.assert_close(dataset[index][0], original.flip(2), atol=1e-6, rtol=1e-6)
        assert dataset[index][1] == 0
    # Validation/default construction remains exactly the original path.
    validation = PatchCoreDataset([pixels], 32, [1])
    assert len(validation) == 1
    torch.testing.assert_close(validation[0][0], original, atol=0, rtol=0)


def test_augmented_views_are_reproducible_without_changing_global_rng(pixels):
    from patchcore_data import PatchCoreDataset
    from torchvision.transforms import functional  # noqa: F401 - initialize before checking RNG state
    dataset = PatchCoreDataset([pixels], 32, augmentation_views=2, seed=91,
        augmentation={"horizontal_flip": .5, "rotation": 15, "color_jitter": .2})
    numpy_state, torch_state, python_state = np.random.get_state(), torch.random.get_rng_state(), random.getstate()
    first = dataset[1][0]
    second = dataset[2][0]
    torch.testing.assert_close(dataset[1][0], first, atol=0, rtol=0)
    assert not torch.equal(first, second)
    assert not torch.equal(first, dataset[0][0])
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, atol=0, rtol=0)
    assert random.getstate() == python_state
    assert torch.isfinite(first).all()


def test_rotation_does_not_add_black_corners(tmp_path):
    from patchcore_data import IMAGENET_MEAN, IMAGENET_STD, PatchCoreDataset
    path = tmp_path / "white.png"
    Image.new("RGB", (48, 40), "white").save(path)
    dataset = PatchCoreDataset([path], 32, augmentation_views=1, augmentation={"rotation": 45})
    rgb = dataset[1][0] * torch.tensor(IMAGENET_STD)[:, None, None] + torch.tensor(IMAGENET_MEAN)[:, None, None]
    torch.testing.assert_close(rgb, torch.ones_like(rgb), atol=1e-6, rtol=0)


@pytest.mark.parametrize("settings", [
    {"augmentation_views": -1}, {"augmentation_views": 11}, {"augmentation_views": 1.5},
    {"augmentation_views": 1, "seed": -1},
    {"augmentation_views": 1, "augmentation": {"rotation": float("nan")}},
    {"augmentation_views": 1, "augmentation": {"horizontal_flip": 2}},
    {"augmentation_views": 1, "augmentation": {"color_jitter": -1}},
    {"augmentation_views": 1, "labels": [1]},
])
def test_invalid_augmentation_is_rejected(pixels, settings):
    from patchcore_data import PatchCoreDataset
    with pytest.raises(ValueError):
        PatchCoreDataset([pixels], 32, **settings)


@pytest.mark.parametrize("ratio,expected,limits", [
    (.1, 6, []), (.5, 16, ["candidate_limit", "memory_limit"]),
    (1., 16, ["candidate_limit", "memory_limit"]),
])
def test_coreset_reports_actual_limit_and_stage_times(weights, ratio, expected, limits):
    from patchcore import PatchCore
    pc = PatchCore(backbone_name="resnet18", pretrained=False, backbone_weights=weights,
                   input_size=32, sampling_ratio=ratio, max_candidates=20, max_memory_bank=16)
    messages = []
    pc.fit([torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(4))],
           progress_callback=lambda current, total, message: messages.append(message))
    metadata = pc.training_metadata
    assert metadata["seen_patches"] == 64
    assert metadata["requested_new_patches"] == max(int(64 * ratio), 1)
    assert len(pc.memory_bank) == metadata["new_patches"] == expected
    assert metadata["effective_new_ratio"] == expected / 64
    assert metadata["limited_by"] == limits
    assert metadata["feature_extraction_seconds"] >= 0
    assert metadata["coreset_selection_seconds"] >= 0
    assert any(f"실제 {expected:,}개" in message for message in messages)


def test_worker_uses_extra_normal_views_but_unmodified_validation_and_records_csv(tmp_path, weights):
    from core.patchcore_trainer import PatchCoreWorker
    from core.project import ProjectManager
    from patchcore import PatchCore
    from patchcore_data import input_tensor
    data = tmp_path / "data"
    for directory, values in (("train/good", [50, 80]), ("val/good", [55]), ("val/defect", [220])):
        folder = data / directory
        folder.mkdir(parents=True)
        for index, value in enumerate(values):
            Image.new("RGB", (40, 48), (value, value // 2, value // 3)).save(folder / f"{index}.png")
    project = ProjectManager.create_new("aug", "anomaly", str(tmp_path / "project"), ["good", "defect"])
    project.data.root = str(data)
    cfg = project.training
    cfg.device, cfg.input_size, cfg.batch_size = "cpu", 32, 2
    cfg.patchcore_weight_source, cfg.patchcore_weights, cfg.patchcore_backbone = "backbone", weights, "resnet18"
    cfg.patchcore_augmentation_views = 2
    cfg.patchcore_max_candidates, cfg.patchcore_max_memory_bank = 32, 8
    cfg.patchcore_sampling_ratio = .5
    worker = PatchCoreWorker(project)
    train, validation = worker._create_dataloaders(project.data, cfg)
    assert len(train.dataset) == 6 and len(validation.dataset) == 2
    assert validation.dataset.augmentation_views == 0
    for index, path in enumerate(validation.dataset.samples):
        torch.testing.assert_close(validation.dataset[index][0], input_tensor(path, 32), atol=0, rtol=0)
    errors = []
    worker.signals.training_error.connect(errors.append)
    worker.run()
    assert errors == []
    checkpoint = project.runs[-1].checkpoint_path
    model = PatchCore.load(checkpoint)
    meta = model.training_metadata
    assert meta["seen_patches"] == 96
    assert meta["augmentation"]["total_views"] == 6
    assert meta["augmentation"]["extra_views_per_image"] == 2
    with (Path(checkpoint).parent / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["hp_training_patchcore_augmentation_views"] == "2"
    assert row["hp_applied_training_metadata_augmentation_total_views"] == "6"
    assert float(row["hp_applied_training_metadata_coreset_selection_seconds"]) >= 0


def test_append_reports_remaining_capacity_and_preserves_old_bank(weights):
    from patchcore import PatchCore
    pc = PatchCore(backbone_name="resnet18", pretrained=False, backbone_weights=weights,
                   input_size=32, sampling_ratio=1., max_candidates=32, max_memory_bank=8)
    images = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(21))
    pc.fit([images])
    old = pc.memory_bank.clone()
    pc.max_memory_bank = 10
    pc.fit([images], append=True)
    torch.testing.assert_close(old, pc.memory_bank[:8], atol=0, rtol=0)
    assert pc.training_metadata["retained_patches"] == 8
    assert pc.training_metadata["requested_new_patches"] == 32
    assert pc.training_metadata["new_patches"] == 2
    assert pc.training_metadata["limited_by"] == ["memory_limit"]
