"""Generated outputs must be real task datasets, not just preview pictures."""
from pathlib import Path

import numpy as np
import pytest
from core.dataset_editor import read_annotations, scan_dataset
from core.project import ProjectManager
from core.project_generation import ProjectGenerationStore, class_options
from PIL import Image
from webapp.storage import digest


def make_project(tmp_path, task):
    names = ["good", "핀찌그러짐", "코팅부족"] if task == "anomaly" else ["정상", "핀찌그러짐", "코팅부족"]
    if task == "segment":
        names = names[1:]
    project = ProjectManager.create_new("사용자 검사", task, str(tmp_path / task), names)
    pixels = np.random.default_rng(4).integers(65, 190, (96, 128, 3), dtype=np.uint8)
    for split in ("train", "val", "test"):
        if task == "anomaly" and split == "val":
            continue
        folder = Path(getattr(project.data, split + "_dir"))
        if task in {"classify", "anomaly"}:
            folder /= names[0]
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(folder / "바탕.png")
        if task == "segment":
            mask = np.zeros(pixels.shape[:2], np.uint8)
            mask[4:14, 4:16] = 1
            mask[16:21, 4:16] = 255
            Image.fromarray(mask).save(Path(project.data.root) / "masks" / split / "바탕.png")
        elif task in {"detect", "obb"}:
            label = "0 .1 .1 .12 .12\n" if task == "detect" else "0 .04 .04 .16 .04 .16 .16 .04 .16\n"
            (Path(project.data.root) / "labels" / split / "바탕.txt").write_text(label, encoding="utf-8")
    ProjectManager.save(project)
    return project


def generate(store, **kwargs):
    return store.generate([{"class_id": 2, "method": "stain", "count": 2}], source_class="정상", seed=41, **kwargs)


@pytest.mark.parametrize("task", ["classify", "segment", "detect", "obb", "anomaly"])
def test_custom_class_generate_publish_reopen_and_relocate(tmp_path, task):
    project = make_project(tmp_path, task)
    before = {str(p): digest(p) for p in Path(project.data.root).rglob("*") if p.is_file()}
    store = ProjectGenerationStore(project)
    result = generate(store)
    batch = store.load(result["batch_id"])
    assert batch["classes"] == project.data.class_names
    assert result["generated"] == 2 and not result["failures"]
    for row in batch["samples"]:
        folder = store.verified_sample(batch["id"], row)
        original, generated = [np.array(Image.open(folder / file)) for file in ("original.png", "image.png")]
        changed = np.array(Image.open(folder / "changed.png")) > 0
        assert np.any(changed)
        np.testing.assert_array_equal(original[~changed], generated[~changed])
        assert Path(row["source"]).is_relative_to(project.data.train_dir)
        if task == "segment":
            target = np.array(Image.open(folder / "mask.png"))
            assert set(np.unique(target)) == {0, 1, 2, 255}
            assert np.all(target[changed] == 2)
            assert np.all(target[4:14, 4:16] == 1)
        elif task in {"detect", "obb"}:
            labels = read_annotations(folder / "label.txt", task, 3)
            assert [r["class_id"] for r in labels] == [0, 2]
    selected = [batch["samples"][0]["id"]]
    saved = store.publish(batch["id"], selected)
    assert store.publish(batch["id"], selected)["reused"]
    reopened = ProjectManager.load(saved["project_path"])
    assert Path(reopened.data.train_dir).is_dir()
    index = scan_dataset(reopened)
    assert all(not r["error"] for r in index["images"])
    synthetic = [r for r in index["images"] if r["name"].startswith("generated-")]
    assert len(synthetic) == 1
    assert synthetic[0]["split"] == ("test" if task == "anomaly" else "train")
    assert "코팅부족" in synthetic[0]["classes"]
    if task == "anomaly":
        assert len(list((Path(reopened.data.root) / "ground_truth/코팅부족").glob("*_mask.png"))) == 1
        assert len(list(Path(reopened.data.train_dir).rglob("*.png"))) == 1
        assert not any(r["split"] == "test" and r["name"] == "바탕.png" for r in index["images"])
    else:
        assert len([r for r in index["images"] if r["split"] == "val"]) == 1
        assert len([r for r in index["images"] if r["split"] == "test"]) == 1
    # The new project is movable; no .partial folder path leaks into its JSON.
    path = Path(saved["project_path"])
    assert ".partial" not in path.read_text(encoding="utf-8")
    moved = path.parent.rename(tmp_path / "다른 PC에 배포")
    relocated = ProjectManager.load(str(moved / path.name))
    assert scan_dataset(relocated)["images"]
    assert before == {str(p): digest(p) for p in Path(project.data.root).rglob("*") if p.is_file()}


@pytest.mark.parametrize("method", ["scratch", "stain", "cutout", "noise", "texture", "elastic", "color"])
def test_each_method_uses_project_class_not_fixed_names(tmp_path, method):
    store = ProjectGenerationStore(make_project(tmp_path, "segment"))
    out = store.generate([{"class_id": 2, "method": method, "count": 1}], seed=9)
    row = store.load(out["batch_id"])["samples"][0]
    assert row["class_id"] == 2
    assert row["method"] == method


def test_stale_class_map_and_tampered_candidates_rejected(tmp_path):
    project = make_project(tmp_path, "segment")
    store = ProjectGenerationStore(project)
    batch = store.load(generate(store)["batch_id"])
    row = batch["samples"][0]
    project.data.class_names[2] = "다른 클래스"
    with pytest.raises(ValueError, match="클래스가 변경"):
        store.publish(batch["id"], [row["id"]])
    project.data.class_names = batch["classes"]
    (store.directory("batches", batch["id"]) / row["id"] / "image.png").write_bytes(b"broken")
    with pytest.raises(ValueError, match="변경"):
        store.publish(batch["id"], [row["id"]])
    assert not list((store.root / "publications").glob("*/manifest.json"))


def test_empty_source_cancel_and_class_zero_contract(tmp_path):
    project = make_project(tmp_path, "segment")
    store = ProjectGenerationStore(project)
    assert class_options(project) == [(1, "핀찌그러짐"), (2, "코팅부족")]
    with pytest.raises(ValueError, match="클래스"):
        store.generate([{"class_id": 0, "method": "noise", "count": 1}])
    class Cancel:
        def cancelled(self):
            return True
    with pytest.raises(InterruptedError):
        generate(store, context=Cancel())
    assert not list((store.root / "batches").glob("*"))
    for p in Path(project.data.train_dir).glob("*.png"):
        p.unlink()
    with pytest.raises(ValueError, match="0장"):
        generate(store)


def test_detection_class_zero_is_foreground_and_seed_is_reproducible(tmp_path):
    project = make_project(tmp_path, "detect")
    store = ProjectGenerationStore(project)
    rule = [{"class_id": 0, "method": "color", "count": 1}]
    a, b = [store.load(store.generate(rule, seed=3)["batch_id"]) for _ in range(2)]
    assert a["samples"][0]["hashes"] == b["samples"][0]["hashes"]
    assert [0, "정상"] == list(class_options(project)[0])
