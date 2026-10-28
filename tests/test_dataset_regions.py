"""Object tasks label regions inside an image, not the image: scan counts and dataset page."""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from core.dataset_editor import mask_regions, scan_dataset
from core.project import ProjectManager

NAMES = ["background", "scratch", "dent"]


def _segment_project(root):
    """a: 2 scratches + 1 dent (+ ignore pixels), b: background-only mask, c: no mask."""
    project = ProjectManager.create_new("regions", "segment", str(root), NAMES)
    images = Path(project.data.train_dir)
    masks = images.parent.parent / "masks" / "train"
    masks.mkdir(parents=True, exist_ok=True)
    for name in "abc":
        Image.new("L", (40, 30), 100).save(images / f"{name}.png")
    mask = np.zeros((30, 40), np.uint8)
    mask[2:5, 2:12] = 1            # scratch 1
    mask[10:12, 20:35] = 1         # scratch 2
    mask[20, 5] = mask[21, 6] = 1  # diagonal neighbours: one region (8-connected)
    mask[24:28, 30:34] = 2         # dent
    mask[0, 39] = 255              # training exclusion: not a region
    Image.fromarray(mask, "L").save(masks / "a.png")
    Image.fromarray(np.zeros((30, 40), np.uint8), "L").save(masks / "b.png")
    return project, images, masks


def test_scan_counts_regions_per_image_and_split(tmp_path):
    project, _, _ = _segment_project(tmp_path / "p")
    index = scan_dataset(project)
    rows = {Path(row["path"]).stem: row for row in index["images"]}
    assert rows["a"]["regions"] == {"scratch": 3, "dent": 1}
    assert rows["b"]["regions"] == {} and rows["b"]["annotated"]
    assert rows["c"]["regions"] == {} and not rows["c"]["annotated"]
    # The existing image-level contract is unchanged (the web page and filters still read it).
    assert rows["a"]["classes"] == ["background", "dent", "scratch"]
    train = index["splits"][0]
    assert train["regions"] == {"scratch": 3, "dent": 1}
    assert (train["empty"], train["unlabeled"]) == (1, 1)


def test_mask_regions_are_cached_until_the_mask_changes(tmp_path):
    path = tmp_path / "m.png"
    mask = np.zeros((16, 16), np.uint8)
    mask[1:3, 1:3] = 1
    Image.fromarray(mask, "L").save(path)
    assert mask_regions(path, 3) == ((0, 1), {1: 1})
    with patch("core.dataset_editor.Image.open", side_effect=AssertionError("decoded again")):
        assert mask_regions(path, 3) == ((0, 1), {1: 1})
    mask[8:10, 8:10] = 1
    Image.fromarray(mask, "L").save(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert mask_regions(path, 3) == ((0, 1), {1: 2})
    with pytest.raises(ValueError, match="범위"):
        mask_regions(path, 1)


def test_detection_regions_are_boxes(tmp_path):
    project = ProjectManager.create_new("boxes", "detect", str(tmp_path / "p"), ["part", "defect"])
    image = Path(project.data.train_dir) / "x.png"
    Image.new("RGB", (20, 20)).save(image)
    label = Path(project.data.root) / "labels/train/x.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("1 .5 .5 .2 .2\n1 .2 .2 .1 .1\n0 .7 .7 .1 .1\n", encoding="utf-8")
    row = scan_dataset(project)["images"][0]
    assert row["regions"] == {"part": 1, "defect": 2}


def test_segmentation_dataset_page_shows_regions_not_image_classes(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from widgets.dataset_widget import DatasetWidget, FILTER_UNLABELED

    app = QApplication.instance() or QApplication([])
    project, _, _ = _segment_project(tmp_path / "p")
    with patch.object(DatasetWidget, "_scan_all"):
        widget = DatasetWidget()
        widget.set_project(project)
        widget.show()
        widget._scan_worker = None  # _scan_ready accepts results from the current scan (sender) only
        widget._scan_ready(scan_dataset(project), {})
        app.processEvents()

        # ① No "add images into a class" selector for segmentation.
        assert not widget.class_combo.isVisible() and not widget.class_label_widget.isVisible()
        # ④ Filters: defect classes (no background), unlabeled, background-only.
        options = [widget.tab_filters["train"].itemText(i) for i in range(widget.tab_filters["train"].count())]
        assert options == ["전체", "scratch", "dent", FILTER_UNLABELED, "배경만 있는 이미지"]
        # ③ Badges count regions inside each image.
        badges = {Path(t.image_path).stem: t.class_label for t in widget._thumbs["train"]}
        assert badges == {"a": "scratch 3 · dent 1", "b": "배경만", "c": "미라벨"}
        for text, expected in (("scratch", {"a"}), ("배경만 있는 이미지", {"b"}), (FILTER_UNLABELED, {"c"})):
            widget.tab_filters["train"].setCurrentText(text)
            assert {Path(t.image_path).stem for t in widget._thumbs["train"]} == expected, text
        # ② Statistics: regions and images per class, then background-only / unlabeled rows.
        table = [[widget.stats_table.item(r, c).text() for c in range(2)]
                 for r in range(widget.stats_table.rowCount())]
        assert table == [["scratch", "3개 · 1장"], ["dent", "1개 · 1장"], ["배경만", "1장"], ["미라벨", "1장"]]
        assert widget.stats_hint.isVisible() and "blob" in widget.stats_hint.text()
        assert "영역 4개" in widget.summary_label.text()
        widget.close()
        widget.deleteLater()
    app.processEvents()


def test_object_task_class_delete_asks_which_class(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QInputDialog
    from widgets.dataset_widget import DatasetWidget

    app = QApplication.instance() or QApplication([])
    project, _, _ = _segment_project(tmp_path / "p")
    with patch.object(DatasetWidget, "_scan_all"):
        widget = DatasetWidget()
        widget.set_project(project)
        asked = {}

        def get_item(parent, title, label, items, current, editable):
            asked["items"] = list(items)
            return "", False  # cancel

        with patch.object(QInputDialog, "getItem", side_effect=get_item), \
                patch.object(widget, "_start_class_operation") as start:
            widget._delete_class()
        assert asked["items"] == ["scratch", "dent"]  # background (0) cannot be deleted
        start.assert_not_called()
        widget.close()
        widget.deleteLater()
    app.processEvents()
