"""Object tasks label regions inside an image, not the image.

Covers the region scan (counts, sizes, editor-consistent objects), training-size warnings,
region-level edits, and the desktop dataset page (badges, statistics, overlay thumbnails,
region list, opening the workspace focused on one region).
"""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from core.dataset_editor import edit_dataset, scan_dataset
from core.dataset_regions import edit_regions, segment_regions, training_footprint, training_scale
from core.mask_annotations import MaskDocument
from core.project import ProjectManager

cv2 = pytest.importorskip("cv2")

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
    mask[2:5, 2:12] = 1            # scratch 1 (3 px thick)
    mask[10:11, 20:35] = 1         # scratch 2 (1 px thick)
    mask[20, 5] = mask[21, 6] = 1  # diagonal neighbours: one region (8-connected)
    mask[24:29, 30:35] = 2         # dent 5×5
    mask[0, 39] = 255              # training exclusion: not a region
    Image.fromarray(mask, "L").save(masks / "a.png")
    Image.fromarray(np.zeros((30, 40), np.uint8), "L").save(masks / "b.png")
    return project, images, masks


def _rows(project):
    return {Path(row["path"]).stem: row for row in scan_dataset(project)["images"]}


# ── scan ───────────────────────────────────────────────────────

def test_scan_lists_regions_with_size_and_thickness(tmp_path):
    project, _, _ = _segment_project(tmp_path / "p")
    index = scan_dataset(project)
    rows = {Path(row["path"]).stem: row for row in index["images"]}
    assert rows["a"]["regions"] == {"scratch": 3, "dent": 1}
    items = sorted((i["class"], i["area"], i["width"], tuple(i["bbox"])) for i in rows["a"]["region_items"])
    assert items == [("dent", 25, 5, (30, 24, 5, 5)), ("scratch", 2, 1, (5, 20, 2, 2)),
                     ("scratch", 15, 1, (20, 10, 15, 1)), ("scratch", 30, 3, (2, 2, 10, 3))]
    assert rows["a"]["size"] == [40, 30]
    assert rows["b"]["regions"] == {} and rows["b"]["annotated"]
    assert rows["c"]["regions"] == {} and not rows["c"]["annotated"]
    # The image-level contract is unchanged (filters and other readers still use it).
    assert rows["a"]["classes"] == ["background", "dent", "scratch"]
    train = index["splits"][0]
    assert train["regions"] == {"scratch": 3, "dent": 1}
    assert (train["empty"], train["unlabeled"]) == (1, 1)


def test_touching_workspace_objects_are_separate_regions(tmp_path):
    """Regions follow workspace objects: two touching objects of one class are two regions."""
    project, images, _ = _segment_project(tmp_path / "p")
    document = MaskDocument(40, 30, 3, shapes=[
        {"kind": "polygon", "class_id": 1, "points": [.1, .1, .4, .1, .4, .5, .1, .5], "object_id": "left"},
        {"kind": "polygon", "class_id": 1, "points": [.4, .1, .7, .1, .7, .5, .4, .5], "object_id": "right"}])
    edit_dataset(project, "mask_annotations", [images / "b.png"], split="train", annotations=document.payload())
    items = _rows(project)["b"]["region_items"]
    assert sorted(i["key"] for i in items) == ["left", "right"]
    assert all(i["class"] == "scratch" for i in items)


def test_polygon_labels_are_rendered_like_the_workspace(tmp_path):
    project, images, _ = _segment_project(tmp_path / "p")
    labels = images.parent.parent / "labels" / "train"
    labels.mkdir(parents=True, exist_ok=True)
    (labels / "c.txt").write_text("2 .1 .1 .5 .1 .5 .5\n", encoding="utf-8")
    row = _rows(project)["c"]
    assert row["regions"] == {"dent": 1} and row["region_items"][0]["area"] > 50


def test_region_scan_is_cached_until_a_label_file_changes(tmp_path):
    project, images, masks = _segment_project(tmp_path / "p")
    first = segment_regions(project, images / "a.png", "train")
    with patch("core.dataset_regions.Image.open", side_effect=AssertionError("decoded again")):
        assert segment_regions(project, images / "a.png", "train") is first
    mask = np.asarray(Image.open(masks / "a.png")).copy()
    mask[15:17, 15:17] = 2
    Image.fromarray(mask, "L").save(masks / "a.png")
    stat = (masks / "a.png").stat()
    os.utime(masks / "a.png", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert len(segment_regions(project, images / "a.png", "train")[1]) == len(first[1]) + 1


def test_detection_regions_are_boxes(tmp_path):
    project = ProjectManager.create_new("boxes", "detect", str(tmp_path / "p"), ["part", "defect"])
    image = Path(project.data.train_dir) / "x.png"
    Image.new("RGB", (200, 100)).save(image)
    label = Path(project.data.root) / "labels/train/x.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("1 .5 .5 .2 .2\n1 .2 .2 .1 .1\n0 .7 .7 .1 .1\n", encoding="utf-8")
    row = scan_dataset(project)["images"][0]
    assert row["regions"] == {"part": 1, "defect": 2}
    first = row["region_items"][0]
    assert (first["key"], first["bbox"], first["area"], first["width"]) == ("row:0", [80, 40, 40, 20], 800, 20)


def test_training_footprint_uses_the_configured_input_size_and_crop(tmp_path):
    project, _, _ = _segment_project(tmp_path / "p")
    project.training.input_size = 20
    scale = training_scale(project, (40, 30))
    assert scale == (0.5, 20 / 30)  # segmentation stretches to a square input
    area, width = training_footprint({"area": 30, "width": 3, "bbox": [0, 0, 10, 3]}, scale)
    assert area == pytest.approx(30 * 0.5 * 20 / 30) and width == pytest.approx(3 * np.sqrt(0.5 * 20 / 30))
    project.training.patchcore_crop_enabled = True
    project.training.patchcore_crop_width, project.training.patchcore_crop_height = 20, 20
    assert training_scale(project, (40, 30)) == (1.0, 1.0)  # centre crop first, then resize


# ── region-level edits ─────────────────────────────────────────

def test_one_region_changes_class_and_other_pixels_stay(tmp_path):
    project, _, masks = _segment_project(tmp_path / "p")
    before = np.asarray(Image.open(masks / "a.png")).copy()
    row = _rows(project)["a"]
    thin = next(i for i in row["region_items"] if i["area"] == 15)
    result = edit_regions(project, "train", row["path"], [thin], class_id=2)
    assert Path(result["backup_path"]).is_dir()
    after = np.asarray(Image.open(masks / "a.png"))
    changed = after != before
    assert changed.sum() == 15 and np.all(after[changed] == 2) and np.all(before[changed] == 1)
    assert _rows(project)["a"]["regions"] == {"scratch": 2, "dent": 2}


def test_deleting_regions_and_stale_selections(tmp_path):
    project, _, masks = _segment_project(tmp_path / "p")
    row = _rows(project)["a"]
    scratches = [i for i in row["region_items"] if i["class"] == "scratch"]
    edit_regions(project, "train", row["path"], scratches, class_id=None)
    after = np.asarray(Image.open(masks / "a.png"))
    assert not (after == 1).any() and (after == 2).sum() == 25 and after[0, 39] == 255
    with pytest.raises(ValueError, match="바뀌었습니다"):  # the scan is now stale
        edit_regions(project, "train", row["path"], scratches[:1], class_id=2)
    with pytest.raises(ValueError):
        edit_regions(project, "train", row["path"], [row["region_items"][0]], class_id=0)  # background


def test_locked_workspace_objects_are_not_edited(tmp_path):
    project, images, _ = _segment_project(tmp_path / "p")
    document = MaskDocument(40, 30, 3, shapes=[
        {"kind": "polygon", "class_id": 1, "points": [.1, .1, .4, .1, .4, .5, .1, .5], "object_id": "kept"}],
        objects={"kept": {"locked": True}})
    edit_dataset(project, "mask_annotations", [images / "b.png"], split="train", annotations=document.payload())
    row = _rows(project)["b"]
    with pytest.raises(ValueError, match="잠긴"):
        edit_regions(project, "train", row["path"], row["region_items"], class_id=2)


def test_detection_region_edit_changes_one_box(tmp_path):
    project = ProjectManager.create_new("boxes", "detect", str(tmp_path / "p"), ["part", "defect"])
    image = Path(project.data.train_dir) / "x.png"
    Image.new("RGB", (100, 100)).save(image)
    label = Path(project.data.root) / "labels/train/x.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("1 .5 .5 .2 .2\n0 .2 .2 .1 .1\n", encoding="utf-8")
    items = scan_dataset(project)["images"][0]["region_items"]
    edit_regions(project, "train", str(image), [items[0]], class_id=0)
    assert label.read_text(encoding="utf-8").splitlines()[0].startswith("0 ")
    edit_regions(project, "train", str(image), [scan_dataset(project)["images"][0]["region_items"][1]], None)
    assert len(label.read_text(encoding="utf-8").splitlines()) == 1


# ── desktop dataset page ───────────────────────────────────────

@pytest.fixture
def page(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from widgets.dataset_widget import DatasetWidget

    app = QApplication.instance() or QApplication([])
    project, images, masks = _segment_project(tmp_path / "p")
    with patch.object(DatasetWidget, "_scan_all"):
        widget = DatasetWidget()
        widget.set_project(project)
        widget.show()

        def load(project=project):
            from core.dataset_scan import DatasetScan
            worker = DatasetScan(project)
            results = []
            worker.ready.connect(lambda index, thumbs: results.append((index, thumbs)))
            worker.run()  # the scan thread's body, synchronously
            widget._scan_worker = None  # _scan_ready accepts the current scan (sender) only
            widget._scan_ready(*results[0])
            app.processEvents()
        load()
        yield app, widget, project, load
        widget.close()
        widget.deleteLater()
    app.processEvents()


def test_segmentation_dataset_page_shows_regions_not_image_classes(page):
    from widgets.dataset_widget import FILTER_UNLABELED

    _app, widget, _project, _load = page
    assert not widget.class_combo.isVisible() and not widget.class_label_widget.isVisible()
    options = [widget.tab_filters["train"].itemText(i) for i in range(widget.tab_filters["train"].count())]
    assert options == ["전체", "scratch", "dent", FILTER_UNLABELED, "배경만 있는 이미지"]
    badges = {Path(t.image_path).stem: t.class_label for t in widget._thumbs["train"]}
    assert badges == {"a": "scratch 3 · dent 1", "b": "배경만", "c": "미라벨"}
    for text, expected in (("scratch", {"a"}), ("배경만 있는 이미지", {"b"}), (FILTER_UNLABELED, {"c"})):
        widget.tab_filters["train"].setCurrentText(text)
        assert {Path(t.image_path).stem for t in widget._thumbs["train"]} == expected, text
    table = [[widget.stats_table.item(r, c).text() for c in range(2)] for r in range(widget.stats_table.rowCount())]
    assert table == [["scratch", "3개 · 1장"], ["dent", "1개 · 1장"], ["배경만", "1장"], ["미라벨 (배경 학습)", "1장"]]
    assert "영역 4개" in widget.summary_label.text()


def test_size_statistics_warn_about_regions_under_one_training_pixel(page):
    app, widget, project, load = page
    project.training.input_size = 16  # 40×30 → 16×16: the 1-px scratches become ~0.5 px
    load()
    sizes = {widget.size_table.item(r, 0).text(): [widget.size_table.item(r, c).text() for c in range(1, 5)]
             for r in range(widget.size_table.rowCount())}
    assert sizes["scratch"] == ["3", "15", "2", "2 ⚠"] and sizes["dent"] == ["1", "25", "25", "0"]
    assert widget.size_warning.isVisible() and "2개" in widget.size_warning.text()
    project.training.input_size = 256
    load()
    assert not widget.size_warning.isVisible()


def test_thumbnails_overlay_regions_in_class_colours(page):
    from core.dataset_regions import class_rgb

    _app, widget, _project, _load = page
    thumb = next(t for t in widget._thumbs["train"] if t.image_path.endswith("a.png"))
    image = widget._dataset_overlays[thumb.image_path]
    plain_image = widget._dataset_thumbnails[thumb.image_path]
    # centre of the 5×5 dent (32, 26) in thumbnail coordinates
    x, y = int(32.5 * image.width() / 40), int(26.5 * image.height() / 30)
    pixel, plain = image.pixelColor(x, y), plain_image.pixelColor(x, y)
    red, green, blue = class_rgb(2)
    assert pixel != plain
    # blended towards the dent colour (alpha 165/255 over grey 100)
    assert abs(pixel.red() - (red * 165 + 100 * 90) / 255) <= 2
    assert abs(pixel.blue() - (blue * 165 + 100 * 90) / 255) <= 2
    widget.overlay_check.setChecked(False)
    shown = next(t for t in widget._thumbs["train"] if t.image_path.endswith("a.png"))
    assert shown is not thumb  # thumbnails were rebuilt without the overlay


def test_region_list_filters_opens_focused_and_edits(page):
    from PySide6.QtWidgets import QMessageBox
    from widgets.dataset_widget import region_classes
    from widgets.region_list_dialog import RegionListDialog

    app, widget, project, load = page
    project.training.input_size = 16
    load()
    dialog = RegionListDialog(widget, region_classes(project))
    assert dialog.model.rowCount() == 4
    dialog.small_check.setChecked(True)
    assert dialog.proxy.rowCount() == 2  # the two 1-px scratches
    dialog.small_check.setChecked(False)
    dialog.class_combo.setCurrentIndex(dialog.class_combo.findData("dent"))
    assert dialog.proxy.rowCount() == 1
    dialog.table.selectRow(0)
    with patch.object(widget, "_open_image_preview") as opened:
        dialog._open()
    path, = opened.call_args.args
    assert path.endswith("a.png") and opened.call_args.kwargs["focus"][0] == [30, 24, 5, 5]
    with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes), \
            patch.object(widget, "_scan_all", side_effect=load):
        assert widget.edit_regions(dialog.selected_rows(), None)
    assert _rows(project)["a"]["regions"] == {"scratch": 3}
    assert dialog.model.rowCount() == 3  # the open list reloads after the rescan
    dialog.deleteLater()


def test_workspace_focus_selects_a_single_region_object_only(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from widgets.segmentation_annotation import SegmentationAnnotationDialog

    app = QApplication.instance() or QApplication([])
    image = tmp_path / "x.png"
    Image.new("RGB", (200, 160), 90).save(image)
    base = np.zeros((160, 200), np.uint8)
    base[10:14, 10:30] = 1
    base[100:104, 120:150] = 1  # same class, second blob: one PNG layer object
    shapes = [{"kind": "polygon", "class_id": 2, "points": [.5, .1, .7, .1, .7, .3, .5, .3], "object_id": "dent"}]
    document = MaskDocument(200, 160, 3, base=base, shapes=shapes)
    dialog = SegmentationAnnotationDialog(str(image), NAMES, document, lambda values: None)
    dialog.show()
    app.processEvents()
    dialog._fit()
    fit_zoom = dialog.view.transform().m11()
    blob = (base[100:104, 120:150] > 0).sum()
    dialog.focus_region([120, 100, 30, 4], "base:1", [121, 101], int(blob))
    assert dialog.view.transform().m11() > fit_zoom and dialog.selected == -1
    assert "연결 부분 나누기" in dialog.status.text()
    owner = dialog.document.owner_at(120, 30)
    area = int(np.count_nonzero(dialog.document._owners == owner))
    dialog.focus_region(dialog.document.bounds(owner), "dent", [120, 30], area)
    assert dialog.selected == owner
    dialog.dirty = False
    dialog.reject()
    dialog.deleteLater()
    app.processEvents()


def test_object_task_class_delete_asks_which_class(page):
    from PySide6.QtWidgets import QInputDialog

    _app, widget, _project, _load = page
    asked = {}

    def get_item(parent, title, label, items, current, editable):
        asked["items"] = list(items)
        return "", False  # cancel

    with patch.object(QInputDialog, "getItem", side_effect=get_item), \
            patch.object(widget, "_start_class_operation") as start:
        widget._delete_class()
    assert asked["items"] == ["scratch", "dent"]  # background (0) cannot be deleted
    start.assert_not_called()


# ── segmentation background (class 0) ──────────────────────────

def test_new_segment_project_gets_background_so_first_defect_is_drawable(tmp_path):
    """'scratch, dent'만 입력해도 0번에 배경이 들어가 scratch가 1번(그릴 수 있는 클래스)이 된다."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from widgets.project_widget import ProjectWidget

    QApplication.instance() or QApplication([])
    widget = ProjectWidget()
    created = []
    widget.project_created.connect(created.append)
    widget._on_task_selected("segment")
    widget.name_edit.setText("defects")
    widget.path_edit.setText(str(tmp_path))
    with patch.object(QMessageBox, "information") as info, patch.object(QMessageBox, "warning") as warning:
        widget.class_edit.setText("background")  # 배경만으로는 그릴 클래스가 없다
        widget._create_project()
        assert not created and warning.called
        widget.class_edit.setText("scratch, dent")
        widget._create_project()
    assert created[0].data.class_names == ["background", "scratch", "dent"]
    assert "배경 클래스를 자동으로 추가" in info.call_args.args[2]
    widget.deleteLater()


def test_dataset_page_inserts_background_into_project_without_one(tmp_path):
    """배경 없이 만든 기존 프로젝트: 버튼으로 0번에 배경을 넣고 마스크 번호를 한 칸씩 올린다."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from widgets.dataset_widget import DatasetWidget

    app = QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("legacy", "segment", str(tmp_path / "p"), ["scratch", "dent"])
    ProjectManager.save(project)
    masks = Path(project.data.root) / "masks" / "train"
    masks.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((4, 4), np.uint8)
    mask[0, 0] = 1  # dent (scratch는 0번이라 그릴 수 없었다)
    Image.fromarray(mask, "L").save(masks / "a.png")
    with patch.object(DatasetWidget, "_scan_all"):
        widget = DatasetWidget()
        widget.set_project(project)
        widget.show()
        assert widget.fix_background_btn.isVisible()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes), \
                patch.object(widget, "_start_class_operation",
                             side_effect=lambda operation, callback, _message: callback(operation())):
            widget._insert_background()
        assert project.data.class_names == ["background", "scratch", "dent"]
        assert np.asarray(Image.open(masks / "a.png"))[0, 0] == 2
        assert not widget.fix_background_btn.isVisible()
        assert "1번으로 그릴 수 있습니다" in widget.summary_label.text()
        widget.close()
        widget.deleteLater()
    app.processEvents()
