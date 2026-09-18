"""Exercise shared class selection against actual split contents in the Qt UI."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "Qt runtime required")
class DatasetFilterTests(unittest.TestCase):
    def test_dataset_image_preview_zoom_is_separate_from_grid(self):
        from PySide6.QtWidgets import QApplication
        from widgets.dataset_widget import DatasetImagePreviewDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.png"
            Image.new("RGB", (320, 180), 120).save(path)
            dialog = DatasetImagePreviewDialog(str(path))
            app.processEvents()
            self.assertEqual(dialog.zoom.value(), 100)
            original_width = dialog.image.pixmap().width()
            dialog.zoom.setValue(200)
            app.processEvents()
            self.assertGreater(dialog.image.pixmap().width(), original_width)
            dialog.close()
            dialog.deleteLater()

    def test_class_selection_follows_tabs_and_top_split_even_when_empty(self):
        from PySide6.QtWidgets import QApplication
        from core.project import ProjectManager
        from core.dataset_editor import scan_dataset
        from widgets.dataset_widget import DatasetWidget, SPLITS
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            project = ProjectManager.create_new("shared-filter", "classify", str(Path(directory) / "project"), ["empty", "full"])
            for split in SPLITS:
                for name in (["full"] if split == "test" else ["empty", "full"]):
                    Image.new("RGB", (32, 24), 120).save(Path(project.data.root) / split / name / f"{name}.png")
            with patch.object(DatasetWidget, "_scan_all"):
                widget = DatasetWidget()
                widget.set_project(project)
                widget._dataset_index = scan_dataset(project)
                for split in SPLITS:
                    widget._refresh_browser(split)
                    self.assertEqual(widget.tab_scrolls[split].horizontalScrollBarPolicy().name, "ScrollBarAlwaysOn")
                    self.assertEqual(widget.tab_scrolls[split].verticalScrollBarPolicy().name, "ScrollBarAlwaysOn")
                    self.assertGreaterEqual(widget.tab_grid_widgets[split].minimumWidth(), 760)
                widget.split_tabs.setCurrentIndex(SPLITS.index("val"))
                widget._select_all("val")
                widget.tab_filters["val"].setCurrentText("empty")
                self.assertFalse(widget._selected_paths, "Hidden images must leave the bulk selection")
                for split in ("train", "test", "val"):
                    widget.split_tabs.setCurrentIndex(SPLITS.index(split))
                    self.assertEqual(widget.tab_filters[split].currentText(), "empty")
                    self.assertEqual(widget.class_combo.currentText(), "empty")
                    self.assertEqual(len(widget._thumbs[split]), 0 if split == "test" else 1)
                    self.assertTrue(all(row.class_label == "empty" for row in widget._thumbs[split]))
                widget.split_combo.setCurrentText("train")
                self.assertEqual(widget.tab_filters["train"].currentText(), "empty")
                widget.set_project(project)
                self.assertTrue(all(combo.currentText() == "empty" for combo in widget.tab_filters.values()))
                widget.tab_filters["train"].setCurrentText("전체")
                widget.split_combo.setCurrentText("test")
                self.assertTrue(all(combo.currentText() == "전체" for combo in widget.tab_filters.values()))
                widget.class_combo.setCurrentText("full")
                self.assertTrue(all(combo.currentText() == "full" for combo in widget.tab_filters.values()))
                project.data.class_names.remove("full")
                widget.set_project(project)
                self.assertTrue(all(combo.currentText() == "전체" for combo in widget.tab_filters.values()))
                widget.tab_filters["val"].setCurrentText("empty")
                other = ProjectManager.create_new("other", "classify", str(Path(directory) / "other"), ["empty", "full"])
                widget.set_project(other)
                self.assertTrue(all(combo.currentText() == "전체" for combo in widget.tab_filters.values()))
                widget.close()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
