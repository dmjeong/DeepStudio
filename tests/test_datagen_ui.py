"""Desktop real-data page and API project isolation integration."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("PySide6") and importlib.util.find_spec("psutil"), "Qt runtime required")
class DesktopDataGenTests(unittest.TestCase):
    def test_custom_item_mask_and_project_restore(self):
        from PySide6.QtWidgets import QApplication
        from core.project import ProjectManager
        from widgets.defect_gen_widget import DefectGenWidget
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = ProjectManager.create_new("custom", "classify", str(root / "project"), ["NG", "OK"])
            widget = DefectGenWidget()
            widget.set_project(project)
            widget.resize(1280, 720)
            widget.show()
            app.processEvents()
            self.assertEqual(widget.tabs.count(), 4)
            widget.item_name.setText("내가 학습할 모양")
            widget.class_name.setCurrentIndex(widget.class_name.findData("NG"))
            widget.save_item(False)
            item_id = widget.item_id()
            source = root / "images"
            source.mkdir()
            Image.new("RGB", (96, 64), (10, 30, 50)).save(source / "real.png")
            widget.source.setText(str(source))
            widget.group.setText("lot-A")
            widget.import_images()
            self.assertEqual(widget.images.count(), 1)
            widget.editor.canvas.mask.fill(0xffffffff)
            widget.save_mask()
            widget.item_name.setText("새 이름")
            widget.save_item(True)
            self.assertEqual(widget.item_id(), item_id)
            restored = DefectGenWidget()
            restored.set_project(project)
            self.assertEqual(restored.items.currentText(), "새 이름")
            self.assertEqual(restored.images.count(), 1)
            self.assertTrue(restored.editor.png()["png"])
            widget.show_legacy()
            widget.legacy._worker = object()
            self.assertIs(widget._run_project, project)
            widget.set_project(None)
            self.assertIs(widget.project, project)
            widget.legacy._worker = None
            self.assertIsNone(widget._run_project)
            restored.close()
            widget.close()
            app.processEvents()


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx") and (Path(__file__).resolve().parents[1] / "webapp/server.py").exists(), "API runtime required")
class DataGenAPITests(unittest.TestCase):
    def test_project_scoped_items_and_mask_image(self):
        from fastapi.testclient import TestClient
        from webapp.server import create_app
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with TestClient(create_app(root / "state"), base_url="http://127.0.0.1", headers={"X-Studio-Request": "1"}) as client:
                project = client.post("/api/projects", json={"name": "first", "task": "classify", "parent": str(root), "class_names": ["NG"]}).json()
                payload = {"project_path": project["filepath"], "values": {"name": "사용자 항목", "class_name": "NG"}}
                response = client.post("/api/datagen/item", json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(client.get("/api/datagen").json()["items"][0]["name"], "사용자 항목")
                client.post("/api/projects", json={"name": "second", "task": "classify", "parent": str(root), "class_names": ["NG"]})
                stale = client.post("/api/datagen/item", json=payload)
                self.assertEqual(stale.status_code, 400)
                self.assertEqual(client.get("/api/datagen").json()["items"], [])


if __name__ == "__main__":
    unittest.main()
