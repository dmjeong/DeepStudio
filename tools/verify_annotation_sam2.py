"""No-argument offline GUI integration check using the bundled SAM2 Tiny.

Run with the same Python environment as the application, after build assets
have been prepared. No user data is read; no weights are downloaded.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from core.annotation_tools import AnnotationDraftStore
from core.dataset_editor import edit_dataset
from core.mask_annotations import load_mask_document
from core.project import ProjectManager
from PIL import Image, ImageDraw
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QDialog
from widgets.segmentation_annotation import SegmentationAnnotationDialog


def verify():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="dvs-sam2-gui-") as directory:
        project = ProjectManager.create_new(
            "sam2-gui", "segment", directory, ["background", "part"]
        )
        ProjectManager.save(project)
        path = Path(project.data.train_dir) / "synthetic-part.png"
        image = Image.new("RGB", (256, 192), "#666666")
        ImageDraw.Draw(image).rectangle((80, 60, 170, 130), fill="#cccccc")
        image.save(path)
        dialog = SegmentationAnnotationDialog(
            path,
            project.data.class_names,
            load_mask_document(project, path, "train"),
            lambda payload: edit_dataset(
                project, "mask_annotations", [str(path)], annotations=payload
            ),
            draft_store=AnnotationDraftStore(Path(directory) / "drafts"),
        )
        dialog.show()
        app.processEvents()
        dialog.ai_device.setCurrentIndex(1)
        ticks = [0]
        ticker = QTimer()
        ticker.setInterval(20)
        ticker.timeout.connect(lambda: ticks.__setitem__(0, ticks[0] + 1))
        ticker.start()
        started = time.monotonic()
        try:
            dialog._ai_prompt({"point": [125.0, 95.0], "label": 1})
            loop = QEventLoop()
            worker = dialog.ai_worker
            worker.finished.connect(loop.quit)
            loop.exec()
            app.processEvents()
            ticker.stop()
            if dialog._ai_proposal is None:
                raise RuntimeError(dialog.ai_status.text())
            proposal = dialog._ai_proposal[1].copy()
            if not proposal.any():
                raise RuntimeError("SAM2 produced an empty proposal.")
            dialog.ai_apply.click()
            app.processEvents()
            expected = dialog.document.render()
            dialog._save()
            dialog.worker.wait()
            app.processEvents()
            if dialog.dirty:
                raise RuntimeError(dialog.status.text())
            actual = load_mask_document(project, path, "train").render()
            np.testing.assert_array_equal(actual, expected)
            return {
                "model": "sam2_hiera_tiny",
                "device": "cpu",
                "image": [256, 192],
                "proposal_pixels": int(proposal.sum()),
                "png_roundtrip_equal": True,
                "save_keeps_workspace_open": dialog.isVisible(),
                "ui_timer_ticks_during_prediction": ticks[0],
                "elapsed_seconds": time.monotonic() - started,
            }
        finally:
            ticker.stop()
            dialog._draft_timer.stop()
            for worker in (dialog.ai_worker, dialog.worker, dialog.draft_worker):
                if worker:
                    worker.requestInterruption()
                    worker.wait()
            app.processEvents()
            QDialog.reject(dialog)
            dialog.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    result = verify()
    output = ROOT / "local-artifacts" / "annotation-sam2-verification.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
