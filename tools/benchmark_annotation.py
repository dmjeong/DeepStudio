"""Run without arguments: benchmark a 4096², 128-object editing fixture.

Requires the application's Python environment. No user images or weights are
read. The result describes this machine only, not Windows/i7 acceptance.
"""

import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]

import numpy as np
from core.annotation_tools import AnnotationDraftStore, grid_boxes
from core.mask_annotations import MaskDocument
from PIL import Image
from PySide6 import __version__ as qt_binding_version
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog
from widgets.segmentation_annotation import SegmentationAnnotationDialog


def measure():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="dvs-annotation-benchmark-") as directory:
        image = Path(directory) / "chips.png"
        Image.new("RGB", (4096, 4096), "gray").save(image)
        document = MaskDocument(4096, 4096, 3)
        for i, row in enumerate(
            grid_boxes(
                [0.08, 0.04, 0.07, 0.04],
                8,
                16,
                column_end=[0.92, 0.04],
                row_end=[0.08, 0.96],
            )
        ):
            document.shapes.append(
                {
                    "kind": "polygon",
                    "class_id": 1,
                    "points": row["points"],
                    "object_id": f"chip-{i}",
                }
            )
        started = time.perf_counter()
        dialog = SegmentationAnnotationDialog(
            image,
            ["background", "part", "defect"],
            document,
            lambda _: None,
            draft_store=AnnotationDraftStore(Path(directory) / "drafts"),
        )
        dialog.show()
        app.processEvents()
        initial_ms = 1000 * (time.perf_counter() - started)
        view = dialog.view
        dialog.erase_scope.setCurrentIndex(2)
        dialog._set_mode("erase")
        dialog.brush_size.setValue(32)
        points = [(400 + i * 8, 160 + (i % 4) * 5) for i in range(160)]
        QTest.mousePress(
            view.viewport(),
            Qt.MouseButton.LeftButton,
            pos=view.mapFromScene(QPointF(*points[0])),
        )
        samples = []
        for point in points[1:]:
            start = time.perf_counter()
            QTest.mouseMove(view.viewport(), view.mapFromScene(QPointF(*point)))
            app.processEvents()
            samples.append(1000 * (time.perf_counter() - start))
        start = time.perf_counter()
        QTest.mouseRelease(
            view.viewport(),
            Qt.MouseButton.LeftButton,
            pos=view.mapFromScene(QPointF(*points[-1])),
        )
        app.processEvents()
        commit_ms = 1000 * (time.perf_counter() - start)
        result = {
            "system": platform.platform(),
            "cpu": platform.processor(),
            "python": platform.python_version(),
            "pyside6": qt_binding_version,
            "qt_platform": os.environ.get("QT_QPA_PLATFORM", "native"),
            "image": [4096, 4096],
            "objects": 128,
            "brush_px": 32,
            "window": [dialog.width(), dialog.height()],
            "initial_window_ms": initial_ms,
            "pointer_to_processed_paint_ms": {
                "median": float(np.median(samples)),
                "p95": float(np.percentile(samples, 95)),
                "max": float(max(samples)),
                "samples": len(samples),
                "over_100ms": sum(s > 100 for s in samples),
            },
            "stroke_commit_ms": commit_ms,
            "notes": "Qt synthetic mouse events plus processEvents; target Windows/native display requires a separate run.",
        }
        dialog._draft_timer.stop()
        if dialog.draft_worker:
            dialog.draft_worker.wait(10000)
            app.processEvents()
        dialog.dirty = False
        QDialog.reject(dialog)
        dialog.deleteLater()
        app.processEvents()
        return result


if __name__ == "__main__":
    output = ROOT / "local-artifacts" / "annotation-benchmark.json"
    output.parent.mkdir(exist_ok=True)
    result = measure()
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")
