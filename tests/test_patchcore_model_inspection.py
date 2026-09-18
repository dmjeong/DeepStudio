"""Saved PatchCore -> actual model-load button -> process inspection -> CPU inference."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gui"), str(ROOT / "python")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "torchvision", "PySide6", "cv2", "matplotlib"))


@unittest.skipUnless(AVAILABLE, "Qt/PyTorch runtime required")
class PatchCoreModelInspectionTests(unittest.TestCase):
    def test_null_class_names_and_invalid_metadata_do_not_leave_false_unloaded_state(self):
        from PySide6.QtWidgets import QApplication
        from widgets.inference_widget import InferenceWidget
        app = QApplication.instance() or QApplication([])
        widget = InferenceWidget()
        try:
            result = {"status": "completed", "output": {"weights": "existing.pt", "device": "cpu",
                      "task": "anomaly", "input_size": [32, 32], "class_names": None}}
            widget._model_inspected(result)
            self.assertTrue(widget._process_model)
            self.assertEqual(widget.class_names, [])
            self.assertTrue(widget.infer_btn.isEnabled())
            self.assertIn("준비 완료", widget.model_info.text())
            result["output"] = {**result["output"], "weights": "invalid.pt", "input_size": [0, 32]}
            widget._model_inspected(result)
            self.assertEqual(widget._active_checkpoint, "existing.pt")
            self.assertIn("모델 로드 실패", widget.model_info.text())
            self.assertIn("기존 모델 유지", widget.model_info.text())
            self.assertTrue(widget.infer_btn.isEnabled())
        finally:
            widget._clear_model()
            widget.deleteLater()
            app.processEvents()

    def test_saved_anomaly_load_button_and_cpu_inference_in_actual_worker_processes(self):
        import numpy as np
        import torch
        from PIL import Image
        from PySide6.QtWidgets import QApplication, QPushButton
        from torch.utils.data import DataLoader
        from core.inference_loading import load_cpu_engine
        from patchcore import PatchCore
        from patchcore_data import PatchCoreDataset
        from webapp.jobs import JobManager
        from widgets.inference_widget import InferenceWidget
        app = QApplication.instance() or QApplication([])
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                image = root / "normal.png"
                Image.fromarray(np.random.default_rng(20).integers(0, 256, (40,48,3), dtype=np.uint8)).save(image)
                # Local model fixture tests state transfer, not pretrained anomaly quality.
                with torch.random.fork_rng():
                    torch.manual_seed(11)
                    model = PatchCore(backbone_name="resnet18", device="cpu", pretrained=False,
                                      input_size=32, max_candidates=16, max_memory_bank=8,
                                      sampling_ratio=.5, n_neighbors=3, center_crop={"width": 32, "height": 24})
                model.fit(DataLoader(PatchCoreDataset([str(image)], 32, center_crop=model.center_crop), batch_size=1))
                model.anomaly_threshold = 3
                weights = root / "best.pt"
                model.save(weights)
                reference = load_cpu_engine(weights)
                self.assertEqual(reference.class_names, [])
                expected = reference.infer(str(image))
                self.assertEqual(expected.status, "ok", expected.error)
                self.assertEqual(expected.details["raw_threshold"], 3)
                self.assertGreaterEqual(expected.score, 0)
                self.assertLessEqual(expected.score, 1)
                manager = JobManager(root / "state")
                with patch("core.job_manager._manager", manager), \
                        patch("widgets.inference_widget.QMessageBox.warning") as warning, \
                        patch("widgets.inference_widget.QMessageBox.critical") as critical:
                    widget = InferenceWidget()
                    def wait_for(field):
                        deadline = time.monotonic() + 90
                        while getattr(widget, field) is not None:
                            app.processEvents()
                            if time.monotonic() > deadline:
                                raise AssertionError(f"{field} timeout: {widget.model_info.text()}")
                            time.sleep(.005)
                        app.processEvents()
                    try:
                        widget.infer_device_combo.setCurrentIndex(widget.infer_device_combo.findData("cpu"))
                        widget.ckpt_edit.setText(str(weights))
                        next(button for button in widget.findChildren(QPushButton) if button.text() == "모델 로드").click()
                        wait_for("_model_inspection")
                        self.assertTrue(widget._process_model, widget.model_info.text())
                        self.assertTrue(widget.infer_btn.isEnabled(), widget.model_info.text())
                        self.assertEqual(widget.class_names, [])
                        self.assertEqual(str(widget._infer_device), "cpu")
                        self.assertIn("PatchCore 준비 완료", widget.model_info.text())
                        self.assertIn("CPU", widget.model_info.text())
                        self.assertIn("메모리 뱅크: 8개", widget.model_info.text())
                        self.assertIn("중앙 크롭: 32×24", widget.model_info.text())
                        self.assertFalse(widget.gradcam_checkbox.isEnabled())
                        self.assertIsNone(widget.model)
                        self.assertIsNone(widget._patchcore_model, "The model lives in the inference process")
                        self.assertEqual(widget.result_review._slider_range, (0, 1))
                        self.assertEqual(widget.result_review.saved_threshold, .5)
                        widget._current_image = str(image)
                        widget.infer_btn.click()
                        wait_for("_inference_worker")
                        actual = widget._inference_results[str(image)]
                        self.assertEqual(actual.status, "ok", actual.error)
                        self.assertEqual(actual.task, "anomaly")
                        self.assertEqual(actual.threshold, .5)
                        self.assertEqual(actual.details["raw_threshold"], 3)
                        self.assertEqual(actual.details["crop_box"], [8, 8, 40, 32])
                        np.testing.assert_allclose(actual.score, expected.score, rtol=1e-5, atol=1e-5)
                        before = weights.read_bytes()
                        json_file = root / "crop.json"
                        crop = {"width": 24, "height": 16}
                        for mode, region in (("json", crop), ("full", None), ("model", model.center_crop)):
                            widget.crop_mode_combo.setCurrentIndex(widget.crop_mode_combo.findData(mode))
                            if mode == "json":
                                json_file.write_text(json.dumps({"center_crop": crop}))
                                widget.crop_json_edit.setText(str(json_file))
                                self.assertIn("24×16", widget.crop_region_info.text())
                            widget.infer_btn.click()
                            self.assertFalse(widget.crop_mode_combo.isEnabled())
                            self.assertFalse(widget.crop_json_edit.isEnabled())
                            json_file.write_text('{"center_crop": null}')
                            wait_for("_inference_worker")
                            actual = widget._inference_results[str(image)]
                            self.assertEqual(actual.status, "ok", actual.error)
                            self.assertEqual(actual.details["input_region"]["mode"], mode)
                            self.assertEqual(actual.details["input_region"]["center_crop"], region)
                            reference.configure_input_region({"mode":"json", "center_crop": region})
                            np.testing.assert_allclose(actual.score, reference.infer(str(image)).score, rtol=1e-5, atol=1e-5)
                        self.assertEqual(weights.read_bytes(), before)
                        self.assertIn("PatchCore 준비 완료", widget.model_info.text())
                        self.assertTrue(widget.infer_btn.isEnabled())
                        jobs = manager.list(limit=10)
                        self.assertEqual({row["kind"] for row in jobs}, {"inspect_model", "infer"})
                        self.assertTrue(all(row["status"] == "completed" for row in jobs))
                        warning.assert_not_called()
                        critical.assert_not_called()
                    finally:
                        for field in ("_model_inspection", "_inference_worker"):
                            worker = getattr(widget, field)
                            if worker is not None:
                                worker.stop()
                                worker.wait(15000)
                                app.processEvents()
                        manager.close()
                        widget._clear_model()
                        widget.deleteLater()
                        app.processEvents()
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
