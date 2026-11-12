"""Exercise the same UI entry points used in the frozen/installed EXE smoke.

Source CPU execution is a separate check; it does not replace Windows freezing.
"""
import json
from pathlib import Path

from core import job_manager
from PySide6.QtWidgets import QMessageBox
from tools.desktop_package_smoke import run


def test_desktop_smoke_ui_contract_on_source_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEP_STUDIO_DESKTOP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(job_manager, "_manager", None)
    # The smoke deliberately raises instead of opening modal error boxes.
    # Restore these process-global functions when the source test finishes.
    monkeypatch.setattr(QMessageBox, "warning", QMessageBox.warning)
    monkeypatch.setattr(QMessageBox, "critical", QMessageBox.critical)
    report = tmp_path / "source-smoke.json"
    try:
        run(str(report), require_cuda_runtime=False)
        result = json.loads(report.read_text(encoding="utf-8"))
        assert not result["frozen"]
        assert result["project_class_generation_saved"]
        assert result["defect_review_saved"]
        assert result["persistent_results"]
        assert result["inference_status"] != "error"
        assert Path(report.with_suffix(".png")).is_file()
    finally:
        job_manager.desktop_manager().close()
