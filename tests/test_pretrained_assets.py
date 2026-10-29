"""Pretrained weight bundle: which weights a run needs, partial preparation, and a clear pre-start check."""

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from builtin_assets import ASSET_FILES, builtin_asset_path, missing_asset_help, required_pretrained_asset


@pytest.mark.parametrize("task, mode, model, kwargs, expected", [
    ("segment", "builtin_finetune", "unet_resnet18", {}, "resnet18"),
    ("segment", "builtin_finetune", "deeplabv3plus_resnet34", {}, "resnet34"),
    ("segment", "builtin_scratch", "unet_resnet18", {}, None),
    ("segment", "builtin_transfer", "deeplabv3plus_resnet34", {}, None),
    ("segment", "sam2_transfer", "sam2_hiera_tiny", {}, "sam2_hiera_tiny"),
    ("classify", "efficientnet_finetune", "efficientnet_b1", {"efficientnet_model": "efficientnet_b1"}, "efficientnet_b1"),
    ("classify", "builtin_finetune", "convnext_v1_tiny", {}, "convnext_v1_tiny"),
    ("detect", "upstream_finetune", "libreyolo_detect_9t", {}, "libreyolo_detect_9t"),
    ("anomaly", "custom", "", {"anomaly_method": "patchcore", "patchcore_source": "imagenet",
                               "patchcore_backbone": "wide_resnet50_2"}, "wide_resnet50_2"),
    ("anomaly", "custom", "", {"anomaly_method": "patchcore", "patchcore_source": "patchcore"}, None),
])
def test_required_pretrained_asset(task, mode, model, kwargs, expected):
    assert required_pretrained_asset(task, mode, model, **kwargs) == expected


def test_missing_weights_message_says_how_to_get_only_that_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DVS_BUILTIN_ASSETS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        builtin_asset_path("resnet34")
    message = missing_asset_help("resnet34")
    assert "--only resnet34" in message and "무작위 초기화" in message and "build.bat" in message


def test_prepare_only_downloads_the_named_weights_and_merges_the_manifest(tmp_path):
    import prepare_builtin_assets as prepare

    def fake_download(url, cache):
        path = Path(cache) / Path(url).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights:" + url.encode())
        return path

    with patch.object(prepare, "cached_imagenet_weights", side_effect=fake_download) as download:
        prepare.prepare(tmp_path, ["resnet18"])
        prepare.prepare(tmp_path, ["resnet34"])
    assert download.call_count == 2
    files = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["files"]
    assert set(files) == {ASSET_FILES["resnet18"], ASSET_FILES["resnet34"]}
    os.environ["DVS_BUILTIN_ASSETS_DIR"] = str(tmp_path)
    try:
        assert builtin_asset_path("resnet18").is_file() and builtin_asset_path("resnet34").is_file()
    finally:
        del os.environ["DVS_BUILTIN_ASSETS_DIR"]
    with pytest.raises(SystemExit):
        prepare.prepare(tmp_path, ["not_a_model"])


def test_training_page_stops_before_starting_when_weights_are_missing(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from core.project import ProjectManager
    from widgets.training_widget import TrainingWidget

    QApplication.instance() or QApplication([])
    monkeypatch.setenv("DVS_BUILTIN_ASSETS_DIR", str(tmp_path / "empty"))
    project = ProjectManager.create_new("w", "segment", str(tmp_path / "p"), ["background", "scratch"])
    project.model.model_id = "unet_resnet18"
    project.training.training_mode = "builtin_finetune"
    widget = TrainingWidget()
    widget.set_project(project)
    widget.model_id_combo.setCurrentIndex(widget.model_id_combo.findData("unet_resnet18"))
    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("builtin_finetune"))
    warnings = []
    with patch.object(QMessageBox, "warning", side_effect=lambda *args: warnings.append(args[1:])):
        widget._start_training()
    titles = [title for title, _text in warnings]
    assert titles == ["사전학습 가중치 없음"], warnings
    assert "--only resnet18" in warnings[0][1]
    assert widget._run_project is None and not (widget.worker and widget.worker.isRunning())
    widget.deleteLater()
