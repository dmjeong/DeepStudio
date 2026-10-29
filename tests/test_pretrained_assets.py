"""Pretrained weight bundle: which weights a run needs, partial preparation, and a clear pre-start check."""

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from builtin_assets import (ASSET_FILES, builtin_asset_path, missing_asset_help, required_pretrained_asset,
                            user_asset_root)


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


def test_missing_weights_message_says_where_to_download_and_put_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DVS_BUILTIN_ASSETS_DIR", str(tmp_path))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    with pytest.raises(FileNotFoundError):
        builtin_asset_path("resnet34")
    message = missing_asset_help("resnet34")
    assert "https://download.pytorch.org/models/resnet34-b627a593.pth" in message
    assert str(user_asset_root()) in message and "manifest는 필요 없습니다" in message
    assert "--only resnet34" in message and "무작위 초기화" in message


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
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    warnings = []
    with patch.object(QMessageBox, "warning", side_effect=lambda *args: warnings.append(args[1:])), \
            patch("widgets.pretrained_weights.offer_weight_import", return_value=False) as offer:
        widget._start_training()
    # 가중치가 없으면 학습을 시작하지 않고, 받는 곳·넣는 곳을 알리며 가져오기를 제안한다
    assert offer.call_args.args[1] == "resnet18" and not warnings, warnings
    assert widget._run_project is None and not (widget.worker and widget.worker.isRunning())
    widget.deleteLater()


# ── 수동으로 받은 가중치 (manifest 없이) ──────────────────────────────

def _zip_weights(path):
    """PyTorch 저장 형식(zip)인 작은 파일."""
    import zipfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"weights")
    return path


def test_manual_file_without_manifest_or_entry_is_accepted(tmp_path, monkeypatch):
    """사용자가 겪은 문제: 수동으로 넣은 파일이 manifest에 없다고 거부됐다."""
    bundle = tmp_path / "bundle"
    monkeypatch.setenv("DVS_BUILTIN_ASSETS_DIR", str(bundle))
    target = _zip_weights(bundle / ASSET_FILES["sam2_hiera_tiny"])
    assert builtin_asset_path("sam2_hiera_tiny") == target.resolve()           # manifest 파일 자체가 없음
    (bundle / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {}}), encoding="utf-8")
    other = _zip_weights(bundle / ASSET_FILES["libreyolo_detect_9t"])
    assert builtin_asset_path("libreyolo_detect_9t") == other.resolve()        # manifest에 항목이 없음
    html = bundle / ASSET_FILES["re_detr_v4_small"]
    html.write_text("<html>download page</html>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PyTorch 가중치 파일 형식이 아닙니다"):
        builtin_asset_path("re_detr_v4_small")


def test_torchvision_file_is_checked_by_the_hash_in_its_official_name(tmp_path, monkeypatch):
    import hashlib
    import builtin_assets
    monkeypatch.setenv("DVS_BUILTIN_ASSETS_DIR", str(tmp_path / "bundle"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    content = b"resnet weights"
    prefix = hashlib.sha256(content).hexdigest()[:8]
    monkeypatch.setitem(builtin_assets.ASSET_FILES, "resnet18", f"torchvision/resnet18-{prefix}.pth")
    cached = tmp_path / "torch" / "hub" / "checkpoints" / f"resnet18-{prefix}.pth"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(content)
    assert builtin_asset_path("resnet18") == cached.resolve()                  # torch hub 캐시도 찾는다
    cached.write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="해시 불일치"):
        builtin_asset_path("resnet18")


def test_user_folder_accepts_flat_files_and_import_records_the_hash(tmp_path, monkeypatch):
    import builtin_assets
    monkeypatch.setenv("DVS_BUILTIN_ASSETS_DIR", str(tmp_path / "bundle"))
    flat = _zip_weights(user_asset_root() / "sam2.1_hiera_small.pt")         # 하위 폴더 없이 넣어도 된다
    assert builtin_asset_path("sam2_hiera_small") == flat.resolve()
    downloaded = _zip_weights(tmp_path / "Downloads" / "LibreYOLO9t (1).pt")  # 브라우저가 이름을 바꿔도 된다
    target = builtin_assets.import_asset("libreyolo_detect_9t", downloaded)
    assert target == (user_asset_root() / ASSET_FILES["libreyolo_detect_9t"]).resolve()
    manifest = json.loads((user_asset_root() / "manifest.json").read_text(encoding="utf-8"))
    assert ASSET_FILES["libreyolo_detect_9t"] in manifest["files"]
    assert builtin_asset_path("libreyolo_detect_9t") == target
    bad = tmp_path / "Downloads" / "page.pt"
    bad.write_text("<html>", encoding="utf-8")
    with pytest.raises(RuntimeError):
        builtin_assets.import_asset("re_detr_v4_small", bad)
    assert builtin_assets.missing_assets() and "libreyolo_detect_9t" not in builtin_assets.missing_assets()


def test_build_keeps_going_when_one_weight_cannot_be_downloaded(tmp_path, capsys):
    import prepare_builtin_assets as prepare

    def fake_download(url, cache):
        if "resnet34" in url:
            raise OSError("blocked by proxy")
        path = Path(cache) / Path(url).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights:" + url.encode())
        return path

    with patch.object(prepare, "cached_imagenet_weights", side_effect=fake_download):
        prepare.prepare(tmp_path, ["resnet18", "resnet34"])
        output = capsys.readouterr().out
        assert "[failed] resnet34" in output and "resnet34-b627a593.pth" in output and "빌드는 계속" in output
        files = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["files"]
        assert set(files) == {ASSET_FILES["resnet18"]}
        with pytest.raises(RuntimeError, match="resnet34"):
            prepare.prepare(tmp_path, ["resnet34"], strict=True)
        # 다시 실행해도 이미 받은 항목은 manifest에 남는다
        assert ASSET_FILES["resnet18"] in json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["files"]


def test_settings_page_lists_weights_and_imports_the_selected_one(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from widgets.pretrained_weights import PretrainedWeightsGroup

    QApplication.instance() or QApplication([])
    monkeypatch.setenv("DVS_BUILTIN_ASSETS_DIR", str(tmp_path / "bundle"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    group = PretrainedWeightsGroup()
    rows = {group.table.item(r, 0).text(): group.table.item(r, 2).text() for r in range(group.table.rowCount())}
    assert set(rows) == set(ASSET_FILES) and set(rows.values()) == {"없음"}
    row = list(ASSET_FILES).index("sam2_hiera_tiny")
    group.table.selectRow(row)
    downloaded = _zip_weights(tmp_path / "sam2.1_hiera_tiny.pt")
    with patch.object(QFileDialog, "getOpenFileName", return_value=(str(downloaded), "")), \
            patch.object(QMessageBox, "information"):
        group._import_selected()
    assert group.table.item(row, 2).text() == "사용 가능"
    group.deleteLater()
