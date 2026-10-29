"""데이터에 맞춘 분할 학습 설정 추천 (가는 결함이 학습 해상도에서 사라지지 않게)."""

from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "gui"), str(Path(__file__).resolve().parents[1] / "python")]

from core.training_advisor import advise, thin_risk  # noqa: E402


def _project(tmp_path, *, crop=(3500, 1504), input_size=512, patch=False, scale=1.0):
    from core.project import ProjectManager
    project = ProjectManager.create_new("adv", "segment", str(tmp_path / "p"), ["background", "crack", "dot"])
    training = project.training
    training.input_size, training.patch_training, training.patch_scale = input_size, patch, scale
    training.patchcore_crop_enabled = crop is not None
    if crop:
        training.patchcore_crop_width, training.patchcore_crop_height = crop
    return project


def _index(widths, size=(3840, 2160), cls="crack"):
    """이미지 4장, 영역 굵기 목록을 나눠 담은 스캔 결과."""
    images = [{"path": f"/data/{i}.png", "split": "train", "name": f"{i}.png", "size": list(size),
               "region_items": []} for i in range(4)]
    for number, width in enumerate(widths):
        images[number % 4]["region_items"].append({"class": cls, "width": width, "area": width * 200})
    return {"images": images}


def test_micro_cracks_keep_original_resolution(tmp_path):
    """폭 1~2px 미세 크랙: 512로 줄이면 사라지고, 배율을 낮춰도 사라지므로 패치 100%."""
    advice = advise(_project(tmp_path), _index([1, 1, 2, 1, 2, 3, 1, 2]), weights_ready=True)
    assert advice["ok"]
    settings = advice["settings"]
    assert settings["patch_training"] is True and settings["patch_scale"] == 1.0 and settings["input_size"] == 512
    assert settings["model_id"] == "unet_resnet18" and settings["training_mode"] == "builtin_finetune"
    assert settings["class_weights"] == "sqrt" and settings["segmentation_loss"] == "ce_dice"
    assert advice["stats"]["area"] == [3500, 1504] and advice["stats"]["thin_px"] == pytest.approx(1)
    assert any("라벨 붓을 2~3px" in note for note in advice["notes"])
    assert "GPU" in advice["reasons"]["patch_scale"]


def test_thicker_defects_get_the_fastest_scale_that_keeps_them(tmp_path):
    """폭 8px 이상: 25%에서도 2px 남으므로 가장 빠른 25%, 패치는 줄인 영역(875×376)에 맞춘 352."""
    advice = advise(_project(tmp_path), _index([8, 9, 10, 12, 8, 9]), weights_ready=True)
    settings = advice["settings"]
    assert settings["patch_training"] and settings["patch_scale"] == .25 and settings["input_size"] == 352
    assert advice["stats"]["compute"]["size"] == [875, 376]
    # 폭 3px이면 75% (50%면 1.5px로 2px 아래)
    assert advise(_project(tmp_path / "b"), _index([3, 3, 4, 5]), weights_ready=True)["settings"]["patch_scale"] == .75


def test_large_defects_do_not_need_patch_training(tmp_path):
    """폭 40px: 512로 줄여도 2px 이상 남으면 전체 이미지 학습(가장 빠름)을 그대로 둔다."""
    advice = advise(_project(tmp_path), _index([40, 50, 60, 45]), weights_ready=True)
    assert advice["settings"]["patch_training"] is False and advice["settings"]["input_size"] == 512
    # 폭 10px: 가로·세로 평균 배율로는 2.2px 남아 보여도, 세로로 뻗은 크랙은 가로 6.8배 축소를 받아 1.5px → 패치 25%
    borderline = advise(_project(tmp_path / "b"), _index([10, 10, 11, 12]), weights_ready=True)["settings"]
    assert borderline["patch_training"] is True and borderline["patch_scale"] == .25


def test_start_check_counts_defects_lost_at_the_current_resolution(tmp_path):
    index = _index([1, 1, 2, 1, 6, 8])
    # 배율 25% 패치: 1~2px 크랙은 0.25~0.5px, 6·8px는 1.5·2px → 4개 손실
    assert thin_risk(_project(tmp_path, patch=True, scale=.25), index)["lost"] == 4
    assert thin_risk(_project(tmp_path / "b", patch=True, scale=1.0), index)["lost"] == 0
    # 전체 축소 512(가로 6.8배, 세로 2.9배 축소): 가는 결함 대부분 손실
    assert thin_risk(_project(tmp_path / "c"), index)["lost"] >= 4


def test_no_labels_or_other_tasks_explain_why(tmp_path):
    assert advise(_project(tmp_path), {"images": []})["ok"] is False
    from core.project import ProjectManager
    classify = ProjectManager.create_new("c", "classify", str(tmp_path / "c"), ["ok", "ng"])
    assert "분할" in advise(classify, {"images": []})["message"]


def _crack_dataset(root):
    """3840×2160 이미지 2장 + 폭 1px 미세 크랙 마스크 (실제 스캔용)."""
    import numpy as np
    from PIL import Image
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        Image.new("L", (3840, 2160), 120).save(root / "images" / split / "a.png")
        mask = np.zeros((2160, 3840), np.uint8)
        mask[1000, 500:900] = 1  # 1px 크랙
        Image.fromarray(mask, "L").save(root / "masks" / split / "a.png")


@pytest.fixture
def training_page(tmp_path):
    pytest.importorskip("PySide6")
    pytest.importorskip("cv2")
    from PySide6.QtWidgets import QApplication
    from widgets.training_widget import TrainingWidget
    app = QApplication.instance() or QApplication([])
    project = _project(tmp_path, patch=True, scale=.25)
    _crack_dataset(tmp_path / "data")
    project.data.root = str(tmp_path / "data")
    project.data.train_dir = str(tmp_path / "data" / "images" / "train")
    project.data.val_dir = str(tmp_path / "data" / "images" / "val")
    project.model.model_id = "unet_resnet18"
    project.training.training_mode = "builtin_finetune"
    widget = TrainingWidget()
    widget.set_project(project)
    widget.show()
    yield app, widget, project
    widget.close()
    widget.deleteLater()


def test_auto_settings_button_scans_and_applies_original_resolution(training_page):
    """배율 25%로 잘못 고른 미세 크랙 프로젝트: 버튼 → 스캔 → 추천(100%) 적용."""
    import time
    from unittest.mock import patch
    from widgets.training_advice_dialog import TrainingAdviceDialog
    app, widget, project = training_page
    assert widget.advise_btn.isVisibleTo(widget)
    shown = {}

    def accept(dialog):
        shown["rows"] = [dialog.table.item(r, 0).text() for r in range(dialog.table.rowCount())]
        shown["summary"] = dialog.summary.text()
        return 1

    with patch.object(TrainingAdviceDialog, "exec", accept):
        widget._open_advisor()
        deadline = time.time() + 60
        while getattr(widget, "_advice_scan", None) is not None and time.time() < deadline:
            app.processEvents()
            time.sleep(.02)
        app.processEvents()
    assert "패치 배율" in shown["rows"] and "1.0px" in shown["summary"] and "1px보다 가늘어집니다" in shown["summary"]
    assert widget.patch_check.isChecked() and widget.patch_scale_combo.currentData() == 1.0
    assert project.training.patch_scale == 1.0 and project.training.class_weights == "sqrt"
    assert "자동 설정 적용" in widget.status_label.text()


def test_start_check_warns_before_thin_defects_vanish(training_page):
    from unittest.mock import patch
    from PySide6.QtWidgets import QMessageBox
    from core.dataset_editor import scan_dataset
    _app, widget, project = training_page
    index = scan_dataset(project)
    widget._known_dataset_index = lambda: index
    asked = []

    def click(label):
        def clicked(box):
            asked.append(box.text())
            return next(button for button in box.buttons() if button.text() == label)
        return clicked

    with patch.object(QMessageBox, "exec", lambda box: 0), patch.object(QMessageBox, "clickedButton", click("취소")):
        assert widget._thin_defects_ok() is False
    assert "1px보다 가늘어집니다" in asked[0] and "패치 배율 25%" in asked[0]
    with patch.object(QMessageBox, "exec", lambda box: 0), patch.object(QMessageBox, "clickedButton", click("그대로 학습")):
        assert widget._thin_defects_ok() is True
    # 원본 해상도면 묻지 않는다
    project.training.patch_scale = 1.0
    with patch.object(QMessageBox, "exec", side_effect=AssertionError("should not ask")):
        assert widget._thin_defects_ok() is True


def test_inference_validation_button_selects_val_images_and_runs(tmp_path):
    pytest.importorskip("PySide6")
    from unittest.mock import patch
    from PySide6.QtWidgets import QApplication
    from widgets.inference_widget import InferenceWidget
    QApplication.instance() or QApplication([])
    project = _project(tmp_path, crop=None)
    from PIL import Image
    folder = tmp_path / "data" / "images" / "val" / "lot1"
    folder.mkdir(parents=True)
    for name in ("b.png", "a.png"):
        Image.new("L", (64, 48)).save(folder / name)
    project.data.root = str(tmp_path / "data")
    project.data.val_dir = str(tmp_path / "data" / "images" / "val")
    widget = InferenceWidget()
    widget.project = project
    with patch.object(widget, "_run_inference") as run:
        widget.infer_btn.setEnabled(False)
        widget._evaluate_validation()
        run.assert_not_called()  # 모델이 없으면 고르기만 한다
        assert [p.rsplit("/", 1)[-1] for p in widget._batch_images] == ["a.png", "b.png"]
        assert "2장" in widget.batch_time_label.text()
        widget.infer_btn.setEnabled(True)
        widget._evaluate_validation()
        run.assert_called_once()
    assert widget.result_review.model.rowCount() == 2
    widget.deleteLater()


def test_validation_images_use_test_folder_when_anomaly_val_is_the_train_alias(tmp_path):
    """이상 탐지 프로젝트는 val = train(정상 학습 폴더)이 기본값이라, 학습 이미지 대신 test를 평가한다."""
    pytest.importorskip("PySide6")
    from PIL import Image
    from PySide6.QtWidgets import QApplication
    from core.project import ProjectManager
    from widgets.inference_widget import InferenceWidget
    QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("a", "anomaly", str(tmp_path / "a"), ["good", "defect"])
    Image.new("L", (32, 32)).save(Path(project.data.train_dir) / "good" / "train.png")
    Image.new("L", (32, 32)).save(Path(project.data.test_dir) / "defect" / "test.png")
    widget = InferenceWidget()
    widget.project = project
    assert [Path(p).name for p in widget._validation_images()] == ["test.png"]
    widget.deleteLater()
