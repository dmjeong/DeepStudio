"""클래스별 판정 기준과 '오검출 0 기준 찾기': 정상 이미지의 덩어리가 모두 걸러지고 라벨 결함은 최대한 남는다."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))

from core.segment_defects import (class_rule, judge, load_model_rules, model_rules_path,  # noqa: E402
                                  per_image_alarm_bound, rules_of, save_model_rules, zero_alarm_rules)


def _blob(i, name, area, score, truth=()):
    return {"id": i, "class_id": 1 if name == "crack" else 2, "class_name": name, "x": 0, "y": 0, "w": 1, "h": 1,
            "area": area, "score": score, "truth": list(truth)}


def test_per_class_rule_overrides_the_default_only_for_that_class():
    rules = rules_of({}, {"min_area": 4, "min_score": 0.5, "per_class": {"crack": {"min_area": 10, "min_score": 0.85}}})
    assert class_rule(rules, "crack") == (10, 0.85) and class_rule(rules, "fm") == (4, 0.5)
    details = {"defects": [_blob(0, "crack", 30, 0.80), _blob(1, "fm", 5, 0.6)]}
    verdict = judge(details, rules)
    assert [item["class_name"] for item in verdict["kept"]] == ["fm"]  # crack 0.80 < 0.85
    # 추론 때 저장한 클래스별 기준이 그대로 다시 판정에 쓰인다
    assert judge({**details, "defect_rules": rules})["kept"] == verdict["kept"]


def _evaluation():
    """정상 3장(오검출 덩어리) + 라벨 결함 이미지 1장 (crack 3개: 강함·강함·흐림)."""
    normal = [{"defects": [_blob(0, "crack", 28, 0.80), _blob(1, "crack", 8, 0.68)]},
              {"defects": [_blob(0, "fm", 6, 0.73)]},
              {"defects": []}]
    labeled = {"defects": [_blob(0, "crack", 120, 0.93, [0]), _blob(1, "crack", 80, 0.91, [1]),
                           _blob(2, "crack", 15, 0.70, [2]), _blob(3, "fm", 130, 0.97, [3]), _blob(4, "fm", 9, 0.75)],
               "ground_truth": {"defects": [{"class_name": "crack", "predicted": [0]},
                                            {"class_name": "crack", "predicted": [1]},
                                            {"class_name": "crack", "predicted": [2]},
                                            {"class_name": "fm", "predicted": [3]}],
                                "confusion": [[0, 0, 0], [0, 1, 0], [0, 0, 1]]}}
    return normal + [labeled]


def test_finder_removes_every_normal_blob_and_keeps_most_labeled_defects():
    result = zero_alarm_rules(_evaluation(), {"min_area": 1, "min_score": 0.0})
    rows = {row["class_name"]: row for row in result["classes"]}
    assert result["complete"] and result["normal_images"] == 3 and result["labeled_images"] == 1
    assert rows["crack"]["false_now"] == 2 and rows["crack"]["false_after"] == 0
    assert rows["crack"]["found_after"] == 2 and rows["crack"]["truth"] == 3  # 흐린 crack(0.70)만 포기
    assert rows["fm"]["false_after"] == 0 and rows["fm"]["found_after"] == 1  # 라벨 이미지의 정답 아닌 fm도 오검출
    rules = result["rules"]
    for details in _evaluation()[:3]:
        assert judge(details, rules)["decision"] == "OK"
    assert judge(_evaluation()[3], rules)["false_positives"] == 0


def test_finder_uses_area_when_score_alone_cannot_reject_a_false_blob():
    details = [{"defects": [_blob(0, "crack", 6, 0.995)]},
               {"defects": [_blob(0, "crack", 50, 0.95, [0])],
                "ground_truth": {"defects": [{"class_name": "crack", "predicted": [0]}]}}]
    row = zero_alarm_rules(details)["classes"][0]
    assert row["proposed"] == [7, 0.0] and row["found_after"] == 1  # 점수 0.995는 못 넘지만 면적 7px 이상으로 거른다


def test_without_labels_only_the_score_rises():
    row = zero_alarm_rules([{"defects": [_blob(0, "crack", 300, 0.6)]}])["classes"][0]
    assert row["proposed"] == [1, 0.62]  # 근거 없이 큰 면적 문턱으로 결함까지 버리지 않는다


def test_statistical_bound_shrinks_with_more_normal_images():
    assert per_image_alarm_bound(0) is None
    assert per_image_alarm_bound(100) == pytest.approx(0.0295, abs=1e-3)  # 약 3/n
    assert per_image_alarm_bound(300) < per_image_alarm_bound(100)


def test_rules_are_saved_next_to_the_model(tmp_path):
    checkpoint = tmp_path / "best.pt"
    assert load_model_rules(checkpoint) is None
    save_model_rules(checkpoint, {"min_area": 3, "min_score": 0.2, "per_class": {"crack": {"min_score": 0.85}}})
    assert model_rules_path(checkpoint).name == "best.rules.json"
    assert load_model_rules(checkpoint) == {"min_area": 3, "min_score": 0.2,
                                            "per_class": {"crack": {"min_area": 3, "min_score": 0.85}}}


def _segment_result(path, defects, truth=None):
    from core.inference_types import InferenceResult
    details = {"class_names": ["background", "crack", "fm"], "defects": defects,
               "defect_rules": {"min_area": 1, "min_score": 0.0}}
    if truth:
        details["ground_truth"] = truth
    return InferenceResult(path, "ok", "segment", "", details=details)


@pytest.fixture
def review(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    from widgets.inference_review import InferenceReview
    widget = InferenceReview()
    yield widget
    widget.deleteLater()


def test_finder_dialog_applies_and_saves_rules_next_to_the_model(review, tmp_path, monkeypatch):
    from widgets.judgement_rules_dialog import ZeroAlarmDialog
    evaluation = _evaluation()
    paths = [f"{i}.png" for i in range(len(evaluation))]
    review.set_context(paths)
    for path, details in zip(paths, evaluation):
        truth = details.get("ground_truth")
        review.update_result(_segment_result(path, details["defects"], truth))
    review.set_model_path(str(tmp_path / "best.pt"))
    assert review.save_rules_btn.isEnabled()  # 저장 안 된 기준
    assert "모델에 저장 안 됨" in review.rules_state.text()
    monkeypatch.setattr(ZeroAlarmDialog, "exec", lambda self: ZeroAlarmDialog.SAVE)
    review._find_zero_alarm_rules()
    assert review.per_class["crack"]["min_score"] >= 0.82
    assert (tmp_path / "best.rules.json").is_file() and "모델에 저장됨" in review.rules_state.text()
    # 다시 불러오면 저장된 기준으로 시작한다 (내보내기·C++과 같은 기준)
    review.set_rules({"min_area": 1, "min_score": 0.0})
    review.set_model_path(str(tmp_path / "best.pt"))
    assert review.per_class["crack"]["min_score"] >= 0.82
    assert review.segment_rules["per_class"] == review.per_class


def test_finder_without_results_explains_what_to_run(review, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: shown.append(args[2]))
    review._find_zero_alarm_rules()
    assert shown and "정상 이미지 폴더" in shown[0]


def test_dialog_lists_each_class_with_now_and_proposed(review):
    from widgets.judgement_rules_dialog import ZeroAlarmDialog, ClassRulesDialog
    dialog = ZeroAlarmDialog(_evaluation(), {"min_area": 1, "min_score": 0.0})
    texts = {dialog.table.item(r, 0).text(): dialog.table.item(r, 3).text() for r in range(dialog.table.rowCount())}
    assert texts == {"crack": "2 → 0", "fm": "2 → 0"}
    assert "정상 이미지가 적습니다" in dialog.summary.text()
    dialog.deleteLater()
    edit = ClassRulesDialog(["background", "crack", "fm"], {"min_area": 3, "min_score": 0.1,
                                                            "per_class": {"crack": {"min_area": 9, "min_score": 0.9}}})
    assert edit.table.rowCount() == 2  # 배경은 기준 대상이 아니다
    assert edit.result_rules()["per_class"] == {"crack": {"min_area": 9, "min_score": 0.9}}
    edit.deleteLater()
