"""분할 추론 결과를 결함 단위로 판정하고 정답과 비교하는지 검증한다 (Qt 없이)."""

from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
pytest.importorskip("cv2")

from core.inference_review import review_record  # noqa: E402
from core.segment_defects import (analyze_segmentation, batch_truth_summary, judge,  # noqa: E402
                                  load_truth, segment_split_dirs, truth_mask_path)

NAMES = ["background", "scratch", "dent"]


def _scene():
    """scratch 18px(정답과 겹침) + scratch 1px 노이즈 + dent 24px(오검출) / 정답에만 있는 dent(놓침)."""
    labels = np.zeros((20, 30), int)
    labels[2:5, 2:8] = 1
    labels[10, 10] = 1
    labels[12:16, 20:26] = 2
    truth = np.zeros_like(labels)
    truth[3:6, 3:9] = 1
    truth[17:19, 1:3] = 2
    truth[0, 0] = 255  # 학습 제외 픽셀은 비교에서 뺀다
    probabilities = np.zeros((3, 10, 15), np.float32)
    probabilities[0], probabilities[1], probabilities[2] = .2, .7, .1
    return labels, truth, probabilities


def test_each_blob_becomes_a_defect_with_box_area_score_in_image_coordinates():
    labels, _, probabilities = _scene()
    analysis = analyze_segmentation(labels, probabilities, class_names=NAMES, offset=(100, 50))
    defects = analysis["defects"]
    # 큰 결함부터, 원본 좌표(크롭 오프셋 반영), 점수 = 결함 픽셀의 그 클래스 평균 확률
    assert [(d["class_name"], d["x"], d["y"], d["w"], d["h"], d["area"]) for d in defects] == [
        ("dent", 120, 62, 6, 4, 24), ("scratch", 102, 52, 6, 3, 18), ("scratch", 110, 60, 1, 1, 1)]
    assert [d["score"] for d in defects] == [pytest.approx(.1), pytest.approx(.7), pytest.approx(.7)]
    assert "ground_truth" not in analysis


def test_judgement_rules_filter_noise_and_decide_ok_ng_without_rerunning():
    labels, _, probabilities = _scene()
    details = {**analyze_segmentation(labels, probabilities, class_names=NAMES), "class_names": NAMES}
    verdict = judge(details)
    assert verdict["decision"] == "NG" and verdict["counts"] == {"dent": 1, "scratch": 2}
    assert judge(details, {"min_area": 2})["counts"] == {"dent": 1, "scratch": 1}     # 1px 노이즈 제거
    assert judge(details, {"min_score": .5})["counts"] == {"scratch": 2}              # 낮은 확신 dent 제거
    assert judge(details, {"min_area": 1000})["decision"] == "OK"
    assert judge({"defects": []})["decision"] == "OK"


def test_ground_truth_gives_class_iou_and_found_missed_false_positives():
    labels, truth, probabilities = _scene()
    details = {**analyze_segmentation(labels, probabilities, class_names=NAMES, truth=truth), "class_names": NAMES}
    verdict = judge(details)
    assert (verdict["truth_defects"], verdict["found"], verdict["missed"], verdict["false_positives"]) == (2, 1, 1, 2)
    scratch = next(row for row in verdict["per_class"] if row["class_name"] == "scratch")
    # 정답 18px, 예측 19px, 겹침 10px → IoU 10/27 (255 픽셀은 제외)
    assert scratch["iou"] == pytest.approx(10 / 27) and scratch["truth_pixels"] == 18
    # 1px 노이즈를 기준으로 빼면 오검출이 하나 줄고, 정답 scratch는 여전히 검출
    assert judge(details, {"min_area": 2})["false_positives"] == 1
    summary = batch_truth_summary([details, details], class_names=NAMES)
    assert summary["images"] == 2 and summary["found"] == 2 and summary["missed"] == 2
    assert summary["miou"] == pytest.approx((10 / 27 + 0) / 2)


def test_truth_mask_is_found_by_the_training_pairing_rule(tmp_path):
    from core.project import ProjectManager
    project = ProjectManager.create_new("p", "segment", str(tmp_path / "p"), NAMES)
    image = Path(project.data.root) / "images" / "val" / "sub" / "a.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (6, 4)).save(image)
    mask = Path(project.data.root) / "masks" / "val" / "sub" / "a.png"
    mask.parent.mkdir(parents=True)
    Image.fromarray(np.ones((4, 6), np.uint8), "L").save(mask)
    dirs = segment_split_dirs(project)
    assert truth_mask_path(str(image), dirs) == str(mask)
    assert truth_mask_path(str(tmp_path / "elsewhere.png"), dirs) is None
    assert load_truth(str(mask), (6, 4), box=(1, 1, 4, 3)).shape == (2, 3)
    assert load_truth(str(mask), (7, 4)) is None  # 크기가 다르면 비교하지 않는다


def test_review_record_judges_desktop_segmentation_and_leaves_web_records_alone():
    labels, _, probabilities = _scene()
    details = {**analyze_segmentation(labels, probabilities, class_names=NAMES), "class_names": NAMES,
               "defect_rules": {"min_area": 1, "min_score": 0}}
    record = {"image_path": "a.png", "status": "ok", "task": "segment", "summary": "NG 3", "details": details}
    row = review_record(record)
    assert (row["decision"], row["summary"], row["display_score"]) == ("NG", "NG 3", pytest.approx(.7))
    assert review_record(record, segment_rules={"min_area": 1000})["decision"] == "OK"
    # 결함 분석이 없는 기록(웹 추론)은 이전처럼 요약을 판정으로 쓴다
    web = {"image_path": "a.png", "status": "ok", "task": "segment", "summary": "세그멘테이션",
           "details": {"pixel_counts": {0: 10}}}
    assert review_record(web)["decision"] == "세그멘테이션"
