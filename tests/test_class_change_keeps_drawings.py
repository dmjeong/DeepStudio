"""데스크톱에서 클래스를 지우거나 0번에 배경을 넣어도 작업실 그리기 기록이 남는다.

    이전: 마스크 픽셀 번호만 바꾸고 기록을 버려, 이미지의 모든 라벨이 작업실에서 'PNG <번호>' 덩어리가 됐다.
    이후: 기록 속 클래스 번호도 같은 규칙으로 바꾼다. 다시 그린 결과가 픽셀과 같을 때만 기록을 남긴다.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from core.class_management import ClassManager
from core.mask_annotations import MaskDocument
from core.project import ProjectManager


def _rect(class_id, x0, y0, x1, y1, name):
    return {"kind": "polygon", "class_id": class_id, "object_id": name,
            "points": [x0 / 60, y0 / 40, x1 / 60, y0 / 40, x1 / 60, y1 / 40, x0 / 60, y1 / 40]}


def _project(tmp_path, names, shapes):
    project = ProjectManager.create_new("keep", "segment", str(tmp_path / "p"), ["background", "x"])
    project.data.class_names, project.data.num_classes = list(names), len(names)
    root = Path(project.data.root)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "masks" / "train").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (60, 40), "gray").save(root / "images" / "train" / "a.png")
    document = MaskDocument(60, 40, len(names), shapes=shapes)
    (root / "masks" / "train" / "a.png").write_bytes(document.encode())
    return project, root / "masks" / "train" / "a.png"


def _reopen(path, classes):
    with Image.open(path) as image:
        image.load()
        return MaskDocument.from_image(image, 60, 40, classes), np.array(image)


SHAPES = [_rect(1, 2, 2, 20, 18, "shift"), _rect(2, 25, 2, 40, 18, "double"), _rect(3, 45, 2, 58, 18, "fm"),
          _rect(2, 10, 10, 30, 30, "double over shift")]


def test_deleting_a_class_keeps_the_other_drawings_editable(tmp_path):
    project, mask = _project(tmp_path, ["background", "shift", "double", "fm"], SHAPES)
    before, _ = _reopen(mask, 4)
    expected = before.render()
    expected = np.where(expected == 2, 0, np.where(expected == 3, 2, expected))   # double → 배경, fm 3 → 2
    preview = ClassManager.preview_delete(project, "double")
    ClassManager.delete(project, "double", preview, keep_drawings=True)
    document, pixels = _reopen(mask, 3)
    assert np.array_equal(pixels, expected)                                        # 학습 정답은 픽셀만 바꿀 때와 같다
    assert [s["class_id"] for s in document.shapes] == [1, 0, 2, 0]               # 기록도 같은 번호로
    assert document.base_classes_left() == ()                                       # 'PNG' 덩어리 없음
    assert np.array_equal(document.render(), expected)


def test_without_keep_drawings_the_record_is_dropped_as_before(tmp_path):
    """웹 경로(기본값)는 이전 동작 그대로: 픽셀만 바꾸고 기록은 버린다."""
    project, mask = _project(tmp_path, ["background", "shift", "double", "fm"], SHAPES)
    ClassManager.delete(project, "double", ClassManager.preview_delete(project, "double"))
    document, _ = _reopen(mask, 3)
    assert not document.shapes and set(document.base_classes_left()) == {1, 2}


def test_inserting_background_keeps_drawings_and_shifts_their_classes(tmp_path):
    project, mask = _project(tmp_path, ["shift", "fm"], [_rect(1, 5, 5, 25, 25, "fm")])
    ClassManager.insert_background(project, keep_drawings=True)
    document, pixels = _reopen(mask, 3)
    assert [s["class_id"] for s in document.shapes] == [2] and int((pixels == 2).sum()) > 0
    assert document.base_classes_left() == ()
