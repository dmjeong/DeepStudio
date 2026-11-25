"""main 병합 후 리뷰에서 찾은 버그의 회귀 테스트.

    ① 학습 폴더에 마스크가 하나도 없으면 학습을 막는다 (전부 배경 정답 → 결함 0개 모델이 Best가 되던 문제)
    ② 같은 경로의 손상된/프로젝트가 아닌 JSON 위에도 프로젝트를 저장할 수 있다
    ③ Custom CSP 탐지: 점수 == 기준이면 앱과 C++ 모두 남긴다 (C++만 버리던 문제)
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]


def _images(folder, count=2, size=(40, 30)):
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", size, (100 + index, 100, 100)).save(folder / f"{index}.png")
    return folder


# ── ① 마스크가 하나도 없는 학습 폴더 ──────────────────────────────
def test_training_split_without_any_mask_is_refused(tmp_path):
    pytest.importorskip("torch")
    from dataset import SegmentationDataset
    images, masks = _images(tmp_path / "images" / "train"), tmp_path / "masks" / "train"
    masks.mkdir(parents=True)
    with pytest.raises(ValueError, match="정답 마스크가 하나도 없습니다"):
        SegmentationDataset(str(images), str(masks), input_size=(32, 32), is_train=True, num_classes=2)


def test_validation_split_of_normal_images_without_masks_is_allowed(tmp_path):
    """검증 폴더는 정상 이미지만(마스크 없음 = 자동 배경) 있어도 된다 (오검출 확인용)."""
    pytest.importorskip("torch")
    from dataset import SegmentationDataset
    images, masks = _images(tmp_path / "images" / "val"), tmp_path / "masks" / "val"
    masks.mkdir(parents=True)
    dataset = SegmentationDataset(str(images), str(masks), input_size=(32, 32), is_train=False, num_classes=2)
    assert len(dataset) == 2 and len(dataset.unlabeled) == 2


def test_training_split_with_one_mask_keeps_unlabeled_images_as_background(tmp_path):
    pytest.importorskip("torch")
    from dataset import SegmentationDataset
    images, masks = _images(tmp_path / "images" / "train", 3), tmp_path / "masks" / "train"
    masks.mkdir(parents=True)
    mask = np.zeros((30, 40), np.uint8)
    mask[5:10, 5:10] = 1
    Image.fromarray(mask).save(masks / "0.png")
    dataset = SegmentationDataset(str(images), str(masks), input_size=(32, 32), is_train=True, num_classes=2)
    assert len(dataset) == 3 and len(dataset.unlabeled) == 2


# ── ② 손상된 파일 위에 저장 ─────────────────────────────────────
@pytest.mark.parametrize("existing", [b"{not json", b"[1, 2, 3]", b"\xff\xfe\x00",
                                      json.dumps({"name": "x"}).encode()])
def test_project_save_over_a_broken_or_foreign_json_file(tmp_path, existing):
    from core.project import ProjectManager
    project = ProjectManager.create_new("save", "segment", str(tmp_path / "p"), ["background", "crack"])
    target = tmp_path / "elsewhere.json"
    target.write_bytes(existing)
    saved = ProjectManager.save(project, str(target))
    data = json.loads(Path(saved).read_text(encoding="utf-8"))
    assert data["name"] == "save" and data["modified"] == project.modified


def test_project_save_of_unchanged_content_keeps_the_modified_time(tmp_path):
    from core.project import ProjectManager
    project = ProjectManager.create_new("same", "segment", str(tmp_path / "p"), ["background", "crack"])
    path = ProjectManager.save(project)
    before = Path(path).read_bytes()
    first = project.modified
    ProjectManager.save(project, path)
    assert Path(path).read_bytes() == before and project.modified == first   # 내용이 같으면 다시 쓰지 않는다


# ── ③ 탐지 점수 == 기준 ─────────────────────────────────────────
def _custom_detector(tmp_path, threshold):
    """출력이 고정된 Custom CSP 탐지 ONNX: 객체성·클래스 로짓 0 → 점수 0.5 × 0.5 = 정확히 0.25.

    JSON은 C++ 규약 테스트 생성기(cpp/tests/make_models.py)의 탐지 설정과 같은 형식이다.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    values = helper.make_tensor("detections", TensorProto.FLOAT, [1, 1, 6], [0.5, 0.5, 0.4, 0.4, 0.0, 0.0])
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["detections"], value=values)], "fixture",
        [helper.make_tensor_value_info("input_image", TensorProto.FLOAT, [1, 1, 2, 3])],
        [helper.make_tensor_value_info("detections", TensorProto.FLOAT, [1, 1, 6])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.save(model, tmp_path / "detect.onnx")
    config = tmp_path / "detect.json"
    config.write_text(json.dumps({
        "schema_version": 5, "backend": "custom", "model_path": "detect.onnx", "task": "detect",
        "num_classes": 1, "input_channels": 1, "input_height": 2, "input_width": 3,
        "input_name": "input_image", "output_name": "detections",
        "normalize_mean": [0.449], "normalize_std": [0.226], "class_names": ["crack"],
        "preprocessing": {"resize_implementation": "opencv_linear_exact_v1", "resize": "bilinear",
                          "interpolation": "INTER_LINEAR_EXACT", "antialias": False,
                          "layout": "NCHW", "value_scale": 255., "color_order": "GRAY"},
        "postprocessing": {"box_format": "normalized_cxcywh", "objectness": "sigmoid",
                           "class_scores": "sigmoid", "confidence_threshold": threshold,
                           "iou_threshold": 0.5, "max_detections": 10},
        "cpp_supported": True}), encoding="utf-8")
    return config


def test_python_custom_detection_keeps_a_score_equal_to_the_threshold():
    from detection import postprocess_detections
    rows = np.array([[0.5, 0.5, 0.4, 0.4, 0.0, 0.0]])
    assert len(postprocess_detections(rows, confidence=0.25)) == 1


@pytest.mark.skipif(not os.environ.get("DETECT_PROBE"), reason="DETECT_PROBE(C++ detect_probe 경로)가 없습니다")
def test_cpp_custom_detection_keeps_a_score_equal_to_the_threshold_like_the_app(tmp_path):
    config = _custom_detector(tmp_path, 0.25)
    image = tmp_path / "in.png"
    Image.new("RGB", (32, 32), "gray").save(image)
    result = subprocess.run([os.environ["DETECT_PROBE"], str(config), str(image)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout.strip().splitlines()[-1])["detections"]
    assert len(rows) == 1 and rows[0]["confidence"] == pytest.approx(0.25)   # 이전: C++만 버렸다
    # 기준을 조금 올리면 앱과 같이 버린다
    doc = json.loads(config.read_text(encoding="utf-8"))
    doc["postprocessing"]["confidence_threshold"] = 0.2501
    config.write_text(json.dumps(doc), encoding="utf-8")
    result = subprocess.run([os.environ["DETECT_PROBE"], str(config), str(image)],
                            capture_output=True, text=True, timeout=60, check=True)
    assert json.loads(result.stdout.strip().splitlines()[-1])["detections"] == []


# ── ④ 저장 파일 그대로 복원할 때 상대 project_dir ───────────────────
def test_restore_project_resolves_a_relative_project_dir_next_to_the_file(tmp_path, monkeypatch):
    """저장 파일의 project_dir은 "."이다. 복원 후 학습 기록이 앱 실행 폴더(현재 폴더)에 생기면 안 된다."""
    from core.project import ProjectManager
    from webapp.storage import restore_project
    project = ProjectManager.create_new("rel", "segment", str(tmp_path / "p"), ["background", "crack"])
    path = ProjectManager.save(project)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    assert raw["project_dir"] == "."
    monkeypatch.chdir(tmp_path)                       # 현재 폴더 ≠ 프로젝트 폴더
    restored = restore_project({**raw, "filepath": path})
    assert Path(restored.project_dir) == Path(path).parent
    snapshot = restore_project({**raw, "project_dir": str(tmp_path / "abs"), "filepath": path})
    assert snapshot.project_dir == str(tmp_path / "abs")   # 작업 스냅샷의 절대 경로는 그대로


# ── ⑤ LibreYOLO 검증 기록이 앱 실행 폴더에 생기지 않음 ─────────────────
def test_libreyolo_training_runs_inside_the_output_folder_and_restores_cwd(tmp_path, monkeypatch):
    """LibreYOLO 검증은 현재 폴더에 runs/val을 만든다 → 학습 중에만 출력 상위 폴더로 옮겼다가 되돌린다."""
    from types import SimpleNamespace
    from libreyolo_workdir import inside_run_directory
    app_dir = tmp_path / "install"
    app_dir.mkdir()
    monkeypatch.chdir(app_dir)
    trainer = SimpleNamespace(config=SimpleNamespace(project="out/runs", name="exp"))
    with inside_run_directory(trainer) as project:
        assert Path.cwd() == app_dir / "out" / "runs" == project
        assert trainer.config.project == str(project)          # 상대 경로는 바꾸기 전에 절대 경로로
        Path("runs/val/x").mkdir(parents=True)                 # LibreYOLO 검증기가 하는 일
    assert Path.cwd() == app_dir and not (app_dir / "runs").exists()
    assert (project / "runs" / "val" / "x").is_dir()
    assert not (project / "exp").exists()                      # 실행 폴더는 미리 만들지 않는다 (name2 방지)
    with pytest.raises(RuntimeError):
        with inside_run_directory(trainer):
            raise RuntimeError("학습 실패")
    assert Path.cwd() == app_dir                               # 실패해도 되돌린다
