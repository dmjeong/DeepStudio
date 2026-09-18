"""
프로젝트 관리 단위 테스트

테스트 항목:
┌─────────────────────────────────────────────────────────┐
│  1. ProjectData 직렬화/역직렬화 (JSON round-trip)        │
│  2. ProjectManager.create_new — 태스크별 폴더 구조 생성  │
│  3. ProjectManager.save / load — 파일 저장/로드          │
│  4. DataConfig / TrainingConfig 기본값 검증               │
│  5. 태스크별 기본 input_size, batch_size 검증             │
└─────────────────────────────────────────────────────────┘
"""

import os
import json
import pytest

from core.project import (
    ProjectData, ProjectManager, DataConfig,
    TrainingConfig, ModelConfig, SUPPORTED_TASKS,
)


class TestProjectCreation:
    """프로젝트 생성 테스트"""

    @pytest.mark.parametrize("task", ["classify", "segment", "detect", "anomaly"])
    def test_create_new_all_tasks(self, tmp_path, task):
        """4가지 태스크 모두 프로젝트 생성 가능"""
        class_names = ["A", "B"] if task != "anomaly" else []
        project = ProjectManager.create_new(
            name=f"test_{task}",
            task=task,
            project_dir=str(tmp_path / f"proj_{task}"),
            class_names=class_names,
        )
        assert project.task == task
        assert project.name == f"test_{task}"
        assert project.data.root != ""

    def test_classify_folders_created(self, tmp_path):
        """Classification: train/val/test/{클래스명} 폴더 자동 생성"""
        project = ProjectManager.create_new(
            name="cls_test", task="classify",
            project_dir=str(tmp_path / "cls"),
            class_names=["OK", "NG"],
        )
        for split in ["train", "val", "test"]:
            for cls in ["OK", "NG"]:
                path = os.path.join(project.data.root, split, cls)
                assert os.path.isdir(path), f"{path} 미생성"

    def test_anomaly_folders_created(self, tmp_path):
        """Anomaly: train/good + test/{good,defect} 폴더 자동 생성"""
        project = ProjectManager.create_new(
            name="ano_test", task="anomaly",
            project_dir=str(tmp_path / "ano"),
        )
        assert os.path.isdir(
            os.path.join(project.data.root, "train", "good")
        )
        assert os.path.isdir(
            os.path.join(project.data.root, "test", "good")
        )
        assert os.path.isdir(
            os.path.join(project.data.root, "test", "defect")
        )

    def test_segment_folders_created(self, tmp_path):
        """Segmentation: images + masks 폴더 자동 생성"""
        project = ProjectManager.create_new(
            name="seg_test", task="segment",
            project_dir=str(tmp_path / "seg"),
            class_names=["bg", "part"],
        )
        for sub in ["images", "masks"]:
            for split in ["train", "val", "test"]:
                path = os.path.join(project.data.root, sub, split)
                assert os.path.isdir(path), f"{path} 미생성"

    def test_detect_folders_created(self, tmp_path):
        """Detection: images + labels 폴더 자동 생성"""
        project = ProjectManager.create_new(
            name="det_test", task="detect",
            project_dir=str(tmp_path / "det"),
            class_names=["car", "person"],
        )
        for sub in ["images", "labels"]:
            for split in ["train", "val", "test"]:
                path = os.path.join(project.data.root, sub, split)
                assert os.path.isdir(path), f"{path} 미생성"

    def test_invalid_task_raises(self, tmp_path):
        """미지원 태스크 → ValueError"""
        with pytest.raises(ValueError):
            ProjectManager.create_new(
                name="bad", task="unknown",
                project_dir=str(tmp_path / "bad"),
            )


class TestProjectSaveLoad:
    """프로젝트 저장/로드 round-trip 테스트"""

    def test_save_creates_file(self, tmp_project):
        """save() → .dvproj 파일 생성"""
        filepath = ProjectManager.save(tmp_project)
        assert os.path.isfile(filepath)
        assert filepath.endswith(ProjectManager.FILE_EXTENSION)

    def test_load_restores_data(self, tmp_project):
        """save() → load() → 데이터 복원"""
        filepath = ProjectManager.save(tmp_project)
        loaded = ProjectManager.load(filepath)

        assert loaded.name == tmp_project.name
        assert loaded.task == tmp_project.task
        assert loaded.data.root == tmp_project.data.root
        assert loaded.data.class_names == tmp_project.data.class_names

    def test_roundtrip_preserves_training_config(self, tmp_project):
        """학습 설정이 save/load 후에도 보존"""
        tmp_project.training.epochs = 200
        tmp_project.training.learning_rate = 0.0005
        tmp_project.training.optimizer = "sgd"

        filepath = ProjectManager.save(tmp_project)
        loaded = ProjectManager.load(filepath)

        assert loaded.training.epochs == 200
        assert loaded.training.learning_rate == 0.0005
        assert loaded.training.optimizer == "sgd"

    def test_save_file_is_valid_json(self, tmp_project):
        """저장된 파일이 유효한 JSON"""
        filepath = ProjectManager.save(tmp_project)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "name" in data
        assert "task" in data


class TestDefaults:
    """기본값 검증"""

    @pytest.mark.parametrize("task,expected_size", [
        ("classify", 224),
        ("segment", 320),
        ("detect", 416),
        ("anomaly", 224),
    ])
    def test_default_input_size(self, task, expected_size):
        """태스크별 기본 입력 크기"""
        assert SUPPORTED_TASKS[task]["default_input_size"] == expected_size

    def test_data_config_defaults(self):
        """DataConfig 기본값 확인"""
        dc = DataConfig()
        assert dc.root == ""
        assert dc.class_names == []
        assert dc.num_classes == 0

    def test_training_config_defaults(self):
        """TrainingConfig 기본값 확인"""
        tc = TrainingConfig()
        assert tc.epochs == 100
        assert tc.optimizer == "adamw"
        assert tc.scheduler == "cosine"
        assert tc.use_amp is True
