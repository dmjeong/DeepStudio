"""Current engine contracts and renamed public entry points remain usable."""
from dataclasses import asdict
from pathlib import Path

import pytest

from core.project import ProjectManager, TrainingConfig
from core.training_modes import MODE_LABELS, training_capabilities, training_engine_name


@pytest.mark.parametrize("task,mode,engine", [
    ("classify", "efficientnet_finetune", "efficientnet"),
    ("classify", "custom", "custom"),
    ("segment", "custom", "custom"),
    ("detect", "custom", "custom"),
    ("anomaly", "custom", "custom"),
])
def test_supported_task_engine_contract(task, mode, engine):
    assert training_capabilities(task, mode)["engine"] == engine
    assert set(MODE_LABELS) == {"custom", "efficientnet_finetune", "efficientnet_transfer", "efficientnet_resume",
                                "efficientnet_scratch", "builtin_finetune", "builtin_transfer", "builtin_scratch",
                                "upstream_finetune", "upstream_transfer", "upstream_resume", "upstream_scratch"}
    assert {key for key in asdict(TrainingConfig()) if key.endswith("_model")} == {"efficientnet_model"}


def test_unknown_modes_do_not_silently_select_a_different_model():
    for mode in ("", "unknown", "unknown_finetune", "unknown_resume"):
        with pytest.raises(ValueError):
            training_engine_name(mode)
        with pytest.raises(ValueError):
            training_capabilities("classify", mode)
    with pytest.raises(ValueError):
        training_capabilities("obb", "custom")


def test_project_format_uses_current_extension_and_can_read_selected_json(tmp_path):
    project = ProjectManager.create_new("inspection", "classify", str(tmp_path / "project"), ["OK", "NG"])
    saved = Path(ProjectManager.save(project))
    assert saved.suffix == ".dvproj"
    selected = saved.with_suffix(".json")
    selected.write_bytes(saved.read_bytes())
    restored = ProjectManager.load(str(selected))
    assert restored.training == project.training
    assert restored.data.class_names == ["OK", "NG"]


def test_new_detection_project_uses_the_shipped_rtdetrv4_native_contract(tmp_path):
    project = ProjectManager.create_new("inspection", "detect", str(tmp_path / "project"), ["NG"])
    assert project.model.model_id == "re_detr_v4_small"
    assert project.training.training_mode == "upstream_finetune"
    assert project.training.input_size == 640


def test_public_cpp_and_packaging_names_agree():
    root = Path(__file__).resolve().parents[1]
    header = (root / "cpp/include/vision_inference.h").read_text(encoding="utf-8")
    assert "class VisionInference" in header
    assert '#include "vision_inference.h"' in (root / "cpp/src/vision_inference.cpp").read_text(encoding="utf-8")
    assert "add_library(vision_inference" in (root / "cpp/CMakeLists.txt").read_text(encoding="utf-8")
    assert '"DeepVisionStudio/"' in (root / "tools/package_web.py").read_text(encoding="utf-8")
