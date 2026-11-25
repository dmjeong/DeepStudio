"""Inference-only saves do not clobber or interrupt another window's training."""
from pathlib import Path
import pytest
from core.project import ProjectManager, RunRecord


def test_unchanged_second_window_does_not_rewrite_snapshot(tmp_path):
    project = ProjectManager.create_new('inspection', 'segment', str(tmp_path / 'project'), ['crack'])
    path = Path(ProjectManager.save(project))
    reader = ProjectManager.load(str(path))
    before, timestamp = path.read_bytes(), path.stat().st_mtime_ns
    ProjectManager.save(reader)
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == timestamp
    assert reader.modified == project.modified


def test_stale_second_window_cannot_delete_new_training_history(tmp_path):
    project = ProjectManager.create_new('inspection', 'segment', str(tmp_path / 'project'), ['crack'])
    path = ProjectManager.save(project)
    reader = ProjectManager.load(path)
    project.runs.append(RunRecord(run_id='new-run', status='completed'))
    ProjectManager.save(project)
    before = Path(path).read_bytes()
    with pytest.raises(RuntimeError, match='다른 앱'):
        ProjectManager.save(reader)
    assert Path(path).read_bytes() == before
    fresh = ProjectManager.load(path)
    assert fresh.runs[0].run_id == 'new-run'
    fresh.training.epochs += 1
    ProjectManager.save(fresh)


def test_external_edit_without_modified_timestamp_is_detected(tmp_path):
    project = ProjectManager.create_new('inspection', 'segment', str(tmp_path / 'project'), ['crack'])
    path = Path(ProjectManager.save(project))
    path.write_text(path.read_text().replace('inspection', 'external'), encoding='utf-8')
    with pytest.raises(RuntimeError, match='다른 앱'):
        ProjectManager.save(project)
    assert ProjectManager.load(str(path)).name == 'external'
