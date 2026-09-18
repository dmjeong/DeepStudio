"""두 UI의 공통 저장 계약, 잘못된 작업 기록과 학습 모드 검증."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.project import ProjectManager
from webapp.jobs import JobManager
from webapp.storage import project_view, restore_project, write_json, read_json


class StudioReliabilityTests(unittest.TestCase):
    def test_malformed_jobs_do_not_block_startup_or_healthy_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            broken = root / "jobs" / ("a" * 32) / "job.json"
            broken.parent.mkdir(parents=True)
            broken.write_text("{broken")
            good = root / "jobs" / ("b" * 32) / "job.json"
            write_json(good, {"id": "b" * 32, "kind": "infer", "status": "running", "created_at": "2026-09-08"})
            (good.parent / "result.json").write_text("[]")
            manager = JobManager(root)
            self.assertEqual(manager.list()[0]["status"], "interrupted")
            self.assertTrue(manager.warnings)
            self.assertEqual(broken.read_text(), "{broken")
            self.assertEqual(len(manager.storage()["jobs"]), 2)
            self.assertTrue(manager.export_archive("a" * 32).is_file())
            manager.delete(["a" * 32])
            self.assertFalse(broken.exists())

    def test_active_record_cannot_be_deleted(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = JobManager(folder)
            manager.active_id = "a" * 32
            with self.assertRaises(RuntimeError):
                manager.delete([manager.active_id])

    def test_desktop_web_desktop_round_trip_retains_resources_and_training_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            p = ProjectManager.create_new("한글 프로젝트", "detect", str(Path(folder) / "project"), ["good", "defect"])
            p.training.training_mode = "efficientnet_transfer"
            p.training.class_weights = "sqrt"
            p.model.freeze_backbone = True
            ProjectManager.save(p)
            original = project_view(p)
            restored = restore_project(original)
            ProjectManager.save(restored)
            loaded = ProjectManager.load(ProjectManager.get_active_filepath(restored))
            self.assertEqual(project_view(loaded)["data"], original["data"])
            self.assertEqual(loaded.training.training_mode, "efficientnet_transfer")
            self.assertTrue(loaded.model.freeze_backbone)


    def test_corrupt_completion_does_not_leave_compute_slot_busy(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = JobManager(folder)
            job_id = "a" * 32
            record = {"id": job_id, "kind": "infer", "status": "running", "created_at": "2026-09-08"}
            manager._jobs[job_id] = record
            directory = manager.root / job_id
            write_json(directory / "job.json", record)
            (directory / "job.json").write_text("{broken")
            write_json(directory / "result.json", [])
            manager.active_id = job_id
            with patch("subprocess.Popen") as process:
                process.wait.return_value = 9
                manager._wait(job_id, process)
            self.assertIsNone(manager.active_id)
            self.assertEqual(read_json(directory / "job.json")["status"], "failed")
