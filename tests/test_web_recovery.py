"""계산 라이브러리 없이 프로젝트 복구와 작업 결과 보존을 검증한다."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import webapp  # noqa: F401,E402
from core.project import ProjectManager
from webapp.jobs import JobManager
from webapp.storage import ProjectStore, read_json, write_json


class WebRecoveryTests(unittest.TestCase):
    def test_unavailable_project_can_be_replaced_without_stale_writes(self):
        for damage in ("missing", "invalid"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                store = ProjectStore(root / "state")
                project = ProjectManager.create_new("original", "classify", str(root / "original"), ["OK"])
                ProjectManager.save(project)
                store.select(project)
                path = Path(ProjectManager.get_active_filepath(project))
                if damage == "missing":
                    path.unlink()
                else:
                    path.write_text("{broken", encoding="utf-8")
                with self.assertRaises((OSError, ValueError)):
                    store.reload()
                self.assertIsNone(store.project)
                self.assertTrue(store.error)
                self.assertIn(str(path), store.recent)
                replacement = ProjectManager.create_new("replacement", "classify", str(root / "replacement"), ["OK"])
                ProjectManager.save(replacement)
                store.select(replacement)
                store.reload()
                self.assertEqual(store.project.name, "replacement")
                self.assertEqual(store.error, "")
                if damage == "missing":
                    self.assertFalse(path.exists())
                else:
                    self.assertEqual(path.read_text(encoding="utf-8"), "{broken")

    def test_project_refresh_failure_preserves_actual_job_result(self):
        for status, error in (("completed", ""), ("failed", "model execution failed"), ("cancelled", "")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                manager = JobManager(folder, on_finish=Mock(side_effect=FileNotFoundError("project moved")))
                job_id = "a" * 32
                directory = manager.root / job_id
                write_json(directory / "job.json", {"id": job_id, "kind": "infer", "status": "running", "created_at": "2026-09-08"})
                result = {"status": status, "error": error, "output": {"count": 3}, "duration_sec": 1.5}
                write_json(directory / "result.json", result)
                manager.active_id = job_id
                process = Mock()
                process.wait.return_value = 0
                manager.process = process
                manager._wait(job_id, process)
                saved = read_json(directory / "job.json")
                for key, value in result.items():
                    self.assertEqual(saved[key], value)
                self.assertIn("project moved", saved["project_error"])
                self.assertIsNone(manager.active_id)
                self.assertIsNone(manager.process)
                self.assertEqual(manager.read(job_id)["job"], saved)


if __name__ == "__main__":
    unittest.main()
