"""로컬 웹 저장 계약, 프로세스 실행, 중단 및 계산 없는 캐시 렌더링 검증."""

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import webapp  # noqa: F401,E402
from webapp.server import create_app
from webapp.storage import read_json, write_json
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = create_app(self.root / "state")
        self.client = TestClient(self.app, base_url="http://127.0.0.1", headers={"X-Studio-Request": "1"})
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def project(self, name="검사 프로젝트", task="classify"):
        result = self.client.post("/api/projects", json={"name": name, "task": task,
                                                       "parent": str(self.root), "class_names": ["NG", "OK"]})
        self.assertEqual(result.status_code, 200, result.text)
        return result.json()

    def wait_job(self, job_id, timeout=150):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.client.get(f"/api/jobs/{job_id}").json()
            if response["job"]["status"] not in {"running", "cancelling"}:
                return response
            time.sleep(.1)
        self.fail(f"작업 시간 초과: {self.client.get(f'/api/jobs/{job_id}').text}")

    def test_project_round_trip_and_transactional_settings(self):
        project = self.project()
        result = self.client.put("/api/project", json={"training": {"epochs": 7, "class_weights": "sqrt",
                     "augmentation": {"horizontal_flip": .4}}, "model": {"freeze_backbone": True}})
        self.assertEqual(result.status_code, 200, result.text)
        saved = Path(project["filepath"]).read_bytes()
        invalid = self.client.put("/api/project", json={"training": {"epochs": 0}})
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(saved, Path(project["filepath"]).read_bytes())
        reopened = self.client.post("/api/projects/open", json={"path": project["filepath"]}).json()
        self.assertEqual(reopened["training"]["epochs"], 7)
        self.assertEqual(reopened["training"]["augmentation"]["horizontal_flip"], .4)
        self.assertTrue(reopened["model"]["freeze_backbone"])
        self.assertEqual(reopened["filepath"], project["filepath"])
        self.assertEqual(self.client.post("/api/projects", json={"name": "검사 프로젝트", "task": "classify",
                            "parent": str(self.root), "class_names": ["NG"]}).status_code, 409)

    def test_non_anomaly_project_requires_an_initial_class(self):
        for task in ("classify", "detect", "segment", "obb"):
            with self.subTest(task=task):
                response = self.client.post("/api/projects", json={
                    "name": f"empty-{task}", "task": task,
                    "parent": str(self.root), "class_names": [],
                })
                self.assertEqual(response.status_code, 400, response.text)
                self.assertFalse((self.root / f"empty-{task}").exists())
        response = self.client.post("/api/projects", json={
            "name": "empty-anomaly", "task": "anomaly",
            "parent": str(self.root), "class_names": [],
        })
        self.assertEqual(response.status_code, 200, response.text)

    def test_center_crop_settings_save_before_optional_model_pack_install(self):
        for task in ("detect", "segment", "anomaly", "obb"):
            with self.subTest(task=task):
                self.project(name=f"crop-{task}", task=task)
                response = self.client.put("/api/project", json={"training": {
                    "patchcore_crop_enabled": True,
                    "patchcore_crop_width": 320,
                    "patchcore_crop_height": 192,
                }})
                self.assertEqual(response.status_code, 200, response.text)
                saved = response.json()["training"]
                self.assertTrue(saved["patchcore_crop_enabled"])
                self.assertEqual(saved["patchcore_crop_width"], 320)
                self.assertEqual(saved["patchcore_crop_height"], 192)

    def test_state_recovers_after_project_is_moved_or_corrupted(self):
        for damage in ("moved", "invalid"):
            with self.subTest(damage=damage):
                project = self.project(name=f"project-{damage}")
                path = Path(project["filepath"])
                if damage == "moved":
                    path.rename(path.with_suffix(".moved"))
                else:
                    path.write_text("{broken", encoding="utf-8")
                response = self.client.get("/api/state")
                self.assertEqual(response.status_code, 200, response.text)
                state = response.json()
                self.assertIsNone(state["project"])
                self.assertTrue(state["error"])
                self.assertIn(project["filepath"], state["recent"])
                self.assertEqual(self.client.put("/api/project", json={"training": {"epochs": 3}}).status_code, 400)
                replacement = self.project(name=f"replacement-{damage}")
                state = self.client.get("/api/state").json()
                self.assertEqual(state["project"]["filepath"], replacement["filepath"])
                self.assertEqual(state["error"], "")
                if damage == "moved":
                    self.assertFalse(path.exists())
                else:
                    self.assertEqual(path.read_text(encoding="utf-8"), "{broken")

    def test_local_origin_and_header_guards(self):
        self.assertEqual(self.client.get("/api/state", headers={"Host": "evil.example"}).status_code, 400)
        self.assertEqual(self.client.get("/api/state", headers={"Origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(self.client.post("/api/projects/open", json={"path": "x"},
                         headers={"X-Studio-Request": ""}).status_code, 403)
        self.assertEqual(self.client.get("/api/options?task=invalid").status_code, 400)

    def test_model_catalog_exposes_requested_variants_without_claiming_release(self):
        response = self.client.get("/api/models?task=detect")
        self.assertEqual(response.status_code, 200, response.text)
        models = {item["model_id"]: item for item in response.json()["models"]}
        self.assertEqual({"re_detr_v4_small", "re_detr_v4_medium", "re_detr_v4_large", "libreyolo_detect_9t"}, set(models))
        self.assertTrue(all(item["release_status"] != "release_ready" for item in models.values()))

    def test_builtin_segmentation_worker_publishes_miou_metric(self):
        from webapp.storage import restore_project
        from webapp.worker import _train_builtin_project

        project_row = self.project(name="기본 분할 모델")
        project_data = read_json(project_row["filepath"])
        project_data["filepath"] = project_row["filepath"]
        project = restore_project(project_data)
        project.task = "segment"
        project.model.model_id = "deeplabv3plus_resnet34"
        project.data.num_classes = 2
        project.data.class_names = ["background", "defect"]
        project.training.input_size = 64
        project.training.epochs = 1
        project.training.batch_size = 1
        best = Path(project.project_dir) / "fake-best.pt"
        best.write_bytes(b"checkpoint")
        events = []

        class Context:
            def emit(self, event, args):
                events.append((event, args))

            def cancelled(self):
                return False

        forwarded = {}

        def fake_train(*args, log, **kwargs):
            forwarded.update(kwargs)
            log({"event": "epoch_finished", "epoch": 1, "total_epochs": 1,
                 "train_loss": 0.4, "val_loss": 0.3, "metric": 0.75})
            return best

        with patch("train_builtin.train_builtin", side_effect=fake_train), \
                patch("torch.load", return_value={"metric": 0.75, "epoch": 0}):
            record = _train_builtin_project(Context(), project, "cpu")

        self.assertEqual(record.best_metric_name, "mIoU")
        self.assertEqual(record.metrics_history, {
            "train_loss": [0.4], "val_loss": [0.3], "mIoU": [0.75],
        })
        self.assertEqual(forwarded["optimizer_name"], project.training.optimizer)
        self.assertEqual(forwarded["weight_decay"], project.training.weight_decay)
        self.assertEqual(forwarded["horizontal_flip"],
                         project.training.augmentation.horizontal_flip)
        epoch = next(args for event, args in events if event == "epoch_finished")
        self.assertEqual(epoch[3], {"mIoU": 0.75})

    def test_model_pack_job_endpoint_keeps_operation_allowlist(self):
        body = {"pack_dir": str(self.root), "data_dir": str(self.root),
                "work_dir": str(self.root), "request": {}}
        response = self.client.post("/api/jobs/model-pack/unknown", json=body)
        self.assertEqual(response.status_code, 400)

    def test_model_pack_install_endpoint_discovers_verified_pack(self):
        manifest = json.dumps({
            "schema_version": 1, "model_id": "vendor.web-pack", "pack_version": "1.0.0",
            "family": "Web Pack", "variant": "Small", "task": "classify",
            "runtimes": ["onnx"], "capabilities": ["infer", "export_onnx"],
            "input_size": [224, 224], "input_channels": [3], "release_status": "scoped",
        }).encode()
        files = {"manifest.json": manifest, "README.ko.md": b"offline"}
        checksums = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
        pack = self.root / "vendor.web-pack.dvmodel"
        with zipfile.ZipFile(pack, "w") as archive:
            for name, value in files.items():
                archive.writestr(name, value)
            archive.writestr("checksums.json", json.dumps({"files": checksums}))
        response = self.client.post("/api/models/install", json={
            "pack_path": str(pack), "allow_unsigned": True,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model"]["model_id"], "vendor.web-pack")
        models = self.client.get("/api/models?task=classify").json()["models"]
        self.assertIn("vendor.web-pack", {item["model_id"] for item in models})

    def test_real_class_delete_process_and_backup(self):
        project = self.project()
        original = Path(project["data"]["train_dir"]) / "NG" / "source.png"
        Image.new("RGB", (16, 16), "red").save(original)
        preview = self.client.post("/api/classes/preview-delete", json={"name": "NG"}).json()
        self.assertEqual(preview["image_count"], 1)
        response = self.client.post("/api/classes/delete", json={"name": "NG", "preview_digest": preview["preview_digest"]})
        self.assertEqual(response.status_code, 200, response.text)
        result = self.wait_job(response.json()["id"])
        self.assertEqual(result["job"]["status"], "completed", result)
        self.assertFalse(original.exists())
        self.assertTrue(Path(result["job"]["output"]["archive"]).exists())
        self.assertEqual(self.client.get("/api/state").json()["project"]["data"]["class_names"], ["OK"])

    def test_cancelled_process_releases_lock_and_persists(self):
        project = self.project()
        original = Path(project["data"]["train_dir"]) / "NG" / "source.png"
        Image.new("RGB", (128, 128), "gray").save(original)
        response = self.client.post("/api/jobs/defects", json={"folder": str(original.parent),
            "output": str(self.root / "generated"), "per_image": 100,
            "params": {"types": ["scratch"], "count": 100, "seed": 42}})
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()["id"]
        self.assertEqual(self.client.put("/api/project", json={"training": {"epochs": 3}}).status_code, 409)
        self.client.post(f"/api/jobs/{job_id}/cancel")
        result = self.wait_job(job_id)
        self.assertEqual(result["job"]["status"], "cancelled", result)
        self.assertIsNone(self.client.get("/api/state").json()["active_job"])
        self.assertEqual(read_json(self.root / "state" / "jobs" / job_id / "job.json")["status"], "cancelled")

    def test_defect_preview_cached_publish_and_project_boundaries(self):
        project = self.project()
        source = Path(project["data"]["train_dir"]) / "OK" / "normal.png"
        Image.new("RGB", (80, 64), "gray").save(source)
        settings = {"folder": str(source.parent), "output": str(self.root / "approved"),
                    "roi": "", "texture": "", "per_image": 2, "params": {"types": ["scratch"], "seed": 42}}
        saved = self.client.put("/api/defects/settings", json={"project_path": project["filepath"], "settings": settings})
        self.assertEqual(saved.status_code, 200, saved.text)
        response = self.client.post("/api/jobs/defects", json={**settings, "preview": True, "project_path": project["filepath"]})
        self.assertEqual(response.status_code, 200, response.text)
        job = self.wait_job(response.json()["id"])["job"]
        self.assertEqual(job["status"], "completed", job)
        rows = self.client.get(f"/api/defects/{job['id']}/candidates").json()
        self.assertEqual(rows["total"], 1)
        original_pixels = Path(rows["samples"][0]["image"]).read_bytes()
        for kind in ("image", "mask", "original"):
            response = self.client.get(f"/api/defects/{job['id']}/image/000001?kind={kind}")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.content.startswith(b"\x89PNG"))
        source.unlink()
        response = self.client.post("/api/jobs/defect-publish", json={"project_path": project["filepath"],
                "source_job": job["id"], "sample_ids": ["000001"], "output": settings["output"]})
        published = self.wait_job(response.json()["id"])["job"]
        self.assertEqual(published["status"], "completed", published)
        folder = Path(published["output"]["output_dir"])
        record = read_json(folder / "manifest.json")["samples"][0]
        self.assertEqual((folder / record["image"]).read_bytes(), original_pixels)
        blocked = self.client.post("/api/jobs/defect-publish", json={"project_path": project["filepath"],
                "source_job": job["id"], "sample_ids": ["000001"], "output": project["data"]["test_dir"]})
        self.assertEqual(blocked.status_code, 400)
        self.project(name="다른 프로젝트")
        self.assertEqual(self.client.get("/api/defects/jobs").json(), [])
        self.assertEqual(self.client.get(f"/api/defects/{job['id']}/candidates").status_code, 400)
        self.assertEqual(self.client.put("/api/defects/settings", json={"project_path": project["filepath"], "settings": settings}).status_code, 400)

    def test_cached_gradcam_range_and_toggle_without_model(self):
        job_id = "a" * 32
        root = self.root / "state" / "jobs" / job_id
        write_json(root / "job.json", {"id": job_id, "kind": "infer", "status": "completed", "created_at": "2026-01-01"})
        original = np.full((24, 32, 3), 120, np.uint8)
        activation = np.linspace(0, 1, 24 * 32, dtype=np.float32).reshape(24, 32)
        write_json(root / "results" / "0.json", {"summary": "NG 90%", "heatmap": {"kind": "Grad-CAM"}})
        np.savez(root / "results" / "0.npz", original=original, preview=original, activation=activation)
        path = f"/api/jobs/{job_id}/image/0"
        first = self.client.get(path + "?cam=true&lower=0&upper=1&alpha=1")
        second = self.client.get(path + "?cam=true&lower=.5&upper=.8&alpha=.5")
        plain = self.client.get(path + "?cam=false")
        self.assertEqual(first.status_code, 200, first.text if first.status_code != 200 else "")
        self.assertNotEqual(first.content, second.content)
        np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(plain.content))), original)
        self.assertEqual(self.client.get(path + "?lower=1&upper=0").status_code, 400)
        self.assertIsNone(self.app.state.jobs.process)

    def test_headless_training_inference_and_csv_processes(self):
        project = self.project()
        for split in ("train", "val"):
            for name, value in (("NG", 220), ("OK", 30)):
                for index in range(2):
                    target = Path(project["data"][f"{split}_dir"]) / name / f"{index}.png"
                    Image.new("RGB", (64, 64), (value + index,) * 3).save(target)
        update = self.client.put("/api/project", json={"training": {"training_mode": "custom", "device": "cpu",
                "epochs": 2, "batch_size": 2, "input_size": 64, "warmup_epochs": 0,
                "class_weights": "balanced", "selection_metric": "f1_macro"},
                "model": {"backbone_channels": [4, 8, 16, 32, 64], "csp_depth": [1, 1, 1, 1]}})
        self.assertEqual(update.status_code, 200, update.text)
        response = self.client.post("/api/jobs/train")
        self.assertEqual(response.status_code, 200, response.text)
        result = self.wait_job(response.json()["id"])
        self.assertEqual(result["job"]["status"], "completed", result)
        project = self.client.get("/api/state").json()["project"]
        record = project["runs"][-1]
        self.assertEqual(record["best_metric_name"], "f1_macro")
        self.assertTrue(Path(record["checkpoint_path"]).is_file())
        csv_response = self.client.get("/api/runs/0/csv")
        self.assertEqual(csv_response.status_code, 200)
        rows = list(csv.DictReader(io.StringIO(csv_response.content.decode("utf-8-sig"))))
        self.assertEqual(sum(int(row["is_best"]) for row in rows), 1)
        self.assertIn("run_total_time_hms", rows[0])
        response = self.client.post("/api/jobs/infer", json={"weights": record["checkpoint_path"],
                    "folder": project["data"]["val_dir"], "gradcam": True})
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()["id"]
        result = self.wait_job(job_id)
        self.assertEqual(result["job"]["status"], "completed", result)
        cached = self.client.get(f"/api/jobs/{job_id}/results").json()
        self.assertEqual(cached["total"], 4)
        self.assertTrue(cached["results"][0]["summary"])
        self.assertEqual(cached["results"][0]["inference_status"], "completed")
        self.assertEqual(cached["results"][0]["gradcam_status"], "completed")
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}/image/0").status_code, 200)
        self.assertEqual(result["job"]["output"]["errors"], 0)
        json_path = self.root / "input-crop.json"
        crop = {"width": 32, "height": 24}
        before = Path(record["checkpoint_path"]).read_bytes()
        for mode in ("json", "full", "model"):
            write_json(json_path, {"preprocessing": {"center_crop": crop}})
            response = self.client.post("/api/jobs/infer", json={"weights": record["checkpoint_path"],
                "folder": project["data"]["val_dir"], "gradcam": False,
                "crop_mode": mode, "crop_json": str(json_path)})
            self.assertEqual(response.status_code, 200, response.text)
            write_json(json_path, {"center_crop": None})
            selected_id = response.json()["id"]
            selected = self.wait_job(selected_id)
            self.assertEqual(selected["job"]["status"], "completed", selected)
            effective = selected["job"]["output"]["input_region"]
            self.assertEqual(effective["mode"], mode)
            self.assertEqual(effective["center_crop"], crop if mode == "json" else None)
            records = self.client.get(f"/api/jobs/{selected_id}/results").json()["results"]
            self.assertEqual(len(records), 4)
            for row in records:
                self.assertEqual(row["status"], "ok", row.get("error"))
                self.assertEqual(row["details"]["input_region"], effective)
                if mode == "json":
                    self.assertEqual(row["details"]["crop_box"], [16,20,48,44])
        self.assertEqual(Path(record["checkpoint_path"]).read_bytes(), before)
        write_json(json_path, {"center_crop": {"width": -1, "height": 32}})
        response = self.client.post("/api/jobs/infer", json={"weights": record["checkpoint_path"],
            "folder": project["data"]["val_dir"], "crop_mode":"json", "crop_json":str(json_path)})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.client.get(f"/api/inference/region?path={json_path}").status_code, 400)

    def test_main_server_and_training_imports_do_not_load_qt(self):
        script = """
import sys
import webapp
from webapp.server import create_app
assert 'torch' not in sys.modules
from core.trainer import TrainWorker
from core.patchcore_trainer import PatchCoreWorker
from core.efficientnet_trainer import EfficientNetTrainWorker
assert not any(key.startswith('PySide6') for key in sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_process_lock_blocks_mutations_after_server_shutdown(self):
        from webapp.locking import exclusive_file
        self.project()
        with exclusive_file(self.app.state.jobs.root / "compute.lock"):
            response = self.client.put("/api/project", json={"training": {"epochs": 3}})
            self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.client.put("/api/project", json={"training": {"epochs": 3}}).status_code, 200)

    def test_restart_recovery_and_partial_event_replay(self):
        from webapp.jobs import JobManager
        state = self.root / "recovery"
        job_id = "b" * 32
        directory = state / "jobs" / job_id
        write_json(directory / "job.json", {"id": job_id, "kind": "export", "status": "running",
                                           "created_at": "2026-01-01"})
        events = directory / "events.jsonl"
        first = json.dumps({"event": "log_message", "args": ["완료한 줄"]}, ensure_ascii=False).encode() + b"\n"
        events.write_bytes(first + b'{"event": "progress_updated", "args": [1, 2]')
        recovered = JobManager(state)
        response = recovered.read(job_id)
        self.assertEqual(response["job"]["status"], "interrupted")
        self.assertEqual(len(response["events"]), 1)
        self.assertEqual(response["offset"], len(first))
        with events.open("ab") as stream:
            stream.write(b"}\n")
        self.assertEqual(recovered.read(job_id, response["offset"])["events"][0]["args"], [1, 2])
        write_json(directory / "result.json", {"status": "completed", "output": {"verification": "passed"}})
        self.assertEqual(recovered.read(job_id)["job"]["status"], "completed")

    def test_external_saved_runs_are_preserved_before_settings_update(self):
        from core.project import ProjectManager, RunRecord
        project = self.project()
        external = ProjectManager.load(project["filepath"])
        external.runs.append(RunRecord(run_id="external-finished", status="completed"))
        ProjectManager.save(external)
        response = self.client.put("/api/project", json={"training": {"epochs": 12}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["runs"][0]["run_id"], "external-finished")

    def test_disk_cache_failure_preserves_computed_prediction(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.inference_types import InferenceResult
        from webapp.worker import JobContext, infer
        directory = self.root / "cache-failure"
        directory.mkdir()
        weights = directory / "fixture.pt"
        weights.write_bytes(b"mock loader fixture")
        source = directory / "source.png"
        Image.new("RGB", (8, 8)).save(source)
        prediction = InferenceResult(str(source), "ok", "classify", "NG 99%",
                                     inference_sec=.02, inference_status="completed")
        engine = SimpleNamespace(infer=lambda _path: prediction, _current_preview_rgb=np.zeros((8, 8, 3), np.uint8),
                                 _heatmap_cache=None, _gradcam=None, info="NG", input_region={"mode":"model", "center_crop":None})
        with patch("core.inference_loading.load_inference_engine", return_value=engine), \
             patch("numpy.savez", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "미리보기 저장 실패"):
                infer(JobContext(directory, monitor_parent=False), {"weights": str(weights), "images": [str(source)]})
        saved = read_json(directory / "results" / "0.json")
        self.assertEqual(saved["summary"], "NG 99%")
        self.assertEqual(saved["inference_sec"], .02)
        self.assertIn("disk full", saved["cache_error"])



if __name__ == "__main__":
    unittest.main()
