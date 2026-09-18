"""실제 Torch 백본, 전이 가중치, 특징 뱅크, 파일/Qt 워커 통합 검사."""
import csv
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "gui"))
RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision"))
QT = RUNTIME and importlib.util.find_spec("PySide6") is not None
if RUNTIME:
    import torch
    from torchvision.models import resnet18
    from patchcore import PatchCore, PatchCoreCancelled
    from patchcore_data import input_tensor, IMAGENET_MEAN, IMAGENET_STD
    from patchcore_sampling import FeatureReservoir, coreset


@unittest.skipUnless(RUNTIME, "Torch/torchvision runtime required")
class PatchCoreFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.weights = Path(cls.temp.name) / "사전학습_테스트.pth"
        # 이 fixture는 파일 전이 검사용. 공식 사전학습은 별도 CI smoke에서 검증한다.
        with torch.random.fork_rng():
            torch.manual_seed(42)
            state = resnet18(weights=None).state_dict()
        torch.save({"state_dict": {"module.encoder." + k: v for k, v in state.items()}}, cls.weights)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.path = Path(self.work.name)

    def tearDown(self):
        self.work.cleanup()

    def model(self, **kwargs):
        defaults = dict(backbone_name="resnet18", device="cpu", pretrained=False,
                        backbone_weights=str(self.weights), input_size=32, max_candidates=20,
                        max_memory_bank=8, sampling_ratio=.5, n_neighbors=3)
        defaults.update(kwargs)
        return PatchCore(**defaults)

    def batches(self):
        generator = torch.Generator().manual_seed(21)
        return [torch.rand(2, 3, 32, 32, generator=generator) for _ in range(2)]

@unittest.skipUnless(RUNTIME, "Torch/torchvision runtime required")
class PatchCoreRuntimeTests(PatchCoreFixture):
    def test_real_local_transfer_fit_roundtrip_and_frozen_batchnorm(self):
        pc = self.model()
        before = {k: v.clone() for k, v in pc.backbone.state_dict().items()}
        pc.backbone.train()  # fit이 eval을 강제해야 한다.
        batches = self.batches()
        pc.fit(batches)
        self.assertFalse(pc.backbone.training)
        self.assertTrue(all(not p.requires_grad for p in pc.backbone.parameters()))
        for key, tensor in pc.backbone.state_dict().items():
            torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
        self.assertEqual(pc.memory_bank.shape, (8, 384))
        self.assertEqual(pc.training_metadata["seen_patches"], 64)
        self.assertEqual(pc.training_metadata["candidate_patches"], 20)
        self.assertTrue(pc.training_metadata["candidate_limit_applied"])
        self.assertEqual(pc.weight_source["kind"], "local_backbone")
        pc.anomaly_threshold = .123
        result = pc.predict(batches[0])
        checkpoint = self.path / "저장된_모델.pt"
        pc.save(checkpoint)
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("offline load")):
            loaded = PatchCore.load(checkpoint)
        self.assertEqual(loaded.anomaly_threshold, .123)
        self.assertEqual(loaded.normalized_threshold, .5)
        self.assertEqual(loaded.get_score_normalization(), pc.get_score_normalization())
        data = torch.load(checkpoint, weights_only=False)
        self.assertEqual(data["score_normalization"]["scale"], .123)
        del data["score_normalization"]
        torch.save(data, checkpoint)
        legacy = PatchCore.load(checkpoint)
        self.assertEqual(legacy.normalized_threshold, .5)
        self.assertEqual(legacy.get_score_normalization(), loaded.get_score_normalization())
        data["score_normalization"] = {"method": "distance_ratio_v1", "scale": 0}
        torch.save(data, checkpoint)
        with self.assertRaisesRegex(ValueError, "정규화"):
            PatchCore.load(checkpoint)
        self.assertEqual(loaded.weight_source, pc.weight_source)
        for expected, actual in zip(result, loaded.predict(batches[0])):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(result[1].shape, (2, 32, 32))

    def test_append_preserves_old_features_and_requires_new_calibration(self):
        pc = self.model()
        pc.fit(self.batches())
        old = pc.memory_bank.clone()
        pc.max_memory_bank = 12
        pc.anomaly_threshold = .4
        self.assertEqual(pc.get_score_normalization()["scale"], .4)
        pc.fit(self.batches(), append=True)
        torch.testing.assert_close(pc.memory_bank[:8], old, rtol=0, atol=0)
        self.assertEqual(len(pc.memory_bank), 12)
        self.assertIsNone(pc.anomaly_threshold)
        self.assertIsNone(pc.score_normalization)
        self.assertEqual(pc.training_metadata["retained_patches"], 8)
        with self.assertRaisesRegex(ValueError, "한도"):
            pc.fit(self.batches(), append=True)

    def test_cancellation_after_first_batch_preserves_existing_bank(self):
        pc = self.model()
        pc.fit(self.batches())
        old = pc.memory_bank.clone()
        progress = []
        with self.assertRaises(PatchCoreCancelled):
            pc.fit(self.batches(), progress_callback=lambda *args: progress.append(args),
                   cancel_callback=lambda: bool(progress))
        self.assertEqual(len(progress), 1)
        torch.testing.assert_close(pc.memory_bank, old, rtol=0, atol=0)

    def test_coreset_cancellation_preserves_existing_bank(self):
        pc = self.model()
        pc.fit(self.batches())
        old = pc.memory_bank.clone()
        stopped = [False]
        def progress(current, total, message):
            if "대표 패치 선별" in message:
                stopped[0] = True
        with self.assertRaises(PatchCoreCancelled):
            pc.fit(self.batches(), progress_callback=progress, cancel_callback=lambda: stopped[0])
        torch.testing.assert_close(pc.memory_bank, old, rtol=0, atol=0)

    def test_knn_merges_all_bank_chunks_for_global_neighbors(self):
        pc = self.model()
        generator = torch.Generator().manual_seed(7)
        pc.memory_bank = torch.randn(2053, 384, generator=generator)
        query = torch.randn(263, 384, generator=generator)
        query[:3] = pc.memory_bank[-3:]
        pc.n_neighbors = 9
        expected = torch.cdist(query, pc.memory_bank).topk(9, largest=False, dim=1).values.mean(dim=1)
        actual = pc._compute_knn_distances(query)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-3)

    def test_bounded_sampling_is_deterministic_without_global_rng_changes(self):
        features = torch.arange(120 * 140, dtype=torch.float32).reshape(120, 140)
        state = torch.get_rng_state().clone()
        banks = []
        for _ in range(2):
            pool = FeatureReservoir(23, seed=12)
            for batch in features.split(17):
                pool.add(batch)
                self.assertLessEqual(len(pool.features), 23)
            self.assertEqual(pool.seen, 120)
            banks.append(coreset(pool.features, 8, seed=13))
        torch.testing.assert_close(banks[0], banks[1], rtol=0, atol=0)
        self.assertEqual(len(torch.unique(banks[0], dim=0)), 8)
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)

    def test_incompatible_or_nan_backbone_is_rejected(self):
        bad = self.path / "bad.pt"
        torch.save({"model": {"model.0.weight": torch.ones(1)}}, bad)
        with self.assertRaisesRegex(ValueError, "누락"):
            self.model(backbone_weights=str(bad))
        data = torch.load(self.weights, weights_only=False)
        data["state_dict"]["module.encoder.conv1.weight"].fill_(float("nan"))
        torch.save(data, bad)
        with self.assertRaisesRegex(ValueError, "conv1.weight"):
            self.model(backbone_weights=str(bad))
        torch.save({"backbone_name": "resnet50"}, bad)
        with self.assertRaisesRegex(ValueError, "불일치"):
            self.model(backbone_weights=str(bad))

    def test_imagenet_download_failure_does_not_fall_back_to_random(self):
        from patchcore_weights import build_backbone
        with patch("model_download.cached_imagenet_weights", side_effect=OSError("network unavailable")), \
                patch("torchvision.models.resnet18") as constructor:
            with self.assertRaisesRegex(RuntimeError, "로컬 백본 가중치"):
                build_backbone("resnet18", True)
        constructor.assert_not_called()

    def test_certificate_failure_preserves_cause_and_gives_offline_instructions(self):
        import ssl
        from urllib.error import URLError
        from patchcore_weights import build_backbone
        failure = URLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))
        with patch("model_download.cached_imagenet_weights", side_effect=failure), \
                patch("torchvision.models.resnet18") as constructor:
            with self.assertRaisesRegex(RuntimeError, "SSL_CERT_FILE") as error:
                build_backbone("resnet18", True)
        self.assertIs(error.exception.__cause__, failure)
        self.assertIn("https://download.pytorch.org/", str(error.exception))
        self.assertIn("로컬 ResNet 백본 가중치", str(error.exception))
        constructor.assert_not_called()

    def test_atomic_save_failure_preserves_existing_checkpoint(self):
        pc = self.model()
        pc.fit(self.batches())
        filename = self.path / "best.pt"
        pc.save(filename)
        before = filename.read_bytes()
        with patch("patchcore.torch.save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                pc.save(filename)
        self.assertEqual(filename.read_bytes(), before)
        self.assertEqual(list(self.path.glob(".patchcore-*")), [])

    def test_corrupt_checkpoint_and_legacy_preprocessing(self):
        pc = self.model()
        pc.fit(self.batches())
        filename = self.path / "model.pt"
        pc.save(filename)
        data = torch.load(filename, weights_only=False)
        state = data.pop("backbone_state_dict")
        torch.save(data, filename)
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("no download")):
            with self.assertRaisesRegex(ValueError, "가중치 누락"):
                PatchCore.load(filename)
        data["backbone_state_dict"] = state
        data["schema_version"] = 2
        data.pop("preprocessing")
        torch.save(data, filename)
        self.assertEqual(PatchCore.load(filename).preprocessing, "legacy_pil_rgb")
        data["memory_bank"][0, 0] = float("nan")
        torch.save(data, filename)
        with self.assertRaisesRegex(ValueError, "메모리 뱅크"):
            PatchCore.load(filename)

    def test_16bit_grayscale_uses_full_range_and_shared_inference_transform(self):
        filename = self.path / "height.png"
        values = np.tile(np.array([0, 32768, 65535], dtype=np.uint16), (3, 1))
        Image.fromarray(values).save(filename)
        tensor = input_tensor(filename, 3)
        restored = tensor * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        np.testing.assert_allclose(restored[0].numpy(), values.astype(np.float32) / 65535, atol=1e-6)
        self.assertGreater(float(input_tensor(filename, 3, "legacy_pil_rgb")[0, 0, 1]), float(tensor[0, 0, 1]))
        pc = self.model()
        pc.fit(self.batches())
        direct = pc.predict(input_tensor(filename, 32).unsqueeze(0))
        for expected, actual in zip(direct, pc.predict_from_path(filename)):
            np.testing.assert_array_equal(actual, expected)

    def test_bad_image_reports_filename(self):
        bad = self.path / "broken.png"
        bad.write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "broken.png"):
            input_tensor(bad, 32)

    def test_cropped_dataset_checkpoint_and_cpu_inference_share_exact_pixels(self):
        from core.inference_loading import load_cpu_engine
        from patchcore_data import PatchCoreDataset
        from center_crop import center_crop_box
        crop = {"width": 18, "height": 12}
        raw = np.arange(31 * 43, dtype=np.uint16).reshape(31, 43) * 40
        path = self.path / "full.png"
        cropped = self.path / "roi.png"
        Image.fromarray(raw).save(path)
        Image.fromarray(raw).crop(center_crop_box((43, 31), crop)).save(cropped)
        tensor, _ = PatchCoreDataset([path], 32, center_crop=crop)[0]
        torch.testing.assert_close(tensor, input_tensor(cropped, 32), rtol=0, atol=0)
        pc = self.model(center_crop=crop)
        pc.fit([tensor.unsqueeze(0)])
        pc.anomaly_threshold = 3
        checkpoint = self.path / "cropped.pt"
        pc.save(checkpoint)
        loaded = PatchCore.load(checkpoint)
        self.assertEqual(loaded.center_crop, crop)
        self.assertEqual(loaded.get_info()["center_crop"], crop)
        self.assertEqual(torch.load(checkpoint, weights_only=False)["schema_version"], 4)
        expected = loaded.predict(tensor.unsqueeze(0))
        for direct, from_path in zip(expected, loaded.predict_from_path(path)):
            np.testing.assert_array_equal(direct, from_path)
        engine = load_cpu_engine(checkpoint)
        result = engine.infer(str(path))
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(result.details["crop_box"], [12, 9, 30, 21])
        self.assertEqual(engine._heatmap_cache["valid_mask"].shape, (31, 43))
        self.assertEqual(int(engine._heatmap_cache["valid_mask"].sum()), 18 * 12)
        self.assertLessEqual(result.score, 1)
        self.assertIn("중앙 18×12", engine.info)
        before = checkpoint.read_bytes()
        for request, region in (({"mode": "full"}, None),
                                ({"mode": "json", "center_crop": {"width": 24, "height": 16}},
                                 {"width": 24, "height": 16}), (None, crop)):
            engine.configure_input_region(request)
            actual = engine.infer(str(path))
            self.assertEqual(actual.status, "ok", actual.error)
            direct_scores, _ = loaded.predict(input_tensor(path, 32, loaded.preprocessing, region).unsqueeze(0))
            np.testing.assert_allclose(actual.details["raw_score"], direct_scores[0], rtol=1e-5, atol=1e-5)
            self.assertEqual(actual.details["input_region"]["center_crop"], region)
            self.assertEqual(engine._patchcore_model.center_crop, crop)
            valid = engine._heatmap_cache["valid_mask"]
            self.assertEqual(int(valid.sum()) if valid is not None else 43*31,
                             region["width"]*region["height"] if region else 43*31)
        self.assertEqual(checkpoint.read_bytes(), before)
        Image.new("RGB", (10, 10)).save(self.path / "too_small.png")
        failed = engine.infer(str(self.path / "too_small.png"))
        self.assertEqual(failed.status, "error")
        self.assertIn("원본", failed.error)
        self.assertEqual(engine.infer(str(path)).status, "ok")
        data = torch.load(checkpoint, weights_only=False)
        del data["center_crop"]
        torch.save(data, checkpoint)
        with self.assertRaisesRegex(ValueError, "크롭 설정 누락"):
            PatchCore.load(checkpoint)


@unittest.skipUnless(QT, "Torch and Qt runtime required")
class PatchCoreWorkerTests(PatchCoreFixture):
    def test_worker_builds_from_local_weights_without_validation(self):
        from core.project import ProjectManager
        from core.patchcore_trainer import PatchCoreWorker
        root = self.path / "data"
        (root / "train/good").mkdir(parents=True)
        (root / "val/good").mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.full((48, 48, 3), 70 + i * 30, np.uint8)).save(root / f"train/good/{i}.png")
        project = ProjectManager.create_new("PatchCore", "anomaly", str(self.path / "project"), ["good", "defect"])
        project.data.root = str(root)
        cfg = project.training
        cfg.patchcore_weight_source, cfg.patchcore_weights, cfg.patchcore_backbone = "backbone", str(self.weights), "resnet18"
        cfg.input_size, cfg.batch_size, cfg.device = 32, 2, "cpu"
        cfg.patchcore_crop_enabled = True
        cfg.patchcore_crop_width, cfg.patchcore_crop_height = 32, 40
        cfg.patchcore_max_candidates, cfg.patchcore_max_memory_bank = 20, 8
        worker = PatchCoreWorker(project)
        errors, done = [], []
        worker.signals.training_error.connect(errors.append)
        worker.signals.training_finished.connect(lambda *args: done.append(args))
        worker.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(done), 1)
        record = project.runs[-1]
        self.assertEqual(record.status, "completed")
        self.assertIsNone(record.eval_results["auroc"])
        self.assertIn("best_unavailable_NA_epoch_1.pt", record.checkpoint_path)
        self.assertEqual(record.config_snapshot["patchcore_transfer"]["kind"], "local_backbone")
        loaded = PatchCore.load(record.checkpoint_path)
        self.assertEqual(loaded.backbone_name, "resnet18")
        self.assertEqual(loaded.center_crop, {"width": 32, "height": 40})
        with (Path(record.checkpoint_path).parent / "results.csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["is_best"], "1")
        self.assertIn(":", row["epoch_time_hms"])
        cfg.patchcore_weight_source, cfg.patchcore_weights = "patchcore", record.checkpoint_path
        cfg.patchcore_append, cfg.patchcore_max_memory_bank = True, 12
        transferred = worker._prepare_model(cfg, "cpu")
        torch.testing.assert_close(transferred.memory_bank, loaded.memory_bank, rtol=0, atol=0)
        cfg.patchcore_crop_width = 30
        with self.assertRaisesRegex(ValueError, "크롭 불일치"):
            worker._prepare_model(cfg, "cpu")
        cfg.patchcore_crop_width = 32
        cfg.input_size = 64
        with self.assertRaisesRegex(ValueError, "입력 크기 불일치"):
            worker._prepare_model(cfg, "cpu")
        cfg.patchcore_append = False
        rebuilt = worker._prepare_model(cfg, "cpu")
        self.assertIsNone(rebuilt.memory_bank)
        self.assertEqual(rebuilt.input_size, 64)

    def test_worker_error_marks_run_failed_and_reports_stage(self):
        from core.project import ProjectManager
        from core.patchcore_trainer import PatchCoreWorker
        project = ProjectManager.create_new("Broken", "anomaly", str(self.path / "broken"), ["good", "defect"])
        project.data.root = str(self.path / "missing")
        worker = PatchCoreWorker(project)
        errors = []
        worker.signals.training_error.connect(errors.append)
        worker.run()
        self.assertEqual(project.runs[-1].status, "failed")
        self.assertIn("데이터 확인", errors[0])
        self.assertTrue((Path(worker._run_dir) / "training_error.txt").is_file())


@unittest.skipUnless(QT, "Torch and Qt runtime required")
class PatchCoreUISettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_ui_restores_transfer_options_and_disables_unused_training_settings(self):
        from core.project import ProjectData
        from widgets.training_widget import TrainingWidget
        project = ProjectData(task="anomaly")
        cfg = project.training
        cfg.patchcore_weight_source, cfg.patchcore_backbone = "backbone", "resnet18"
        cfg.patchcore_weights = "local.pth"
        cfg.patchcore_max_candidates, cfg.patchcore_max_memory_bank, cfg.patchcore_seed = 200, 37, 15
        cfg.patchcore_crop_enabled = True
        cfg.patchcore_crop_width, cfg.patchcore_crop_height = 320, 192
        widget = TrainingWidget()
        try:
            widget.set_project(project)
            self.assertTrue(widget.mode_group.isHidden())
            self.assertEqual(widget.pc_source_combo.currentData(), "backbone")
            self.assertTrue(widget.pc_weights_edit.isEnabled())
            self.assertFalse(widget.pc_append_check.isEnabled())
            self.assertFalse(widget.lr_spin.isEnabled())
            self.assertFalse(widget.epochs_spin.isEnabled())
            self.assertFalse(widget.amp_check.isEnabled())
            self.assertTrue(widget.batch_spin.isEnabled())
            self.assertTrue(widget.tl_group.isHidden())
            self.assertTrue(widget.pc_crop_check.isChecked())
            self.assertEqual(widget.pc_crop_width_spin.value(), 320)
            self.assertEqual(widget.pc_crop_height_spin.value(), 192)
            self.assertTrue(widget.pc_crop_width_spin.isEnabled())
            widget.pc_crop_width_spin.setValue(384)
            widget.pc_source_combo.setCurrentIndex(widget.pc_source_combo.findData("patchcore"))
            self.assertTrue(widget.pc_append_check.isEnabled())
            self.assertFalse(widget.pc_backbone_combo.isEnabled())
            widget.pc_append_check.setChecked(True)
            widget.collect_config()
            self.assertTrue(cfg.patchcore_append)
            self.assertEqual(cfg.patchcore_crop_width, 384)
            self.assertEqual(cfg.patchcore_crop_height, 192)
            self.assertEqual(cfg.patchcore_max_memory_bank, 37)
            self.assertEqual(cfg.patchcore_seed, 15)
            self.assertEqual(cfg.patchcore_weights, "local.pth")
            widget.anomaly_method_combo.setCurrentIndex(1)
            self.assertTrue(widget.lr_spin.isEnabled())
            self.assertFalse(widget.pc_source_combo.isEnabled())
            self.assertTrue(widget.pc_crop_check.isEnabled())
            self.assertTrue(widget.pc_crop_width_spin.isEnabled())
            self.assertFalse(widget.tl_group.isHidden())
        finally:
            widget.close()
            widget.deleteLater()


if __name__ == "__main__":
    unittest.main()
