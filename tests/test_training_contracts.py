"""데이터와 학습 경계 회귀 테스트. Qt/GPU 없이 원본 메서드를 실행한다."""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def source_objects(path, names, namespace):
    """무거운 프레임워크 import만 피하고 실제 함수/클래스 본문을 사용한다."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
                         names=[ast.alias(name="annotations")])] + selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace


def source_method(path, class_name, method_name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
                         names=[ast.alias(name="annotations")]), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace[method_name]


class Loader:
    def __init__(self, dataset, batch_size, **kwargs):
        self.dataset, self.batch_size, self.options = dataset, batch_size, kwargs

    def __len__(self):
        count = len(self.dataset)
        return count // self.batch_size if self.options.get("drop_last") else math.ceil(count / self.batch_size)


class Subset:
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, list(indices)

    def __len__(self):
        return len(self.indices)


class Generator:
    def manual_seed(self, seed):
        self.seed = seed
        return self


def split(dataset, sizes, generator):
    indices = list(range(len(dataset)))
    random.Random(generator.seed).shuffle(indices)
    return Subset(dataset, indices[:sizes[0]]), Subset(dataset, indices[sizes[0]:])


def dataset_namespace():
    tensor_ops = SimpleNamespace(to_tensor=lambda image: np.asarray(image),
                                normalize=lambda image, mean, std: image)
    torch = SimpleNamespace(Generator=Generator, from_numpy=np.asarray,
                            utils=SimpleNamespace(data=SimpleNamespace(random_split=split, Subset=Subset)))
    names = {"ClassificationDataset", "SegmentationDataset", "AnomalyDataset",
             "_TransformOverrideSubset", "create_classification_loaders", "create_anomaly_loaders"}
    from opencv_preprocess import read_image, resize
    return source_objects("python/dataset.py", names, {
        "read_image": read_image, "resize": resize,
        "os": os, "random": random, "np": np, "Image": Image, "Dataset": object,
        "DataLoader": Loader, "torch": torch, "T": SimpleNamespace(functional=tensor_ops),
        "get_normalize_params": lambda channels: ([0], [1]),
        "get_classification_transforms": lambda *args, **kwargs: None,
    })


def write_image(path, values=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8) if values is None else values).save(path)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV image loading unavailable")
class DataContracts(unittest.TestCase):
    def test_validation_missing_class_preserves_training_id(self):
        ns = dataset_namespace()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("train/defect/a.png", "train/good/b.png", "val/good/c.png"):
                write_image(root / name)
            train, val, names = ns["create_classification_loaders"](str(root), batch_size=8)
            self.assertEqual(names, ["defect", "good"])
            self.assertEqual(val.dataset.samples[0][1], 1)
            self.assertEqual(len(train), 1)
            write_image(root / "val/unknown/d.png")
            with self.assertRaisesRegex(ValueError, "학습에 없는"):
                ns["create_classification_loaders"](str(root))

    def test_small_auto_split_is_nonempty_and_repeatable(self):
        ns = dataset_namespace()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(3):
                write_image(root / "good" / f"{index}.png")
            first = ns["create_classification_loaders"](str(root), batch_size=16, val_split=.5)
            second = ns["create_classification_loaders"](str(root), batch_size=16, val_split=.5)
            self.assertEqual(len(first[0]), 1)
            self.assertEqual(len(first[1]), 1)
            self.assertEqual(first[0].dataset.indices, second[0].dataset.indices)

    def test_palette_mask_preserves_ids_and_rectangular_shape(self):
        ns = dataset_namespace()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_image(root / "images/a.png", np.zeros((2, 2), dtype=np.uint8))
            (root / "masks").mkdir()
            mask = Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).convert("P")
            palette = [0] * 768
            palette[3:6] = [255, 0, 0]
            mask.putpalette(palette)
            mask.save(root / "masks/a.png")
            dataset = ns["SegmentationDataset"](str(root / "images"), str(root / "masks"),
                                                 input_size=(8, 12), is_train=False, num_classes=2)
            _, result = dataset[0]
            self.assertEqual(result.shape, (8, 12))
            self.assertEqual(set(np.unique(result)), {0, 1})
            Image.fromarray(np.array([[0, 255], [1, 255]], dtype=np.uint8)).save(root / "masks/a.png")
            self.assertEqual(set(np.unique(dataset[0][1])), {0, 1, 255})
            Image.fromarray(np.array([[0, 3], [1, 0]], dtype=np.uint8)).save(root / "masks/a.png")
            with self.assertRaisesRegex(ValueError, "클래스 범위"):
                dataset[0]
            mask.convert("RGB").save(root / "masks/a.png")
            with self.assertRaisesRegex(ValueError, "클래스 인덱스"):
                dataset[0]

    def test_reconstruction_validation_has_no_training_flip(self):
        ns = dataset_namespace()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(4):
                write_image(root / "train/good" / f"{index}.png")
            write_image(root / "train/defect/ignored.png")
            train, val = ns["create_anomaly_loaders"](str(root), batch_size=16, val_split=.5)
            self.assertEqual(len(train.dataset), 2)
            self.assertEqual(len(val.dataset), 2)
            self.assertEqual(val.dataset.dataset.flip_prob, 0)
            self.assertTrue(val.dataset.dataset.is_train)  # reconstruction-pair contract
            self.assertTrue(all("good" in p for p in train.dataset.dataset.samples))
            self.assertFalse(set(train.dataset.indices) & set(val.dataset.indices))


class MetadataAndAdapters(unittest.TestCase):
    def test_gui_loader_receives_split_and_mask_class_count(self):
        import sys
        from unittest.mock import patch
        captured = {}

        def classify(**kwargs):
            captured.update(kwargs)
            self.assertNotIn("num_classes", kwargs)
            return [1], [1], ["defect", "good"]

        def other(**kwargs):
            captured.update(kwargs)
            return [1], [1]

        module = SimpleNamespace(create_classification_loaders=classify,
                                 create_segmentation_loaders=other,
                                 create_anomaly_loaders=other, create_detection_loaders=other)
        fn = source_method("gui/core/trainer.py", "TrainWorker", "_create_dataloaders", {})
        worker = SimpleNamespace(signals=SimpleNamespace(log_message=SimpleNamespace(emit=lambda *a: None)))
        cfg = SimpleNamespace(input_size=64, batch_size=2, in_channels=1)
        data = SimpleNamespace(root="data", val_split=.35, class_names=[], num_classes=7)
        with patch.dict(sys.modules, {"dataset": module}):
            fn(worker, "classify", data, cfg)
            self.assertEqual(captured["val_split"], .35)
            self.assertEqual(data.num_classes, 2)
            captured.clear()
            fn(worker, "segment", data, cfg)
            self.assertEqual(captured["num_classes"], 2)
            captured.clear()
            fn(worker, "anomaly", data, cfg)
            self.assertEqual(captured["val_split"], .35)

    def test_checkpoint_contains_restore_and_grayscale_preprocessing(self):
        spec = importlib.util.spec_from_file_location("contract_checkpoint", ROOT / "python/checkpoint.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.make_checkpoint_metadata("anomaly", 2, ["good", "defect"], (64, 96), 1,
                         {"backbone_channels": [8, 16, 32, 64, 128], "csp_depth": [1, 1, 1, 1], "dropout": .1})
        self.assertEqual(result["preprocessing"]["input_size"], [64, 96])
        self.assertEqual(result["preprocessing"]["mean"], [.449])
        self.assertEqual(result["model_config"]["backbone_channels"][0], 8)
        self.assertIsNone(result["anomaly_threshold"])
        json.dumps(result, allow_nan=False)



    def test_short_cosine_schedule_has_finite_nonnegative_rates(self):
        namespace = {"math": math, "optim": SimpleNamespace(lr_scheduler=SimpleNamespace(
            LambdaLR=lambda optimizer, function: function))}
        fn = source_method("gui/core/trainer.py", "TrainWorker", "_create_scheduler", namespace)
        schedule = fn(None, object(), SimpleNamespace(scheduler="cosine", epochs=3, warmup_epochs=3))
        self.assertTrue(all(math.isfinite(schedule(i)) and schedule(i) >= 0 for i in range(4)))


class TrainingOrchestration(unittest.TestCase):
    def run_fake_training(self, task, val_scores, *, loss_pairs=None, events=None,
                          fail_best_save=False):
        from datetime import datetime
        spec = importlib.util.spec_from_file_location("contract_checkpoint_run", ROOT / "python/checkpoint.py")
        checkpoint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checkpoint)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        stored, constructed_counts = {}, []
        events = events if events is not None else []

        class Model:
            def __init__(self, **kwargs):
                constructed_counts.append(kwargs["num_classes"])

            def to(self, device):
                return self

            def get_param_count(self):
                return {"total": 1, "total_MB": 0}

            def state_dict(self):
                return {"weight": "sentinel"}

            def load_state_dict(self, state):
                pass

        def save(state, path):
            if path.endswith("best.pt") and fail_best_save:
                raise OSError("체크포인트 저장 실패")
            stored[path] = state
            Path(path).touch()
            events.append(("save", state["epoch"], Path(path).name))

        metric_name = {"classify": "accuracy", "segment": "mIoU",
                       "detect": "mAP_50", "anomaly": "recon_loss"}[task]
        from core.project import TrainingConfig
        cfg = TrainingConfig(epochs=len(val_scores), batch_size=8, learning_rate=.01, in_channels=1,
                             input_size=64, use_amp=False, early_stop_patience=10,
                             training_mode="custom", anomaly_method="reconstruction")
        data = SimpleNamespace(num_classes=1, class_names=["stale"])
        model_cfg = SimpleNamespace(backbone_channels=[8, 16, 32, 64, 128], csp_depth=[1, 1, 1, 1],
                                    dropout=.1, freeze_backbone=False, pretrained_weights="")
        project = SimpleNamespace(training=cfg, data=data, model=model_cfg, task=task, project_dir=temp.name, runs=[])
        device_manager = SimpleNamespace(get_device=lambda d: "cpu", supports_amp=lambda d: False,
                          get_device_label=lambda d: "cpu", get_optimal_num_workers=lambda d: 0)
        signals = SimpleNamespace(**{name: SimpleNamespace(
            emit=lambda *args, name=name: events.append((name, *args))) for name in (
            "log_message", "epoch_finished", "best_epoch_updated", "progress_updated", "lr_updated", "early_stopped", "eval_finished", "training_finished")})
        namespace = {"os": os, "math": math, "np": np, "datetime": datetime,
                     "get_device_manager": lambda: device_manager, "CustomCSP": Model,
                     "make_checkpoint_metadata": checkpoint.make_checkpoint_metadata,
                     "ProjectManager": SimpleNamespace(new_run_id=lambda **kwargs: "test_run"),
                     "RunRecord": lambda **kwargs: SimpleNamespace(config_snapshot={}, **kwargs), "TASK_METRIC_NAMES": {task: {"primary": metric_name, "display": []}},
                     "torch": SimpleNamespace(save=save, load=lambda path, **kwargs: stored[path])}
        fn = source_method("gui/core/trainer.py", "TrainWorker", "_run_training", namespace)
        worker = SimpleNamespace(project=project, signals=signals, _stop_requested=False, engine_name="custom")
        # 모델별 확장 지점도 실제 기본 구현을 실행한다. 저장 I/O만 관찰용으로 대체한다.
        for method_name in ("_prepare_training", "_build_model", "_initialize_model",
                            "_checkpoint_metadata", "_restore_training_state", "_checkpoint_extra"):
            method = source_method("gui/core/trainer.py", "TrainWorker", method_name, namespace)
            setattr(worker, method_name, method.__get__(worker))
        worker._save_checkpoint = save

        def loaders(*args, **kwargs):
            data.num_classes, data.class_names = 2, ["defect", "good"]
            return [1], [1]

        values = iter(val_scores)
        worker._create_dataloaders = loaders
        worker._create_optimizer = lambda *args: SimpleNamespace(param_groups=[{"lr": .01}], state_dict=lambda: {})
        worker._create_scheduler = lambda *args: None
        worker._create_criterion = lambda *args, **kwargs: None
        losses = loss_pairs or [(1.0, 1.0)] * len(val_scores)
        current_epoch = [0]

        def train(*args, **kwargs):
            current_epoch[0] = args[5] - 1
            return losses[current_epoch[0]][0]

        worker._train_one_epoch = train
        worker._validate = lambda *args: (
            losses[current_epoch[0]][1], {metric_name: next(values), "auroc": None})
        worker._full_evaluation = lambda *args: {"auroc": None}
        worker._cpu_state = lambda value: value
        fn(worker)
        return stored, constructed_counts, project.runs[0]

    def test_zero_accuracy_saves_first_checkpoint_after_class_discovery(self):
        stored, counts, record = self.run_fake_training("classify", [0.0])
        self.assertEqual(counts, [2])
        best = next(value for path, value in stored.items() if path.endswith("best.pt"))
        self.assertEqual(best["best_metric"], 0)
        self.assertEqual(best["model_config"]["backbone_channels"][0], 8)
        self.assertEqual(record.epochs_done, 1)
        self.assertEqual(record.status, "completed")

    def test_normal_reconstruction_selects_lower_loss_not_unavailable_auroc(self):
        stored, _, record = self.run_fake_training("anomaly", [.8, .3, .5])
        best = next(value for path, value in stored.items() if path.endswith("best.pt"))
        self.assertEqual(best["epoch"], 2)
        self.assertEqual(best["best_metric_name"], "recon_loss")
        self.assertEqual(record.best_epoch, 2)
        self.assertEqual(record.epochs_done, 3)
        json.dumps(best, allow_nan=False)


if __name__ == "__main__":
    unittest.main()


class AtomicCheckpointTests(unittest.TestCase):
    def test_partial_save_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "best.pt"
            target.write_bytes(b"previous")
            def failing_save(state, filename):
                Path(filename).write_bytes(b"partial")
                raise OSError("disk write failed")
            save = source_method("gui/core/trainer.py", "TrainWorker", "_save_checkpoint",
                                 {"os": os, "torch": SimpleNamespace(save=failing_save)})
            with self.assertRaisesRegex(OSError, "disk write failed"):
                save({}, str(target))
            self.assertEqual(target.read_bytes(), b"previous")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_successful_save_replaces_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "best.pt"
            target.write_bytes(b"previous")
            save = source_method("gui/core/trainer.py", "TrainWorker", "_save_checkpoint",
                {"os": os, "torch": SimpleNamespace(save=lambda state, filename: Path(filename).write_bytes(b"complete"))})
            save({}, str(target))
            self.assertEqual(target.read_bytes(), b"complete")
            self.assertEqual(list(Path(directory).iterdir()), [target])
