"""Train through the desktop/web worker, then audit saved classification results.

This is a small real-image integration experiment, not an accuracy benchmark.
The torchvision controls replace only the EfficientNet constructor; the data,
pretrained loading, optimizer, evaluation and persistence remain the same.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import socket
import subprocess
import sys
from types import MethodType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(ROOT / "gui")]
ROLES = ["custom_b0", "official_b0", "custom_b1", "official_b1"]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare(output, dataset_name="cifar10"):
    import numpy as np
    if dataset_name == "digits":
        from PIL import Image
        from sklearn.datasets import load_digits
        dataset = load_digits()
        rng = np.random.default_rng(20260915)
        manifest = []
        for label in [0, 1, 2]:
            indices = rng.permutation(np.flatnonzero(dataset.target == label))[:60]
            for split, selected in [("train", indices[:40]), ("val", indices[40:])]:
                folder = output / "data" / split / str(label)
                folder.mkdir(parents=True, exist_ok=True)
                for index in selected:
                    pixels = (dataset.images[index] * 255 / 16).round().astype(np.uint8)
                    path = folder / f"{int(index):05d}.png"
                    Image.fromarray(pixels).convert("RGB").save(path)
                    manifest.append({"path": path.relative_to(output).as_posix(),
                                     "source_index": int(index),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        write(output / "dataset.json", manifest)
        write(output / "dataset-info.json", {"name": "sklearn bundled real handwritten digits 0, 1, 2",
              "split": "40 train and 20 disjoint validation samples per class; seed 20260915",
              "native_size": "8x8", "purpose": "Routing and results integrity, not an accuracy benchmark"})
        return
    socket.setdefaulttimeout(60)
    from torchvision.datasets import CIFAR10
    manifest = []
    for split, train, count in [("train", True, 40), ("val", False, 20)]:
        dataset = CIFAR10(str(output / "download"), train=train, download=True)
        rng = np.random.default_rng(20260915)
        for label in [0, 3, 5]:
            name = dataset.classes[label]
            indices = rng.permutation(np.flatnonzero(np.array(dataset.targets) == label))[:count]
            folder = output / "data" / split / name
            folder.mkdir(parents=True, exist_ok=True)
            for index in indices:
                image, _ = dataset[int(index)]
                path = folder / f"{int(index):05d}.png"
                image.save(path)
                manifest.append({"path": path.relative_to(output).as_posix(),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write(output / "dataset.json", manifest)
    write(output / "dataset-info.json", {"name": "CIFAR10 airplane, cat, dog", "split": "distinct original train/test splits"})


def official_factory(architecture, num_classes, in_channels, dropout, *, input_adapter="native"):
    import torch
    from torchvision import models
    from efficientnet_contract import NATIVE_INPUT, IMPLEMENTATION_VERSION
    assert in_channels in (1, 3) and input_adapter == NATIVE_INPUT
    model = getattr(models, architecture)(weights=None, num_classes=num_classes, dropout=dropout)
    if in_channels == 1:
        model.features[0][0] = torch.nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
    # Non-registering aliases retain the official state_dict without duplicate keys.
    object.__setattr__(model, "backbone", model.features)
    object.__setattr__(model, "head", model.classifier)
    model.architecture, model.in_channels, model.task = architecture, in_channels, "classify"
    model.input_adapter = input_adapter

    def checkpoint_config(self):
        return {"architecture": self.architecture, "dropout": dropout,
                "implementation_version": IMPLEMENTATION_VERSION, "input_adapter": self.input_adapter,
                "in_channels": self.in_channels, "stem_in_channels": self.features[0][0].in_channels}

    model.checkpoint_config = MethodType(checkpoint_config, model)

    def count(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_MB": total * 4 / 1024**2}

    model.get_param_count = MethodType(count, model)
    return model


def state_fingerprint(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_role(output, role, epochs, size, channels=1):
    import numpy as np
    import torch
    from core.project import ProjectManager
    from core.device_manager import get_device_manager
    from core.efficientnet_trainer import EfficientNetTrainWorker
    from webapp.storage import digest, project_view, read_json
    from webapp.worker import JobContext, train
    from core.metrics import ClassificationMetrics

    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    role_dir = output / role
    role_dir.mkdir(parents=True, exist_ok=False)
    project_path = output / "project-path.txt"
    if project_path.exists():
        project = ProjectManager.load(project_path.read_text())
    else:
        project = ProjectManager.create_new("Training comparison", "classify", str(output / "project"),
                                            sorted(p.name for p in (output / "data" / "train").iterdir() if p.is_dir()))
    project.data.root = str(output / "data")
    cfg = project.training
    cfg.training_mode = "efficientnet_finetune"
    cfg.efficientnet_model = "efficientnet_b1" if role.endswith("b1") else "efficientnet_b0"
    cfg.device, cfg.use_amp, cfg.in_channels = "cpu", False, channels
    cfg.epochs, cfg.batch_size, cfg.input_size = epochs, 8, size
    cfg.learning_rate, cfg.warmup_epochs, cfg.early_stop_patience = .001, 0, 0
    cfg.selection_metric, cfg.class_weights = "accuracy", "none"
    cfg.augmentation.horizontal_flip = 0
    cfg.augmentation.vertical_flip = 0
    cfg.augmentation.rotation = 0
    cfg.augmentation.color_jitter = 0
    cfg.augmentation.mixup_alpha = 0
    project.model.pretrained_weights, project.model.freeze_backbone = "", False
    ProjectManager.save(project)
    path = ProjectManager.get_active_filepath(project)
    project_path.write_text(path)
    previous = [digest(r.checkpoint_path) for r in project.runs]
    context = JobContext(role_dir, monitor_parent=False)
    observed = {}
    original_initialize = EfficientNetTrainWorker._initialize_model

    def initialize(worker, model, weights_path, task):
        result = original_initialize(worker, model, weights_path, task)
        observed.update(model_class=type(model).__module__ + "." + type(model).__name__,
                        parameter_count=sum(p.numel() for p in model.parameters()),
                        initialized_state_sha256=state_fingerprint(model),
                        weight_provenance=worker._weight_provenance)
        return result

    factory = patch("core.efficientnet_trainer.EfficientNet", official_factory) if role.startswith("official") else nullcontext()
    with factory, patch.object(EfficientNetTrainWorker, "_initialize_model", initialize), \
            patch.object(get_device_manager(), "get_optimal_num_workers", return_value=0):
        result = train(context, {"project": project_view(project), "project_digest": digest(path)})
    assert result["status"] == "completed", result
    saved = ProjectManager.load(path)
    assert len(saved.runs) == len(previous) + 1
    assert [digest(r.checkpoint_path) for r in saved.runs[:-1]] == previous
    record = saved.runs[-1]
    assert record.config_snapshot["job_id"] == role_dir.name
    assert record.config_snapshot["checkpoint_sha256"] == digest(record.checkpoint_path)
    events = [json.loads(line) for line in (role_dir / "events.jsonl").read_text().splitlines()]
    emitted = [e["args"][0] for e in events if e["event"] == "eval_finished"][-1]
    assert emitted["confusion_matrix"] == record.eval_results["confusion_matrix"]
    job_record = read_json(role_dir / "project_result.json")["runs"][-1]
    assert job_record["eval_results"] == record.eval_results
    assert record.epochs_done == epochs
    checkpoint = torch.load(record.checkpoint_path, map_location="cpu", weights_only=False)
    paths = sorted((output / "data" / "val").glob("*/*.png"))
    targets = np.array([saved.data.class_names.index(p.parent.name) for p in paths])
    from opencv_preprocess import read_image
    from dataset import get_classification_transforms
    from efficientnet import EfficientNet
    assert checkpoint["engine"] == "efficientnet"
    assert checkpoint["model_config"]["architecture"] == cfg.efficientnet_model
    constructor = official_factory if role.startswith("official") else EfficientNet
    network = constructor(cfg.efficientnet_model, 3, channels, .2).eval()
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    transform = get_classification_transforms((size, size), is_train=False, in_channels=channels)
    batches = []
    with torch.no_grad():
        for offset in range(0, len(paths), 8):
            images = torch.stack([transform(read_image(p, channels)) for p in paths[offset:offset + 8]])
            batches.append(network(images).softmax(1).numpy())
    probabilities = np.concatenate(batches)
    fingerprint = state_fingerprint(network)
    predictions = probabilities.argmax(1)
    # Independent NumPy reconstruction, then comparison to persisted GUI metrics.
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (targets, predictions), 1)
    meter = ClassificationMetrics(3, saved.data.class_names)
    meter.update(predictions, targets)
    measured = meter.compute()
    cm_equal = cm.tolist() == record.eval_results["confusion_matrix"]
    checked_metrics = {}
    for key, value in measured.items():
        if isinstance(value, (int, float)) and key in record.eval_results:
            expected = record.eval_results[key]
            if isinstance(expected, (int, float)):
                checked_metrics[key] = bool(np.isclose(value, expected, rtol=1e-6, atol=1e-7, equal_nan=True))
    report = {"role": role, "epochs": epochs, "input_size": size, "train_images": 120,
              "input_channels": channels, "model_config": network.checkpoint_config(),
              "val_images": len(paths), "observed": observed, "run_id": record.run_id,
              "checkpoint_sha256": digest(record.checkpoint_path), "trained_state_sha256": fingerprint,
              "best_epoch": record.best_epoch, "training_mode": record.config_snapshot["training"]["training_mode"],
              "history": record.metrics_history, "saved_metrics": record.eval_results,
              "recomputed_confusion_matrix": cm.tolist(), "saved_cm_matches_predictions": cm_equal,
              "saved_metrics_match_predictions": checked_metrics,
              "probabilities": probabilities.tolist(), "predictions": predictions.tolist(),
              "targets": targets.tolist(), "files": [p.relative_to(output).as_posix() for p in paths],
              "recomputed_accuracy": float(measured["accuracy"])}
    write(role_dir / "report.json", report)
    print("TRAINING_COMPARISON_ROLE=" + json.dumps(report), flush=True)
    assert cm_equal, "Saved confusion matrix does not match predictions from the saved checkpoint"
    assert checked_metrics and all(checked_metrics.values()), checked_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="training-comparison")
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--dataset", choices=["cifar10", "digits"], default="cifar10")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.role:
        run_role(output, args.role, args.epochs, args.input_size, args.channels)
        return
    if not args.summarize_only:
        prepare(output, args.dataset)
    if args.prepare_only:
        return
    failures, reports = [], {}
    for role in ROLES:
        if not args.summarize_only:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                                      "--role", role, "--epochs", str(args.epochs), "--input-size", str(args.input_size),
                                      "--channels", str(args.channels)])
            if process.returncode:
                failures.append(role)
        path = output / role / "report.json"
        if path.exists():
            reports[role] = json.loads(path.read_text())
            if not reports[role]["saved_cm_matches_predictions"]:
                failures.append(role + ":saved_confusion_matrix")
            checks = reports[role].get("saved_metrics_match_predictions", {})
            if not checks or not all(checks.values()):
                failures.append(role + ":saved_metrics")
        elif role not in failures:
            failures.append(role + ":missing_report")
    import numpy as np
    comparisons = []
    for left, right in [("custom_b0", "official_b0"), ("custom_b1", "official_b1")]:
        if left not in reports or right not in reports:
            continue
        a, b = reports[left], reports[right]
        if a.get("input_channels", 3) != b.get("input_channels", 3):
            failures.append(left + ":channel_mismatch:" + right)
            continue
        error = float(np.max(np.abs(np.array(a["probabilities"]) - np.array(b["probabilities"]))))
        comparisons.append({"left": left, "right": right, "max_probability_difference": error,
                            "different_predictions": int(np.count_nonzero(np.array(a["predictions"]) != b["predictions"])),
                            "same_confusion_matrix": a["saved_metrics"]["confusion_matrix"] == b["saved_metrics"]["confusion_matrix"],
                            "same_checkpoint_bytes": a["checkpoint_sha256"] == b["checkpoint_sha256"]})
    summary = {"dataset": json.loads((output / "dataset-info.json").read_text()),
               "purpose": "Integration diagnosis; not a production accuracy or speed benchmark",
               "device": "cpu", "failures": failures, "comparisons": comparisons,
               "results": [{k: v for k, v in report.items() if k not in {"probabilities", "targets", "files", "predictions"}}
                           for report in reports.values()]}
    write(output / "summary.json", summary)
    print("TRAINING_COMPARISON_SUMMARY=" + json.dumps(summary), flush=True)
    if failures:
        raise SystemExit("Comparison failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
