"""Train built-in adapters and write deployable checkpoints.

This is deliberately a small, dependency-stable worker entry point.  It uses
the repository's existing image/mask loaders and emits the same metadata that
``export_onnx.py`` consumes, so a trained adapter never needs a second model
definition during export.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Iterable

import torch
from torch import nn

from builtin_models import build_builtin_model, get_builtin_spec, make_builtin_checkpoint
from dataset import create_classification_loaders, create_segmentation_loaders


def _classification_epoch(model, loader, criterion, optimizer=None, device="cpu", metrics_out=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    samples = 0
    confusion = None
    non_blocking = torch.device(device).type == "cuda"
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * labels.shape[0]
            correct += int(logits.argmax(dim=1).eq(labels).sum())
            if metrics_out is not None:
                classes = logits.shape[1]
                counts = torch.bincount((labels * classes + logits.argmax(1)).detach().cpu(),
                                        minlength=classes * classes).reshape(classes, classes)
                confusion = counts if confusion is None else confusion + counts
            samples += labels.shape[0]
    if samples == 0:
        raise ValueError("classification loader contains no samples")
    if metrics_out is not None:
        counts = confusion.double()
        tp = counts.diag()
        precision = tp / counts.sum(0).clamp(min=1)
        recall = tp / counts.sum(1).clamp(min=1)
        f1 = 2 * tp / (counts.sum(0) + counts.sum(1)).clamp(min=1)
        metrics_out.update(accuracy=correct / samples, precision_macro=float(precision.mean()),
                           recall_macro=float(recall.mean()),
                           f1_macro=float(f1.mean()), val_loss=total_loss / samples)
    return total_loss / samples, correct / samples


def _segmentation_epoch(model, loader, criterion, optimizer=None, device="cpu", metrics_out=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    samples = 0
    intersection = None
    union = None
    non_blocking = torch.device(device).type == "cuda"
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, masks in loader:
            images = images.to(device, non_blocking=non_blocking)
            masks = masks.to(device, dtype=torch.long, non_blocking=non_blocking)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * images.shape[0]
            samples += images.shape[0]
            predicted = logits.argmax(dim=1)
            valid = masks != 255
            classes = logits.shape[1]
            if intersection is None:
                intersection = torch.zeros(classes, dtype=torch.float64)
                union = torch.zeros(classes, dtype=torch.float64)
            for cls in range(classes):
                truth = valid & (masks == cls)
                guess = valid & (predicted == cls)
                intersection[cls] += (truth & guess).sum().item()
                union[cls] += (truth | guess).sum().item()
    if samples == 0 or intersection is None:
        raise ValueError("segmentation loader contains no samples")
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    if metrics_out is not None:
        dice_denominator = union + intersection
        dice = torch.where(dice_denominator > 0, 2 * intersection / dice_denominator, torch.ones_like(union))
        metrics_out.update(mIoU=float(iou.mean()), dice_score=float(dice.mean()), val_loss=total_loss / samples)
    return total_loss / samples, float(iou.mean())


def save_builtin_checkpoint(path: str | Path, model: nn.Module, *, model_id: str,
                            num_classes: int, input_size: int | Iterable[int],
                            in_channels: int, class_names: Iterable[str], epoch: int,
                            optimizer: torch.optim.Optimizer | None = None,
                            scheduler=None, metric: float | None = None,
                            training_config: dict | None = None,
                            metrics_history: list[dict] | None = None) -> None:
    model.eval()
    state = make_builtin_checkpoint(model_id, model, num_classes=num_classes,
                                    input_size=input_size, in_channels=in_channels,
                                    class_names=class_names)
    state["epoch"] = int(epoch)
    state["metric"] = None if metric is None else float(metric)
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    state["training_config"] = dict(training_config or {})
    state["metrics_history"] = list(metrics_history or [])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)


def train_builtin(model_id: str, data_root: str | Path, *, num_classes: int = 0,
                  input_size: int | tuple[int, int] | None = None, in_channels: int = 3,
                  epochs: int = 1, batch_size: int = 4, learning_rate: float = 1e-3,
                  weight_decay: float = 5e-4, optimizer_name: str = "adamw",
                  scheduler_name: str = "cosine", warmup_epochs: int = 0,
                  early_stop_patience: int = 0, label_smoothing: float = 0.0,
                  horizontal_flip: float = 0.5, rotation: float = 15.0,
                  color_jitter: float = 0.2, val_split: float = 0.2,
                  num_workers: int | None = None, freeze_backbone: bool = False,
                  backbone_lr_mult: float = 0.1,
                  output_dir: str | Path = "runs/builtin", device: str = "cpu",
                  resume: str | Path | None = None,
                  initial_weights: str | Path | None = None, pretrained: bool = False, log=print,
                  selection_metric: str = "engine_default",
                  should_stop=lambda: False) -> Path:
    spec = get_builtin_spec(model_id)
    gui_path = str(Path(__file__).resolve().parents[1] / "gui")
    if gui_path not in sys.path:
        sys.path.insert(0, gui_path)
    from core.model_selection import selection_policy
    from core.training_time import TrainingClock, format_hms, log_total_time
    policy = selection_policy(SimpleNamespace(selection_metric=selection_metric), "builtin", spec.task)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    optimizer_name = optimizer_name.lower()
    scheduler_name = scheduler_name.lower()
    if optimizer_name not in {"adamw", "adam", "sgd"}:
        raise ValueError("optimizer_name must be adamw, adam or sgd")
    if scheduler_name not in {"cosine", "step", "none"}:
        raise ValueError("scheduler_name must be cosine, step or none")
    if not 0 <= horizontal_flip <= 1 or rotation < 0 or not 0 <= color_jitter <= 1:
        raise ValueError("augmentation values are outside their supported ranges")
    if not 0 < val_split < 1 or not 0 <= label_smoothing < 1:
        raise ValueError("validation or label smoothing settings are invalid")
    if warmup_epochs < 0 or early_stop_patience < 0 or not 0 < backbone_lr_mult <= 1:
        raise ValueError("scheduler, early-stop or backbone LR settings are invalid")
    if resume is not None and initial_weights is not None:
        raise ValueError("resume and initial_weights are mutually exclusive")
    if pretrained and (resume is not None or initial_weights is not None):
        raise ValueError("ImageNet, resume and initial_weights are mutually exclusive")
    if input_size is None:
        input_size = spec.default_size
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    input_size = tuple(int(value) for value in input_size)
    if len(input_size) != 2 or min(input_size) < 32:
        raise ValueError("input_size must contain two values >= 32")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    target = torch.device(device)
    if num_workers is None:
        cores = os.cpu_count() or 1
        num_workers = max(2, min(cores // 2, 8)) if target.type == "cuda" else 0
    if num_workers < 0:
        raise ValueError("num_workers must be zero or greater")
    pin_memory = target.type == "cuda"
    log(f"data loader: workers={num_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={num_workers > 0}, prefetch_factor={2 if num_workers else 0}")
    if spec.task == "classify":
        train_loader, val_loader, class_names = create_classification_loaders(
            str(data_root), input_size=input_size, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory,
            in_channels=in_channels, val_split=val_split,
            flip_prob=horizontal_flip, rotation=rotation, color_jitter=color_jitter)
        num_classes = len(class_names) if num_classes == 0 else num_classes
        if num_classes != len(class_names):
            raise ValueError("num_classes does not match classification folders")
    else:
        if num_classes < 1:
            raise ValueError("num_classes is required for segmentation")
        train_loader, val_loader = create_segmentation_loaders(
            str(data_root), input_size=input_size, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory,
            in_channels=in_channels, num_classes=num_classes,
            flip_prob=horizontal_flip, rotation=rotation, color_jitter=color_jitter)
        class_names = [str(index) for index in range(num_classes)]
    initial_checkpoint = None
    backbone_weights = None
    if initial_weights is not None:
        payload = torch.load(initial_weights, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            if payload.get("model_id") != model_id:
                raise ValueError("initial checkpoint model_id does not match the requested adapter")
            initial_checkpoint = payload
        else:
            backbone_weights = initial_weights
    model = build_builtin_model(model_id, num_classes, in_channels,
                                pretrained=pretrained, backbone_weights=backbone_weights)
    if initial_checkpoint is not None:
        # Class order matters even when the number of classes is unchanged.
        source = dict(initial_checkpoint["model_state_dict"])
        head = {"resnet18": "fc.", "resnet50": "fc.",
                "convnext_v1_tiny": "classifier.2.",
                "deeplabv3plus_resnet34": "decoder.6.", "unet_resnet18": "head."}[model_id]
        if list(initial_checkpoint.get("class_names", [])) != class_names:
            source = {key: value for key, value in source.items() if not key.startswith(head)}
            missing, unexpected = model.load_state_dict(source, strict=False)
            if unexpected or any(not key.startswith(head) for key in missing):
                raise ValueError("initial checkpoint backbone does not match the requested adapter")
        else:
            model.load_state_dict(source, strict=True)
        model.weight_provenance = {"source": "local_checkpoint", "file": Path(initial_weights).name}
    model = model.to(target)
    log(f"가중치 로드 완료: {model_id} / {model.weight_provenance}")
    head_prefixes = {
        "resnet18": ("fc.",), "resnet50": ("fc.",),
        "convnext_v1_tiny": ("classifier.",),
        "deeplabv3plus_resnet34": ("aspp.", "low_projection.", "decoder."),
        "unet_resnet18": ("dec", "head."),
    }[model_id]
    head_parameters, backbone_parameters = [], []
    for name, parameter in model.named_parameters():
        is_head = any(name.startswith(prefix) for prefix in head_prefixes)
        if freeze_backbone and not is_head:
            parameter.requires_grad_(False)
        (head_parameters if is_head else backbone_parameters).append(parameter)
    parameter_groups = [{"params": head_parameters, "lr": learning_rate}]
    trainable_backbone = [parameter for parameter in backbone_parameters if parameter.requires_grad]
    if trainable_backbone:
        parameter_groups.append({"params": trainable_backbone,
                                 "lr": learning_rate * backbone_lr_mult})
    optimizer_types = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
                       "sgd": torch.optim.SGD}
    optimizer_kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
    if optimizer_name == "sgd":
        optimizer_kwargs["momentum"] = 0.9
    optimizer = optimizer_types[optimizer_name](parameter_groups, **optimizer_kwargs)
    if scheduler_name == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
    elif scheduler_name == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=0.1
        )
    else:
        main_scheduler = None
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / max(2, warmup_epochs), total_iters=warmup_epochs
        )
        scheduler = (warmup if main_scheduler is None else
                     torch.optim.lr_scheduler.SequentialLR(
                         optimizer, [warmup, main_scheduler], milestones=[warmup_epochs]
                     ))
    else:
        scheduler = main_scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=label_smoothing)
    training_config = {
        "optimizer": optimizer_name, "scheduler": scheduler_name,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs, "early_stop_patience": early_stop_patience,
        "label_smoothing": label_smoothing, "freeze_backbone": freeze_backbone,
        "backbone_lr_mult": backbone_lr_mult, "val_split": val_split,
        "num_workers": num_workers,
        "pretrained": pretrained, "weight_provenance": model.weight_provenance,
        "selection_metric": policy.metric, "selection_direction": policy.direction,
        "augmentation": {"horizontal_flip": horizontal_flip, "rotation": rotation,
                         "color_jitter": color_jitter},
    }
    start_epoch = 0
    history: list[dict] = []
    best_metric = float("inf") if policy.direction == "min" else float("-inf")
    best_epoch = 0
    resume_best_checkpoint = None
    if resume is not None:
        resume_path = Path(resume)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_id") != model_id:
            raise ValueError("resume checkpoint model_id does not match the requested adapter")
        saved_metric = checkpoint.get("training_config", {}).get("selection_metric", "engine_default")
        saved_policy = selection_policy(SimpleNamespace(selection_metric=saved_metric), "builtin", spec.task)
        if saved_policy.metric != policy.metric:
            raise ValueError("resume Best 기준이 다릅니다. 같은 기준으로 재개하거나 로컬 가중치로 새 학습을 시작하세요.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.weight_provenance = dict(checkpoint.get("weight_provenance", {}))
        training_config["weight_provenance"] = model.weight_provenance
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        history = list(checkpoint.get("metrics_history") or [])
        previous_best_path = resume_path.parent / "best.pt"
        if previous_best_path.is_file():
            resume_best_checkpoint = torch.load(
                previous_best_path, map_location="cpu", weights_only=False
            )
            if resume_best_checkpoint.get("model_id") != model_id:
                raise ValueError("best checkpoint model_id does not match the resume adapter")
        else:
            resume_best_checkpoint = checkpoint
        if resume_best_checkpoint.get("metric") is not None:
            best_metric = float(resume_best_checkpoint["metric"])
            best_epoch = int(resume_best_checkpoint.get("epoch", start_epoch - 1)) + 1
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    training_clock = TrainingClock()
    if resume_best_checkpoint is not None:
        torch.save(resume_best_checkpoint, destination / "best.pt")
    completed_epochs = start_epoch
    epochs_without_improvement = max(0, start_epoch - best_epoch) if best_epoch else 0
    for epoch in range(start_epoch, epochs):
        if should_stop():
            break
        training_clock.start_epoch()
        validation_metrics = {}
        if spec.task == "classify":
            train_loss, train_metric = _classification_epoch(model, train_loader, criterion, optimizer, target)
            val_loss, val_metric = _classification_epoch(model, val_loader, criterion, None, target, validation_metrics)
        else:
            train_loss, train_metric = _segmentation_epoch(model, train_loader, criterion, optimizer, target)
            val_loss, val_metric = _segmentation_epoch(model, val_loader, criterion, None, target, validation_metrics)
        selected_value = policy.value(validation_metrics)
        if not all(torch.isfinite(torch.tensor(value)) for value in (train_loss, train_metric, val_loss, val_metric)):
            raise ValueError("training produced a non-finite metric")
        epoch_time_sec, elapsed_time_sec = training_clock.end_epoch()
        event_metrics = dict(validation_metrics)
        event_metrics.update(epoch_time_sec=epoch_time_sec, elapsed_time_sec=elapsed_time_sec)
        log(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} metric={val_metric:.6f}")
        log(f"  Epoch {epoch + 1} 시간 | 소요 {format_hms(epoch_time_sec)} | "
            f"누적 {format_hms(elapsed_time_sec)}")
        completed_epochs = epoch + 1
        log({"event": "epoch_finished", "epoch": completed_epochs,
             "train_loss": train_loss, "val_loss": val_loss,
             "metric": val_metric, "total_epochs": epochs, "metrics": event_metrics,
             "selected_metric": policy.metric, "selected_value": selected_value})
        history.append({"epoch": completed_epochs, "train_loss": train_loss,
                        "train_metric": train_metric, "val_loss": val_loss,
                        "val_metric": val_metric,
                        "metrics": validation_metrics, "selected_value": selected_value,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "epoch_time_sec": epoch_time_sec,
                        "elapsed_time_sec": elapsed_time_sec})
        if scheduler is not None:
            scheduler.step()
        save_builtin_checkpoint(destination / "last.pt", model, model_id=model_id,
                                num_classes=num_classes, input_size=input_size,
                                in_channels=in_channels, class_names=class_names,
                                epoch=epoch, optimizer=optimizer, scheduler=scheduler,
                                metric=selected_value, training_config=training_config,
                                metrics_history=history)
        improved = selected_value < best_metric if policy.direction == "min" else selected_value > best_metric
        if improved:
            best_metric = selected_value
            best_epoch = completed_epochs
            epochs_without_improvement = 0
            save_builtin_checkpoint(destination / "best.pt", model, model_id=model_id,
                                    num_classes=num_classes, input_size=input_size,
                                    in_channels=in_channels, class_names=class_names,
                                    epoch=epoch, optimizer=optimizer, scheduler=scheduler,
                                    metric=selected_value, training_config=training_config,
                                    metrics_history=history)
            log({"event": "best_epoch_updated", "epoch": completed_epochs,
                 "train_loss": train_loss, "val_loss": val_loss, "metrics": validation_metrics})
        else:
            epochs_without_improvement += 1
        if early_stop_patience and epochs_without_improvement >= early_stop_patience:
            log(f"early stopping at epoch {completed_epochs}")
            break
    if completed_epochs == start_epoch and start_epoch >= epochs:
        raise ValueError("resume checkpoint already reached the requested epochs")
    log_total_time(log, training_clock.elapsed())
    (destination / "training.json").write_text(json.dumps({"model_id": model_id,
        "task": spec.task, "epochs": epochs, "input_size": list(input_size),
        "in_channels": in_channels,
        "best_metric": best_metric if torch.isfinite(torch.tensor(best_metric)) else None,
        "best_metric_name": policy.metric,
        "best_epoch": best_epoch, "completed_epochs": completed_epochs,
        "metrics_history": history, "training_config": training_config,
        "cancelled": bool(should_stop())}, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination / "best.pt"


def main() -> None:
    specs = tuple(get_builtin_spec(model_id) for model_id in (
        "resnet18", "resnet50", "convnext_v1_tiny", "deeplabv3plus_resnet34", "unet_resnet18"))
    parser = argparse.ArgumentParser(description="Train a built-in Deep Vision Studio adapter")
    parser.add_argument("--model", choices=[spec.model_id for spec in specs], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--input_size", type=int, default=0)
    parser.add_argument("--in_channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="runs/builtin")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume")
    parser.add_argument("--pretrained", action="store_true", help="Load the selected ImageNet backbone")
    parser.add_argument("--initial_weights", help="Local torchvision backbone or matching Studio checkpoint")
    parser.add_argument("--selection_metric", default="engine_default")
    args = parser.parse_args()
    train_builtin(args.model, args.data_root, num_classes=args.num_classes,
                  input_size=args.input_size or None, in_channels=args.in_channels,
                  epochs=args.epochs, batch_size=args.batch_size,
                  learning_rate=args.learning_rate, output_dir=args.output_dir,
                  device=args.device, num_workers=args.num_workers,
                  resume=args.resume, pretrained=args.pretrained,
                  initial_weights=args.initial_weights, selection_metric=args.selection_metric)


if __name__ == "__main__":
    main()
