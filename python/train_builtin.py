"""Train the weight-free built-in adapters and write deployable checkpoints.

This is deliberately a small, dependency-stable worker entry point.  It uses
the repository's existing image/mask loaders and emits the same metadata that
``export_onnx.py`` consumes, so a trained adapter never needs a second model
definition during export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from builtin_models import build_builtin_model, get_builtin_spec, make_builtin_checkpoint
from dataset import create_classification_loaders, create_segmentation_loaders


def _classification_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * labels.shape[0]
            correct += int(logits.argmax(dim=1).eq(labels).sum())
            samples += labels.shape[0]
    if samples == 0:
        raise ValueError("classification loader contains no samples")
    return total_loss / samples, correct / samples


def _segmentation_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    samples = 0
    intersection = None
    union = None
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
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
    return total_loss / samples, float(iou.mean())


def save_builtin_checkpoint(path: str | Path, model: nn.Module, *, model_id: str,
                            num_classes: int, input_size: int | Iterable[int],
                            in_channels: int, class_names: Iterable[str], epoch: int,
                            optimizer: torch.optim.Optimizer | None = None,
                            metric: float | None = None) -> None:
    model.eval()
    state = make_builtin_checkpoint(model_id, model, num_classes=num_classes,
                                    input_size=input_size, in_channels=in_channels,
                                    class_names=class_names)
    state["epoch"] = int(epoch)
    state["metric"] = None if metric is None else float(metric)
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)


def train_builtin(model_id: str, data_root: str | Path, *, num_classes: int = 0,
                  input_size: int | tuple[int, int] | None = None, in_channels: int = 3,
                  epochs: int = 1, batch_size: int = 4, learning_rate: float = 1e-3,
                  output_dir: str | Path = "runs/builtin", device: str = "cpu",
                  resume: str | Path | None = None, log=print,
                  should_stop=lambda: False) -> Path:
    spec = get_builtin_spec(model_id)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
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
    if spec.task == "classify":
        train_loader, val_loader, class_names = create_classification_loaders(
            str(data_root), input_size=input_size, batch_size=batch_size, in_channels=in_channels)
        num_classes = len(class_names) if num_classes == 0 else num_classes
        if num_classes != len(class_names):
            raise ValueError("num_classes does not match classification folders")
    else:
        if num_classes < 1:
            raise ValueError("num_classes is required for segmentation")
        train_loader, val_loader = create_segmentation_loaders(
            str(data_root), input_size=input_size, batch_size=batch_size,
            in_channels=in_channels, num_classes=num_classes)
        class_names = [str(index) for index in range(num_classes)]
    model = build_builtin_model(model_id, num_classes, in_channels).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    start_epoch = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("model_id") != model_id:
            raise ValueError("resume checkpoint model_id does not match the requested adapter")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
    destination = Path(output_dir)
    best_metric = float("-inf")
    completed_epochs = start_epoch
    for epoch in range(start_epoch, epochs):
        if should_stop():
            break
        if spec.task == "classify":
            train_loss, train_metric = _classification_epoch(model, train_loader, criterion, optimizer, target)
            val_loss, val_metric = _classification_epoch(model, val_loader, criterion, None, target)
        else:
            train_loss, train_metric = _segmentation_epoch(model, train_loader, criterion, optimizer, target)
            val_loss, val_metric = _segmentation_epoch(model, val_loader, criterion, None, target)
        if not all(torch.isfinite(torch.tensor(value)) for value in (train_loss, train_metric, val_loss, val_metric)):
            raise ValueError("training produced a non-finite metric")
        log(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} metric={val_metric:.6f}")
        completed_epochs = epoch + 1
        log({"event": "epoch_finished", "epoch": completed_epochs,
             "train_loss": train_loss, "val_loss": val_loss,
             "metric": val_metric, "total_epochs": epochs})
        save_builtin_checkpoint(destination / "last.pt", model, model_id=model_id,
                                num_classes=num_classes, input_size=input_size,
                                in_channels=in_channels, class_names=class_names,
                                epoch=epoch, optimizer=optimizer, metric=val_metric)
        if val_metric > best_metric:
            best_metric = val_metric
            save_builtin_checkpoint(destination / "best.pt", model, model_id=model_id,
                                    num_classes=num_classes, input_size=input_size,
                                    in_channels=in_channels, class_names=class_names,
                                    epoch=epoch, optimizer=optimizer, metric=val_metric)
    if completed_epochs == start_epoch and start_epoch >= epochs:
        raise ValueError("resume checkpoint already reached the requested epochs")
    (destination / "training.json").write_text(json.dumps({"model_id": model_id,
        "task": spec.task, "epochs": epochs, "input_size": list(input_size),
        "in_channels": in_channels, "best_metric": best_metric,
        "completed_epochs": completed_epochs}, ensure_ascii=False, indent=2), encoding="utf-8")
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
    parser.add_argument("--resume")
    args = parser.parse_args()
    train_builtin(args.model, args.data_root, num_classes=args.num_classes,
                  input_size=args.input_size or None, in_channels=args.in_channels,
                  epochs=args.epochs, batch_size=args.batch_size,
                  learning_rate=args.learning_rate, output_dir=args.output_dir,
                  device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
