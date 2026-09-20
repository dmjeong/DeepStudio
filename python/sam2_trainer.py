"""Windows-native SAM2 prompt-mask fine-tuning.

Deep Vision Studio's segmentation editor stores semantic class-index masks.
SAM2 is an interactive object-mask model, so this worker deterministically
turns every non-background semantic region into a binary target and supplies a
positive point from that region.  It fine-tunes SAM2's prompt encoder and mask
decoder while the Hiera image encoder stays frozen.  The generated checkpoint
contains the compatible Hiera ID and changed-module state, which the ONNX
exporter loads on top of the bundled official checkpoint.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from dataset import create_segmentation_loaders
from sam2_assets import load_sam2_checkpoint


def _prompt_targets(masks: torch.Tensor, *, class_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return binary masks, SAM pixel coordinates, and valid-batch indices.

    Pixels ``0`` and ``255`` are background and ignore respectively.  An image
    can contain several semantic classes; the deterministic offset rotates the
    selected class each epoch while preserving SAM2's one-object contract.
    """
    targets, points, valid = [], [], []
    for index, mask in enumerate(masks):
        classes = torch.unique(mask[(mask > 0) & (mask != 255)])
        if not len(classes):
            continue
        target = mask == classes[(class_offset + index) % len(classes)]
        coordinates = torch.nonzero(target, as_tuple=False)
        # ``coordinates`` is y,x.  The middle foreground pixel is guaranteed
        # positive and avoids a centroid falling into a hollow object.
        y, x = coordinates[len(coordinates) // 2]
        targets.append(target)
        points.append(torch.stack((x, y)).to(dtype=torch.float32))
        valid.append(index)
    if not valid:
        empty = masks.new_empty((0, *masks.shape[-2:]), dtype=torch.bool)
        return empty, masks.new_empty((0, 1, 2), dtype=torch.float32), masks.new_empty((0,), dtype=torch.long)
    return (torch.stack(targets), torch.stack(points).unsqueeze(1),
            torch.tensor(valid, dtype=torch.long, device=masks.device))


def _sam2_logits(model, images: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    """Run the official image path used by the paired ONNX exporter."""
    features = model.forward_image(images)["backbone_fpn"]
    image_embeddings = features[2]
    if model.directly_add_no_mem_embed:
        image_embeddings = image_embeddings + model.no_mem_embed.reshape(1, -1, 1, 1)
    point_labels = torch.ones(point_coords.shape[:2], device=images.device, dtype=torch.int32)
    sparse, dense = model.sam_prompt_encoder(
        points=(point_coords, point_labels), boxes=None, masks=None)
    logits, _, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=[features[0], features[1]],
    )
    return logits[:, 0]


def _loss_and_scores(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    resized = F.interpolate(targets.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest")[:, 0]
    bce = F.binary_cross_entropy_with_logits(logits, resized)
    probabilities = logits.sigmoid()
    intersection = (probabilities * resized).sum(dim=(1, 2))
    denominator = probabilities.sum(dim=(1, 2)) + resized.sum(dim=(1, 2))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    prediction = probabilities >= 0.5
    truth = resized >= 0.5
    discrete_intersection = (prediction & truth).sum(dim=(1, 2)).float()
    union = (prediction | truth).sum(dim=(1, 2)).float()
    dice = ((2.0 * discrete_intersection + 1.0) /
            (prediction.sum(dim=(1, 2)).float() + truth.sum(dim=(1, 2)).float() + 1.0)).mean()
    iou = ((discrete_intersection + 1.0) / (union + 1.0)).mean()
    return bce + dice_loss, float(dice.detach().cpu()), float(iou.detach().cpu())


def _checkpoint_payload(model, optimizer, *, model_id: str, epoch: int, best_dice: float,
                        history: list[dict], source_checkpoint: str | None) -> dict:
    return {
        "format_version": 1,
        "type": "sam2_finetune",
        "backend": "sam2",
        "task": "segment",
        "model_id": model_id,
        "training_contract": "semantic-mask-to-positive-point-object-mask-v1",
        "frozen_modules": ["image_encoder"],
        "epoch": epoch,
        "best_metric": best_dice,
        "metric_name": "prompt_dice",
        "metrics_history": history,
        "source_checkpoint": source_checkpoint or "bundled_official_sam2.1",
        # Keep only the modules this worker changes.  The immutable Hiera image
        # encoder remains the verified bundled asset, so a fine-tune result does
        # not duplicate hundreds of MB of pretrained parameters.
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
            if name.startswith(("sam_prompt_encoder.", "sam_mask_decoder."))
        },
        "optimizer_state_dict": optimizer.state_dict(),
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_sam2(model_id: str, data_root: str | Path, *, output_dir: str | Path,
               epochs: int, batch_size: int, learning_rate: float, weight_decay: float,
               device: str, use_amp: bool, input_size: int = 1024,
               initial_checkpoint: str | Path | None = None,
               horizontal_flip: float = 0.5, rotation: float = 0.0,
               color_jitter: float = 0.0, emit: Callable[[dict], None] | None = None,
               should_stop: Callable[[], bool] | None = None) -> dict:
    """Fine-tune one shipped SAM2.1 Hiera variant and return run artifacts."""
    if int(input_size) != 1024:
        # SAM2.1 Hiera checkpoints and the official ONNX ABI use a fixed 1024
        # image side.  Do not silently resize to a different model contract.
        raise ValueError("SAM2 학습 입력 크기는 1024여야 합니다.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("SAM2 epochs와 batch_size는 1 이상이어야 합니다.")
    emit = emit or (lambda _event: None)
    should_stop = should_stop or (lambda: False)
    torch_device = torch.device(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = load_sam2_checkpoint(model_id, checkpoint_path=initial_checkpoint, device=str(torch_device))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.sam_prompt_encoder, model.sam_mask_decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.train()
    model.image_encoder.eval()
    train_loader, val_loader = create_segmentation_loaders(
        str(data_root), input_size=(1024, 1024), batch_size=batch_size, in_channels=3,
        flip_prob=horizontal_flip, rotation=rotation, color_jitter=color_jitter)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, weight_decay=weight_decay)
    amp_enabled = bool(use_amp and torch_device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_dice, best_epoch, history = -1.0, 0, []
    best_path, last_path = output / "best.pt", output / "last.pt"
    started = time.perf_counter()
    emit({"event": "training_started", "family": "SAM2.1", "size": model_id,
          "epoch": 1, "total_epochs": epochs, "input_size": 1024,
          "frozen_modules": ["image_encoder"]})

    for epoch in range(1, epochs + 1):
        if should_stop():
            break
        epoch_started = time.perf_counter()
        model.train()
        model.image_encoder.eval()
        train_loss, train_dice, train_count = 0.0, 0.0, 0
        for images, masks in train_loader:
            if should_stop():
                break
            targets, coords, indices = _prompt_targets(masks, class_offset=epoch - 1)
            if not len(indices):
                continue
            images = images.index_select(0, indices).to(torch_device, non_blocking=True)
            targets, coords = targets.to(torch_device), coords.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            with (torch.autocast(device_type="cuda", enabled=True) if amp_enabled else nullcontext()):
                logits = _sam2_logits(model, images, coords)
                loss, dice, _ = _loss_and_scores(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * len(indices)
            train_dice += dice * len(indices)
            train_count += len(indices)
        if not train_count:
            raise RuntimeError("SAM2 학습 마스크에 전경 클래스(1 이상)가 없습니다. 0은 배경, 255는 ignore여야 합니다.")

        model.eval()
        val_loss, val_dice, val_iou, val_count = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                targets, coords, indices = _prompt_targets(masks)
                if not len(indices):
                    continue
                images = images.index_select(0, indices).to(torch_device, non_blocking=True)
                targets, coords = targets.to(torch_device), coords.to(torch_device)
                logits = _sam2_logits(model, images, coords)
                loss, dice, iou = _loss_and_scores(logits, targets)
                val_loss += float(loss.detach().cpu()) * len(indices)
                val_dice += dice * len(indices)
                val_iou += iou * len(indices)
                val_count += len(indices)
        if not val_count:
            raise RuntimeError("SAM2 검증 마스크에 전경 클래스(1 이상)가 없습니다. images/val과 masks/val을 확인하세요.")
        event = {
            "event": "epoch_finished", "epoch": epoch, "total_epochs": epochs,
            "train_loss": train_loss / train_count, "val_loss": val_loss / val_count,
            "metric_name": "prompt_dice", "metric": val_dice / val_count,
            "metrics": {"prompt_dice": val_dice / val_count, "prompt_iou": val_iou / val_count},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.perf_counter() - epoch_started,
            "elapsed_time_sec": time.perf_counter() - started,
        }
        event["is_best"] = event["metric"] > best_dice
        if event["is_best"]:
            best_dice, best_epoch = event["metric"], epoch
        event["best_epoch"] = best_epoch
        history.append(event)
        payload = _checkpoint_payload(model, optimizer, model_id=model_id, epoch=epoch,
                                      best_dice=best_dice, history=history,
                                      source_checkpoint=str(initial_checkpoint) if initial_checkpoint else None)
        _save_checkpoint(last_path, payload)
        if event["is_best"]:
            _save_checkpoint(best_path, payload)
        emit(event)
    return {
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "last_checkpoint": str(last_path) if last_path.is_file() else None,
        "best_metric": best_dice if best_dice >= 0 else None,
        "best_epoch": best_epoch,
        "history": history,
        "cancelled": should_stop(),
    }
