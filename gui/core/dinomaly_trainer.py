"""Desktop/web event bridge for the normal-only Dinomaly trainer."""

import json
from datetime import datetime
from pathlib import Path


def train_project(context, project, device):
    import torch
    from center_crop import configured_center_crop
    from core.project import ProjectManager, RunRecord
    from core.training_artifacts import publish_best, training_hyperparameters
    from train_dinomaly import train_dinomaly

    cfg = project.training
    run_id = ProjectManager.new_run_id(
        task="anomaly",
        model_name="dinomaly_vits14",
        input_size=cfg.input_size,
        project_dir=project.project_dir,
    )
    run_dir = Path(project.project_dir) / "runs" / run_id

    def log(message):
        if isinstance(message, dict):
            if message["event"] == "epoch_finished":
                context.emit(
                    "epoch_finished",
                    [
                        message["epoch"],
                        message["train_loss"],
                        message["val_loss"],
                        message["metrics"],
                    ],
                )
                context.emit(
                    "progress_updated", [message["epoch"], message["total_epochs"]]
                )
                context.emit("lr_updated", [message["epoch"], message["learning_rate"]])
            elif message["event"] == "best_epoch_updated":
                context.emit(
                    "best_epoch_updated",
                    [
                        message["epoch"],
                        message["train_loss"],
                        message["val_loss"],
                        message["metrics"],
                    ],
                )
        else:
            context.emit("log_message", [str(message)])

    weights = project.model.pretrained_weights or None
    resume = weights if cfg.training_mode == "builtin_resume" else None
    aug = cfg.augmentation
    started = datetime.now().isoformat()
    context.emit(
        "log_message",
        [f"Dinomaly 정상 재구성 학습 · 장치: {device} · 입력: {cfg.input_size}"],
    )
    best = train_dinomaly(
        project.data.root,
        output_dir=run_dir,
        input_size=cfg.input_size,
        in_channels=cfg.in_channels,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        optimizer_name=cfg.optimizer,
        scheduler_name=cfg.scheduler,
        warmup_epochs=cfg.warmup_epochs,
        early_stop_patience=cfg.early_stop_patience,
        device=str(device),
        pretrained=cfg.training_mode == "builtin_finetune",
        resume=resume,
        initial_weights=None if resume else weights,
        patch_training=cfg.patch_training,
        patches_per_image=cfg.patches_per_image,
        patch_scale=cfg.patch_scale,
        center_crop=configured_center_crop(cfg),
        horizontal_flip=aug.horizontal_flip,
        rotation=aug.rotation,
        color_jitter=aug.color_jitter,
        log=log,
        should_stop=context.cancelled,
    )
    state = json.loads((run_dir / "training.json").read_text())
    history = state["metrics_history"]
    metrics = {
        key: [row[key] for row in history]
        for key in ("epoch", "train_loss", "val_loss")
    }
    for key in ("epoch_time_sec", "elapsed_time_sec"):
        metrics[key] = [row["metrics"][key] for row in history]
    selection = {}
    named = ""
    best_value = 0.0
    best_epoch = state["best_epoch"]
    if best.is_file():
        ckpt = torch.load(best, map_location="cpu", weights_only=True)
        best_value = ckpt["metric"]
        named, selection = publish_best(
            best,
            run_dir,
            metrics,
            epoch=best_epoch,
            metric="val_loss",
            value=best_value,
            direction="min",
            engine="dinomaly",
            task="anomaly",
            policy="normal_validation_loss",
            formula="normal feature reconstruction cosine loss",
            hyperparameters=training_hyperparameters(
                project,
                engine="dinomaly",
                effective={"device": str(device)},
                applied=state["training_config"],
            ),
        )
    record = RunRecord(
        run_id=run_id,
        started_at=started,
        finished_at=datetime.now().isoformat(),
        status="cancelled" if context.cancelled() else "completed",
        epochs_done=state["completed_epochs"],
        best_metric=best_value,
        best_epoch=best_epoch,
        best_metric_name="val_loss",
        checkpoint_path=named,
        metrics_history=metrics,
        lr_history=[row["learning_rate"] for row in history],
        config_snapshot={
            "engine": "dinomaly",
            "model_id": "dinomaly_vits14",
            "training": state["training_config"],
            "best_selection": selection,
        },
    )
    project.runs.append(record)
    context.emit("training_finished", [best_value, best_epoch, named])
    return record
