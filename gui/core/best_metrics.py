"""Display the metrics saved for one best checkpoint, never the latest epoch."""
import math
from numbers import Real


NON_METRICS = {"epoch", "best_epoch", "is_best", "learning_rate", "lr",
               "selected_value", "selection_value", "sample_count", "num_classes",
               "num_samples", "threshold", "optimal_threshold", "anomaly_threshold", "total_seconds"}


def scalar_metrics(values):
    return {key: (float(value) if value is not None and math.isfinite(value) else None)
            for key, value in values.items()
            if key not in NON_METRICS and not key.endswith(("_sec", "_seconds", "_hms"))
            and (value is None or isinstance(value, Real) and not isinstance(value, bool))}


def best_epoch_metrics(run):
    if run.best_epoch < 1:
        return {}
    values = scalar_metrics(run.eval_results or {})
    history = run.metrics_history or {}
    epochs = history.get("epoch", list(range(1, max((len(v) for v in history.values()), default=0) + 1)))
    if "epoch" not in history and (run.config_snapshot or {}).get("training", {}).get("training_mode") == "upstream_resume":
        epochs = []  # Legacy upstream records omitted the resume offset. Do not guess.
    if run.best_epoch in epochs:
        index = epochs.index(run.best_epoch)
        values.update(scalar_metrics({key: sequence[index] for key, sequence in history.items()
                                      if index < len(sequence)}))
    saved = (run.config_snapshot or {}).get("best_selection", {})
    if saved.get("epoch") == run.best_epoch:
        values.update(scalar_metrics(saved.get("metrics", {})))
    if run.best_metric_name and run.best_metric_name != "unavailable":
        values.update(scalar_metrics({run.best_metric_name: run.best_metric}))
    return values


def metric_label(key, task="classify"):
    from core.metrics import TASK_METRIC_NAMES
    labels = {"train_loss": "Train Loss", "val_loss": "Val Loss", "prompt_dice": "Prompt Dice",
              "prompt_iou": "Prompt IoU", "accuracy_top5": "Top-5 Accuracy", "recon_loss": "Reconstruction Loss"}
    labels.update(TASK_METRIC_NAMES.get(task, {}).get("labels", {}))
    return labels.get(key, key)
