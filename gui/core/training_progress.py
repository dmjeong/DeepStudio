"""Epoch coordinates and elapsed-time estimates shared by training views."""
from bisect import bisect_left
import math
from numbers import Real


def record_epoch(epochs, histories, epoch, values):
    """Upsert one epoch; keep missing observations as gaps, never shift them left."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("Epoch must be a positive integer")
    index = bisect_left(epochs, epoch)
    exists = index < len(epochs) and epochs[index] == epoch
    for key in values:
        histories.setdefault(key, [None] * len(epochs))
    if not exists:
        epochs.insert(index, epoch)
        for history in histories.values():
            history.insert(index, None)
    for key, history in histories.items():
        value = values.get(key)
        history[index] = float(value) if isinstance(value, Real) and math.isfinite(value) else None


def remaining_seconds(elapsed, current_epoch, total_epochs, start_epoch):
    completed = current_epoch - start_epoch
    if completed <= 0 or elapsed < 0 or not math.isfinite(elapsed):
        return None
    return elapsed / completed * max(0, total_epochs - current_epoch)


def run_description(run):
    """Describe the selected run from recorded execution settings, not the form."""
    config = run.config_snapshot
    training = config.get("training", {})
    mode = str(training.get("training_mode", ""))
    if config.get("engine") == "builtin" or mode.startswith("builtin"):
        model = config.get("model_id") or config.get("model", {}).get("model_id") or "기본 모델"
    elif mode.startswith("efficientnet"):
        model = training.get("efficientnet_model", "EfficientNet")
    elif run.config_snapshot.get("task") == "anomaly" and training.get("anomaly_method") == "patchcore":
        model = "PatchCore / " + training.get("patchcore_backbone", "")
    elif mode == "custom" or config.get("task") == "anomaly":
        model = "Custom CSP"
    else:
        model = "모델 미기록"
    checksum = config.get("checkpoint_sha256")
    return f"Run: {run.run_id} | 모델: {model} | SHA-256: {checksum or '미기록'}"
