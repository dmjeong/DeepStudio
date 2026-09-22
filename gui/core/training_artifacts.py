"""실제 Best 저장 결과를 CSV, 파일명, 선정 근거에 동일하게 기록한다."""

import csv
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from collections.abc import Mapping

from core.version import APP_VERSION
from core.training_time import format_hms


def _cell(value):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _cell(value) for key, value in row.items()} for row in rows)
    _atomic_write(path, write)


def _object_dict(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    raise TypeError(f"설정 객체를 저장할 수 없습니다: {type(value).__name__}")


def training_hyperparameters(project, *, engine, effective=None, applied=None):
    """Capture requested and effective training settings before publishing CSVs."""
    training = _object_dict(getattr(project, "training", {}))
    model_object = getattr(project, "model", {})
    model = _object_dict(model_object)
    data = getattr(project, "data", None)
    return {
        "training": training,
        "model": model,
        "data": {
            "val_split": getattr(data, "val_split", None),
            "num_classes": getattr(data, "num_classes", None),
            "class_names": list(getattr(data, "class_names", [])),
        },
        "effective": {
            "engine": engine,
            "task": getattr(project, "task", ""),
            "model_id": getattr(model_object, "model_id", ""),
            **dict(effective or {}),
        },
        "applied": dict(applied or {}),
    }


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {str(key): _plain(item) for key, item in vars(value).items()
                if not str(key).startswith("_")}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _flatten_hyperparameters(values):
    columns = {}
    def visit(parts, value):
        if isinstance(value, Mapping):
            for key in sorted(value):
                safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(key)).strip("_") or "value"
                visit([*parts, safe], value[key])
            return
        name = "hp_" + "_".join(parts)
        columns[name] = (json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                         if isinstance(value, list) else value)
    visit([], _plain(values or {}))
    return columns


def publish_best(checkpoint, run_dir, history, *, epoch, metric, value,
                 direction, engine, task, policy, version="", timing=None, formula="", evaluation_metrics=None,
                 hyperparameters=None):
    """점수를 다시 비교하지 않고 엔진이 실제 저장한 Best만 내보낸다.

    학습 중에는 ``best.pt``를 임시 고정 이름으로 써서 평가·재개 로직을 단순하게
    유지한다. 최종 평가가 끝나면 그 파일을 선택 지표가 들어간 배포용 이름으로
    원자적으로 바꾸며, 평범한 ``best.pt``는 남기지 않는다.
    """
    checkpoint, run_dir = Path(checkpoint), Path(run_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best 체크포인트가 없습니다: {checkpoint}")
    if value is not None and not math.isfinite(float(value)):
        raise ValueError("Best 선정 값은 유한값 또는 평가 불가(None)여야 합니다")
    if epoch < 1:
        raise ValueError("Best 에폭 근거가 없습니다")
    metric = re.sub(r"[^A-Za-z0-9_-]+", "_", metric).strip("_") or "unavailable"
    score = "NA" if value is None else f"{float(value):.6f}"
    named = checkpoint.with_name(f"best_{metric}_{score}_epoch_{epoch}{checkpoint.suffix}")
    fd, temporary = tempfile.mkstemp(prefix=".best-", dir=checkpoint.parent)
    os.close(fd)
    try:
        shutil.copyfile(checkpoint, temporary)
        os.replace(temporary, named)
        # Named artifact is now durable.  Keep ``last.pt`` for resumable
        # workers when it had to be used as a cancellation fallback.
        if checkpoint.name == "best.pt":
            os.unlink(checkpoint)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    total_seconds = timing.elapsed() if timing is not None else None
    hyperparameters = _plain(hyperparameters or {})
    csv_context = {"engine": engine, "task": task, "studio_version": APP_VERSION,
                   **_flatten_hyperparameters(hyperparameters)}
    epochs = history.get("epoch") or list(range(1, max((len(v) for v in history.values()), default=0) + 1))
    by_epoch = {int(e): {key: values[index] for key, values in history.items()
                         if key != "epoch" and index < len(values)}
                for index, e in enumerate(epochs)}
    csv_path = run_dir / "results.csv"
    rows = [{"epoch": e, **values} for e, values in by_epoch.items()]
    if not rows:
        rows = [{"epoch": epoch}]
    best_row = None
    for row in rows:
        row_epoch = int(float(row["epoch"]))
        row.update(by_epoch.get(row_epoch, {}))
        row["is_best"] = int(row_epoch == epoch)
        row["selection_metric"] = metric
        row["selection_direction"] = direction
        row["selection_value"] = value if row_epoch == epoch else by_epoch.get(row_epoch, {}).get(metric)
        for key in ("epoch_time", "elapsed_time"):
            seconds = row.get(f"{key}_sec")
            row[f"{key}_hms"] = format_hms(float(seconds)) if seconds not in (None, "") else ""
        row["run_total_time_sec"] = total_seconds
        row["run_total_time_hms"] = format_hms(total_seconds) if total_seconds is not None else ""
        row.update(csv_context)
        if row_epoch == epoch:
            best_row = dict(row)
    # 이어학습 CSV가 없는 경우 이전 Best의 손실을 만들어 채우지 않는다.
    if best_row is None:
        best_row = {"epoch": epoch, "is_best": 1, "selection_metric": metric,
                    "selection_direction": direction, "selection_value": value, **csv_context}
        rows.insert(0, best_row.copy())
    best_row["checkpoint"] = named.name
    from core.best_metrics import scalar_metrics
    best_metrics = scalar_metrics({**(evaluation_metrics or {}), **by_epoch.get(epoch, {}),
                                   **({metric: value} if metric != "unavailable" else {})})
    best_row.update(best_metrics)
    _write_csv(csv_path, rows)
    _write_csv(run_dir / "best_result.csv", [best_row])
    details = {"epoch": epoch, "metric": metric, "value": value, "direction": direction,
               "engine": engine, "task": task, "policy": policy, "engine_version": version or APP_VERSION,
               "studio_version": APP_VERSION, "formula": formula,
               "checkpoint": os.path.relpath(named, run_dir), "filename_decimals": 6}
    details["metrics"] = best_metrics
    details["hyperparameters"] = hyperparameters
    if total_seconds is not None:
        details["total_seconds"] = total_seconds
        details["total_hms"] = format_hms(total_seconds)
        _write_csv(run_dir / "training_summary.csv", [{"engine": engine, "task": task,
                   "best_epoch": epoch, "epochs_in_session": len(epochs),
                   "total_training_time_sec": total_seconds,
                   "total_training_time_hms": format_hms(total_seconds),
                   **_flatten_hyperparameters(hyperparameters)}])
    _atomic_write(run_dir / "best_selection.json",
                  lambda stream: json.dump(details, stream, ensure_ascii=False, indent=2, allow_nan=False))
    return str(named), details
