"""Bounded per-batch observations; no customer pixels or activation arrays saved."""
import json
import math
import os
from pathlib import Path


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


class LayerDebugSession:
    def __init__(self, config, run_dir, record, emit, inspector=None):
        if inspector is None:
            from layer_debug import LayerInspector
            inspector = LayerInspector(
                [value.strip() for value in config.layer_debug_patterns.split(",") if value.strip()],
                keep_values=False, max_layers=128)
        self.inspector = inspector
        self.limit = config.layer_debug_batches
        self.record = record
        self.emit = emit
        self.path = Path(run_dir) / "layer_debug.json"
        self.snapshots = []
        self.active = False

    def begin(self, model, gradient_scale=1.0):
        if len(self.snapshots) >= self.limit:
            return
        if not math.isfinite(gradient_scale) or gradient_scale <= 0:
            raise ValueError("Invalid gradient scale")
        self.inspector.gradient_scale = gradient_scale
        self.inspector.attach(model)
        self.active = True

    def capture(self, epoch, batch):
        if not self.active:
            return
        snapshot = finite_json({"run_id": self.record.run_id, "epoch": epoch, "batch": batch,
                                "gradient_scale": self.inspector.gradient_scale,
                                "gradient_unscaled": True, "layers": self.inspector.records})
        # Detach before validation, even when observation spans multiple epochs.
        self.close()
        self.snapshots.append(snapshot)
        temporary = self.path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps({"schema_version": 1, "samples": self.snapshots},
                                             ensure_ascii=False, allow_nan=False), encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.record.config_snapshot["layer_debug"] = {
            "path": str(self.path), "samples": len(self.snapshots), "latest": snapshot}
        self.emit(snapshot)

    def close(self):
        self.inspector.close()
        self.active = False


def debug_rows(snapshot):
    """Display the same fields for trainable and frozen selected layers."""
    def number(value):
        return f"{value:.5g}" if isinstance(value, (int, float)) and math.isfinite(value) else "N/A"
    rows = []
    for name, record in snapshot.get("layers", {}).items():
        output = record.get("output", record.get("eval_output", {}))
        gradient = record.get("gradient", {})
        rows.append([name, str(record.get("input_shapes", [])), str(output.get("shape", [])),
                     number(output.get("min")), number(output.get("max")), number(output.get("mean")),
                     str(output.get("finite", "N/A")), number(gradient.get("mean")),
                     str(gradient.get("finite", "N/A"))])
    return rows
