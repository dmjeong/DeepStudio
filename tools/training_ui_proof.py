"""Capture real Qt training widgets using synthetic projects and result events.

This checks rendering and event display, not model training or model accuracy.
Missing dependencies and unhandled Qt callbacks fail the check; nothing is skipped.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]


def write(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def render(scale, output):
    report = {"scale": scale, "status": "running", "cases": [], "issues": [],
              "event_source": "synthetic UI events; no model training",
              "commit": os.environ.get("GITHUB_SHA", "local checkout")}
    path = output / f"report-{scale}.json"
    write(path, report)
    old_hook = sys.excepthook
    page = None
    window = None
    try:
        import numpy as np
        from PySide6.QtCore import QPoint, QRect
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication, QScrollArea
        sys.path[:0] = [str(ROOT), str(ROOT / "gui"), str(ROOT / "python")]
        from core.project import ProjectManager
        from app.main_window import MainWindow
        from widgets.training_results import HAS_MATPLOTLIB
        if not HAS_MATPLOTLIB:
            raise RuntimeError("Matplotlib Qt canvas is required")

        def record_exception(kind, value, tb):
            report["issues"].append("".join(traceback.format_exception(kind, value, tb)))
        sys.excepthook = record_exception
        app = QApplication.instance() or QApplication([])
        app.setStyleSheet((ROOT / "gui/resources/styles/dark_theme.qss").read_text(encoding="utf-8"))
        window = MainWindow()
        page = window.training_page
        window.content_stack.setCurrentWidget(page)
        window.show()

        def capture(name, width, height):
            window.resize(width, height)
            QTest.qWait(100)
            app.processEvents()
            case = {"name": name, "requested_size": [width, height],
                    "actual_size": [window.width(), window.height()],
                    "training_page_size": [page.width(), page.height()],
                    "mode": page.mode_combo.currentData(),
                    "selection": page.selection_combo.currentData(), "issues": []}
            if case["requested_size"] != case["actual_size"]:
                case["issues"].append("Layout forces the full window beyond the requested viewport")
            controls = {"start": page.start_btn, "stop": page.stop_btn,
                        "status": page.status_label, "progress": page.progress_bar,
                        "eta": page.eta_label, "best": page.best_selection_label,
                        "identity": page.run_identity_label}
            controls.update({f"metric-{i}": label for i, label in enumerate(page.metric_cards.values())})
            rects = {}
            for key, control in controls.items():
                parent = control.parentWidget()
                while parent is not None and parent is not page:
                    if isinstance(parent, QScrollArea):
                        parent.ensureWidgetVisible(control, 0, 0)
                        app.processEvents()
                        rect = QRect(control.mapTo(parent.viewport(), QPoint(0, 0)), control.size())
                        if not parent.viewport().rect().contains(rect):
                            case["issues"].append(f"Control unreachable by scrolling: {key}")
                    parent = parent.parentWidget()
            for area in page.findChildren(QScrollArea):
                area.verticalScrollBar().setValue(0)
                area.horizontalScrollBar().setValue(0)
            app.processEvents()
            for key, control in controls.items():
                rects[key] = QRect(control.mapTo(window, QPoint(0, 0)), control.size())
            for a, b in (("start", "stop"), ("stop", "status"), ("progress", "eta")):
                if rects[a].intersects(rects[b]):
                    case["issues"].append(f"Controls overlap: {a}, {b}")
            png = output / f"{scale}-{name}-{width}x{height}.png"
            if not window.grab().save(str(png)):
                raise RuntimeError(f"Screenshot could not be saved: {png.name}")
            case["screenshot"] = png.name
            case["geometry"] = {key: [r.x(), r.y(), r.width(), r.height()] for key, r in rects.items()}
            report["cases"].append(case)
            write(path, report)

        modes = [
            ("efficientnet_finetune", "efficientnet_b0", "val_loss"),
            ("efficientnet_finetune", "efficientnet_b1", "recall_macro"),
            ("efficientnet_transfer", "efficientnet_b0", "engine_default"),
            ("efficientnet_resume", "efficientnet_b0", "engine_default"),
            ("efficientnet_scratch", "efficientnet_b0", "engine_default"),
        ]
        with tempfile.TemporaryDirectory(prefix="studio-ui-proof-") as temp:
            project = ProjectManager.create_new("UI Proof", "classify", str(Path(temp) / "project"), ["OK", "NG"])
            for index, (mode, architecture, metric) in enumerate(modes):
                project.training.training_mode = mode
                project.training.efficientnet_model = architecture
                project.training.selection_metric = metric
                project.model.pretrained_weights = ""
                page.set_project(project)
                page.advanced_check.setChecked(index % 2 == 1)
                assert page.mode_combo.currentData() == mode
                assert page.selection_combo.currentData() == metric
                assert page.epochs_spin.isEnabled() == (not mode.endswith("_resume"))
                assert page.resume_frame.isVisible() == mode.endswith(("_resume", "_transfer"))
                for width, height in ((1024, 680), (1280, 800), (1600, 900)):
                    capture(f"{index}-{mode}", width, height)

            project.training.training_mode = "efficientnet_finetune"
            project.training.selection_metric = "accuracy"
            page.set_project(project)
            for index, matrix in enumerate(([[7, 1], [2, 6]], [[5, 3], [1, 7]])):
                expected = np.array(matrix)
                accuracy = float(expected.trace() / expected.sum())
                page._on_epoch_finished(index + 1, 0.4, 0.5, {"accuracy": accuracy})
                page._on_best_epoch_updated(index + 1, 0.4, 0.5, {"accuracy": accuracy})
                page._on_progress(index + 1, 2)
                page._on_eval_finished({"task": "classify", "accuracy": accuracy, "confusion_matrix": matrix})
                np.testing.assert_array_equal(page.cm_chart.ax.images[0].get_array(), expected)
                assert len(page.cm_chart.figure.axes) == 2
                capture(f"synthetic-result-{index}", 1280, 800)
            page.set_project(project)
            assert len(page.cm_chart.figure.axes) == 1
            assert not page.cm_chart.ax.images
            assert page.progress_bar.value() == 0
            assert page.eval_widget.summary_table.rowCount() == 0
            capture("new-project-cleared", 1280, 800)
        report["status"] = "failed" if report["issues"] or any(c["issues"] for c in report["cases"]) else "passed"
    except ModuleNotFoundError as exc:
        report["status"] = "blocked"
        report["issues"].append(str(exc))
    except Exception:
        report["status"] = "failed"
        report["issues"].append(traceback.format_exc())
    finally:
        if window is not None:
            window.close()
            window.deleteLater()
        sys.excepthook = old_hook
        write(path, report)
    print("TRAINING_UI_PROOF " + json.dumps({"scale": scale, "status": report["status"],
          "captures": len(report["cases"]), "issues": report["issues"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("1", "1.25", "1.5"))
    parser.add_argument("--output", type=Path, default=ROOT / "training-ui-proof")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.scale:
        return render(args.scale, args.output)
    results = []
    for scale in ("1", "1.25", "1.5"):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR=scale)
        try:
            run = subprocess.run([sys.executable, __file__, "--scale", scale, "--output", str(args.output)],
                                 env=env, timeout=120, check=False)
            results.append({"scale": scale, "exit_code": run.returncode})
        except subprocess.TimeoutExpired:
            results.append({"scale": scale, "exit_code": 1, "error": "120 second timeout"})
    passed = all(r["exit_code"] == 0 for r in results)
    write(args.output / "summary.json", {"passed": passed, "runs": results})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
