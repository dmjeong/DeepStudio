"""실제 워커 제어 흐름에서 Best 체크포인트와 요약 값의 일치를 검증한다."""
from datetime import datetime
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from tests import test_training_contracts as contracts

source_method = contracts.source_method


def capture_signals(events):
    names = ("epoch_finished", "best_epoch_updated", "progress_updated", "lr_updated",
             "log_message", "eval_finished", "training_finished", "training_error")
    return SimpleNamespace(**{
        name: SimpleNamespace(emit=lambda *args, name=name: events.append((name, *args)))
        for name in names
    })


class CustomBestEpoch(unittest.TestCase):
    def test_checkpoint_metric_selects_both_losses_from_one_epoch(self):
        for task in ("classify", "segment", "detect"):
            with self.subTest(task=task):
                events = []
                stored, _, record = contracts.TrainingOrchestration.run_fake_training(
                    self, task, [.6, .9, .7], events=events,
                    loss_pairs=[(.8, .7), (.5, .6), (.1, .2)],
                )
                updates = [event for event in events if event[0] == "best_epoch_updated"]
                self.assertEqual([event[1] for event in updates], [1, 2])
                self.assertEqual(updates[-1][1:4], (2, .5, .6))
                best = next(state for path, state in stored.items() if path.endswith("best.pt"))
                self.assertEqual(best["epoch"], updates[-1][1])
                self.assertEqual(record.best_epoch, 2)
                for event in updates:
                    before = events[:events.index(event)]
                    self.assertIn(("save", event[1], "best.pt"), before)
                    self.assertTrue(any(row[0] == "epoch_finished" and row[1] == event[1]
                                        for row in before))

    def test_tied_custom_metric_keeps_first_best_loss_pair(self):
        events = []
        contracts.TrainingOrchestration.run_fake_training(
            self, "classify", [.8, .8], events=events,
            loss_pairs=[(.7, .6), (.1, .2)],
        )
        updates = [event for event in events if event[0] == "best_epoch_updated"]
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0][1:4], (1, .7, .6))

    def test_reconstruction_selects_lower_loss_and_its_matching_train_loss(self):
        events = []
        contracts.TrainingOrchestration.run_fake_training(
            self, "anomaly", [.7, .2, .4], events=events,
            loss_pairs=[(.8, .7), (.5, .2), (.1, .4)],
        )
        updates = [event for event in events if event[0] == "best_epoch_updated"]
        self.assertEqual(updates[-1][1:4], (2, .5, .2))

    def test_failed_checkpoint_save_does_not_announce_best(self):
        events = []
        with self.assertRaisesRegex(OSError, "체크포인트 저장 실패"):
            contracts.TrainingOrchestration.run_fake_training(
                self, "classify", [.8], events=events, fail_best_save=True,
            )
        self.assertFalse(any(event[0] == "best_epoch_updated" for event in events))




class PatchCoreBestEpoch(unittest.TestCase):
    def test_saved_single_pass_has_metrics_and_no_artificial_losses(self):
        for auroc in (.9, None):
            with self.subTest(auroc=auroc), tempfile.TemporaryDirectory() as temp:
                events = []
                model = SimpleNamespace(
                    preprocessing="full_range_v1", weight_source={}, training_metadata={},
                    center_crop=None,
                    backbone=SimpleNamespace(feature_dim=8),
                    fit=lambda *args, **kwargs: None,
                    get_info=lambda: {"memory_bank_size": 4},
                    save=lambda path: (Path(path).write_bytes(b"patchcore"), events.append(("save", path))),
                )
                dm = SimpleNamespace(get_device=lambda mode: "cpu",
                                     get_device_label=lambda device: "CPU",
                                     get_optimal_num_workers=lambda device: 0)
                method = source_method("gui/core/patchcore_trainer.py", "PatchCoreWorker",
                    "_run_patchcore", {
                        "os": os, "time": time, "datetime": datetime,
                        "get_device_manager": lambda: dm, "PatchCore": lambda **kwargs: model,
                        "ProjectManager": SimpleNamespace(new_run_id=lambda **kwargs: "run"),
                        "RunRecord": lambda **kwargs: SimpleNamespace(config_snapshot={}, **kwargs), "PatchCoreCancelled": RuntimeError,
                    })
                project = SimpleNamespace(training=SimpleNamespace(input_size=64, patchcore_sampling_ratio=.01, patchcore_n_neighbors=9),
                                          data=SimpleNamespace(), project_dir=temp, runs=[])
                evaluation = {"auroc": auroc, "f1": .8 if auroc is not None else None,
                              "evaluable": auroc is not None, "optimal_threshold": None}
                def loader():
                    return SimpleNamespace(dataset=SimpleNamespace(samples=[], preprocessing="full_range_v1"))
                worker = SimpleNamespace(
                    project=project, signals=capture_signals(events), _stop_requested=False,
                    _prepare_model=lambda *args: model,
                    _create_dataloaders=lambda *args: (loader(), loader()),
                    _evaluate_patchcore=lambda *args: (auroc, evaluation),
                )
                method(worker)
                update = next(event for event in events if event[0] == "best_epoch_updated")
                self.assertEqual(update[1], 1)
                self.assertTrue(math.isnan(update[2]) and math.isnan(update[3]))
                self.assertEqual(update[4]["auroc"], auroc)
                self.assertTrue(any(event[0] == "save" for event in events[:events.index(update)]))


if __name__ == "__main__":
    unittest.main()
