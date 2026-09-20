"""Layer observation lifecycle, persisted identity and model option boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
from core.project import ProjectData, RunRecord
from core.training_modes import training_capabilities, validate_training_options
from core.layer_debug_session import LayerDebugSession, debug_rows


class InspectorDouble:
    def __init__(self):
        self.records = {}
        self.attached = False
        self.attach_count = 0

    def attach(self, model):
        self.attach_count += 1
        self.attached = True
        self.records = {"layer": {"output": {"finite": False, "mean": float("nan"), "shape": [2, 3]},
                                  "gradient": {"finite": True, "mean": 0.0}}}

    def close(self):
        self.attached = False


class LayerDebugContracts(unittest.TestCase):
    def setUp(self):
        self.project = ProjectData()
        self.project.training.training_mode = "efficientnet_finetune"
        self.project.training.layer_debug_enabled = True

    def test_shared_capabilities(self):
        for variant in ("efficientnet_b0", "efficientnet_b1"):
            self.project.training.efficientnet_model = variant
            caps = validate_training_options(self.project)
            self.assertIn(variant, caps["models"])
            self.assertTrue(caps["layer_debug"])
        self.assertTrue(training_capabilities("classify", "efficientnet_resume")["resume"])
        self.assertNotIn("mixup_alpha", caps["augmentation"])
        with self.assertRaisesRegex(ValueError, "지원하지 않는"):
            training_capabilities("classify", "unsupported_transfer")
        for task, mode in (("detect", "efficientnet_finetune"), ("obb", "custom")):
            with self.assertRaises(ValueError):
                training_capabilities(task, mode)

    def test_adapter_without_hooks_cannot_silently_accept_observation(self):
        self.project.model.model_id = "resnet18"
        self.project.training.training_mode = "custom"
        capabilities = training_capabilities("classify", "custom", model_id="resnet18")
        self.assertFalse(capabilities["layer_debug"])
        with self.assertRaisesRegex(ValueError, "연결되어 있지"):
            validate_training_options(self.project)

    def test_unsupported_and_invalid_observation_rejected(self):
        for mode in ("unsupported_finetune", "unsupported_resume"):
            self.project.training.training_mode = mode
            with self.assertRaises(ValueError):
                validate_training_options(self.project)
        self.project.training.training_mode = "efficientnet_finetune"
        for count in (0, 11, True, 1.5):
            self.project.training.layer_debug_batches = count
            with self.assertRaises(ValueError):
                validate_training_options(self.project)
        self.project.training.layer_debug_batches = 1
        self.project.training.layer_debug_patterns = " , "
        with self.assertRaises(ValueError):
            validate_training_options(self.project)

    def test_previously_ignored_augmentation_is_explicit_error(self):
        self.project.training.augmentation.vertical_flip = .5
        with self.assertRaisesRegex(ValueError, "미지원"):
            validate_training_options(self.project)

    def test_bounded_samples_detach_and_preserve_nonfinite_diagnostic(self):
        self.project.training.layer_debug_batches = 2
        inspector = InspectorDouble()
        record, events = RunRecord(run_id="run-A"), []
        with tempfile.TemporaryDirectory() as directory:
            session = LayerDebugSession(self.project.training, directory, record, events.append, inspector)
            for step in range(5):
                session.begin(object(), 1024.0)
                session.capture(1 + step, 1)
                self.assertFalse(inspector.attached)
            content = json.loads(session.path.read_text())
            self.assertEqual(len(content["samples"]), 2)
            self.assertEqual(inspector.attach_count, 2)
            self.assertEqual(content["samples"][0]["run_id"], "run-A")
            self.assertIsNone(events[0]["layers"]["layer"]["output"]["mean"])
            self.assertFalse(events[0]["layers"]["layer"]["output"]["finite"])
            self.assertEqual(record.config_snapshot["layer_debug"]["samples"], 2)
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            rows = debug_rows(events[0])
            self.assertEqual(rows[0][5], "N/A")
            self.assertEqual(rows[0][7], "0")

    def test_write_failure_detaches_and_preserves_previous_artifact(self):
        inspector = InspectorDouble()
        with tempfile.TemporaryDirectory() as directory:
            session = LayerDebugSession(self.project.training, directory, RunRecord(), lambda _: None, inspector)
            session.path.write_text("previous")
            session.begin(object())
            with patch("core.layer_debug_session.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    session.capture(1, 1)
            self.assertFalse(inspector.attached)
            self.assertEqual(session.path.read_text(), "previous")
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
    def test_real_backward_does_not_change_weights_or_rng(self):
        import copy
        import torch
        from layer_debug import LayerInspector
        torch.manual_seed(5)
        baseline = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Dropout(.2), torch.nn.Linear(6, 2))
        observed = copy.deepcopy(baseline)
        inputs, targets = torch.randn(3, 4), torch.tensor([0, 1, 0])
        with tempfile.TemporaryDirectory() as directory:
            inspector = LayerInspector(["0", "3"], max_layers=2)
            session = LayerDebugSession(self.project.training, directory, RunRecord(run_id="real"), lambda _: None, inspector)
            states = []
            for model in (baseline, observed):
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                torch.manual_seed(81)
                for step in range(2):
                    optimizer.zero_grad()
                    if model is observed:
                        session.begin(model, 128.0)
                    (torch.nn.functional.cross_entropy(model(inputs), targets) * 128.0).backward()
                    if model is observed:
                        session.capture(1, step + 1)
                    optimizer.step()
                states.append(torch.get_rng_state())
            for expected, actual in zip(baseline.parameters(), observed.parameters()):
                torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            torch.testing.assert_close(states[0], states[1], atol=0, rtol=0)
            gradient = session.snapshots[0]["layers"]["3"]["gradient"]
            self.assertTrue(gradient["finite"])
            self.assertLessEqual(abs(gradient["max"]), 1 / 3 + 1e-6)
            self.assertFalse(any(module._forward_hooks for module in observed.modules()))
            self.assertFalse(inspector._gradient_handles)
            with self.assertRaisesRegex(ValueError, "패턴"):
                LayerInspector(["*"], max_layers=1).attach(observed)
            self.assertFalse(any(module._forward_hooks for module in observed.modules()))


if __name__ == "__main__":
    unittest.main()
