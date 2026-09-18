"""학습에 실제 사용되는 클래스 가중치와 손실 기울기 계약 검증."""

import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from class_weights import class_counts, weights_from_counts

try:
    import torch
except ImportError:
    torch = None



def samples(labels):
    # 확장 표본는 경로와 라벨 외에 캐시 경로와 이미지를 함께 보관한다.
    return SimpleNamespace(samples=[(f"image_{i}.png", label, None, None)
                                    for i, label in enumerate(labels)])


class ClassCountContracts(unittest.TestCase):
    def test_counts_read_four_column_samples_in_model_class_order(self):
        self.assertEqual(class_counts(samples([2, 0, 2, 1, 2]), 3), [1, 1, 3])

    def test_counts_use_actual_nested_training_subset(self):
        dataset = samples([0] * 9 + [1])
        outer = SimpleNamespace(dataset=dataset, indices=[0, 1, 2, 9])
        inner = SimpleNamespace(dataset=outer, indices=[2, 3])
        self.assertEqual(class_counts(inner, 2), [1, 1])
        self.assertEqual(class_counts(outer, 2), [3, 1])

    def test_counting_does_not_open_images_or_augment_training_data(self):
        class CachedDataset:
            samples = [("missing/ok.png", 0), ("missing/ng.png", 1)]

            def __getitem__(self, index):
                raise AssertionError("클래스 개수 계산에서 이미지 로드는 필요하지 않음")

        self.assertEqual(class_counts(CachedDataset(), 2), [1, 1])

    def test_unknown_or_out_of_range_labels_fail_explicitly(self):
        for labels in ([0, -1], [0, 2], [0, "1"], [0, 1.5]):
            with self.subTest(labels=labels), self.assertRaises((ValueError, TypeError)):
                class_counts(samples(labels), 2)

    def test_empty_or_unreadable_dataset_cannot_claim_balanced_training(self):
        for dataset in (samples([]), SimpleNamespace()):
            with self.subTest(dataset=dataset), self.assertRaises((ValueError, TypeError)):
                class_counts(dataset, 2)


class WeightFormulaContracts(unittest.TestCase):
    def assertWeightsEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for found, wanted in zip(actual, expected):
            self.assertAlmostEqual(found, wanted, places=7)

    def test_none_preserves_unweighted_mode(self):
        self.assertIsNone(weights_from_counts([90, 10], "none"))

    def test_balanced_equalizes_total_contribution_per_class(self):
        weights = weights_from_counts([90, 10], "balanced")
        self.assertWeightsEqual(weights, [5 / 9, 5])
        self.assertAlmostEqual(90 * weights[0], 10 * weights[1])
        self.assertAlmostEqual((90 * weights[0] + 10 * weights[1]) / 100, 1)

    def test_sqrt_is_weaker_than_balanced_and_keeps_mean_scale(self):
        weights = weights_from_counts([90, 10], "sqrt")
        self.assertWeightsEqual(weights, [5 / 6, 2.5])
        self.assertAlmostEqual(weights[1] / weights[0], 3)
        self.assertAlmostEqual((90 * weights[0] + 10 * weights[1]) / 100, 1)

    def test_equal_counts_produce_identity_for_both_modes(self):
        for mode in ("balanced", "sqrt"):
            with self.subTest(mode=mode):
                self.assertWeightsEqual(weights_from_counts([7, 7, 7], mode), [1, 1, 1])

    def test_missing_training_class_rejected_instead_of_infinite_weight(self):
        for mode in ("balanced", "sqrt"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                weights_from_counts([10, 0], mode)

    def test_bad_counts_and_unknown_mode_are_not_silently_ignored(self):
        for counts in ([], [-1, 3], [0, 0]):
            with self.subTest(counts=counts), self.assertRaises((ValueError, TypeError)):
                weights_from_counts(counts, "balanced")
        with self.assertRaises(ValueError):
            weights_from_counts([90, 10], "balnaced")


@unittest.skipIf(torch is None, "Torch 미설치: 실제 손실과 역전파 검증 생략")
class WeightedLossContracts(unittest.TestCase):
    def criterion(self, counts=(90, 10), mode="balanced", **kwargs):
        from classification_loss import WeightedClassificationLoss
        return WeightedClassificationLoss(weights_from_counts(counts, mode), **kwargs)

    def test_minority_gradient_does_not_cancel_with_batch_size_one(self):
        logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
        labels = torch.tensor([1])
        base = torch.nn.functional.cross_entropy(logits, labels)
        base_grad = torch.autograd.grad(base, logits)[0]
        for mode, scale in (("balanced", 5.0), ("sqrt", 2.5)):
            loss = self.criterion(mode=mode)(logits, labels)
            grad = torch.autograd.grad(loss, logits)[0]
            self.assertTrue(torch.allclose(loss, base * scale))
            self.assertTrue(torch.allclose(grad, base_grad * scale))

    def test_majority_only_batch_still_receives_lower_contribution(self):
        logits = torch.tensor([[0.2, 0.8], [1.0, -0.2]], requires_grad=True)
        labels = torch.tensor([0, 0])
        base = torch.nn.functional.cross_entropy(logits, labels)
        loss = self.criterion()(logits, labels)
        self.assertTrue(torch.allclose(loss, base * (5 / 9)))

    def test_mixed_batch_changes_each_examples_gradient_by_its_weight(self):
        logits = torch.tensor([[0.2, 0.8], [1.0, -0.2]], requires_grad=True)
        labels = torch.tensor([0, 1])
        base = torch.nn.functional.cross_entropy(logits, labels)
        base_grad = torch.autograd.grad(base, logits)[0]
        loss = self.criterion()(logits, labels)
        grad = torch.autograd.grad(loss, logits)[0]
        expected_scale = torch.tensor([5 / 9, 5]).unsqueeze(1)
        self.assertTrue(torch.allclose(grad, base_grad * expected_scale))

    def test_equal_counts_match_original_cross_entropy_loss_and_gradient(self):
        logits = torch.tensor([[0.2, 0.8], [1.0, -0.2]], requires_grad=True)
        labels = torch.tensor([0, 1])
        base = torch.nn.functional.cross_entropy(logits, labels)
        base_grad = torch.autograd.grad(base, logits)[0]
        for mode in ("none", "balanced", "sqrt"):
            loss = self.criterion(counts=[10, 10], mode=mode)(logits, labels)
            self.assertTrue(torch.equal(loss, base))
            self.assertTrue(torch.equal(torch.autograd.grad(loss, logits)[0], base_grad))

    def test_label_smoothing_keeps_sample_mean_reduction(self):
        logits = torch.tensor([[0.2, 0.8], [1.0, -0.2]], requires_grad=True)
        labels = torch.tensor([0, 1])
        expected = torch.nn.functional.cross_entropy(
            logits, labels, weight=torch.tensor([5 / 9, 5]),
            label_smoothing=.1, reduction="none").mean()
        actual = self.criterion(label_smoothing=.1)(logits, labels)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_large_weight_with_half_logits_stays_finite_in_float32_loss(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.tensor([[5.0, -5.0]], dtype=dtype, requires_grad=True)
                labels = torch.tensor([0])
                criterion = self.criterion(counts=[1, 199999])
                loss = criterion(logits, labels)
                expected = torch.nn.functional.cross_entropy(logits.float(), labels) * 100000
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.allclose(loss, expected))
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_custom_validation_excludes_training_weights_and_preserves_smoothing(self):
        from tests.test_training_contracts import source_method

        seen = []
        meter = SimpleNamespace(
            update=lambda predictions, labels: seen.append((predictions.tolist(), labels.tolist())),
            compute=lambda: {"accuracy": .5, "precision_macro": .5,
                             "recall_macro": .5, "f1_macro": .5},
        )
        validate = source_method("gui/core/trainer.py", "TrainWorker", "_validate", {
            "torch": torch, "nn": torch.nn, "F": torch.nn.functional,
            "create_metrics": lambda *args: meter,
        })
        logits = torch.tensor([[0.2, 0.8], [1.0, -0.2]])
        labels = torch.tensor([0, 1])
        loader = [(logits, labels)]
        model = torch.nn.Identity()
        data = SimpleNamespace(num_classes=2, class_names=["OK", "NG"])
        expected = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=.1).item()
        results = []
        for mode in ("none", "balanced", "sqrt"):
            criterion = self.criterion(mode=mode, label_smoothing=.1)
            with torch.no_grad():
                loss, metrics = validate(SimpleNamespace(), model, loader, criterion,
                                         "cpu", "classify", data)
            self.assertAlmostEqual(loss, expected)
            self.assertEqual(metrics["val_loss"], loss)
            results.append(metrics)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])
        self.assertEqual(seen, [([1, 0], [0, 1])] * 3)




if __name__ == "__main__":
    unittest.main()
