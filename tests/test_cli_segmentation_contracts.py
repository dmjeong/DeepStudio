"""CLI semantic segmentation의 제외 라벨 계약 회귀 검사."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


SOURCE = Path(__file__).resolve().parents[1] / "python" / "train_segmentation.py"


def load_definitions(names, namespace):
    """모델/데이터셋 의존성 없이 실제 소스 정의만 실행한다."""
    nodes = []
    for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class NumpyLogits(np.ndarray):
    def argmax(self, dim):
        return np.asarray(self).argmax(axis=dim)


class SegmentationMetricContractTests(unittest.TestCase):
    def setUp(self):
        namespace = load_definitions(
            {"compute_metrics"}, {"torch": SimpleNamespace(Tensor=np.ndarray)}
        )
        self.compute = namespace["compute_metrics"]

    def test_ignored_pixels_do_not_change_metrics(self):
        target = np.array([[[0, 1, 255]]])
        logits = np.array([[[[8, 0, 9]], [[0, 8, 0]]]]).view(NumpyLogits)
        metrics = self.compute(logits, target, 2)
        self.assertEqual(metrics["pixel_acc"], 1.0)
        self.assertEqual(metrics["mean_iou"], 1.0)
        self.assertEqual(metrics["class_iou"], [1.0, 1.0])
        logits[0, :, 0, 2] = [0, 9]
        self.assertEqual(metrics, self.compute(logits, target, 2))

    def test_all_ignored_validation_batch_fails_explicitly(self):
        logits = np.zeros((1, 2, 1, 2)).view(NumpyLogits)
        with self.assertRaisesRegex(ValueError, "평가 가능한 정답 픽셀 없음"):
            self.compute(logits, np.full((1, 1, 2), 255), 2)


@unittest.skipIf(torch is None, "Torch 미설치: 실제 손실/역전파 검증 생략")
class SegmentationLossContractTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_definitions(
            {"DiceLoss", "CombinedSegLoss"},
            {"torch": torch, "nn": torch.nn, "F": torch.nn.functional},
        )

    def test_ignored_pixels_and_samples_do_not_change_loss(self):
        target = torch.tensor([[[0, 1, 255]]])
        logits = torch.tensor([[[[3.0, 0.0, 9.0]], [[0.0, 3.0, 0.0]]]])
        for name in ("DiceLoss", "CombinedSegLoss"):
            criterion = self.namespace[name](num_classes=2)
            expected = criterion(logits[:, :, :, :2], target[:, :, :2])
            self.assertTrue(torch.allclose(criterion(logits, target), expected))
            mixed_logits = torch.cat([logits, torch.zeros_like(logits)])
            mixed_target = torch.cat([target, torch.full_like(target, 255)])
            self.assertTrue(torch.allclose(criterion(mixed_logits, mixed_target), expected))

    def test_all_ignored_training_loss_has_zero_gradient(self):
        for name in ("DiceLoss", "CombinedSegLoss"):
            logits = torch.randn(2, 3, 2, 2, requires_grad=True)
            target = torch.full((2, 2, 2), 255, dtype=torch.long)
            loss = self.namespace[name](num_classes=3)(logits, target)
            self.assertEqual(loss.item(), 0.0)
            loss.backward()
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


if __name__ == "__main__":
    unittest.main()
