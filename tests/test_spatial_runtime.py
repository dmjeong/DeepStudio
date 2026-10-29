"""실제 검출/분할 학습, 체크포인트, Qt 결과와 ONNX 회귀 검사."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'gui'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in (
    'torch', 'torchvision', 'PySide6', 'onnx', 'onnxruntime', 'cv2'))
SMALL = dict(backbone_channels=[8, 16, 32, 64, 128], csp_depth=[1, 1, 1, 1], dropout=0.)


@unittest.skipUnless(AVAILABLE, '실제 Torch, Qt, ONNX 런타임 필요')
class SpatialRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        import torch
        torch.manual_seed(123)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def model(self, task, classes=2, encoding='grid_sigmoid_xywh'):
        from model import CustomCSP
        return CustomCSP(task=task, num_classes=classes, detection_box_encoding=encoding, **SMALL)

    def checkpoint(self, model, size=(65, 97), legacy=False):
        import torch
        from checkpoint import make_checkpoint_metadata
        payload = make_checkpoint_metadata(model.task, model.num_classes,
                                           ['OK', 'NG'][:model.num_classes], size, 3, SMALL)
        if legacy:
            payload['model_config'].pop('detection_box_encoding', None)
        payload['model_state_dict'] = model.state_dict()
        path = self.root / f'{model.task}_{legacy}.pt'
        torch.save(payload, path)
        return path, payload

    def test_single_class_detection_has_valid_boxes_and_learns_class_and_box(self):
        import torch
        from core.trainer import TrainWorker
        model = self.model('detect', 1).train()
        worker = TrainWorker(SimpleNamespace())
        inputs = torch.rand(2, 3, 65, 97)
        outputs = model(inputs)
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue(((outputs[..., :4] > 0) & (outputs[..., :4] <= 1)).all())
        target = torch.tensor([[[0, .25, .5, .2, .3]], [[0, .75, .5, .2, .3]]])
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        before = model.head.cls_pred.weight.detach().clone()
        loss = worker._compute_detection_loss(outputs, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.head.cls_pred.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.head.bbox_pred.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.head.cls_pred.weight))
        background = worker._compute_detection_loss(model(inputs), torch.zeros(2, 3, 5))
        self.assertTrue(torch.isfinite(background))
        background.backward()
        with self.assertRaises(ValueError):
            worker._compute_detection_loss(outputs, torch.ones(2, 3))

    def test_one_prediction_is_not_trained_against_conflicting_classes(self):
        import torch
        from core.trainer import TrainWorker
        outputs = torch.tensor([[[.5, .5, .3, .3, 0., 0., 0.]]], requires_grad=True)
        targets = torch.tensor([[[0., .5, .5, .3, .3], [1., .6, .5, .3, .3]]])
        loss = TrainWorker(SimpleNamespace())._compute_detection_loss(outputs, targets)
        loss.backward()
        self.assertLess(float(outputs.grad[0, 0, 5]), 0)
        self.assertGreater(float(outputs.grad[0, 0, 6]), 0)

    def test_semantic_odd_rectangular_input_matches_target_and_backpropagates(self):
        import torch
        model = self.model('segment').train()
        outputs = model(torch.rand(2, 3, 65, 97))
        self.assertEqual(tuple(outputs.shape), (2, 2, 65, 97))
        targets = torch.randint(0, 2, (2, 65, 97))
        targets[:, :2] = 255
        loss = torch.nn.functional.cross_entropy(outputs, targets, ignore_index=255)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters()))

    def test_real_datasets_keep_flip_coordinates_and_reject_mask_misalignment(self):
        from dataset import DetectionDataset, SegmentationDataset
        images, labels, masks = [self.root / name for name in ('images', 'labels', 'masks')]
        for folder in (images, labels, masks):
            folder.mkdir()
        Image.new('RGB', (12, 8)).save(images / 'a.png')
        (labels / 'a.txt').write_text('0 .2 .5 .1 .4', encoding='utf-8')
        no_flip = DetectionDataset(str(images), str(labels), num_classes=1, flip_prob=0)
        flip = DetectionDataset(str(images), str(labels), num_classes=1, flip_prob=1)
        self.assertAlmostEqual(float(no_flip[0][1][0, 1]), .2)
        self.assertAlmostEqual(float(flip[0][1][0, 1]), .8)
        Image.new('L', (4, 4)).save(masks / 'a.png')
        dataset = SegmentationDataset(str(images), str(masks), num_classes=2)
        with self.assertRaisesRegex(ValueError, '원본 크기 불일치'):
            dataset[0]
        (labels / 'a.txt').write_text('0 .2 .5 .1 .4\n' * 51, encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '객체 수 51'):
            no_flip[0]

    def test_legacy_checkpoint_preserves_original_box_equations(self):
        import torch
        from export_onnx import load_custom_model
        model = self.model('detect', encoding='legacy_raw').eval()
        with torch.no_grad():
            model.head.bbox_pred.weight.zero_()
            model.head.bbox_pred.bias.fill_(-1)
        path, payload = self.checkpoint(model, legacy=True)
        restored = load_custom_model(payload)
        self.assertEqual(restored.detection_box_encoding, 'legacy_raw')
        inputs = torch.rand(1, 3, 65, 97)
        with torch.no_grad():
            np.testing.assert_array_equal(model(inputs).numpy(), restored(inputs).numpy())
            self.assertTrue((restored(inputs)[..., :4] == -1).all())
        from core.trainer import TrainWorker
        target = self.model('detect').eval()
        head_before = {key: value.clone() for key, value in target.head.state_dict().items()}
        TrainWorker(SimpleNamespace())._load_pretrained(target, str(path), 'detect')
        for key, value in model.backbone.state_dict().items():
            self.assertTrue(torch.equal(value, target.backbone.state_dict()[key]), key)
        for key, value in head_before.items():
            self.assertTrue(torch.equal(value, target.head.state_dict()[key]), key)

    def test_both_custom_tasks_export_and_restore_with_real_onnx_runtime(self):
        import torch
        from export_onnx import export_checkpoint, load_custom_model
        for task in ('detect', 'segment'):
            with self.subTest(task=task):
                model = self.model(task).eval()
                path, payload = self.checkpoint(model)
                restored = load_custom_model(payload)
                inputs = torch.rand(1, 3, 65, 97)
                with torch.no_grad():
                    np.testing.assert_array_equal(model(inputs).numpy(), restored(inputs).numpy())
                output = path.with_suffix('.onnx')
                # 내보내기 함수 내부에서 random/zero 입력과 배치 2를 ORT로 비교한다.
                export_checkpoint(path, output, dynamic_batch=True, log=lambda _: None)
                manifest = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
                self.assertEqual(manifest['verification'], 'passed')
                self.assertEqual(manifest['cpp_supported'], task in {'segment', 'detect'})
                if task == 'detect':
                    self.assertEqual(manifest['postprocessing']['box_encoding'], 'grid_sigmoid_xywh')
                    self.assertTrue(manifest['postprocessing']['class_aware_nms'])

    def test_segmentation_inference_lists_defects_judges_and_compares_with_ground_truth(self):
        """분할 추론 = 결함 목록(클래스·박스·면적·점수) + OK/NG + 정답 비교. Grad-CAM은 보이지 않는다."""
        import torch
        from PySide6.QtCore import QSettings
        from core.project import ProjectManager
        self.addCleanup(lambda: QSettings('DeepVisionStudio', 'Inference').clear())
        QSettings('DeepVisionStudio', 'Inference').clear()
        project = ProjectManager.create_new('p', 'segment', str(self.root / 'project'), ['background', 'scratch'])
        image = Path(project.data.root) / 'images' / 'train' / 'a.png'
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (101, 73), (40, 40, 40)).save(image)
        truth = np.zeros((73, 101), np.uint8)
        truth[12:22, 22:44] = 1   # 예측과 겹치는 정답 scratch → 검출
        truth[60:66, 80:90] = 1   # 예측이 없는 정답 scratch → 놓침
        mask = Path(project.data.root) / 'masks' / 'train' / 'a.png'
        mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(truth, 'L').save(mask)
        model = self.model('segment').eval()
        checkpoint, _ = self.checkpoint(model)
        widget = self.widget(checkpoint)
        widget.project = project
        widget.result_review.set_project(project)
        self.assertFalse(widget.gradcam_checkbox.isVisibleTo(widget))
        self.assertFalse(widget.heatmap_range_widget.isVisibleTo(widget))

        def logits(inputs):  # 모델 해상도(65×97)의 10×20 블록만 scratch
            output = torch.zeros(inputs.shape[0], 2, 65, 97)
            output[:, 1, 10:20, 20:40] = 6
            return output
        with patch.object(widget.model, 'forward', side_effect=logits):
            widget._batch_images = [str(image)]
            widget._run_inference()
            from tests.test_async_inference import wait_for_inference
            wait_for_inference(widget)
        result = widget._inference_results[str(image)]
        self.assertEqual(result.status, 'ok', result.error)
        self.assertEqual(result.summary, 'NG 1')
        defect, = result.details['defects']
        self.assertEqual(defect['class_name'], 'NG')  # 모델(체크포인트)의 클래스 이름
        self.assertAlmostEqual(defect['w'], 20 * 101 / 97, delta=1.5)
        self.assertAlmostEqual(defect['h'], 10 * 73 / 65, delta=1.5)
        self.assertGreater(defect['score'], .99)
        self.assertEqual(result.details['ground_truth']['mask'], str(mask))
        # 결함 표: 검출 1 + 놓친 정답 1, 이미지에는 결함 사각형
        self.assertTrue(widget.defect_table.isVisibleTo(widget))
        self.assertEqual([widget.defect_table.item(r, 8).text() for r in range(2)], ['검출', '놓침'])
        self.assertIn('결함 검출 1/2', widget.task_details_label.text())
        self.assertIn('NG 1장', widget.result_review.segment_summary.text())
        self.assertIn('놓침 1', widget.result_review.segment_summary.text())
        clean = widget._defect_base[1]
        self.assertFalse(np.array_equal(clean, widget._current_preview_rgb))
        # 행을 고르면 그 결함으로 확대한다
        widget.defect_table.setCurrentCell(0, 1)
        self.assertGreater(widget.image_label._zoom, 1)
        # 기준을 올리면 다시 추론하지 않고 OK로 바뀐다
        with patch.object(widget.model, 'forward', side_effect=AssertionError('rules must not rerun the model')):
            widget.result_review.min_area.setValue(10_000)
        self.application.processEvents()
        self.assertEqual(widget.image_label._result_text, 'OK')
        self.assertEqual(widget.result_review.model.item(0, 4).text(), 'OK')
        self.assertEqual(widget.defect_table.item(0, 8).text(), '놓침')
        self.assertEqual(QSettings('DeepVisionStudio', 'Inference').value('segment/min_area'), 10_000)

    def test_process_engine_reports_defects_only_when_desktop_asks(self):
        """데스크톱 추론 프로세스는 segment_review로 결함을 받고, 설정이 없는 요청(웹)은 이전 결과 그대로."""
        from core.inference_loading import load_inference_engine
        checkpoint, _ = self.checkpoint(self.model('segment').eval())
        image = self.root / 'a.png'
        Image.new('RGB', (40, 30), (90, 90, 90)).save(image)
        desktop = load_inference_engine(checkpoint, segment_review={'min_area': 1, 'min_score': 0, 'split_dirs': []})
        result = desktop.infer(str(image))
        self.assertEqual(result.status, 'ok', result.error)
        self.assertIn('defects', result.details)
        self.assertIn(result.summary.split()[0], ('OK', 'NG'))
        web = load_inference_engine(checkpoint).infer(str(image))
        self.assertNotIn('defects', web.details)
        self.assertEqual(web.summary, '세그멘테이션')

    def widget(self, checkpoint):
        from widgets.inference_widget import InferenceWidget
        widget = InferenceWidget()
        widget.infer_device_combo.setCurrentIndex(widget.infer_device_combo.findData('cpu'))
        def cleanup():
            widget._heatmap_timer.stop()
            if widget._gradcam:
                widget._gradcam.release()
            widget._preview_cache.clear()
            widget.hide()
            widget.deleteLater()
            self.application.processEvents()
        self.addCleanup(cleanup)
        with patch('widgets.inference_widget.QMessageBox.critical') as error:
            self.assertTrue(widget._load_model(str(checkpoint)), error.call_args)
            error.assert_not_called()
        return widget

    def assert_cache_only_clicks(self, widget):
        for index in (0, 1, 0):
            widget.result_review.select_path(widget._batch_images[index])
            self.application.processEvents()
            self.assertEqual(widget._current_result.status, 'ok', widget._current_result.error)
        widget._switch_view_mode('grid')
        widget._on_grid_cell_clicked(1)
        self.application.processEvents()

    def test_custom_results_show_boxes_and_masks_and_clicks_do_not_run_model(self):
        import torch
        for task in ('detect', 'segment'):
            with self.subTest(task=task):
                model = self.model(task).eval()
                if task == 'detect':
                    with torch.no_grad():
                        model.head.obj_pred.weight.zero_()
                        model.head.obj_pred.bias.fill_(5)
                        model.head.cls_pred.weight.zero_()
                        model.head.cls_pred.bias.copy_(torch.tensor([5., -5.]))
                checkpoint, _ = self.checkpoint(model)
                widget = self.widget(checkpoint)
                widget.gradcam_checkbox.setChecked(False)
                paths = []
                for index in range(2):
                    path = self.root / f'{task}_{index}.png'
                    Image.new('RGB', (101, 73), (30 + index, 30, 30)).save(path)
                    paths.append(str(path))
                widget._batch_images = paths
                widget._run_inference()
                from tests.test_async_inference import wait_for_inference
                wait_for_inference(widget)
                self.assertEqual(widget._current_result.status, 'ok', widget._current_result.error)
                if task == 'detect':
                    self.assertGreater(widget._current_result.details['num_detections'], 0)
                    self.assertNotIn('candidate_cells', widget._current_result.details)
                else:
                    self.assertEqual(sum(widget._current_result.details['pixel_counts'].values()), 101 * 73)
                self.assertEqual(widget._current_preview_rgb.shape, (73, 101, 3))
                self.assertGreater(len(np.unique(widget._current_preview_rgb.reshape(-1, 3), axis=0)), 1 if task == 'detect' else 0)
                self.assertGreater(float(np.abs(widget._current_preview_rgb.astype(float) - 31).sum()), 0)
                with patch.object(widget.model, 'forward', side_effect=AssertionError('click must use cache')):
                    self.assert_cache_only_clicks(widget)
                    widget.gradcam_checkbox.setChecked(True)
                    widget._render_cached_heatmap()





if __name__ == '__main__':
    unittest.main()
