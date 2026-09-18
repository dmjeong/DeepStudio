"""검출/분할 데이터, 후처리, 표시와 캐시 경계의 회귀 검사."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'gui'))
from spatial_data import read_spatial_labels, semantic_pairs, validate_spatial_dataset
from detection import postprocess_detections
from core.spatial_preview import draw_detection_boxes, segmentation_preview
from core.inference_cache import InferencePreviewCache


class SpatialContracts(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def dataset(self, label):
        for split in ('train', 'val'):
            path = self.root / 'images' / split / 'nested' / 'sample.png'
            path.parent.mkdir(parents=True)
            Image.new('RGB', (64, 64)).save(path)
            self.write(f'labels/{split}/nested/sample.txt', label)

    def test_native_polygon_dataset_and_background_image(self):
        self.dataset('0 .2 .2 .8 .2 .8 .8 .2 .8\n')
        Image.new('RGB', (64, 64)).save(self.root / 'images/train/background.png')
        result = validate_spatial_dataset(self.root, 'segment', 1)
        self.assertEqual(result['train'], {'images': 2, 'objects': 1, 'background_images': 1})

    def test_boxes_cannot_be_used_as_polygons(self):
        self.dataset('0 .5 .5 .2 .2')
        self.assertEqual(validate_spatial_dataset(self.root, 'detect', 1)['val']['objects'], 1)
        with self.assertRaisesRegex(ValueError, 'sample.txt:1.*폴리곤'):
            validate_spatial_dataset(self.root, 'segment', 1)

    def test_invalid_labels_report_file_and_line(self):
        for row in ('0 .5 .5 -1 .1', '2 .5 .5 .2 .2', '.5 .5 .5 .2 .2',
                    '0 nan .5 .2 .2', '0 .5 .5 .2 .2 0', '0 .5 2 .2 .2'):
            with self.subTest(row=row):
                path = self.write('bad.txt', '\n' + row)
                with self.assertRaisesRegex(ValueError, 'bad.txt:2'):
                    read_spatial_labels(path, 'detect', 2)

    def test_all_background_and_missing_label_directory_fail_before_training(self):
        self.dataset('')
        with self.assertRaisesRegex(ValueError, '객체 라벨 없음'):
            validate_spatial_dataset(self.root, 'detect', 1)
        path = self.root / 'labels/train/nested/sample.txt'
        path.unlink()
        path.parent.rmdir()
        path.parent.parent.rmdir()
        with self.assertRaisesRegex(ValueError, '픽셀 마스크.*폴리곤'):
            validate_spatial_dataset(self.root, 'segment', 1)

    def test_semantic_pairing_requires_all_images_and_unique_relative_names(self):
        self.write('images/nested/a.png', '')
        with self.assertRaisesRegex(ValueError, '마스크 누락'):
            semantic_pairs(self.root / 'images', self.root / 'masks')
        self.write('masks/nested/a.bmp', '')
        pairs = semantic_pairs(self.root / 'images', self.root / 'masks')
        self.assertEqual(len(pairs), 1)
        self.write('images/nested/a.jpg', '')
        with self.assertRaisesRegex(ValueError, '이름 중복'):
            semantic_pairs(self.root / 'images', self.root / 'masks')

    def test_nms_removes_duplicates_within_class_and_uses_joint_confidence(self):
        rows = np.array([[.5, .5, .4, .4, 8, 8, -8],
                         [.5, .5, .4, .4, 7, 8, -8],
                         [.5, .5, .4, .4, 8, -8, 8],
                         [.1, .1, .1, .1, 8, -8, -8]])
        detections = postprocess_detections(rows)
        self.assertEqual([item['class_id'] for item in detections], [0, 1])
        np.testing.assert_allclose(detections[0]['bbox'], [.3, .3, .7, .7])
        self.assertAlmostEqual(detections[0]['confidence'], (1 / (1 + np.exp(-8))) ** 2)

    def test_single_class_empty_and_invalid_boxes(self):
        self.assertEqual(postprocess_detections(np.empty((0, 6))), [])
        rows = [[.5, .5, .2, .2, 8, -8], [.5, .5, -.2, .2, 8, 8], [2, 2, .2, .2, 8, 8]]
        self.assertEqual(postprocess_detections(rows), [])
        with self.assertRaises(ValueError):
            postprocess_detections([[.5, .5, .2, .2, float('nan'), 1]])

    def test_boxes_render_at_image_edges_without_changing_source(self):
        original = np.zeros((60, 80, 3), dtype=np.uint8)
        boxes = [{'class_id': 0, 'confidence': .9, 'bbox': [.9999, .9999, 1, 1]},
                 {'class_id': 1, 'confidence': .8, 'bbox': [.2, .3, .7, .8]}]
        rendered = draw_detection_boxes(original, boxes)
        self.assertEqual(rendered.shape, original.shape)
        self.assertTrue(rendered.flags.c_contiguous)
        self.assertTrue(rendered.any())
        self.assertFalse(original.any())

    def test_semantic_preview_counts_original_resolution_without_interpolated_ids(self):
        original = np.zeros((60, 80, 3), dtype=np.uint8)
        rendered, counts = segmentation_preview(np.array([[0, 1], [0, 1]]), original, 2)
        self.assertEqual(counts, {0: 2400, 1: 2400})
        self.assertEqual(rendered.shape, original.shape)
        self.assertEqual(len(np.unique(rendered.reshape(-1, 3), axis=0)), 2)
        self.assertFalse(original.any())
        with self.assertRaises(ValueError):
            segmentation_preview(np.array([[2]]), original, 2)

    def test_cached_box_coordinates_are_owned_and_preserved_for_grid(self):
        cache = InferencePreviewCache()
        self.addCleanup(cache.clear)
        original = np.zeros((60, 80, 3), dtype=np.uint8)
        detections = [{'class_id': 1, 'confidence': .9, 'bbox': [.2, .3, .7, .8]}]
        heatmap = dict(original=original, activation=np.ones((60, 80)), kind='Grad-CAM',
                       info='', detections=detections, base_image=original)
        cache.put('a.png', original, heatmap)
        detections[0]['bbox'][0] = .9
        for thumbnail in (False, True):
            first = cache.get('a.png', thumbnail=thumbnail)
            self.assertEqual(first['heatmap']['detections'][0]['bbox'][0], .2)
            first['heatmap']['detections'][0]['bbox'][0] = .8
            self.assertEqual(cache.get('a.png')['heatmap']['detections'][0]['bbox'][0], .2)


if __name__ == '__main__':
    unittest.main()
