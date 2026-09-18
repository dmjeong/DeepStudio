"""합성 저장의 원자성, 프로젝트 경계, 캐시 무결성과 개별 시드 검증."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gui'))
from core.defect_generator import DefectParams, generate_sample
from core.defect_io import DefectBatchWriter, read_roi
from core.defect_workflow import (candidate, sample_seed, settings_for, validate_output,
                                  validate_settings, validate_source, source_paths, publish_candidates)
from core.project import ProjectManager


def encode(image):
    output = io.BytesIO()
    Image.fromarray(image).save(output, format='PNG')
    return output.getvalue()


class DefectTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = np.full((64, 80, 3), 128, dtype=np.uint8)
        self.source = self.root / '정상.png'
        Image.fromarray(self.image).save(self.source)
        self.sample = generate_sample(self.image, DefectParams(seed=42))
        self.codec = patch('core.defect_io.png_bytes', side_effect=encode)
        self.codec.start()
        self.addCleanup(self.codec.stop)

    def test_batch_is_invisible_until_all_files_are_committed(self):
        output = self.root / 'out'
        with DefectBatchWriter(output) as writer:
            target = Path(writer.save(self.sample, self.source, original=self.image))
            self.assertFalse(target.exists())
        record = json.loads((writer.output_dir / 'manifest.json').read_text(encoding="utf-8"))['samples'][0]
        self.assertTrue(target.is_file())
        self.assertTrue((writer.output_dir / record['mask']).is_file())
        self.assertTrue(writer.archive_path.is_file())
        self.assertEqual(list(output.glob('.defect-*')), [])
        self.assertLess(len(target.name), 40)

    def test_failure_at_every_commit_boundary_leaves_no_public_batch(self):
        from core.defect_io import durable_bytes
        for failure in ('encoding', 'image', 'mask', 'recipe', 'zip', 'manifest', 'rename'):
            with self.subTest(failure=failure):
                output = self.root / failure
                def fail_write(path, content):
                    path = Path(path)
                    if ((failure == 'image' and path.parent.name == 'images') or
                        (failure == 'mask' and path.parent.name == 'masks') or
                        (failure == 'recipe' and path.parent.name == 'recipes') or
                        (failure == 'manifest' and path.name == 'manifest.json')):
                        raise OSError('injected write failure')
                    return durable_bytes(path, content)
                target = 'core.defect_io.durable_bytes'
                kwargs = {'side_effect': fail_write}
                if failure == 'encoding':
                    target, kwargs = 'core.defect_io.png_bytes', {'side_effect': OSError('encode')}
                elif failure == 'zip':
                    target, kwargs = 'core.defect_io.zipfile.ZipFile', {'side_effect': OSError('zip')}
                elif failure == 'rename':
                    target, kwargs = 'core.defect_io.os.rename', {'side_effect': OSError('publish')}
                with self.assertRaises(OSError), patch(target, **kwargs):
                    with DefectBatchWriter(output) as writer:
                        writer.save(self.sample, self.source, original=self.image)
                self.assertEqual(list(output.iterdir()), [])

    def test_separate_candidates_survive_later_failure_and_detect_corruption(self):
        job = self.root / 'job'
        with DefectBatchWriter(job / 'candidates', batch_id='000001') as writer:
            writer.save(self.sample, self.source, original=self.image)
        with self.assertRaises(OSError):
            with DefectBatchWriter(job / 'candidates', batch_id='000002'):
                raise OSError('cancelled candidate')
        row = candidate(job, '000001', verify=True)
        Path(row['image']).write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, '손상'):
            candidate(job, '000001', verify=True)
        self.assertFalse((job / 'candidates' / '000002').exists())

    def test_roi_dimensions_are_never_silently_changed(self):
        roi = self.root / 'roi.png'
        Image.fromarray(np.ones((4, 5), dtype=np.uint8) * 255).save(roi)
        with self.assertRaisesRegex(ValueError, '크기 불일치'):
            read_roi(roi, (64, 80))
        self.assertEqual(read_roi(roi, (4, 5)).shape, (4, 5))

    def test_sample_seed_is_independent_of_source_order(self):
        first = sample_seed(42, 'abc', 0)
        self.assertEqual(first, sample_seed(42, 'abc', 0))
        self.assertNotEqual(first, sample_seed(42, 'abc', 1))
        self.assertNotEqual(first, sample_seed(42, 'def', 0))

    def test_project_settings_and_synthetic_paths_round_trip_after_move(self):
        project = ProjectManager.create_new('검사', 'anomaly', str(self.root / 'project'), ['good', 'defect'])
        cfg = settings_for(project)
        self.assertEqual(Path(cfg['output']), Path(project.project_dir) / 'synthetic')
        project.defect_generation = validate_settings(cfg)
        path = ProjectManager.save(project)
        moved = self.root / 'moved'
        Path(project.project_dir).rename(moved)
        restored = ProjectManager.load(str(moved / Path(path).name))
        self.assertEqual(restored.defect_generation['output'], str(moved / 'synthetic'))
        self.assertEqual(restored.defect_generation['params']['seed'], 42)
        with self.assertRaisesRegex(ValueError, '데이터셋'):
            validate_output(restored, str(moved / 'data' / 'test' / 'defect'))
        with self.assertRaisesRegex(ValueError, '평가'):
            validate_source(restored, moved / 'data' / 'test' / 'good' / 'test.png')
        validate_source(restored, moved / 'data' / 'train' / 'good' / 'normal.png')

    def test_invalid_settings_rejected_before_worker_start(self):
        project = ProjectManager.create_new('검사', 'classify', str(self.root / 'project'), ['good', 'bad'])
        base = settings_for(project)
        for key, value in [('seed', -1), ('seed', True), ('count', True), ('intensity', float('nan')),
                           ('mix_types', 'false'), ('size_ratio', 0), ('types', [])]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_settings({**base, 'params': {**base['params'], key: value}})

    def source_project(self):
        project = ProjectManager.create_new('검사', 'classify', str(self.root / 'project'), ['good', 'bad'])
        source = Path(project.data.train_dir) / 'good' / 'normal.png'
        Image.fromarray(self.image).save(source)
        return project, source, settings_for(project)

    def test_empty_folder_never_scans_current_directory(self):
        project, _, payload = self.source_project()
        with patch('core.defect_workflow.os.walk') as walk:
            with self.assertRaises(ValueError):
                source_paths(project, {**payload, 'folder': ''})
            walk.assert_not_called()

    def test_evaluation_image_cannot_be_used_as_texture(self):
        project, _, payload = self.source_project()
        texture = Path(project.data.test_dir) / 'bad' / 'reference.png'
        texture.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.image).save(texture)
        with self.assertRaisesRegex(ValueError, '평가'):
            source_paths(project, {**payload, 'texture': str(texture)})

    def test_synthetic_texture_cannot_be_reused(self):
        project, _, payload = self.source_project()
        texture = self.root / 'synthetic' / 'images' / 'reference.png'
        texture.parent.mkdir(parents=True)
        Image.fromarray(self.image).save(texture)
        (texture.parent.parent / 'manifest.json').write_text('{"synthetic": true}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '재사용'):
            source_paths(project, {**payload, 'texture': str(texture)})

    def test_duplicate_bytes_with_different_names_are_rejected(self):
        project, source, payload = self.source_project()
        source.with_name('copy.png').write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, '중복'):
            source_paths(project, payload)

    def test_distinct_training_images_and_training_texture_are_allowed(self):
        project, source, payload = self.source_project()
        other = source.with_name('other.png')
        Image.fromarray(self.image + 1).save(other)
        result = source_paths(project, {**payload, 'texture': str(source)})
        self.assertEqual(result, [source.resolve(), other.resolve()])

    def test_overlapping_train_and_evaluation_directories_are_rejected(self):
        project, source, _ = self.source_project()
        project.data.val_dir = project.data.train_dir
        with self.assertRaisesRegex(ValueError, '평가'):
            validate_source(project, source)

    def test_repeat_publish_rejects_incomplete_saved_bundle(self):
        project, source, payload = self.source_project()
        job = self.root / 'job'
        with DefectBatchWriter(job / 'candidates', batch_id='000001') as writer:
            writer.save(self.sample, source, original=self.image)
        identity = {'filepath': str(self.root / 'project' / 'project.json')}
        (job / 'request.json').write_text(json.dumps({'kind': 'defects', 'payload': {'project': identity}}), encoding='utf-8')
        context = SimpleNamespace(cancelled=lambda: False, emit=lambda *_: None)
        def pixels(path):
            with Image.open(path) as image:
                return np.array(image)
        with patch('core.defect_workflow.read_image', side_effect=pixels):
            for missing in ('original', 'archive', 'recipe'):
                with self.subTest(missing=missing):
                    request = {'project': identity, 'source_directory': str(job), 'sample_ids': ['000001'],
                               'output': str(self.root / f'export-{missing}')}
                    result = publish_candidates(context, request, project)
                    folder = Path(result['output']['output_dir'])
                    record = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))['samples'][0]
                    target = {'original': record['original'], 'archive': 'annotations.zip',
                              'recipe': 'recipes/000001.json'}[missing]
                    (folder / target).unlink()
                    with self.assertRaisesRegex(ValueError, '손상'):
                        publish_candidates(context, request, project)


if __name__ == '__main__':
    unittest.main()
