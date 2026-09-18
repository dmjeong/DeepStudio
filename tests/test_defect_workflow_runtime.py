"""실제 파일로 공통 합성, 중단 복원, 재생성 없는 내보내기를 검증한다."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'gui'))
from core.project import ProjectManager
from core.defect_workflow import generate_candidates, publish_candidates, candidates, settings_for
from webapp.storage import project_view, write_json


class Context:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True)
        self.stop = False
        self.stop_after_first = False

    def cancelled(self):
        return self.stop

    def emit(self, name, args):
        if self.stop_after_first and name == 'progress_updated':
            self.stop = True


@unittest.skipUnless(importlib.util.find_spec('cv2'), '실제 OpenCV PNG 코덱 필요')
class DefectWorkflowRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = ProjectManager.create_new('검사', 'classify', str(self.root / 'project'), ['good', 'bad'])
        ProjectManager.save(self.project)
        self.source = Path(self.project.data.train_dir) / 'good' / 'source.png'
        Image.fromarray(np.full((64, 80, 3), 128, np.uint8)).save(self.source)
        self.payload = {**settings_for(self.project), 'per_image': 3, 'project': project_view(self.project)}
        self.context = Context(self.root / 'job')
        write_json(self.context.directory / 'request.json', {'kind': 'defects', 'payload': self.payload})

    def test_cancel_retains_complete_candidate_and_publish_is_idempotent(self):
        self.context.stop_after_first = True
        result = generate_candidates(self.context, self.payload, self.project)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['output']['generated'], 1)
        row = candidates(self.context.directory)[0]
        expected = Path(row['image']).read_bytes()
        self.source.unlink()
        export = Context(self.root / 'export-job')
        payload = {'project': project_view(self.project), 'source_directory': str(self.context.directory),
                   'sample_ids': ['000001'], 'output': str(self.root / 'approved')}
        with patch('core.defect_workflow.generate_sample', side_effect=AssertionError('검수 저장에서 재생성 금지')):
            first = publish_candidates(export, payload, self.project)
            second = publish_candidates(export, payload, self.project)
        self.assertTrue(second['output']['reused'])
        self.assertEqual(first['output']['output_dir'], second['output']['output_dir'])
        out = Path(first['output']['output_dir'])
        manifest = json.loads((out / 'manifest.json').read_text(encoding="utf-8"))
        self.assertEqual((out / manifest['samples'][0]['image']).read_bytes(), expected)
        self.assertFalse(any(Path(self.project.data.test_dir).rglob('*.png')))

    def test_preview_and_batch_first_sample_have_identical_pixels(self):
        generate_candidates(self.context, {**self.payload, 'preview': True}, self.project)
        batch = Context(self.root / 'batch')
        generate_candidates(batch, self.payload, self.project)
        a, b = candidates(self.context.directory)[0], candidates(batch.directory)[0]
        self.assertEqual(a['seed'], b['seed'])
        self.assertEqual(Path(a['image']).read_bytes(), Path(b['image']).read_bytes())
        self.assertEqual(len(candidates(batch.directory)), 3)

    def test_publish_cancellation_and_corrupted_original_do_not_publish(self):
        generate_candidates(self.context, {**self.payload, 'preview': True}, self.project)
        export = Context(self.root / 'export-job')
        export.stop = True
        payload = {'project': project_view(self.project), 'source_directory': str(self.context.directory),
                   'sample_ids': ['000001'], 'output': str(self.root / 'approved')}
        result = publish_candidates(export, payload, self.project)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(list((self.root / 'approved').iterdir()), [])
        row = candidates(self.context.directory)[0]
        Path(row['original']).write_bytes(b'broken original')
        export.stop = False
        with self.assertRaisesRegex(ValueError, '손상'):
            publish_candidates(export, payload, self.project)
        self.assertEqual(list((self.root / 'approved').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
