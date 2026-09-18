"""프로젝트 이동, 저장 실패, UI 저장/학습 수명에 대한 회귀 검증.

Qt 미설치 환경에서도 실행하도록 GUI 메서드는 원본 AST에서 읽고 출력 객체만 대체한다.
실제 Qt event loop 또는 GPU 통합 테스트를 대체하지 않는다.
"""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gui')]
from core.project import ProjectManager, ProjectData, RunRecord
from core.training_progress import run_description


def source_method(relative_path, class_name, name, namespace):
    tree = ast.parse((ROOT / relative_path).read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / relative_path), 'exec'), namespace)
    return namespace[name]


class PassiveControl:
    def __init__(self):
        self.values = []

    def __getattr__(self, name):
        if name in ('count', 'currentIndex'):
            return lambda: 0
        if name == 'findData':
            return lambda _value: 0
        if name == 'currentData':
            return lambda: 'cpu'
        if name in ('model', 'item'):
            return lambda *args: self
        if name in ('setValue', 'setText'):
            return lambda value: self.values.append(value)
        return lambda *args, **kwargs: None


class PassiveForm:
    def __init__(self, project=None):
        self.project = project
        self.worker = None
        self._run_project = None
        self.metrics_card_row = PassiveControl()
        self.metric_cards = {}
        self.window = lambda: types.SimpleNamespace()
        self._device_manager = types.SimpleNamespace(get_device=lambda mode: 'cpu', get_device_label=lambda device: 'CPU')

    def __getattr__(self, name):
        if name.startswith('_') or name == 'collect_config':
            return lambda *args, **kwargs: None
        control = PassiveControl()
        setattr(self, name, control)
        return control


class ProjectPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.project = ProjectManager.create_new('Inspection', 'classify', str(self.base / 'original'), ['OK', 'NG'])
        self.filepath = Path(ProjectManager.save(self.project))

    def tearDown(self):
        self.temp.cleanup()

    def test_serialization_failure_preserves_file_and_runtime_state(self):
        before = self.filepath.read_bytes()
        modified = self.project.modified
        self.project.runs = [RunRecord(eval_results={'invalid': object()})]
        with self.assertRaises(TypeError):
            ProjectManager.save(self.project)
        self.assertEqual(self.filepath.read_bytes(), before)
        self.assertEqual(self.project.modified, modified)
        self.assertEqual(ProjectManager.get_active_filepath(self.project), str(self.filepath))

    def test_replace_failure_preserves_file_and_cleans_temporary(self):
        before = self.filepath.read_bytes()
        modified = self.project.modified
        self.project.training.epochs = 42
        with patch('core.project.os.replace', side_effect=PermissionError('locked')):
            with self.assertRaises(PermissionError):
                ProjectManager.save(self.project)
        self.assertEqual(self.filepath.read_bytes(), before)
        self.assertEqual(self.project.modified, modified)
        self.assertEqual(list(self.filepath.parent.glob('.project-*.tmp')), [])

    def test_renamed_opened_file_remains_save_target(self):
        renamed = self.filepath.with_name('renamed.dvproj')
        self.filepath.rename(renamed)
        loaded = ProjectManager.load(str(renamed))
        loaded.training.epochs = 19
        self.assertEqual(ProjectManager.save(loaded), str(renamed))
        self.assertFalse(self.filepath.exists())
        self.assertEqual(ProjectManager.load(str(renamed)).training.epochs, 19)

    def test_portable_paths_and_external_resource_preservation(self):
        external = self.base / 'external.pt'
        external.write_bytes(b'weights')
        self.project.model.pretrained_weights = str(external)
        checkpoint = Path(self.project.project_dir) / 'runs' / 'best.pt'
        checkpoint.write_bytes(b'checkpoint')
        self.project.runs = [RunRecord(checkpoint_path=str(checkpoint))]
        ProjectManager.save(self.project)
        payload = json.loads(self.filepath.read_text())
        self.assertEqual(payload['data']['root'], 'data')
        self.assertEqual(payload['runs'][0]['checkpoint_path'], 'runs/best.pt')
        self.assertNotIn('_active_filepath', payload)
        moved = self.base / 'moved'
        shutil.copytree(self.filepath.parent, moved)
        loaded = ProjectManager.load(str(moved / self.filepath.name))
        self.assertEqual(loaded.project_dir, str(moved))
        self.assertEqual(loaded.data.root, str(moved / 'data'))
        self.assertEqual(loaded.runs[0].checkpoint_path, str(moved / 'runs' / 'best.pt'))
        self.assertEqual(loaded.model.pretrained_weights, str(external))
        ProjectManager.save(loaded)
        self.assertEqual(json.loads(self.filepath.read_text())['modified'], self.project.modified)

    def test_legacy_absolute_paths_rebase_to_opened_location(self):
        payload = asdict(self.project)
        legacy = self.base / 'legacy'
        legacy.mkdir()
        target = legacy / 'loaded.dvproj'
        target.write_text(json.dumps(payload), encoding='utf-8')
        loaded = ProjectManager.load(str(target))
        self.assertEqual(loaded.data.train_dir, str(legacy / 'data' / 'train'))
        self.assertEqual(ProjectManager.get_active_filepath(loaded), str(target))

    def test_legacy_windows_paths_rebase_without_touching_external_paths(self):
        payload = asdict(self.project)
        payload['project_dir'] = r'C:\projects\inspect'
        payload['data']['root'] = r'C:\projects\inspect\data'
        payload['model']['pretrained_weights'] = r'D:\models\external.pt'
        target = self.base / 'windows.dvproj'
        target.write_text(json.dumps(payload), encoding='utf-8')
        loaded = ProjectManager.load(str(target))
        self.assertEqual(loaded.data.root, str(self.base / 'data'))
        self.assertEqual(loaded.model.pretrained_weights, r'D:\models\external.pt')

    def test_existing_project_cannot_be_recreated(self):
        before = self.filepath.read_bytes()
        with self.assertRaises(FileExistsError):
            ProjectManager.create_new('Inspection', 'classify', self.project.project_dir, ['A'])
        self.assertEqual(self.filepath.read_bytes(), before)

    def test_invalid_names_rejected_before_creating_folders(self):
        for name in ('../escape', r'..\escape', 'CON', 'bad:name', 'trailing.', ' spaced', 'NUL.txt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ProjectManager.create_new(name, 'classify', str(self.base / 'invalid'), ['OK'])
        with self.assertRaises(ValueError):
            ProjectManager.create_new('Valid', 'classify', str(self.base / 'invalid'), ['../escape'])
        with self.assertRaises(ValueError):
            ProjectManager.create_new('Valid', 'classify', str(self.base / 'invalid'), ['NG', 'ng'])
        self.assertFalse((self.base / 'invalid').exists())

    def test_run_ids_unique_with_fixed_clock(self):
        with patch('core.project.datetime') as clock:
            clock.now.return_value.strftime.return_value = '260907_10h47m05s'
            ids = [ProjectManager.new_run_id('classify', 'efficientnet_b0.pt', 640,
                                            project_dir=self.project.project_dir) for _ in range(100)]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(all('/' not in value and '\\' not in value for value in ids))
        self.assertEqual(ids[0], 'Classification_efficientnetb0_640_260907_10h47m05s')
        self.assertEqual(ids[1], ids[0] + '_2')
        self.assertTrue(all((Path(self.project.project_dir) / 'runs' / value).is_dir() for value in ids))
        clock.now.return_value.strftime.assert_called_with('%y%m%d_%Hh%Mm%Ss')

    def test_run_folder_reservation_preserves_existing_files(self):
        with patch('core.project.datetime') as clock:
            clock.now.return_value.strftime.return_value = '20260907_104705'
            root = Path(self.project.project_dir) / 'runs'
            (root / 'run_20260907_104705').write_text('existing result', encoding='utf-8')
            name = ProjectManager.new_run_id(project_dir=self.project.project_dir)
            self.assertEqual(name, 'run_20260907_104705_2')
            self.assertEqual((root / 'run_20260907_104705').read_text(), 'existing result')
            self.assertTrue((root / name).is_dir())
            clock.now.return_value.strftime.assert_called_with('%Y%m%d_%H%M%S')

    def test_ui_save_collects_pending_settings_first(self):
        save = source_method('gui/app/main_window.py', 'MainWindow', '_save_current_project', {'ProjectManager': ProjectManager})
        fake = types.SimpleNamespace(project=self.project, dataset_page=types.SimpleNamespace(),
            training_page=types.SimpleNamespace(collect_config=lambda: setattr(self.project.training, 'input_size', 640)))
        save(fake)
        self.assertEqual(ProjectManager.load(str(self.filepath)).training.input_size, 640)

    def test_ui_restore_retains_saved_input_size(self):
        restore = source_method('gui/widgets/training_widget.py', 'TrainingWidget', 'set_project', {'ProjectData': ProjectData, 'MODE_LABELS': __import__('core.training_modes', fromlist=['MODE_LABELS']).MODE_LABELS})
        self.project.training.input_size = 640
        ui = PassiveForm()
        ui.selection_combo.findData = lambda _value: 0
        restore(ui, self.project)
        self.assertEqual(ui.input_size_spin.values, [640])

    def test_default_legacy_anomaly_uses_isolated_patchcore_worker(self):
        created = []
        class Worker:
            def __init__(self, project):
                self.project = project
                self.signals = types.SimpleNamespace(**{name: PassiveControl() for name in (
                    'epoch_finished', 'best_epoch_updated', 'batch_finished', 'lr_updated', 'eval_finished', 'training_finished',
                    'training_error', 'log_message', 'progress_updated', 'early_stopped', 'layer_debug')})
                self.finished = PassiveControl()
                created.append(self)
            def start(self):
                pass
        start = source_method('gui/widgets/training_widget.py', 'TrainingWidget', '_start_training', {
            'copy': copy, 'asdict': asdict, 'os': os, 'time': types.SimpleNamespace(time=lambda: 1),
            'ProjectManager': ProjectManager,
            'PatchCoreWorker': Worker, '_metric_info_for': lambda project: {},
            'QMessageBox': types.SimpleNamespace(warning=lambda *args: self.fail('valid anomaly was rejected'),
                                                critical=lambda *args: self.fail(str(args)))})
        self.project.task = 'anomaly'
        self.project.training.training_mode = 'unsupported_finetune'  # 기존 JSON 기본값
        ui = PassiveForm(self.project)
        start(ui)
        self.assertEqual(len(created), 1)
        self.assertIsNot(created[0].project, self.project)
        created[0].project.training.epochs = 7
        self.assertNotEqual(self.project.training.epochs, 7)

    def test_cancelled_worker_merges_snapshot_and_restores_controls(self):
        finish = source_method('gui/widgets/training_widget.py', 'TrainingWidget', '_on_worker_finished', {
            'copy': copy, 'asdict': asdict, 'ProjectManager': ProjectManager,
            'run_description': run_description,
            'QMessageBox': types.SimpleNamespace(critical=lambda *args: self.fail(str(args)))})
        ui = PassiveForm(self.project)
        ui._run_project = self.project
        ui._run_snapshot = copy.deepcopy(self.project)
        ui._initial_run_count = 0
        ui._run_snapshot.runs.append(RunRecord(run_id='cancelled', status='cancelled'))
        ui._run_config = {'task': 'classify', 'training': {'epochs': 100}}
        ui._pending_result = None
        ui._pending_error = None
        ui._cancel_requested = True
        finish(ui)
        self.assertIsNone(ui._run_project)
        self.assertEqual(self.project.runs[0].status, 'cancelled')
        self.assertEqual(self.project.runs[0].config_snapshot['requested_config']['training']['epochs'], 100)
        self.assertNotIn('training', self.project.runs[0].config_snapshot)
        loaded = ProjectManager.load(str(self.filepath))
        self.assertEqual(loaded.runs[0].status, 'cancelled')

    def test_active_worker_blocks_close_without_saving(self):
        close = source_method('gui/app/main_window.py', 'MainWindow', 'closeEvent', {
            'QMessageBox': types.SimpleNamespace(warning=lambda *args: None)})
        calls = []
        ui = types.SimpleNamespace(has_active_jobs=lambda: True)
        event = types.SimpleNamespace(ignore=lambda: calls.append('ignored'), accept=lambda: calls.append('accepted'))
        close(ui, event)
        self.assertEqual(calls, ['ignored'])

    def test_finished_class_worker_stays_busy_until_ui_callback(self):
        busy = source_method('gui/app/main_window.py', 'MainWindow', 'has_active_jobs', {})
        pending = types.SimpleNamespace(isRunning=lambda: False)
        ui = types.SimpleNamespace(dataset_page=types.SimpleNamespace(_class_worker=pending))
        self.assertTrue(busy(ui))
        save = source_method('gui/app/main_window.py', 'MainWindow', '_save_current_project', {})
        with self.assertRaisesRegex(RuntimeError, '클래스 변경'):
            save(ui)

    def test_same_project_class_change_clears_inference_before_rebind(self):
        refresh = source_method('gui/app/main_window.py', 'MainWindow', '_on_dataset_project_changed', {})
        calls = []
        page = types.SimpleNamespace(set_project=lambda project: None)
        inference = types.SimpleNamespace(
            _clear_model=lambda: calls.append('cleared'),
            set_project=lambda project: calls.append(('rebound', project)))
        ui = types.SimpleNamespace(project=self.project, training_page=page, inference_page=inference,
                                   export_page=page, defect_gen_page=page)
        refresh(ui, self.project)
        self.assertEqual(calls, ['cleared', ('rebound', self.project)])


if __name__ == '__main__':
    unittest.main()
