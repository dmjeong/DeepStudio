"""Two desktop managers share history without sharing a CPU inference slot."""
import os
import time

import psutil
import pytest
from webapp.jobs import JobManager, TERMINAL
from webapp.locking import exclusive_file
from webapp.storage import read_json, write_json


def test_cpu_readers_do_not_bypass_gpu_or_mutation_protection(tmp_path):
    manager = JobManager(tmp_path)
    with exclusive_file(manager.root / 'compute.lock'):
        for kind in ('infer', 'inspect_model'):
            manager.require_idle(kind, {'device': 'cpu'})
            with pytest.raises(RuntimeError):
                manager.require_idle(kind, {'device': 'cuda:0'})
        for kind in ('train', 'export', 'delete_class', None):
            with pytest.raises(RuntimeError):
                manager.require_idle(kind, {'device': 'cpu'})
    manager.active_id = 'a' * 32
    with pytest.raises(RuntimeError):
        manager.require_idle('infer', {'device': 'cpu'})


def test_other_app_live_job_is_not_interrupted_or_deleted(tmp_path):
    directory = tmp_path / 'jobs' / ('a' * 32)
    job = {'id': directory.name, 'kind': 'train', 'status': 'running', 'created_at': '2026-11-11',
           'owner_pid': os.getpid(), 'owner_started': psutil.Process().create_time()}
    write_json(directory / 'job.json', job)
    manager = JobManager(tmp_path)
    assert read_json(directory / 'job.json') == job
    assert manager.list()[0]['status'] == 'running'
    with pytest.raises(RuntimeError, match='다른 앱'):
        manager.delete([job['id']])
    with pytest.raises(RuntimeError, match='다른 앱'):
        manager.export_archive(job['id'])
    write_json(directory / 'job.json', {**job, 'status': 'completed'})
    assert manager.list()[0]['status'] == 'completed'
    manager.delete([job['id']])


def test_reused_pid_does_not_mask_interrupted_job(tmp_path):
    directory = tmp_path / 'jobs' / ('a' * 32)
    job = {'id': directory.name, 'kind': 'train', 'status': 'running', 'created_at': '2026-11-11',
           'owner_pid': os.getpid(), 'owner_started': psutil.Process().create_time() - 100}
    write_json(directory / 'job.json', job)
    manager = JobManager(tmp_path)
    assert manager.list()[0]['status'] == 'interrupted'


def test_two_real_cpu_inference_processes_with_training_lock(tmp_path):
    torch = pytest.importorskip('torch')
    from PIL import Image
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    model = build_builtin_model('unet_lite', 2, 3).eval()
    weights = tmp_path / 'best.pt'
    torch.save(make_builtin_checkpoint('unet_lite', model, num_classes=2, in_channels=3,
                                      input_size=(32, 32), class_names=['background', 'crack']), weights)
    images = []
    for i, color in enumerate(('red', 'green')):
        path = tmp_path / str(i) / 'same.png'
        path.parent.mkdir()
        Image.new('RGB', (32, 32), color).save(path)
        images.append(str(path))
    state = tmp_path / 'state'
    first = JobManager(state)
    managers = [first]
    ids = []
    try:
        with exclusive_file(first.root / 'compute.lock'):
            ids.append(first.start('infer', {'weights': str(weights), 'device': 'cpu', 'threads': 1,
                                            'gradcam': False, 'images': images})['id'])
            second = JobManager(state)
            managers.append(second)
            assert second.read(ids[0])['job']['status'] == 'running'
            ids.append(second.start('infer', {'weights': str(weights), 'device': 'cpu', 'threads': 1,
                                             'gradcam': False, 'images': images[::-1]})['id'])
            deadline = time.monotonic() + 90
            while any(manager.active_id for manager in managers):
                assert time.monotonic() < deadline, 'CPU inference process timeout'
                time.sleep(.02)
        for manager, job_id, paths in zip(managers, ids, (images, images[::-1])):
            result = manager.read(job_id)
            assert result['job']['status'] == 'completed', result
            import numpy as np
            for i, path in enumerate(paths):
                root = manager.directory(job_id) / 'results'
                assert read_json(root / f'{i}.json')['image_path'] == path
                with np.load(root / f'{i}.npz', allow_pickle=False) as cache:
                    np.testing.assert_array_equal(cache['original'], np.asarray(Image.open(path)))
        assert all(row['status'] in TERMINAL for row in first.list())
    finally:
        for manager in managers:
            manager.close()
            process = manager.process
            if process is not None:
                try:
                    process.wait(timeout=10)
                except Exception:
                    process.kill()
                    process.wait(timeout=10)


def test_failed_request_is_not_left_as_a_live_job(tmp_path):
    manager = JobManager(tmp_path)
    with pytest.raises(TypeError):
        manager.start('infer', {'device': 'cpu', 'invalid': object()})
    assert manager.active_id is None
    assert JobManager(tmp_path).list()[0]['status'] == 'failed'
