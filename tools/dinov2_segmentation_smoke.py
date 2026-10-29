"""Opt-in real-weight training/export/CPU smoke. Synthetic masks are NOT accuracy evidence.

python tools/dinov2_segmentation_smoke.py --weights /path/dinov2_vits14_pretrain.pth
Requires the normal app dependencies; does not download files or change the asset bundle.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'gui')]


def fixture(root, channels):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(73)
    for split, count in [('train', 8), ('val', 4)]:
        for index in range(count):
            mask = np.zeros((96, 96), dtype=np.uint8)
            mask[10:80, 12+index:14+index] = 1  # two-pixel scratch
            mask[35:60, 45+index:68+index] = 2
            raw = np.clip(25 + mask.astype(float)*90 + rng.normal(0, 2, mask.shape), 0, 255).astype(np.uint8)
            if channels == 3:
                raw = np.repeat(raw[..., None], 3, axis=-1)
            for name, array in [('images', raw), ('masks', mask)]:
                directory = root/name/split; directory.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(directory/f'{index}.png')


def bench(model, channels, work, threads, iterations):
    import numpy as np
    import onnxruntime as ort
    import torch
    from torchvision import transforms
    from opencv_preprocess import OpenCVResize
    from dataset import get_normalize_params
    from export_onnx import export_to_onnx
    from onnx_session import cpu_session_options

    size = 392
    onnx = work/'benchmark.onnx'
    x = torch.randn(1, channels, size, size)
    export_to_onnx(model, x, onnx, 17, False, 'segment')
    session = ort.InferenceSession(str(onnx), cpu_session_options(num_threads=threads), providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        expected = model(x).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    np.testing.assert_allclose(expected, actual, atol=1e-3, rtol=5e-4)
    mean, std = get_normalize_params(channels)
    transform = transforms.Compose([OpenCVResize((size, size)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    raw = np.zeros((size, size), np.uint8) if channels == 1 else np.zeros((size, size, 3), np.uint8)
    def run():
        tensor = transform(raw).unsqueeze(0).numpy()
        return session.run(None, {session.get_inputs()[0].name: tensor})[0].argmax(axis=1)
    for _ in range(5):
        run()
    durations = []
    for _ in range(iterations):
        start = time.perf_counter(); run(); durations.append((time.perf_counter()-start)*1000)
    result = {'size': [size,size], 'channels': channels, 'threads': threads,
              'warmup': 5, 'iterations': iterations, 'median_ms': float(np.median(durations)),
              'p95_ms': float(np.percentile(durations,95)),
              'scope': 'in-memory uint8 resize+normalize+ORT+argmax; excludes disk, rendering and camera',
              'max_logit_error': float(np.max(np.abs(expected-actual)))}
    del session
    onnx.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--reference-repo', type=Path)
    parser.add_argument('--unet-weights', type=Path)
    parser.add_argument('--report', type=Path, default=Path('dinov2-smoke.json'))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--class-weights', choices=['none', 'sqrt', 'balanced'], default='balanced')
    parser.add_argument('--native-fixtures', type=Path, help='Optional new directory for native SDK tests (plain/encrypted model, test image/mask/key)')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    if args.native_fixtures and args.native_fixtures.exists():
        parser.error('--native-fixtures must be a new directory')
    if min(args.epochs, args.threads, args.iterations) < 1:
        parser.error('epochs, threads and iterations must be positive')
    import numpy as np
    import onnxruntime as ort
    import torch
    from builtin_assets import ASSET_FILES, file_sha256
    from builtin_models import build_builtin_model, load_builtin_checkpoint
    from dinov2_segmentation import DINOV2_SHA256, DinoV2SmallBackbone
    from prepare_builtin_assets import prepare
    from train_builtin import train_builtin
    from export_onnx import export_checkpoint
    from core.inference_loading import load_inference_engine
    from model_crypto import create_key, read_key, load_package
    from onnx_session import cpu_session_options, deployment_session_settings
    ort.disable_telemetry_events()
    torch.set_num_threads(args.threads)
    torch.manual_seed(73)
    if file_sha256(args.weights) != DINOV2_SHA256:
        raise ValueError('Not the pinned official DINOv2 ViT-S/14 checkpoint')
    report = {'platform': platform.platform(), 'processor': platform.processor(),
              'torch': torch.__version__, 'onnxruntime': ort.__version__,
              'weights_sha256': DINOV2_SHA256, 'data': 'synthetic geometric masks, not inspection accuracy', 'runs': []}
    try:
        report['cpu'] = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report['cpu'] = platform.processor()
    if args.reference_repo:
        sys.path.insert(0, str(args.reference_repo.resolve()))
        os.environ['XFORMERS_DISABLED'] = '1'
        from dinov2.hub.backbones import dinov2_vits14
        reference = dinov2_vits14(pretrained=False).eval()
        local = DinoV2SmallBackbone().eval()
        state = torch.load(args.weights, map_location='cpu', weights_only=True)
        reference.load_state_dict(state); local.load_state_dict(state)
        report['reference_revision'] = subprocess.check_output(['git','-C',str(args.reference_repo),'rev-parse','HEAD'],text=True).strip()
        report['reference_parity'] = []
        with torch.inference_mode():
            for shape in [(224,224),(70,98),(518,518)]:
                x = torch.randn(1,3,*shape)
                a = reference.get_intermediate_layers(x, n=[2,5,8,11], reshape=True)
                b = local(x)
                errors = [float((u-v).abs().max()) for u,v in zip(a,b)]
                for u,v in zip(a,b): torch.testing.assert_close(u,v,atol=1e-5,rtol=1e-5)
                report['reference_parity'].append({'input':shape,'max_errors':errors})
        del reference, local, state
    previous = os.environ.get('DVS_BUILTIN_ASSETS_DIR')
    try:
        with tempfile.TemporaryDirectory(prefix='dvs-dino-smoke-') as temporary:
            root = Path(temporary); assets = root/'assets'
            target = assets/ASSET_FILES['dinov2_vits14']; target.parent.mkdir(parents=True)
            shutil.copyfile(args.weights, target)
            os.environ['DVS_BUILTIN_ASSETS_DIR'] = str(assets)
            prepare(assets, ['dinov2_vits14'], register_existing=True)
            for channels in (1,3):
                work = root/f'ch{channels}'; data = work/'data'
                fixture(data, channels)
                best = train_builtin('dinov2_vits14', data, num_classes=3, input_size=96,
                    in_channels=channels, epochs=args.epochs, batch_size=2, learning_rate=1e-3,
                    pretrained=True, freeze_backbone=True, num_workers=0, device='cpu',
                    class_weights=args.class_weights,
                    horizontal_flip=0, rotation=0, color_jitter=0, class_names=['background','scratch','dent'],
                    output_dir=work/'train', log=lambda *_: None)
                ckpt = torch.load(best,map_location='cpu',weights_only=False)
                model = load_builtin_checkpoint(ckpt)
                assert model.weight_provenance['source'] == 'dinov2_lvd142m'
                engine = load_inference_engine(best, gradcam=False, threads=args.threads)
                inferred = engine.infer(str(data/'images/val/0.png'))
                assert inferred.status == 'ok', inferred.error
                plain = export_checkpoint(best,work/'model.onnx',dynamic_batch=True,log=lambda *_: None)
                key = create_key(work/'model.key')
                encrypted = export_checkpoint(best,work/'model.dvsenc',dynamic_batch=True,encryption_key_path=key,log=lambda *_: None)
                config, graphs = load_package(encrypted['output_path'],read_key(key))
                options = deployment_session_settings(config)
                s1 = ort.InferenceSession(plain['output_path'], cpu_session_options(**options), providers=['CPUExecutionProvider'])
                s2 = ort.InferenceSession(graphs[config['model_path']], cpu_session_options(**options), providers=['CPUExecutionProvider'])
                for batch in (1,2):
                    x = np.random.default_rng(73).normal(size=(batch,channels,96,96)).astype(np.float32)
                    expected = s1.run(None,{s1.get_inputs()[0].name:x})[0]
                    actual = s2.run(None,{s2.get_inputs()[0].name:x})[0]
                    np.testing.assert_array_equal(expected,actual)
                    with torch.inference_mode(): pt = model(torch.from_numpy(x)).numpy()
                    np.testing.assert_allclose(pt,actual,atol=1e-3,rtol=5e-4)
                summary = json.loads((work/'train/training.json').read_text())
                row = {'channels':channels,'training_epochs':summary['completed_epochs'],
                       'class_weights':args.class_weights,
                       'synthetic_validation_mIoU':summary['best_metric'], 'best_epoch':summary['best_epoch'],
                       'checkpoint_reload_inference':'passed','onnx':plain['verification'],
                       'encrypted_batch1_2_exact':True,
                       'cpu_benchmark':bench(model,channels,work,args.threads,args.iterations)}
                if args.native_fixtures and channels == 1:
                    from PIL import Image
                    from torchvision import transforms
                    from opencv_preprocess import OpenCVResize, read_image
                    from dataset import get_normalize_params
                    destination = args.native_fixtures
                    destination.mkdir(parents=True)
                    for source in [Path(plain['output_path']),Path(plain['config_path']),Path(encrypted['output_path']),key]:
                        shutil.copyfile(source,destination/source.name)
                    source = data/'images/val/0.png'
                    shutil.copyfile(source,destination/'input.png')
                    mean,std = get_normalize_params(channels)
                    transform = transforms.Compose([OpenCVResize((96,96)),transforms.ToTensor(),transforms.Normalize(mean,std)])
                    x = transform(read_image(source,channels)).unsqueeze(0).numpy()
                    mask = s1.run(None,{s1.get_inputs()[0].name:x})[0].argmax(1)[0].astype(np.uint8)
                    Image.fromarray(mask).save(destination/'expected-mask.png')
                report['runs'].append(row); print(json.dumps(row),flush=True)
                del s1,s2,graphs,engine,model,ckpt
                gc.collect(); shutil.rmtree(work)
            if args.unet_weights:
                model = build_builtin_model('unet_resnet18',3,1,backbone_weights=args.unet_weights).eval()
                report['unet_reference_same_settings'] = bench(model,1,root,args.threads,args.iterations)
    finally:
        if previous is None: os.environ.pop('DVS_BUILTIN_ASSETS_DIR',None)
        else: os.environ['DVS_BUILTIN_ASSETS_DIR'] = previous
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(args.report)


if __name__ == '__main__':
    main()
