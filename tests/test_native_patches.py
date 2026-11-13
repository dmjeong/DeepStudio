"""Native patch training: preserve labels/edges and deployment aggregation."""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from detection_patch_dataset import prepare_detection_patches
from native_patches import (
    bag_logits,
    detection_nms,
    patch_contract,
    tiles,
    validate_patches,
)
from PIL import Image
from studio_augmentation import augment_rgb

ZERO = {"horizontal_flip": 0., "rotation": 0., "color_jitter": 0.}


@pytest.mark.parametrize("family", ["yolo9", "rtdetrv4", "dfine"])
def test_detector_validation_matches_inference_pixels_and_boxes(family):
    from native_detection import DetectorValidationTransform, detector_pixels
    rgb = np.random.default_rng(3).integers(0, 256, (79, 117, 3), dtype=np.uint8)
    boxes = np.array([[1, 2, 65, 73, 1]], np.float32)
    transform = DetectorValidationTransform(family, 640)
    tensor, labels = transform(rgb[:, :, ::-1], boxes, (640, 640))
    pixels, (sx, sy) = detector_pixels(rgb, 640, family)
    np.testing.assert_array_equal(tensor, pixels.transpose(2, 0, 1).astype(np.float32) / 255)
    np.testing.assert_allclose(labels[0], [sx, 2*sy, 65*sx, 73*sy, 1])
    assert transform.wants_unresized_image and transform.custom_normalization
    # Low-intensity images must still remain on the /255 scale in validation.
    dark, _ = transform(np.ones_like(rgb), np.empty((0, 5)), (640, 640))
    assert dark[0, 0, 0] == np.float32(1/255)


def test_augmentation_disabled_and_forced_flip_boxes():
    image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    boxes = np.array([[1, 2, 7, 8, 0]], np.float32)
    pixels, labels = augment_rgb(image, ZERO, boxes)
    np.testing.assert_array_equal(pixels, image)
    np.testing.assert_array_equal(labels, boxes)
    pixels, labels = augment_rgb(image, {**ZERO, "horizontal_flip": 1.}, boxes)
    np.testing.assert_array_equal(pixels, image[:, ::-1])
    np.testing.assert_array_equal(labels, [[13, 2, 19, 8, 0]])
    for _ in range(10):
        pixels, labels = augment_rgb(image, {**ZERO, "rotation": 45., "color_jitter": .3}, boxes)
        assert pixels.dtype == np.uint8 and pixels.shape == image.shape
        assert np.isfinite(labels).all()
        assert np.all(labels[:, [0, 2]] <= 20) and np.all(labels[:, [1, 3]] <= 10)


def test_tile_coverage_padding_and_bag_label_semantics():
    contract = patch_contract("classify", 32, batch=2)
    image = np.zeros((19, 81, 3), np.uint8)
    covered = np.zeros(image.shape[:2], np.uint8)
    pieces = list(tiles(image, contract))
    for tile, (x, y, w, h) in pieces:
        assert tile.shape == (32, 32, 3)
        assert np.all(tile[h:] == 114)
        covered[y:y+h, x:x+w] = 1
    assert covered.all() and pieces[-1][1][0] == 49
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
    inputs = torch.tensor([[[[[2.]], [[0.]], [[0.]]], [[[0.]], [[4.]], [[0.]]]]], requires_grad=True)
    logits = bag_logits(model, inputs, chunk=1)
    torch.testing.assert_close(logits, torch.tensor([[2., 4.]]))
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    assert inputs.grad is not None and logits.shape == (1, 2)  # one original label
    with pytest.raises(ValueError):
        validate_patches({**contract, "scale": None})


def make_detection_data(root, size=(96, 72)):
    for split in ("train", "val"):
        for i in range(2):
            image = root / "images" / split / f"{i}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            target = root / "labels" / split / f"{i}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((size[1], size[0], 3), 60 + i * 70, np.uint8)
            pixels[20:40, 30:50] = (250, 10, 20)
            Image.fromarray(pixels).save(image)
            target.write_text("0 .5 .5 .5 .5\n" if i == 0 else "")
    return root


def test_detection_patch_boxes_and_source_splits(tmp_path):
    root = make_detection_data(tmp_path / "source")
    contract = patch_contract("detect", 32)
    result = prepare_detection_patches(root, tmp_path / "prepared", contract, num_classes=2,
                                       patches_per_image=2, seed=7)
    info = json.loads((result / "patches.json").read_text())
    assert info["counts"]["train"] == 4
    assert info["counts"]["val"] > 4  # validation always covers whole source
    from spatial_data import read_spatial_labels
    positives = 0
    for path in (result / "labels").rglob("*.txt"):
        rows = read_spatial_labels(path, "detect", 2)
        positives += len(rows)
        for cls, cx, cy, w, h in rows:
            assert 0 <= cx-w/2 < cx+w/2 <= 1.000001
            assert 0 <= cy-h/2 < cy+h/2 <= 1.000001
    assert positives and len(list(root.rglob("*.png"))) == 4
    rows = [{"class_id": c, "confidence": score, "bbox": box} for c, score, box in
            [(0, .9, [0, 0, 10, 10]), (0, .8, [1, 1, 10, 10]), (1, .7, [0, 0, 10, 10])]]
    assert [r["class_id"] for r in detection_nms(rows)] == [0, 1]


def test_small_object_coverage_preserves_full_box_and_validation(tmp_path):
    source = make_detection_data(tmp_path / "source", (232, 64))
    target = source / "labels/train/0.txt"
    target.write_text(f"0 {64/232:.12g} {20/64:.12g} {4/232:.12g} {6/64:.12g}\n", encoding="utf-8")
    contract = patch_contract("detect", 64)
    options = dict(num_classes=2, patches_per_image=2, seed=2)
    baseline = prepare_detection_patches(source, tmp_path / "baseline", contract, **options)
    covered = prepare_detection_patches(source, tmp_path / "covered", contract,
                                       small_object_coverage=True, **options)
    info = json.loads((covered / "patches.json").read_text(encoding="utf-8"))
    assert info["small_object_coverage"] == {
        "small_targets": 1, "randomly_covered": 0, "selected_covered": 1,
    }
    assert info["counts"] == json.loads((baseline / "patches.json").read_text(encoding="utf-8"))["counts"]
    from spatial_data import read_spatial_labels
    from hashlib import sha256
    prefix = sha256(b"train/0.png").hexdigest()[:20]
    positives = [row for path in (covered / "labels/train").glob(f"{prefix}_*.txt")
                 for row in read_spatial_labels(path, "detect", 2)]
    np.testing.assert_allclose(positives, [[0, 16/64, 20/64, 4/64, 6/64]])
    for folder in ("images/val", "labels/val"):
        assert {p.name: p.read_bytes() for p in (covered / folder).iterdir()} == {
            p.name: p.read_bytes() for p in (baseline / folder).iterdir()
        }


def test_resume_matches_the_actual_best_checkpoint_and_validation(tmp_path):
    pytest.importorskip("libreyolo")
    from native_detection_training import _resume_best_checkpoint

    trainer = SimpleNamespace(best_epoch=2, best_mAP50_95=.4)
    saved = {"epoch": 4, "best_epoch": 2, "best_metric_value": .4, "task": "detect",
             "model_family": "yolo9", "size": "t", "nc": 1, "names": {0: "crack"}, "imgsz": 640}
    current = tmp_path / "last.pt"
    best = tmp_path / "best_mAP_50_95_0.4_epoch_2.pt"
    report = {"epoch": 2, "metrics": {"mAP_50_95": .4}, "per_class": []}
    best_data = {**saved, "epoch": 1, "studio_validation": report}
    torch.save({**best_data, "epoch": 3}, best)
    assert _resume_best_checkpoint(current, saved, trainer) == (None, None)
    torch.save(best_data, best)
    assert _resume_best_checkpoint(current, saved, trainer) == (best, report)
    torch.save(best_data, tmp_path / "best_other_epoch_2.pt")
    assert _resume_best_checkpoint(current, saved, trainer) == (None, None)
    assert _resume_best_checkpoint(best, best_data, trainer) == (best, report)


def test_native_operating_point_uses_the_coco_validator_predictions_and_images():
    pytest.importorskip("libreyolo")
    from native_detection_training import _operating_point_from_coco

    truth = {"category_id": 7, "bbox": [10, 20, 5, 7]}
    evaluator = SimpleNamespace(
        _img_ids={2, 3, 4}, label_to_category_id={0: 7, 1: 9},
        coco_gt=SimpleNamespace(imgToAnns={1: [truth], 2: [truth], 3: [truth]}),
        results=[{**truth, "image_id": 2, "score": .9}, {**truth, "image_id": 2, "score": .8},
                 {**truth, "image_id": 3, "score": .2}, {**truth, "image_id": 4, "score": .5, "category_id": 9}],
    )
    point = _operating_point_from_coco(evaluator, ["crack", "FM"])
    assert (point["detection_tp"], point["detection_fp"], point["detection_fn"]) == (1, 2, 1)
    assert point["precision"] == pytest.approx(1/3) and point["recall"] == .5
    assert point["per_class"][1]["fp"] == 1 and point["per_class"][1]["recall"] is None
    evaluator.results.clear()
    empty = _operating_point_from_coco(evaluator, ["crack", "FM"])
    assert empty["detection_support"] == empty["detection_fn"] == 2
    assert empty["precision"] == empty["recall"] == 0


@pytest.mark.parametrize("model_id", ["libreyolo_detect_9t", "re_detr_v4_small", "libreyolo_detect_dfine_nano"])
def test_real_detector_training_settings_and_export(tmp_path, model_id):
    pytest.importorskip("libreyolo")
    from export_onnx import export_checkpoint
    from native_detection_training import train_detector
    from upstream_models import (
        _ProgressCallback,
        build_upstream_model,
        load_upstream_checkpoint,
        write_detection_dataset_yaml,
    )
    source = make_detection_data(tmp_path / "data")
    contract = patch_contract("detect", 640, batch=2)
    prepared = prepare_detection_patches(source, tmp_path / "patches", contract, num_classes=2,
                                         patches_per_image=1, small_object_coverage=True)
    data = write_detection_dataset_yaml(prepared, ["crack", "FM"], tmp_path / "data.json")
    model = build_upstream_model(model_id, num_classes=2, device="cpu")
    from native_detection import DetectorValidationTransform
    assert isinstance(model._get_val_preprocessor(), DetectorValidationTransform)
    events = []
    result = train_detector(model, data=str(data), augmentation=ZERO, patches=contract,
        small_object_coverage=True, train_patch_limit=1,
        epochs=1, batch=2, imgsz=640, workers=0, device="cpu", project=str(tmp_path), name="train",
        exist_ok=True, amp=False, val_loss=True, eval_interval=1, seed=1, cuda_graph=False,
        callbacks=_ProgressCallback(events.append))
    saved = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert saved["studio_augmentation"] == ZERO and saved["studio_patches"] == contract
    assert result["small_object_coverage"] and saved["studio_small_object_coverage"]
    assert saved["studio_train_patch_limit"] == 1
    validation = saved["studio_validation"]
    metrics = validation["metrics"]
    assert result["best_validation"] == validation and validation["epoch"] == 1
    assert metrics["mAP_50_95"] == result["best_metric"] == result["best_mAP50_95"]
    assert metrics["detection_support"] == 1
    assert metrics["detection_tp"] + metrics["detection_fn"] == 1
    assert validation["per_class"][1]["support"] == 0 and validation["per_class"][1]["recall"] is None
    epoch = next(event for event in events if event["event"] == "epoch_finished")
    assert epoch["epoch"] == epoch["best_epoch"] == 1
    assert epoch["metric_name"] == "mAP_50_95" and epoch["metrics"]["precision"] == metrics["precision"]
    assert epoch["detection_per_class"] == validation["per_class"]
    if model_id == "libreyolo_detect_9t":
        resume = build_upstream_model(model_id, num_classes=2, device="cpu",
                                     weights=result["last_checkpoint"])
        with pytest.raises(ValueError, match="재개 모델의 작은 객체 우선 설정"):
            train_detector(resume, data=str(data), augmentation=ZERO, patches=contract,
                small_object_coverage=True, train_patch_limit=2, resume=True,
                epochs=2, batch=2, imgsz=640, workers=0, device="cpu",
                project=str(tmp_path), name="resume", exist_ok=True, amp=False)
    loaded = load_upstream_checkpoint(result["best_checkpoint"], device="cpu")
    assert all(not module.training for module in loaded.model.modules())
    from native_patch_inference import predict_patches
    prediction = predict_patches(loaded, Image.open(source / "images/val/0.png"), contract)
    assert prediction.studio_tiles == 1
    output = tmp_path / "model.onnx"
    export_checkpoint(result["best_checkpoint"], output, verify=True, log=lambda _: None)
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert manifest["patches"] == contract and manifest["schema_version"] == 7
    assert manifest["normalize_mean"] == [0., 0., 0.]
    probe = os.getenv("DETECT_PROBE")
    if probe:
        run = subprocess.run([probe, str(output.with_suffix(".json")), str(source / "images/val/0.png")],
                             capture_output=True, text=True, check=True)
        payload = json.loads(run.stdout.strip().splitlines()[-1])
        assert len(payload["detections"]) == len(prediction.boxes.conf)


@pytest.mark.parametrize("variant", ["small", "medium"])
def test_real_patch_classifier_train_export_and_cpp(tmp_path, variant):
    pytest.importorskip("libreyolo")
    from export_onnx import export_checkpoint
    from libreyolo_training import train_classifier
    from native_patch_inference import predict_patches
    from upstream_models import build_upstream_model, load_upstream_checkpoint
    root = tmp_path / "data"
    for split in ("train", "val"):
        for c in ("ok", "ng"):
            folder = root / split / c
            folder.mkdir(parents=True)
            image = np.full((180, 450, 3), 60 if c == "ok" else 150, np.uint8)
            Image.fromarray(image).save(folder / "0.png")
    model = build_upstream_model("libreyolo_classify_mobilenetv4_" + variant, num_classes=2, device="cpu")
    contract = patch_contract("classify", 224, batch=2)
    result = train_classifier(model, data=str(root), augmentation=ZERO, patches=contract,
        epochs=1, batch=2, imgsz=224, workers=0, device="cpu", project=str(tmp_path), name="train",
        exist_ok=True, amp=False, val_loss=True, eval_interval=1, seed=1)
    saved = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert saved["studio_augmentation"] == ZERO and saved["studio_patches"] == contract
    loaded = load_upstream_checkpoint(result["best_checkpoint"], device="cpu")
    source = root / "val/ng/0.png"
    prediction = predict_patches(loaded, Image.open(source), contract)
    assert prediction.studio_tiles == 3
    from core.inference_loading import load_inference_engine
    engine = load_inference_engine(result["best_checkpoint"], gradcam=True)
    visible = engine.infer(str(source))
    assert visible.status == "ok", visible.error
    assert visible.gradcam_status == "completed", engine.info
    assert visible.details["patches"]["tiles"] == 3
    assert engine._heatmap_cache["valid_mask"].all()
    engine._gradcam.release()
    output = tmp_path / "model.onnx"
    export_checkpoint(result["best_checkpoint"], output, verify=True, log=lambda _: None)
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert manifest["patches"] == contract and manifest["schema_version"] == 7
    if probe := os.getenv("CLASSIFY_PROBE"):
        run = subprocess.run([probe, str(output.with_suffix(".json")), str(source)], capture_output=True, text=True, check=True)
        payload = json.loads(run.stdout.strip().splitlines()[-1])
        np.testing.assert_allclose(payload["probabilities"], prediction.probs.data.numpy(), atol=1e-4, rtol=1e-4)
        from model_crypto import create_key, encrypt_config, read_key
        key = create_key(tmp_path / "model.key")
        encrypted = tmp_path / "model.dvsenc"
        encrypt_config(output.with_suffix(".json"), encrypted, read_key(key))
        sealed = subprocess.run([probe, str(encrypted), str(source), str(key)], capture_output=True, text=True, check=True)
        np.testing.assert_array_equal(json.loads(sealed.stdout.strip().splitlines()[-1])["probabilities"], payload["probabilities"])


def test_detr_set_verification_keeps_pairs_and_rejects_changed_detections():
    from detection_export_validation import validate_detection_sets
    logits = np.array([[[3., -2.], [-2., 4.], [-8., -8.]]], np.float32)
    boxes = np.array([[[.2, .2, .1, .1], [.8, .8, .1, .1], [.5, .5, .1, .1]]], np.float32)
    perm = [1, 2, 0]
    assert validate_detection_sets([logits, boxes], [logits[:, perm], boxes[:, perm]])
    bad = boxes.copy(); bad[0, 0, 0] += .1
    with pytest.raises(ValueError):
        validate_detection_sets([logits, boxes], [logits, bad])
    bad = logits.copy(); bad[0, 0] = [-2., 3.]
    with pytest.raises(ValueError):
        validate_detection_sets([logits, boxes], [bad, boxes])
    bad = boxes.copy(); bad[0, 2, 0] += .1
    assert validate_detection_sets([logits, boxes], [logits, bad])  # below saved .25 floor only
    bad[0, 2, 0] = np.nan
    with pytest.raises(ValueError):
        validate_detection_sets([logits, boxes], [logits, bad])


def test_single_tile_training_does_not_break_pooled_bn():
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.BatchNorm2d(3),
                                torch.nn.Flatten(), torch.nn.Linear(3, 2)).train()
    inputs = torch.randn(1, 1, 3, 32, 32, requires_grad=True)
    output = bag_logits(model, inputs, chunk=2)
    output.sum().backward()
    assert model[1].training and model[1].weight.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_cpp_nonempty_detection_tiles_and_verification_floor(tmp_path):
    probe = os.getenv("DETECT_PROBE")
    if not probe:
        pytest.skip("DETECT_PROBE not configured")
    import onnx
    from export_onnx import create_inference_config
    from native_detection import preprocessing_contract
    from onnx import TensorProto, helper, numpy_helper
    logits = np.array([[[4., 3.], [-8., -8.]]], np.float32)
    boxes = np.array([[[.5, .5, .5, .5], [.1, .1, .1, .1]]], np.float32)
    graph = helper.make_graph([
        helper.make_node("Constant", [], ["scores"], value=numpy_helper.from_array(logits)),
        helper.make_node("Constant", [], ["boxes"], value=numpy_helper.from_array(boxes))], "fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2, 2]),
         helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 2, 4])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, tmp_path / "model.onnx")
    config = create_inference_config(tmp_path, "detect", 2, 32, 3, "model.onnx", ["crack", "FM"],
        preprocessing=preprocessing_contract("dfine", 32), backend="libreyolo_dfine", input_name="images",
        output_names=["scores", "boxes"])
    config = Path(config)
    doc = json.loads(config.read_text()); doc["schema_version"] = 7
    doc["postprocessing"].update(selection="query_class_topk", validated_confidence_floor=.25, class_aware_nms=False)
    doc["patches"] = patch_contract("detect", 32, scale=.5)
    doc["preprocessing"]["center_crop"] = {"width": 120, "height": 64}
    config.write_text(json.dumps(doc))
    source = tmp_path / "input.png"
    Image.new("RGB", (140, 80), (5, 20, 80)).save(source)
    result = subprocess.run([probe, str(config), str(source)], capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout.strip().splitlines()[-1])["detections"]
    assert {r["class_id"] for r in rows} == {0, 1}  # both classes from same query preserved
    assert len(rows) == 4  # final overlap gets merged independently per class
    assert min(r["bbox"][0] for r in rows) == pytest.approx(26.)
    assert all(r["bbox"][1:] == pytest.approx([24., r["bbox"][2], 56.]) for r in rows)
    from model_crypto import create_key, encrypt_config, read_key
    key = create_key(tmp_path / "model.key")
    encrypted = tmp_path / "model.dvsenc"
    encrypt_config(config, encrypted, read_key(key))
    sealed = subprocess.run([probe, str(encrypted), str(source), str(key)], capture_output=True, text=True, check=True)
    assert json.loads(sealed.stdout.strip().splitlines()[-1])["detections"] == rows
    doc["postprocessing"]["confidence_threshold"] = .1
    config.write_text(json.dumps(doc))
    rejected = subprocess.run([probe, str(config), str(source)], capture_output=True, text=True, check=False)
    assert rejected.returncode != 0 and "verification floor" in rejected.stderr


@pytest.mark.parametrize("model_id,asset,task", [
    ("libreyolo_detect_dfine_nano", "LibreDFINEn.pt", "detect"),
    ("libreyolo_classify_mobilenetv4_medium", "LibreMobileNetV4m-cls.pt", "classify"),
])
def test_downloaded_pretrained_weights_finetune_and_export(tmp_path, model_id, asset, task):
    root = os.getenv("NATIVE_PATCH_ASSETS")
    if not root:
        pytest.skip("NATIVE_PATCH_ASSETS not configured; no implicit downloads in tests")
    from pathlib import Path

    from export_onnx import export_checkpoint
    from upstream_models import build_upstream_model, write_detection_dataset_yaml
    model = build_upstream_model(model_id, num_classes=2, device="cpu", weights=str(Path(root) / asset))
    # Native wrapper loads the original 80/1000-class backbone before head adaptation.
    assert model.nb_classes == (80 if task == "detect" else 1000)
    source = torch.load(Path(root) / asset, map_location="cpu", weights_only=False)["model"]
    network = model.model.state_dict()
    key = next(k for k in source if k.endswith("weight") and source[k].ndim == 4 and k in network)
    torch.testing.assert_close(network[key], source[key], rtol=0, atol=0)
    settings = {"epochs": 1, "batch": 2, "workers": 0, "device": "cpu", "project": str(tmp_path),
                    "name": "train", "exist_ok": True, "amp": False, "val_loss": True, "eval_interval": 1, "seed": 2}
    if task == "detect":
        from native_detection_training import train_detector
        data = make_detection_data(tmp_path / "data", size=(720, 520))
        yaml = write_detection_dataset_yaml(data, ["crack", "FM"], tmp_path / "data.json")
        result = train_detector(model, data=str(yaml), augmentation=ZERO, imgsz=640, cuda_graph=False, **settings)
    else:
        from libreyolo_training import train_classifier
        data = tmp_path / "data"
        for split in ("train", "val"):
            for name in ("ng", "ok"):
                folder = data / split / name
                folder.mkdir(parents=True)
                Image.new("RGB", (320, 240), (40, 120 if name == "ok" else 220, 80)).save(folder / "0.png")
        result = train_classifier(model, data=str(data), augmentation=ZERO, imgsz=224, **settings)
    assert model.nb_classes == 2
    output = tmp_path / "pretrained.onnx"
    export_checkpoint(result["best_checkpoint"], output, verify=True, log=lambda _: None)
    assert json.loads(output.with_suffix(".json").read_text())["num_classes"] == 2
