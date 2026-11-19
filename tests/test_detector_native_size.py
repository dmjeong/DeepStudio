"""Verify pixel scale and box coordinates on odd, non-square native inputs."""
import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")


def test_custom_loader_uses_original_pixels_and_box_and_validates_val_size(tmp_path, monkeypatch):
    from dataset import create_detection_loaders
    import dataset

    shape = (41, 73)
    pixels = np.arange(np.prod(shape) * 3, dtype=np.uint8).reshape(*shape, 3)
    for split in ("train", "val"):
        images, labels = tmp_path / "images" / split, tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.fromarray(pixels).save(images / "sample.png")
        (labels / "sample.txt").write_text("0 0.5 0.5 0.5 0.25\n")

    def no_resize(*args):
        pytest.fail("native detection called resize")
    monkeypatch.setattr(dataset, "resize", no_resize)
    train, val = create_detection_loaders(tmp_path, input_size=(64, 64), batch_size=1,
                                          num_classes=1, flip_prob=0., original_size=True)
    for loader in (train, val):
        images, boxes = next(iter(loader))
        assert tuple(images.shape) == (1, 3, *shape)
        torch.testing.assert_close(boxes[0, 0], torch.tensor([0., .5, .5, .5, .25]))
    Image.fromarray(pixels[:-1]).save(tmp_path / "images/val/sample.png")
    with pytest.raises(ValueError, match="크기"):
        create_detection_loaders(tmp_path, num_classes=1, original_size=True)


@pytest.mark.parametrize("family", ["yolo9", "dfine", "rtdetrv4"])
def test_train_val_keep_pixels_and_box_and_filter_canvas_only_candidates(family):
    pytest.importorskip("libreyolo")
    from native_detection import DetectorValidationTransform, source_frame_outputs
    from native_detection_training import DetectionTransform
    from upstream_original_size import checkpoint_region, detector_size, input_hw

    h, w = 107, 173
    canvas = detector_size(family, h, w)
    ch, cw = input_hw(canvas)
    bgr = np.zeros((h, w, 3), np.uint8)
    bgr[-1, -1] = [30, 70, 130]
    labels = np.array([[w - 5, h - 6, w, h, 1]], np.float32)
    augment = dict(horizontal_flip=0., rotation=0., color_jitter=0.)
    train = DetectionTransform(family, canvas, augment, (w, h))
    val = DetectorValidationTransform(family, canvas, (w, h))
    train_pixels, train_targets = train(bgr, labels.copy(), canvas)
    val_pixels, val_targets = val(bgr, labels.copy(), canvas)
    np.testing.assert_array_equal(train_pixels, val_pixels)
    np.testing.assert_allclose(train_pixels[:, h - 1, w - 1] * 255, [130, 70, 30])
    assert train_pixels.shape == (3, ch, cw)
    assert train_pixels[0, h, w] == pytest.approx(114 / 255 if family == "yolo9" else 0)
    np.testing.assert_array_equal(val_targets[0], labels[0])
    if family == "yolo9":
        np.testing.assert_allclose(train_targets[0], [1, (w - 5) / cw, (h - 6) / ch, w / cw, h / ch])
        raw = torch.tensor([[w - 5, h - 6, w + 2, h + 2, .8], [w + 1, h, cw, ch, .9]]).T
        (mapped,) = source_frame_outputs(family, (w, h), [raw])
        torch.testing.assert_close(mapped[:, 0], torch.tensor([w - 5, h - 6, w, h, .8]))
        assert mapped[4, 1] == 0
    else:
        np.testing.assert_allclose(train_targets[0], [1, w - 2.5, h - 3, 5, 6])
        xyxy = torch.tensor([[w - 5, h - 6, w + 2, h + 2], [w + 1, h + 1, cw, ch]], dtype=torch.float32)
        lo, hi = xyxy[:, :2], xyxy[:, 2:]
        canvas_wh = torch.tensor([cw, ch])
        cxcywh = torch.cat(((hi + lo) / 2 / canvas_wh, (hi - lo) / canvas_wh), 1)
        scores, boxes = source_frame_outputs(family, (w, h), [torch.tensor([[4.], [6.]]), cxcywh])
        torch.testing.assert_close(boxes[0], torch.tensor([(w - 2.5) / w, (h - 3) / h, 5 / w, 6 / h]))
        assert scores[0] == 4 and scores[1].sigmoid() < 1e-12
    saved = dict(task="detect", model_family=family, original_size_region=[w, h], imgsz_h=ch, imgsz_w=cw)
    assert checkpoint_region(saved) == (w, h)
    with pytest.raises(ValueError, match="원본 크기"):
        checkpoint_region(dict(saved, imgsz_w=cw + 32))
    with pytest.raises(ValueError, match="원본 학습 영역"):
        train(bgr[:-1], labels.copy(), canvas)


@pytest.mark.parametrize("family", ["yolo9", "rtdetrv4"])
def test_padded_onnx_exposes_original_pixels_and_source_frame_boxes(tmp_path, family):
    ort = pytest.importorskip("onnxruntime")
    from native_detection import source_frame_outputs
    from native_detection_export import add_source_frame_interface
    from upstream_original_size import detector_size, input_hw

    h, w = 77, 119
    ch, cw = input_hw(detector_size(family, h, w))
    fill = 114 / 255 if family == "yolo9" else 0.
    rng = torch.Generator().manual_seed(42)
    raw = ([torch.rand(1, 6, 20, generator=rng)] if family == "yolo9" else
           [torch.rand(1, 20, 2, generator=rng), torch.rand(1, 20, 4, generator=rng)])
    if family == "yolo9":
        raw[0][:, :4] *= w

    class Canvas(torch.nn.Module):
        def forward(self, image):
            # Depend on the last source pixel as well as the bottom-right padding.
            marker = (image[:, 0, h - 1, w - 1] + image[:, 0, -1, -1]).reshape(-1, 1, 1)
            return tuple(values + marker for values in raw)

    path = tmp_path / "test.onnx"
    names = [f"output{i}" for i in range(len(raw))]
    torch.onnx.export(Canvas(), torch.ones(1, 3, ch, cw), path, input_names=["rgb"], output_names=names,
                      opset_version=17, dynamic_axes={key: {0: "batch"} for key in ["rgb", *names]}, dynamo=False)
    add_source_frame_interface(path, family, (w, h), 2, 17)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[-2:] == [h, w]
    images = torch.rand(2, 3, h, w, generator=rng)
    canvas = torch.nn.functional.pad(images, (0, cw - w, 0, ch - h), value=fill)
    expected = source_frame_outputs(family, (w, h), Canvas()(canvas))
    for actual, reference in zip(session.run(None, {"rgb": images.numpy()}), expected):
        np.testing.assert_allclose(actual, reference.numpy(), atol=1e-6)
