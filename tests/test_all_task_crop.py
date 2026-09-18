"""Cropped labels, isolated snapshots and actual CPU model reload for every task."""
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
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]
from center_crop import center_crop_box, checkpoint_center_crop, restore_detections
from crop_dataset import crop_spatial_rows, prepare_crop_dataset


class CropSnapshotTests(unittest.TestCase):
    def test_patchcore_version_string_and_custom_mapping_both_restore(self):
        crop={"width":32,"height":24}
        self.assertEqual(checkpoint_center_crop({"type":"patchcore","preprocessing":"full_range_v1","center_crop":crop}),crop)
        self.assertIsNone(checkpoint_center_crop({"type":"patchcore","preprocessing":"legacy_pil_rgb"}))
        self.assertEqual(checkpoint_center_crop({"preprocessing":{"center_crop":crop}}),crop)

    def test_classification_and_reconstruction_preserve_splits_pixels_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.arange(11*9, dtype=np.uint16).reshape(9, 11)*500
            for split in ("train", "val", "test"):
                folder = root/"data"/split/"good"
                folder.mkdir(parents=True)
                Image.fromarray(pixels).save(folder/"a.tif")
            original = (root/"data/train/good/a.tif").read_bytes()
            for task in ("classify", "anomaly"):
                snapshot = Path(prepare_crop_dataset(root/"data", root/"generated", task,
                                                    {"width":4, "height":2}))
                for split in ("train", "val", "test"):
                    with Image.open(snapshot/split/"good/a.png") as image:
                        np.testing.assert_array_equal(np.asarray(image), pixels[3:5,3:7])
                self.assertEqual(json.loads((snapshot/"center_crop.json").read_text())["status"], "completed")
                self.assertEqual((root/"data/train/good/a.tif").read_bytes(), original)

    def test_detection_clips_partial_and_removes_outside_boxes(self):
        crop = {"width": 40, "height": 20}
        # ROI x=30..70, y=30..50. First box overlaps left border; second is outside.
        rows = crop_spatial_rows([[2,.3,.5,.4,.25],[0,.05,.05,.04,.04]], "detect", (100,80), crop)
        np.testing.assert_allclose(rows, [[2,.25,.5,.5,1.]])
        restored = restore_detections([{"bbox":[0,0,.5,1],"class_id":2}], (100,80), crop)
        np.testing.assert_allclose(restored[0]["bbox"], [.3,.375,.5,.625])

    @unittest.skipUnless(importlib.util.find_spec("shapely"), "Polygon geometry runtime required")
    def test_concave_instance_splits_without_bridging_background(self):
        # U shape with connector below the crop becomes two independent rectangles.
        points = [(1,1),(3,1),(3,7),(7,7),(7,1),(9,1),(9,9),(1,9)]
        row = [1] + [value/10 for point in points for value in point]
        cropped = crop_spatial_rows([row], "segment", (10,10), {"width":10,"height":2})
        self.assertEqual(len(cropped), 2)
        from shapely.geometry import Polygon
        polygons = [Polygon(np.asarray(r[1:]).reshape(-1,2)) for r in cropped]
        self.assertAlmostEqual(sum(p.area for p in polygons), .4)
        self.assertTrue(all(p.is_valid for p in polygons))

    def test_semantic_palette_indices_and_detection_labels_share_source_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"data/images/train").mkdir(parents=True)
            (root/"data/masks/train").mkdir(parents=True)
            (root/"data/labels/train").mkdir(parents=True)
            pixels = np.arange(99, dtype=np.uint8).reshape(9,11)
            Image.fromarray(pixels).save(root/"data/images/train/a.png")
            mask = Image.fromarray(pixels%3).convert("P")
            mask.save(root/"data/masks/train/a.png")
            crop = {"width":4,"height":2}
            snapshot = Path(prepare_crop_dataset(root/"data",root/"out","segment",crop))
            with Image.open(snapshot/"masks/train/a.png") as saved:
                self.assertEqual(saved.mode,"P")
                np.testing.assert_array_equal(np.asarray(saved), (pixels%3)[3:5,3:7])
            (root/"data/labels/train/a.txt").write_text("0 .5 .5 1 1\n")
            snapshot = Path(prepare_crop_dataset(root/"data",root/"out","detect",crop))
            np.testing.assert_allclose(np.loadtxt(snapshot/"labels/train/a.txt"), [0,.5,.5,1,1])

    def test_save_failure_cancellation_and_name_collision_never_publish_partial_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"data/good").mkdir(parents=True)
            Image.new("RGB",(10,10)).save(root/"data/good/a.png")
            args=(root/"data",root/"out","classify",{"width":4,"height":4})
            with patch.object(Image.Image,"save",side_effect=OSError("disk full")):
                with self.assertRaisesRegex(ValueError,"disk full"):
                    prepare_crop_dataset(*args)
            self.assertEqual(list((root/"out").iterdir()),[])
            with self.assertRaises(InterruptedError):
                prepare_crop_dataset(*args,should_stop=lambda:True)
            self.assertEqual(list((root/"out").iterdir()),[])
            Image.new("RGB",(10,10)).save(root/"data/good/a.jpg")
            with self.assertRaisesRegex(ValueError,"중복"):
                prepare_crop_dataset(*args)
            self.assertEqual(list((root/"out").iterdir()),[])


RUNTIME = all(importlib.util.find_spec(name) for name in ("torch","torchvision","unsupported","PySide6"))


@unittest.skipUnless(RUNTIME,"Actual CPU training/inference dependencies required")
class AllTaskCropRuntimeTests(unittest.TestCase):
    def test_custom_model_reload_matches_explicit_crop_for_all_four_tasks(self):
        import torch
        from model import CustomCSP
        from checkpoint import make_checkpoint_metadata
        from core.inference_loading import load_cpu_engine
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            crop={"width":40,"height":32}
            pixels=np.random.default_rng(12).integers(0,256,(57,71,3),dtype=np.uint8)
            Image.fromarray(pixels).save(root/"source.png")
            Image.fromarray(pixels).crop(center_crop_box((71,57),crop)).save(root/"roi.png")
            for task in ("classify","detect","segment","anomaly"):
                with self.subTest(task=task):
                    architecture={"backbone_channels":[8,16,32,64,128],"csp_depth":[1,1,1,1],"dropout":0.0}
                    model=CustomCSP(task=task,num_classes=2,in_channels=3,**architecture).eval()
                    metadata=make_checkpoint_metadata(task,2,["a","b"],64,3,architecture,center_crop=crop)
                    if task == "anomaly":
                        # This synthetic checkpoint needs a saved boundary to return a calibrated result.
                        metadata["anomaly_threshold"] = .5
                    metadata["model_state_dict"]=model.state_dict()
                    torch.save(metadata,root/"crop.pt")
                    metadata.pop("center_crop")
                    metadata["preprocessing"].pop("center_crop")
                    torch.save(metadata,root/"plain.pt")
                    cropped=load_cpu_engine(root/"crop.pt",gradcam=True)
                    plain=load_cpu_engine(root/"plain.pt",gradcam=True)
                    seen=[]
                    hook=cropped.model.register_forward_pre_hook(lambda _,args:seen.append(args[0].detach().clone()))
                    result=cropped.infer(str(root/"source.png"))
                    hook.remove()
                    reference=[]
                    hook=plain.model.register_forward_pre_hook(lambda _,args:reference.append(args[0].detach().clone()))
                    expected=plain.infer(str(root/"roi.png"))
                    hook.remove()
                    self.assertEqual(result.status,"ok",result.error)
                    self.assertEqual(expected.status,"ok",expected.error)
                    torch.testing.assert_close(seen[0],reference[0],rtol=0,atol=0)
                    self.assertEqual(result.details["crop_box"],[15,12,55,44])
                    self.assertEqual(cropped._current_preview_rgb.shape,pixels.shape)
                    if task=="classify":
                        np.testing.assert_allclose(result.details["probabilities"],expected.details["probabilities"])
                    if task=="anomaly":
                        self.assertEqual(result.score,expected.score)
                    if task=="segment":
                        self.assertEqual(sum(result.details["pixel_counts"].values()),40*32)
                    if cropped._heatmap_cache:
                        valid=cropped._heatmap_cache["valid_mask"]
                        self.assertEqual(int(valid.sum()),40*32)
                    from core.inference_region import read_input_region
                    json_crop = {"width": 24, "height": 16}
                    json_path = root / "input.json"
                    json_path.write_text(json.dumps({"preprocessing": {"center_crop": json_crop}}))
                    request = read_input_region("json", json_path)
                    json_path.write_text('{"center_crop": null}')
                    before = (root/"crop.pt").read_bytes()
                    for choice, region in ((request, json_crop), ({"mode": "full"}, None), (None, crop)):
                        cropped.configure_input_region(choice)
                        seen.clear()
                        reference.clear()
                        with Image.open(root/"source.png") as source:
                            source.crop(center_crop_box(source.size, region)).save(root/"selected.png")
                        hook = cropped.model.register_forward_pre_hook(lambda _,args:seen.append(args[0].detach().clone()))
                        actual = cropped.infer(str(root/"source.png"))
                        hook.remove()
                        hook = plain.model.register_forward_pre_hook(lambda _,args:reference.append(args[0].detach().clone()))
                        explicit = plain.infer(str(root/"selected.png"))
                        hook.remove()
                        self.assertEqual(actual.status, "ok", actual.error)
                        self.assertEqual(explicit.status, "ok", explicit.error)
                        torch.testing.assert_close(seen[0], reference[0], rtol=0, atol=0)
                        self.assertEqual(actual.details["input_region"]["center_crop"], region)
                        self.assertEqual(actual.details["input_region"]["saved_center_crop"], crop)
                        if region:
                            self.assertEqual(actual.details["crop_box"], list(center_crop_box((71,57),region)))
                        if task == "segment":
                            self.assertEqual(sum(actual.details["pixel_counts"].values()),
                                             region["width"]*region["height"] if region else 71*57)
                        if cropped._heatmap_cache:
                            valid = cropped._heatmap_cache["valid_mask"]
                            self.assertEqual(int(valid.sum()) if valid is not None else 71*57,
                                             region["width"]*region["height"] if region else 71*57)
                    self.assertEqual((root/"crop.pt").read_bytes(), before)
                    Image.new("RGB",(8,8)).save(root/"small.png")
                    self.assertEqual(cropped.infer(str(root/"small.png")).status,"error")
                    self.assertEqual(cropped.infer(str(root/"source.png")).status,"ok")

    def test_qt_crop_is_visible_and_persistent_for_every_task(self):
        from PySide6.QtWidgets import QApplication
        from core.project import ProjectData
        from widgets.training_widget import TrainingWidget
        app=QApplication.instance() or QApplication([])
        widget=TrainingWidget()
        try:
            for task in ("classify","detect","segment","anomaly","obb"):
                project=ProjectData(task=task)
                project.training.patchcore_crop_enabled=True
                project.training.patchcore_crop_width=320
                project.training.patchcore_crop_height=192
                widget.set_project(project)
                self.assertEqual(widget.mode_combo.model().item(widget.mode_combo.findData("custom")).isEnabled(), task != "obb")
                widget.show()
                app.processEvents()
                self.assertTrue(widget.pc_crop_check.isVisible(),task)
                self.assertTrue(widget.pc_crop_width_spin.isEnabled(),task)
                widget.pc_crop_height_spin.setValue(160)
                widget.collect_config()
                self.assertEqual(project.training.patchcore_crop_height,160)
        finally:
            widget.close()
            widget.deleteLater()
            app.processEvents()
