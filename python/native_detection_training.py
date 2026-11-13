"""Pinned native detector trainers with explicit, editable Studio augmentation."""
import numpy as np
from libreyolo.models.dfine.trainer import DFINETrainer
from libreyolo.models.rtdetrv4.trainer import RTDETRv4Trainer
from libreyolo.models.yolo9.trainer import YOLO9Trainer
from studio_augmentation import augment_rgb, augmentation_contract


class DetectionTransform:
    wants_unresized_image = True

    def __init__(self, family, size, settings):
        self.settings = augmentation_contract(settings)
        self.family, self.size = family, size

    def __call__(self, image, targets, input_dim):
        from native_detection import detector_pixels
        rgb, labels = augment_rgb(image[:, :, ::-1], self.settings, targets)
        pixels, (sx, sy) = detector_pixels(rgb, self.size, self.family)
        if len(labels) > 300:
            raise ValueError("한 패치의 박스가 300개를 넘습니다. 패치 크기를 조절하세요.")
        padded = np.zeros((300, 5), np.float32)
        if self.family == "yolo9":
            padded[:, 0] = -1
        if len(labels):
            boxes = labels[:, :4] * np.asarray([sx, sy, sx, sy], np.float32)
            if self.family == "yolo9":
                boxes /= self.size
            else:
                # Native DETR trainer expects pixel cxcywh; it normalizes in on_forward.
                boxes = np.column_stack(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]))
            padded[:len(labels), 0] = labels[:, 4]
            padded[:len(labels), 1:] = boxes
        return np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32) / 255, padded


class StudioDetectionMixin:
    def __init__(self, *args, studio_augmentation=None, studio_patches=None, studio_center_crop=None,
                 studio_small_object_coverage=False, studio_train_patch_limit=0, **kwargs):
        self.studio_augmentation = augmentation_contract(studio_augmentation)
        self.studio_patches = studio_patches
        self.studio_center_crop = studio_center_crop
        self.studio_small_object_coverage = studio_small_object_coverage
        self.studio_train_patch_limit = studio_train_patch_limit
        # Only user-visible augmentations run; no hidden mosaic, IoU crop or scale jitter.
        kwargs.update(mosaic_prob=0., mixup_prob=0., hsv_prob=0., degrees=0., translate=0.,
                      flip_prob=0., no_aug_epochs=0)
        if self.get_model_family() != "yolo9":
            kwargs["multi_scale"] = False
        super().__init__(*args, **kwargs)

    def create_transforms(self):
        from libreyolo.models.deim.transforms import DEIMPassThroughDataset
        return DetectionTransform(self.get_model_family(), self.config.imgsz, self.studio_augmentation), DEIMPassThroughDataset

    def _checkpoint_extra_metadata(self):
        metadata = {**super()._checkpoint_extra_metadata(), "studio_augmentation": self.studio_augmentation,
                    "studio_patches": self.studio_patches, "center_crop": self.studio_center_crop}
        if self.studio_small_object_coverage:
            metadata["studio_small_object_coverage"] = True
            metadata["studio_train_patch_limit"] = self.studio_train_patch_limit
        return metadata


class StudioYOLO9Trainer(StudioDetectionMixin, YOLO9Trainer):
    pass


class StudioRTDETRTrainer(StudioDetectionMixin, RTDETRv4Trainer):
    pass


class StudioDFINETrainer(StudioDetectionMixin, DFINETrainer):
    pass


def train_detector(wrapper, *, augmentation=None, patches=None, center_crop=None, resume=False,
                   small_object_coverage=False, train_patch_limit=0, **settings):
    from pathlib import Path

    from libreyolo.data import load_data_config
    if type(small_object_coverage) is not bool or (
        small_object_coverage and (not patches or type(train_patch_limit) is not int or not 1 <= train_patch_limit <= 256)
    ):
        raise ValueError("작은 객체 우선에는 탐지 패치 학습과 이미지당 패치 수 1~256이 필요합니다")
    if wrapper.FAMILY in {"dfine", "rtdetrv4"}:
        try:
            from pycocotools.coco import COCO  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("탐지 평가 모듈이 없습니다. gui/requirements-upstream-models.txt를 설치하세요.") from exc
    data = load_data_config(settings["data"], autodownload=False)
    if data["nc"] != wrapper.nb_classes:
        wrapper._rebuild_for_new_classes(data["nc"])
    wrapper._seed_scratch_initialization(settings.get("seed", 0))
    wrapper.names = {i: name for i, name in enumerate(data["names"])} if isinstance(data["names"], list) else data["names"]
    trainer_type = {"yolo9": StudioYOLO9Trainer, "rtdetrv4": StudioRTDETRTrainer, "dfine": StudioDFINETrainer}[wrapper.FAMILY]
    trainer = trainer_type(model=wrapper.model, wrapper_model=wrapper, size=wrapper.size,
        num_classes=wrapper.nb_classes, studio_augmentation=augmentation, studio_patches=patches,
        studio_center_crop=center_crop, studio_small_object_coverage=small_object_coverage,
        studio_train_patch_limit=train_patch_limit, resume=resume, **settings)
    if resume:
        import torch
        saved = torch.load(wrapper.model_path, map_location="cpu", weights_only=False)
        if (saved.get("studio_patches") != patches or saved.get("center_crop") != center_crop or
                saved.get("studio_augmentation") != trainer.studio_augmentation):
            raise ValueError("재개 모델의 패치/증강/입력 영역이 다릅니다. 설정을 맞추거나 추가 학습을 선택하세요.")
        if (saved.get("studio_small_object_coverage", False) != small_object_coverage or
                (small_object_coverage and saved.get("studio_train_patch_limit") != train_patch_limit)):
            raise ValueError("재개 모델의 작은 객체 우선 설정이 다릅니다. 설정을 맞추거나 추가 학습을 선택하세요.")
        trainer.setup()
        trainer.resume(str(wrapper.model_path))
    result = trainer.train()
    result["applied_augmentation"] = trainer.studio_augmentation
    result["patches"] = patches
    result["small_object_coverage"] = small_object_coverage
    best = result.get("best_checkpoint")
    if best and Path(best).is_file():
        wrapper._load_weights(str(best))
    return result
