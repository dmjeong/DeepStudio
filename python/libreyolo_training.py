"""Pinned native training with Studio's full-frame classification transforms."""
from pathlib import Path

import torch
from libreyolo.models.mobilenetv4.trainer import MobileNetV4Trainer
from libreyolo.validation import ClassifyValidator, ValidationConfig
from libreyolo_classification import FullFrameTransform, preprocessing_contract


def _set_transform(loader, size, *, augment=False, augmentation=None, patches=None):
    # ClassifyDataset wraps ImageFolder in the pinned LibreYOLO 1.5 runtime.
    dataset = loader.dataset
    if not hasattr(dataset, "_impl") or not hasattr(dataset._impl, "transform"):
        raise RuntimeError("LibreYOLO 데이터셋 구조가 변경되어 전체 영상 전처리를 연결할 수 없습니다")
    if patches:
        from native_patches import PatchBagTransform
        dataset._impl.transform = PatchBagTransform(patches, augment=augment, augmentation=augmentation)
    else:
        dataset._impl.transform = FullFrameTransform(size, augment=augment, augmentation=augmentation)
    return loader


class FullFrameValidator(ClassifyValidator):
    def __init__(self, *args, studio_patches=None, **kwargs):
        self.studio_patches = studio_patches
        super().__init__(*args, **kwargs)

    def _setup_dataloader(self):
        return _set_transform(super()._setup_dataloader(), self.config.imgsz, patches=self.studio_patches)

    def _inference(self, images):
        if self.studio_patches:
            from native_patches import bag_logits
            with self._autocast_context():
                return bag_logits(self.model.model, images.to(self.device), self.studio_patches["batch"])
        return super()._inference(images)


class FullFrameTrainer(MobileNetV4Trainer):
    def __init__(self, *args, studio_center_crop=None, studio_augmentation=None, studio_patches=None, **kwargs):
        from center_crop import validate_center_crop
        from studio_augmentation import augmentation_contract
        self.studio_center_crop = validate_center_crop(studio_center_crop)
        self.studio_augmentation = augmentation_contract(studio_augmentation)
        self.studio_patches = studio_patches
        super().__init__(*args, **kwargs)

    def _setup_classify_data(self):
        if self.studio_patches:
            # The native image classifier rejects batch=1 for its pooled BN head.
            # Our bag forward batches tiles instead; build the native class mapping
            # first, then replace only its sampler/loader with one original per bag.
            from torch.utils.data import DataLoader
            requested = self.config.batch
            self.config.batch = max(2, requested)
            try:
                super()._setup_classify_data()
            finally:
                self.config.batch = requested
            previous = self.train_loader
            self.train_loader = DataLoader(previous.dataset, batch_size=1, shuffle=True,
                num_workers=previous.num_workers, collate_fn=previous.collate_fn,
                pin_memory=previous.pin_memory, persistent_workers=previous.num_workers > 0)
        else:
            super()._setup_classify_data()
        _set_transform(self.train_loader, self.config.imgsz, augment=True,
                       augmentation=self.studio_augmentation, patches=self.studio_patches)

    def on_forward(self, imgs, targets, polygons=None):
        if not self.studio_patches:
            return super().on_forward(imgs, targets, polygons)
        from native_patches import bag_logits
        logits = bag_logits(self.model, imgs, self.studio_patches["batch"])
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return {"total_loss": loss, "loss_ce": loss.detach()}

    def _checkpoint_extra_metadata(self):
        return {**super()._checkpoint_extra_metadata(),
                "center_crop": self.studio_center_crop,
                "studio_augmentation": self.studio_augmentation,
                "studio_patches": self.studio_patches,
                "studio_preprocessing": preprocessing_contract(self.config.imgsz)}

    def resume(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("studio_preprocessing") is None:
            raise ValueError("중앙 자르기를 사용한 구형 LibreYOLO 학습은 중단 재개할 수 없습니다. "
                             "기존 가중치로 추가 학습을 선택하세요. 추론에는 그대로 사용할 수 있습니다.")
        preprocessing_contract(self.config.imgsz, checkpoint["studio_preprocessing"])
        if checkpoint.get("studio_patches") != self.studio_patches:
            raise ValueError("중단 재개 모델의 패치 설정이 다릅니다. 추가 학습을 선택하세요.")
        if checkpoint.get("studio_augmentation", {"horizontal_flip": .5, "rotation": 0., "color_jitter": 0.}) != self.studio_augmentation:
            raise ValueError("재개 모델의 증강 설정이 다릅니다. 저장된 설정을 사용하세요.")
        from center_crop import checkpoint_center_crop
        if checkpoint_center_crop(checkpoint) != self.studio_center_crop:
            raise ValueError("중단 재개는 저장된 입력 영역과 같아야 합니다. 영역 변경은 추가 학습을 사용하세요.")
        return super().resume(checkpoint_path)

    def _run_classify_validation(self, epoch):
        # Keep native metrics, label mapping, EMA and validation loss machinery.
        # Select our validator explicitly instead of replacing global classes.
        from libreyolo.training.distributed import unwrap_model
        config = ValidationConfig(
            data=self.config.data, imgsz=self.config.imgsz,
            batch_size=1 if self.studio_patches else max(1, self.config.batch // max(self.world_size, 1)),
            device=str(self.device), half=self.config.amp and self.device.type == "cuda",
            amp_dtype=self.config.amp_dtype, verbose=False,
            num_workers=self.config.workers, split="val")
        network = self.ema_model.ema if self.ema_model else unwrap_model(self.model)
        original = self.wrapper_model.model
        self.wrapper_model.model = network
        try:
            result = FullFrameValidator(self.wrapper_model, config,
                                       studio_patches=self.studio_patches,
                                       **self._validation_loss_kwargs(network)).run()
        finally:
            self.wrapper_model.model = original
        metrics = self._scalar_mapping(result)
        top1 = metrics["metrics/accuracy_top1"]
        return {"mAP50": top1, "mAP50_95": top1, "best_metric": top1,
                "best_metric_key": "metrics/accuracy_top1", "metrics": metrics}


def train_classifier(wrapper, *, resume=False, center_crop=None, augmentation=None, patches=None, **settings):
    if patches:
        # Original image labels are preserved; one image contains variable K tiles.
        # The requested batch controls each patch forward, not label duplication.
        settings["batch"] = 1
        settings["cuda_graph"] = False
    wrapper._seed_scratch_initialization(settings.get("seed", 0))
    trainer = FullFrameTrainer(model=wrapper.model, wrapper_model=wrapper, size=wrapper.size,
                               num_classes=wrapper.nb_classes, resume=resume,
                               studio_center_crop=center_crop, studio_augmentation=augmentation,
                               studio_patches=patches, **settings)
    if resume:
        if not wrapper.model_path:
            raise ValueError("중단 재개할 LibreYOLO 체크포인트가 없습니다")
        trainer.setup()
        trainer.resume(str(wrapper.model_path))
    result = trainer.train()
    result["applied_augmentation"] = trainer.studio_augmentation
    result["patches"] = patches
    best = result.get("best_checkpoint")
    if best and Path(best).is_file():
        wrapper._load_weights(str(best))
    return result
