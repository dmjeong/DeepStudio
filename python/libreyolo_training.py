"""Pinned native training with Studio's full-frame classification transforms."""
from pathlib import Path

import torch
from libreyolo.models.mobilenetv4.trainer import MobileNetV4Trainer
from libreyolo.validation import ClassifyValidator, ValidationConfig
from libreyolo_classification import FullFrameTransform, preprocessing_contract


def _set_transform(loader, size, *, augment=False):
    # ClassifyDataset wraps ImageFolder in the pinned LibreYOLO 1.5 runtime.
    dataset = loader.dataset
    if not hasattr(dataset, "_impl") or not hasattr(dataset._impl, "transform"):
        raise RuntimeError("LibreYOLO 데이터셋 구조가 변경되어 전체 영상 전처리를 연결할 수 없습니다")
    dataset._impl.transform = FullFrameTransform(size, augment=augment)
    return loader


class FullFrameValidator(ClassifyValidator):
    def _setup_dataloader(self):
        return _set_transform(super()._setup_dataloader(), self.config.imgsz)


class FullFrameTrainer(MobileNetV4Trainer):
    def __init__(self, *args, studio_center_crop=None, **kwargs):
        from center_crop import validate_center_crop
        self.studio_center_crop = validate_center_crop(studio_center_crop)
        super().__init__(*args, **kwargs)

    def _setup_classify_data(self):
        super()._setup_classify_data()
        _set_transform(self.train_loader, self.config.imgsz, augment=True)

    def _checkpoint_extra_metadata(self):
        return {**super()._checkpoint_extra_metadata(),
                "center_crop": self.studio_center_crop,
                "studio_preprocessing": preprocessing_contract(self.config.imgsz)}

    def resume(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("studio_preprocessing") is None:
            raise ValueError("중앙 자르기를 사용한 구형 LibreYOLO 학습은 중단 재개할 수 없습니다. "
                             "기존 가중치로 추가 학습을 선택하세요. 추론에는 그대로 사용할 수 있습니다.")
        preprocessing_contract(self.config.imgsz, checkpoint["studio_preprocessing"])
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
            batch_size=max(1, self.config.batch // max(self.world_size, 1)),
            device=str(self.device), half=self.config.amp and self.device.type == "cuda",
            amp_dtype=self.config.amp_dtype, verbose=False,
            num_workers=self.config.workers, split="val")
        network = self.ema_model.ema if self.ema_model else unwrap_model(self.model)
        original = self.wrapper_model.model
        self.wrapper_model.model = network
        try:
            result = FullFrameValidator(self.wrapper_model, config,
                                       **self._validation_loss_kwargs(network)).run()
        finally:
            self.wrapper_model.model = original
        metrics = self._scalar_mapping(result)
        top1 = metrics["metrics/accuracy_top1"]
        return {"mAP50": top1, "mAP50_95": top1, "best_metric": top1,
                "best_metric_key": "metrics/accuracy_top1", "metrics": metrics}


def train_classifier(wrapper, *, resume=False, center_crop=None, **settings):
    trainer = FullFrameTrainer(model=wrapper.model, wrapper_model=wrapper, size=wrapper.size,
                               num_classes=wrapper.nb_classes, resume=resume,
                               studio_center_crop=center_crop, **settings)
    if resume:
        if not wrapper.model_path:
            raise ValueError("중단 재개할 LibreYOLO 체크포인트가 없습니다")
        trainer.setup()
        trainer.resume(str(wrapper.model_path))
    result = trainer.train()
    best = result.get("best_checkpoint")
    if best and Path(best).is_file():
        wrapper._load_weights(str(best))
    return result
