"""기존 분류 학습 루프를 사용하는 EfficientNet 초기화와 복원 계약."""

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from core.project import AugmentationConfig, ModelConfig, TrainingConfig
from core.trainer import TrainWorker
from efficientnet import EfficientNet, VARIANTS, load_imagenet
from efficientnet_contract import NATIVE_INPUT, LEGACY_GRAY_INPUT, checkpoint_input_contract, channel_description


MODES = {"efficientnet_finetune", "efficientnet_transfer", "efficientnet_resume", "efficientnet_scratch"}


def dataset_signature(data):
    """원본 파일의 경로, 크기와 수정 시각으로 재개 대상 변경을 확인한다."""
    root = Path(data.root).resolve()
    if not data.root or not root.is_dir():
        raise FileNotFoundError("분류 데이터 폴더 없음")
    items = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            stat = path.stat()
            items.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    value = {"root": str(root), "files": items, "val_split": data.val_split}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class EfficientNetTrainWorker(TrainWorker):
    engine_name = "efficientnet"

    def _prepare_training(self):
        cfg = self.project.training
        if self.project.task != "classify" or cfg.training_mode not in MODES:
            raise ValueError("EfficientNet은 분류 전용 학습 모드 필요")
        if cfg.efficientnet_model not in VARIANTS:
            raise ValueError("EfficientNet B0 또는 B1 선택 필요")
        self._source_checkpoint = None
        self._best_state = None
        self._weight_provenance = {}
        self._data_signature = dataset_signature(self.project.data)
        if cfg.training_mode in {"efficientnet_finetune", "efficientnet_scratch"}:
            return
        source = self.project.model.pretrained_weights
        if not source or not Path(source).is_file():
            raise FileNotFoundError("EfficientNet 로컬 가중치 파일 선택 필요")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("EfficientNet 체크포인트 형식 오류")
        self._source_checkpoint = checkpoint
        if cfg.training_mode != "efficientnet_resume":
            return
        from opencv_preprocess import require_resume_contract
        require_resume_contract(checkpoint.get("preprocessing"))
        required = {"training_config", "model_settings", "optimizer_state_dict",
                    "scheduler_state_dict", "rng_state", "best_state_dict", "resume_compatible",
                    "metrics_history", "lr_history", "patience_counter", "epoch", "best_epoch", "best_metric"}
        if (checkpoint.get("engine") != "efficientnet" or not required.issubset(checkpoint)
                or not checkpoint.get("resume_compatible") or checkpoint.get("debug_run")
                or checkpoint["optimizer_state_dict"] is None):
            raise ValueError("중단 재개에는 학습 상태가 포함된 EfficientNet last.pt 필요")
        saved = dict(checkpoint["training_config"])
        # 이전 optimizer의 파라미터 그룹 수와 정확한 재개 상태를 보존한다.
        saved.setdefault("efficientnet_no_decay", False)
        epoch = checkpoint["epoch"]
        if not isinstance(epoch, int) or epoch < 1 or epoch >= saved["epochs"]:
            raise ValueError("완료된 학습은 '내 가중치로 추가 학습' 선택 필요")
        patience = saved.get("early_stop_patience", 0)
        if patience > 0 and checkpoint["patience_counter"] >= patience:
            raise ValueError("조기 종료된 학습은 내 가중치로 추가 학습 선택 필요")
        if checkpoint.get("data_signature") != self._data_signature:
            raise ValueError("학습 재개 데이터 변경 감지: 내 가중치로 추가 학습 선택 필요")
        saved["augmentation"] = AugmentationConfig(**saved["augmentation"])
        saved.update(training_mode="efficientnet_resume", device=cfg.device,
                     layer_debug_enabled=cfg.layer_debug_enabled,
                     layer_debug_patterns=cfg.layer_debug_patterns,
                     layer_debug_batches=cfg.layer_debug_batches)
        self.project.training = TrainingConfig(**saved)
        self.project.model = ModelConfig(**checkpoint["model_settings"])
        self.project.model.pretrained_weights = source
        self.signals.log_message.emit(f"  재개 설정 복원: epoch {epoch + 1}, 모델 {saved['efficientnet_model']}")

    def _build_model(self):
        cfg = self.project.training
        if cfg.augmentation.vertical_flip or cfg.augmentation.mixup_alpha:
            raise ValueError("EfficientNet의 수직 반전과 Mixup은 미지원: 두 값을 0으로 설정 필요")
        if self._source_checkpoint is not None and cfg.training_mode == "efficientnet_resume":
            if self._source_checkpoint.get("class_names") != self.project.data.class_names:
                raise ValueError("재개 체크포인트와 데이터 클래스 순서 불일치")
        adapter = NATIVE_INPUT
        source = self._source_checkpoint
        if source is not None and "model_state_dict" in source:
            if source.get("engine") != "efficientnet":
                raise ValueError("EfficientNet 체크포인트 필요")
            adapter = checkpoint_input_contract(source, cfg.in_channels)["input_adapter"]
        model = EfficientNet(cfg.efficientnet_model, self.project.data.num_classes,
                             cfg.in_channels, self.project.model.dropout, input_adapter=adapter)
        self._efficientnet_definition = model.checkpoint_config()
        self.signals.log_message.emit("  EfficientNet: " + channel_description(self._efficientnet_definition))
        if adapter == LEGACY_GRAY_INPUT:
            self.signals.log_message.emit("  구형 가중치의 RGB 확장 구조를 유지합니다. "
                                          "실제 1ch Conv로 새 학습하려면 ImageNet 사전학습을 선택하세요.")
        return model

    def _initialize_model(self, model, weights_path, task):
        mode = self.project.training.training_mode
        if mode == "efficientnet_finetune":
            self._weight_provenance = load_imagenet(model, weights_path=weights_path or None)
        elif mode == "efficientnet_scratch":
            self._weight_provenance = {"source": "random", "architecture": model.architecture}
            self.signals.log_message.emit("  EfficientNet 무작위 초기화에서 학습 시작")
            return False
        else:
            checkpoint = self._source_checkpoint
            if "model_state_dict" not in checkpoint:
                if mode == "efficientnet_resume":
                    raise ValueError("ImageNet 가중치에는 중단 재개 상태 없음")
                self._weight_provenance = load_imagenet(model, weights_path=weights_path)
            else:
                if (checkpoint.get("engine") != "efficientnet"
                        or checkpoint.get("model_config", {}).get("architecture") != model.architecture):
                    raise ValueError("EfficientNet 아키텍처와 체크포인트 불일치")
                contract = checkpoint_input_contract(checkpoint, model.in_channels)
                if contract["input_adapter"] != model.input_adapter:
                    raise ValueError("EfficientNet 입력 구조와 체크포인트 불일치")
                state = dict(checkpoint["model_state_dict"])
                changed_classes = checkpoint.get("class_names") != self.project.data.class_names
                if changed_classes:
                    if mode == "efficientnet_resume":
                        raise ValueError("클래스 변경 모델의 학습 재개 불가")
                    state["classifier.1.weight"] = model.classifier[1].weight.detach().cpu()
                    state["classifier.1.bias"] = model.classifier[1].bias.detach().cpu()
                model.load_state_dict(state, strict=True)
                self._weight_provenance = {"local_checkpoint": weights_path,
                                          "classifier_reinitialized": changed_classes}
        self.signals.log_message.emit("  EfficientNet 가중치 검증 및 로드 완료: " +
                                      json.dumps(self._weight_provenance, ensure_ascii=False))
        return True

    def _checkpoint_metadata(self):
        result = super()._checkpoint_metadata()
        cfg = self.project.training
        result.update(engine="efficientnet", architecture_name=cfg.efficientnet_model,
                      model_config=dict(self._efficientnet_definition),
                      source_weights=dict(self._weight_provenance))
        if cfg.in_channels == 1:
            result["preprocessing"]["grayscale_adapter"] = self._efficientnet_definition["input_adapter"]
        return result

    def _create_optimizer(self, model, cfg, model_cfg):
        if not cfg.efficientnet_no_decay:
            return super()._create_optimizer(model, cfg, model_cfg)
        groups = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            lr = cfg.learning_rate * (model_cfg.backbone_lr_mult if name.startswith("features.") else 1.0)
            decay = 0.0 if parameter.ndim <= 1 or name.endswith(".bias") else cfg.weight_decay
            groups.setdefault((lr, decay), []).append(parameter)
        params = [{"params": values, "lr": lr, "weight_decay": decay}
                  for (lr, decay), values in groups.items()]
        self.signals.log_message.emit("  EfficientNet: BatchNorm과 bias의 weight decay 제외")
        if cfg.optimizer == "sgd":
            return torch.optim.SGD(params, lr=cfg.learning_rate, momentum=.9)
        optimizer = torch.optim.Adam if cfg.optimizer == "adam" else torch.optim.AdamW
        return optimizer(params, lr=cfg.learning_rate)

    def _checkpoint_extra(self, model, is_better, metrics_history, lr_history, patience):
        if is_better:
            # CPU에서도 이후 optimizer.step이 Best 텐서를 변경하지 못하게 복사한다.
            self._best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        rng = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {"training_config": asdict(self.project.training),
                "model_settings": asdict(self.project.model),
                "data_signature": self._data_signature, "resume_compatible": True,
                "debug_run": bool(getattr(self, "debug_batch_limit", None)),
                "rng_state": rng, "best_state_dict": self._best_state,
                "metrics_history": copy.deepcopy(metrics_history), "lr_history": list(lr_history),
                "patience_counter": patience}

    def _restore_training_state(self, model, optimizer, scheduler, scaler, run_dir):
        if self.project.training.training_mode != "efficientnet_resume":
            return {}
        checkpoint = self._source_checkpoint
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            if checkpoint["scheduler_state_dict"] is None:
                raise ValueError("학습률 스케줄러 복원 상태 없음")
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self._best_state = copy.deepcopy(checkpoint["best_state_dict"])
        previous_times = checkpoint["metrics_history"].get("elapsed_time_sec", [])
        if previous_times:
            self._training_clock.started -= float(previous_times[-1])
        previous_best = dict(checkpoint)
        previous_best.update(model_state_dict=self._best_state, epoch=checkpoint["best_epoch"], resume_compatible=False)
        self._save_checkpoint(previous_best, str(Path(run_dir) / "best.pt"))
        rng = checkpoint["rng_state"]
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        if "cuda" in rng and torch.cuda.is_available():
            if len(rng["cuda"]) != torch.cuda.device_count():
                raise ValueError("재개 체크포인트와 CUDA 장치 수 불일치")
            torch.cuda.set_rng_state_all(rng["cuda"])
        return {"next_epoch": checkpoint["epoch"] + 1,
                "best_metric": checkpoint["best_metric"], "best_epoch": checkpoint["best_epoch"],
                "patience_counter": checkpoint["patience_counter"],
                "metrics_history": copy.deepcopy(checkpoint["metrics_history"]),
                "lr_history": list(checkpoint["lr_history"])}
