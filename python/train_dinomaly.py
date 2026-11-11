"""Dinomaly normal-only training, validation-only selection and calibration.

Test images are never used for selection or threshold fitting. With no labelled
NG validation examples the threshold is explicitly a normal-only limit.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from dense_deployment import PreparedWindows, region_pixels
from dinomaly import Dinomaly, load_checkpoint
from dinomaly_runtime import DinomalyRuntime, deployment_config, read_bgr
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def discover(root):
    from patchcore_data import NORMAL_NAMES, image_paths

    root = Path(root)
    training_root = root / "train"
    if not training_root.is_dir():
        training_root = root
    folders = [p for p in training_root.iterdir() if p.is_dir()]
    normals = [p for p in folders if p.name.casefold() in NORMAL_NAMES]
    if normals:
        training = [f for p in normals for f in image_paths(p)]
    elif not folders or training_root.name.casefold() in NORMAL_NAMES:
        training = image_paths(training_root)
    else:
        raise ValueError(
            "Dinomaly는 train/good 또는 train/normal의 정상 이미지만 학습합니다"
        )
    validation, labels = [], []
    if training_root != root and (root / "val").is_dir():
        for path in image_paths(root / "val"):
            parts = Path(path).relative_to(root / "val").parts
            if len(parts) < 2:
                raise ValueError(
                    "검증 이미지는 val/good 또는 val/불량명 폴더에 넣으세요"
                )
            validation.append(path)
            labels.append(0 if parts[0].casefold() in NORMAL_NAMES else 1)
    if not training:
        raise ValueError("정상 학습 이미지가 없습니다")
    # No test fallback. A deterministic held-out normal split when val is absent.
    if not any(label == 0 for label in labels):
        if len(training) < 2:
            raise ValueError("정상 학습·검증에 최소 2장이 필요합니다")
        count = max(1, len(training) // 5)
        held = training[-count:]
        training = training[:-count]
        validation = held + validation
        labels = [0] * len(held) + labels
    return training, validation, labels


class NormalDataset(Dataset):
    def __init__(
        self,
        paths,
        config,
        *,
        training=False,
        patches_per_image=0,
        horizontal_flip=0.5,
        rotation=15.0,
        color_jitter=0.2,
    ):
        self.paths, self.config, self.training = list(paths), config, training
        self.sample_random = training and patches_per_image > 0
        self.flip, self.rotation = horizontal_flip, rotation
        self.jitter = (
            T.ColorJitter(color_jitter, color_jitter, color_jitter)
            if training and color_jitter
            else None
        )
        self.items = []
        for path in paths:
            count = len(PreparedWindows(config, region_pixels(config, read_bgr(path))))
            self.items.extend(
                (path, index, count)
                for index in range(
                    min(count, patches_per_image)
                    if training and patches_per_image
                    else count
                )
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        import cv2

        path, tile, count = self.items[index]
        image = read_bgr(path)
        if self.jitter:
            if image.ndim == 2:
                image = np.asarray(self.jitter(Image.fromarray(image)))
            else:
                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB if image.shape[2] == 3 else cv2.COLOR_BGRA2RGB,
                )
                image = cv2.cvtColor(
                    np.asarray(self.jitter(Image.fromarray(rgb))), cv2.COLOR_RGB2BGR
                )
        if self.sample_random:
            tile = random.randrange(count)
        tensor = PreparedWindows(self.config, region_pixels(self.config, image))[tile][
            2
        ][0]
        tensor = torch.from_numpy(tensor.copy())
        if self.training:
            if random.random() < self.flip:
                tensor = tensor.flip(-1)
            if self.rotation:
                tensor = T.functional.rotate(
                    tensor,
                    random.uniform(-self.rotation, self.rotation),
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=0.0,
                )
        return tensor


def train_dinomaly(
    data_root,
    *,
    output_dir,
    input_size=224,
    in_channels=3,
    epochs=20,
    batch_size=4,
    learning_rate=2e-4,
    weight_decay=1e-4,
    optimizer_name="adamw",
    scheduler_name="cosine",
    warmup_epochs=0,
    early_stop_patience=0,
    device="cpu",
    pretrained=True,
    initial_weights=None,
    resume=None,
    patch_training=False,
    patch_scale=1.0,
    patches_per_image=0,
    center_crop=None,
    horizontal_flip=0.5,
    rotation=15.0,
    color_jitter=0.2,
    num_workers=0,
    seed=0,
    log=print,
    should_stop=lambda: False,
):
    if (
        epochs < 1
        or batch_size < 1
        or input_size < 32
        or input_size > 4096
        or in_channels not in (1, 3)
    ):
        raise ValueError("Dinomaly 학습 크기·배치·에폭 설정 오류")
    if (
        learning_rate <= 0
        or weight_decay < 0
        or not 0 <= horizontal_flip <= 1
        or rotation < 0
        or not 0 <= color_jitter <= 1
    ):
        raise ValueError("Dinomaly 학습률·증강 설정 오류")
    if (
        patches_per_image < 0
        or num_workers < 0
        or warmup_epochs < 0
        or early_stop_patience < 0
    ):
        raise ValueError("Dinomaly 패치·작업자·스케줄 설정 오류")
    if not pretrained and not initial_weights and not resume:
        raise ValueError("Dinomaly 고정 인코더에는 사전학습 가중치가 필요합니다")
    if sum(bool(v) for v in (pretrained, initial_weights, resume)) != 1:
        raise ValueError("가중치 시작 방식을 하나만 선택하세요")
    from center_crop import validate_center_crop
    from checkpoint import tiling_contract

    center_crop = validate_center_crop(center_crop)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tiling = tiling_contract(input_size, patch_scale) if patch_training else None
    if tiling:
        tiling["blend"] = "center_weighted_map"
    meta = {
        "type": "dinomaly",
        "backend": "dinomaly",
        "model_id": "dinomaly_vits14",
        "architecture_version": 1,
        "task": "anomaly",
        "input_size": input_size,
        "in_channels": in_channels,
        "center_crop": center_crop,
        "tiling": tiling,
    }
    contract = deployment_config(meta)
    train_paths, val_paths, val_labels = discover(data_root)
    datasets = [
        NormalDataset(
            paths,
            contract,
            training=training,
            patches_per_image=patches_per_image,
            horizontal_flip=horizontal_flip,
            rotation=rotation,
            color_jitter=color_jitter,
        )
        for paths, training in (
            (train_paths, True),
            ([p for p, y in zip(val_paths, val_labels) if y == 0], False),
        )
    ]
    target = torch.device(device)
    loaders = [
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=i == 0,
            num_workers=num_workers,
            pin_memory=target.type == "cuda",
        )
        for i, ds in enumerate(datasets)
    ]
    source = (
        torch.load(resume or initial_weights, map_location="cpu", weights_only=True)
        if resume or initial_weights
        else None
    )
    if (
        source
        and source.get("type") == "dinomaly"
        and source.get("in_channels") != in_channels
    ):
        raise ValueError("Dinomaly 체크포인트 입력 채널이 다릅니다")
    if source and source.get("type") == "dinomaly":
        model = load_checkpoint(source, target)
    else:
        model = Dinomaly(
            in_channels, pretrained=pretrained, weights_path=initial_weights
        ).to(target)
    optimizers = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    if optimizer_name not in optimizers or scheduler_name not in {
        "cosine",
        "step",
        "none",
    }:
        raise ValueError("미지원 optimizer/scheduler")
    optimizer = optimizers[optimizer_name](
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def factor(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        if scheduler_name == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        if scheduler_name == "step":
            return 0.1 ** int(progress * 3)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    settings = {
        "model_id": "dinomaly_vits14",
        "input_size": input_size,
        "in_channels": in_channels,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "warmup_epochs": warmup_epochs,
        "early_stop_patience": early_stop_patience,
        "augmentation": {
            "horizontal_flip": horizontal_flip,
            "rotation": rotation,
            "color_jitter": color_jitter,
        },
        "tiling": tiling,
        "patches_per_image": patches_per_image,
        "center_crop": center_crop,
        "seed": seed,
        "encoder_frozen": True,
        "hard_fraction": 0.9,
        "pretrained": pretrained,
        "num_workers": num_workers,
    }
    start = 0
    history = []
    best = float("inf")
    best_epoch = 0
    if resume:
        for key in ("input_size", "in_channels", "tiling", "center_crop"):
            if source.get(key) != meta.get(key):
                raise ValueError(f"Dinomaly 재개 입력 계약 불일치: {key}")
        for key in (
            "batch_size",
            "learning_rate",
            "weight_decay",
            "optimizer",
            "scheduler",
            "warmup_epochs",
            "augmentation",
            "patches_per_image",
            "seed",
        ):
            if source["training_config"].get(key) != settings[key]:
                raise ValueError(f"Dinomaly 재개 학습 설정 불일치: {key}")
        optimizer.load_state_dict(source["optimizer_state_dict"])
        scheduler.load_state_dict(source["scheduler_state_dict"])
        start = int(source["epoch"])
        history = source["metrics_history"]
        best = float(source["best_loss"])
        best_epoch = int(source["best_epoch"])
        if epochs <= start:
            raise ValueError(f"총 에폭은 저장된 {start}에폭보다 커야 합니다")
        rng = source.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
            np.random.set_state(
                (
                    rng["numpy"][0],
                    np.asarray(rng["numpy"][1], np.uint32),
                    *rng["numpy"][2:],
                )
            )
            if target.type == "cuda" and rng.get("cuda"):
                torch.cuda.set_rng_state_all(rng["cuda"])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if resume:
        from checkpoint import previous_best_path

        previous = previous_best_path(resume)
        if previous is None:
            raise ValueError(
                "재개하려면 같은 실행 폴더의 Best 체크포인트와 선정 기록도 필요합니다"
            )
        previous_state = torch.load(previous, map_location="cpu", weights_only=True)
        if (
            previous_state.get("model_id") != "dinomaly_vits14"
            or previous_state.get("epoch") != best_epoch
            or previous_state.get("metric") != best
        ):
            raise ValueError("재개 체크포인트와 Best 선정 기록이 다릅니다")
        torch.save(previous_state, destination / "best.pt")
    begun = time.perf_counter()
    stopped = False
    for epoch in range(start, epochs):
        epoch_start = time.perf_counter()
        losses = []
        for i, loader in enumerate(loaders):
            model.train(i == 0)
            total = seen = 0
            with torch.set_grad_enabled(i == 0):
                for images in loader:
                    if should_stop():
                        stopped = True
                        break
                    images = images.to(target, non_blocking=target.type == "cuda")
                    loss = model.training_loss(images)
                    if not torch.isfinite(loss):
                        raise ValueError("Dinomaly 손실 NaN/Inf")
                    if i == 0:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    total += float(loss.detach()) * len(images)
                    seen += len(images)
            if stopped:
                break
            losses.append(total / seen)
        if stopped:
            break
        scheduler.step()
        metrics = {
            "val_loss": losses[1],
            "epoch_time_sec": time.perf_counter() - epoch_start,
            "elapsed_time_sec": time.perf_counter() - begun,
        }
        row = {
            "epoch": epoch + 1,
            "train_loss": losses[0],
            "val_loss": losses[1],
            "val_metric": losses[1],
            "selected_value": losses[1],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "metrics": metrics,
        }
        history.append(row)
        improved = losses[1] < best
        if improved:
            best, best_epoch = losses[1], epoch + 1
        numpy_rng = np.random.get_state()
        state = {
            **meta,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "epoch": epoch + 1,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_config": settings,
            "metrics_history": history,
            "best_epoch": best_epoch,
            "best_loss": best,
            "metric": losses[1],
            "weight_provenance": model.weight_provenance,
            "rng_state": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "numpy": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                "cuda": torch.cuda.get_rng_state_all() if target.type == "cuda" else [],
            },
        }
        torch.save(state, destination / "last.pt")
        if improved:
            torch.save(state, destination / "best.pt")
        log(
            f"Dinomaly {epoch + 1}/{epochs} · train {losses[0]:.6f} · val {losses[1]:.6f} · {metrics['epoch_time_sec']:.2f}초"
        )
        log(
            {
                "event": "epoch_finished",
                "total_epochs": epochs,
                "metric": losses[1],
                **row,
            }
        )
        if improved:
            log({"event": "best_epoch_updated", **row})
        if early_stop_patience and epoch + 1 - best_epoch >= early_stop_patience:
            break
    best_path = destination / "best.pt"
    if best_path.is_file() and not stopped:
        state = torch.load(best_path, map_location="cpu", weights_only=True)
        runtime = DinomalyRuntime(state, target)
        scores = []
        for path in val_paths:
            if should_stop():
                stopped = True
                break
            scores.append(
                float(runtime.predict_from_path(path, center_crop=center_crop)[0][0])
            )
        if not stopped:
            normal = [s for s, y in zip(scores, val_labels) if y == 0]
            threshold = float(np.nextafter(np.float32(max(normal)), np.float32(np.inf)))
            threshold = max(threshold, float(np.finfo(np.float32).eps))
            state["anomaly_threshold"] = threshold
            state["calibration"] = {
                "source": "validation_only",
                "method": "above_max_normal_v1",
                "normal_count": len(normal),
                "ng_count": sum(val_labels),
                "ng_detected": sum(
                    s >= threshold for s, y in zip(scores, val_labels) if y
                ),
                "normal_only": not any(val_labels),
                "test_used": False,
            }
            torch.save(state, best_path)
            log(
                f"Dinomaly 검증 정상 상한 보정: {threshold:.6f} · {state['calibration']}"
            )
    (destination / "training.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "completed_epochs": history[-1]["epoch"] if history else 0,
                "metrics_history": history,
                "training_config": settings,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_path
