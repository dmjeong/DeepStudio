"""웹 서버와 메모리를 공유하지 않는 계산 프로세스. GUI 모듈을 로드하지 않는다."""

from dataclasses import asdict
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from datetime import datetime

from webapp.storage import digest, json_value, read_json, restore_project, write_json
from webapp.locking import exclusive_file


class JobContext:
    def __init__(self, directory, monitor_parent=True, parent_pid=None, parent_started=None):
        self.directory = Path(directory)
        self.parent_closed = threading.Event()
        self.failure = ""
        self.monitor_done = threading.Event()
        if monitor_parent:
            threading.Thread(target=self._parent_process, args=(parent_pid, parent_started), daemon=True).start()

    def _parent_process(self, pid, started):
        # stdin.read 대기와 torch/cv2의 Windows 표준입력 검사를 겹치지 않는다.
        import psutil
        try:
            parent = psutil.Process(pid or os.getppid())
            while not self.monitor_done.is_set():
                if not parent.is_running() or (started is not None and parent.create_time() != started):
                    self.parent_closed.set()
                    return
                self.monitor_done.wait(.5)
        except psutil.NoSuchProcess:
            self.parent_closed.set()

    def cancelled(self):
        return self.parent_closed.is_set() or (self.directory / "cancel").exists()

    def emit(self, event, args):
        if event == "training_error":
            self.failure = str(args[0])
        message = json.dumps(json_value({"event": event, "args": args}), ensure_ascii=False, allow_nan=False)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def _container_model_spec(project):
    """Return a container-only catalog spec, if the project selects one."""
    model_id = getattr(project.model, "model_id", "")
    if not model_id:
        return None
    try:
        from core.model_registry import registry_with_installed_packs
        registry, _ = registry_with_installed_packs()
        spec = registry.get(model_id)
    except (ImportError, KeyError, ValueError):
        spec = None
    if spec is not None:
        return spec if "container" in spec.runtimes and "windows_native" not in spec.runtimes else None
    # Web jobs can use a state-directory model root that is different from
    # the worker process' default user-data root.  The project stores the
    # activated pack path, so inspect only its manifest as a safe fallback.
    pack_path = getattr(project.model, "pack_path", "")
    if not isinstance(pack_path, str) or not pack_path:
        return None
    try:
        pack_root = Path(pack_path).expanduser().resolve(strict=True)
        manifest_path = pack_root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (isinstance(manifest, dict) and manifest.get("model_id") == model_id and
            "container" in manifest.get("runtimes", ()) and
            "windows_native" not in manifest.get("runtimes", ())):
        from types import SimpleNamespace
        return SimpleNamespace(display_name=(manifest.get("family") or model_id), runtimes=("container",))
    return None


def _train_builtin_project(context, project, device):
    """Train a registered adapter with the selected weight source."""
    from core.project import ProjectManager, RunRecord
    from core.training_artifacts import publish_best
    from train_builtin import train_builtin

    model_id = project.model.model_id
    from builtin_models import BUILTIN_MODEL_SPECS
    spec = BUILTIN_MODEL_SPECS.get(model_id)
    if spec is None or spec.task != project.task:
        return None
    cfg, data = project.training, project.data
    run_id = ProjectManager.new_run_id(task=project.task, model_name=model_id,
                                       input_size=cfg.input_size,
                                       project_dir=project.project_dir)
    run_dir = Path(project.project_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    context.emit("log_message", [f"기본 모델 학습: {model_id}"])
    history = []

    def log(message):
        if isinstance(message, dict) and message.get("event") == "epoch_finished":
            history.append(dict(message))
            epoch = int(message["epoch"])
            total = int(message["total_epochs"])
            metric = float(message["metric"])
            metric_name = "accuracy" if project.task == "classify" else "mIoU"
            event_metrics = dict(message.get("metrics", {metric_name: metric}))
            for key in ("epoch_time_sec", "elapsed_time_sec"):
                if key in message and key not in event_metrics:
                    event_metrics[key] = message[key]
            context.emit("epoch_finished", [epoch, float(message["train_loss"]),
                                              float(message["val_loss"]), event_metrics])
            context.emit("progress_updated", [epoch, total])
        elif isinstance(message, dict) and message.get("event") == "best_epoch_updated":
            context.emit("best_epoch_updated", [message["epoch"], message["train_loss"],
                                               message["val_loss"], message["metrics"]])
        else:
            context.emit("log_message", [str(message)])

    data_root = data.root
    from center_crop import configured_center_crop
    crop = configured_center_crop(cfg)
    if crop:
        from crop_dataset import prepare_crop_dataset
        data_root = prepare_crop_dataset(
            data.root, Path(project.project_dir) / "generated", project.task, crop,
            num_classes=data.num_classes, should_stop=context.cancelled,
            log=lambda message: context.emit("log_message", [message]),
        )
    weights = project.model.pretrained_weights or None
    resume = weights if str(cfg.training_mode).endswith("_resume") else None
    initial_weights = None if resume else weights
    augmentation = cfg.augmentation
    best = train_builtin(model_id, data_root, num_classes=data.num_classes,
                         input_size=cfg.input_size, in_channels=cfg.in_channels,
                         epochs=cfg.epochs, batch_size=cfg.batch_size,
                         learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay,
                         optimizer_name=cfg.optimizer, scheduler_name=cfg.scheduler,
                         warmup_epochs=cfg.warmup_epochs,
                         early_stop_patience=cfg.early_stop_patience,
                         label_smoothing=(cfg.label_smoothing if project.task == "classify" else 0.0),
                         horizontal_flip=augmentation.horizontal_flip,
                         rotation=augmentation.rotation,
                         color_jitter=augmentation.color_jitter,
                         val_split=data.val_split,
                         freeze_backbone=project.model.freeze_backbone,
                         backbone_lr_mult=project.model.backbone_lr_mult,
                         output_dir=run_dir,
                         device=str(device),
                         resume=resume, initial_weights=initial_weights,
                         pretrained=cfg.training_mode == "builtin_finetune",
                         selection_metric=cfg.selection_metric,
                         log=log, should_stop=context.cancelled)
    if not best.is_file():
        record = RunRecord(run_id=run_id, started_at=datetime.now().isoformat(),
                           finished_at=datetime.now().isoformat(), status="cancelled",
                           config_snapshot={"engine": "builtin", "model_id": model_id})
        project.runs.append(record)
        return record
    import torch
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    training_state_path = run_dir / "training.json"
    training_state = (json.loads(training_state_path.read_text(encoding="utf-8"))
                      if training_state_path.is_file() else {})
    metric = float(checkpoint.get("metric") or 0.0)
    best_epoch = int(training_state.get("best_epoch") or (int(checkpoint.get("epoch", 0)) + 1))
    completed_epochs = int(training_state.get("completed_epochs") or best_epoch)
    from core.model_selection import selection_policy
    metric_name = selection_policy(cfg, "builtin", project.task).metric
    full_history = training_state.get("metrics_history") or checkpoint.get("metrics_history") or []
    if full_history:
        metric_history = [float(item.get("selected_value", item["val_metric"])) for item in full_history]
        train_loss_history = [float(item["train_loss"]) for item in full_history]
        val_loss_history = [float(item["val_loss"]) for item in full_history]
    else:
        metric_history = [float(item.get("selected_value", item["metric"])) for item in history]
        train_loss_history = [float(item["train_loss"]) for item in history]
        val_loss_history = [float(item["val_loss"]) for item in history]
    metrics_history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        metric_name: metric_history,
    }
    records = full_history or history
    metrics_history["epoch"] = [int(item["epoch"]) for item in records]
    metric_keys = set.union(*(set(item.get("metrics", {})) for item in records)) if records else set()
    for name in sorted(metric_keys - {"train_loss", "val_loss", metric_name}):
        metrics_history[name] = [item.get("metrics", {}).get(name) for item in records]
    for name in ("epoch_time_sec", "elapsed_time_sec"):
        if any(name in item for item in records):
            metrics_history[name] = [item.get(name) for item in records]
    lr_history = [float(item["learning_rate"]) for item in full_history
                  if "learning_rate" in item]
    policy = selection_policy(cfg, "builtin", project.task)
    named_checkpoint, selection = publish_best(
        best, run_dir, metrics_history, epoch=best_epoch, metric=metric_name, value=metric,
        direction=policy.direction, engine="builtin", task=project.task,
        policy="native_adapter_strict_improvement", formula=policy.formula)
    record = RunRecord(run_id=run_id, started_at=datetime.now().isoformat(),
                       finished_at=datetime.now().isoformat(), status=(
                           "cancelled" if context.cancelled() else "completed"),
                       epochs_done=completed_epochs, best_metric=metric, best_epoch=best_epoch,
                       best_metric_name=metric_name, checkpoint_path=named_checkpoint,
                       metrics_history=metrics_history, lr_history=lr_history,
                       config_snapshot={"engine": "builtin", "model_id": model_id,
                                        "training": training_state.get("training_config", {}),
                                        "center_crop": crop, "best_selection": selection})
    project.runs.append(record)
    context.emit("training_finished", [metric, best_epoch, named_checkpoint])
    return record


def _train_upstream_project(context, project, device):
    """Train the shipped LibreYOLO/RT-DETRv4 family without a CSP fallback."""
    from upstream_models import train_upstream_model, upstream_model_ids, get_upstream_spec
    model_id = project.model.model_id
    if model_id not in upstream_model_ids():
        return None
    from core.project import ProjectManager, RunRecord
    from core.training_artifacts import publish_best
    cfg, data = project.training, project.data
    run_id = ProjectManager.new_run_id(task=project.task, model_name=model_id,
                                       input_size=cfg.input_size, project_dir=project.project_dir)
    run_dir = Path(project.project_dir) / "runs" / run_id
    history = []

    def progress(event):
        if event.get("event") == "training_started":
            context.emit("progress_updated", [max(0, int(event["epoch"]) - 1), int(event["total_epochs"])])
            context.emit("log_message", [f"native worker 준비 완료: {event['family']}/{event['size']} | {event['total_epochs']} epochs"])
            return
        if event.get("event") == "training_exception":
            context.emit("log_message", ["native worker 오류: " + event["message"]])
            return
        if event.get("event") == "dataset_prepared":
            automatic = event.get("automatic_classes", [])
            suffix = f" | 자동 검증 분할: {', '.join(automatic)}" if automatic else ""
            context.emit("log_message", [
                "LibreYOLO 데이터 준비: train {}장, val {}장{}".format(
                    event["train_images"], event["val_images"], suffix,
                )
            ])
            return
        if event.get("event") != "epoch_finished":
            return
        history.append(dict(event))
        metrics = dict(event.get("metrics", {}))
        metrics.setdefault(event.get("metric_name", "metric"), event["metric"])
        metrics["epoch_time_sec"] = event.get("epoch_time_sec", 0.0)
        metrics["elapsed_time_sec"] = event.get("elapsed_time_sec", 0.0)
        context.emit("epoch_finished", [int(event["epoch"]), float(event["train_loss"]),
                                        float(event["val_loss"]), metrics])
        if event.get("learning_rate") is not None:
            context.emit("lr_updated", [int(event["epoch"]), float(event["learning_rate"])])
        if event.get("is_best"):
            context.emit("best_epoch_updated", [int(event.get("best_epoch") or event["epoch"]),
                                                 float(event["train_loss"]), float(event["val_loss"]), metrics])
        context.emit("progress_updated", [int(event["epoch"]), int(event["total_epochs"])])
        context.emit("log_message", [f"Epoch {event['epoch']}/{event['total_epochs']} | train={event['train_loss']:.5f} | val={event['val_loss']:.5f} | {event['metric_name']}={event['metric']:.5f} | {event['epoch_time_sec']:.1f}s"])

    spec = get_upstream_spec(model_id)
    mode = str(cfg.training_mode)
    source = project.model.pretrained_weights or None
    context.emit("log_message", [f"기본 native 모델 학습: {spec.family}/{spec.size}"])
    result = train_upstream_model(
        model_id, data_root=data.root, class_names=list(data.class_names), output_dir=run_dir,
        epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
        device=str(device), weights=source, pretrained=mode == "upstream_finetune", resume=mode == "upstream_resume",
        use_amp=cfg.use_amp, patience=cfg.early_stop_patience,
        val_split=data.val_split, seed=0, emit=progress)
    checkpoint = result.get("best_checkpoint") or result.get("last_checkpoint")
    if not checkpoint or not Path(checkpoint).is_file():
        raise RuntimeError("upstream 학습이 best.pt 또는 last.pt를 만들지 않았습니다")
    metric_name = (history[-1].get("metric_name", "metric") if history else "metric")
    best_metric = float(result.get("best_metric") if result.get("best_metric") is not None
                        else (history[-1]["metric"] if history else 0.0))
    best_epoch = int(result.get("best_epoch") or (history[-1]["epoch"] if history else 0))
    metrics_history = {"train_loss": [float(item["train_loss"]) for item in history],
                       "epoch": [int(item["epoch"]) for item in history],
                       "val_loss": [float(item["val_loss"]) for item in history],
                       metric_name: [float(item["metric"]) for item in history],
                       "epoch_time_sec": [float(item.get("epoch_time_sec", 0.0)) for item in history],
                       "elapsed_time_sec": [float(item.get("elapsed_time_sec", 0.0)) for item in history]}
    metric_keys = set.union(*(set(item.get("metrics", {})) for item in history)) if history else set()
    for name in sorted(metric_keys - {"epoch_time_sec", "elapsed_time_sec", metric_name}):
        metrics_history[name] = [item.get("metrics", {}).get(name) for item in history]
    direction = "min" if "loss" in metric_name.lower() else "max"
    named_checkpoint, selection = publish_best(
        checkpoint, run_dir, metrics_history, epoch=best_epoch, metric=metric_name, value=best_metric,
        direction=direction, engine="upstream", task=project.task,
        policy="upstream_native_is_best", formula=metric_name)
    record = RunRecord(run_id=run_id, started_at=datetime.now().isoformat(),
                       finished_at=datetime.now().isoformat(), status="completed",
                       epochs_done=len(history), best_metric=best_metric, best_epoch=best_epoch,
                       best_metric_name=metric_name, checkpoint_path=named_checkpoint,
                       metrics_history=metrics_history,
                       config_snapshot={"engine": "upstream", "model_id": model_id,
                                        "upstream_family": spec.family, "upstream_size": spec.size,
                                        "upstream_result": result, "best_selection": selection})
    project.runs.append(record)
    context.emit("training_finished", [best_metric, best_epoch, named_checkpoint])
    return record


def _train_sam2_project(context, project, device):
    """Run the shipped SAM2 prompt-mask adapter instead of Custom CSP."""
    from core.training_modes import SAM2_ADAPTER_IDS
    from core.training_artifacts import publish_best
    model_id = project.model.model_id
    if model_id not in SAM2_ADAPTER_IDS:
        return None
    if project.training.training_mode not in {"sam2_finetune", "sam2_transfer"}:
        raise RuntimeError("SAM2는 '기본 제공 가중치로 미세조정' 또는 '로컬 미세조정 체크포인트' 모드로 실행하세요.")
    from core.project import ProjectManager, RunRecord
    from sam2_trainer import train_sam2
    cfg, data = project.training, project.data
    run_id = ProjectManager.new_run_id(task=project.task, model_name=model_id,
                                       input_size=1024, project_dir=project.project_dir)
    run_dir = Path(project.project_dir) / "runs" / run_id
    history = []

    def progress(event):
        if event.get("event") == "training_started":
            context.emit("progress_updated", [0, int(event["total_epochs"])])
            context.emit("log_message", [
                f"SAM2.1 native worker 준비 완료: {event['size']} | 1024 입력 | "
                "image encoder 고정, prompt encoder·mask decoder 학습"
            ])
            return
        if event.get("event") != "epoch_finished":
            return
        history.append(dict(event))
        metrics = dict(event.get("metrics", {}))
        metrics["epoch_time_sec"] = event.get("epoch_time_sec", 0.0)
        metrics["elapsed_time_sec"] = event.get("elapsed_time_sec", 0.0)
        context.emit("epoch_finished", [int(event["epoch"]), float(event["train_loss"]),
                                        float(event["val_loss"]), metrics])
        context.emit("lr_updated", [int(event["epoch"]), float(event["learning_rate"])])
        if event.get("is_best"):
            context.emit("best_epoch_updated", [int(event["best_epoch"]), float(event["train_loss"]),
                                                 float(event["val_loss"]), metrics])
        context.emit("progress_updated", [int(event["epoch"]), int(event["total_epochs"])])
        context.emit("log_message", [
            f"Epoch {event['epoch']}/{event['total_epochs']} | train={event['train_loss']:.5f} | "
            f"val={event['val_loss']:.5f} | prompt_dice={event['metric']:.5f} | "
            f"{event['epoch_time_sec']:.1f}s"
        ])

    source = project.model.pretrained_weights if cfg.training_mode == "sam2_transfer" else None
    result = train_sam2(
        model_id, data.root, output_dir=run_dir, epochs=cfg.epochs, batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay, device=str(device),
        use_amp=cfg.use_amp, input_size=1024, initial_checkpoint=source,
        horizontal_flip=cfg.augmentation.horizontal_flip, rotation=cfg.augmentation.rotation,
        color_jitter=cfg.augmentation.color_jitter, emit=progress, should_stop=context.cancelled,
    )
    checkpoint = result.get("best_checkpoint") or result.get("last_checkpoint")
    if not checkpoint or not Path(checkpoint).is_file():
        raise RuntimeError("SAM2 학습이 best.pt 또는 last.pt를 만들지 않았습니다.")
    metric_name = "prompt_dice"
    metrics_history = {
        "epoch": [int(item["epoch"]) for item in history],
        "train_loss": [float(item["train_loss"]) for item in history],
        "val_loss": [float(item["val_loss"]) for item in history],
        metric_name: [float(item["metric"]) for item in history],
        "prompt_iou": [float(item["metrics"]["prompt_iou"]) for item in history],
        "epoch_time_sec": [float(item.get("epoch_time_sec", 0.0)) for item in history],
        "elapsed_time_sec": [float(item.get("elapsed_time_sec", 0.0)) for item in history],
    }
    named_checkpoint, selection = publish_best(
        checkpoint, run_dir, metrics_history, epoch=int(result.get("best_epoch") or 0),
        metric=metric_name, value=float(result.get("best_metric") or 0.0), direction="max",
        engine="sam2", task="segment", policy="prompt_dice_strict_improvement", formula="Prompt Dice")
    record = RunRecord(
        run_id=run_id, started_at=datetime.now().isoformat(), finished_at=datetime.now().isoformat(),
        status="cancelled" if result.get("cancelled") else "completed", epochs_done=len(history),
        best_metric=float(result.get("best_metric") or 0.0), best_epoch=int(result.get("best_epoch") or 0),
        best_metric_name=metric_name, checkpoint_path=named_checkpoint, metrics_history=metrics_history,
        config_snapshot={
            "engine": "sam2", "model_id": model_id, "input_size": 1024,
            "training_contract": "semantic-mask-to-positive-point-object-mask-v1",
            "frozen_modules": ["image_encoder"], "sam2_result": result, "best_selection": selection,
        },
    )
    project.runs.append(record)
    context.emit("training_finished", [record.best_metric, record.best_epoch, named_checkpoint])
    return record


def train(context, payload):
    from core.project import ProjectManager
    from core.training_engine import TrainingEvents
    from core.trainer import TrainWorker
    from core.patchcore_trainer import PatchCoreWorker
    project = restore_project(payload["project"])
    from core.training_modes import validate_training_options
    validate_training_options(project)
    container_spec = _container_model_spec(project)
    if container_spec is not None:
        # A pack request has a different result/event contract and must go
        # through ``/api/jobs/model-pack/{operation}``.  Failing here is safer
        # than silently training the selected Re-DETR/SAM2/LibreYOLO ID with
        # the legacy Custom CSP engine.
        raise RuntimeError(
            f"{container_spec.display_name}은 일반 train 작업에서 실행할 수 없습니다. "
            "설치된 모델 팩 경로로 pack_train 작업을 사용하세요."
        )
    from core.device_manager import get_device_manager
    from core.accelerator import runtime_report
    device = get_device_manager().get_device(project.training.device)
    context.emit("log_message", [f"학습 장치 검사: 요청={project.training.device}, 실제={device}"])
    runtime = runtime_report(str(device), check=True)
    if runtime.get("error"):
        raise RuntimeError("학습 장치 연산 검사 실패: " + runtime["error"])
    context.emit("log_message", [f"실행 장치: {runtime['name']} | PyTorch {runtime['torch']} | CUDA {runtime['cuda_runtime'] or '없음'}"])
    if runtime.get("backward_checked"):
        context.emit("log_message", ["GPU 합성곱과 역전파 검사 통과"])
    before = len(project.runs)
    config = {key: payload["project"][key] for key in ("task", "data", "model", "training")}
    builtin_record = _train_builtin_project(context, project, device)
    upstream_record = None if builtin_record is not None else _train_upstream_project(context, project, device)
    sam2_record = None if builtin_record is not None or upstream_record is not None else _train_sam2_project(context, project, device)
    if builtin_record is not None or upstream_record is not None or sam2_record is not None:
        engine = None
    elif project.task == "anomaly" and project.training.anomaly_method == "patchcore":
        engine_class = PatchCoreWorker
    elif project.training.training_mode.startswith("efficientnet"):
        from core.efficientnet_trainer import EfficientNetTrainWorker
        engine_class = EfficientNetTrainWorker
    else:
        engine_class = TrainWorker
    if builtin_record is None and upstream_record is None and sam2_record is None:
        engine = engine_class(project, signals=TrainingEvents(context.emit), should_stop=context.cancelled)
        engine.run()
    if (builtin_record is not None or upstream_record is not None or sam2_record is not None or
            getattr(engine, "engine_name", None) == "efficientnet"):
        config.update(training=asdict(project.training), model=asdict(project.model), data=asdict(project.data))
    for record in project.runs[before:]:
        record.config_snapshot = {**config, **record.config_snapshot, "runtime": runtime,
                                  "requested_config": {key: payload["project"][key]
                                                       for key in ("task", "data", "model", "training")},
                                  "job_id": context.directory.name,
                                  "checkpoint_sha256": digest(record.checkpoint_path)
                                  if record.checkpoint_path and Path(record.checkpoint_path).is_file() else None}
    write_json(context.directory / "project_result.json", {**asdict(project), "filepath": payload["project"]["filepath"]})
    if digest(payload["project"]["filepath"]) != payload["project_digest"]:
        raise RuntimeError("외부에서 프로젝트 파일 변경됨. 작업 폴더의 project_result.json에 학습 기록 보존")
    if payload.get("persist_project", True):
        ProjectManager.save(project, payload["project"]["filepath"])
    if context.failure:
        raise RuntimeError(context.failure)
    status = project.runs[-1].status if len(project.runs) > before else "cancelled" if context.cancelled() else "failed"
    return {"status": status, "output": {"runs": [asdict(run) for run in project.runs[before:]]}}


def infer(context, payload):
    import numpy as np
    from core.device_manager import get_device_manager
    from core.inference_loading import load_inference_engine
    from core.inference_timing import batch_stage_summary
    from core.inference_review import source_class
    device = get_device_manager().get_device(payload.get("device", "cpu"))
    engine = load_inference_engine(payload["weights"], device=device, gradcam=payload.get("gradcam", True),
                                   input_region=payload.get("input_region"), runtime=payload.get("runtime", "auto"),
                                   threads=payload.get("threads", max(1, min(4, os.cpu_count() or 1))))
    if callable(getattr(engine, "prepare", None)):
        engine.prepare()
    if getattr(engine, "runtime_warning", ""):
        context.emit("log_message", [engine.runtime_warning])
    root = context.directory / "results"
    root.mkdir(exist_ok=True)
    results = []
    write_json(context.directory / "input_manifest.json", {"weights": payload["weights"],
        "input_region": engine.input_region,
        "weights_sha256": digest(payload["weights"]), "project_path": payload.get("project", {}).get("filepath"),
        "images": [{"path": p, "bytes": Path(p).stat().st_size, "mtime_ns": Path(p).stat().st_mtime_ns,
                    "sha256": digest(p)} for p in payload["images"]]})
    started = time.perf_counter()
    try:
        for index, path in enumerate(payload["images"]):
            if context.cancelled():
                break
            result = engine.infer(path)
            record = {**asdict(result), "index": index, "source_class": source_class(path, payload.get("project"))}
            arrays = {}
            if engine._current_preview_rgb is not None:
                arrays["preview"] = engine._current_preview_rgb
            if result.status != "error" and (engine._heatmap_cache is None or engine._heatmap_cache.get("original") is None):
                from PIL import Image
                from core.image_display import display_rgb
                with Image.open(path) as source:
                    arrays["original"] = display_rgb(np.array(source.convert("RGB") if source.mode == "P" else source))
            cache = engine._heatmap_cache
            if cache is not None:
                for key in ("original", "activation", "base_image", "valid_mask"):
                    if cache.get(key) is not None:
                        arrays[key] = cache[key]
                record["heatmap"] = {"kind": cache["kind"], "info": cache["info"],
                                     "detections": cache.get("detections")}
            record["info"] = engine.info
            record["cache_ready"] = False
            temporary = root / f"{index}.tmp"
            # 큰 이미지 배열 저장에 실패해도 이미 계산한 판정/시간은 보존한다.
            write_json(root / f"{index}.json", record)
            try:
                with temporary.open("wb") as stream:
                    np.savez(stream, **arrays)
                os.replace(temporary, root / f"{index}.npz")
                from PIL import Image
                thumb = arrays.get("preview", arrays.get("original"))
                if thumb is not None:
                    preview = Image.fromarray(thumb)
                    preview.thumbnail((256, 256))
                    preview.save(root / f"{index}.thumb.png")
                record["cache_ready"] = True
                write_json(root / f"{index}.json", record)
            except OSError as exc:
                record["cache_error"] = f"미리보기 저장 실패: {exc}"
                record.pop("heatmap", None)
                write_json(root / f"{index}.json", record)
                raise RuntimeError(record["cache_error"]) from exc
            finally:
                with suppress(OSError):
                    temporary.unlink()
            results.append(result)
            context.emit("inference_result", [record])
            context.emit("progress_updated", [index + 1, len(payload["images"])])
    finally:
        if engine._gradcam is not None:
            engine._gradcam.release()
    errors = sum(result.status == "error" for result in results)
    output = {"completed": len(results), "total": len(payload["images"]), "errors": errors,
              "input_region": engine.input_region,
              "elapsed_sec": time.perf_counter() - started,
              "inference_summary": batch_stage_summary(results, "inference"),
              "gradcam_summary": batch_stage_summary(results, "gradcam")}
    return {"status": "cancelled" if context.cancelled() else "failed" if results and errors == len(results) else "completed",
            "error": "모든 이미지 추론 실패" if results and errors == len(results) else "", "output": output}


def export(context, payload):
    from export_onnx import export_checkpoint
    if context.cancelled():
        return {"status": "cancelled"}
    validation_dir = None
    if payload.get("dataset_validation"):
        from classification_export_validation import collect_images
        validation_dir = payload.get("validation_dir")
        if not validation_dir:
            data = payload.get("project", {}).get("data", {})
            for split in ("val_dir", "test_dir", "train_dir"):
                folder = data.get(split)
                if not folder:
                    continue
                try:
                    collect_images(folder, data.get("class_names", []))
                except (ValueError, OSError):
                    continue
                validation_dir = folder
                break
        if not validation_dir:
            raise ValueError("실제 이미지 ONNX 출력 검증에는 이미지가 1장 이상 있는 비교 경로가 필요합니다")
        context.emit("log_message", [f"실제 이미지 비교 폴더: {validation_dir}"])
    result = export_checkpoint(payload["weights"], payload["output"],
                               opset_version=payload.get("opset", 17),
                               dynamic_batch=payload.get("dynamic_batch", False), verify=True,
                               validation_dir=validation_dir,
                               allow_precision_fallback=payload.get("allow_precision_fallback", False),
                               log=lambda line: context.emit("log_message", [line]))
    validation = result.get("classification_validation")
    if validation:
        selected = validation["selected"]
        if validation["policy"] == "classification_dataset_v1":
            message = (f"실제 이미지 {validation['image_count']}장 검증 통과. 전처리+추론+후처리 "
                       f"중앙값 {selected['pipeline_median_ms']:.2f}ms / p95 {selected['pipeline_p95_ms']:.2f}ms / "
                       f"최대 {selected['pipeline_max_ms']:.2f}ms. "
                       f"관측 8ms 목표: {'충족' if validation['target_met'] else '미달'} (파일 디코딩 제외)")
        else:
            message = (f"결정론적 합성 입력 {validation['probe_count']}개에서 top-1·확률·판정 마진 검증 통과. "
                       f"모델 추론 중앙값 {selected['model_median_ms']:.2f}ms. 실제 이미지 자료는 검증하지 않음")
        context.emit("log_message", [message])
    return {"status": "completed", "output": result}


def model_pack_operation(context, payload):
    """Run one train/infer/export command in an installed Docker model pack.

    The request is JSON metadata; large images, checkpoints, and results stay
    in the mounted data/work directories.  The pack process is long-lived for
    this operation and is always closed on the way out.
    """
    from core.model_pack_worker import ModelPackWorker

    operation = payload.get("operation")
    if operation not in {"train", "infer", "export"}:
        raise ValueError("model pack operation must be train, infer, or export")
    if context.cancelled():
        return {"status": "cancelled"}
    worker = ModelPackWorker.from_installed_pack(
        payload["pack_dir"], data_dir=payload["data_dir"], work_dir=payload["work_dir"],
        cpus=payload.get("cpus", 4), memory=payload.get("memory", "8g"),
        name=payload.get("container_name"))
    try:
        worker.start()
        hello = worker.json_request("hello")
        context.emit("log_message", [f"모델 팩 worker 준비: {worker.model_id}"])
        # Keep the lifecycle explicit even for packs that use the protocol's
        # default no-op handler.  A pack may validate its dataset/checkpoint
        # here and fail before a long train/export operation starts.
        prepared = worker.json_request("prepare", {
            "operation": operation,
            "data_dir": str(payload["data_dir"]),
            "work_dir": str(payload["work_dir"]),
            "request": payload.get("request", {}),
        })
        context.emit("model_pack_prepared", [prepared])
        if context.cancelled():
            try:
                worker.json_request("cancel")
            except Exception:
                pass
            return {"status": "cancelled"}
        result = worker.json_request(operation, payload.get("request", {}))
        context.emit("model_pack_result", [result])
        return {"status": "completed", "output": {"hello": hello, "prepared": prepared, "result": result}}
    finally:
        worker.close()


def pack_train(context, payload):
    return model_pack_operation(context, {**payload, "operation": "train"})


def pack_infer(context, payload):
    return model_pack_operation(context, {**payload, "operation": "infer"})


def pack_export(context, payload):
    return model_pack_operation(context, {**payload, "operation": "export"})


def defects(context, payload):
    from core.defect_workflow import generate_candidates
    return generate_candidates(context, payload, restore_project(payload["project"]))


def defect_publish(context, payload):
    from core.defect_workflow import publish_candidates
    return publish_candidates(context, payload, restore_project(payload["project"]))


def delete_class(context, payload):
    from core.class_management import ClassManager
    project = restore_project(payload["project"])
    if digest(payload["project"]["filepath"]) != payload["project_digest"]:
        raise ValueError("외부에서 프로젝트 변경됨. 다시 열고 삭제 내용 확인 필요")
    preview = ClassManager.preview_delete(project, payload["class_name"])
    current = hashlib.sha256(json.dumps(json_value(asdict(preview)), sort_keys=True).encode()).hexdigest()
    if current != payload["preview_digest"]:
        raise ValueError("미리보기 이후 데이터 변경. 클래스 삭제 내용을 다시 확인 필요")
    if context.cancelled():
        return {"status": "cancelled"}
    archive = ClassManager.delete(project, payload["class_name"], preview=preview)
    return {"status": "completed", "output": {"archive": archive}}



def inspect_model(context, payload):
    from core.device_manager import get_device_manager
    from core.inference_loading import load_inference_engine
    device = get_device_manager().get_device(payload.get("device", "cpu"))
    engine = load_inference_engine(payload["weights"], device=device, gradcam=False)
    task = ("anomaly" if engine._patchcore_model is not None else
            engine._upstream_task if getattr(engine, "_upstream_model", None) is not None else
            getattr(engine.model, "task", "classify"))
    return {"status": "completed", "output": {"class_names": list(engine.class_names or []),
        "input_size": engine._input_size, "task": task, "weights": payload["weights"],
        "center_crop": engine._center_crop,
        "device": str(engine._infer_device), "anomaly_threshold": engine._anomaly_threshold,
        "patchcore": engine._patchcore_model.get_info() if engine._patchcore_model is not None else None,
        "upstream": getattr(engine, "_upstream_model", None) is not None,
        "weights_sha256": digest(payload["weights"])}}


def dataset_edit(context, payload):
    from core.dataset_editor import edit_dataset
    project = restore_project(payload["project"])
    if digest(payload["project"]["filepath"]) != payload["project_digest"]:
        raise ValueError("외부에서 프로젝트 변경됨. 다시 열기 필요")
    result = edit_dataset(project, cancelled=context.cancelled, **payload["edit"])
    return {"status": "completed", "output": result}


def datagen_train(context, payload):
    from core.datagen_learning import train
    from core.datagen_store import DataGenStore
    project = restore_project(payload["project"])
    with exclusive_file(Path.home() / ".deep-vision-studio-datagen" / "compute.lock"), exclusive_file(DataGenStore(project).root / "compute.lock"):
        return train(context, payload, project)


def datagen_generate(context, payload):
    from core.datagen_learning import generate
    from core.datagen_store import DataGenStore
    project = restore_project(payload["project"])
    with exclusive_file(Path.home() / ".deep-vision-studio-datagen" / "compute.lock"), exclusive_file(DataGenStore(project).root / "compute.lock"):
        return generate(context, payload, project)



OPERATIONS = {"datagen_train": datagen_train, "datagen_generate": datagen_generate,
              "train": train, "infer": infer, "export": export,
              "pack_train": pack_train, "pack_infer": pack_infer, "pack_export": pack_export,
              "defects": defects, "defect_publish": defect_publish, "delete_class": delete_class,
              "dataset_edit": dataset_edit, "inspect_model": inspect_model}


def run_job(request_path, monitor_parent=True):
    request_path = Path(request_path)
    context = None
    started = time.perf_counter()
    try:
        request = read_json(request_path)
        context = JobContext(request_path.parent, monitor_parent=monitor_parent,
                             parent_pid=request.get("parent_pid"), parent_started=request.get("parent_started"))
        # 작업 프로세스에서만 모델 라이브러리를 가져오고 CPU 자원을 제한한다.
        threads = request["payload"].get("threads", max(1, min(4, os.cpu_count() or 1)))
        os.environ["OMP_NUM_THREADS"] = str(threads)
        if request["kind"] in {"train", "infer", "export"}:
            import torch
            torch.set_num_threads(threads)
        with exclusive_file(request_path.parent.parent / "compute.lock"):
            result = OPERATIONS[request["kind"]](context, request["payload"])
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "failed", "error": str(exc)}
    finally:
        if context is not None:
            context.monitor_done.set()
    result["duration_sec"] = time.perf_counter() - started
    write_json(request_path.parent / "result.json", result)
    return result


if __name__ == "__main__":
    run_job(sys.argv[1])
