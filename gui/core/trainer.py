"""
Deep Vision Studio — 학습 엔진

학습 파이프라인 (QThread 기반):
┌──────────────────────────────────────────────────────┐
│  GUI Thread (메인)                                    │
│  ┌─────────────────────────────────────────────────┐ │
│  │  TrainingWidget                                  │ │
│  │  • 학습 시작/중지 버튼                            │ │
│  │  • 실시간 Loss / Metric / LR 차트 업데이트         │ │
│  │  • 혼동 행렬 + 평가 결과 테이블                    │ │
│  └───────────────────┬─────────────────────────────┘ │
│                      │ signals                        │
│  ┌───────────────────▼─────────────────────────────┐ │
│  │  TrainWorker (QThread)                           │ │
│  │  • 데이터 로딩 → 모델 학습 → 검증                  │ │
│  │  • 에폭별 시그널 emit → GUI 업데이트              │ │
│  │  • 체크포인트 자동 저장                            │ │
│  │  • 학습 후 종합 평가 (Confusion Matrix 등)         │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘

트랜스퍼 러닝:
┌───────────────────────────────────────────────────────┐
│ 1. 사전학습 가중치 로드 (backbone만 또는 전체)          │
│ 2. 백본 동결 옵션 (freeze_backbone=True)              │
│ 3. 백본/헤드 차등 학습률 (backbone_lr_mult)            │
│ 4. 다른 태스크 → 백본 전이 가능                        │
│    (classify best.pt → detect 백본으로 재사용)         │
└───────────────────────────────────────────────────────┘

태스크별 평가 지표:
┌───────────────────────────────────────────────────────┐
│ Classification: Accuracy, Precision, Recall, F1, CM   │
│ Segmentation:   mIoU, Dice Score, Pixel Accuracy      │
│ Detection:      mAP@0.5, mAP@0.5:0.95, P/R           │
│ Anomaly:        AUROC, F1, P/R, Confusion Matrix      │
└───────────────────────────────────────────────────────┘
"""

import os
import math
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.training_engine import TrainingEngine

# 프로젝트 모듈 경로 추가 (PyInstaller EXE 호환)
from core.paths import ensure_python_path
ensure_python_path()

from model import CustomCSP
from checkpoint import make_checkpoint_metadata
from class_weights import class_counts, weights_from_counts, describe_class_weights
from classification_loss import WeightedClassificationLoss
from core.project import ProjectData, RunRecord, ProjectManager
from core.metrics import (
    create_metrics, TASK_METRIC_NAMES,
)
from core.device_manager import get_device_manager


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  학습 워커 (QThread)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TrainWorker(TrainingEngine):
    """
    별도 스레드에서 학습 수행

    시그널로 GUI에 진행 상황 전달:
    - epoch_finished → Loss / Metric 차트 업데이트
    - lr_updated → Learning Rate 차트 업데이트
    - batch_finished → 프로그레스 바 업데이트
    - eval_finished → 종합 평가 결과 (Confusion Matrix 등)
    - training_finished → 완료 알림
    - training_error → 에러 팝업
    """

    def __init__(self, project: ProjectData, parent=None, *, signals=None, should_stop=None):
        super().__init__(project, signals=signals, should_stop=should_stop)
        self._stop_requested = False

    def stop(self):
        """학습 중지 요청 (현재 에폭 완료 후 종료)"""
        self._stop_requested = True

    def run(self):
        """학습 메인 루프 (별도 스레드에서 실행)"""
        try:
            self._run_training()
        except Exception as e:
            if getattr(self, "_run_record", None) is not None:
                self._run_record.status = "failed"
                self._run_record.finished_at = datetime.now().isoformat()
            self.signals.training_error.emit(
                f"학습 오류: {str(e)}\n{traceback.format_exc()}"
            )

        finally:
            if getattr(self, "_layer_debug_session", None) is not None:
                self._layer_debug_session.close()

    def _run_training(self):
        """실제 학습 로직"""
        from core.training_modes import validate_training_options
        validate_training_options(self.project)
        self._prepare_training()
        validate_training_options(self.project)
        from core.training_time import TrainingClock, record_epoch_time, log_total_time
        self._training_clock = TrainingClock()
        cfg = self.project.training
        data_cfg = self.project.data
        model_cfg = self.project.model
        task = self.project.task

        self.signals.log_message.emit(f"학습 시작 — 태스크: {task}")
        self.signals.log_message.emit(f"  에폭: {cfg.epochs}, 배치: {cfg.batch_size}")
        self.signals.log_message.emit(f"  학습률: {cfg.learning_rate}")

        # ── 1. 디바이스 설정 (GPU 자동 감지) ─────────────
        dm = get_device_manager()
        device_mode = getattr(cfg, "device", "auto")  # TrainingConfig.device
        device = dm.get_device(device_mode)
        # AMP는 GPU 지원 + 사용자 설정 모두 충족 시에만 활성화
        cfg_amp = getattr(cfg, "use_amp", True)
        use_amp = dm.supports_amp(device) and cfg_amp and not getattr(self, "debug_fp32", False)
        self.signals.log_message.emit(f"  디바이스: {dm.get_device_label(device)}")
        if use_amp:
            self.signals.log_message.emit("  Mixed Precision (FP16) 활성화")
        if cfg.epochs < 1 or cfg.batch_size < 1:
            raise ValueError("에폭과 배치 크기는 1 이상 필요")
        # 실제 클래스 목록을 확정한 뒤 출력 헤드를 생성한다.
        num_workers = getattr(self, "debug_num_workers", dm.get_optimal_num_workers(device))
        pin_memory = getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"
        train_loader, val_loader = self._create_dataloaders(
            task, data_cfg, cfg, num_workers=num_workers, pin_memory=pin_memory
        )
        if train_loader is None or len(train_loader) == 0:
            detail = getattr(self, "_dataloader_error", "")
            message = "학습 가능한 데이터 배치 없음"
            if detail:
                message += f"\n데이터 로딩 원인: {detail}"
            raise ValueError(message)
        if val_loader is None or len(val_loader) == 0:
            raise ValueError("모델 선택을 위한 검증 데이터 필요")
        model = self._build_model().to(device)

        params = model.get_param_count()
        self.signals.log_message.emit(
            f"  모델 파라미터: {params['total']:,} ({params['total_MB']:.1f} MB)"
        )

        # ── 2. 트랜스퍼 러닝 (사전학습 가중치 로드) ────────
        #  경로가 지정됐는데 파일이 없으면 조용히 건너뛰지 않고 알린다.
        #  (이전에는 무시하고 스크래치 학습을 진행해 사용자가
        #   사전학습이 적용된 줄 알았다.)
        weights_path = (model_cfg.pretrained_weights or "").strip()
        pretrained_loaded = self._initialize_model(model, weights_path, task)
        if getattr(self, "debug_observer", None) is not None:
            self.debug_observer(model)

        # ── 3. 백본 동결 (선택) ────────────────────────
        if model_cfg.freeze_backbone:
            if not pretrained_loaded:
                # 랜덤 초기화된 백본을 얼리면 특징 추출기가 학습되지 않아
                # 헤드만으로는 어떤 것도 배울 수 없다.
                raise ValueError(
                    "'백본 레이어 동결'이 켜져 있지만 사전학습 가중치가 "
                    "없습니다.\n"
                    "랜덤 초기화된 백본을 동결하면 특징 추출기가 전혀 "
                    "학습되지 않아 성능이 나오지 않습니다.\n\n"
                    "다음 중 하나를 선택하세요:\n"
                    "  • 사전학습 가중치를 지정한다\n"
                    "  • '백본 레이어 동결'을 해제한다"
                )
            for param in model.backbone.parameters():
                param.requires_grad = False
            self.signals.log_message.emit("  백본 동결됨 (헤드만 학습)")

        # ── 4. 옵티마이저 설정 ─────────────────────────
        optimizer = self._create_optimizer(model, cfg, model_cfg)
        scheduler = self._create_scheduler(optimizer, cfg)

        # ── 4.5. AMP GradScaler (Mixed Precision) ─────
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # ── 6. 손실 함수 ──────────────────────────────
        #  클래스 가중치는 학습 데이터의 실제 분포에서 계산하므로
        #  데이터 로더 생성 이후에 손실 함수를 만든다.
        criterion = self._create_criterion(
            task, cfg, train_loader=train_loader,
            num_classes=data_cfg.num_classes, device=device,
        )

        # ── 7. 학습 실행 기록 생성 ─────────────────────
        # 폴더명: {Task}_{Model}_{ImgSize}_{날짜}
        # 예: Classification_custom_224_260907_10h47m
        pretrained_name = (cfg.efficientnet_model if self.engine_name == "efficientnet"
                           else os.path.basename(weights_path) if weights_path else "custom")
        run_id = ProjectManager.new_run_id(
            task=task,
            model_name=pretrained_name,
            input_size=cfg.input_size,
            project_dir=self.project.project_dir,
        )
        run_dir = os.path.join(self.project.project_dir, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)

        # 태스크별 주요 메트릭 이름
        metric_info = TASK_METRIC_NAMES.get(task, {})
        from core.model_selection import selection_policy
        selected_policy = selection_policy(cfg, self.engine_name, task)
        primary_metric_name = selected_policy.metric
        self.signals.log_message.emit("  Best 기준: " + selected_policy.describe())

        run_record = RunRecord(
            run_id=run_id,
            started_at=datetime.now().isoformat(),
            status="running",
            best_metric_name=primary_metric_name,
        )

        if getattr(self, "_class_weight_details", None) is not None:
            run_record.config_snapshot["class_weight_details"] = self._class_weight_details
        self._run_record = run_record
        self.project.runs.append(run_record)
        self._layer_debug_session = None
        if cfg.layer_debug_enabled:
            from core.layer_debug_session import LayerDebugSession
            self._layer_debug_session = LayerDebugSession(cfg, run_dir, run_record, self.signals.layer_debug.emit)
            self.signals.log_message.emit("선택 레이어 관찰 활성화: 초기 배치의 통계만 저장, 전체 학습은 계속 진행")

        # ── 8. 메트릭 이력 추적 딕셔너리 ──────────────
        metrics_history = {"train_loss": [], "val_loss": []}
        for metric_name in set(metric_info.get("display", []) + [primary_metric_name]):
            metrics_history[metric_name] = []
        lr_history = []

        # ── 9. 학습 루프 ──────────────────────────────
        # anomaly의 primary metric은 auroc (높을수록 좋음)
        # 단, 에폭 중 val_loss만 볼 때는 낮을수록 좋음
        minimize_metric = primary_metric_name in ("recon_loss", "val_loss")
        best_metric = math.inf if minimize_metric else -math.inf
        best_epoch = 0
        epochs_done = 0
        patience_counter = 0

        restored = self._restore_training_state(model, optimizer, scheduler, scaler, run_dir)
        start_epoch = restored.get("next_epoch", 1)
        if restored:
            best_metric = restored["best_metric"]
            best_epoch = restored["best_epoch"]
            epochs_done = start_epoch - 1
            patience_counter = restored["patience_counter"]
            metrics_history = restored["metrics_history"]
            lr_history = restored["lr_history"]

        self.signals.log_message.emit(
            f"  주요 메트릭: {primary_metric_name}"
        )

        # Publish the resume baseline before any batch or completed-epoch event.
        self.signals.progress_updated.emit(start_epoch - 1, cfg.epochs)
        for epoch in range(start_epoch, cfg.epochs + 1):
            if self._stop_requested:
                self.signals.log_message.emit("학습 중지됨")
                break

            # ── Train (AMP 적용) ──
            self._training_clock.start_epoch()
            train_loss = self._train_one_epoch(
                model, train_loader, criterion, optimizer,
                device, epoch, cfg.epochs,
                scaler=scaler, use_amp=use_amp,
            )

            if self._stop_requested:
                break

            # ── Validate (간략 메트릭) ──
            val_loss, epoch_metrics = self._validate(
                model, val_loader, criterion, device, task, data_cfg
            )
            if primary_metric_name == "val_loss":
                epoch_metrics["val_loss"] = val_loss

            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise ValueError("학습 또는 검증 손실이 유한한 값이 아님")
            epochs_done = epoch
            run_record.epochs_done = epochs_done

            # ── Scheduler step ──
            if scheduler is not None:
                scheduler.step()

            # ── Learning Rate 기록 ──
            current_lr = optimizer.param_groups[0]["lr"]
            lr_history.append(current_lr)
            self.signals.lr_updated.emit(epoch, current_lr)

            # ── 메트릭 이력 저장 ──
            metrics_history["train_loss"].append(train_loss)
            metrics_history["val_loss"].append(val_loss)
            for key in set(metric_info.get("display", []) + [primary_metric_name]):
                if key not in {"train_loss", "val_loss"}:
                    metrics_history.setdefault(key, []).append(epoch_metrics.get(key))

            # 최초 유효 에폭은 0점이어도 보존한다.
            current_metric = val_loss if primary_metric_name == "val_loss" else epoch_metrics.get(primary_metric_name)
            if current_metric is None or not math.isfinite(current_metric):
                raise ValueError(f"모델 선택 지표 계산 불가: {primary_metric_name}")
            is_better = (best_epoch == 0 or
                         (current_metric < best_metric if minimize_metric
                          else current_metric > best_metric))
            if is_better:
                best_metric, best_epoch, patience_counter = current_metric, epoch, 0
            else:
                patience_counter += 1
            checkpoint = self._checkpoint_metadata()
            checkpoint.update({
                "epoch": epoch,
                "model_state_dict": self._cpu_state(model.state_dict()),
                "optimizer_state_dict": self._cpu_state(optimizer.state_dict()),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_metric": best_metric, "best_epoch": best_epoch,
                "best_metric_name": primary_metric_name, "trained_device": str(device),
            })
            if getattr(self, "_class_weight_details", None) is not None:
                checkpoint["class_weight_details"] = self._class_weight_details
            record_epoch_time(self._training_clock, metrics_history,
                              self.signals.log_message.emit, epoch)
            epoch_metrics = {**epoch_metrics,
                             "epoch_time_sec": metrics_history["epoch_time_sec"][-1],
                             "elapsed_time_sec": metrics_history["elapsed_time_sec"][-1]}

            # ── 시그널 emit → GUI 차트 및 시간 카드 업데이트 ──
            self.signals.epoch_finished.emit(epoch, train_loss, val_loss, epoch_metrics)
            self.signals.progress_updated.emit(epoch, cfg.epochs)

            # ── 로그 ──
            metric_str = ", ".join(
                f"{metric_info.get('labels', {}).get(k, k)}: {v:.4f}"
                for k, v in epoch_metrics.items()
                if k not in {"epoch_time_sec", "elapsed_time_sec"}
                and isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v)
            )
            self.signals.log_message.emit(
                f"  Epoch {epoch}/{cfg.epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"LR: {current_lr:.6f} | {metric_str}"
            )
            checkpoint.update(self._checkpoint_extra(model, is_better, metrics_history,
                                                     lr_history, patience_counter))
            self._save_checkpoint(checkpoint, os.path.join(run_dir, "last.pt"))
            if is_better:
                self._save_checkpoint(checkpoint, os.path.join(run_dir, "best.pt"))
                self.signals.best_epoch_updated.emit(
                    epoch, train_loss, val_loss, dict(epoch_metrics)
                )
                self.signals.log_message.emit(
                    f"  Best 모델 저장 ({primary_metric_name}: {best_metric:.4f})"
                )

            # ── 조기 종료 ──
            if cfg.early_stop_patience > 0 and patience_counter >= cfg.early_stop_patience:
                self.signals.log_message.emit(
                    f"  조기 종료 (patience {cfg.early_stop_patience} 도달)"
                )
                self.signals.early_stopped.emit(epoch, best_metric)
                break

        if best_epoch == 0:
            run_record.status = "cancelled" if self._stop_requested else "failed"
            run_record.finished_at = datetime.now().isoformat()
            log_total_time(self.signals.log_message.emit, self._training_clock.elapsed())
            return

        # ── 10. 학습 후 종합 평가 ─────────────────────
        self.signals.log_message.emit("\n종합 평가 시작...")

        # Best 모델 로드 후 종합 평가 (CPU로 로드 후 device로 이동)
        ckpt_path = os.path.join(run_dir, "best.pt")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            self.signals.log_message.emit(
                f"  Best 모델 로드 (epoch {best_epoch})"
            )

        eval_results = {} if self._stop_requested else self._full_evaluation(
            model, val_loader, device, task, data_cfg
        )

        # 평가 시그널 emit → GUI 혼동 행렬 / 결과 테이블 업데이트
        self.signals.eval_finished.emit(eval_results)

        # ── 11. 학습 완료 처리 ─────────────────────────
        run_record.finished_at = datetime.now().isoformat()
        run_record.status = "cancelled" if self._stop_requested else "completed"
        run_record.best_metric = best_metric
        run_record.best_epoch = best_epoch
        run_record.checkpoint_path = ckpt_path
        run_record.epochs_done = epochs_done
        run_record.metrics_history = {
            k: v for k, v in metrics_history.items() if v
        }
        run_record.lr_history = lr_history

        # eval_results에서 confusion_matrix는 ndarray → 리스트로 변환 (JSON 저장용)
        eval_for_save = {}
        for k, v in eval_results.items():
            if isinstance(v, np.ndarray):
                eval_for_save[k] = v.tolist()
            elif isinstance(v, dict):
                # nested dict도 변환
                eval_for_save[k] = {
                    sk: sv.tolist() if isinstance(sv, np.ndarray) else sv
                    for sk, sv in v.items()
                }
            else:
                eval_for_save[k] = v
        run_record.eval_results = eval_for_save

        from core.training_artifacts import publish_best, training_hyperparameters
        hyperparameters = training_hyperparameters(
            self.project, engine=self.engine_name,
            effective={"device": str(device), "use_amp": use_amp,
                       "num_workers": num_workers, "pin_memory": pin_memory,
                       "pretrained_loaded": pretrained_loaded},
            applied={"class_weight_details": getattr(self, "_class_weight_details", None)},
        )
        ckpt_path, selection = publish_best(
            ckpt_path, run_dir, run_record.metrics_history, epoch=best_epoch,
            metric=primary_metric_name, value=best_metric,
            direction="min" if minimize_metric else "max", engine=self.engine_name, task=task,
            policy="strict_improvement_first_tie", timing=self._training_clock,
            formula=selected_policy.formula, evaluation_metrics=eval_for_save,
            hyperparameters=hyperparameters)
        run_record.checkpoint_path = ckpt_path
        run_record.config_snapshot["best_selection"] = selection
        log_total_time(self.signals.log_message.emit, selection["total_seconds"])

        self.signals.training_finished.emit(best_metric, best_epoch, ckpt_path)
        self.signals.log_message.emit(
            f"\n학습 완료! Best: epoch {best_epoch}, "
            f"{primary_metric_name}: {best_metric:.4f}"
        )

    engine_name = "custom"

    def _prepare_training(self):
        """모델별 시작 설정을 확인한다."""

    def _build_model(self):
        cfg, model_cfg = self.project.training, self.project.model
        return CustomCSP(task=self.project.task, in_channels=cfg.in_channels,
                      num_classes=self.project.data.num_classes,
                      backbone_channels=model_cfg.backbone_channels,
                      csp_depth=model_cfg.csp_depth, dropout=model_cfg.dropout)

    def _initialize_model(self, model, weights_path, task):
        if weights_path:
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"사전학습 가중치 파일 없음: {weights_path}")
            self._load_pretrained(model, weights_path, task)
            return True
        self.signals.log_message.emit("  사전학습 없음 — 랜덤 초기화에서 학습 시작")
        return False

    def _checkpoint_metadata(self):
        from center_crop import configured_center_crop
        cfg, data, model = self.project.training, self.project.data, self.project.model
        return make_checkpoint_metadata(self.project.task, data.num_classes, data.class_names,
            cfg.input_size, cfg.in_channels,
            {"backbone_channels": list(model.backbone_channels),
             "csp_depth": list(model.csp_depth), "dropout": model.dropout},
            center_crop=configured_center_crop(cfg))

    def _restore_training_state(self, model, optimizer, scheduler, scaler, run_dir):
        return {}

    def _checkpoint_extra(self, model, is_better, metrics_history, lr_history, patience):
        return {}

    @staticmethod
    def _save_checkpoint(checkpoint, path):
        import tempfile
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=os.path.dirname(path))
        os.close(fd)
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _cpu_state(value):
        """중첩 optimizer 상태까지 CPU 텐서로 저장한다."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: TrainWorker._cpu_state(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(TrainWorker._cpu_state(item) for item in value)
        return value

    # ── 사전학습 가중치 로드 (트랜스퍼 러닝) ─────────────
    def _load_pretrained(self, model, weights_path, target_task):
        """
        트랜스퍼 러닝: 사전학습 가중치 로드

        전략:
        1. 같은 태스크 → 전체 모델 로드
        2. 다른 태스크 → 백본만 로드 (헤드 무시)

         실패는 반드시 예외로 알린다.
        이전 구현은 이름이 하나도 맞지 않아도 strict=False로
        빈 dict를 로드한 뒤 "전이 완료"를 출력해, 사용자가
        사전학습을 쓰고 있다고 믿는 채로 랜덤 초기화 모델을
        학습하는 조용한 실패가 발생했다.
        """
        self.signals.log_message.emit(f"  사전학습 가중치: {weights_path}")

        # CPU로 로드 후 모델에 적용 (모델은 이미 device에 있음)
        checkpoint = torch.load(
            weights_path, map_location="cpu", weights_only=False
        )

        # Module-serialized and foreign training checkpoints cannot be transferred.
        if isinstance(checkpoint, dict) and (
            isinstance(checkpoint.get("model"), nn.Module)
            or isinstance(checkpoint.get("ema"), nn.Module)
            or "train_args" in checkpoint
        ):
            raise ValueError("지원하지 않는 체크포인트 구조입니다. 동일한 자체 백본의 state_dict 가중치를 선택하세요.")

        # ── state_dict 추출 (텐서만) ──
        raw = checkpoint.get("model_state_dict", checkpoint) \
            if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(raw, dict):
            raise ValueError(
                f"지원하지 않는 체크포인트 형식입니다: {type(raw).__name__}"
            )
        state_dict = {
            k: v for k, v in raw.items() if isinstance(v, torch.Tensor)
        }
        if not state_dict:
            raise ValueError(
                "체크포인트에 가중치 텐서 없음.\n"
                "이 GUI가 저장한 best.pt / last.pt 인지 확인하세요."
            )

        source_task = checkpoint.get("task") if isinstance(checkpoint, dict) else None

        source_encoding = (checkpoint.get("model_config") or {}).get("detection_box_encoding", "legacy_raw")
        incompatible_boxes = (source_task == target_task == "detect"
                              and source_encoding != getattr(model, "detection_box_encoding", "grid_sigmoid_xywh"))
        if incompatible_boxes:
            self.signals.log_message.emit("  검출 좌표 형식 변경: 기존 가중치의 백본만 전이하고 새 검출 헤드 학습")
        # 같은 태스크라도 검출 좌표 정의가 다르면 백본만 전이한다.
        if source_task == target_task and not incompatible_boxes:
            candidate = state_dict
            scope = "전체 모델"
        else:
            candidate = {
                k: v for k, v in state_dict.items() if k.startswith("backbone.")
            }
            scope = "백본"

        # ── 이름과 shape이 모두 맞는 것만 로드 ──
        #  백본 채널/깊이 설정이 다르면 이름은 같아도 shape이 달라
        #  load_state_dict가 예외를 던지므로 미리 걸러낸다.
        model_sd = model.state_dict()
        loadable, shape_mismatch, unknown = {}, [], []
        for k, v in candidate.items():
            if k not in model_sd:
                unknown.append(k)
            elif model_sd[k].shape != v.shape:
                shape_mismatch.append(k)
            else:
                loadable[k] = v

        if not loadable:
            raise ValueError(
                f"체크포인트에서 옮길 수 있는 가중치가 없습니다 "
                f"(이름 불일치 {len(unknown)}개, 크기 불일치 "
                f"{len(shape_mismatch)}개).\n"
                f"소스 태스크: {source_task} / 타겟 태스크: {target_task}\n\n"
                "다른 아키텍처의 체크포인트이거나 백본 설정(채널·깊이)이 "
                "다른 것으로 보입니다.\n"
                "이 GUI에서 같은 모델 설정으로 학습한 best.pt 를 사용하세요."
            )

        model.load_state_dict(loadable, strict=False)

        # ── 실제로 적재된 양을 정직하게 보고 ──
        target_total = len([
            k for k in model_sd
            if (scope == "전체 모델" or k.startswith("backbone."))
        ])
        ratio = len(loadable) / max(target_total, 1) * 100
        self.signals.log_message.emit(
            f"  {scope} 가중치 로드 "
            f"(소스: {source_task} → 타겟: {target_task})"
        )
        self.signals.log_message.emit(
            f"  적재 {len(loadable)}개 / 대상 {target_total}개 ({ratio:.0f}%)"
        )
        if shape_mismatch:
            self.signals.log_message.emit(
                f"  크기 불일치로 건너뜀: {len(shape_mismatch)}개 "
                f"(예: {shape_mismatch[0]})"
            )
        if unknown:
            self.signals.log_message.emit(
                f"  모델에 없는 키: {len(unknown)}개 (예: {unknown[0]})"
            )
        if ratio < 50:
            self.signals.log_message.emit(
                "  절반도 채우지 못했습니다 — 체크포인트가 "
                "현재 모델 설정과 맞는지 확인하세요."
            )

    # ── 옵티마이저 생성 ───────────────────────────────
    def _create_optimizer(self, model, cfg, model_cfg):
        """차등 학습률 지원 옵티마이저"""
        if model_cfg.freeze_backbone:
            # 백본 동결 시 → 헤드 파라미터만
            params = [p for p in model.head.parameters() if p.requires_grad]
        elif (model_cfg.pretrained_weights or self.engine_name == "efficientnet") and model_cfg.backbone_lr_mult < 1.0:
            # 차등 학습률: 백본 낮은 LR, 헤드 높은 LR
            params = [
                {"params": model.backbone.parameters(),
                 "lr": cfg.learning_rate * model_cfg.backbone_lr_mult},
                {"params": model.head.parameters(),
                 "lr": cfg.learning_rate},
            ]
        else:
            params = model.parameters()

        if cfg.optimizer == "adamw":
            return optim.AdamW(params, lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "sgd":
            return optim.SGD(params, lr=cfg.learning_rate,
                             momentum=0.9, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "adam":
            return optim.Adam(params, lr=cfg.learning_rate,
                              weight_decay=cfg.weight_decay)
        else:
            return optim.AdamW(params, lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)

    # ── 스케줄러 생성 ─────────────────────────────────
    def _create_scheduler(self, optimizer, cfg):
        if cfg.scheduler == "cosine":
            if cfg.epochs < 1 or cfg.warmup_epochs < 0:
                raise ValueError("에폭은 1 이상, warmup은 0 이상 필요")
            warmup = min(cfg.warmup_epochs, cfg.epochs - 1)

            def lr_factor(epoch):
                if warmup and epoch < warmup:
                    return (epoch + 1) / warmup
                progress = min(max((epoch - warmup) / max(cfg.epochs - warmup, 1), 0), 1)
                return 0.5 * (1 + math.cos(math.pi * progress))

            return optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        elif cfg.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=max(cfg.epochs // 3, 1), gamma=0.1
            )
        return None

    # ── 클래스 가중치 계산 ────────────────────────────
    def _compute_class_weights(self, train_loader, num_classes, mode, device):
        """현재 학습 subset의 클래스 ID 순서에 맞춰 가중치와 기록 생성."""
        counts = class_counts(train_loader.dataset, num_classes)
        weights = weights_from_counts(counts, mode)
        names = list(self.project.data.class_names)
        if len(names) != num_classes:
            raise ValueError("클래스 이름과 모델 출력 수 불일치")
        self._class_weight_details = {
            "mode": mode, "class_names": names, "counts": counts,
            "weights": weights or [1.0] * num_classes,
            "normalization": "training_sample_mean_1", "reduction": "batch_mean",
        }
        self.signals.log_message.emit(
            "  " + describe_class_weights(names, counts, weights, mode))
        self.signals.log_message.emit(
            "  클래스 가중치는 학습 손실에 적용. 검증 손실은 가중치 없이 계산하며 "
            "설정한 Label smoothing 유지. 클래스별 Recall과 F1 비교 필요"
        )
        return None if weights is None else torch.tensor(weights, device=device)

    # ── 손실 함수 생성 ────────────────────────────────
    def _create_criterion(self, task, cfg, train_loader=None,
                          num_classes=None, device=None):
        """태스크별 손실 함수 (classification은 클래스 가중치 지원)"""
        if task == "classify":
            weights = None
            mode = getattr(cfg, "class_weights", "none")
            if train_loader is not None and num_classes:
                weights = self._compute_class_weights(
                    train_loader, num_classes, mode, device
                )
            elif mode != "none":
                raise ValueError("클래스 가중치 적용을 위한 학습 데이터 로더 필요")
            return WeightedClassificationLoss(
                weights=weights, label_smoothing=cfg.label_smoothing
            )
        elif task == "segment":
            return nn.CrossEntropyLoss(ignore_index=255)
        elif task == "detect":
            # Detection은 별도 복합 손실 (cls + bbox + obj)
            return None  # _train_one_epoch에서 직접 계산
        elif task == "anomaly":
            return nn.MSELoss()
        return nn.CrossEntropyLoss()

    # ── 데이터 로더 생성 ──────────────────────────────
    def _create_dataloaders(self, task, data_cfg, cfg, num_workers=0, pin_memory=False):
        """
        태스크별 데이터 로더 생성

        각 태스크에 맞는 전용 Dataset + DataLoader 사용:
        - classify: ClassificationDataset (폴더 기반)
        - segment: SegmentationDataset (이미지+마스크 쌍)
        - detect: DetectionDataset (정규화 좌표 포맷 레이블)
        - anomaly: AnomalyDataset (정상 이미지 재구성)

        GPU 사용 시 num_workers > 0으로 데이터 로딩 병렬화
        """
        self._dataloader_error = ""
        try:
            from center_crop import configured_center_crop
            from crop_dataset import prepare_crop_dataset
            crop = configured_center_crop(cfg)
            data_root = data_cfg.root
            if crop:
                data_root = prepare_crop_dataset(
                    data_cfg.root, os.path.join(self.project.project_dir, "generated"),
                    task, crop, num_classes=data_cfg.num_classes,
                    should_stop=lambda: self._stop_requested, log=self.signals.log_message.emit)
            from dataset import (
                create_classification_loaders,
                create_segmentation_loaders,
                create_anomaly_loaders,
                create_detection_loaders,
            )

            worker_mode = "persistent/prefetch=2" if num_workers else "main process"
            self.signals.log_message.emit(
                f"  데이터 로더: workers={num_workers}, pin_memory={pin_memory}, {worker_mode}"
            )

            # ── 입력 크기를 (H, W) 튜플로 정규화 ──
            #  정수를 그대로 넘기면 T.Resize(int)가 종횡비를 보존해
            #  224×298 같은 비정사각 텐서가 나오고, 추론/내보내기
            #  (224×224)와 전처리가 어긋난다(train/serve skew).
            #  PIL Image.resize()는 정수를 아예 받지 못해 크래시한다.
            size = cfg.input_size
            input_size = (int(size), int(size)) if isinstance(size, (int, float)) \
                else tuple(size)

            # ── GUI 데이터 증강 설정 ──
            #  이 값을 전달하지 않으면 dataset.py의 기본값
            #  (좌우반전 0.5 / 회전 15° / 색상 0.2)이 항상 적용되어
            #  방향·위치로 정의되는 클래스의 학습 신호가 파괴된다.
            aug = getattr(cfg, "augmentation", None)
            flip_prob = getattr(aug, "horizontal_flip", 0.5) if aug else 0.5
            rotation = getattr(aug, "rotation", 15.0) if aug else 15.0
            color_jitter = getattr(aug, "color_jitter", 0.2) if aug else 0.2
            self.signals.log_message.emit(
                f"  입력 크기: {input_size[0]}×{input_size[1]} | "
                f"증강 — 좌우반전 {flip_prob}, 회전 {rotation}°, 색상 {color_jitter}"
            )

            if task == "classify":
                result = create_classification_loaders(
                    data_root=data_root,
                    input_size=input_size,
                    batch_size=cfg.batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    in_channels=cfg.in_channels,
                    flip_prob=flip_prob,
                    rotation=rotation,
                    color_jitter=color_jitter,
                    val_split=data_cfg.val_split,
                )
                # (train_loader, val_loader, class_names) 반환
                if len(result) == 3:
                    train_loader, val_loader, class_names = result
                    # ── 클래스 순서 강제 동기화 ──────────────
                    # 데이터셋은 폴더명을 sorted()로 정렬하여
                    # 인덱스를 부여함 (예: flip_over=0, ok=1).
                    # 사용자가 프로젝트 생성 시 입력한 순서
                    # (예: ok=0, flip_over=1)와 다를 수 있으므로,
                    # 항상 데이터셋의 실제 정렬 순서로 덮어써야
                    # Confusion Matrix 라벨이 올바르게 매핑됨.
                    if class_names:
                        if list(class_names) != data_cfg.class_names:
                            self.signals.log_message.emit(
                                f"  클래스 순서 동기화: "
                                f"{data_cfg.class_names} → {list(class_names)}"
                            )
                        data_cfg.class_names = list(class_names)
                        data_cfg.num_classes = len(class_names)
                    self.signals.log_message.emit(
                        f"  클래스: {data_cfg.class_names}"
                    )
                    return train_loader, val_loader
                return result

            elif task == "segment":
                return create_segmentation_loaders(
                    data_root=data_root,
                    input_size=input_size,
                    batch_size=cfg.batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    in_channels=cfg.in_channels,
                    flip_prob=flip_prob,
                    num_classes=data_cfg.num_classes,
                )

            elif task == "detect":
                self.signals.log_message.emit(
                    "  Detection 데이터 로더 (정규화 좌표 포맷)"
                )
                return create_detection_loaders(
                    data_root=data_root,
                    input_size=input_size,
                    batch_size=cfg.batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    num_classes=data_cfg.num_classes,
                    in_channels=cfg.in_channels,
                    flip_prob=flip_prob,
                )

            elif task == "anomaly":
                self.signals.log_message.emit(
                    "  Anomaly 데이터 로더 (정상 이미지 재구성)"
                )
                return create_anomaly_loaders(
                    data_root=data_root,
                    input_size=input_size,
                    batch_size=cfg.batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    in_channels=cfg.in_channels,
                    val_split=data_cfg.val_split,
                    flip_prob=flip_prob,
                )

        except Exception as e:
            self._dataloader_error = f"{type(e).__name__}: {e}"
            self.signals.log_message.emit(f"데이터 로딩 오류: {self._dataloader_error}")
            import traceback
            self.signals.log_message.emit(traceback.format_exc())
            return None, None

    def _criterion_loss(self, criterion, outputs, targets):
        """전체 ignore 마스크의 CE NaN을 피하고 나머지는 원래 손실을 사용한다."""
        if self.project.task == "segment" and not bool((targets != 255).any()):
            return outputs.sum() * 0.0
        return criterion(outputs, targets)

    # ── 1 에폭 학습 ───────────────────────────────────
    def _train_one_epoch(self, model, loader, criterion, optimizer,
                         device, epoch, total_epochs,
                         scaler=None, use_amp=False):
        """
        한 에폭 학습 수행

        AMP (Mixed Precision) 지원:
        ┌──────────────────────────────────────────────┐
        │ GPU + AMP 활성화 시:                          │
        │  1. autocast로 FP16 순전파 → 메모리 50% 절약  │
        │  2. GradScaler로 안전한 역전파                 │
        │  3. 학습 속도 2~3배 향상                       │
        │                                              │
        │ CPU 사용 시:                                  │
        │  기존 FP32 학습 그대로 유지                     │
        └──────────────────────────────────────────────┘
        """
        model.train()
        if self.project.model.freeze_backbone:
            model.backbone.eval()
        total_loss = 0.0
        num_batches = len(loader)
        if num_batches == 0:
            raise ValueError("학습 배치 없음")
        processed_batches = 0

        for batch_idx, batch in enumerate(loader):
            if self._stop_requested:
                break

            # 배치 데이터 언팩 (태스크에 따라 다름)
            # ┌─────────────────────────────────────────────────────┐
            # │ AnomalyDataset 학습 시 반환:                        │
            # │   images  = 정규화된 입력 (backbone 투입용)          │
            # │   targets = 원본 [0,1] 이미지 (재구성 타겟)          │
            # │                                                     │
            # │ ※ loss = MSE(model(normalized), original_[0,1])     │
            # │   정규화된 입력을 타겟으로 쓰면 identity mapping     │
            # │   학습이 되어 이상 탐지가 불가능해진다               │
            # └─────────────────────────────────────────────────────┘
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
            else:
                images = batch
                targets = images

            # GPU로 데이터 전송 (non_blocking으로 비동기 전송)
            images = images.to(device, non_blocking=True)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            debug_session = getattr(self, "_layer_debug_session", None)
            if debug_session is not None:
                debug_session.begin(model, scaler.get_scale() if use_amp and scaler is not None else 1.0)

            # AMP 활성화 시 autocast + GradScaler 사용
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    if self.project.task == "anomaly":
                        # 재구성 타겟 = 원본 [0,1] 이미지 (targets)
                        loss = self._criterion_loss(criterion, outputs, targets)
                    elif criterion is not None:
                        loss = self._criterion_loss(criterion, outputs, targets)
                    else:
                        loss = self._compute_detection_loss(outputs, targets)

                # Scaled 역전파
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 기본 FP32 학습 (CPU 또는 AMP 미지원 GPU)
                outputs = model(images)
                if self.project.task == "anomaly":
                    loss = self._criterion_loss(criterion, outputs, targets)
                elif criterion is not None:
                    loss = self._criterion_loss(criterion, outputs, targets)
                else:
                    loss = self._compute_detection_loss(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            if debug_session is not None:
                debug_session.capture(epoch, batch_idx + 1)
            total_loss += loss.item()
            processed_batches += 1

            # 배치 시그널 (10배치마다)
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                self.signals.batch_finished.emit(
                    epoch, batch_idx + 1, num_batches, loss.item()
                )

        # GPU 메모리 정리
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return total_loss / max(processed_batches, 1)

    # ── 검증 (에폭 단위 — 간략 메트릭) ────────────────
    @torch.no_grad()
    def _validate(self, model, loader, criterion, device, task, data_cfg):
        """
        에폭 단위 검증 — 간략 메트릭 계산 (실시간 차트용)

        각 태스크별 적절한 메트릭을 계산하여 반환:
        - classify: accuracy, precision, recall, f1
        - segment: mIoU, dice, pixel accuracy
        - detect: val_loss (전체 mAP는 학습 후 종합 평가에서)
        - anomaly: recon_loss, auroc (테스트셋 있을 때)
        """
        model.eval()
        total_loss = 0.0
        metrics = {}

        if loader is None:
            return 0.0, {"val_metric": 0.0}
        non_blocking = getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"

        # 태스크별 메트릭 수집기 생성
        num_classes = max(data_cfg.num_classes, 1)
        meter = create_metrics(task, num_classes, data_cfg.class_names)

        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
            else:
                images = batch
                targets = images

            images = images.to(device, non_blocking=non_blocking)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device, non_blocking=non_blocking)

            outputs = model(images)

            # ── 손실 계산 ──
            # anomaly: 재구성 타겟 = 원본 [0,1] 이미지 (targets)
            if task == "anomaly":
                loss = nn.functional.mse_loss(outputs, targets)
            elif task == "classify":
                # 모드 간 검증 손실 비교를 위해 학습 가중치를 제외한다.
                loss = F.cross_entropy(
                    outputs, targets,
                    label_smoothing=getattr(criterion, "label_smoothing", 0.0),
                )
            elif criterion is not None:
                loss = self._criterion_loss(criterion, outputs, targets)
            else:
                loss = self._compute_detection_loss(outputs, targets)

            total_loss += loss.item()

            # ── 태스크별 메트릭 누적 ──
            if task == "classify":
                preds = outputs.argmax(dim=1).cpu().numpy()
                gt = targets.cpu().numpy()
                meter.update(preds, gt)

            elif task == "segment":
                preds = outputs.argmax(dim=1).cpu().numpy()
                gt = targets.cpu().numpy()
                meter.update(preds, gt)

            elif task == "detect":
                from detection import detection_metric_records
                for bi in range(outputs.shape[0]):
                    predictions, ground_truth = detection_metric_records(outputs, targets, bi)
                    meter.update(predictions, ground_truth)

            elif task == "anomaly":
                # 재구성 오차를 이상 스코어로 사용 (원본 타겟 기준)
                recon_error = (outputs - targets).pow(2).mean(dim=(1, 2, 3))
                scores = recon_error.cpu().numpy()
                # anomaly 데이터 로더가 (image, label) 형태인 경우
                if isinstance(targets, torch.Tensor) and targets.ndim == 1:
                    labels = targets.cpu().numpy()
                else:
                    # 학습 중에는 모두 정상(0) 처리
                    labels = np.zeros(len(scores))
                meter.update(scores, labels)

        num_batches = max(len(loader), 1)
        avg_loss = total_loss / num_batches

        # ── 메트릭 계산 ──
        computed = meter.compute()
        if task == "segment" and computed.get("evaluable") is False:
            raise ValueError("검증 마스크에 평가 가능한 픽셀 없음 (모두 ignore)")

        if task == "classify":
            metrics = {
                "accuracy": computed["accuracy"],
                "precision_macro": computed["precision_macro"],
                "recall_macro": computed["recall_macro"],
                "f1_macro": computed["f1_macro"],
                "val_loss": avg_loss,
            }
        elif task == "segment":
            metrics = {
                "mIoU": computed["mIoU"],
                "dice_score": computed["dice_score"],
                "pixel_accuracy": computed["pixel_accuracy"],
                "val_loss": avg_loss,
            }
        elif task == "detect":
            metrics = {
                "mAP_50": computed.get("mAP_50", 0.0),
                "mAP_50_95": computed.get("mAP_50_95", 0.0),
                "precision": computed.get("precision", 0.0),
                "recall": computed.get("recall", 0.0),
                "val_loss": avg_loss,
            }
        elif task == "anomaly":
            metrics = {
                "auroc": computed.get("auroc"),
                "f1": computed.get("f1"),
                "precision": computed.get("precision"),
                "recall": computed.get("recall"),
                "recon_loss": avg_loss,
            }

        return avg_loss, metrics

    # ── 종합 평가 (학습 완료 후) ──────────────────────
    @torch.no_grad()
    def _full_evaluation(self, model, loader, device, task, data_cfg):
        """
        학습 완료 후 전체 검증셋에 대한 종합 평가

        Confusion Matrix, Per-class 메트릭, ROC Curve 등
        모든 상세 지표를 계산하여 반환
        """
        model.eval()

        if loader is None:
            self.signals.log_message.emit("  검증 데이터 없음 — 평가 스킵")
            return {}
        non_blocking = getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"

        num_classes = max(data_cfg.num_classes, 1)
        meter = create_metrics(task, num_classes, data_cfg.class_names)

        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
            else:
                images = batch
                targets = images

            images = images.to(device, non_blocking=non_blocking)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device, non_blocking=non_blocking)

            outputs = model(images)

            if task == "classify":
                preds = outputs.argmax(dim=1).cpu().numpy()
                gt = targets.cpu().numpy()
                meter.update(preds, gt)

            elif task == "segment":
                preds = outputs.argmax(dim=1).cpu().numpy()
                gt = targets.cpu().numpy()
                meter.update(preds, gt)

            elif task == "detect":
                from detection import detection_metric_records
                for bi in range(outputs.shape[0]):
                    predictions, ground_truth = detection_metric_records(outputs, targets, bi)
                    meter.update(predictions, ground_truth)

            elif task == "anomaly":
                # 재구성 오차: 원본 [0,1] 이미지(targets) 기준
                # ※ images는 정규화된 입력 — 타겟으로 쓰면 identity mapping
                recon_error = (outputs - targets).pow(2).mean(dim=(1, 2, 3))
                scores = recon_error.cpu().numpy()
                if isinstance(targets, torch.Tensor) and targets.ndim == 1:
                    labels = targets.cpu().numpy()
                else:
                    labels = np.zeros(len(scores))
                meter.update(scores, labels)

        results = meter.compute()

        # 로그 출력
        self.signals.log_message.emit("  ─── 종합 평가 결과 ───")
        metric_info = TASK_METRIC_NAMES.get(task, {})
        for key in metric_info.get("display", []):
            if key in results and isinstance(results[key], (int, float)):
                label = metric_info.get("labels", {}).get(key, key)
                self.signals.log_message.emit(f"  {label}: {results[key]:.4f}")

        # per_class 로그
        if "per_class" in results:
            self.signals.log_message.emit("  ─── 클래스별 ───")
            def shown(value):
                return f"{value:.3f}" if isinstance(value, (int, float)) and math.isfinite(value) else "N/A"
            for pc in results["per_class"]:
                name = pc["name"]
                if task == "classify":
                    self.signals.log_message.emit(
                        f"  {name}: P={shown(pc.get('precision'))} "
                        f"R={shown(pc.get('recall'))} F1={shown(pc.get('f1'))} "
                        f"(n={pc['support']})"
                    )
                elif task == "segment":
                    self.signals.log_message.emit(
                        f"  {name}: IoU={shown(pc.get('iou'))} "
                        f"Dice={shown(pc.get('dice'))}"
                    )
                elif task == "detect":
                    self.signals.log_message.emit(
                        f"  {name}: AP@0.5={shown(pc.get('ap_50'))} "
                        f"AP@.5:.95={shown(pc.get('ap_50_95'))}"
                    )

        # 태스크 이름 추가
        results["task"] = task
        return results

    # ── Detection 손실 (Focal + CIoU + BCE) ─────────────
    def _compute_detection_loss(self, outputs, targets):
        """
        Detection 복합 손실 함수

        구성:
        ┌──────────────────────────────────────────────────┐
        │ L_total = λ_box × L_CIoU                         │
        │        + λ_obj × L_obj  (Focal BCE)               │
        │        + λ_cls × L_cls  (BCE)                     │
        │                                                    │
        │ L_CIoU : Complete IoU — 겹침+중심거리+종횡비 반영   │
        │ L_obj  : Focal Loss — 양/음성 불균형 해결           │
        │ L_cls  : BCE — 다중 라벨 분류                      │
        └──────────────────────────────────────────────────┘

        Args:
            outputs: (B, N, 5+C) — [cx, cy, w, h, obj, cls...]
            targets: list of (M, 6) — [batch_idx, cls, cx, cy, w, h]
                     또는 (B, M, 5) — [cls, cx, cy, w, h]
        """
        outputs = outputs.float()
        if outputs.ndim != 3 or outputs.shape[-1] < 6 or not torch.isfinite(outputs).all():
            raise ValueError("검출 출력 형식/유한값 오류")
        device = outputs.device
        B, N, dim = outputs.shape
        num_classes = dim - 5

        # 손실 가중치
        lambda_box = 5.0
        lambda_obj = 1.0
        lambda_cls = 1.0

        # 예측 분리
        pred_bbox = outputs[:, :, :4]       # (B, N, 4) — cx, cy, w, h
        pred_obj = outputs[:, :, 4]         # (B, N) — objectness
        pred_cls = outputs[:, :, 5:]        # (B, N, C) — class logits

        # ── 타겟 파싱 ──────────────────────────────────
        # 타겟을 배치별로 분리
        if isinstance(targets, torch.Tensor):
            if targets.dim() == 2 and targets.shape[-1] == 6:
                # (M, 6) 형태: [batch_idx, cls, cx, cy, w, h]
                target_list = []
                for b in range(B):
                    mask = targets[:, 0].long() == b
                    target_list.append(targets[mask, 1:])  # (Mi, 5)
            elif targets.dim() == 3 and targets.shape[0] == B and targets.shape[-1] == 5:
                # (B, M, 5) 형태: [cls, cx, cy, w, h]
                target_list = [targets[b] for b in range(B)]
            else:
                raise ValueError("검출 정답 텐서 형식 오류")
        elif isinstance(targets, (list, tuple)) and len(targets) == B:
            target_list = targets
        else:
            raise ValueError("검출 정답 배치 형식 오류")

        # ── 배치별 손실 계산 ───────────────────────────
        total_box_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        obj_target = torch.zeros_like(pred_obj)  # (B, N)

        num_pos = 0  # 양성 샘플 수

        for b in range(B):
            gt = target_list[b].to(device)
            if gt.ndim != 2 or gt.shape[-1] != 5 or not torch.isfinite(gt).all():
                raise ValueError("검출 정답은 유한한 (M,5) 배열 필요")

            # 패딩된 빈 행 제거 (w=0 또는 h=0)
            if gt.dim() == 2 and gt.shape[0] > 0:
                valid = (gt[:, 3] > 0) & (gt[:, 4] > 0)
                gt = gt[valid]

            if gt.shape[0] == 0:
                continue  # 이 배치에 GT 없음

            if (not torch.isfinite(gt).all() or (gt[:, 0] != gt[:, 0].long()).any()
                    or (gt[:, 0] < 0).any() or (gt[:, 0] >= num_classes).any()):
                raise ValueError("검출 정답의 클래스 번호 또는 좌표 오류")
            gt_cls = gt[:, 0].long()          # (M,)
            gt_bbox = gt[:, 1:5]              # (M, 4): cx, cy, w, h

            # ── 양성 매칭: 각 GT에 가장 가까운 예측 할당 ──
            # 예측 bbox 중심과 GT 중심의 L1 거리로 매칭
            pred_centers = pred_bbox[b, :, :2]  # (N, 2)
            gt_centers = gt_bbox[:, :2]          # (M, 2)

            # (N, M) 거리 행렬
            dist = torch.cdist(pred_centers.detach(), gt_centers.float(), p=1)

            # 각 GT의 후보 중 거리가 가장 가까운 GT 하나에만 예측을 배정한다.
            k = min(10, N)
            candidates = torch.full_like(dist, float("inf"))
            nearest = dist.topk(k, largest=False, dim=0).indices
            candidates.scatter_(0, nearest, dist.gather(0, nearest))
            costs, assigned = candidates.min(dim=1)
            for m in range(gt.shape[0]):
                indices = torch.where(torch.isfinite(costs) & (assigned == m))[0]
                if not len(indices):
                    continue
                ciou = self._ciou_loss(pred_bbox[b, indices], gt_bbox[m:m + 1].expand(len(indices), -1))
                total_box_loss = total_box_loss + ciou.sum()
                obj_target[b, indices] = 1.0
                cls_target = torch.zeros(len(indices), num_classes, device=device)
                cls_target[:, gt_cls[m]] = 1.0
                total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                    pred_cls[b, indices], cls_target, reduction="sum")
                num_pos += len(indices)

        # ── 전체 손실 합산 ─────────────────────────────
        num_pos = max(num_pos, 1)

        loss_box = total_box_loss / num_pos
        loss_obj = self._focal_bce(pred_obj, obj_target)
        loss_cls = total_cls_loss / num_pos

        total_loss = (
            lambda_box * loss_box
            + lambda_obj * loss_obj
            + lambda_cls * loss_cls
        )

        return total_loss

    @staticmethod
    def _ciou_loss(pred, target, eps=1e-7):
        """
        Complete IoU (CIoU) Loss

        CIoU = IoU − (ρ²(b, b_gt) / c²) − αv

        ┌────────────────────────────────────────────┐
        │ IoU  : 겹침 비율                            │
        │ ρ²/c²: 중심점 거리 / 최소 포함 대각선²       │
        │ αv   : 종횡비 일관성 패널티                  │
        └────────────────────────────────────────────┘

        Args:
            pred:   (N, 4) — cx, cy, w, h
            target: (N, 4) — cx, cy, w, h
        Returns:
            loss: (N,) — 1 - CIoU
        """
        import math

        # cx, cy, w, h → x1, y1, x2, y2
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        gt_x1 = target[:, 0] - target[:, 2] / 2
        gt_y1 = target[:, 1] - target[:, 3] / 2
        gt_x2 = target[:, 0] + target[:, 2] / 2
        gt_y2 = target[:, 1] + target[:, 3] / 2

        # 교집합
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                     (inter_y2 - inter_y1).clamp(min=0)

        # 합집합
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union_area = pred_area + gt_area - inter_area + eps

        iou = inter_area / union_area

        # 중심점 거리²
        center_dist = (pred[:, 0] - target[:, 0]) ** 2 + \
                      (pred[:, 1] - target[:, 1]) ** 2

        # 최소 포함 사각형 대각선²
        enclose_x1 = torch.min(pred_x1, gt_x1)
        enclose_y1 = torch.min(pred_y1, gt_y1)
        enclose_x2 = torch.max(pred_x2, gt_x2)
        enclose_y2 = torch.max(pred_y2, gt_y2)
        enclose_diag = (enclose_x2 - enclose_x1) ** 2 + \
                       (enclose_y2 - enclose_y1) ** 2 + eps

        # 종횡비 패널티
        v = (4 / (math.pi ** 2)) * (
            torch.atan(target[:, 2] / (target[:, 3] + eps))
            - torch.atan(pred[:, 2] / (pred[:, 3] + eps))
        ) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        ciou = iou - center_dist / enclose_diag - alpha * v

        return 1 - ciou  # loss = 1 - CIoU

    @staticmethod
    def _focal_bce(pred, target, gamma=2.0, alpha=0.25):
        """
        Focal Loss (BCE 기반)

        FL(p_t) = −α_t (1 − p_t)^γ log(p_t)

        양/음성 불균형 해결 — Detection에서 배경(음성)이 압도적으로 많은 문제
        """
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )
        p_t = torch.sigmoid(pred)
        p_t = target * p_t + (1 - target) * (1 - p_t)
        focal_weight = (1 - p_t) ** gamma

        alpha_t = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_t * focal_weight * bce

        return loss.mean()

    # ── 주요 메트릭 선택 ──────────────────────────────
    def _get_primary_metric(self, metrics, task):
        """태스크별 Best 판단 기준 메트릭"""
        metric_info = TASK_METRIC_NAMES.get(task, {})
        primary = metric_info.get("primary", "accuracy")
        return metrics.get(primary, 0.0)
