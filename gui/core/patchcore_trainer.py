"""Deep Vision Studio의 PatchCore 사전학습 전이 워커.

고정 백본으로 정상 특징을 한 번 추출하고 메모리 뱅크를 구축한다.
ImageNet, 로컬 백본 및 기존 PatchCore의 재구축/추가 학습을 지원한다.
실행 상태, 평가, Best 파일과 시간 CSV를 일반 학습 기록에 연결한다.
"""

import os
import traceback
from datetime import datetime

import numpy as np
import torch

from core.training_engine import TrainingEngine

# 프로젝트 모듈 경로 추가 (PyInstaller EXE 호환)
from core.paths import ensure_python_path
ensure_python_path()

from patchcore import PatchCore, PatchCoreCancelled
from core.project import ProjectData, RunRecord, ProjectManager
from core.device_manager import get_device_manager


class PatchCoreWorker(TrainingEngine):
    """
    PatchCore 메모리 뱅크 구축 워커 (별도 스레드)

    시그널:
    - progress_updated → 프로그레스 바 (배치/전체)
    - log_message → 학습 로그
    - training_finished → 완료 (메모리 뱅크 경로 포함)
    - training_error → 오류
    """

    def __init__(self, project: ProjectData, parent=None, *, signals=None, should_stop=None):
        super().__init__(project, signals=signals, should_stop=should_stop)
        self._stop_requested = False

    def stop(self):
        """중지 요청"""
        self._stop_requested = True

    def run(self):
        """실패/취소도 실행 기록에 반영하고 원인 단계를 로그에 남긴다."""
        try:
            self._run_patchcore()
        except PatchCoreCancelled:
            self.cancelled = True
            self._finish_failed_run("cancelled")
            self.signals.log_message.emit("PatchCore 구축 취소")
        except Exception as exc:
            self._finish_failed_run("failed")
            stage = getattr(self, "_stage", "초기화")
            message = f"PatchCore {stage} 오류: {exc}\n{traceback.format_exc()}"
            if getattr(self, "_run_dir", None):
                try:
                    with open(os.path.join(self._run_dir, "training_error.txt"), "w", encoding="utf-8") as stream:
                        stream.write(message)
                except OSError:
                    pass
            self.signals.training_error.emit(message)

    def _finish_failed_run(self, status):
        record = getattr(self, "_run_record", None)
        if record is not None:
            record.status = status
            record.finished_at = datetime.now().isoformat()
        clock = getattr(self, "_training_clock", None)
        if clock is not None:
            from core.training_time import log_total_time
            log_total_time(self.signals.log_message.emit, clock.elapsed())

    def _prepare_model(self, cfg, device):
        """선택한 사전학습 백본을 사용한다. 로드 실패를 무작위 가중치로 대체하지 않는다."""
        from center_crop import configured_center_crop
        crop = configured_center_crop(cfg)
        source = getattr(cfg, "patchcore_weight_source", "imagenet")
        name = getattr(cfg, "patchcore_backbone", "wide_resnet50_2")
        path = getattr(cfg, "patchcore_weights", "").strip()
        append = getattr(cfg, "patchcore_append", False)
        candidates = getattr(cfg, "patchcore_max_candidates", 20000)
        bank_limit = getattr(cfg, "patchcore_max_memory_bank", 4096)
        if source not in ("imagenet", "backbone", "patchcore"):
            raise ValueError(f"미지원 PatchCore 가중치 소스: {source}")
        if append and source != "patchcore":
            raise ValueError("추가 학습은 기존 PatchCore 체크포인트를 선택해야 합니다")
        if source != "imagenet" and not os.path.isfile(path):
            raise FileNotFoundError(f"선택한 가중치 파일 없음: {path}")
        if source == "patchcore":
            pc = PatchCore.load(path, device="cpu")
            if append:
                if pc.center_crop != crop:
                    raise ValueError(f"추가 학습 중앙 크롭 불일치: 기존 {pc.center_crop}, 현재 {crop}. 기존 설정을 사용하거나 새 뱅크를 구축해 주세요")
                if pc.input_size != cfg.input_size:
                    raise ValueError(f"추가 학습 입력 크기 불일치: 기존 {pc.input_size}, 현재 {cfg.input_size}. 기존 크기를 선택하거나 새 뱅크 구축 사용")
                if pc.memory_bank.shape[0] >= bank_limit:
                    raise ValueError(f"기존 대표 패치 {pc.memory_bank.shape[0]:,}개. 최대 대표 패치 수를 더 크게 설정 필요")
            else:
                pc.memory_bank = None
                pc.input_size, pc.preprocessing = int(cfg.input_size), "opencv_full_range_v2"
                pc.center_crop = crop
            pc.device = torch.device(device)
            pc.backbone.to(pc.device).eval()
            if pc.memory_bank is not None:
                pc.memory_bank = pc.memory_bank.to(pc.device)
            pc.max_candidates, pc.max_memory_bank = candidates, bank_limit
            pc.sampling_ratio, pc.n_neighbors = cfg.patchcore_sampling_ratio, cfg.patchcore_n_neighbors
            pc.seed = getattr(cfg, "patchcore_seed", 0)
            pc.weight_source = {**pc.weight_source, "transfer_checkpoint": os.path.basename(path),
                                "transfer_mode": "append" if append else "rebuild"}
        else:
            if source == "imagenet":
                self.signals.log_message.emit(
                    "  ImageNet 백본 준비: 캐시 확인 후 필요하면 시스템 인증서로 다운로드합니다.")
            pc = PatchCore(backbone_name=name, device=device,
                           sampling_ratio=cfg.patchcore_sampling_ratio, n_neighbors=cfg.patchcore_n_neighbors,
                           input_size=cfg.input_size, pretrained=source == "imagenet",
                           backbone_weights=path if source == "backbone" else "",
                           max_candidates=candidates, max_memory_bank=bank_limit,
                           seed=getattr(cfg, "patchcore_seed", 0), center_crop=crop)
        self.signals.log_message.emit(f"  백본: {pc.backbone_name} | 가중치: {pc.weight_source}")
        self.signals.log_message.emit("  백본 가중치와 BatchNorm 고정. 현재 정상 데이터로 메모리 뱅크 구축")
        self.signals.log_message.emit(
            f"  특징 {pc.backbone.feature_dim}차원 | 후보 최대 {candidates:,} / 대표 최대 {bank_limit:,} 패치")
        return pc

    def _run_patchcore(self):
        from core.training_time import TrainingClock, record_epoch_time, log_total_time
        from core.training_artifacts import publish_best
        self._training_clock = TrainingClock()
        cfg, data_cfg = self.project.training, self.project.data
        self._stage = "설정 검증"
        if not 0 < cfg.patchcore_sampling_ratio <= 1 or cfg.patchcore_n_neighbors < 1:
            raise ValueError("PatchCore 비율/이웃 수 범위 오류")
        if min(getattr(cfg, "patchcore_max_candidates", 20000), getattr(cfg, "patchcore_max_memory_bank", 4096)) < 1:
            raise ValueError("후보/대표 패치 한도는 1 이상 필요")
        model_name = ("patchcore_transfer" if getattr(cfg, "patchcore_weight_source", "imagenet") == "patchcore"
                      else getattr(cfg, "patchcore_backbone", "wide_resnet50_2"))
        run_id = ProjectManager.new_run_id(task="anomaly", model_name=model_name,
                                          input_size=cfg.input_size, project_dir=self.project.project_dir)
        self._run_dir = run_dir = os.path.join(self.project.project_dir, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        self._run_record = record = RunRecord(run_id=run_id, started_at=datetime.now().isoformat(), status="running")
        self.project.runs.append(record)
        self.signals.log_message.emit("PatchCore 사전학습 특징 전이 시작")
        self._stage = "데이터 확인"
        # 빈 폴더/잘못된 라벨은 대형 가중치 다운로드 전에 확인한다.
        train_loader, val_loader = self._create_dataloaders(data_cfg, cfg, 0)
        self._stage = "사전학습 가중치 로드"
        dm = get_device_manager()
        device = dm.get_device(getattr(cfg, "device", "auto"))
        self.signals.log_message.emit(f"  장치: {dm.get_device_label(device)}")
        if self._stop_requested:
            raise PatchCoreCancelled()
        pc = self._prepare_model(cfg, device)
        from patchcore_data import validate_crop_images
        def check_cancel():
            if self._stop_requested:
                raise PatchCoreCancelled()
        self._stage = "중앙 크롭 크기 확인"
        for loader in (train_loader, val_loader):
            if loader is not None:
                loader.dataset.preprocessing = pc.preprocessing
                validate_crop_images(loader.dataset.samples, pc.center_crop, pc.preprocessing, check_cancel)
        record.config_snapshot["patchcore_transfer"] = pc.weight_source
        if self._stop_requested:
            raise PatchCoreCancelled()
        self._stage = "정상 특징 추출 및 대표 패치 선별"
        self._training_clock.start_epoch()
        def progress(current, total, message):
            self.signals.progress_updated.emit(current, total)
            self.signals.log_message.emit(f"  {message}")
        pc.fit(train_loader, progress_callback=progress, cancel_callback=lambda: self._stop_requested,
               append=getattr(cfg, "patchcore_append", False))
        if self._stop_requested:
            raise PatchCoreCancelled()
        record.config_snapshot["patchcore_training"] = pc.training_metadata
        self._stage = "평가 및 임계값 보정"
        auroc = None
        evaluation = {"task": "anomaly", "method": "patchcore", "evaluable": False,
                      "auroc": None, "optimal_threshold": None, "reason": "평가 데이터 없음",
                      "split": ""}
        if val_loader is not None:
            auroc, evaluation = self._evaluate_patchcore(pc, val_loader, device)
        if self._stop_requested:
            raise PatchCoreCancelled()
        pc.anomaly_threshold = evaluation.get("optimal_threshold")
        pc.calibration = {"evaluable": evaluation.get("evaluable", False),
                          "reason": evaluation.get("reason", ""), "sample_count": evaluation.get("sample_count", 0),
                          "method": "validation_f1", "threshold_comparator": ">="}
        readiness = "ready" if pc.anomaly_threshold is not None else "uncalibrated"
        record.config_snapshot["patchcore_readiness"] = {
            "state": readiness, "evaluation_split": evaluation.get("split", ""),
            "reason": evaluation.get("reason", ""),
            "sample_count": evaluation.get("sample_count", 0),
        }
        self._stage = "체크포인트 및 CSV 저장"
        checkpoint = os.path.join(run_dir, "best.pt")
        pc.save(checkpoint)
        history = {"epoch": [1], "auroc": [auroc]}
        record_epoch_time(self._training_clock, history, self.signals.log_message.emit, 1)
        self.signals.epoch_finished.emit(1, float("nan"), float("nan"), {
            "epoch_time_sec": history["epoch_time_sec"][-1],
            "elapsed_time_sec": history["elapsed_time_sec"][-1],
        })
        self.signals.best_epoch_updated.emit(1, float("nan"), float("nan"),
            {key: evaluation.get(key) for key in ("auroc", "f1", "precision", "recall")})
        self.signals.eval_finished.emit(evaluation)
        checkpoint, selection = publish_best(
            checkpoint, run_dir, history, epoch=1, metric="auroc" if auroc is not None else "unavailable",
            value=auroc, direction="single_fit", engine="patchcore", task="anomaly",
            policy="single_memory_bank_snapshot_auroc_is_evaluation_only", timing=self._training_clock)
        record.finished_at = datetime.now().isoformat()
        record.status, record.epochs_done, record.best_epoch = "completed", 1, 1
        record.best_metric, record.best_metric_name = auroc if auroc is not None else 0., "auroc" if auroc is not None else "unavailable"
        record.checkpoint_path, record.metrics_history, record.eval_results = checkpoint, history, evaluation
        record.config_snapshot["best_selection"] = selection
        log_total_time(self.signals.log_message.emit, selection["total_seconds"])
        self.signals.training_finished.emit(auroc if auroc is not None else 0., 1, checkpoint)
        if readiness == "ready":
            self.signals.log_message.emit(f"PatchCore 완료 및 판정 보정: {checkpoint}")
        else:
            self.signals.log_message.emit(
                f"PatchCore 특징 뱅크 구축 완료 — 판정 미보정: {checkpoint}\n"
                "정상·불량이 모두 있는 val 또는 test 폴더로 다시 보정해야 C++ ONNX 배포와 OK/NG 판정이 가능합니다.")

    def _create_dataloaders(self, data_cfg, cfg, num_workers=0):
        from torch.utils.data import DataLoader
        from patchcore_data import discover_data, PatchCoreDataset
        from center_crop import configured_center_crop
        crop = configured_center_crop(cfg)
        if cfg.batch_size < 1 or cfg.input_size < 32:
            raise ValueError("배치 크기는 1 이상, 입력 크기는 32 이상 필요")
        training, validation, labels, split = discover_data(data_cfg.root)
        self._patchcore_evaluation_split = split
        batch = min(cfg.batch_size, max(1, 1048576 // (cfg.input_size * cfg.input_size)))
        self.signals.log_message.emit(f"  정상 학습 {len(training)}장 | 실제 배치 {batch} (설정 {cfg.batch_size})")
        self.signals.log_message.emit(f"  보정 평가 {split}: {len(validation)}장" if validation else "  평가 이미지 없음: 정상 뱅크만 구축, AUROC/임계값은 평가 불가")
        self.signals.log_message.emit("  OpenCV 정확 리사이즈 + RGB/회색조 ImageNet 정규화, 16비트 전체 범위 보존")
        if crop:
            self.signals.log_message.emit(f"  원본 중앙 {crop['width']}×{crop['height']} px 크롭 → 입력 {cfg.input_size}×{cfg.input_size} px. 불량 평가 이미지는 크롭 내부에 불량이 있어야 합니다")
        # GUI QThread 안에서 Windows spawn 워커를 다시 만들지 않는다.
        train_loader = DataLoader(PatchCoreDataset(training, cfg.input_size, center_crop=crop), batch_size=batch,
                                  shuffle=False, num_workers=0, drop_last=False)
        val_loader = (DataLoader(PatchCoreDataset(validation, cfg.input_size, labels, center_crop=crop), batch_size=batch,
                                 shuffle=False, num_workers=0, drop_last=False) if validation else None)
        return train_loader, val_loader

    @torch.no_grad()
    def _evaluate_patchcore(self, pc, val_loader, device):
        """
        PatchCore 검증 — AUROC 계산

        Args:
            pc: PatchCore 인스턴스
            val_loader: 검증 데이터 로더
            device: 연산 디바이스

        Returns:
            auroc: AUROC 값 (0~1)
            results: 평가 결과 딕셔너리
        """
        from core.metrics import create_metrics

        meter = create_metrics("anomaly", 2, ["normal", "anomaly"])

        for batch in val_loader:
            if self._stop_requested:
                raise PatchCoreCancelled()
            if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                raise ValueError("PatchCore 평가 데이터는 이미지와 정답 라벨을 반환해야 합니다")
            images, labels = batch

            images = images.to(device)
            scores, _ = pc.predict(images)

            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = np.asarray(labels)

            meter.update(scores, labels_np)

        results = meter.compute()
        auroc = results.get("auroc")

        self.signals.log_message.emit(
            f"  검증 AUROC: {auroc:.4f}" if auroc is not None
            else f"  검증 평가 불가: {results.get('reason', '')}"
        )
        if results.get("f1") is not None:
            self.signals.log_message.emit(
                f"  검증 F1: {results['f1']:.4f}"
            )

        results["task"] = "anomaly"
        results["method"] = "patchcore"
        results["split"] = getattr(self, "_patchcore_evaluation_split", "")
        return auroc, results
