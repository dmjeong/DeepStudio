"""학습 작업 조정과 화면 상태 관리. 폼과 결과 표시는 별도 모듈."""

import os
import time
import copy
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QWidget, QFileDialog, QMessageBox
from PySide6.QtGui import QTextCursor

from core.project import ProjectData, ProjectManager
from core.qt_training import TrainWorker, PatchCoreWorker
from core.training_modes import training_engine_name, training_capabilities, MODE_LABELS
from core.training_progress import remaining_seconds, run_description
# 마우스 휠로 하이퍼파라미터가 실수로 바뀌는 것을 막는 위젯


from widgets.training_results import _finite_metric, _format_metric, _metric_info_for
from widgets.training_form import TrainingForm

class TrainingWidget(TrainingForm, QWidget):
    """학습 페이지 — 실시간 모니터링 + 종합 평가 + 모델 비교"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.worker = None
        self._pack_job = None
        self._pack_error = ""
        self._run_project = None
        self._run_snapshot = None
        self._pending_result = None
        self._pending_error = None
        self._cancel_requested = False
        self._start_time = None
        self._progress_start_epoch = None
        self._progress_total_epochs = 0
        self._init_ui()

    def _on_mode_changed(self, index: int):
        """
        학습 모드 변경 → UI 표시 전환

        모드별 가시 위젯:
        ┌──────────────────────┬─────────┬──────────┬──────────┐
        │                      │ 파인튜닝 │ 이어학습 │ 커스텀   │
        ├──────────────────────┼─────────┼──────────┼──────────┤
        │ 모델 피커            │    O    │    -     │    -     │
        │ 이어학습 경로        │    -    │    O     │    -     │
        │ 학습 전략 (freeze)   │    O    │    O     │    -     │
        │ 커스텀 트랜스퍼러닝  │    -    │    -     │    O     │
        └──────────────────────┴─────────┴──────────┴──────────┘
        """
        is_anomaly = self.project is not None and self.project.task == "anomaly"
        mode = "custom" if is_anomaly else self.mode_combo.currentData()
        self.mode_group.setVisible(not is_anomaly)
        # 모델 피커: 파인튜닝에서만
        self.efficientnet_frame.setVisible(str(mode).startswith("efficientnet") and mode != "efficientnet_resume")
        self.efficientnet_model_combo.setEnabled(mode != "efficientnet_resume")
        # 이전 모델 경로: 이어학습에서만
        self.resume_frame.setVisible(mode in {"efficientnet_resume", "efficientnet_transfer"})
        from core.training_modes import MODE_HELP
        self.mode_help.setText(MODE_HELP.get(mode, "지원하지 않는 학습 모드입니다. 현재 엔진을 선택하세요."))
        self.hp_group.setEnabled(True)
        self.aug_group.setEnabled(mode not in {"efficientnet_resume"})
        self.resume_edit.setPlaceholderText("중단 시 last.pt" if str(mode).endswith("_resume") else "추가 학습할 best.pt")
        # 커스텀 트랜스퍼 러닝: 커스텀에서만
        self.tl_group.setVisible(mode == "custom")
        # EfficientNet 학습 전략 (freeze 등): 파인튜닝·이어학습에서만
        self.finetune_frame.setVisible(
            mode in ("efficientnet_finetune", "efficientnet_transfer")
        )
        self._update_selection_options()
        # Refresh the metric list before settings callbacks validate its value.
        self._on_patchcore_settings_changed()
        self._update_pack_controls()

    def _set_model_options(self, project):
        """Filter the shared catalog to the current task and restore model_id."""
        if not hasattr(self, "model_id_combo"):
            return
        from core.model_registry import registry_with_installed_packs
        registry, _ = registry_with_installed_packs()
        model_id = getattr(project.model, "model_id", "")
        self.model_id_combo.blockSignals(True)
        try:
            self.model_id_combo.clear()
            for spec in registry.list(project.task):
                self.model_id_combo.addItem(
                    spec.display_name, spec.model_id)
            if model_id and self.model_id_combo.findData(model_id) < 0:
                self.model_id_combo.addItem(f"저장된 모델 팩 · {model_id}", model_id)
            index = self.model_id_combo.findData(model_id)
            self.model_id_combo.setCurrentIndex(max(0, index))
        finally:
            self.model_id_combo.blockSignals(False)

    def _install_model_pack(self):
        """Import and validate one offline Docker model pack from the GUI."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "모델 팩 가져오기", "", "Deep Vision model pack (*.dvmodel)"
        )
        if not filepath:
            return
        try:
            from core.model_registry import (default_installed_model_root,
                                             registry_with_installed_packs)
            from model_runtime.pack_installer import PackInstaller
            root = default_installed_model_root()
            installed = PackInstaller(root).install(filepath)
            registry, errors = registry_with_installed_packs(root)
            if errors:
                raise ValueError("; ".join(errors))
            if self.project is not None and registry.get(installed.model_id).task == self.project.task:
                # Importing a pack makes it the explicit model selection for
                # this project.  Persist the directory so the worker cannot
                # accidentally use the legacy native training engine.
                self.project.model.model_id = installed.model_id
                self.project.model.pack_path = str(installed.path)
                ProjectManager.save(self.project)
                self._set_model_options(self.project)
                self._update_pack_controls()
            QMessageBox.information(
                self, "모델 팩 설치 완료",
                f"{installed.model_id} {installed.pack_version} 팩을 설치했습니다.\n{installed.path}"
            )
        except Exception as exc:
            QMessageBox.warning(self, "모델 팩 설치 실패", str(exc))

    def _on_model_id_changed(self, *_):
        """Keep the legacy mode selector consistent with a catalog adapter."""
        if self.project is None or not hasattr(self, "model_id_combo"):
            return
        model_id = self.model_id_combo.currentData() or ""
        try:
            from core.model_registry import installed_model_path, registry_with_installed_packs
            registry, _ = registry_with_installed_packs()
            spec = registry.get(model_id)
            if len(spec.input_channels) == 1:
                self.project.training.in_channels = spec.input_channels[0]
            if hasattr(self, "input_size_spin") and model_id not in {"efficientnet_b0", "efficientnet_b1"}:
                self.input_size_spin.setValue(spec.input_size[0])
            if "container" in spec.runtimes and "windows_native" not in spec.runtimes:
                installed = installed_model_path(model_id)
                self.project.model.pack_path = str(installed) if installed else ""
            else:
                self.project.model.pack_path = ""
        except ValueError:
            spec = None
        if model_id not in {"efficientnet_b0", "efficientnet_b1"}:
            current = self.mode_combo.currentData()
            if current in {"efficientnet_finetune", "efficientnet_transfer", "efficientnet_resume"}:
                custom = self.mode_combo.findData("custom")
                if custom >= 0:
                    self.mode_combo.setCurrentIndex(custom)
        if model_id in {"efficientnet_b0", "efficientnet_b1"}:
            self._on_efficientnet_model_changed()
        self._on_patchcore_settings_changed()
        self._update_pack_controls()

    def _selected_container_spec(self):
        """Return the selected container-only model and its installed pack path."""
        if self.project is None:
            return None, None
        model_id = getattr(self.project.model, "model_id", "")
        if not model_id:
            return None, None
        try:
            from core.model_registry import registry_with_installed_packs
            registry, _ = registry_with_installed_packs()
            spec = registry.get(model_id)
        except ValueError:
            return None, None
        if "container" not in spec.runtimes or "windows_native" in spec.runtimes:
            return None, None
        pack_path = Path(getattr(self.project.model, "pack_path", "")).expanduser()
        if not pack_path.is_absolute() or pack_path.is_symlink() or not pack_path.is_dir():
            return spec, None
        return spec, pack_path.resolve()

    def _update_pack_controls(self):
        """Show Docker pack actions only for an installed container-only model."""
        controls = getattr(self, "model_pack_train_button", None)
        if controls is None:
            return
        spec, pack_path = self._selected_container_spec()
        data_root = Path(getattr(getattr(self.project, "data", None), "root", "")).expanduser()
        data_ready = (data_root.is_absolute() and data_root.is_dir()
                      and not data_root.is_symlink())
        visible = (spec is not None and pack_path is not None and data_ready
                   and self._pack_job is None)
        for name in ("model_pack_train_button", "model_pack_infer_button", "model_pack_export_button"):
            getattr(self, name).setVisible(visible)
        if visible:
            self.model_pack_train_button.setToolTip(f"{spec.display_name} 팩의 학습 작업 실행\n{pack_path}")
            self.model_pack_infer_button.setToolTip(f"{spec.display_name} 팩의 추론 작업 실행\n{pack_path}")
            self.model_pack_export_button.setToolTip(f"{spec.display_name} 팩의 ONNX export 실행\n{pack_path}")

    def _start_model_pack_operation(self, operation: str):
        """Run a container model pack from the desktop training page."""
        if operation not in {"train", "infer", "export"} or self._pack_job is not None:
            return
        if self._run_project is not None or self._pack_job is not None or (self.worker and self.worker.isRunning()):
            QMessageBox.warning(self, "작업 진행 중", "현재 작업이 끝난 뒤 모델 팩 작업을 실행하세요.")
            return
        from core.job_manager import desktop_manager
        try:
            # Fail before saving or changing the page when an inference,
            # training, or dataset job already owns the process lock.
            desktop_manager().require_idle()
        except RuntimeError as exc:
            QMessageBox.warning(self, "작업 진행 중", str(exc))
            return
        spec, pack_path = self._selected_container_spec()
        if spec is None or pack_path is None:
            QMessageBox.warning(self, "모델 팩 필요", "서명된 컨테이너 모델 팩을 먼저 설치하고 선택하세요.")
            self._update_pack_controls()
            return
        data_dir = Path(getattr(self.project.data, "root", "")).expanduser()
        if not data_dir.is_absolute() or not data_dir.is_dir() or data_dir.is_symlink():
            QMessageBox.warning(self, "데이터 경로 확인", "모델 팩 작업에 사용할 데이터 루트가 없습니다.")
            return
        data_dir = data_dir.resolve()
        project_dir = Path(getattr(self.project, "project_dir", "")).expanduser()
        if not project_dir.is_absolute() or project_dir.is_symlink() or not project_dir.is_dir():
            project_dir = data_dir
        work_dir = (project_dir.resolve() / ".deepvision-model-pack-work")
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            ProjectManager.save(self.project)
            from core.desktop_jobs import DesktopJob
            request = {
                "task": self.project.task,
                "model": asdict(self.project.model),
                "training": asdict(self.project.training),
                "data": asdict(self.project.data),
                "project_path": ProjectManager.get_active_filepath(self.project),
            }
            payload = {
                "operation": operation,
                "pack_dir": str(pack_path),
                "data_dir": str(data_dir),
                "work_dir": str(work_dir),
                "request": request,
                "cpus": max(1, min(128, os.cpu_count() or 4)),
                "memory": "8g",
            }
            self._pack_error = ""
            self._pack_job = DesktopJob("pack_" + operation, payload, self)
            self._pack_job.event.connect(self._on_pack_event)
            self._pack_job.failed.connect(self._on_pack_failed)
            self._pack_job.completed.connect(self._on_pack_completed)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.settings_panel.setEnabled(False)
            self.progress_bar.setRange(0, 0)
            self.status_label.setText(f"{spec.display_name} 팩 {operation} 중...")
            self.run_identity_label.setText(f"팩 작업: {operation} · {pack_path}")
            self._on_log_message(f"모델 팩 작업 시작: {operation} / {spec.model_id}")
            self._pack_job.start()
        except Exception as exc:
            self._pack_job = None
            self.settings_panel.setEnabled(True)
            self.stop_btn.setEnabled(False)
            QMessageBox.critical(self, "팩 작업 시작 오류", str(exc))

    def _on_pack_event(self, event: str, args):
        if event == "log_message":
            self._on_log_message(str(args[0] if args else ""))
        elif event == "model_pack_prepared":
            self._on_log_message(f"팩 준비 완료: {args[0] if args else {}}")
        elif event == "model_pack_result":
            self._on_log_message(f"팩 결과 수신: {args[0] if args else {}}")

    def _on_pack_failed(self, message: str):
        self._pack_error = str(message)
        self._on_log_message(f"팩 작업 실패: {message}")

    def _on_pack_completed(self, job):
        """Release the desktop job and show its opaque pack result."""
        error = self._pack_error or str(job.get("error", ""))
        status = job.get("status", "failed")
        output = job.get("output") or {}
        current = self._pack_job
        self._pack_job = None
        if current is not None:
            current.deleteLater()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.settings_panel.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if status == "completed" and not error else 0)
        self.status_label.setText("팩 작업 완료" if status == "completed" and not error else "팩 작업 실패")
        self._update_pack_controls()
        if error or status != "completed":
            QMessageBox.critical(self, "팩 작업 실패", error or "모델 팩 작업이 실패했습니다.")
        else:
            QMessageBox.information(self, "팩 작업 완료", f"{output}")

    def _training_capabilities(self):
        if self.mode_combo.currentData() not in MODE_LABELS or (self.project and self.project.task == "obb"):
            return {"layer_debug": False, "layer_debug_reason": "지원하지 않는 모델 형식입니다.",
                    "augmentation": [], "default_layer_patterns": "features.0,features.1.*,classifier.1"}
        return training_capabilities(self.project.task if self.project else "classify",
                                     self.mode_combo.currentData(), self.anomaly_method_combo.currentData(),
                                     model_id=self.model_id_combo.currentData() or "")

    def _reset_unsupported_augmentation(self):
        if self.project:
            self.project.training.augmentation.vertical_flip = 0.0
            self.project.training.augmentation.mixup_alpha = 0.0
            self.unsupported_aug_reset.hide()

    def _default_debug_patterns(self):
        self.debug_patterns.setText(self._training_capabilities()["default_layer_patterns"])

    def _on_layer_debug(self, snapshot):
        from PySide6.QtWidgets import QTableWidgetItem
        from core.layer_debug_session import debug_rows
        rows = debug_rows(snapshot)
        self.debug_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.debug_table.setItem(row, column, QTableWidgetItem(value))
        self.debug_table.resizeColumnsToContents()
        self._on_log_message(
            f"레이어 관찰 기록: 에폭 {snapshot.get('epoch')}, 배치 {snapshot.get('batch')}, "
            f"{len(rows)}개 레이어. 설정한 초기 배치만 기록하며 이후에는 마지막 결과를 유지합니다.")
        self.debug_table.setToolTip(f"Run {snapshot.get('run_id', '')}, epoch {snapshot.get('epoch', '')}, batch {snapshot.get('batch', '')}. Gradient는 AMP 배율 제거 후 통계입니다.")

    def _on_efficientnet_model_changed(self, *_):
        if hasattr(self, "input_size_spin"):
            self.input_size_spin.setValue(240 if self.efficientnet_model_combo.currentData() == "efficientnet_b1" else 224)

    def _on_patchcore_settings_changed(self, *_):
        """이상 탐지 방법에 실제로 적용되는 설정만 활성화."""
        if not hasattr(self, "tl_group"):
            return
        is_anomaly = self.project is not None and self.project.task == "anomaly"
        is_pc = is_anomaly and self.anomaly_method_combo.currentData() == "patchcore"
        resume = not is_anomaly and self.mode_combo.currentData() in {"efficientnet_resume"}
        source = self.pc_source_combo.currentData()
        self.pc_backbone_combo.setEnabled(is_pc and source != "patchcore")
        self.pc_weights_edit.setEnabled(is_pc and source != "imagenet")
        self.pc_weights_browse.setEnabled(is_pc and source != "imagenet")
        self.pc_append_check.setEnabled(is_pc and source == "patchcore")
        self.pc_crop_check.setEnabled(not resume)
        cropped = self.pc_crop_check.isChecked()
        self.pc_crop_width_spin.setEnabled(cropped and not resume)
        self.pc_crop_height_spin.setEnabled(cropped and not resume)
        self.pc_crop_info.setVisible(True)
        self.pc_crop_info.setText(
            f"중앙 {self.pc_crop_width_spin.value()}×{self.pc_crop_height_spin.value()} px → 모델 입력 {self.input_size_spin.value()}×{self.input_size_spin.value()} px\n"
            "모든 태스크의 학습과 추론에 적용합니다. 박스와 마스크도 함께 자릅니다. 분류와 이상 탐지는 크롭 안의 내용에 맞는 이미지 라벨이 필요합니다."
            if cropped else "전체 이미지를 모델 입력 크기로 리사이즈합니다.")
        for control in (self.pc_source_combo, self.pc_sampling_spin, self.pc_neighbors_spin,
                        self.pc_candidates_spin, self.pc_bank_spin, self.pc_seed_spin):
            control.setEnabled(is_pc)
        for control in (self.epochs_spin, self.lr_spin, self.wd_spin, self.optimizer_combo,
                        self.scheduler_combo, self.patience_spin, self.hflip_spin,
                        self.rotation_spin, self.color_jitter_spin):
            control.setEnabled(not is_pc and not resume)
        for control in (self.batch_spin, self.input_size_spin, self.class_weight_combo):
            control.setEnabled(not resume)
        self.selection_combo.setEnabled(not is_pc and not resume)
        self._selection_changed()
        capabilities = self._training_capabilities()
        for control in (self.debug_check, self.debug_patterns, self.debug_batches, self.debug_default):
            control.setEnabled(capabilities["layer_debug"])
        self.debug_info.setText("선택한 초기 배치만 관찰한 뒤 전체 학습을 계속합니다. Gradient는 AMP 배율을 제거합니다."
                                if capabilities["layer_debug"] else capabilities["layer_debug_reason"])
        for key, control in (("horizontal_flip", self.hflip_spin), ("rotation", self.rotation_spin),
                             ("color_jitter", self.color_jitter_spin)):
            control.setEnabled(not resume and key in capabilities["augmentation"])
        self.amp_check.setEnabled(not is_pc and not resume and self._device_manager.cuda_available)
        self.tl_group.setVisible(not is_pc and (is_anomaly or self.mode_combo.currentData() == "custom"))

    def _update_selection_options(self):
        if not hasattr(self, "selection_combo") or not self.project:
            return
        from core.model_selection import available_metrics, LABELS
        if self.project.task == "obb" or self.mode_combo.currentData() not in MODE_LABELS:
            self.selection_combo.clear()
            self.selection_info.setText("현재 태스크 또는 저장된 학습 모드는 지원하지 않습니다.")
            return
        engine = training_engine_name(self.mode_combo.currentData())
        current = self.selection_combo.currentData() or "engine_default"
        self.selection_combo.blockSignals(True)
        self.selection_combo.clear()
        for metric in available_metrics(engine, self.project.task):
            self.selection_combo.addItem(LABELS[metric], metric)
        self.selection_combo.setCurrentIndex(max(0, self.selection_combo.findData(current)))
        self.selection_combo.blockSignals(False)
        self._selection_changed()

    def _selection_changed(self, *_):
        if not hasattr(self, "selection_info") or not self.project:
            return
        if self.project.task == "obb" or self.mode_combo.currentData() not in MODE_LABELS:
            self.selection_info.setText("현재 태스크 또는 저장된 학습 모드는 지원하지 않습니다.")
            return
        from core.model_selection import selection_policy
        from types import SimpleNamespace
        if self.project.task == "anomaly" and self.anomaly_method_combo.currentData() == "patchcore":
            self.selection_info.setText("PatchCore는 메모리 뱅크 1회 구축: 에폭별 Best 비교 없음")
            return
        engine = training_engine_name(self.mode_combo.currentData())
        policy = selection_policy(SimpleNamespace(selection_metric=self.selection_combo.currentData() or "engine_default"),
                                  engine, self.project.task)
        self.selection_info.setText(policy.describe() + "\nBest 저장과 조기 종료에 동일하게 적용")

    def _browse_patchcore_weights(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "PatchCore 전이학습 가중치 선택", "", "PyTorch 체크포인트 (*.pt *.pth)"
        )
        if filepath:
            self.pc_weights_edit.setText(filepath)

    def _browse_resume_weights(self):
        """이어학습용 이전 체크포인트 선택"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "이전 학습 모델 선택", "",
            "PyTorch 체크포인트 (*.pt *.pth)"
        )
        if filepath:
            self.resume_edit.setText(filepath)



    def set_project(self, project: ProjectData):
        """프로젝트 설정 → UI에 반영"""
        if self._run_project is not None:
            raise RuntimeError("학습 결과 반영 전 프로젝트 변경 불가")
        self.project = project
        if hasattr(self, "_set_model_options"):
            self._set_model_options(project)
        self.debug_table.setRowCount(0)
        cfg = project.training
        self.debug_check.setChecked(cfg.layer_debug_enabled)
        self.debug_patterns.setText(cfg.layer_debug_patterns)
        self.debug_batches.setValue(cfg.layer_debug_batches)

        # ── 학습 모드 복원 ──
        mode = "custom" if project.task == "anomaly" else getattr(cfg, "training_mode", "efficientnet_finetune")
        for i in range(self.mode_combo.count() - 1, -1, -1):
            if self.mode_combo.itemData(i) not in MODE_LABELS:
                self.mode_combo.removeItem(i)
        if mode not in MODE_LABELS:
            self.mode_combo.addItem("지원하지 않는 학습 모드 - 현재 엔진 선택 필요", mode)
            self.mode_combo.model().item(self.mode_combo.count() - 1).setEnabled(False)
        self.mode_combo.model().item(self.mode_combo.findData("custom")).setEnabled(project.task != "obb")
        for efficientnet_mode in ("efficientnet_finetune", "efficientnet_transfer", "efficientnet_resume"):
            self.mode_combo.model().item(self.mode_combo.findData(efficientnet_mode)).setEnabled(project.task == "classify")
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(mode))
        self.mode_combo.setEnabled(project.task != "anomaly")
        self._on_mode_changed(self.mode_combo.currentIndex())
        if hasattr(self, "_on_model_id_changed"):
            self._on_model_id_changed()

        self.efficientnet_model_combo.setCurrentIndex(max(0, self.efficientnet_model_combo.findData(cfg.efficientnet_model)))

        # ── 하이퍼파라미터 복원 ──
        self._update_selection_options()
        self.selection_combo.setCurrentIndex(max(0, self.selection_combo.findData(cfg.selection_metric)))
        self.epochs_spin.setValue(cfg.epochs)
        self.batch_spin.setValue(cfg.batch_size)
        self.lr_spin.setValue(cfg.learning_rate)
        self.wd_spin.setValue(cfg.weight_decay)
        self.input_size_spin.setValue(cfg.input_size)
        self.patience_spin.setValue(cfg.early_stop_patience)

        # 옵티마이저 매핑
        opt_map = {"adamw": 0, "sgd": 1, "adam": 2}
        self.optimizer_combo.setCurrentIndex(opt_map.get(cfg.optimizer, 0))

        sch_map = {"cosine": 0, "step": 1, "none": 2}
        self.scheduler_combo.setCurrentIndex(sch_map.get(cfg.scheduler, 0))

        # 클래스 가중치 설정 복원
        cw_mode = getattr(cfg, "class_weights", "none")
        for i in range(self.class_weight_combo.count()):
            if self.class_weight_combo.itemData(i) == cw_mode:
                self.class_weight_combo.setCurrentIndex(i)
                break

        # 데이터 증강 설정 복원
        aug = cfg.augmentation
        self.unsupported_aug_reset.setVisible(bool(aug.vertical_flip or aug.mixup_alpha))
        self.hflip_spin.setValue(aug.horizontal_flip)
        self.rotation_spin.setValue(aug.rotation)
        self.color_jitter_spin.setValue(aug.color_jitter)

        # 디바이스 설정 복원
        device_mode = getattr(cfg, "device", "auto")
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == device_mode:
                self.device_combo.setCurrentIndex(i)
                break

        self.amp_check.setChecked(getattr(cfg, "use_amp", True))

        # 트랜스퍼 러닝 설정 복원 (커스텀 모드용)
        mcfg = project.model
        self.weights_edit.setText(getattr(mcfg, "pretrained_weights", "") or "")
        freeze_val = getattr(mcfg, "freeze_backbone", False)
        self.freeze_check.setChecked(freeze_val)        # 커스텀 모드용
        self.finetune_freeze_check.setChecked(freeze_val)     # EfficientNet 모드용
        self.backbone_lr_spin.setValue(getattr(mcfg, "backbone_lr_mult", 0.1))

        # 이어학습 경로 복원
        self.resume_edit.setText(
            (getattr(mcfg, "pretrained_weights", "") or "")
            if mode in {"efficientnet_resume", "efficientnet_transfer"}
            else ""
        )

        # 태스크별 UI 갱신
        self._rebuild_metric_cards(project.task)
        self.metric_chart.set_task(project.task)

        # 클래스 가중치: classify의 두 학습 엔진 모두 지원
        # (segment/detect/anomaly는 클래스 가중치 미지원)
        is_classify = (project.task == "classify")
        self.class_weight_label.setVisible(is_classify)
        self.class_weight_combo.setVisible(is_classify)

        # ── PatchCore 설정 복원 (이상 탐지 전용) ──────────
        # anomaly 태스크일 때만 PatchCore 옵션 그룹 표시
        is_anomaly = (project.task == "anomaly")
        self.patchcore_group.setVisible(is_anomaly)

        if is_anomaly:
            # 이상 탐지 방법 복원 (patchcore / reconstruction)
            method = getattr(cfg, "anomaly_method", "patchcore")
            idx = 0 if method == "patchcore" else 1
            self.anomaly_method_combo.setCurrentIndex(idx)

            # 코어셋 비율 · kNN 이웃 수 복원
            self.pc_sampling_spin.setValue(
                getattr(cfg, "patchcore_sampling_ratio", 0.01)
            )
            self.pc_neighbors_spin.setValue(
                getattr(cfg, "patchcore_n_neighbors", 9)
            )
            for control, value in (
                (self.pc_backbone_combo, cfg.patchcore_backbone),
                (self.pc_source_combo, cfg.patchcore_weight_source),
            ):
                control.setCurrentIndex(max(0, control.findData(value)))
            self.pc_weights_edit.setText(cfg.patchcore_weights)
            self.pc_append_check.setChecked(cfg.patchcore_append)
            self.pc_candidates_spin.setValue(cfg.patchcore_max_candidates)
            self.pc_bank_spin.setValue(cfg.patchcore_max_memory_bank)
            self.pc_seed_spin.setValue(cfg.patchcore_seed)

        self.pc_crop_check.setChecked(cfg.patchcore_crop_enabled)
        self.pc_crop_width_spin.setValue(cfg.patchcore_crop_width)
        self.pc_crop_height_spin.setValue(cfg.patchcore_crop_height)

        self._on_patchcore_settings_changed()

        self.efficientnet_no_decay_check.setChecked(cfg.efficientnet_no_decay)

        # 기존 학습 기록이 있으면 비교 테이블 갱신
        if project.runs:
            self.compare_widget.update_comparison(project)
        else:
            self.compare_widget.clear()
        self.loss_chart.clear()
        self.metric_chart.clear()
        self.lr_chart.clear()
        self.cm_chart.clear()
        self.eval_widget.clear()
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("대기 중")
        self.eta_label.setText("ETA: —")
        self.best_selection_label.setText("Best 선정 결과: 대기")
        self._start_time = None
        self._progress_start_epoch = None
        self._progress_total_epochs = 0
        self.progress_bar.setRange(0, 100)
        self.run_identity_label.setText("선택된 학습 결과 없음")

    def collect_config(self):
        """저장과 클래스 변경에서 호출하는 공개 설정 동기화 진입점."""
        self._sync_config()

    def _browse_weights(self):
        """사전학습 가중치 파일 선택"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "가중치 파일 선택", "",
            "PyTorch 체크포인트 (*.pt *.pth)"
        )
        if filepath:
            self.weights_edit.setText(filepath)

    def _sync_config(self):
        """UI → 프로젝트 설정 동기화"""
        if self.project is None:
            return

        cfg = self.project.training
        cfg.epochs = self.epochs_spin.value()
        cfg.batch_size = self.batch_spin.value()
        cfg.learning_rate = self.lr_spin.value()
        cfg.weight_decay = self.wd_spin.value()
        cfg.input_size = self.input_size_spin.value()
        cfg.early_stop_patience = self.patience_spin.value()
        cfg.selection_metric = self.selection_combo.currentData() or "engine_default"

        opt_map = {0: "adamw", 1: "sgd", 2: "adam"}
        cfg.optimizer = opt_map.get(self.optimizer_combo.currentIndex(), "adamw")

        sch_map = {0: "cosine", 1: "step", 2: "none"}
        cfg.scheduler = sch_map.get(self.scheduler_combo.currentIndex(), "cosine")

        # ── 학습 모드 ──
        cfg.training_mode = "custom" if self.project.task == "anomaly" else self.mode_combo.currentData()
        cfg.efficientnet_model = self.efficientnet_model_combo.currentData() or "efficientnet_b0"
        cfg.efficientnet_no_decay = self.efficientnet_no_decay_check.isChecked()
        cfg.layer_debug_enabled = self.debug_check.isChecked() and self._training_capabilities()["layer_debug"]
        cfg.layer_debug_patterns = self.debug_patterns.text().strip()
        cfg.layer_debug_batches = self.debug_batches.value()

        # 트랜스퍼 러닝 설정 (모드에 따라 다른 소스)
        mcfg = self.project.model
        selected_model = self.model_id_combo.currentData() if hasattr(self, "model_id_combo") else None
        if selected_model:
            mcfg.model_id = selected_model
        if mcfg.model_id not in {"efficientnet_b0", "efficientnet_b1"} and cfg.training_mode.startswith("efficientnet"):
            cfg.training_mode = "custom"
        if cfg.training_mode in {"efficientnet_resume", "efficientnet_transfer"}:
            # 이어학습: resume_edit → pretrained_weights
            mcfg.pretrained_weights = self.resume_edit.text().strip()
        elif cfg.training_mode == "custom":
            # 커스텀: weights_edit → pretrained_weights
            mcfg.pretrained_weights = self.weights_edit.text().strip()
        elif cfg.training_mode == "efficientnet_finetune":
            # 파인튜닝: 프리트레인드 모델이 소스이므로 경로 불필요
            mcfg.pretrained_weights = ""

        # freeze_backbone: 모드에 따라 다른 체크박스에서 읽기
        # ┌──────────────────────┬──────────────────────────────┐
        # │ custom 모드          │ freeze_check (커스텀 TL)      │
        # └──────────────────────┴──────────────────────────────┘
        if cfg.training_mode in ("efficientnet_finetune", "efficientnet_transfer"):
            mcfg.freeze_backbone = self.finetune_freeze_check.isChecked()
        else:
            mcfg.freeze_backbone = self.freeze_check.isChecked()
        mcfg.backbone_lr_mult = self.backbone_lr_spin.value()

        # 디바이스 설정
        cfg.device = self.device_combo.currentData()
        cfg.use_amp = self.amp_check.isChecked()

        # 증강 설정
        cfg.class_weights = (self.class_weight_combo.currentData()
                             if self.project.task == "classify" else "none")
        cfg.augmentation.horizontal_flip = self.hflip_spin.value()
        cfg.augmentation.rotation = self.rotation_spin.value()
        cfg.augmentation.color_jitter = self.color_jitter_spin.value()

        # ── PatchCore 설정 동기화 (이상 탐지 전용) ──────
        if self.project and self.project.task == "anomaly":
            method_idx = self.anomaly_method_combo.currentIndex()
            cfg.anomaly_method = "patchcore" if method_idx == 0 else "reconstruction"
            cfg.patchcore_sampling_ratio = self.pc_sampling_spin.value()
            cfg.patchcore_n_neighbors = self.pc_neighbors_spin.value()
            cfg.patchcore_backbone = self.pc_backbone_combo.currentData()
            cfg.patchcore_weight_source = self.pc_source_combo.currentData()
            cfg.patchcore_weights = self.pc_weights_edit.text().strip()
            cfg.patchcore_append = (
                cfg.patchcore_weight_source == "patchcore" and self.pc_append_check.isChecked()
            )
            cfg.patchcore_max_candidates = self.pc_candidates_spin.value()
            cfg.patchcore_max_memory_bank = self.pc_bank_spin.value()
            cfg.patchcore_seed = self.pc_seed_spin.value()

        cfg.patchcore_crop_enabled = self.pc_crop_check.isChecked()
        cfg.patchcore_crop_width = self.pc_crop_width_spin.value()
        cfg.patchcore_crop_height = self.pc_crop_height_spin.value()

    def _start_training(self):
        """
        학습 시작 — 선택된 모드에 따라 적절한 워커 생성

        ┌──────────────────────────────────────────────────────────┐
        │ anomaly + patchcore  → PatchCoreWorker (메모리 뱅크)     │
        │ custom               → TrainWorker (기존)                │
        └──────────────────────────────────────────────────────────┘
        """
        if self._run_project is not None or (self.worker and self.worker.isRunning()):
            return
        from core.job_manager import desktop_manager
        try:
            desktop_manager().require_idle()
        except RuntimeError as exc:
            QMessageBox.warning(self, "작업 진행 중", str(exc))
            return
        dataset_page = getattr(self.window(), "dataset_page", None)
        class_worker = getattr(dataset_page, "_class_worker", None)
        if class_worker is not None:
            QMessageBox.warning(self, "데이터 변경 중", "데이터 변경 완료 후 학습 시작 가능")
            return
        if self.project is None:
            QMessageBox.warning(self, "알림", "프로젝트를 먼저 열어 주세요.")
            return

        if not self.project.data.root:
            QMessageBox.warning(self, "알림", "데이터 경로를 먼저 설정해 주세요.")
            return

        # UI → 설정 동기화
        self._sync_config()

        try:
            from core.training_modes import validate_training_options
            validate_training_options(self.project)
        except ValueError as error:
            QMessageBox.warning(self, "학습 설정 확인", str(error))
            return

        # ── 모드별 유효성 검사 ──
        mode = self.project.training.training_mode
        is_anomaly = self.project.task == "anomaly"
        is_patchcore = is_anomaly and self.project.training.anomaly_method == "patchcore"
        model_id = getattr(self.project.model, "model_id", "")
        if model_id:
            from core.model_registry import registry_with_installed_packs
            try:
                registry, _ = registry_with_installed_packs()
                spec = registry.get(model_id)
            except ValueError:
                spec = None
            if spec is not None and "container" in spec.runtimes and "windows_native" not in spec.runtimes:
                QMessageBox.warning(
                    self, "모델 팩 필요",
                    f"{spec.display_name}은 설치된 Docker 모델 팩 worker에서 학습해야 합니다.\n"
                    "모델 관리에서 .dvmodel 팩을 설치한 뒤 팩 작업으로 실행하세요."
                )
                return
        if is_patchcore and self.project.training.patchcore_weight_source != "imagenet":
            path = self.project.training.patchcore_weights
            if not path or not os.path.isfile(path):
                QMessageBox.warning(self, "가중치 파일 확인", f"전이학습 가중치 파일이 없습니다:\n{path}")
                return
        if not is_anomaly and mode in {"efficientnet_resume", "efficientnet_transfer"}:
            resume_path = self.project.model.pretrained_weights
            if not resume_path:
                QMessageBox.warning(
                    self, "알림",
                    "추가 학습 또는 중단 재개에 사용할 가중치(.pt) 선택 필요"
                )
                return
            if not os.path.isfile(resume_path):
                QMessageBox.warning(
                    self, "알림",
                    f"이전 모델 파일 없음:\n{resume_path}"
                )
                return

        if self.project.task == "obb":
            QMessageBox.warning(self, "OBB 학습", "OBB 학습 엔진은 제공하지 않습니다")
            return

        if mode.startswith("efficientnet") and self.project.task != "classify":
            QMessageBox.warning(self, "EfficientNet 학습", "EfficientNet은 분류 태스크만 지원")
            return

        # Prepare persistence before clearing results or disabling Start.
        # A read-only project or snapshot error must leave the page usable.
        try:
            from webapp.storage import digest
            ProjectManager.save(self.project)
            run_file_digest = digest(ProjectManager.get_active_filepath(self.project))
            run_snapshot = copy.deepcopy(self.project)
            run_config = {"task": self.project.task, "training": asdict(self.project.training),
                          "model": asdict(self.project.model), "data": asdict(self.project.data)}
            device = self._device_manager.get_device(self.device_combo.currentData())
            device_label = self._device_manager.get_device_label(device)
        except Exception as exc:
            QMessageBox.critical(self, "학습 시작 오류", f"학습 준비 실패:\n{exc}")
            return

        # 차트 + 메트릭 카드 초기화
        self._rebuild_metric_cards(self.project.task)
        self.metric_chart.metric_labels = _metric_info_for(self.project).get("labels", {})
        # ┌──────────────────────────────────────────────────┐
        # │ 이전 학습 결과가 남아 있으면 혼동을 줄 수 있으므로 │
        # │ 차트·카드·로그·프로그레스를 모두 초기 상태로 리셋  │
        # └──────────────────────────────────────────────────┘
        self.loss_chart.clear()
        self.metric_chart.clear()
        self.lr_chart.clear()
        self.cm_chart.clear()
        self.eval_widget.clear()
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.best_selection_label.setText("Best 선정 결과: 대기")

        # 메트릭 카드 값 초기화 (Best Accuracy, Best Epoch, Loss 등)
        for val_label in self.metric_cards.values():
            val_label.setText("—")
            val_label.setStyleSheet("")  # 초록색 강조 제거

        # 버튼 상태 변경
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.settings_panel.setEnabled(False)

        # 모드 표시
        mode_labels = {
            "efficientnet_finetune": "EfficientNet 사전학습",
            "efficientnet_transfer": "EfficientNet 추가 학습",
            "efficientnet_resume": "EfficientNet 학습 재개",
            "custom": "커스텀",
        }
        mode_label = mode_labels.get(mode, "학습")
        if is_patchcore:
            mode_label = "PatchCore 전이학습"
        self.status_label.setText(
            f"{mode_label} 중... ({device_label})"
        )
        self._start_time = None
        self._progress_start_epoch = None
        self._progress_total_epochs = 0
        self.progress_bar.setRange(0, 0)
        self.eta_label.setText("ETA: 준비 중")
        self.run_identity_label.setText("현재 작업: 모델 준비 중")

        # Loss 차트 탭으로 이동
        if self.project.training.layer_debug_enabled:
            self.chart_tabs.setCurrentWidget(self.debug_table)
            self._on_log_message(
                f"레이어 관찰 대기: 학습 시작 후 첫 {self.project.training.layer_debug_batches}개 배치를 기록합니다.")
        else:
            self.chart_tabs.setCurrentIndex(0)
        self.debug_table.setRowCount(0)

        # ── 모드에 따라 워커 생성 ──────────────────────
        # PatchCore는 일반 학습과 완전히 다른 파이프라인이므로
        # anomaly + patchcore 조합일 때 전용 워커를 사용
        self._run_file_digest = run_file_digest
        self._run_project = self.project
        self._run_snapshot = run_snapshot
        self._initial_run_count = len(self._run_snapshot.runs)
        self._run_config = run_config
        self._pending_result = None
        self._pending_error = None
        self._cancel_requested = False
        try:
            if is_patchcore:
                self.worker = PatchCoreWorker(self._run_snapshot)
            else:
                self.worker = TrainWorker(self._run_snapshot)
        except Exception as exc:
            self._pending_error = str(exc)
            self._on_worker_finished()
            return

        # 시그널 연결 (모든 워커가 동일한 인터페이스 사용)
        self.worker.signals.epoch_finished.connect(self._on_epoch_finished)
        self.worker.signals.best_epoch_updated.connect(self._on_best_epoch_updated)
        self.worker.signals.batch_finished.connect(self._on_batch_finished)
        self.worker.signals.lr_updated.connect(self._on_lr_updated)
        self.worker.signals.eval_finished.connect(self._on_eval_finished)
        self.worker.signals.layer_debug.connect(self._on_layer_debug)
        self.worker.signals.training_finished.connect(self._on_training_finished)
        self.worker.signals.training_error.connect(self._on_training_error)
        self.worker.signals.log_message.connect(self._on_log_message)
        self.worker.signals.progress_updated.connect(self._on_progress)
        self.worker.signals.early_stopped.connect(self._on_early_stop)
        self.worker.finished.connect(self._on_worker_finished)
        try:
            self.worker.start()
        except Exception as exc:
            self._pending_error = str(exc)
            self._on_worker_finished()

    def _stop_training(self):
        """학습 중지"""
        if self._pack_job is not None:
            self._pack_job.stop()
            self.status_label.setText("팩 작업 중지 요청 중...")
            self.stop_btn.setEnabled(False)
            return
        if self.worker:
            self._cancel_requested = True
            self.worker.stop()
            self.status_label.setText("중지 요청 중...")
            self.stop_btn.setEnabled(False)

    # ── 시그널 핸들러 ─────────────────────────────

    def _on_epoch_finished(self, epoch, train_loss, val_loss, metrics):
        """각 에폭의 손실과 메트릭 추이를 차트에 기록."""
        # Loss 차트
        self.loss_chart.update_chart(epoch, train_loss, val_loss)

        # Metric 차트
        self.metric_chart.update_metrics(epoch, metrics)

    def _on_best_epoch_updated(self, epoch, train_loss, val_loss, metrics):
        """저장된 베스트 모델의 에폭, 손실, 주요 지표를 함께 표시."""
        if "Best Epoch" in self.metric_cards:
            self.metric_cards["Best Epoch"].setText(str(epoch))
        if "Train Loss" in self.metric_cards:
            self.metric_cards["Train Loss"].setText(_format_metric(train_loss))
        if "Val Loss" in self.metric_cards:
            self.metric_cards["Val Loss"].setText(_format_metric(val_loss))

        project = self._run_snapshot or self.project
        if project and hasattr(self, "best_selection_label"):
            from core.model_selection import selection_policy
            engine = training_engine_name(project.training.training_mode)
            if project.task == "anomaly" and project.training.anomaly_method == "patchcore":
                self.best_selection_label.setText("PatchCore 메모리 뱅크 구축 완료")
            else:
                policy = selection_policy(project.training, engine, project.task)
                self.best_selection_label.setText(
                    f"Best epoch {epoch} | {policy.describe()} | 값 {_format_metric(metrics.get(policy.metric))}")
        if project:
            metric_info = _metric_info_for(project)
            primary = metric_info.get("primary", "")
            primary_label = metric_info.get("labels", {}).get(primary, "Best Metric")
            card_name = f"Best {primary_label}"

            if card_name in self.metric_cards:
                current = metrics.get(primary)
                self.metric_cards[card_name].setText(_format_metric(current))
                self.metric_cards[card_name].setStyleSheet(
                    "color: #34C759;" if _finite_metric(current) else "")

    def _on_batch_finished(self, epoch, batch_idx, total_batches, loss):
        """배치 완료"""
        total_epochs = self._progress_total_epochs
        if total_epochs <= 0 or total_batches <= 0:
            return
        completed = epoch - 1 + min(1, max(0, batch_idx / total_batches))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(min(99, max(0, int(completed / total_epochs * 100))))
        if not self._cancel_requested:
            self.status_label.setText(f"Epoch {epoch}/{total_epochs} | Batch {batch_idx}/{total_batches}")

    def _on_lr_updated(self, epoch, lr):
        """Learning Rate 업데이트 → LR 차트 갱신"""
        self.lr_chart.update_lr(epoch, lr)

    def _on_eval_finished(self, eval_results):
        """
        종합 평가 완료 → 혼동 행렬 + 평가 테이블 업데이트

        학습 완료 후 호출되어 상세 지표를 표시
        """
        # 혼동 행렬 표시 (Classification / Anomaly)
        if "confusion_matrix" in eval_results:
            cm = eval_results["confusion_matrix"]
            if isinstance(cm, list):
                cm = np.array(cm)

            task = eval_results.get("task", "classify")
            if task == "classify":
                class_names = (
                    (self._run_snapshot or self.project).data.class_names if self.project else None
                )
            elif task == "anomaly":
                class_names = ["Normal", "Anomaly"]
            else:
                class_names = None

            if class_names:
                self.cm_chart.update_matrix(cm, class_names)

        # 평가 결과 테이블
        self.eval_widget.update_results(eval_results)

        # Confusion Matrix 탭으로 전환
        if "confusion_matrix" in eval_results:
            self.chart_tabs.setCurrentIndex(3)  # CM 탭

    def _on_progress(self, current_epoch, total_epochs):
        """진행률 업데이트"""
        now = time.monotonic()
        if self._progress_start_epoch is None:
            self._progress_start_epoch = current_epoch
            self._start_time = now
        self._progress_total_epochs = total_epochs
        self.progress_bar.setRange(0, 100)
        pct = int(current_epoch / max(total_epochs, 1) * 100)
        self.progress_bar.setValue(min(99, max(0, pct)))

        # ETA 계산
        remaining = remaining_seconds(now - self._start_time, current_epoch,
                                      total_epochs, self._progress_start_epoch)
        if remaining is not None:
            mins, secs = divmod(int(remaining), 60)
            self.eta_label.setText(f"ETA: {mins}m {secs}s")

        else:
            self.eta_label.setText("ETA: 첫 에폭 측정 중")
        if not self._cancel_requested:
            self.status_label.setText("최종 평가 및 저장 중" if current_epoch >= total_epochs
                                      else f"Epoch {current_epoch}/{total_epochs}")

    def _on_training_finished(self, best_metric, best_epoch, ckpt_path):
        """결과는 보관하고 실제 QThread 종료 후 프로젝트에 반영."""
        self._pending_result = (best_metric, best_epoch, ckpt_path)

    def _on_training_error(self, error_msg):
        self._pending_error = error_msg
        self.log_text.append(f"\n{error_msg}")

    def _on_worker_finished(self):
        """성공, 실패, 취소 모두 동일한 경로로 실행 상태 정리."""
        original = self._run_project
        snapshot = self._run_snapshot
        if original is None:
            return
        new_runs = snapshot.runs[self._initial_run_count:] if snapshot else []
        for run in new_runs:
            saved_run = copy.deepcopy(run)
            # Worker output is authoritative, including restored resume settings.
            saved_run.config_snapshot.setdefault("requested_config", copy.deepcopy(self._run_config))
            original.runs.append(saved_run)
        if snapshot is not None:
            original.data.class_names = list(snapshot.data.class_names)
            original.data.num_classes = snapshot.data.num_classes
        result = self._pending_result
        error = self._pending_error
        display_error = None
        status = "failed" if error else new_runs[-1].status if new_runs else (
            "cancelled" if self._cancel_requested else "failed")
        if new_runs:
            self.run_identity_label.setText(run_description(new_runs[-1]))
            try:
                self._display_finished_run(snapshot, new_runs[-1])
            except Exception as exc:
                # A rendering failure must not prevent persistence or unlock.
                display_error = str(exc)
                self.log_text.append(f"결과 화면 갱신 실패: {exc}")
        else:
            self.run_identity_label.setText("현재 작업의 저장된 학습 결과 없음")
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None
        self._run_project = None
        self._run_snapshot = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.settings_panel.setEnabled(True)
        self._start_time = None
        self._progress_start_epoch = None
        self.progress_bar.setRange(0, 100)
        self.eta_label.setText("ETA: —")
        self.status_label.setText({"completed": "학습 완료", "cancelled": "학습 중단",
                                   "failed": "학습 실패"}.get(status, "학습 종료"))
        try:
            self.compare_widget.update_comparison(original)
        except Exception as exc:
            display_error = str(exc)
            self.log_text.append(f"비교 화면 갱신 실패: {exc}")
        try:
            from webapp.storage import digest
            if isinstance(getattr(self, "_run_file_digest", None), str) and digest(ProjectManager.get_active_filepath(original)) != self._run_file_digest:
                raise RuntimeError("외부 프로젝트 변경 감지. 작업 기록의 project_result.json에 결과 보존. 프로젝트 다시 열기 필요")
            self.collect_config()
            ProjectManager.save(original)
        except Exception as exc:
            self.status_label.setText("결과 저장 실패")
            QMessageBox.critical(self, "결과 저장 오류", f"실행 결과 저장 실패:\n{exc}")
            return
        if status == "failed" and error:
            QMessageBox.critical(self, "학습 오류", error)
        elif display_error:
            self.status_label.setText("결과 저장됨 / 화면 갱신 실패")
            QMessageBox.warning(self, "결과 표시 오류", f"학습 기록은 저장되었습니다.\n{display_error}")
        elif status == "completed" and result:
            best_metric, best_epoch, ckpt_path = result
            metric_name = new_runs[-1].best_metric_name if new_runs else "metric"
            metric_text = "평가 불가" if metric_name == "unavailable" else _format_metric(best_metric)
            self.progress_bar.setValue(100)
            epoch_text = str(best_epoch) if best_epoch > 0 else "N/A"
            if "Best Epoch" in self.metric_cards:
                self.metric_cards["Best Epoch"].setText(epoch_text)
            QMessageBox.information(self, "학습 완료",
                f"학습 완료\n\n{metric_name}: {metric_text}\n"
                f"Best Epoch: {epoch_text}\n체크포인트: {ckpt_path}")

    def _display_finished_run(self, snapshot, run):
        """Render the completed record with the worker's class order and settings."""
        self._rebuild_metric_cards(snapshot.task, project=snapshot)
        self.metric_chart.metric_labels = _metric_info_for(snapshot).get("labels", {})
        if run.config_snapshot.get("layer_debug", {}).get("latest"):
            self._on_layer_debug(run.config_snapshot["layer_debug"]["latest"])
        history = run.metrics_history
        epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))
        self.loss_chart.clear()
        self.metric_chart.clear()
        self.lr_chart.clear()
        display = _metric_info_for(snapshot).get("display", [])
        for index, epoch in enumerate(epochs):
            values = {key: sequence[index] if index < len(sequence) else None
                      for key, sequence in history.items() if key != "epoch"}
            redraw = index == len(epochs) - 1
            self.loss_chart.update_chart(epoch, values.get("train_loss"), values.get("val_loss"), redraw=redraw)
            self.metric_chart.update_metrics(epoch, {key: values.get(key) for key in display}, redraw=redraw)
            if index < len(run.lr_history):
                self.lr_chart.update_lr(epoch, run.lr_history[index], redraw=index == min(len(epochs), len(run.lr_history)) - 1)
        best_index = epochs.index(run.best_epoch) if run.best_epoch in epochs else -1
        def best_loss(name):
            values = history.get(name, [])
            return values[best_index] if 0 <= best_index < len(values) else None
        if run.best_metric_name and (run.best_epoch > 0 or run.eval_results):
            self._on_best_epoch_updated(run.best_epoch, best_loss("train_loss"), best_loss("val_loss"),
                                        {run.best_metric_name: run.best_metric})
        self.cm_chart.clear()
        self.eval_widget.clear()
        if run.eval_results:
            self._on_eval_finished(run.eval_results)

    def _on_log_message(self, message):
        """로그 메시지"""
        self.log_text.append(message)
        # 자동 스크롤
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    def _on_early_stop(self, epoch, best_metric):
        """조기 종료"""
        self.status_label.setText(f"조기 종료 (epoch {epoch})")

from widgets.training_results import LossChart as LossChart, MetricChart as MetricChart, LRChart as LRChart, ConfusionMatrixChart as ConfusionMatrixChart, EvalResultsWidget as EvalResultsWidget, ModelCompareWidget as ModelCompareWidget
