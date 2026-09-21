"""
Deep Vision Studio — 모델 내보내기 페이지

기능:
┌─────────────────────────────────────────────────────────────┐
│ 1. PyTorch (.pt) → ONNX (.onnx) 변환                       │
│ 2. ONNX 옵션: opset 버전, 동적 배치, 단순화                  │
│ 3. 변환 결과 검증 (PyTorch vs ONNX 출력 비교)                │
│ 4. C++ 추론용 inference_config.json 생성                     │
│ 5. ONNX 모델 정보 표시 (크기, 입출력 형태)                   │
└─────────────────────────────────────────────────────────────┘
"""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, QGroupBox,
    QFormLayout, QSpinBox,
    QCheckBox, QTextEdit, QProgressBar,
)
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont, QTextCursor


# 프로젝트 모듈 경로 추가 (PyInstaller EXE 호환)
from core.paths import ensure_python_path
ensure_python_path()

from export_onnx import export_checkpoint
from core.project import ProjectData


class ExportWorker(QThread):
    """로드부터 검증까지 동일한 내보내기 서비스를 작업 스레드에서 실행한다."""

    log = Signal(str)
    completed = Signal(dict)
    error = Signal(str)

    def __init__(self, checkpoint_path, output_path, opset_version,
                 dynamic_batch, simplify, verify, parent=None, *, sam2_model_id=""):
        super().__init__(parent)
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        self.options = dict(opset_version=opset_version, dynamic_batch=dynamic_batch,
                            simplify=simplify, verify=verify)
        self.sam2_model_id = sam2_model_id

    def run(self):
        try:
            if self.sam2_model_id:
                from export_sam2_onnx import export_official_sam2_checkpoint
                # SAM2 deployment contains encoder+decoder graphs, so use the
                # chosen ONNX file's directory as the bundle directory.
                if self.options["dynamic_batch"] or self.options["simplify"]:
                    self.log.emit("SAM2는 고정 1-image encoder와 dynamic prompt point 계약을 사용합니다. 선택한 동적 배치/단순화 옵션은 적용하지 않습니다.")
                result = export_official_sam2_checkpoint(
                    self.sam2_model_id, self.checkpoint_path,
                    os.path.dirname(os.path.abspath(self.output_path)),
                    verify=self.options["verify"], opset=self.options["opset_version"],
                    log=self.log.emit)
            else:
                result = export_checkpoint(self.checkpoint_path, self.output_path,
                                           log=self.log.emit, **self.options)
            self.completed.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class ExportWidget(QWidget):
    """모델 내보내기 페이지"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)

        # ── 타이틀 ──
        title = QLabel("모델 내보내기 (ONNX)")
        title.setObjectName("page_title")
        layout.addWidget(title)

        subtitle = QLabel(
            "학습된 PyTorch 모델을 ONNX 형식으로 변환할 수 있습니다. "
            "변환된 모델은 지원하는 ONNX Runtime 추론기에서 바로 추론에 사용할 수 있습니다."
        )
        subtitle.setObjectName("page_subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # ── 입력 설정 ──
        input_group = QGroupBox("입력 설정")
        input_layout = QFormLayout(input_group)

        # 체크포인트 경로
        ckpt_row = QHBoxLayout()
        self.ckpt_edit = QLineEdit()
        self.ckpt_edit.setPlaceholderText("학습된 체크포인트 (.pt)")
        ckpt_row.addWidget(self.ckpt_edit)
        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(self._browse_checkpoint)
        ckpt_row.addWidget(browse_btn)
        input_layout.addRow("체크포인트:", ckpt_row)

        # 출력 경로
        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("출력 ONNX 파일 경로")
        out_row.addWidget(self.output_edit)
        out_browse = QPushButton("찾아보기...")
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(out_browse)
        input_layout.addRow("출력 경로:", out_row)

        layout.addWidget(input_group)

        # ── ONNX 옵션 ──
        opt_group = QGroupBox("ONNX 옵션")
        opt_layout = QFormLayout(opt_group)

        self.opset_spin = QSpinBox()
        self.opset_spin.setRange(11, 20)
        self.opset_spin.setValue(17)
        self.opset_spin.setFixedWidth(120)
        # QFormLayout 은 필드 열을 가득 채우므로 HBox + stretch 로 폭을 고정한다
        opset_row = QHBoxLayout()
        opset_row.addWidget(self.opset_spin)
        opset_row.addStretch(1)
        opt_layout.addRow("Opset 버전:", opset_row)

        self.dynamic_check = QCheckBox("동적 배치 크기 지원")
        opt_layout.addRow("", self.dynamic_check)

        self.simplify_check = QCheckBox("ONNX 단순화 (onnxsim)")
        opt_layout.addRow("", self.simplify_check)

        self.verify_check = QCheckBox("출력 검증 (PyTorch vs ONNX)")
        self.verify_check.setChecked(True)
        opt_layout.addRow("", self.verify_check)

        layout.addWidget(opt_group)

        # ── 내보내기 버튼 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.export_btn = QPushButton("ONNX Export")
        self.export_btn.setProperty("cssClass", "primary")
        self.export_btn.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.export_btn.setFixedSize(240, 48)
        self.export_btn.clicked.connect(self._start_export)
        btn_row.addWidget(self.export_btn)

        layout.addLayout(btn_row)

        # ── 프로그레스 ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 무한 진행
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # ── 로그 ──
        log_group = QGroupBox("변환 로그")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(250)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        layout.addStretch()

    def set_project(self, project: ProjectData):
        """프로젝트 설정"""
        self.project = project
        # A previous project's checkpoint must never be exported under this
        # project's output directory.  External checkpoints require an
        # explicit browse action after this reset.
        self.ckpt_edit.clear()
        self.log_text.clear()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._auto_checkpoint = ""

        # 최근 체크포인트 자동 설정
        if project.runs:
            latest = project.runs[-1]
            if latest.checkpoint_path:
                self.ckpt_edit.setText(latest.checkpoint_path)
                self._auto_checkpoint = os.path.abspath(latest.checkpoint_path)
        # A fresh SAM2 project exports the bundled official checkpoint.  A
        # SAM2 training run has its own compatible fine-tune checkpoint and
        # must remain the source selected above.
        if (project.task == "segment" and
                str(getattr(project.model, "model_id", "")).startswith("sam2_hiera_") and
                not self.ckpt_edit.text().strip()):
            from builtin_assets import builtin_asset_path
            self.ckpt_edit.setText(str(builtin_asset_path(project.model.model_id)))
            self._auto_checkpoint = self.ckpt_edit.text()

        # 기본 출력 경로
        export_dir = os.path.join(project.project_dir, "exports")
        os.makedirs(export_dir, exist_ok=True)
        self.output_edit.setText(
            os.path.join(export_dir, f"model_{project.task}.onnx")
        )

    def _browse_checkpoint(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "체크포인트 선택", "",
            "PyTorch 체크포인트 (*.pt *.pth)"
        )
        if filepath:
            self.ckpt_edit.setText(filepath)

    def _browse_output(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "ONNX 저장", "",
            "ONNX 모델 (*.onnx)"
        )
        if filepath:
            self.output_edit.setText(filepath)

    def _start_export(self):
        """내보내기 시작"""
        ckpt_path = self.ckpt_edit.text().strip()
        output_path = self.output_edit.text().strip()

        # The selection can change on the Training page after this page was
        # initially bound to the project.  A blank SAM2 source resolves to the
        # bundled official checkpoint; a selected SAM2 fine-tune result stays
        # selected so its learned prompt/mask-decoder weights are exported.
        if (self.project is not None and self.project.task == "segment" and
                str(getattr(self.project.model, "model_id", "")).startswith("sam2_hiera_") and
                not ckpt_path):
            from builtin_assets import builtin_asset_path
            ckpt_path = str(builtin_asset_path(self.project.model.model_id))
            self.ckpt_edit.setText(ckpt_path)

        if not ckpt_path or not os.path.isfile(ckpt_path):
            QMessageBox.warning(self, "알림", "체크포인트 파일을 선택해 주세요.")
            return

        if not output_path:
            QMessageBox.warning(self, "알림", "출력 경로를 지정해 주세요.")
            return

        try:
            self._validate_checkpoint_for_project(ckpt_path)
        except ValueError as exc:
            QMessageBox.warning(self, "체크포인트 확인", str(exc))
            return

        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            return
        self.log_text.clear()
        self.export_btn.setEnabled(False)
        self.progress_bar.show()
        self.worker = ExportWorker(
            checkpoint_path=ckpt_path, output_path=output_path,
            opset_version=self.opset_spin.value(),
            dynamic_batch=self.dynamic_check.isChecked(),
            simplify=self.simplify_check.isChecked(),
            verify=self.verify_check.isChecked(), parent=self,
            sam2_model_id=(getattr(self.project.model, "model_id", "")
                           if self.project is not None and self.project.task == "segment" and
                           str(getattr(self.project.model, "model_id", "")).startswith("sam2_hiera_") else ""),
        )
        self.worker.log.connect(self._on_log)
        self.worker.completed.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _validate_checkpoint_for_project(self, checkpoint_path):
        """Reject a checkpoint whose declared task/model contradicts the project.

        A user may explicitly browse a compatible external checkpoint, but a
        stale checkpoint from a different project must not silently become a
        new project's deployment artifact.
        """
        if self.project is None:
            return
        import torch
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(f"체크포인트를 읽을 수 없습니다: {exc}") from exc
        if not isinstance(checkpoint, dict):
            raise ValueError("체크포인트 메타데이터를 확인할 수 없습니다.")
        task = checkpoint.get("task")
        if task and task != self.project.task:
            raise ValueError(
                f"현재 프로젝트 태스크는 {self.project.task}이지만 체크포인트는 {task}입니다. "
                "같은 프로젝트의 학습 결과를 선택하거나 호환되는 프로젝트를 여세요.")
        selected = getattr(self.project.model, "model_id", "")
        model_config = checkpoint.get("model_config") or {}
        checkpoint_model = checkpoint.get("model_id") or model_config.get("model_id")
        if selected and checkpoint_model and selected != checkpoint_model:
            raise ValueError(
                f"현재 선택 모델은 {selected}이지만 체크포인트 모델은 {checkpoint_model}입니다.")
        architecture = model_config.get("architecture")
        if selected.startswith("efficientnet_") and architecture and selected != architecture:
            raise ValueError(
                f"현재 선택 EfficientNet은 {selected}이지만 체크포인트 구조는 {architecture}입니다.")

    def _worker_finished(self):
        self.worker = None
        self.export_btn.setEnabled(True)
        self.progress_bar.hide()

    def _on_log(self, message):
        self.log_text.append(message)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    def _on_finished(self, result):
        status = "검증 통과" if result["verification"] == "passed" else "검증 건너뜀"
        text = (f"ONNX 내보내기 완료 ({status})\n\n"
                f"모델: {result['output_path']}\n"
                f"설정: {result['config_path']}\n"
                f"크기: {result['file_size_mb']:.1f} MB")
        if result.get("backend") == "sam2":
            text += "\nSAM2는 encoder ONNX, decoder ONNX, sam2.json 세 파일을 함께 배포합니다."
        if result.get("cpp_supported"):
            text += "\n제공된 C++ 추론기에서 ONNX와 JSON을 함께 로드할 수 있습니다."
        if result.get("runtime_settings"):
            settings = result["runtime_settings"]
            level = {"all": "전체", "basic": "기본", "disabled": "끔"}[settings["graph_optimization_level"]]
            text += f"\n검증된 실행 설정: ONNX Runtime 최적화 {level}, CPU {settings['num_threads']} 스레드"
        optimization = result.get("optimization", {})
        if optimization.get("fallback") == "mixed_precision":
            text += f"\n고정밀 계산 구간: {optimization['fp64_stages']} (나머지는 FP32)"
        elif optimization.get("fallback") == "portable_fp64":
            text += "\n전체 고정밀 계산을 유지합니다. 추론이 느릴 수 있습니다."
        timing = optimization.get("latency_measurement")
        if timing:
            text += (f"\n현재 PC 모델 추론 중앙값: {timing['baseline_median_ms']:.2f} → "
                     f"{timing['selected_median_ms']:.2f} ms (전·후처리 제외)")
        self.log_text.append(text)
        QMessageBox.information(self, "완료", text)

    def _on_error(self, error_msg):
        self.log_text.append(f"\n오류: {error_msg}")
        QMessageBox.critical(self, "오류", f"내보내기 실패:\n{error_msg}")
