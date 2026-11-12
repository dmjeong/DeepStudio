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
    QFormLayout, QSpinBox, QComboBox,
    QCheckBox, QTextEdit, QProgressBar,
)
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont, QTextCursor


# 프로젝트 모듈 경로 추가 (PyInstaller EXE 호환)
from core.paths import ensure_python_path
ensure_python_path()

from export_onnx import deployment_preview, export_checkpoint
from core.project import ProjectData


class ExportWorker(QThread):
    """로드부터 검증까지 동일한 내보내기 서비스를 작업 스레드에서 실행한다."""

    log = Signal(str)
    completed = Signal(dict)
    error = Signal(str)

    def __init__(self, checkpoint_path, output_path, opset_version,
                 dynamic_batch, simplify, verify, parent=None, *, sam2_model_id="",
                 validation_dir=None, allow_precision_fallback=False, encryption_key_path=None, overrides=None,
                 merge_with="", precision="float32", calibration_dir=None):
        super().__init__(parent)
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        self.options = dict(opset_version=opset_version, dynamic_batch=dynamic_batch,
                            precision=precision, calibration_dir=calibration_dir,
                            simplify=simplify, verify=verify, validation_dir=validation_dir,
                            allow_precision_fallback=allow_precision_fallback,
                            encryption_key_path=encryption_key_path, overrides=overrides)
        self.sam2_model_id = sam2_model_id
        self.merge_with = merge_with  # 표면 결함 모델: 이 체크포인트(칩 상태)와 합쳐 ONNX 하나로

    def run(self):
        try:
            if self.merge_with:
                from merge_segmenters import export_merged
                region = (self.options.get("overrides") or {}).get("region_size")
                result = export_merged(self.checkpoint_path, self.merge_with, self.output_path, region=region,
                                       opset_version=self.options["opset_version"], verify=self.options["verify"],
                                       log=self.log.emit)
                result.update(verification="passed" if self.options["verify"] else "skipped", cpp_supported=True,
                              file_size_mb=os.path.getsize(result["output_path"]) / (1024 * 1024), merged=True)
                self.completed.emit(result)
                return
            result = export_checkpoint(self.checkpoint_path, self.output_path,
                                       log=self.log.emit, sam2_model_id=self.sam2_model_id,
                                       **self.options)
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

        # C++ 배포 안내: 패치 학습 모델의 판정 영역(고정 입력)과 함께 담길 판정 기준
        self.region_row = QWidget()
        region_layout = QHBoxLayout(self.region_row)
        region_layout.setContentsMargins(0, 0, 0, 0)
        self.region_w, self.region_h = QSpinBox(), QSpinBox()
        for spin, name in ((self.region_w, "가로"), (self.region_h, "세로")):
            spin.setRange(0, 65536)
            spin.setSpecialValueText("입력 필요")  # 0 = 아직 정하지 않음 (예전 체크포인트)
            spin.setSuffix(" px")
            spin.setAccessibleName(f"판정 영역 {name}")
        region_layout.addWidget(self.region_w)
        region_layout.addWidget(QLabel("×"))
        region_layout.addWidget(self.region_h)
        region_layout.addStretch(1)
        self.region_row.setToolTip("C++ SDK에 넣을 영상(판정 영역) 크기. 이 크기 전체를 한 번에 추론하는 고정 입력으로 내보냅니다.\n"
                                   "기본값은 학습 영상 크기입니다. 설비에서 넘기는 영상 크기와 같아야 합니다.")
        input_layout.addRow("판정 영역:", self.region_row)
        # 두 분할 모델 합치기: 칩 상태(위 체크포인트) + 표면 결함(우선) → ONNX 하나, 설비는 Segment 한 번
        self.merge_check = QCheckBox("표면 결함 모델과 합쳐 파일 하나로")
        self.merge_check.setToolTip("위 체크포인트(예: 칩 상태 shift·double·flip)와 표면 결함 모델(crack·FM)을 ONNX 하나로 합칩니다.\n"
                                    "표면 결함이 있는 픽셀은 표면 모델, 나머지는 위 모델의 판정을 씁니다. 두 모델의 판정 기준도 함께 담깁니다.\n"
                                    "두 모델 모두 패치 학습(배율 100%·50%·25%) 모델이어야 합니다.")
        merge_row = QWidget()
        merge_layout = QHBoxLayout(merge_row)
        merge_layout.setContentsMargins(0, 0, 0, 0)
        self.merge_edit = QLineEdit()
        self.merge_edit.setPlaceholderText("표면 결함 모델 체크포인트 (.pt) — 결함이 겹치면 이쪽이 우선")
        self.merge_edit.setEnabled(False)
        merge_browse = QPushButton("찾아보기...")
        merge_browse.setEnabled(False)
        merge_browse.clicked.connect(self._browse_merge)
        self.merge_check.toggled.connect(self.merge_edit.setEnabled)
        self.merge_check.toggled.connect(merge_browse.setEnabled)
        merge_layout.addWidget(self.merge_check)
        merge_layout.addWidget(self.merge_edit, 1)
        merge_layout.addWidget(merge_browse)
        input_layout.addRow("모델 합치기:", merge_row)
        self.deploy_info = QLabel()
        self.deploy_info.setWordWrap(True)
        self.deploy_info.setStyleSheet("font-size: 11px;")
        input_layout.addRow("C++ 배포:", self.deploy_info)
        self._deploy_preview = {}
        self._show_deploy_preview()
        self.ckpt_edit.editingFinished.connect(self._refresh_deploy_preview)
        self.ckpt_edit.textChanged.connect(lambda _text: self._preview_timer.start())
        from PySide6.QtCore import QTimer
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._refresh_deploy_preview)

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

        self.export_precision = QComboBox()
        self.export_precision.addItem("FP32 (원본 정밀도)", "float32")
        self.export_precision.addItem("INT8 QDQ (분할·Dinomaly, 보정 필요)", "int8")
        opt_layout.addRow("정밀도:", self.export_precision)
        self.calibration_edit = QLineEdit()
        self.calibration_edit.setPlaceholderText("INT8 보정용 학습 이미지 폴더 (정상·불량 혼합)")
        calibration_row = QHBoxLayout()
        calibration_row.addWidget(self.calibration_edit)
        calibration_browse = QPushButton("찾아보기...")
        calibration_browse.clicked.connect(self._browse_calibration)
        calibration_row.addWidget(calibration_browse)
        opt_layout.addRow("INT8 보정 이미지:", calibration_row)
        self.calibration_edit.setToolTip("INT8은 FP32와 값이 달라질 수 있습니다. 출력 차이 리포트를 저장하며 검증/시험 데이터의 실제 판정도 비교하세요.")

        self.dynamic_check = QCheckBox("동적 배치 크기 지원")
        opt_layout.addRow("", self.dynamic_check)

        self.simplify_check = QCheckBox("ONNX 단순화 (onnxsim)")
        opt_layout.addRow("", self.simplify_check)

        self.verify_check = QCheckBox("출력 검증 (PyTorch vs ONNX)")
        self.verify_check.setChecked(True)
        opt_layout.addRow("", self.verify_check)

        self.dataset_verify_check = QCheckBox("실제 이미지 추가 검증 (PyTorch·ONNX 판정 일치·확률 차이 0.1%p 이하)")
        self.dataset_verify_check.setChecked(False)
        opt_layout.addRow("", self.dataset_verify_check)
        self.validation_edit = QLineEdit()
        self.validation_edit.setPlaceholderText("이미지가 있는 폴더 (평면 또는 클래스별 하위 폴더)")
        validation_row = QHBoxLayout()
        validation_row.addWidget(self.validation_edit)
        validation_browse = QPushButton("찾아보기...")
        validation_browse.clicked.connect(self._browse_validation)
        validation_row.addWidget(validation_browse)
        opt_layout.addRow("비교 이미지:", validation_row)
        validation_hint = QLabel(
            "정답률을 측정하는 기능이 아니라, 같은 이미지에서 PyTorch와 ONNX 출력이 "
            "같은지 확인합니다. 이미지 0장 클래스는 자동 제외됩니다."
        )
        validation_hint.setWordWrap(True)
        validation_hint.setStyleSheet("color: #8A92A4;")
        opt_layout.addRow("", validation_hint)
        self.precision_check = QCheckBox("고정밀 호환 내보내기 허용 (느려질 수 있음, 실제 이미지 검증 모드에서는 미사용)")
        opt_layout.addRow("", self.precision_check)

        self.encrypt_check = QCheckBox("모델과 설정을 암호화해서 내보내기 (.dvsenc)")
        self.encrypt_check.toggled.connect(self._encryption_changed)
        opt_layout.addRow("", self.encrypt_check)
        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("별도로 보관할 암호키 파일 (.key)")
        key_row = QHBoxLayout()
        key_row.addWidget(self.key_edit)
        key_browse = QPushButton("기존 키 선택")
        key_browse.clicked.connect(self._browse_key)
        key_row.addWidget(key_browse)
        key_create = QPushButton("새 키 생성")
        key_create.clicked.connect(self._create_key)
        key_row.addWidget(key_create)
        opt_layout.addRow("암호키:", key_row)
        key_hint = QLabel("모든 태스크 지원. 복호화는 프로그램 시작 때 한 번만 수행합니다. 키를 잃으면 모델을 열 수 없으므로 별도로 보관하세요.")
        key_hint.setWordWrap(True)
        opt_layout.addRow("", key_hint)

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
        if self.project is not project:
            self.encrypt_check.setChecked(False)
            self.key_edit.clear()
        self.project = project
        self.export_precision.setCurrentIndex(0)
        self.calibration_edit.setText(project.data.train_dir or "")
        self.validation_edit.clear()
        efficientnet = project.task == "classify" and str(getattr(project.model, "model_id", "")).startswith("efficientnet_")
        self.dataset_verify_check.setChecked(efficientnet)
        self.dataset_verify_check.setEnabled(efficientnet)
        if efficientnet:
            from classification_export_validation import collect_images
            for folder in (project.data.val_dir, project.data.test_dir, project.data.train_dir):
                try:
                    collect_images(folder, project.data.class_names)
                except (ValueError, OSError):
                    continue
                self.validation_edit.setText(folder)
                break
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
            os.path.join(export_dir, f"model_{project.task}" + (".dvsenc" if self.encrypt_check.isChecked() else ".onnx"))
        )

    def _refresh_deploy_preview(self):
        path = self.ckpt_edit.text().strip()
        self._deploy_preview = deployment_preview(path) if path and os.path.isfile(path) else {}
        self._show_deploy_preview()

    def _show_deploy_preview(self):
        """체크포인트의 판정 영역과 함께 담길 판정 기준을 보여 준다."""
        preview = self._deploy_preview
        patch = bool(preview.get("patch")) and not preview.get("vit")
        self.region_row.setVisible(patch)
        lines = []
        if preview.get("region_error"):
            lines.append(f"<b>판정 영역 오류</b>: {preview['region_error']}")
        elif preview.get("original_size_region"):
            width, height = preview["original_size_region"]
            lines.append(f"원본 크기 학습 모델 · 고정 판정 영역 {width}×{height} px")
        elif patch:
            region = preview.get("region") or [0, 0]
            self.region_w.setValue(int(region[0]))
            self.region_h.setValue(int(region[1]))
            scale = preview.get("scale", 1.0)
            lines.append("패치 학습 모델: 판정 영역 전체를 한 번에 넣는 고정 입력으로 내보냅니다"
                         + ("" if scale == 1 else f" (배율 {scale:.0%})")
                         + (". 학습 영상 크기를 기본으로 채웠습니다." if region[0] else
                            ". <b>판정 영역 크기를 입력하세요</b> — 설비에서 넘기는 영상 크기"
                            "(예전 체크포인트라 학습 영상 크기 기록이 없습니다)."))
        elif preview.get("patch") and preview.get("vit"):
            lines.append("전역 문맥 모델: 학습 크기 타일로 내보냅니다. 새 C++ SDK(schema 8)가 자동으로 영역을 나누고 결과를 합칩니다.")
        if preview.get("task") == "segment":
            rules = preview.get("rules")
            if preview.get("rules_error"):
                lines.append(f"<b>판정 기준 오류</b>: {preview['rules_error']}")
            elif rules:
                parts = [f"기본 면적 ≥ {rules['min_area']}px · 점수 ≥ {rules['min_score']:.2f}"]
                parts += [f"{c['class_name']} 면적 ≥ {c['min_area']}px · 점수 ≥ {c['min_score']:.2f}" for c in rules["classes"]]
                lines.append("판정 기준 포함 → C++ SDK가 기준 미달 덩어리를 마스크에서 지웁니다: " + " | ".join(parts))
            else:
                lines.append("판정 기준 없음: 추론 화면에서 기준을 정하고 '모델에 저장'하면 함께 담깁니다.")
        self.deploy_info.setText("<br>".join(lines) or "체크포인트를 고르면 C++ 배포 정보를 보여 줍니다.")
        self.deploy_info.setVisible(bool(lines))

    def _browse_merge(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "표면 결함 모델 선택", "", "PyTorch Checkpoint (*.pt *.pth)")
        if filepath:
            self.merge_edit.setText(filepath)

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
            "암호화 모델 (*.dvsenc)" if self.encrypt_check.isChecked() else "ONNX 모델 (*.onnx)"
        )
        if filepath:
            self.output_edit.setText(filepath)

    def _encryption_changed(self, checked):
        from pathlib import Path
        if self.output_edit.text().strip():
            self.output_edit.setText(str(Path(self.output_edit.text()).with_suffix(".dvsenc" if checked else ".onnx")))

    def _browse_key(self):
        path, _ = QFileDialog.getOpenFileName(self, "암호키 선택", "", "암호키 (*.key);;모든 파일 (*)")
        if path:
            self.key_edit.setText(path)

    def _create_key(self):
        path, _ = QFileDialog.getSaveFileName(self, "새 암호키를 별도로 보관할 위치", "", "암호키 (*.key)")
        if path:
            try:
                from model_crypto import create_key
                create_key(path)
                self.key_edit.setText(path)
                self.encrypt_check.setChecked(True)
            except Exception as exc:
                QMessageBox.warning(self, "키 생성 실패", str(exc))

    def _browse_calibration(self):
        folder = QFileDialog.getExistingDirectory(self, "INT8 보정 이미지 폴더", self.calibration_edit.text())
        if folder:
            self.calibration_edit.setText(folder)

    def _browse_validation(self):
        folder = QFileDialog.getExistingDirectory(self, "분류 비교 이미지 폴더", self.validation_edit.text())
        if folder:
            self.validation_edit.setText(folder)

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

        encryption_key_path = None
        if self.encrypt_check.isChecked():
            encryption_key_path = self.key_edit.text().strip()
            try:
                from model_crypto import read_key
                read_key(encryption_key_path)
            except Exception as exc:
                QMessageBox.warning(self, "암호키 확인", str(exc))
                return

        try:
            self._validate_checkpoint_for_project(ckpt_path)
        except ValueError as exc:
            QMessageBox.warning(self, "체크포인트 확인", str(exc))
            return
        if not self.region_row.isHidden() and not (self.region_w.value() and self.region_h.value()):
            QMessageBox.warning(self, "판정 영역", "판정 영역 크기(설비에서 넘기는 영상의 가로×세로 px)를 입력하세요.")
            return
        merge_with = self.merge_edit.text().strip() if self.merge_check.isChecked() else ""
        if self.merge_check.isChecked():
            if not merge_with or not os.path.isfile(merge_with):
                QMessageBox.warning(self, "모델 합치기", "합칠 표면 결함 모델 체크포인트를 선택하세요.")
                return
            if encryption_key_path:
                QMessageBox.warning(self, "모델 합치기", "합친 모델은 아직 암호화 내보내기를 지원하지 않습니다. 암호화를 끄세요.")
                return

        if self.export_precision.currentData() == "int8":
            if merge_with or not self.verify_check.isChecked() or not os.path.isdir(self.calibration_edit.text().strip()):
                QMessageBox.warning(self, "INT8 설정", "출력 검증과 보정 이미지 폴더가 필요합니다. 모델 합치기는 FP32로 먼저 내보내세요.")
                return
        validation_dir = None
        if self.dataset_verify_check.isChecked() and self.verify_check.isChecked():
            validation_dir = self.validation_edit.text().strip()
            if not validation_dir or not os.path.isdir(validation_dir):
                QMessageBox.warning(self, "비교 이미지 필요", "이미지가 1장 이상 있는 폴더를 선택하세요. 평면 폴더와 클래스별 하위 폴더 모두 지원하며, 이미지와 가중치는 이 PC에서만 검사합니다.")
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
            validation_dir=validation_dir,
            precision=self.export_precision.currentData(), calibration_dir=self.calibration_edit.text().strip() or None,
            allow_precision_fallback=self.precision_check.isChecked(),
            encryption_key_path=encryption_key_path,
            merge_with=merge_with,
            overrides=({"region_size": [self.region_w.value(), self.region_h.value()]}
                       if self.region_row.isVisible() or (self._deploy_preview.get("patch") and
                                                          not self._deploy_preview.get("vit")) else None),
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
        status = ("INT8 실행·보정 영상 비교 완료" if result.get("precision") == "int8" else
                  "검증 통과" if result["verification"] == "passed" else "검증 건너뜀")
        if result.get("merged"):
            text = (f"두 모델 합치기 완료 ({status})\n\n모델: {result['output_path']}\n설정: {result['config_path']}\n"
                    f"크기: {result['file_size_mb']:.1f} MB\n클래스: {', '.join(result['class_names'])}\n"
                    f"입력: 판정 영역 {result['input_size'][1]}×{result['input_size'][0]} 한 번에\n"
                    "설비는 지금처럼 Segment를 한 번 부르면 됩니다. 마스크의 클래스 번호는 위 순서입니다.")
            if result.get("rules"):
                text += "\n두 모델의 판정 기준이 함께 담겼습니다 (C++ SDK가 기준 미달 덩어리를 지움)."
            self.log_text.append(text)
            QMessageBox.information(self, "완료", text)
            return
        text = (f"ONNX 내보내기 완료 ({status})\n\n"
                f"모델: {result['output_path']}\n"
                f"설정: {result.get('config_path') or '암호화 파일에 포함'}\n"
                f"크기: {result['file_size_mb']:.1f} MB")
        if result.get("precision") == "int8":
            text += "\nINT8은 FP32와 수치·정확도가 달라질 수 있습니다. 보정 영상 비교 결과는 로그와 JSON에 저장했습니다. 별도 시험 영상으로 확인하세요."
        if result.get("backend") == "sam2" and not result.get("encrypted"):
            text += "\nSAM2는 encoder ONNX, decoder ONNX, sam2.json 세 파일을 함께 배포합니다."
        if result.get("encrypted"):
            text += "\nC++/C# 암호화 로더와 별도 암호키로 엽니다. 키 파일을 모델과 함께 공개하지 마세요."
        elif result.get("cpp_supported"):
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
        validation = result.get("classification_validation")
        if validation:
            measured = validation["selected"]
            if validation["policy"] == "classification_dataset_v1":
                text += (f"\n실제 이미지 {validation['image_count']}장: 분류 일치·확률 검증 통과 (FP32)"
                         f"\n전처리+추론+후처리: 중앙값 {measured['pipeline_median_ms']:.2f} / "
                         f"p95 {measured['pipeline_p95_ms']:.2f} / 최대 {measured['pipeline_max_ms']:.2f} ms"
                         f"\n8ms 목표: {'측정 이미지 모두 충족' if validation['target_met'] else '미달성 — 내보내기 검증 통과와 별개'}")
            else:
                text += (f"\n결정론적 합성 입력 {validation['probe_count']}개: top-1·확률·판정 마진 검증 통과 (FP32)"
                         f"\n현재 PC 모델 추론 중앙값: {measured['model_median_ms']:.2f} ms"
                         "\n실제 이미지 검증 자료가 없어 합성 입력 범위만 인증했습니다.")
        self.log_text.append(text)
        QMessageBox.information(self, "완료", text)

    def _on_error(self, error_msg):
        self.log_text.append(f"\n오류: {error_msg}")
        QMessageBox.critical(self, "오류", f"내보내기 실패:\n{error_msg}")
