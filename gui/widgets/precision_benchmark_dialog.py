"""Compare deployment ONNX models on the desktop CPU."""

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


class PrecisionBenchmarkWorker(QThread):
    progress = Signal(str)
    completed = Signal(list)
    failed = Signal(str)

    def __init__(self, options, parent):
        super().__init__(parent)
        self.options = options

    def run(self):
        try:
            from precision_benchmark import benchmark_precisions

            rows = benchmark_precisions(
                **self.options, log=self.progress.emit, cancelled=self.isInterruptionRequested,
            )
            if not self.isInterruptionRequested():
                self.completed.emit(rows)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


class PrecisionBenchmarkDialog(QDialog):
    def __init__(self, parent=None, *, fp32_path="", fp16_path="", int8_path="",
                 image_dir="", calibration_dir="", browse_dir=""):
        super().__init__(parent)
        self.setWindowTitle("ONNX 정밀도·CPU 스레드 비교")
        self.resize(960, 700)
        self.worker = None
        self._cancel_requested = False
        self.browse_dir = browse_dir
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "같은 이미지로 CPU에서 FP32·FP16·INT8을 비교합니다. "
            "FP16 JSON을 비우면 비교용 모델을 만듭니다. 탐지·분할·Dinomaly는 "
            "보정 이미지로 INT8 모델도 만들 수 있습니다. 기존 모델은 평문 ONNX와 배포 JSON을 사용하세요."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.settings = QGroupBox("모델과 이미지")
        form = QFormLayout(self.settings)
        self.fp32, _ = self._path_row(form, "FP32 배포 JSON", fp32_path, "기준 모델의 ONNX 설정 (.json)")
        choices = QHBoxLayout()
        self.run_fp16 = QCheckBox("FP16")
        self.run_int8 = QCheckBox("INT8")
        for choice in (self.run_fp16, self.run_int8):
            choice.setChecked(True)
            choices.addWidget(choice)
        choices.addStretch()
        form.addRow("비교 정밀도 (FP32 포함)", choices)
        self.fp16, fp16_row = self._path_row(form, "FP16 배포 JSON", fp16_path, "비워 두면 FP32 모델에서 생성")
        self.int8, int8_row = self._path_row(form, "INT8 배포 JSON", int8_path, "비워 두면 보정 이미지로 생성")
        self.images, _ = self._path_row(form, "시험 이미지 폴더", image_dir, "세 모델에 공통으로 사용할 이미지", folder=True)
        self.calibration, calibration_row = self._path_row(
            form, "INT8 보정 이미지 폴더", calibration_dir, "기존 INT8 배포 JSON을 선택하면 생략 가능", folder=True,
        )
        self.run_fp16.toggled.connect(fp16_row.setEnabled)
        self.run_int8.toggled.connect(int8_row.setEnabled)
        self.run_int8.toggled.connect(calibration_row.setEnabled)
        self.threads = QLineEdit("1, 2, 4, 8")
        self.threads.setToolTip("쉼표로 구분한 ONNX Runtime CPU 스레드 수. 0은 자동입니다.")
        form.addRow("CPU 스레드 수 (0=자동)", self.threads)
        counts = QHBoxLayout()
        self.image_count = self._count(counts, "이미지 수", 1, 100, 5)
        self.repeats = self._count(counts, "반복", 1, 30, 2)
        self.warmup = self._count(counts, "워밍업", 0, 20, 2)
        counts.addStretch()
        form.addRow("측정 설정", counts)
        layout.addWidget(self.settings)

        scope = QLabel(
            "전체 시간은 이미지 읽기, 전처리, 추론, 후처리를 포함합니다. "
            "p50은 중앙값이고 p95는 95%의 측정이 이 값 이하였다는 뜻입니다. "
            "출력은 같은 스레드 수의 FP32와 비교하며 정답률은 계산하지 않습니다. "
            "FP16은 CPU에서 FP32 변환이 들어가거나 지원되지 않을 수 있습니다."
        )
        scope.setWordWrap(True)
        layout.addWidget(scope)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["정밀도", "CPU 스레드", "전체 p50 (ms)", "전체 p95 (ms)", "모델 p50 (ms)", "FP32 대비", "상태"]
        )
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(155)
        layout.addWidget(self.table, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(104)
        self.log.setAccessibleName("비교 진행 로그")
        layout.addWidget(self.log)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)

        controls = QHBoxLayout()
        controls.addStretch()
        self.start_button = QPushButton("비교 시작")
        self.start_button.setProperty("cssClass", "primary")
        self.start_button.clicked.connect(self._start)
        controls.addWidget(self.start_button)
        self.cancel_button = QPushButton("중단")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        controls.addWidget(self.cancel_button)
        close_button = QPushButton("닫기")
        close_button.clicked.connect(self.reject)
        controls.addWidget(close_button)
        layout.addLayout(controls)

    @staticmethod
    def _count(layout, label, low, high, value):
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        spin.setAccessibleName(label)
        layout.addWidget(QLabel(label))
        layout.addWidget(spin)
        return spin

    def _path_row(self, form, label, value, placeholder, *, folder=False):
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        field = QLineEdit(value)
        field.setPlaceholderText(placeholder)
        field.setAccessibleName(label)
        row.addWidget(field, 1)
        button = QPushButton("찾아보기…")
        button.setAccessibleName(f"{label} 찾아보기")
        button.clicked.connect(lambda: self._browse(field, label, folder))
        row.addWidget(button)
        form.addRow(label, container)
        return field, container

    def _browse(self, field, label, folder):
        current = field.text().strip() or self.browse_dir
        if folder:
            path = QFileDialog.getExistingDirectory(self, label, current)
        else:
            path, _ = QFileDialog.getOpenFileName(self, label, current, "ONNX 배포 설정 (*.json)")
        if path:
            field.setText(path)

    def _start(self):
        if self.worker is not None:
            return
        try:
            from precision_benchmark import parse_thread_counts

            thread_counts = parse_thread_counts(self.threads.text())
            fp32_path = self.fp32.text().strip()
            image_dir = self.images.text().strip()
            fp16_path = self.fp16.text().strip() if self.run_fp16.isChecked() else ""
            int8_path = self.int8.text().strip() if self.run_int8.isChecked() else ""
            for name, path in (("FP32", fp32_path), ("FP16", fp16_path), ("INT8", int8_path)):
                if (path or name == "FP32") and (not Path(path).is_file() or Path(path).suffix.lower() != ".json"):
                    raise ValueError(f"{name} ONNX 배포 JSON을 선택하세요.")
            if not image_dir or not Path(image_dir).is_dir():
                raise ValueError("시험 이미지 폴더를 선택하세요.")
            calibration_dir = self.calibration.text().strip() if self.run_int8.isChecked() and not int8_path else ""
            if calibration_dir and not Path(calibration_dir).is_dir():
                raise ValueError("INT8 보정 이미지 폴더를 확인하세요.")
        except (ImportError, ValueError, OSError) as exc:
            self.status.setText(str(exc))
            return
        options = {
            "fp32_path": fp32_path, "image_dir": image_dir, "thread_counts": thread_counts,
            "fp16_path": fp16_path or None, "int8_path": int8_path or None,
            "calibration_dir": calibration_dir or None,
            "run_fp16": self.run_fp16.isChecked(), "run_int8": self.run_int8.isChecked(),
            "image_count": self.image_count.value(), "repeats": self.repeats.value(), "warmup": self.warmup.value(),
        }
        self.table.setRowCount(0)
        self.log.clear()
        self.status.setText("측정 중…")
        self._cancel_requested = False
        self.worker = PrecisionBenchmarkWorker(options, self)
        self.worker.progress.connect(self.log.appendPlainText)
        self.worker.completed.connect(self._show_results)
        self.worker.failed.connect(self.status.setText)
        self.worker.finished.connect(self._finished)
        self.settings.setEnabled(False)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.show()
        self.worker.start()

    @staticmethod
    def _format_ms(value):
        return f"{value:.2f}" if isinstance(value, (int, float)) else "—"

    def _show_results(self, rows):
        if self._cancel_requested:
            return
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            threads = row["threads"]
            thread_label = "—" if threads is None else str(threads or "자동")
            values = (
                row["precision"], thread_label,
                *(self._format_ms(row.get(key)) for key in ("p50_ms", "p95_ms", "model_p50_ms")),
                str(row.get("comparison") or "—"), row["status"],
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if row.get("detail"):
                    cell.setToolTip(row["detail"])
                self.table.setItem(index, column, cell)
            if row.get("detail"):
                self.log.appendPlainText(f"{row['precision']} · {values[1]}: {row['detail']}")
        self.status.setText("측정 완료")

    def _cancel(self):
        if self.worker is not None and self.worker.isRunning():
            self._cancel_requested = True
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status.setText("현재 작업이 끝나면 중단합니다…")

    def _finished(self):
        worker, self.worker = self.worker, None
        if self._cancel_requested:
            self.status.setText("측정을 중단했습니다.")
        worker.deleteLater()
        self.settings.setEnabled(True)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.hide()

    def isRunning(self):
        return self.worker is not None

    def reject(self):
        if self.worker is not None:
            self._cancel()
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self._cancel()
            event.ignore()
            return
        super().closeEvent(event)
