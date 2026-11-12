"""Project classes and generator methods are separate, explicit choices."""
from pathlib import Path

import numpy as np
from core.defect_generator import DEFECT_INFO
from core.desktop_jobs import DesktopJob
from core.project_generation import ProjectGenerationStore, TASK_OUTPUTS, class_options
from PIL import Image
from PySide6.QtCore import QPointF, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                              QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
                              QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
                              QTableWidgetItem, QVBoxLayout, QWidget)
from webapp.storage import project_view
from widgets.datagen_editor import Comparison


class ProjectGenerationWidget(QWidget):
    generation_finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = self._worker = self.batch = self._result = None
        self.output_path = ""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        scroll.setWidget(body)
        outer.addWidget(scroll)
        layout = QVBoxLayout(body)
        self.description = QLabel("프로젝트를 선택하세요.")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        hint = QLabel("검사 클래스와 생성 방식은 다릅니다. 등록한 클래스에 어울리는 방식을 직접 선택하세요.\n"
                      "중앙 크롭을 설정했다면 그 판정 영역 안에만 생성합니다. 원본과 새 이미지·정답을 확인한 뒤 승인하세요.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.controls = QWidget()
        controls = QVBoxLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        self.classes = QTableWidget(0, 3)
        self.classes.setHorizontalHeaderLabels(["프로젝트 클래스 (선택)", "생성 방법", "생성 장수"])
        self.classes.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.classes.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.classes.setColumnWidth(1, 220)
        self.classes.verticalHeader().setDefaultSectionSize(46)
        self.classes.setMaximumHeight(155)
        controls.addWidget(self.classes)
        form = QFormLayout()
        self.source = QComboBox()
        form.addRow("Train 바탕 클래스", self.source)
        self.size_min, self.size_max = self.percent(.1, 50, 1), self.percent(.1, 50, 8)
        self.strength_min, self.strength_max = self.percent(1, 100, 30), self.percent(1, 100, 80)
        for name, a, b in (("생성 영역 크기 범위 (%)", self.size_min, self.size_max),
                           ("강도 범위 (%)", self.strength_min, self.strength_max)):
            row = QHBoxLayout()
            row.addWidget(a)
            row.addWidget(QLabel("~"))
            row.addWidget(b)
            form.addRow(name, row)
        self.size_max.setToolTip("패턴 후보 사각형의 면적 비율입니다. 실제 변경 픽셀 면적과는 다릅니다.")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        form.addRow("재현 시드", self.seed)
        controls.addLayout(form)
        self.generate_button = self.button("1. 선택 클래스 후보 생성", self.generate)
        controls.addWidget(self.generate_button)
        layout.addWidget(self.controls)
        self.message = QLabel("프로젝트 클래스를 선택하세요.")
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.message)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.review_controls = QWidget()
        review = QVBoxLayout(self.review_controls)
        review.setContentsMargins(0, 0, 0, 0)
        self.batches = QComboBox()
        self.batches.currentIndexChanged.connect(lambda: self.guard(self.load_batch))
        review.addWidget(self.batches)
        split = QSplitter()
        self.samples = QListWidget()
        self.samples.currentRowChanged.connect(lambda: self.guard(self.show_sample))
        split.addWidget(self.samples)
        self.compare = Comparison()
        self.compare.view.setMinimumHeight(220)
        split.addWidget(self.compare)
        split.setStretchFactor(1, 4)
        review.addWidget(split, 1)
        self.overlay = QCheckBox("정답 영역 표시 (원본 쪽은 그대로 유지)")
        self.overlay.toggled.connect(lambda: self.guard(self.show_sample))
        review.addWidget(self.overlay)
        row = QHBoxLayout()
        row.addWidget(self.button("2. 선택 후보 승인 · 새 프로젝트 저장", self.publish))
        row.addWidget(self.button("저장 폴더 열기", self.open_output))
        review.addLayout(row)
        layout.addWidget(self.review_controls, 1)
        self.stop_button = self.button("작업 중단", self.stop)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

    @staticmethod
    def percent(minimum, maximum, value):
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setSuffix(" %")
        return widget

    def button(self, title, callback):
        widget = QPushButton(title)
        widget.clicked.connect(lambda: self.guard(callback))
        return widget

    def guard(self, callback):
        try:
            return callback()
        except Exception as exc:
            self.message.setText("작업 실패: " + str(exc))

    def store(self):
        if self.project is None:
            raise ValueError("프로젝트를 먼저 여세요.")
        return ProjectGenerationStore(self.project)

    def set_project(self, project):
        if self._worker is not None:
            return
        self.project, self.batch = project, None
        self.output_path = ""
        self.classes.setRowCount(0)
        self.source.clear()
        self.controls.setEnabled(project is not None)
        self.review_controls.setEnabled(project is not None)
        if project is None:
            self.batches.clear()
            self.samples.clear()
            return
        self.description.setText(TASK_OUTPUTS[project.task] + ("\n합성 불량은 정상 Train에 추가하지 않습니다. 별도 합성 평가 프로젝트로 저장합니다."
                                if project.task == "anomaly" else "\n원본 Train에서 생성하며 승인본은 새 프로젝트의 Train에만 추가합니다."))
        self.source.setEnabled(project.task == "classify")
        self.source.addItems(project.data.class_names if project.task == "classify" else
                             ["Train / good" if project.task == "anomaly" else "Train · 기존 정답 영역 보호"])
        options = class_options(project)
        for cid, name in options:
            row = self.classes.rowCount()
            self.classes.insertRow(row)
            item = QTableWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, cid)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.classes.setItem(row, 0, item)
            method = QComboBox()
            for kind, info in DEFECT_INFO.items():
                method.addItem(info["name"], kind.value)
            self.classes.setCellWidget(row, 1, method)
            count = QSpinBox()
            count.setRange(1, 100)
            count.setValue(3)
            self.classes.setCellWidget(row, 2, count)
        self.message.setText("클래스를 선택하고 생성 방법을 지정하세요." if options else
                             "생성할 클래스가 없습니다. 데이터셋 화면에서 프로젝트 클래스를 먼저 등록하세요.")
        self.guard(self.refresh)

    def refresh(self):
        selected = self.batches.currentData()
        self.batches.blockSignals(True)
        self.batches.clear()
        for batch in self.store().batches():
            self.batches.addItem(f"{batch['created'][:19]} · {len(batch['samples'])}장", batch["id"])
        self.batches.setCurrentIndex(max(0, self.batches.findData(selected)))
        self.batches.blockSignals(False)
        self.load_batch()

    def load_batch(self):
        self.batch = None
        self.samples.clear()
        if not self.batches.currentData():
            self.compare.view.scene().clear()
            self.compare.overlay = None
            return
        self.batch = self.store().load(self.batches.currentData())
        for row in self.batch["samples"]:
            item = QListWidgetItem(f"{self.batch['classes'][row['class_id']]} · 크기 {row['size_ratio']:.1%} · 강도 {row['strength']:.0%}")
            item.setData(Qt.ItemDataRole.UserRole, row["id"])
            item.setCheckState(Qt.CheckState.Unchecked)
            self.samples.addItem(item)
        self.samples.setCurrentRow(0)

    def show_sample(self):
        if not self.batch or self.samples.currentRow() < 0:
            return
        row = self.batch["samples"][self.samples.currentRow()]
        folder = self.store().verified_sample(self.batch["id"], row)
        self.compare.load(folder / "original.png", folder / "image.png")
        if self.overlay.isChecked():
            with Image.open(folder / "image.png") as image:
                pixels = np.array(image.convert("RGB"))
            if (folder / "mask.png").exists():
                with Image.open(folder / "mask.png") as mask:
                    labels = np.array(mask)
                for cid in np.unique(labels):
                    if cid in (0, 255):
                        continue
                    color = QColor.fromHsv((int(cid)*67+205) % 360, 180, 255)
                    region = labels == cid
                    pixels[region] = (pixels[region]*.6 + np.array([color.red(), color.green(), color.blue()])*.4).astype(np.uint8)
            else:
                with Image.open(folder / "changed.png") as mask:
                    region = np.array(mask) > 0
                pixels[region] = (pixels[region]*.6 + np.array([0, 220, 180])*.4).astype(np.uint8)
            qimage = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888).copy()
            if (folder / "label.txt").exists():
                from core.dataset_editor import read_annotations
                rows = read_annotations(folder / "label.txt", self.project.task, len(self.batch["classes"]))
                painter = QPainter(qimage)
                try:
                    h, w = pixels.shape[:2]
                    for label in rows:
                        cid = label["class_id"]
                        painter.setPen(QPen(QColor.fromHsv((cid*67+205) % 360, 180, 255), max(1, w//500)))
                        coordinates = label["coordinates"]
                        if self.project.task == "detect":
                            x, y, bw, bh = coordinates
                            points = [(x-bw/2, y-bh/2), (x+bw/2, y-bh/2),
                                      (x+bw/2, y+bh/2), (x-bw/2, y+bh/2)]
                        else:
                            points = list(zip(coordinates[::2], coordinates[1::2]))
                        painter.drawPolygon(QPolygonF([QPointF(x*w, y*h) for x, y in points]))
                        painter.drawText(QPointF(points[0][0]*w, max(12, points[0][1]*h)), self.batch["classes"][cid])
                finally:
                    painter.end()
            # Overlay sits below the original-side pixmap, preserving comparison.
            overlay = self.compare.view.scene().addPixmap(QPixmap.fromImage(qimage))
            overlay.setZValue(.5)
            self.compare.overlay.setZValue(1)
        self.message.setText(f"{self.batch['classes'][row['class_id']]} · 원본: {row['source']}\n"
                             "저장할 후보만 목록에서 체크하세요. 체크한 후보를 승인하여 새 프로젝트로 저장합니다.")

    def generate(self):
        rules = [{"class_id": self.classes.item(i, 0).data(Qt.ItemDataRole.UserRole),
                  "method": self.classes.cellWidget(i, 1).currentData(), "count": self.classes.cellWidget(i, 2).value()}
                 for i in range(self.classes.rowCount()) if self.classes.item(i, 0).checkState() == Qt.CheckState.Checked]
        if not rules:
            raise ValueError("생성할 프로젝트 클래스를 체크하세요.")
        self.launch("generate", {"rules": rules, "source_class": self.source.currentText(), "seed": self.seed.value(),
                                 "size_min": self.size_min.value()/100, "size_max": self.size_max.value()/100,
                                 "strength_min": self.strength_min.value()/100, "strength_max": self.strength_max.value()/100})

    def publish(self):
        if not self.batch:
            raise ValueError("생성 후보가 없습니다.")
        ids = [self.samples.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.samples.count())
               if self.samples.item(i).checkState() == Qt.CheckState.Checked]
        if not ids:
            raise ValueError("원본·생성 결과·정답을 확인한 뒤 저장할 후보를 체크하세요.")
        self.launch("publish", {"batch_id": self.batch["id"], "sample_ids": ids})

    def launch(self, operation, values):
        if self._worker is not None:
            raise ValueError("실행 중인 작업이 있습니다.")
        self._result = None
        self._worker = DesktopJob("project_generation", {"project": project_view(self.project), "operation": operation, **values}, self)
        self._worker.completed.connect(lambda result: setattr(self, "_result", result))
        self._worker.failed.connect(self.message.setText)
        self._worker.event.connect(self.on_event)
        self._worker.finished.connect(self.finished)
        self.controls.setEnabled(False)
        self.review_controls.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setValue(0)
        self.message.setText("후보 생성 중…" if operation == "generate" else "승인 결과 저장 중…")
        self._worker.start()

    def on_event(self, name, args):
        if name == "progress_updated" and args[1]:
            self.progress.setValue(round(100*args[0]/args[1]))
        elif name == "log_message":
            self.message.setText(str(args[0]))

    def finished(self):
        worker, self._worker = self._worker, None
        result = self._result or {"status": "failed", "error": worker.error or "작업 결과 없음"}
        worker.deleteLater()
        self.controls.setEnabled(True)
        self.review_controls.setEnabled(True)
        self.stop_button.setEnabled(False)
        if result.get("status") != "completed":
            self.message.setText("작업 중단" if result.get("status") == "cancelled" else "작업 실패: " + result.get("error", ""))
            return
        output = result["output"]
        if "batch_id" in output:
            self.guard(self.refresh)
            self.batches.setCurrentIndex(self.batches.findData(output["batch_id"]))
            self.message.setText(f"{output['generated']}장 생성 완료. 원본·정답을 확인하고 저장할 후보를 체크하세요."
                                 + (f" 배경 부족으로 {len(output['failures'])}장 제외." if output["failures"] else ""))
        else:
            self.output_path = output["project_path"]
            self.message.setText(f"{output['saved']}장 저장 완료. 새 프로젝트: {self.output_path}")
            self.generation_finished.emit(output["saved"])

    def open_output(self):
        if not self.output_path:
            raise ValueError("먼저 승인본을 저장하세요.")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.output_path).parent)))

    def stop(self):
        if self._worker is not None:
            self._worker.stop()
