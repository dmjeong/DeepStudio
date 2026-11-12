"""Project annotation -> native defect candidates -> reviewed training project."""

from pathlib import Path

import numpy as np
from core.desktop_jobs import DesktopJob
from core.mask_annotations import MaskDocument
from core.segmentation_synthesis import POLICIES, default_policy
from core.segmentation_synthesis_store import SynthesisStore, format_analysis_report
from core.synthesis_variation import DEFAULT_VARIATION, variation_settings
from PIL import Image
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from widgets.result_image_label import ResultImageLabel

from webapp.storage import project_view


def pixmap(pixels):
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    pixels = np.ascontiguousarray(pixels)
    h, w = pixels.shape[:2]
    return QPixmap.fromImage(
        QImage(pixels.data, w, h, pixels.strides[0], QImage.Format.Format_RGB888).copy()
    )


class SegmentationSynthesisWidget(QWidget):
    generation_finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = self._worker = self._result = None
        self.batch = None
        self.output_path = ""
        self._analysis = None
        self._operation = None
        self._selection_context = None
        self._available_classes = set()
        self._last_error = ""
        self._job_details = ""
        self._notice = QMessageBox(self)
        self._notice.setWindowModality(Qt.WindowModality.NonModal)
        layout = QVBoxLayout(self)
        note = QLabel(
            "SEG 가상 불량 · 균열 경로·가지 / 이물 윤곽을 새로 생성 · 크기·강도 조절 · 원본 해상도 유지\n"
            "참조에서 색·질감을 가져옵니다. shift·double·flip은 물체 배치를 함께 변형합니다. Train → 학습 후보 / Val → 검수 후보."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.controls = QWidget()
        controls = QVBoxLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.profiles = QComboBox()
        self.profiles.currentIndexChanged.connect(lambda: self.guard(self.load_profile))
        row.addWidget(QLabel("정답 분석 버전"))
        row.addWidget(self.profiles, 1)
        self.analyze_button = self.button("1. 프로젝트 정답 분석", self.analyze)
        row.addWidget(self.analyze_button)
        self.unlabelled = QCheckBox("정답 없는 사진도 정상 바탕으로")
        self.unlabelled.setToolTip(
            "체크하지 않으면 마스크/폴리곤 정답이 있는 사진만 바탕으로 사용합니다. 정상 사진은 빈 마스크도 가능합니다."
        )
        row.addWidget(self.unlabelled)
        controls.addLayout(row)
        self.classes = QTableWidget(0, 5)
        self.classes.setStyleSheet("""
            QTableWidget QComboBox, QTableWidget QSpinBox {
                min-height: 22px; padding: 1px 20px 1px 5px;
            }
        """)
        self.classes.verticalHeader().setDefaultSectionSize(32)
        self.classes.setHorizontalHeaderLabels(
            [
                "생성 클래스",
                "Train 참조",
                "Val 참조",
                "방식 (변경 후 재분석)",
                "생성 수",
            ]
        )
        self.classes.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.classes.setMaximumHeight(210)
        controls.addWidget(self.classes)
        row = QHBoxLayout()
        row.addWidget(QLabel("생성 방식"))
        self.variation_mode = QComboBox()
        self.variation_mode.addItem("새 형상 생성 (균열·이물)", "new_shape")
        self.variation_mode.addItem("참조 형상 회전·크기 변형", "deform")
        self.variation_mode.addItem("참조 형상 유지 (강도만 조절)", "reference")
        row.addWidget(self.variation_mode, 1)
        self.variation_inputs = {}
        for title, low, high, minimum, maximum in (
            ("크기", "size_min", "size_max", 35, 300),
            ("강도", "strength_min", "strength_max", 20, 300),
        ):
            row.addWidget(QLabel(title))
            for index, key in enumerate((low, high)):
                if index:
                    row.addWidget(QLabel("~"))
                spin = QSpinBox()
                spin.setRange(minimum, maximum)
                spin.setSuffix("%")
                spin.setValue(round(DEFAULT_VARIATION[key] * 100))
                spin.setToolTip(
                    "100%는 참조 결함 기준입니다. 범위 안에서 후보마다 무작위로 선택합니다."
                )
                self.variation_inputs[key] = spin
                row.addWidget(spin)
        row.addWidget(QLabel("형상 변화"))
        self.shape_amount = QSpinBox()
        self.shape_amount.setRange(0, 100)
        self.shape_amount.setSuffix("%")
        self.shape_amount.setValue(round(DEFAULT_VARIATION["shape_amount"] * 100))
        self.shape_amount.setToolTip(
            "높일수록 균열 굽음·가지와 이물 윤곽의 불규칙성이 커집니다."
        )
        row.addWidget(self.shape_amount)
        self.variation_mode.currentIndexChanged.connect(self.variation_mode_changed)
        controls.addLayout(row)
        row = QHBoxLayout()
        self.split = QComboBox()
        self.split.addItem("Train — 학습용 후보", "train")
        self.split.addItem("Val — 검수 전용 후보", "val")
        self.split.currentIndexChanged.connect(self.update_available)
        row.addWidget(self.split)
        row.addWidget(QLabel("시드"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        row.addWidget(self.seed)
        row.addWidget(QLabel("표면 허용차"))
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setDecimals(3)
        self.tolerance.setRange(0.005, 0.3)
        self.tolerance.setSingleStep(0.01)
        self.tolerance.setValue(0.12)
        self.tolerance.setToolTip(
            "주변 밝기를 맞춘 뒤 표면 차이 허용값. 작을수록 엄격하며 합성 가능한 위치가 줄어듭니다."
        )
        row.addWidget(self.tolerance)
        self.preview_button = self.button(
            "2. 클래스당 1장 미리 생성", lambda: self.generate(True)
        )
        self.generate_button = self.button("설정 수량 생성", self.generate)
        row.addWidget(self.preview_button)
        row.addWidget(self.generate_button)
        controls.addLayout(row)
        row = QHBoxLayout()
        self.allowed = QLineEdit()
        self.allowed.setPlaceholderText(
            "합성 허용 영역 PNG (선택) — 원본 크기, 흰색만 허용"
        )
        row.addWidget(self.allowed, 1)
        row.addWidget(self.button("영역 파일", self.browse_roi))
        controls.addLayout(row)
        layout.addWidget(self.controls)
        row = QHBoxLayout()
        self.batches = QComboBox()
        self.batches.currentIndexChanged.connect(lambda: self.guard(self.load_batch))
        row.addWidget(QLabel("생성 기록"))
        row.addWidget(self.batches, 1)
        row.addWidget(self.button("기록 새로고침", self.refresh))
        row.addWidget(self.button("후보 폴더", self.open_batch))
        self.overlay = QCheckBox("클래스 마스크 표시")
        self.overlay.setChecked(True)
        self.overlay.toggled.connect(lambda: self.guard(self.show_sample))
        row.addWidget(self.overlay)
        row.addWidget(self.button("결함 위치 확대", self.focus_defect))
        layout.addLayout(row)
        splitter = QSplitter()
        self.samples = QListWidget()
        self.samples.currentRowChanged.connect(lambda: self.guard(self.show_sample))
        splitter.addWidget(self.samples)
        panels = QSplitter()
        self.original = ResultImageLabel("바탕 원본")
        self.generated = ResultImageLabel("합성 결과 + 클래스 정답")
        self.reference = ResultImageLabel("실제 결함 참조")
        self.reference.setMaximumWidth(230)
        for title, view in (
            ("바탕 원본", self.original),
            ("합성 결과 / 마스크", self.generated),
            ("실제 결함 참조", self.reference),
        ):
            box = QWidget()
            content = QVBoxLayout(box)
            content.addWidget(QLabel(title))
            content.addWidget(view, 1)
            panels.addWidget(box)
        self.original.view_changed.connect(self.generated.set_view)
        self.generated.view_changed.connect(self.original.set_view)
        splitter.addWidget(panels)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([180, 1000])
        layout.addWidget(splitter, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.review_controls = QWidget()
        review_layout = QVBoxLayout(self.review_controls)
        review_layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(self.button("정답 마스크 수정", self.edit_mask))
        row.addWidget(self.button("3. 이 후보 승인", lambda: self.review("approved")))
        row.addWidget(self.button("제외", lambda: self.review("rejected")))
        row.addWidget(self.button("미검수로 되돌리기", lambda: self.review("pending")))
        review_layout.addLayout(row)
        row = QHBoxLayout()
        self.save_scope = QComboBox()
        self.save_scope.addItem("선택 후보", "selected")
        self.save_scope.addItem("승인 후보 전체", "approved")
        self.save_scope.addItem("제외하지 않은 후보 전체", "active")
        row.addWidget(self.save_scope)
        self.save_images_button = self.button("이미지·정답 PNG 저장", self.save_images)
        row.addWidget(self.save_images_button)
        self.publish_button = self.button(
            "4. 승인본 포함 새 학습 프로젝트 저장", self.publish
        )
        row.addWidget(self.publish_button)
        review_layout.addLayout(row)
        layout.addWidget(self.review_controls)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.message = QLabel("프로젝트 Train/Val에서 클래스별 불량 정답을 분석하세요.")
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.message)
        row = QHBoxLayout()
        self.stop = self.button(
            "작업 중단", lambda: self._worker.stop() if self._worker else None
        )
        self.stop.setEnabled(False)
        row.addWidget(self.stop)
        self.open_button = self.button("저장 폴더 열기", self.open_output)
        self.open_button.setEnabled(False)
        row.addWidget(self.open_button)
        layout.addLayout(row)
        # Keep status and actionable errors above the table even on a small
        # Windows display; a bottom-only label looked like a no-op.
        layout.removeWidget(self.message)
        layout.removeWidget(self.progress)
        status = QWidget()
        status_layout = QVBoxLayout(status)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.message.setMaximumHeight(64)
        status_layout.addWidget(self.message)
        self.analysis_summary = QLabel("아직 정답을 분석하지 않았습니다.")
        self.analysis_summary.setWordWrap(True)
        status_layout.addWidget(self.analysis_summary)
        buttons = QHBoxLayout()
        buttons.addWidget(self.progress, 1)
        self.details_button = self.button("분석 상세 / 오류 보기", self.show_details)
        self.copy_button = self.button("내용 복사", self.copy_details)
        buttons.addWidget(self.details_button)
        buttons.addWidget(self.copy_button)
        status_layout.addLayout(buttons)
        layout.insertWidget(1, status)

    def diagnostic_text(self):
        text = format_analysis_report(self._analysis)
        if self._last_error:
            text = "오류: " + self._last_error + "\n\n" + text
        if self._job_details:
            text += "\n\n실행 기록:\n" + self._job_details
        return text

    def copy_details(self):
        QApplication.clipboard().setText(self.diagnostic_text())

    def show_details(self):
        self._notice.setWindowTitle("SEG 정답 분석 상세")
        self._notice.setIcon(QMessageBox.Icon.Information)
        self._notice.setInformativeText("")
        self._notice.setText(self._last_error or self.analysis_summary.text())
        self._notice.setDetailedText(self.diagnostic_text())
        self._notice.show()
        self._notice.raise_()

    def show_error(self, message):
        self._last_error = str(message)
        self.message.setText("작업 실패: " + self._last_error)
        self.progress.setFormat("실패 — %p%")
        self._notice.setWindowTitle("SEG 가상 불량 작업 실패")
        self._notice.setIcon(QMessageBox.Icon.Warning)
        self._notice.setText(self._last_error)
        self._notice.setInformativeText(
            "상단 ‘분석 상세 / 오류 보기’에서 경로와 제외 이유를 확인하고 ‘내용 복사’로 복사할 수 있습니다."
        )
        self._notice.setDetailedText(self.diagnostic_text())
        self._notice.show()
        self._notice.raise_()

    def update_summary(self):
        if self._analysis is None:
            self.analysis_summary.setText(
                "분석 기록 없음 — 프로젝트 정답 분석을 실행하세요."
            )
            return
        parts = []
        for split in ("train", "val"):
            row = self._analysis["splits"][split]
            usable = sum(v["accepted"] for v in row["classes"].values())
            parts.append(
                f"{split.upper()} 이미지 {row['images']}장 · 정답 파일 {row['annotated']}장 · 사용 가능한 참조 {usable}개"
            )
        prefix = (
            "분석 실패(처리된 범위의 집계) | "
            if self._analysis.get("status") == "failed"
            else ""
        )
        self.analysis_summary.setText(prefix + " | ".join(parts))

    def button(self, title, action):
        button = QPushButton(title)
        button.clicked.connect(lambda _checked=False: self.guard(action))
        return button

    def guard(self, action):
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - display failures at the UI boundary
            self.show_error(str(exc))

    def store(self):
        return SynthesisStore(self.project)

    def set_project(self, project):
        if self._worker:
            return
        self.project = project
        self._analysis = None
        self._selection_context = None
        self._available_classes = set()
        self._last_error = self._job_details = ""
        self._notice.close()
        self.update_summary()
        self.output_path = ""
        self.open_button.setEnabled(False)
        self.classes.setRowCount(0)
        if project:
            for cid, name in enumerate(project.data.class_names[1:], 1):
                index = self.classes.rowCount()
                self.classes.insertRow(index)
                item = QTableWidgetItem(name)
                item.setData(Qt.ItemDataRole.UserRole, cid)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(Qt.CheckState.Checked)
                self.classes.setItem(index, 0, item)
                policy = QComboBox()
                for value, title in POLICIES.items():
                    policy.addItem(title, value)
                policy.setCurrentIndex(policy.findData(default_policy(name)))
                self.classes.setCellWidget(index, 3, policy)
                count = QSpinBox()
                count.setRange(1, 100)
                count.setValue(5)
                self.classes.setCellWidget(index, 4, count)
            self.guard(self.refresh)

    def fill_combo(self, combo, values, title):
        selected = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for row in values:
            combo.addItem(title(row), row["id"])
        combo.setCurrentIndex(max(0, combo.findData(selected)))
        combo.blockSignals(False)

    def refresh(self):
        store = self.store()
        self.fill_combo(
            self.profiles,
            store.list("profiles"),
            lambda v: f"{v['created'][:19]} / {len(v['donors'])}개 참조",
        )
        self.fill_combo(
            self.batches,
            store.list("batches"),
            lambda v: f"{v['created'][:19]} / {v['split']} / {len(v['samples'])}장",
        )
        self.load_profile()
        self.load_batch()
        publications = store.list("publications")
        if publications and not self.output_path:
            self.output_path = str(
                store.directory("publications", publications[0]["id"])
                / publications[0]["project"]
            )
            self.open_button.setEnabled(True)

    def policies(self):
        return {
            str(
                self.classes.item(i, 0).data(Qt.ItemDataRole.UserRole)
            ): self.classes.cellWidget(i, 3).currentData()
            for i in range(self.classes.rowCount())
        }

    def load_profile(self):
        profile = (
            self.store().load("profiles", self.profiles.currentData())
            if self.profiles.currentData()
            else None
        )
        for i in range(self.classes.rowCount()):
            cid = self.classes.item(i, 0).data(Qt.ItemDataRole.UserRole)
            for col, split in ((1, "train"), (2, "val")):
                count = (
                    sum(
                        d["class_id"] == cid and d["split"] == split
                        for d in profile["donors"]
                    )
                    if profile
                    else 0
                )
                item = QTableWidgetItem(str(count))
                examples = (
                    [
                        d
                        for d in profile["donors"]
                        if d["class_id"] == cid and d["split"] == split
                    ]
                    if profile
                    else []
                )
                if examples:
                    item.setToolTip(
                        f"결함 면적 {min(d['pixels'] for d in examples)}~{max(d['pixels'] for d in examples)} px\n"
                        f"문맥 포함 너비 {min(d['box'][2] for d in examples)}~{max(d['box'][2] for d in examples)} px"
                    )
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.classes.setItem(i, col, item)
            if profile:
                combo = self.classes.cellWidget(i, 3)
                combo.setCurrentIndex(combo.findData(profile["policies"][str(cid)]))
        self.update_available()
        if profile:
            self._analysis = profile.get("analysis")
            self.update_summary()
            self.unlabelled.setChecked(profile["include_unlabelled"])
            if profile["notes"]:
                self.message.setText("\n".join(profile["notes"][:10]))

    def update_available(self):
        col = 1 if self.split.currentData() == "train" else 2
        context = (self.profiles.currentData(), self.split.currentData())
        available = set()
        for i in range(self.classes.rowCount()):
            count = self.classes.item(i, col)
            item = self.classes.item(i, 0)
            cid = item.data(Qt.ItemDataRole.UserRole)
            usable = bool(context[0] and count is not None and int(count.text()) > 0)
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | (
                    Qt.ItemFlag.ItemIsUserCheckable
                    if usable
                    else Qt.ItemFlag.NoItemFlags
                )
            )
            if usable:
                available.add(cid)
                if (
                    context != self._selection_context
                    or cid not in self._available_classes
                ):
                    item.setCheckState(Qt.CheckState.Checked)
                item.setToolTip("사용 가능한 실제 불량 참조가 있습니다.")
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setToolTip(
                    "이 분할의 참조가 없습니다. 정답 분석 상세에서 경로와 제외 사유를 확인하세요."
                )
        self._selection_context, self._available_classes = context, available

    def analyze(self):
        self.launch(
            "analyze",
            {
                "include_unlabelled": self.unlabelled.isChecked(),
                "policies": self.policies(),
            },
        )

    def variation_mode_changed(self):
        enabled = self.variation_mode.currentData() != "reference"
        for key in ("size_min", "size_max"):
            self.variation_inputs[key].setEnabled(enabled)
        self.shape_amount.setEnabled(enabled)

    def variation(self):
        settings = {
            "mode": self.variation_mode.currentData(),
            **{
                key: spin.value() / 100
                for key, spin in self.variation_inputs.items()
            },
            "shape_amount": self.shape_amount.value() / 100,
        }
        if settings["mode"] == "reference":
            # Disabled geometry controls must not prevent strength-only
            # generation (e.g. an unfinished min/max edit in another mode).
            settings.update(size_min=1.0, size_max=1.0, shape_amount=0.0)
        return variation_settings(settings)

    def generate(self, preview=False):
        key = self.profiles.currentData()
        if not key:
            raise ValueError("먼저 프로젝트 정답을 분석하세요.")
        profile = self.store().load("profiles", key)
        if (
            self.policies() != profile["policies"]
            or self.unlabelled.isChecked() != profile["include_unlabelled"]
        ):
            raise ValueError(
                "분석 설정이 바뀌었습니다. ‘프로젝트 정답 분석’을 다시 실행하세요."
            )
        counts = {
            str(self.classes.item(i, 0).data(Qt.ItemDataRole.UserRole)): 1
            if preview
            else self.classes.cellWidget(i, 4).value()
            for i in range(self.classes.rowCount())
            if self.classes.item(i, 0).checkState() == Qt.CheckState.Checked
        }
        if not counts:
            raise ValueError("참조가 있는 생성 클래스를 하나 이상 선택하세요.")
        self.launch(
            "generate",
            {
                "profile_id": key,
                "counts": counts,
                "split": self.split.currentData(),
                "seed": self.seed.value(),
                "tolerance": self.tolerance.value(),
                "allowed_path": self.allowed.text(),
                "variation": self.variation(),
            },
        )

    def browse_roi(self):
        path = QFileDialog.getOpenFileName(self, "허용 영역", "", "PNG (*.png)")[0]
        if path:
            self.allowed.setText(path)

    def load_batch(self):
        self.batch = (
            self.store().load("batches", self.batches.currentData())
            if self.batches.currentData()
            else None
        )
        selected = self.samples.currentRow()
        self.samples.blockSignals(True)
        self.samples.clear()
        if self.batch:
            for row in self.batch["samples"]:
                state = {"pending": "미검수", "approved": "승인", "rejected": "제외"}[
                    row["status"]
                ]
                self.samples.addItem(
                    QListWidgetItem(
                        f"{row['id']} · {self.batch['classes'][row['class_id']]} · {state}"
                    )
                )
            if self.batch["failures"]:
                self.message.setText(
                    "생성 제외:\n"
                    + "\n".join(
                        f"{self.batch['classes'][v['class_id']]}: {v['reason']}"
                        for v in self.batch["failures"][:8]
                    )
                )
        self.samples.setCurrentRow(min(max(0, selected), self.samples.count() - 1))
        self.samples.blockSignals(False)
        self.publish_button.setEnabled(
            bool(self.batch and self.batch["split"] == "train")
        )
        self.save_images_button.setEnabled(bool(self.batch and self.batch["samples"]))
        self.show_sample()

    def selected(self):
        if not self.batch or self.samples.currentRow() < 0:
            raise ValueError("생성 후보를 선택하세요.")
        return self.batch["samples"][self.samples.currentRow()]

    def show_sample(self):
        if not self.batch or self.samples.currentRow() < 0:
            for label in (self.original, self.generated, self.reference):
                label.setPixmap(QPixmap())
            self.details.clear()
            return
        row = self.selected()
        store = self.store()
        directory = store.directory("batches", self.batch["id"])
        images = {}
        for key in ("image", "original", "reference", "mask"):
            with Image.open(store.file(directory, row[key])) as image:
                images[key] = np.array(image)
        generated = images["image"]
        if generated.ndim == 2:
            generated = np.repeat(generated[..., None], 3, axis=2)
        if self.overlay.isChecked():
            generated = generated.copy()
            for cid in np.unique(images["mask"]):
                if cid in (0, 255):
                    continue
                color = QColor.fromHsv((int(cid) * 67 + 205) % 360, 180, 255)
                chosen = images["mask"] == cid
                generated[chosen] = np.rint(
                    0.6 * generated[chosen]
                    + 0.4 * np.array([color.red(), color.green(), color.blue()])
                ).astype(np.uint8)
        self.original.setPixmap(pixmap(images["original"]))
        self.generated.setPixmap(pixmap(generated))
        self.reference.setPixmap(pixmap(images["reference"]))
        variation = row["recipe"].get("variation")
        varied = (
            f"크기 {row['recipe']['scale']:.0%} · 강도 {row['recipe']['strength_sample']:.0%} · "
            if variation
            else ""
        )
        self.details.setText(
            f"{self.batch['split']} · {self.batch['classes'][row['class_id']]} · {POLICIES[row['policy']]} · "
            + varied
            + f"표면 차이 {row['recipe']['context_rmse']:.4f} · 위치 {row['recipe']['box']} · "
            f"원본 {images['image'].shape[1]}×{images['image'].shape[0]} / 휠 확대, 드래그 이동"
        )

    def focus_defect(self):
        box = self.selected()["recipe"]["box"]
        for view in (self.original, self.generated):
            view.focus_rect(*box)

    def review(self, status):
        row = self.selected()
        self.store().review(
            self.batch["id"], row["id"], status, expected_revision=row["revision"]
        )
        self.load_batch()

    def edit_mask(self):
        from widgets.segmentation_annotation import SegmentationAnnotationDialog

        row = self.selected()
        store, batch_id = self.store(), self.batch["id"]
        directory = store.directory("batches", batch_id)
        image_path = store.file(directory, row["image"])
        with Image.open(store.file(directory, row["mask"])) as mask:
            width, height = mask.size
            doc = MaskDocument.from_image(
                mask, width, height, len(self.batch["classes"])
            )

        revision = [row["revision"]]

        def save(payload):
            pixels = MaskDocument.from_payload(
                payload, width, height, doc.classes
            ).render()
            updated = store.review(
                batch_id,
                row["id"],
                "pending",
                mask=pixels,
                expected_revision=revision[0],
            )
            revision[0] = updated["revision"]

        dialog = SegmentationAnnotationDialog(
            str(image_path), self.batch["classes"], doc, save, self
        )
        dialog.exec()
        self.load_batch()

    def save_images(self):
        row = self.selected()
        scope = self.save_scope.currentData()
        ids = (
            [row["id"]]
            if scope == "selected"
            else [
                sample["id"]
                for sample in self.batch["samples"]
                if (
                    sample["status"] == "approved"
                    if scope == "approved"
                    else sample["status"] != "rejected"
                )
            ]
        )
        if not ids:
            raise ValueError("선택한 저장 범위에 후보가 없습니다.")
        folder = QFileDialog.getExistingDirectory(self, "새 이미지·정답 저장 폴더")
        if folder:
            self.launch(
                "save_images",
                {
                    "batch_id": self.batch["id"],
                    "sample_ids": ids,
                    "output_dir": folder,
                },
            )

    def publish(self):
        self.launch("publish", {"batch_id": self.batch["id"]})

    def open_batch(self):
        if not self.batch:
            raise ValueError("생성 기록을 선택하세요.")
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.store().directory("batches", self.batch["id"])))
        )

    def open_output(self):
        if self.output_path:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(Path(self.output_path).parent))
            )

    def launch(self, operation, values):
        if self._worker:
            raise ValueError("실행 중인 작업을 먼저 마치거나 중단하세요.")
        self._result = None
        self._operation = operation
        self._last_error = self._job_details = ""
        self._notice.close()
        if operation == "analyze":
            self.profiles.setCurrentIndex(-1)
            self._analysis = None
            self.update_summary()
        self._worker = DesktopJob(
            "seg_synthesis",
            {"project": project_view(self.project), "operation": operation, **values},
            self,
        )
        self._worker.event.connect(self.on_job_event)
        self._worker.failed.connect(self.message.setText)
        self._worker.completed.connect(lambda result: setattr(self, "_result", result))
        self._worker.finished.connect(self.finished)
        self.controls.setEnabled(False)
        self.review_controls.setEnabled(False)
        self.stop.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.message.setText("작업 중…")
        self._worker.start()

    def on_job_event(self, name, args):
        if name == "log_message":
            self.message.setText(str(args[0]))
        elif name == "progress_updated" and args[1]:
            self.progress.setValue(round(100 * args[0] / args[1]))

    def finished(self):
        worker = self._worker
        result = self._result or {
            "status": "failed",
            "error": worker.error or "작업 완료 결과를 받지 못했습니다.",
        }
        operation = self._operation
        self._worker = None
        worker.deleteLater()
        self.controls.setEnabled(True)
        self.review_controls.setEnabled(True)
        self.stop.setEnabled(False)
        output = result.get("output", {})
        if "analysis" in output:
            self._analysis = output["analysis"]
            self.update_summary()
        if result.get("status") != "completed":
            if worker.job_id:
                from core.job_manager import desktop_manager

                try:
                    log = desktop_manager().directory(worker.job_id) / "console.log"
                    self._job_details = (
                        str(log)
                        + "\n"
                        + log.read_text(encoding="utf-8", errors="replace")[-12000:]
                    )
                except OSError:
                    self._job_details = "실행 로그를 읽지 못했습니다."
            if result.get("status") == "cancelled":
                self.message.setText("작업이 중단되었습니다.")
                self.progress.setFormat("중단 — %p%")
            else:
                self.show_error(result.get("error") or "작업이 실패했습니다.")
            return
        try:
            self.refresh()
            for field, combo in (
                ("profile_id", self.profiles),
                ("batch_id", self.batches),
            ):
                if field in output:
                    index = combo.findData(output[field])
                    if index < 0:
                        raise ValueError(
                            "작업은 끝났지만 저장된 결과를 화면에서 읽지 못했습니다. 기록 폴더와 프로젝트 클래스를 확인하세요."
                        )
                    combo.setCurrentIndex(index)
            # An old profile may have been selected during refresh. Load the
            # newly returned profile explicitly even if its index did not change.
            if operation == "analyze":
                self.load_profile()
        except Exception as exc:  # noqa: BLE001 - do not turn a result-loading error into success
            self.show_error(str(exc))
            return
        self.progress.setValue(100)
        if "project_path" in output:
            self.output_path = output["project_path"]
            self.open_button.setEnabled(True)
            self.message.setText(
                f"학습 프로젝트 저장 완료: {self.output_path}\n앱의 ‘프로젝트 열기’에서 이 파일을 선택하세요. 원본 프로젝트는 그대로입니다."
            )
        elif "saved_images" in output:
            self.output_path = output["manifest_path"]
            self.open_button.setEnabled(True)
            self.message.setText(
                f"새 이미지 {output['saved_images']}장과 정답 PNG 저장 완료: {output['export_dir']}\n"
                "images에는 합성 사진, masks에는 클래스 번호 정답이 저장됩니다. 원본 파일은 그대로입니다."
            )
        elif "generated" in output:
            message = f"후보 {output['generated']}장 생성, {len(output['failures'])}장 제외. 결과와 정답을 검수한 뒤 승인하세요."
            if output["failures"]:
                message += "\n" + output["failures"][0]["reason"]
            if not output["generated"]:
                self.show_error(message)
            else:
                self.message.setText(message)
            self.generation_finished.emit(output["generated"])
        else:
            selected_split = self.split.currentData()
            available = sum(
                int(self.classes.item(i, 1 if selected_split == "train" else 2).text())
                for i in range(self.classes.rowCount())
            )
            message = f"정답 분석 완료: 참조 {output.get('donors', 0)}개. {selected_split.upper()} 사용 가능 {available}개."
            if available:
                message += " 선택된 클래스로 ‘클래스당 1장 미리 생성’을 누르세요."
            else:
                message += " 이 분할에는 참조가 없습니다. Train/Val을 확인하세요."
            self.message.setText(message)
