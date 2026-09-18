"""공통 프로세스로 합성하고 저장된 후보를 검수하는 데스크톱 화면."""
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QCheckBox, QListWidget, QListWidgetItem,
    QProgressBar, QSplitter, QScrollArea)

from core.desktop_jobs import DesktopJob
from core.job_manager import desktop_manager
from core.project import ProjectManager
from core.defect_generator import DefectParams, DefectType, DEFECT_INFO
from core.defect_workflow import settings_for, validate_settings, validate_output, candidates, candidate_ids
from core.defect_io import read_image
from core.image_display import display_rgb
from widgets.common import NoWheelSpinBox, NoWheelDoubleSpinBox, NoWheelComboBox
from webapp.storage import project_view


class DefectGenWidget(QWidget):
    generation_finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._worker = None
        self._run_project = None
        self._loading = False
        self._job_id = None
        self._job_result = None
        self._selected = set()
        self._rows = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Defect Gen - 생성 결과 검수"))
        note = QLabel("규칙 기반 표면 합성입니다. 후보를 확인한 뒤 선택한 결과를 저장하세요.\n"
                      "변경 마스크는 실제 픽셀 변화이며, 불량 정답 라벨은 별도 검수가 필요합니다.")
        note.setWordWrap(True)
        layout.addWidget(note)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.controls = QWidget()
        form = QFormLayout(self.controls)
        scroll.setWidget(self.controls)
        split.addWidget(scroll)
        self.source_edit = self._path(form, "정상 이미지 폴더", True)
        self.output_edit = self._path(form, "검수 결과 저장 폴더", True)
        self.roi_edit = self._path(form, "ROI 마스크 (원본과 동일 크기)", False)
        self.texture_edit = self._path(form, "참조 질감", False)
        self.type_checkboxes = {}
        for kind in DefectType:
            box = QCheckBox(DEFECT_INFO[kind]["name"])
            box.setChecked(kind == DefectType.SCRATCH)
            box.toggled.connect(self._remember)
            self.type_checkboxes[kind] = box
            form.addRow(box)
        for name, title, lo, hi, value, step in (
            ("intensity_spin", "강도", .01, 1, .5, .05),
            ("size_spin", "후보 사각형 면적 비율", .001, 1, .15, .01),
            ("feather_spin", "경계 부드러움", 0, 1, .25, .05),
            ("roughness_spin", "불규칙도", 0, 1, .65, .05),
            ("seed_spin", "시드 (-1: 무작위)", -1, 4294967295, 42, 1),
        ):
            widget = NoWheelDoubleSpinBox()
            widget.setDecimals(0 if name == "seed_spin" else 3)
            widget.setRange(lo, hi)
            widget.setValue(value)
            widget.setSingleStep(step)
            widget.valueChanged.connect(self._remember)
            setattr(self, name, widget)
            form.addRow(title, widget)
        self.count_spin = NoWheelSpinBox()
        self.count_spin.setRange(1, 100)
        self.per_image_spin = NoWheelSpinBox()
        self.per_image_spin.setRange(1, 100)
        for title, widget in (("이미지당 패턴 수", self.count_spin), ("원본당 생성 수", self.per_image_spin)):
            widget.valueChanged.connect(self._remember)
            form.addRow(title, widget)
        self.mix_check = QCheckBox("선택한 유형 혼합")
        self.mix_check.toggled.connect(self._remember)
        form.addRow(self.mix_check)
        self.settings_btn = QPushButton("설정 저장")
        self.settings_btn.clicked.connect(self._save_settings)
        self.preview_btn = QPushButton("첫 이미지 샘플 생성")
        self.preview_btn.clicked.connect(lambda: self._start_generation(preview=True))
        self.generate_btn = QPushButton("대량 후보 생성")
        self.generate_btn.clicked.connect(lambda: self._start_generation(preview=False))
        for button in (self.settings_btn, self.preview_btn, self.generate_btn):
            form.addRow(button)
        right = QWidget()
        content = QVBoxLayout(right)
        split.addWidget(right)
        split.setStretchFactor(1, 2)
        self.history = NoWheelComboBox()
        self.history.currentIndexChanged.connect(self._select_job)
        content.addWidget(self.history)
        row = QHBoxLayout()
        self.orig_image_label = QLabel("원본")
        self.defect_image_label = QLabel("합성 결과")
        self.mask_image_label = QLabel("실제 변경 마스크")
        for label in (self.orig_image_label, self.defect_image_label, self.mask_image_label):
            label.setMinimumSize(160, 220)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(label, 1)
        content.addLayout(row)
        self.applied_label = QLabel("")
        self.applied_label.setWordWrap(True)
        content.addWidget(self.applied_label)
        self.result_list = QListWidget()
        self.result_list.currentItemChanged.connect(self._show_result)
        self.result_list.itemChanged.connect(self._select_sample)
        content.addWidget(self.result_list, 1)
        buttons = QHBoxLayout()
        self.select_all_btn = QPushButton("표시된 결과 선택")
        self.select_all_btn.clicked.connect(self._select_all)
        self.save_btn = QPushButton("선택한 결과 그대로 저장")
        self.save_btn.clicked.connect(self._publish)
        buttons.addWidget(self.select_all_btn)
        buttons.addWidget(self.save_btn)
        content.addLayout(buttons)
        page_row = QHBoxLayout()
        self.page_spin = NoWheelSpinBox()
        self.page_spin.setRange(1, 1)
        self.page_spin.valueChanged.connect(self._load_candidates)
        page_row.addWidget(QLabel("후보 페이지 (48장씩)"))
        page_row.addWidget(self.page_spin)
        content.addLayout(page_row)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("")
        self.stop_btn = QPushButton("작업 중단")
        self.stop_btn.clicked.connect(self._stop_generation)
        self.stop_btn.setVisible(False)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.stop_btn)
        self.controls.setEnabled(False)
        self.save_btn.setEnabled(False)

    def _path(self, form, title, directory):
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        button = QPushButton("찾기")
        def browse():
            value = QFileDialog.getExistingDirectory(self, title, edit.text()) if directory else QFileDialog.getOpenFileName(
                self, title, edit.text(), "이미지 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")[0]
            if value:
                edit.setText(value)
        button.clicked.connect(browse)
        edit.textChanged.connect(self._remember)
        row.addWidget(edit, 1)
        row.addWidget(button)
        form.addRow(title, container)
        return edit

    def _get_params(self):
        return DefectParams(intensity=self.intensity_spin.value(), size_ratio=self.size_spin.value(),
            count=self.count_spin.value(), types=[kind for kind, box in self.type_checkboxes.items() if box.isChecked()],
            seed=None if self.seed_spin.value() < 0 else int(self.seed_spin.value()),
            feather=self.feather_spin.value(), roughness=self.roughness_spin.value(), mix_types=self.mix_check.isChecked())

    def _settings(self):
        params = self._get_params()
        return {"folder": self.source_edit.text().strip(), "output": self.output_edit.text().strip(),
                "roi": self.roi_edit.text().strip(), "texture": self.texture_edit.text().strip(),
                "per_image": self.per_image_spin.value(), "params": {**vars(params), "types": [t.value for t in params.types]}}

    def _remember(self, *_args):
        if not self._loading and self.project and hasattr(self, "mix_check"):
            self.project.defect_generation = self._settings()

    def _save_settings(self):
        try:
            self.project.defect_generation = validate_settings(self._settings())
            ProjectManager.save(self.project)
            self.progress_label.setText("합성 설정 저장 완료")
            return True
        except Exception as exc:
            self.progress_label.setText(f"설정 저장 실패: {exc}")
            return False

    def set_project(self, project):
        if self._worker is not None:
            return
        self.project = project
        self.controls.setEnabled(project is not None)
        if project is None:
            self.history.clear()
            self.result_list.clear()
            return
        self._loading = True
        try:
            cfg = settings_for(project)
            for key, edit in (("folder", self.source_edit), ("output", self.output_edit), ("roi", self.roi_edit), ("texture", self.texture_edit)):
                edit.setText(cfg[key])
            params = cfg["params"]
            for key, spin in (("intensity", self.intensity_spin), ("size_ratio", self.size_spin), ("feather", self.feather_spin),
                              ("roughness", self.roughness_spin), ("count", self.count_spin)):
                spin.setValue(params[key])
            self.seed_spin.setValue(-1 if params["seed"] is None else params["seed"])
            self.per_image_spin.setValue(cfg["per_image"])
            self.mix_check.setChecked(params["mix_types"])
            for kind, box in self.type_checkboxes.items():
                box.setChecked(kind.value in params["types"])
        finally:
            self._loading = False
        self._restore_jobs()

    def _restore_jobs(self, selected=None):
        self.history.blockSignals(True)
        self.history.clear()
        for job in desktop_manager().list(ProjectManager.get_active_filepath(self.project), limit=10000):
            if job["kind"] == "defects":
                self.history.addItem(f"{job['created_at'][:19]} | {job['status']}", job["id"])
        index = self.history.findData(selected) if selected else 0
        self.history.setCurrentIndex(max(0, index) if self.history.count() else -1)
        self.history.blockSignals(False)
        self._select_job()

    def _select_job(self, *_args):
        self._job_id = self.history.currentData()
        self._selected.clear()
        self.page_spin.blockSignals(True)
        self.page_spin.setValue(1)
        self.page_spin.blockSignals(False)
        self._load_candidates()

    def _load_candidates(self, *_args):
        self._loading = True
        try:
            self.result_list.clear()
            self._rows = []
            if self._job_id:
                directory = desktop_manager().directory(self._job_id)
                ids = candidate_ids(directory)
                self.page_spin.setMaximum(max(1, (len(ids) + 47) // 48))
                self._rows = candidates(directory, offset=(self.page_spin.value() - 1) * 48, limit=48)
                for sample in self._rows:
                    item = QListWidgetItem(f"{sample['id']} | {Path(sample['source']).name} | 시드 {sample['seed']}")
                    item.setData(Qt.ItemDataRole.UserRole, sample)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked if sample["id"] in self._selected else Qt.CheckState.Unchecked)
                    self.result_list.addItem(item)
            if self.result_list.count():
                self.result_list.setCurrentRow(0)
            else:
                for label in (self.orig_image_label, self.defect_image_label, self.mask_image_label):
                    label.clear()
        except Exception as exc:
            self.progress_label.setText(f"후보 복원 실패: {exc}")
        finally:
            self._loading = False
        self.save_btn.setEnabled(bool(self._selected) and self._worker is None)

    def _show_result(self, item, *_args):
        if item is None:
            return
        try:
            sample = item.data(Qt.ItemDataRole.UserRole)
            for key, label in (("original", self.orig_image_label), ("image", self.defect_image_label), ("mask", self.mask_image_label)):
                self._display_image(label, read_image(sample[key]))
            self.applied_label.setText(f"실제 변경 {sample['changed_fraction']:.2%} | 생성 {sample.get('generation_sec', 0):.3f}초 | "
                                      "선택한 후보의 저장된 픽셀을 그대로 사용")
        except Exception as exc:
            self.applied_label.setText(f"후보 읽기 실패: {exc}")

    def _display_image(self, label, image):
        pixels = display_rgb(image)
        h, w = pixels.shape[:2]
        qimage = QImage(pixels.data, w, h, pixels.strides[0], QImage.Format.Format_RGB888).copy()
        label.setPixmap(QPixmap.fromImage(qimage).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _select_sample(self, item):
        if self._loading:
            return
        sample_id = item.data(Qt.ItemDataRole.UserRole)["id"]
        if item.checkState() == Qt.CheckState.Checked:
            self._selected.add(sample_id)
        else:
            self._selected.discard(sample_id)
        self.save_btn.setEnabled(bool(self._selected) and self._worker is None)

    def _select_all(self):
        for index in range(self.result_list.count()):
            self.result_list.item(index).setCheckState(Qt.CheckState.Checked)

    def _launch(self, kind, payload):
        if self._worker is not None:
            return
        self._run_project = self.project
        self._job_result = None
        self._worker = DesktopJob(kind, payload, self)
        self._worker.event.connect(self._event)
        self._worker.completed.connect(self._completed)
        self._worker.failed.connect(lambda message: self.progress_label.setText(message))
        self._worker.finished.connect(self._finished)
        self.controls.setEnabled(False)
        self.history.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.stop_btn.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("작업 시작")
        self._worker.start()

    def _start_generation(self, preview=False):
        if not self.project or not self._save_settings():
            return
        try:
            validate_output(self.project, self.output_edit.text())
            self._launch("defects", {**self._settings(), "preview": preview, "project": project_view(self.project)})
        except Exception as exc:
            self.progress_label.setText(str(exc))

    def _publish(self):
        if not self._job_id or not self._selected:
            return
        self._launch("defect_publish", {"project": project_view(self.project), "source_directory": str(desktop_manager().directory(self._job_id)),
                     "sample_ids": sorted(self._selected), "output": self.output_edit.text().strip()})

    def _event(self, name, args):
        if name == "progress_updated":
            current, total = args
            self.progress_bar.setValue(round(current * 100 / max(total, 1)))
            self.progress_label.setText(f"{current}/{total}")

    def _completed(self, job):
        self._job_result = job

    def _finished(self):
        worker = self._worker
        self._worker = None
        self._run_project = None
        self.controls.setEnabled(True)
        self.history.setEnabled(True)
        self.stop_btn.setVisible(False)
        job = self._job_result
        if job:
            output = job.get("output", {})
            if worker.kind == "defects":
                self._restore_jobs(job["id"])
                self.progress_label.setText(f"{job['status']} | 생성 후보 {len(candidate_ids(desktop_manager().directory(job['id'])))}장. 결과 선택 후 저장 가능")
            else:
                self.progress_label.setText(f"{job['status']} | 저장 {output.get('saved', 0)}장 | {output.get('output_dir', '')}")
            if job.get("error"):
                self.progress_label.setText(job["error"])
        self.save_btn.setEnabled(bool(self._selected))
        worker.deleteLater()

    def _stop_generation(self):
        if self._worker is not None:
            self._worker.stop()
            self.progress_label.setText("중단 요청됨. 현재 저장의 안전한 종료 대기")
