"""Data Gen: user-defined real defect learning, generation and persistent review."""
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QTabWidget, QSplitter, QScrollArea, QTableWidget, QTableWidgetItem, QProgressBar, QGroupBox)
from core.datagen_store import DataGenStore, mutate
from core.desktop_jobs import DesktopJob
from core.job_manager import desktop_manager
from webapp.storage import project_view
from widgets.datagen_editor import MaskEditor, Comparison


class DefectGenWidget(QWidget):
    generation_finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._worker = None
        self._job_result = None
        self.data = {"items": [], "images": [], "datasets": [], "models": [], "samples": []}
        layout = QVBoxLayout(self)
        title = QHBoxLayout()
        title.addWidget(QLabel("Data Gen | 실제 불량 데이터 학습과 생성"), 1)
        self.legacy_btn = self.button("기존 규칙 증강 도구", self.show_legacy)
        title.addWidget(self.legacy_btn)
        layout.addLayout(title)
        row = QHBoxLayout()
        row.addWidget(QLabel("학습 항목"))
        self.items = QComboBox()
        self.items.currentIndexChanged.connect(self.select_item)
        row.addWidget(self.items, 1)
        self.summary = QLabel()
        row.addWidget(self.summary)
        layout.addLayout(row)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.build_data()
        self.build_training()
        self.build_generation()
        self.build_review()
        self.progress = QProgressBar()
        self.message = QLabel("프로젝트를 선택하세요")
        self.message.setWordWrap(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.message)
        self.stop = self.button("작업 중단", lambda: self._worker.stop() if self._worker else None)
        self.stop.setEnabled(False)
        layout.addWidget(self.stop)
        self.tabs.setEnabled(False)
        self.legacy = None

    def button(self, title, callback):
        button = QPushButton(title)
        button.clicked.connect(lambda _checked=False: self.guard(callback))
        return button

    def guard(self, callback):
        try:
            return callback()
        except Exception as exc:
            self.message.setText(str(exc))

    def tab(self, title):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        self.tabs.addTab(scroll, title)
        return layout

    def path(self, form, label, directory=False):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        def browse():
            value = QFileDialog.getExistingDirectory(self, label, edit.text()) if directory else QFileDialog.getOpenFileName(self, label, edit.text())[0]
            if value:
                edit.setText(value)
        layout.addWidget(edit, 1)
        layout.addWidget(self.button("찾기", browse))
        form.addRow(label, widget)
        return edit

    def combo(self, options):
        widget = QComboBox()
        for title, value in options:
            widget.addItem(title, value)
        return widget

    def build_data(self):
        layout = self.tab("1 불량 데이터")
        form = QFormLayout()
        self.item_name = QLineEdit()
        self.class_name = QComboBox()
        form.addRow("학습 항목명", self.item_name)
        form.addRow("검사 클래스 연결", self.class_name)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        for title, callback in (("학습 항목 추가", lambda: self.save_item(False)), ("이름과 클래스 저장", lambda: self.save_item(True)),
                                ("항목 보관", lambda: self.save_item(True, True))):
            buttons.addWidget(self.button(title, callback))
        layout.addLayout(buttons)
        form = QFormLayout()
        self.source = self.path(form, "실제 이미지 폴더", True)
        self.mask_folder = self.path(form, "마스크 폴더 (선택)", True)
        self.role = self.combo((("실제 불량", "defect"), ("실제 정상", "normal")))
        self.split = self.combo((("Train", "train"), ("Val", "val")))
        self.group = QLineEdit()
        self.group.setPlaceholderText("같은 개체와 로트는 같은 그룹")
        form.addRow("데이터 종류", self.role)
        form.addRow("생성 모델 분할", self.split)
        form.addRow("개체 / 생산 로트 그룹", self.group)
        layout.addLayout(form)
        layout.addWidget(self.button("실제 데이터 가져오기", self.import_images))
        split = QSplitter()
        self.images = QListWidget()
        self.images.setIconSize(QSize(112, 80))
        self.images.currentItemChanged.connect(lambda *_: self.guard(self.load_image))
        self.editor = MaskEditor()
        split.addWidget(self.images)
        split.addWidget(self.editor)
        split.setStretchFactor(1, 4)
        layout.addWidget(split)
        layout.addWidget(self.button("불량 마스크 검수 저장", self.save_mask))
        layout.addWidget(self.button("검수 데이터를 새 학습 버전으로 고정", lambda: self.action("freeze", item_id=self.item_id())))

    def build_training(self):
        layout = self.tab("2 불량 학습")
        note = QLabel("최소 Val Loss를 자동 Best로 보존합니다. 품질 Best는 생성 샘플을 검수한 후 별도로 지정합니다.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.base = self.path(form, "사전학습 Inpainting 모델 폴더", True)
        self.prompt = QLineEdit()
        form.addRow("학습할 불량과 표면 설명", self.prompt)
        layout.addLayout(form)
        advanced = QGroupBox("고급 학습 설정")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        form = QFormLayout(advanced)
        self.training = {}
        for key, title, lo, hi, value, decimals in (("steps", "학습 Step", 1, 100000, 500, 0),
                ("validate_every", "검증 주기 Step", 1, 10000, 50, 0), ("resolution", "해상도 (64 배수)", 256, 1024, 512, 0),
                ("rank", "LoRA rank", 1, 128, 8, 0), ("learning_rate", "학습률", .000001, .1, .0001, 6),
                ("patience", "조기 종료 Patience (0: 사용 안 함)", 0, 1000, 0, 0),
                ("min_delta", "조기 종료 최소 개선 폭", 0, 100, 0, 6)):
            spin = QDoubleSpinBox() if decimals else QSpinBox()
            if decimals:
                spin.setDecimals(decimals)
            spin.setRange(lo, hi)
            spin.setValue(value)
            self.training[key] = spin
            form.addRow(title, spin)
        self.precision = self.combo((("FP32", "fp32"), ("FP16", "fp16"), ("BF16", "bf16")))
        form.addRow("정밀도", self.precision)
        layout.addWidget(advanced)
        layout.addWidget(self.button("학습 설정 저장", self.save_training_settings))
        layout.addWidget(self.button("새 불량 학습", self.train))
        self.models = QComboBox()
        self.models.currentIndexChanged.connect(self.select_model)
        layout.addWidget(self.models)
        self.best_label = QLabel()
        layout.addWidget(self.best_label)
        self.checkpoints = QTableWidget(0, 6)
        self.checkpoints.setHorizontalHeaderLabels(["Step", "Epoch", "Train Loss", "Val Loss", "불량 Val", "배경 Val"])
        self.checkpoints.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.checkpoints)
        self.quality_reason = QLineEdit()
        self.quality_reason.setPlaceholderText("고정 생성 샘플의 품질 검수 근거")
        layout.addWidget(self.quality_reason)
        layout.addWidget(self.button("선택 체크포인트를 품질 Best로 지정", self.quality))
        layout.addWidget(self.button("Last에서 학습 재개", lambda: self.launch("train", {"resume_model": self.models.currentData()})))
        note = QLabel("CUDA GPU와 Data Gen 전용 환경이 필요합니다. 설치와 실제 GPU 검증 절차는 docs/DATA_GEN.md를 확인하세요.")
        note.setWordWrap(True)
        layout.addWidget(note)

    def build_generation(self):
        layout = self.tab("3 생성")
        form = QFormLayout()
        self.generation_models = QComboBox()
        self.normals = QComboBox()
        self.normals.currentIndexChanged.connect(lambda *_: self.guard(self.load_normal))
        form.addRow("학습한 불량 모델", self.generation_models)
        form.addRow("실제 정상 이미지", self.normals)
        self.allowed = self.path(form, "허용 영역 (비우면 요청 영역만 허용)")
        self.count = QSpinBox()
        self.count.setRange(1, 100)
        self.count.setValue(10)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        form.addRow("대량 생성 수", self.count)
        form.addRow("시드", self.seed)
        layout.addLayout(form)
        self.generation_editor = MaskEditor("생성 요청 영역")
        layout.addWidget(self.generation_editor)
        row = QHBoxLayout()
        row.addWidget(self.button("샘플 생성", lambda: self.generate(1)))
        row.addWidget(self.button("대량 생성", lambda: self.generate(self.count.value())))
        layout.addLayout(row)
        layout.addWidget(QLabel("생성 후 검수 탭에서 저장된 결과를 확인하세요."))

    def build_review(self):
        layout = self.tab("4 검수")
        self.filter = self.combo((("미검수", "pending"), ("승인", "approved"), ("제외", "rejected"), ("전체", "all")))
        self.filter.currentIndexChanged.connect(self.load_samples)
        layout.addWidget(self.filter)
        split = QSplitter()
        self.samples = QListWidget()
        self.samples.setIconSize(QSize(112, 80))
        self.samples.currentItemChanged.connect(lambda *_: self.guard(self.load_sample))
        split.addWidget(self.samples)
        right = QWidget()
        content = QVBoxLayout(right)
        self.comparison = Comparison()
        self.review_editor = MaskEditor("검수 불량 마스크 (요청 영역은 초안)")
        content.addWidget(self.comparison)
        content.addWidget(self.review_editor)
        self.layer = self.combo((("생성 결과", "image"), ("허용 영역", "allowed"), ("생성 요청", "requested"), ("실제 변경", "changed")))
        self.layer.currentIndexChanged.connect(lambda *_: self.guard(self.show_layer))
        content.addWidget(self.layer)
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("검수 사유")
        content.addWidget(self.reason)
        row = QHBoxLayout()
        for title, status in (("현재 불량 라벨로 승인", "approved"), ("제외", "rejected"), ("미검수로 되돌리기", "pending")):
            row.addWidget(self.button(title, lambda value=status: self.review(value)))
        content.addLayout(row)
        split.addWidget(right)
        split.setStretchFactor(1, 4)
        layout.addWidget(split)
        form = QFormLayout()
        self.output = self.path(form, "새 학습 버전 저장 폴더", True)
        layout.addLayout(form)
        layout.addWidget(self.button("승인본을 새 학습 버전으로 저장", self.publish))
        label = QLabel("검수된 이미지와 불량 마스크, 클래스 연결을 별도 train 버전으로 저장합니다. PatchCore 정상 학습에는 사용하지 않습니다.")
        label.setWordWrap(True)
        layout.addWidget(label)

    @property
    def _run_project(self):
        # MainWindow also checks this while a finished signal is still queued.
        if self._worker is not None or self.legacy is not None and self.legacy._worker is not None:
            return self.project
        return None

    def set_project(self, project):
        if self._run_project is not None:
            return
        self.project = project
        self.tabs.setEnabled(project is not None)
        self.class_name.clear()
        self.class_name.addItem("미연결: 생성 검수만 가능", "")
        if project:
            for name in project.data.class_names:
                self.class_name.addItem(name, name)
            self.output.setText(str(Path(project.project_dir) / "synthetic"))
            self.refresh()
            saved = self.data.get("settings", {})
            self.base.setText(saved.get("base_model", ""))
            self.prompt.setText(saved.get("prompt", ""))
            for key, widget in self.training.items():
                if key in saved:
                    widget.setValue(saved[key])
            self.precision.setCurrentIndex(max(0, self.precision.findData(saved.get("precision", "fp32"))))
            self.message.setText("학습 항목을 추가하거나 선택하세요")
        else:
            self.items.clear()
        if self.legacy:
            self.legacy.set_project(project)

    def refresh(self):
        self.data = DataGenStore(self.project).state()
        selected = self.items.currentData()
        self.items.blockSignals(True)
        self.items.clear()
        for item in self.data["items"]:
            if not item["archived"]:
                self.items.addItem(item["name"], item["id"])
        index = self.items.findData(selected)
        self.items.setCurrentIndex(index if index >= 0 else 0)
        self.items.blockSignals(False)
        self.select_item()

    def item_id(self):
        if not self.items.currentData():
            raise ValueError("학습 항목을 추가하거나 선택하세요")
        return self.items.currentData()

    def select_item(self, *_):
        key = self.items.currentData()
        item = next((v for v in self.data["items"] if v["id"] == key), None)
        if item:
            self.item_name.setText(item["name"])
            self.class_name.setCurrentIndex(max(0, self.class_name.findData(item["class_name"])))
        images = [v for v in self.data["images"] if v["item_id"] == key]
        self.summary.setText(f"불량 {sum(v['role'] == 'defect' for v in images)} / 정상 {sum(v['role'] == 'normal' for v in images)}")
        selected_image = self.images.currentItem().data(Qt.ItemDataRole.UserRole) if self.images.currentItem() else None
        selected_normal = self.normals.currentData()
        self.images.clear()
        self.normals.clear()
        for row in images:
            if row["role"] == "normal":
                self.normals.addItem(Path(row["source"]).name, row["id"])
                continue
            entry = QListWidgetItem(QIcon(str(DataGenStore(self.project).file("images", row["id"]))),
                f"{row['split']} / {row['group']}\n" + ("마스크 저장됨" if row["mask"] else "마스크 필요"))
            entry.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.images.addItem(entry)
            if row["id"] == selected_image:
                self.images.setCurrentItem(entry)
        if not self.images.currentItem() and self.images.count():
            self.images.setCurrentRow(0)
        if self.normals.findData(selected_normal) >= 0:
            self.normals.setCurrentIndex(self.normals.findData(selected_normal))
        for combo in (self.models, self.generation_models):
            selected = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for model in self.data["models"]:
                if model["item_id"] == key:
                    combo.addItem(f"{model['id'][:8]} / {model['status']}", model["id"])
            combo.setCurrentIndex(max(0, combo.findData(selected)))
            combo.blockSignals(False)
        self.select_model()
        self.load_samples()

    def action(self, action, **values):
        desktop_manager().require_idle()
        result = mutate(self.project, action, values)
        self.refresh()
        self.message.setText("저장 완료" + (f" | {result['output']}" if isinstance(result, dict) and result.get("output") else ""))
        return result

    def save_item(self, update, archived=False):
        self.action("item", name=self.item_name.text(), class_name=self.class_name.currentData() or "",
                    item_id=self.item_id() if update else None, archived=archived)

    def import_images(self):
        self.action("import", item_id=self.item_id(), folder=self.source.text(), masks=self.mask_folder.text(),
                    role=self.role.currentData(), split=self.split.currentData(), group=self.group.text())

    def load_image(self):
        entry = self.images.currentItem()
        if not entry:
            return
        key = entry.data(Qt.ItemDataRole.UserRole)
        row = next(v for v in self.data["images"] if v["id"] == key)
        store = DataGenStore(self.project)
        self.editor.load(store.file("images", key), store.file("images", key, "mask") if row["mask"] else None)

    def save_mask(self):
        if not self.images.currentItem():
            raise ValueError("실제 불량 이미지를 선택하세요")
        self.action("mask", image_id=self.images.currentItem().data(Qt.ItemDataRole.UserRole), mask=self.editor.png())

    def select_model(self, *_):
        model = next((v for v in self.data["models"] if v["id"] == self.models.currentData()), None)
        self.checkpoints.setRowCount(0)
        if not model:
            self.best_label.setText("학습 모델 없음. 실제 Train과 Val 데이터를 고정한 후 학습하세요.")
            return
        self.best_label.setText(f"Best Val: {(model.get('best_val_loss') or '미선정')[:8]} / Best Quality: {(model.get('best_quality') or '미선정')[:8]}")
        for row in model["checkpoints"]:
            index = self.checkpoints.rowCount()
            self.checkpoints.insertRow(index)
            for column, key in enumerate(("step", "epoch", "train_loss", "val_loss", "val_defect_loss", "val_background_loss")):
                value = row[key]
                entry = QTableWidgetItem("미측정" if value is None else str(round(value, 6)))
                entry.setData(Qt.ItemDataRole.UserRole, row["id"])
                self.checkpoints.setItem(index, column, entry)

    def save_training_settings(self):
        cfg = {key: widget.value() for key, widget in self.training.items()}
        cfg.update(base_model=self.base.text(), prompt=self.prompt.text(), precision=self.precision.currentData())
        self.action("settings", values=cfg)

    def train(self):
        datasets = [v for v in self.data["datasets"] if v["item_id"] == self.item_id()]
        if not datasets:
            raise ValueError("실제 Train과 Val 데이터를 새 학습 버전으로 고정하세요")
        cfg = {key: widget.value() for key, widget in self.training.items()}
        cfg.update(base_model=self.base.text(), prompt=self.prompt.text(), precision=self.precision.currentData())
        self.launch("train", {"dataset_id": datasets[-1]["id"], "config": cfg})

    def quality(self):
        row = self.checkpoints.currentRow()
        if row < 0:
            raise ValueError("체크포인트를 선택하세요")
        self.action("quality", model_id=self.models.currentData(), checkpoint=self.checkpoints.item(row, 0).data(Qt.ItemDataRole.UserRole), reason=self.quality_reason.text())

    def load_normal(self):
        if self.project and self.normals.currentData():
            self.generation_editor.load(DataGenStore(self.project).file("images", self.normals.currentData()))

    def generate(self, count):
        self.launch("generate", {"model_id": self.generation_models.currentData(), "image_id": self.normals.currentData(),
                    "requested": self.generation_editor.png(), "allowed": self.allowed.text() or None, "count": count, "seed": self.seed.value()})

    def load_samples(self, *_):
        selected = self.samples.currentItem().data(Qt.ItemDataRole.UserRole) if self.samples.currentItem() else None
        self.samples.clear()
        for sample in self.data["samples"]:
            if sample["item_id"] != self.items.currentData() or self.filter.currentData() not in {"all", sample["review"]["status"]}:
                continue
            entry = QListWidgetItem(QIcon(str(DataGenStore(self.project).file("samples", sample["id"]))),
                                    f"{sample['id'][:8]} / {sample['review']['status']}")
            entry.setData(Qt.ItemDataRole.UserRole, sample["id"])
            self.samples.addItem(entry)
            if sample["id"] == selected:
                self.samples.setCurrentItem(entry)
        if not self.samples.currentItem() and self.samples.count():
            self.samples.setCurrentRow(0)

    def selected_sample(self):
        if not self.samples.currentItem():
            raise ValueError("생성 후보를 선택하세요")
        return self.samples.currentItem().data(Qt.ItemDataRole.UserRole)

    def load_sample(self):
        if not self.samples.currentItem():
            return
        key = self.selected_sample()
        store = DataGenStore(self.project)
        sample = store.record("samples", key)
        self.comparison.load(store.file("samples", key, "original"), store.file("samples", key))
        self.review_editor.load(store.file("samples", key), store.file("samples", key, "reviewed" if sample["review"]["mask"] else "requested"))
        self.reason.setText(sample["review"].get("reason", ""))

    def show_layer(self):
        key = self.selected_sample()
        store = DataGenStore(self.project)
        self.comparison.load(store.file("samples", key, "original"), store.file("samples", key, self.layer.currentData()))

    def review(self, status):
        self.action("review", sample_id=self.selected_sample(), status=status, reason=self.reason.text(),
                    mask=self.review_editor.png() if status == "approved" else None)

    def publish(self):
        ids = [v["id"] for v in self.data["samples"] if v["item_id"] == self.item_id() and v["review"]["status"] == "approved"]
        self.action("publish", sample_ids=ids, output=self.output.text())

    def launch(self, operation, values):
        if not self.project or self._worker:
            return
        self._worker = DesktopJob("datagen_" + operation, {**values, "project": project_view(self.project)}, self)
        self._job_result = None
        self._worker.event.connect(self._event)
        self._worker.completed.connect(lambda job: setattr(self, "_job_result", job))
        self._worker.failed.connect(self.message.setText)
        self._worker.finished.connect(self.finished)
        self.tabs.setEnabled(False)
        self.items.setEnabled(False)
        self.legacy_btn.setEnabled(False)
        self.stop.setEnabled(True)
        self.message.setText("별도 프로세스에서 작업을 시작합니다")
        self._worker.start()

    def _event(self, name, args):
        if name == "progress_updated":
            self.progress.setValue(round(args[0] * 100 / max(1, args[1])))
            self.message.setText(f"{args[0]} / {args[1]}")
        elif name == "log_message":
            self.message.setText(str(args[0]))
        elif name == "datagen_checkpoint":
            self.refresh()
            self.message.setText(f"체크포인트 저장 | Step {args[0]['step']} | Val Loss {args[0]['val_loss']}")

    def finished(self):
        worker = self._worker
        self._worker = None
        self.tabs.setEnabled(True)
        self.items.setEnabled(True)
        self.legacy_btn.setEnabled(True)
        self.stop.setEnabled(False)
        self.refresh()
        job = self._job_result
        self.message.setText((job.get("error") or job["status"]) if job else worker.error)
        worker.deleteLater()

    def show_legacy(self):
        if self.legacy is None:
            from widgets.legacy_defect_gen_widget import DefectGenWidget as Legacy
            self.legacy = Legacy(self)
            self.tabs.addTab(self.legacy, "기존 규칙 증강")
            self.legacy.set_project(self.project)
        self.tabs.setCurrentWidget(self.legacy)
