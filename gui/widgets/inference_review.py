"""Sortable inference records with stable image identity and review-only thresholds."""
from dataclasses import asdict

from PySide6.QtCore import Qt, Signal, QSortFilterProxyModel, QSignalBlocker
from PySide6.QtGui import QStandardItem, QStandardItemModel, QColor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QCheckBox, QPushButton, QDoubleSpinBox, QSlider, QTableView, QAbstractItemView, QHeaderView, QSizePolicy)
from core.inference_review import review_record, source_class, finite, normalized_patchcore
from core.inference_timing import format_result_timing, format_runtime_stages


class ReviewFilter(QSortFilterProxyModel):
    def __init__(self, parent):
        super().__init__(parent)
        self.class_name = self.decision = self.search = ""
        self.selected_only = False

    def filterAcceptsRow(self, row, parent):
        model = self.sourceModel()
        def value(col):
            return model.index(row, col, parent).data() or ""
        return ((not self.class_name or value(2) == self.class_name) and
                (not self.decision or value(4) == self.decision) and
                (not self.search or self.search in str(value(3)).casefold()) and
                (not self.selected_only or model.item(row, 0).checkState() == Qt.CheckState.Checked))


class InferenceReview(QWidget):
    image_selected = Signal(str)
    threshold_changed = Signal()
    HEADERS = ["선택", "인덱스", "클래스", "파일명", "판정", "스코어", "추론 시간 (ms)"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.paths, self.raw, self.rows = [], {}, {}
        self.project = None
        self.saved_threshold = None
        self.score_normalized = False
        self._updating = False
        self._bulk = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.setStyleSheet("""
            QPushButton, QComboBox, QLineEdit, QDoubleSpinBox { padding: 4px 6px; min-height: 20px; font-size: 12px; }
            QCheckBox { font-size: 12px; }
            QHeaderView::section { padding: 6px 4px; font-size: 11px; }
            QTableView::item { padding: 3px 5px; }
        """)
        self.threshold_panel = QWidget()
        box = QVBoxLayout(self.threshold_panel)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(3)
        line = QHBoxLayout()
        self.override = QCheckBox("사용자 임계값")
        self.value = QDoubleSpinBox()
        self.value.setAccessibleName("Anomaly 임계값")
        self.value.setDecimals(6)
        self.value.setRange(-1e12, 1e12)
        self.value.setSingleStep(.01)
        self.value.setKeyboardTracking(False)
        self.reset = QPushButton("저장된 임계값 복원")
        line.addWidget(self.override)
        line.addWidget(self.value, 1)
        line.addWidget(self.reset)
        box.addLayout(line)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setMaximumHeight(16)
        self.slider.setAccessibleName("Anomaly 임계값 슬라이더")
        slider_row = QHBoxLayout()
        slider_row.addWidget(self.slider, 1)
        self.threshold_note = QLabel("스코어 ≥ 임계값이면 NG. 모델 파일은 변경하지 않습니다.")
        self.threshold_note.setWordWrap(False)
        self.threshold_note.setStyleSheet("font-size: 11px;")
        self.threshold_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        slider_row.addWidget(self.threshold_note, 2)
        box.addLayout(slider_row)
        layout.addWidget(self.threshold_panel)
        filters = QHBoxLayout()
        self.class_filter = QComboBox()
        self.class_filter.addItem("전체 클래스", "")
        self.decision_filter = QComboBox()
        for value in ("", "OK", "NG", "미보정", "ERROR", "대기"):
            self.decision_filter.addItem(value or "전체 판정", value)
        self.search = QLineEdit()
        self.search.setPlaceholderText("파일명 검색")
        self.search.setAccessibleName("결과 파일명 검색")
        filters.addWidget(self.class_filter)
        filters.addWidget(self.decision_filter)
        filters.addWidget(self.search, 1)
        layout.addLayout(filters)
        selection = QHBoxLayout()
        self.selected_only = QCheckBox("선택한 행만 보기")
        self.select_visible = QPushButton("표시 행 선택")
        self.clear_selection = QPushButton("선택 해제")
        selection.addWidget(self.selected_only)
        selection.addWidget(self.select_visible)
        selection.addWidget(self.clear_selection)
        layout.addLayout(selection)
        self.model = QStandardItemModel(0, len(self.HEADERS), self)
        self.model.setHorizontalHeaderLabels(self.HEADERS)
        self.proxy = ReviewFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for col, width in enumerate((38, 52, 72, 130, 55, 88, 105)):
            self.table.setColumnWidth(col, width)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.selectionModel().currentRowChanged.connect(self._current_changed)
        layout.addWidget(self.table, 1)
        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        for control in (self.class_filter, self.decision_filter):
            control.currentIndexChanged.connect(self._filter)
        self.search.textChanged.connect(self._filter)
        self.selected_only.toggled.connect(self._filter)
        self.select_visible.clicked.connect(self._select_visible)
        self.clear_selection.clicked.connect(self._clear_selection)
        self.model.itemChanged.connect(self._item_changed)
        self.override.toggled.connect(self._threshold_changed)
        self.value.valueChanged.connect(self._threshold_changed)
        self.slider.valueChanged.connect(self._slider_changed)
        self.reset.clicked.connect(lambda: self.override.setChecked(False))
        self.set_saved_threshold(None)

    @property
    def threshold(self):
        return self.value.value() if self.override.isChecked() else None

    def set_saved_threshold(self, value, *, normalized=False):
        self.score_normalized = normalized
        self.saved_threshold = float(value) if finite(value) else None
        with QSignalBlocker(self.override), QSignalBlocker(self.value):
            self.override.setChecked(False)
            self.value.setRange(0, 1) if normalized else self.value.setRange(-1e12, 1e12)
            self.value.setValue(self.saved_threshold if self.saved_threshold is not None else .5)
        self._threshold_changed()

    def set_context(self, paths, project=None):
        self.project = project
        self.paths, self.raw, self.rows = [], {}, {}
        # Structural row signals must reach the proxy; only suppress view selection callbacks.
        with QSignalBlocker(self.table.selectionModel()):
            self.model.removeRows(0, self.model.rowCount())
            for path in paths:
                self._add_path(path)
        self._update_classes()
        self._sync_threshold()
        self._filter()

    def _add_path(self, path):
        if path in self.rows:
            return
        row = self.model.rowCount()
        self.rows[path] = row
        self.paths.append(path)
        items = [QStandardItem() for _ in self.HEADERS]
        items[0].setCheckable(True)
        items[0].setData(path, Qt.ItemDataRole.UserRole)
        for col, value in ((1, row + 1), (2, source_class(path, self.project) or "—"),
                           (3, path.replace("\\", "/").rsplit("/", 1)[-1]), (4, "대기")):
            items[col].setData(value, Qt.ItemDataRole.DisplayRole)
        items[3].setToolTip(path)
        self.model.appendRow(items)

    def update_result(self, result):
        self.raw[result.image_path] = result
        self._updating = True
        try:
            self._add_path(result.image_path)
            row = self.rows[result.image_path]
            record = review_record(asdict(result), self.threshold)
            values = {2: record["source_class"] or source_class(result.image_path, self.project) or "—",
                      4: record["decision"], 5: float(result.score) if finite(result.score) else None,
                      6: round(result.inference_sec * 1000, 3) if finite(result.inference_sec) else None}
            for col, value in values.items():
                self.model.item(row, col).setData(value, Qt.ItemDataRole.DisplayRole)
            self.model.item(row, 4).setForeground(QColor(record.get("color", "#5590F0")))
            self.model.item(row, 4).setToolTip(record.get("error") or record.get("summary", ""))
            self.model.item(row, 6).setToolTip(
                f"{format_result_timing(result)}\n{format_runtime_stages(result).rstrip()}"
                + ("\n" + (result.details or {})["runtime_warning"] if (result.details or {}).get("runtime_warning") else ""))
        finally:
            self._updating = False
        if not self._bulk:
            self._update_classes()
            self._sync_threshold()
            self._count()

    def _update_classes(self):
        current = self.class_filter.currentData()
        values = sorted({self.model.item(i, 2).text() for i in range(self.model.rowCount())})
        with QSignalBlocker(self.class_filter):
            self.class_filter.clear()
            self.class_filter.addItem("전체 클래스", "")
            for name in values:
                self.class_filter.addItem(name, name)
            self.class_filter.setCurrentIndex(max(0, self.class_filter.findData(current)))

    def _sync_threshold(self):
        results = [r for r in self.raw.values() if r.task == "anomaly"]
        scored = [r for r in results if r.status != "error" and finite(r.score)]
        normalized = self.score_normalized or (bool(scored) and all(normalized_patchcore(asdict(r)) for r in scored))
        with QSignalBlocker(self.value):
            self.value.setRange(0, 1) if normalized else self.value.setRange(-1e12, 1e12)
        self.threshold_panel.setVisible(bool(results) or getattr(self.project, "task", "") == "anomaly" or self.saved_threshold is not None)
        saved = sorted({float(r.threshold) for r in results if finite(r.threshold)})
        if not self.override.isChecked():
            with QSignalBlocker(self.value):
                self.value.setValue(saved[0] if len(saved) == 1 else self.saved_threshold if self.saved_threshold is not None else .5)
        values = [float(r.score) for r in results if finite(r.score)] + saved + [self.value.value()]
        low, high = min(values), max(values)
        margin = max((high - low) * .1, .01)
        self._slider_range = (0.0, 1.0) if normalized else (low - margin, high + margin)
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(1000 * (self.value.value() - self._slider_range[0]) / (self._slider_range[1] - self._slider_range[0])))
        self.value.setEnabled(self.override.isChecked())
        self.slider.setEnabled(self.override.isChecked())
        note = ", ".join(f"{v:g}" for v in saved) or (f"{self.saved_threshold:g}" if self.saved_threshold is not None else "미보정")
        self.threshold_note.setText(("0~1 | " if normalized else "") + f"스코어 ≥ 임계값: NG | 저장값: {note}")
        meaning = "0에 가까울수록 정상 학습 데이터와 유사, 1에 가까울수록 멂. 불량 확률이 아닙니다.\n" if normalized else ""
        self.threshold_note.setToolTip(meaning + self.threshold_note.text() + "\n임계값 미만: OK, 이상: NG. 모델 파일과 저장된 원본 결과는 변경하지 않습니다.")
        self.slider.setToolTip(meaning + "임계값 미만: OK | 임계값 이상: NG")

    def _threshold_changed(self, *_):
        self._bulk = True
        try:
            for result in list(self.raw.values()):
                self.update_result(result)
        finally:
            self._bulk = False
        self._sync_threshold()
        self._count()
        self.threshold_changed.emit()

    def _slider_changed(self, value):
        low, high = self._slider_range
        self.value.setValue(low + (high - low) * value / 1000)

    def _filter(self, *_):
        self.proxy.class_name = self.class_filter.currentData() or ""
        self.proxy.decision = self.decision_filter.currentData() or ""
        self.proxy.search = self.search.text().casefold().strip()
        self.proxy.selected_only = self.selected_only.isChecked()
        self.proxy.invalidateFilter()
        self._count()

    def _count(self):
        checked = sum(self.model.item(i, 0).checkState() == Qt.CheckState.Checked for i in range(self.model.rowCount()))
        self.count_label.setText(f"표시 {self.proxy.rowCount()} / 전체 {self.model.rowCount()} | 선택 {checked}")

    def _item_changed(self, item):
        if not self._updating and item.column() == 0:
            self._filter()

    def _select_visible(self):
        indexes = [self.proxy.mapToSource(self.proxy.index(i, 0)).row() for i in range(self.proxy.rowCount())]
        for index in indexes:
            self.model.item(index, 0).setCheckState(Qt.CheckState.Checked)

    def _clear_selection(self):
        for i in range(self.model.rowCount()):
            self.model.item(i, 0).setCheckState(Qt.CheckState.Unchecked)

    def _current_changed(self, current, _previous):
        if current.isValid():
            row = self.proxy.mapToSource(current).row()
            self.image_selected.emit(self.model.item(row, 0).data(Qt.ItemDataRole.UserRole))

    def select_path(self, path):
        if path in self.rows:
            index = self.proxy.mapFromSource(self.model.index(self.rows[path], 1))
            if index.isValid():
                self.table.setCurrentIndex(index)
                self.table.scrollTo(index)
