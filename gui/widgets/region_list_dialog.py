"""데이터셋 전체의 클래스 영역(blob·박스)을 한 줄씩 보여주는 목록.

┌ 분할 ▾ ┐┌ 클래스 ▾ ┐ [ ] 학습 크기 1px 미만만        입력 512px 기준
┌──────┬──────────┬────────┬────────┬────────┬──────────┬──────────┐
│ 분할  │ 이미지     │ 클래스   │ 면적 px │ 굵기 px │ 학습 면적 │ 학습 굵기   │
│ train│ a.png    │ scratch│ 600    │ 2      │ 37.5     │ 0.50 ⚠   │
└──────┴──────────┴────────┴────────┴────────┴──────────┴──────────┘
[작업실에서 열기] [클래스 변경…] [영역 삭제]                     [닫기]

더블클릭하면 작업실이 그 영역을 확대해 선택한 상태로 열린다. 클래스 변경·삭제는 선택한 영역만
바꾸고(이미지의 다른 영역은 그대로), 저장 전 원본을 백업한다.
"""

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableView, QVBoxLayout,
)

from core.dataset_regions import training_footprint, training_scale
from widgets.common import NoWheelComboBox

COLUMNS = ("분할", "이미지", "클래스", "면적 px", "굵기 px", "크기 px", "학습 면적 px", "학습 굵기 px", "위치")


def region_rows(project, records):
    """스캔 기록 → 영역 행. 학습 입력 크기 기준 면적·폭은 현재 학습 설정으로 계산한다."""
    rows = []
    for record in records:
        size = record.get("size")
        if not size or record.get("error"):
            continue
        scale = training_scale(project, size)
        for item in record.get("region_items") or ():
            area, short = training_footprint(item, scale)
            rows.append({
                "split": record["split"], "path": record["path"], "name": record["name"],
                "item": item, "train_area": area, "train_short": short,
                "small": area < 1 or short < 1,
            })
    return rows


class RegionTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        item, column = row["item"], index.column()
        x, y, w, h = item["bbox"]
        if role == Qt.ItemDataRole.UserRole:  # 정렬 기준 (숫자 열은 숫자로)
            return (row["split"], row["name"], item["class"], item["area"], item.get("width", 1), w * h,
                    row["train_area"], row["train_short"], (y, x))[column]
        if role == Qt.ItemDataRole.DisplayRole:
            warn = " ⚠" if row["small"] and column in (6, 7) else ""
            return (row["split"].upper(), row["name"], item["class"], f"{item['area']:,}",
                    str(item.get("width", 1)), f"{w}×{h}",
                    f"{row['train_area']:.2f}{warn}", f"{row['train_short']:.2f}{warn}", f"{x}, {y}")[column]
        if role == Qt.ItemDataRole.ForegroundRole and row["small"] and column in (6, 7):
            return QBrush(QColor("#E0A030"))
        if role == Qt.ItemDataRole.TextAlignmentRole and column in (3, 4, 5, 6, 7):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole and column == 4:
            return "영역의 실제 굵기(가장 큰 내접원의 지름). 비스듬한 흠집은 bbox가 커도 굵기는 몇 px입니다."
        if role == Qt.ItemDataRole.ToolTipRole and column in (6, 7):
            return ("학습 입력 크기로 줄였을 때의 면적과 굵기입니다. 1px 미만이면 라벨에는 남지만 "
                    "모델이 배우기 어렵습니다. 입력 크기를 키우거나 클래스 가중치 Sqrt를 사용하세요.")
        return None


class RegionFilter(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.split = self.class_name = ""
        self.small_only = False
        self.setSortRole(Qt.ItemDataRole.UserRole)

    def set_filter(self, split, class_name, small_only):
        if hasattr(self, "beginFilterChange"):  # Qt 6.10+: invalidateRowsFilter는 폐기 예정
            self.beginFilterChange()
            self.split, self.class_name, self.small_only = split, class_name, small_only
            self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
        else:
            self.split, self.class_name, self.small_only = split, class_name, small_only
            self.invalidateRowsFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        row = self.sourceModel().rows[source_row]
        return ((not self.split or row["split"] == self.split)
                and (not self.class_name or row["item"]["class"] == self.class_name)
                and (not self.small_only or row["small"]))


class RegionListDialog(QDialog):
    """owner(DatasetWidget)가 제공: project, _dataset_index, dataset_index_updated 시그널,
    open_region(row), edit_regions(rows, class_id | None)."""

    def __init__(self, owner, region_classes):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("영역 목록 · 이미지 안의 클래스 영역")
        self.resize(980, 620)
        layout = QVBoxLayout(self)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("분할:"))
        self.split_combo = NoWheelComboBox()
        for label, value in (("전체", ""), ("TRAIN", "train"), ("VAL", "val"), ("TEST", "test")):
            self.split_combo.addItem(label, value)
        filters.addWidget(self.split_combo)
        filters.addWidget(QLabel("클래스:"))
        self.class_combo = NoWheelComboBox()
        self.class_combo.addItem("전체", "")
        for name in region_classes:
            self.class_combo.addItem(name, name)
        filters.addWidget(self.class_combo)
        self.small_check = QCheckBox("학습 크기 1px 미만만")
        filters.addWidget(self.small_check)
        filters.addStretch(1)
        self.scale_label = QLabel()
        self.scale_label.setStyleSheet("color: #8A92A4;")
        filters.addWidget(self.scale_label)
        layout.addLayout(filters)

        self.model = RegionTableModel(self)
        self.proxy = RegionFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(7, Qt.SortOrder.AscendingOrder)  # 학습 크기에서 가장 얇은 영역부터
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.doubleClicked.connect(lambda _index: self._open())
        layout.addWidget(self.table, 1)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        buttons = QHBoxLayout()
        self.open_btn = QPushButton("작업실에서 열기")
        self.open_btn.clicked.connect(self._open)
        self.reclass_btn = QPushButton("클래스 변경…")
        self.reclass_btn.clicked.connect(self._reclass)
        self.delete_btn = QPushButton("영역 삭제")
        self.delete_btn.setStyleSheet("QPushButton { color: #E05555; }")
        self.delete_btn.clicked.connect(self._delete)
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        for button in (self.open_btn, self.reclass_btn, self.delete_btn):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        for combo in (self.split_combo, self.class_combo):
            combo.currentIndexChanged.connect(self._apply_filter)
        self.small_check.toggled.connect(self._apply_filter)
        self.table.selectionModel().selectionChanged.connect(self._update_buttons)
        owner.dataset_index_updated.connect(self.reload)
        self.region_classes = list(region_classes)
        self.reload()

    # ── 데이터 ──
    def reload(self):
        index = getattr(self.owner, "_dataset_index", None) or {"images": []}
        project = self.owner.project
        self.model.set_rows(region_rows(project, index["images"]))
        size = project.training.input_size
        self.scale_label.setText(f"학습 입력 {size}px 기준 (학습 설정)")
        self._apply_filter()

    def _apply_filter(self, *_):
        self.proxy.set_filter(self.split_combo.currentData(), self.class_combo.currentData(),
                              self.small_check.isChecked())
        shown = self.proxy.rowCount()
        small = sum(row["small"] for row in self.model.rows)
        self.summary.setText(
            f"영역 {shown:,}개 표시 / 전체 {len(self.model.rows):,}개 · 학습 크기 1px 미만 {small:,}개. "
            "더블클릭하면 작업실이 그 영역을 확대해 선택한 상태로 열립니다.")
        self._update_buttons()

    def selected_rows(self):
        rows = {self.proxy.mapToSource(index).row() for index in self.table.selectionModel().selectedRows()}
        return [self.model.rows[row] for row in sorted(rows)]

    def _update_buttons(self, *_):
        count = len(self.selected_rows())
        self.open_btn.setEnabled(count == 1)
        self.reclass_btn.setEnabled(count > 0 and len(self.region_classes) > 1)
        self.delete_btn.setEnabled(count > 0)

    # ── 동작 ──
    def _open(self):
        rows = self.selected_rows()
        if len(rows) == 1:
            self.owner.open_region(rows[0])

    def _reclass(self):
        from PySide6.QtWidgets import QInputDialog

        rows = self.selected_rows()
        if not rows:
            return
        current = rows[0]["item"]["class"]
        options = [name for name in self.region_classes if name != current] or self.region_classes
        name, accepted = QInputDialog.getItem(
            self, "영역 클래스 변경", f"선택한 영역 {len(rows)}개의 새 클래스:", options, 0, False)
        if accepted and name:
            self.owner.edit_regions(rows, self.owner.project.data.class_names.index(name))

    def _delete(self):
        rows = self.selected_rows()
        if rows:
            self.owner.edit_regions(rows, None)
