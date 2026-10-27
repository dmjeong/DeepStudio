"""Shared Windows annotation workspace for axis-aligned boxes and semantic masks."""

from __future__ import annotations

from core.annotation_tools import grid_boxes
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class GridDialog(QDialog):
    def __init__(self, workspace, seed):
        super().__init__(workspace)
        self.workspace = workspace
        self.source_revision = workspace.revision
        self.seed_key = workspace._key(workspace.selected)
        self.seed = list(seed)
        self.candidates = []
        self.setWindowTitle("격자 후보 · 확인 후 적용")
        self.resize(420, 640)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.columns = QSpinBox()
        self.columns.setRange(1, 128)
        self.columns.setValue(8)
        self.rows = QSpinBox()
        self.rows.setRange(1, 128)
        self.rows.setValue(16)
        form.addRow("열 개수", self.columns)
        form.addRow("행 개수", self.rows)
        self.anchors = []
        x, y, w, h = seed
        for label, value in [
            ("가로 끝 중심 X", x + 7 * w * 1.15),
            ("가로 끝 중심 Y", y),
            ("세로 끝 중심 X", x),
            ("세로 끝 중심 Y", y + 15 * h * 1.15),
            ("대각 끝 중심 X", x + 7 * w * 1.15),
            ("대각 끝 중심 Y", y + 15 * h * 1.15),
        ]:
            spin = QDoubleSpinBox()
            spin.setRange(-200, 300)
            spin.setDecimals(3)
            spin.setSuffix(" %")
            spin.setValue(value * 100)
            self.anchors.append(spin)
            form.addRow(label, spin)
        self.perspective = QCheckBox("4개 기준점 원근 보정")
        layout.addWidget(self.perspective)
        for label, index in [
            ("가로 마지막 칩 중심 찍기", 0),
            ("세로 마지막 칩 중심 찍기", 1),
            ("대각 마지막 칩 중심 찍기", 2),
        ]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, i=index: self.pick_anchor(i))
            layout.addWidget(button)
        self._pick_index = None
        self._previous_mode = workspace.view.mode
        workspace.view.point_picked.connect(self._picked)
        self.replace_seed = QCheckBox("기준 영역을 후보로 교체")
        self.replace_seed.setToolTip(
            "끄면 기준 영역은 그대로 두고 첫 칸의 복제를 생략합니다."
        )
        self.replace_seed.setChecked(True)
        if workspace.object_meta.get(self.seed_key, {}).get("locked"):
            self.replace_seed.setChecked(False)
            self.replace_seed.setEnabled(False)
        self.replace_seed.toggled.connect(self._seed_policy)
        layout.addWidget(self.replace_seed)
        self.info = QLabel(
            "좌표는 원본 이미지의 백분율입니다. 제외할 칸은 체크를 해제하세요."
        )
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        preview = QPushButton("후보 미리보기")
        preview.clicked.connect(self.preview)
        layout.addWidget(preview)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["적용", "행 / 열", "범위"])
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table, 1)
        self.table.itemChanged.connect(self._refresh_preview)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.apply
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.preview()

    def _seed_policy(self):
        if self.candidates:
            with QSignalBlocker(self.table):
                item = self.table.item(0, 0)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if self.replace_seed.isChecked() and self.candidates[0]["valid"]
                    else Qt.CheckState.Unchecked
                )
                item.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                    if self.replace_seed.isChecked()
                    else Qt.ItemFlag.NoItemFlags
                )
        self._refresh_preview()

    def pick_anchor(self, index):
        self._pick_index = index
        self.workspace.view.set_mode("grid_anchor")
        self.workspace.status.setText(
            "이미지에서 "
            + ["가로 마지막", "세로 마지막", "대각 마지막"][index]
            + " 칩의 중심을 클릭하세요."
        )
        self.workspace.view.setFocus()

    def _picked(self, point):
        if self._pick_index is None:
            return
        denominator = [self.workspace.width_px, self.workspace.height_px]
        if self.workspace.task == "segment":
            denominator = [max(1, v - 1) for v in denominator]
        for j in range(2):
            self.anchors[2 * self._pick_index + j].setValue(
                100 * point[j] / denominator[j]
            )
        self._pick_index = None
        self.workspace.view.set_mode(self._previous_mode)
        self.preview()

    def done(self, result):
        self.workspace.view.point_picked.disconnect(self._picked)
        if self.workspace.view.mode == "grid_anchor":
            self.workspace.view.set_mode(self._previous_mode)
        super().done(result)

    def preview(self):
        try:
            a = [s.value() / 100 for s in self.anchors]
            self.candidates = grid_boxes(
                self.seed,
                self.columns.value(),
                self.rows.value(),
                column_end=a[:2],
                row_end=a[2:4],
                far_corner=a[4:] if self.perspective.isChecked() else None,
            )
            with QSignalBlocker(self.table):
                self.table.setRowCount(len(self.candidates))
                for i, row in enumerate(self.candidates):
                    check = QTableWidgetItem()
                    check.setFlags(
                        Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                    )
                    check.setCheckState(
                        Qt.CheckState.Checked
                        if row["valid"]
                        else Qt.CheckState.Unchecked
                    )
                    self.table.setItem(i, 0, check)
                    self.table.setItem(
                        i,
                        1,
                        QTableWidgetItem(f"R{row['row']:02d} C{row['column']:02d}"),
                    )
                    self.table.setItem(
                        i,
                        2,
                        QTableWidgetItem("이미지 안" if row["valid"] else "범위 밖"),
                    )
            self._seed_policy()
        except ValueError as exc:
            self.info.setText(str(exc))
            self.candidates = []

    def selected(self):
        return [
            row
            for i, row in enumerate(self.candidates)
            if self.table.item(i, 0).checkState() == Qt.CheckState.Checked
            and (i != 0 or self.replace_seed.isChecked())
        ]

    def _refresh_preview(self):
        selected = self.selected()
        self.info.setText(
            f"전체 {len(self.candidates)}개 · 적용 {len(selected)}개 · 범위 밖 {sum(not r['valid'] for r in self.candidates)}개"
        )
        self.workspace.view.grid_preview = selected
        self.workspace.view.viewport().update()

    def apply(self):
        if self.workspace.revision != self.source_revision:
            self.info.setText(
                "미리보기 중 원본이 수정됐습니다. 이 창을 닫고 기준 영역을 다시 선택하세요."
            )
            return
        selected = self.selected()
        if not selected or any(not row["valid"] for row in selected):
            self.info.setText("적용할 유효 후보를 선택하세요.")
            return
        # Seed removal and creation are one Undo operation.
        workspace = self.workspace
        before = workspace._capture()
        undo_count = len(workspace.undo_stack)
        try:
            seed_index = next(
                (i for i in workspace._indices() if workspace._key(i) == self.seed_key),
                -1,
            )
            if self.replace_seed.isChecked() and seed_index in workspace._indices():
                workspace._select_many({seed_index})
                workspace._remove(seed_index)
            workspace._apply_grid(selected)
            workspace.undo_stack = workspace.undo_stack[:undo_count] + [
                ("격자 생성", before)
            ]
            workspace._render()
            self.accept()
        except ValueError as exc:
            workspace._apply_state(before)
            workspace.undo_stack = workspace.undo_stack[:undo_count]
            workspace._render()
            self.info.setText(str(exc))
