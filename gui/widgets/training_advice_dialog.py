"""'데이터에 맞게 자동 설정' 결과: 지금 설정 → 추천 설정과 이유. 항목마다 골라 적용한다."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

# 표에 보이는 순서와 이름. (설정 키, 이름)
ROWS = (("patch_training", "패치 학습"), ("patch_scale", "패치 배율"), ("input_size", "입력(패치) 크기"),
        ("model_id", "모델"), ("training_mode", "학습 방식"), ("segmentation_loss", "분할 손실"),
        ("class_weights", "클래스 가중치"))


def _text(key, value, labels):
    if key == "patch_training":
        return "켜짐" if value else "꺼짐"
    if key == "patch_scale":
        return f"{float(value):.0%}"
    if key == "input_size":
        return f"{int(value)}px"
    return labels.get(key, {}).get(value, str(value) or "—")


class TrainingAdviceDialog(QDialog):
    """labels: {설정 키: {값: 화면 이름}} — 콤보 이름으로 보여 준다."""

    def __init__(self, advice, labels=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("데이터에 맞게 자동 설정")
        self.resize(900, 560)
        self.advice = advice
        layout = QVBoxLayout(self)
        stats = advice["stats"]
        classes = " · ".join(f"{name} {value['regions']}개(가는 쪽 {value['thin_px']:.1f}px)"
                             for name, value in stats["per_class"].items())
        risk = advice.get("risk") or {}
        summary = (f"라벨된 결함 {stats['regions']}개 / 이미지 {stats['images']}장 · 판정 영역 "
                   f"{stats['area'][0]}×{stats['area'][1]}px\n"
                   f"결함 굵기: 가는 쪽(하위 10%) {stats['thin_px']:.1f}px · 중간값 {stats['median_px']:.1f}px — {classes}\n"
                   f"결함 픽셀 비율 {stats['defect_share']:.2%}")
        if risk.get("lost"):
            summary += f"\n⚠ 지금 설정에서는 결함 {risk['lost']}개가 학습 해상도에서 1px보다 가늘어집니다(사라짐)."
        if stats.get("compute"):
            compute = stats["compute"]
            summary += (f"\n추천대로면 추론은 {compute['size'][0]}×{compute['size'][1]}px를 한 번에 계산합니다 "
                        f"(전체 축소 학습의 약 {compute['vs_resize']:g}배 계산량).")
        self.summary = QLabel(summary)
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)

        labels = labels or {}
        current, settings, reasons = advice["current"], advice["settings"], advice["reasons"]
        self.keys = [key for key, _name in ROWS if key in settings]
        self.table = QTableWidget(len(self.keys), 4)
        self.table.setHorizontalHeaderLabels(["항목 (체크 = 적용)", "지금", "추천", "이유"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setWordWrap(True)
        names = dict(ROWS)
        for row, key in enumerate(self.keys):
            same = current.get(key) == settings[key]
            check = QTableWidgetItem(names[key])
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check.setCheckState(Qt.CheckState.Unchecked if same else Qt.CheckState.Checked)
            self.table.setItem(row, 0, check)
            self.table.setItem(row, 1, QTableWidgetItem(_text(key, current.get(key), labels)))
            arrow = QTableWidgetItem(("= " if same else "→ ") + _text(key, settings[key], labels))
            if not same:
                font = arrow.font()
                font.setBold(True)
                arrow.setFont(font)
            self.table.setItem(row, 2, arrow)
            self.table.setItem(row, 3, QTableWidgetItem(reasons.get(key, "")))
        header = self.table.horizontalHeader()
        for column in range(3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        # 이유 열 폭은 창이 뜬 뒤 정해진다. 그 전에 행 높이를 맞추면 좁은 폭으로 줄바꿈해 행 하나가 표를 다 채운다.
        header.sectionResized.connect(lambda *_: self.table.resizeRowsToContents())
        layout.addWidget(self.table, 1)
        if advice.get("notes"):
            notes = QLabel("\n".join("• " + note for note in advice["notes"]))
            notes.setWordWrap(True)
            notes.setStyleSheet("color: #E5A832;")
            layout.addWidget(notes)
        buttons = QDialogButtonBox()
        self.apply_button = buttons.addButton("선택 항목 적용", QDialogButtonBox.ButtonRole.AcceptRole)
        self.apply_button.setDefault(True)  # Enter = 적용. 처음 포커스도 표가 아니라 적용 버튼
        self.apply_button.setFocus()
        buttons.addButton("닫기", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def showEvent(self, event):
        super().showEvent(event)
        self.table.resizeRowsToContents()

    def selected_settings(self):
        """체크한 항목의 추천 값."""
        return {key: self.advice["settings"][key] for row, key in enumerate(self.keys)
                if self.table.item(row, 0).checkState() == Qt.CheckState.Checked}


class AdviceScan(QThread):
    """자동 설정용 데이터셋 스캔 (영역 굵기·이미지 크기). 썸네일은 만들지 않는다."""

    done = Signal(object, str)  # (스캔 결과 또는 None, 오류 문구)

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project

    def run(self):
        try:
            from core.dataset_editor import scan_dataset
            self.done.emit(scan_dataset(self.project, self.isInterruptionRequested), "")
        except Exception as exc:  # 스캔 실패도 화면에서 알린다
            self.done.emit(None, str(exc))
