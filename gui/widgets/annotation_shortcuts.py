"""Canvas-scoped, user-editable shortcuts. Text fields keep their own keys."""

import json

from PySide6.QtCore import QEvent, QObject, QSettings, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QKeySequenceEdit,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class AnnotationShortcuts(QObject):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace
        self.specs = {
            "select": ("선택", "V"),
            "erase": ("지우개 / 클릭 삭제", "E"),
            "pan": ("화면 이동", "H"),
            "grid": ("격자", "G"),
            "overlay": ("정답 표시 / 숨김", "O"),
            "fit": ("화면 맞춤", "0"),
            "actual_size": ("원본 1:1", "Ctrl+1"),
            "undo": ("실행 취소", "Ctrl+Z"),
            "redo": ("다시 실행", "Ctrl+Y"),
            "duplicate": ("선택 복제", "Ctrl+D"),
            "delete": ("선택 삭제", "Del"),
            "save": ("정답 저장", "Ctrl+S"),
            "review_next": ("검수 완료 후 다음", "Ctrl+Return"),
            "previous": ("이전 이미지", "J"),
            "next": ("다음 이미지", "K"),
        }
        if workspace.task == "segment":
            self.specs.update(
                rectangle=("사각형", "R"),
                polygon=("다각형", "P"),
                brush=("브러시", "B"),
                fill=("채우기", "F"),
            )
        else:
            self.specs["draw"] = ("박스", "R")
        self.defaults = {key: value[1] for key, value in self.specs.items()}
        self.bindings = dict(self.defaults)
        value = workspace.preferences.get("_shortcuts")
        if value is None and workspace.preferences.get("_persistent"):
            try:
                value = json.loads(
                    QSettings("DeepVisionStudio", "AnnotationWorkspace").value(
                        "shortcuts/" + workspace.task, "{}"
                    )
                )
            except (ValueError, TypeError):
                value = {}
        if isinstance(value, dict):
            proposed = {
                key: value.get(key, default) for key, default in self.defaults.items()
            }
            try:
                self._validate(proposed)
            except (ValueError, TypeError):
                pass
            else:
                self.bindings = proposed
        workspace.view.installEventFilter(self)
        workspace.view.viewport().installEventFilter(self)
        self._refresh_labels()

    @staticmethod
    def canonical(value):
        return QKeySequence.fromString(
            value, QKeySequence.SequenceFormat.PortableText
        ).toString(QKeySequence.SequenceFormat.PortableText)

    def _validate(self, bindings):
        seen = set()
        reserved = {
            "Space",
            "Esc",
            "Return",
            "Enter",
            "Backspace",
            "[",
            "]",
            *(str(i) for i in range(1, 10)),
        }
        for value in bindings.values():
            if not isinstance(value, str):
                raise TypeError("단축키 문자열이 필요합니다.")
            sequence = QKeySequence.fromString(
                value, QKeySequence.SequenceFormat.PortableText
            )
            if sequence.count() > 1:
                raise ValueError("단축키는 키 조합 하나만 사용할 수 있습니다.")
            key = self.canonical(value)
            if not key:
                continue
            if sequence[0].key() in {
                Qt.Key.Key_unknown,
                Qt.Key.Key_Control,
                Qt.Key.Key_Shift,
                Qt.Key.Key_Alt,
                Qt.Key.Key_Meta,
            }:
                raise ValueError("Ctrl·Shift 같은 보조키만 지정할 수 없습니다.")
            if key in reserved:
                raise ValueError(
                    "Space·Enter·Esc·Backspace·숫자·브러시 크기 키는 고정입니다."
                )
            if key in seen:
                raise ValueError("같은 단축키를 두 작업에 지정할 수 없습니다.")
            seen.add(key)

    def update(self, bindings):
        proposed = {
            key: bindings.get(key, default) for key, default in self.defaults.items()
        }
        self._validate(proposed)
        self.bindings = {key: self.canonical(value) for key, value in proposed.items()}
        self.workspace.preferences["_shortcuts"] = dict(self.bindings)
        if self.workspace.preferences.get("_persistent"):
            settings = QSettings("DeepVisionStudio", "AnnotationWorkspace")
            settings.setValue(
                "shortcuts/" + self.workspace.task, json.dumps(self.bindings)
            )
            settings.sync()
        self._refresh_labels()

    def _refresh_labels(self):
        short = {
            "select": "선택",
            "erase": "지우개" if self.workspace.task == "segment" else "클릭 삭제",
            "pan": "이동",
            "draw": "박스",
            "rectangle": "사각형",
            "polygon": "다각형",
            "brush": "브러시",
            "fill": "채우기",
        }
        for command, button in self.workspace.mode_buttons.items():
            button.setToolTip(
                self.specs.get(command, (command, ""))[0]
                + "  "
                + self.bindings.get(command, "")
            )
            if command in short:
                text = short[command] + " " + self.bindings.get(command, "")
                button.setText(
                    button.fontMetrics().elidedText(
                        text, Qt.TextElideMode.ElideRight, 110
                    )
                )
        self.workspace.save_button.setText("정답 저장 " + self.bindings["save"])
        self.workspace.review_button.setText(
            "검수 완료·다음 " + self.bindings["review_next"]
        )
        self.workspace.grid_button.setText("격자 " + self.bindings["grid"])
        hint = self.workspace.findChild(QLabel, "annotation-shortcut-hint")
        if hint is not None:
            hint.setText(
                "Space·휠버튼 드래그 이동\n휠 확대\n"
                + (self.bindings["undo"] or "—")
                + " 취소\n"
                + (self.bindings["previous"] or "—")
                + " / "
                + (self.bindings["next"] or "—")
                + " 탐색"
            )

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Type.KeyPress:
            return False
        key = QKeySequence(event.keyCombination()).toString(
            QKeySequence.SequenceFormat.PortableText
        )
        for command, binding in self.bindings.items():
            if binding and key == binding:
                self.workspace._command(command)
                event.accept()
                return True
        aliases = {"D": "select", "Ctrl+Shift+Z": "redo", "Ctrl+Enter": "review_next"}
        if self.workspace.task == "detect":
            aliases["B"] = "draw"
        command = aliases.get(key)
        if command and self.bindings[command] == self.defaults[command]:
            return False
        if key in self.defaults.values() or command:
            event.accept()
            return True
        return False

    def edit(self):
        dialog = QDialog(self.workspace)
        dialog.setWindowTitle("라벨링 단축키")
        dialog.resize(460, 600)
        layout = QVBoxLayout(dialog)
        hint = QLabel(
            "캔버스에서 사용하는 단축키입니다. 입력란의 문자 편집은 그대로 동작합니다.\nSpace·Enter·Esc·숫자키는 고정이며 빈 값은 해당 단축키를 끕니다."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        table = QTableWidget(len(self.specs), 2)
        table.setHorizontalHeaderLabels(["작업", "단축키"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        edits = {}
        for i, (key, (label, _)) in enumerate(self.specs.items()):
            table.setItem(i, 0, QTableWidgetItem(label))
            field = QKeySequenceEdit(QKeySequence(self.bindings[key]))
            table.setCellWidget(i, 1, field)
            edits[key] = field
        error = QLabel()
        error.setWordWrap(True)
        layout.addWidget(error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).setText(
            "기본값"
        )
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            lambda: [
                field.setKeySequence(QKeySequence(self.defaults[key]))
                for key, field in edits.items()
            ]
        )

        def apply():
            values = {
                key: field.keySequence().toString(
                    QKeySequence.SequenceFormat.PortableText
                )
                for key, field in edits.items()
            }
            try:
                self.update(values)
            except (ValueError, TypeError) as exc:
                error.setText(str(exc))
                return
            dialog.accept()

        buttons.accepted.connect(apply)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
