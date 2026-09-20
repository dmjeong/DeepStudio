"""Semantic mask teaching: polygons, brush, eraser and ordered editable objects."""
import copy

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
    QSplitter, QWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox, QInputDialog,
)
from widgets.common import NoWheelComboBox
from widgets.obb_annotation import _SaveLabels
from widgets.annotation_navigation import AnnotationNavigation
from widgets.segmentation_canvas import SegmentationCanvas


class SegmentationAnnotationDialog(AnnotationNavigation, QDialog):
    def __init__(self, image_path, class_names, document, save, parent=None, *, add_class=None, navigation=None):
        super().__init__(parent)
        self.setWindowTitle("분할 마스크 티칭 — " + str(image_path).replace("\\", "/").split("/")[-1])
        self.resize(1280, 850)
        self.document, self.names = document, list(class_names)
        self._save_callback, self._add_class_callback = save, add_class
        self.undo_stack, self.redo_stack = [], []
        self.selected, self.worker, self.pending, self.dirty = -1, None, False, not document.persisted
        self.fit_scale = 1.0
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            raise ValueError("이미지를 열 수 없습니다.")
        layout = QVBoxLayout(self)
        hint = QLabel("P 다각형 · Enter/첫 점 클릭 완료 · B 브러시 · E 지우개 · V 선택/꼭짓점 이동 · Space 이동 · 휠 확대 · Ctrl+Z 실행 취소")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("정답 클래스"))
        self.class_combo = NoWheelComboBox()
        self.class_combo.addItems(self.names)
        self.class_combo.setCurrentIndex(min(1, len(self.names)-1))
        toolbar.addWidget(self.class_combo)
        add = QPushButton("클래스 추가")
        add.setVisible(add_class is not None)
        add.clicked.connect(self._add_class)
        toolbar.addWidget(add)
        toolbar.addWidget(QLabel("브러시 px"))
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 1024)
        self.brush_size.setValue(20)
        toolbar.addWidget(self.brush_size)
        toolbar.addWidget(QLabel("마스크 농도"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        toolbar.addWidget(self.opacity)
        self.zoom_label = QLabel("100%")
        toolbar.addWidget(self.zoom_label)
        layout.addLayout(toolbar)
        tools = QHBoxLayout()
        self.mode_buttons = {}
        for mode, label in (("polygon", "다각형 P"), ("rectangle", "사각형 R"), ("brush", "브러시 B"),
                            ("erase", "지우개 E"), ("select", "선택 V"), ("pan", "이동 H")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, mode=mode: self._set_mode(mode))
            self.mode_buttons[mode] = button
            tools.addWidget(button)
        self.finish_button = QPushButton("다각형 완료 Enter")
        tools.addWidget(self.finish_button)
        fit = QPushButton("화면 맞춤 0")
        tools.addWidget(fit)
        layout.addLayout(tools)
        self.view = SegmentationCanvas(pixmap, self)
        self.view.shape_created.connect(self._create)
        self.view.shape_edited.connect(self._edit)
        self.view.selection_changed.connect(self._select)
        self.view.command.connect(self._command)
        self.view.pending_changed.connect(self._pending)
        self.view.zoom_requested.connect(self._zoom)
        self.finish_button.clicked.connect(self.view.finish_polygon)
        self.class_combo.currentIndexChanged.connect(lambda value: setattr(self.view, "class_id", max(0, value)))
        self.view.class_id = max(0, self.class_combo.currentIndex())
        self.brush_size.valueChanged.connect(lambda value: setattr(self.view, "brush_size", value))
        self.opacity.valueChanged.connect(self.view.set_opacity)
        fit.clicked.connect(self._fit)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.view)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        background = self.names[0] if self.names else "먼저 배경 클래스를 추가하세요"
        info = QLabel(f"그리지 않은 곳과 지우개: 클래스 0 ({background})\n숫자 1~9: 클래스 선택\nF: 마스크 표시/숨김\n브러시·다각형은 목록의 아래 항목이 위에 그려집니다.")
        info.setWordWrap(True)
        side_layout.addWidget(info)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["영역", "클래스"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 85)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.cellClicked.connect(lambda row, column: self._select(row))
        side_layout.addWidget(self.table, 1)
        remove = QPushButton("선택 영역 삭제 Delete")
        remove.clicked.connect(self._remove)
        side_layout.addWidget(remove)
        history = QHBoxLayout()
        self.undo_button, self.redo_button = QPushButton("실행 취소"), QPushButton("다시 실행")
        self.undo_button.clicked.connect(self._undo)
        self.redo_button.clicked.connect(self._redo)
        history.addWidget(self.undo_button)
        history.addWidget(self.redo_button)
        side_layout.addLayout(history)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([940, 290])
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)
        self.status = QLabel("마스크는 원본 크기의 클래스 번호 PNG로 저장합니다. 배경만 있는 이미지도 정답을 저장하세요.")
        self.status.setWordWrap(True)
        self.view.status_changed.connect(self.status.setText)
        layout.addWidget(self.status)
        self.save_button = QPushButton("정답 저장 Ctrl+S")
        self.save_button.clicked.connect(self._save)
        layout.addWidget(self.save_button)
        self._add_navigation(layout, navigation)
        self._set_mode("polygon")
        self._render()
        QTimer.singleShot(0, self._fit)

    def _pending(self, value):
        self.pending = value
        self._buttons()

    def _buttons(self):
        self.save_button.setEnabled(self.dirty and not self.pending and self.worker is None and bool(self.names))
        self.finish_button.setEnabled(len(self.view.points) >= 6 and self.worker is None)
        self.undo_button.setEnabled(bool(self.undo_stack) and self.worker is None)
        self.redo_button.setEnabled(bool(self.redo_stack) and self.worker is None)

    def _checkpoint(self):
        self.undo_stack.append(copy.deepcopy(self.document.shapes))
        self.undo_stack = self.undo_stack[-100:]
        self.redo_stack.clear()

    def _create(self, shape):
        self._checkpoint()
        self.document.shapes.append(self.document.validate_shape(shape))
        self.selected = len(self.document.shapes)-1
        self.dirty = True
        self._render()

    def _edit(self, index, shape):
        self._checkpoint()
        self.document.shapes[index] = self.document.validate_shape(shape)
        self.dirty = True
        self._render()

    def _select(self, index):
        self.selected = index
        self.view.selected = index
        self.table.clearSelection()
        if index >= 0:
            self.table.selectRow(index)
        self.view.viewport().update()

    def _change_class(self, index, class_id):
        shape = dict(self.document.shapes[index], class_id=class_id)
        self._edit(index, shape)

    def _remove(self):
        if self.worker or not 0 <= self.selected < len(self.document.shapes):
            return
        self._checkpoint()
        self.document.shapes.pop(self.selected)
        self.selected = -1
        self.dirty = True
        self._render()

    def _undo(self):
        if self.worker or not self.undo_stack:
            return
        self.view.cancel_gesture()
        self.redo_stack.append(copy.deepcopy(self.document.shapes))
        self.document.shapes = self.undo_stack.pop()
        self.selected, self.dirty = -1, True
        self._render()

    def _redo(self):
        if self.worker or not self.redo_stack:
            return
        self.view.cancel_gesture()
        self.undo_stack.append(copy.deepcopy(self.document.shapes))
        self.document.shapes = self.redo_stack.pop()
        self.selected, self.dirty = -1, True
        self._render()

    def _render(self):
        self.view.set_document(self.document, self.names, self.selected)
        self.table.setRowCount(len(self.document.shapes))
        for i, shape in enumerate(self.document.shapes):
            label = "다각형" if shape["kind"] == "polygon" else "브러시"
            item = QTableWidgetItem(f"{i+1} {label}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, item)
            combo = NoWheelComboBox()
            combo.addItems(self.names)
            if shape["class_id"] == 255:
                combo.addItem("학습 제외 (255)")
                combo.setCurrentIndex(len(self.names))
            else:
                combo.setCurrentIndex(shape["class_id"])
            combo.currentIndexChanged.connect(lambda value, index=i: self._change_class(index, value if value < len(self.names) else 255))
            self.table.setCellWidget(i, 1, combo)
        self.table.resizeRowsToContents()
        if self.selected >= 0:
            self.table.selectRow(self.selected)
        self._buttons()

    def _set_mode(self, mode):
        self.view.set_mode(mode)
        for key, button in self.mode_buttons.items():
            button.setChecked(mode == key)
            button.setStyleSheet("background: #294b80; border: 1px solid #5590f0;" if mode == key else "")
        self.view.setFocus()

    def _command(self, command):
        if command in self.mode_buttons:
            self._set_mode(command)
        elif command in {"undo", "redo", "fit", "delete"}:
            {"undo": self._undo, "redo": self._redo, "fit": self._fit, "delete": self._remove}[command]()
        elif command == "overlay":
            self.view.toggle_overlay()
        elif command == "save" and self.save_button.isEnabled():
            self._save()
        elif command.startswith("class:"):
            value = int(command.split(":")[1])
            if value < len(self.names):
                self.class_combo.setCurrentIndex(value)
        elif command in {"previous", "next"}:
            self._navigate(-1 if command == "previous" else 1)

    def _fit(self):
        self.view.fitInView(self.view.image_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.fit_scale = self.view.transform().m11()
        self.zoom_label.setText("100%")

    def _zoom(self, direction):
        current = self.view.transform().m11()
        ratio = max(.1, min(16., current / max(self.fit_scale, .001) * (1.2 if direction > 0 else 1/1.2)))
        self.view.scale(ratio * self.fit_scale / current, ratio * self.fit_scale / current)
        self.zoom_label.setText(f"{ratio*100:.0f}%")

    def _add_class(self):
        name, accepted = QInputDialog.getText(self, "클래스 추가", "클래스 이름 (첫 클래스는 배경):")
        if accepted and name.strip():
            try:
                self.names = list(self._add_class_callback(name.strip()))
                self.document.classes = len(self.names)
                self.class_combo.clear()
                self.class_combo.addItems(self.names)
                self.class_combo.setCurrentIndex(len(self.names)-1)
                self._render()
            except Exception as exc:
                self.status.setText(str(exc))

    def _save(self):
        if self.worker or self.pending or not self.names:
            return
        self.worker = _SaveLabels(self._save_callback, self.document.payload(), self)
        self.worker.finished.connect(self._saved)
        self.setEnabled(False)
        self.worker.start()

    def _saved(self):
        worker, self.worker = self.worker, None
        self.setEnabled(True)
        if worker.error:
            self.navigation_delta = 0
            self.status.setText("저장 실패: " + worker.error)
            self._buttons()
        else:
            self.dirty = False
            self.accept()
        worker.deleteLater()

    def reject(self):
        if self.worker:
            return
        if (self.dirty or self.pending) and QMessageBox.question(self, "분할 티칭", "저장하지 않은 변경을 버리고 닫을까요?") != QMessageBox.StandardButton.Yes:
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker:
            event.ignore()
            return
        self.reject()
        event.setAccepted(self.result() == QDialog.DialogCode.Rejected and not self.isVisible())
