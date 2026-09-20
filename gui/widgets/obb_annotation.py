"""OBB annotation at original image coordinates, with transactional async saves."""
import copy

from PySide6.QtCore import Qt, Signal, QThread, QTimer, QPointF
from PySide6.QtGui import QColor, QPen, QBrush, QPolygonF, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QGraphicsScene, QGraphicsView, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QInputDialog, QSplitter, QWidget, QAbstractItemView,
)
from core.paths import ensure_python_path
from widgets.common import NoWheelComboBox
from widgets.annotation_navigation import AnnotationNavigation

ensure_python_path()
from obb import rectangle_from_three_points


class _ImageScene(QGraphicsScene):
    point_added = Signal(float, float)

    def mousePressEvent(self, event):
        p = event.scenePos()
        if event.button() == Qt.MouseButton.LeftButton and self.sceneRect().contains(p):
            self.point_added.emit(p.x(), p.y())
            event.accept()
        else:
            super().mousePressEvent(event)


class _SaveLabels(QThread):
    def __init__(self, save, rows, parent):
        super().__init__(parent)
        self.save, self.rows, self.error = save, copy.deepcopy(rows), None

    def run(self):
        try:
            self.save(self.rows)
        except Exception as exc:
            self.error = str(exc)


class OBBAnnotationDialog(AnnotationNavigation, QDialog):
    def __init__(self, image_path, class_names, rows, save, parent=None, *, task="obb", add_class=None, navigation=None):
        super().__init__(parent)
        self.selected = -1
        self.undo_stack, self.redo_stack = [], []
        self.task = task
        self._add_class_callback = add_class
        self.setWindowTitle("객체 박스와 클래스 편집" if task == "detect" else "OBB 정답 편집")
        self.resize(1080, 800)
        self.rows, self.names = copy.deepcopy(rows), list(class_names)
        self.points, self.dirty, self.worker = [], False, None
        self.save_empty = task == "detect" and not rows
        self._save_callback = save
        self.pixmap = QPixmap(image_path)
        if self.pixmap.isNull():
            raise ValueError("이미지를 열 수 없습니다")
        self.width_px, self.height_px = self.pixmap.width(), self.pixmap.height()
        layout = QVBoxLayout(self)
        help_text = QLabel("첫 두 번 클릭으로 한 변을 정하고, 세 번째 클릭으로 폭을 정하세요. "
                           "확대 후 가로와 세로 스크롤로 이동할 수 있습니다.")
        if task == "detect":
            help_text.setText("B 박스 그리기 · V 선택/이동 · Space 화면 이동 · 휠 확대 · Delete 삭제 · Ctrl+Z 실행 취소 · Ctrl+D 복제 · 숫자 1~9 클래스")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        toolbar = QHBoxLayout()
        self.class_combo = NoWheelComboBox()
        self.class_combo.addItems(self.names)
        toolbar.addWidget(QLabel("정답 클래스"))
        toolbar.addWidget(self.class_combo)
        self.add_class_button = QPushButton("클래스 추가")
        self.add_class_button.setVisible(add_class is not None)
        self.add_class_button.clicked.connect(self._add_class)
        toolbar.addWidget(self.add_class_button)
        toolbar.addWidget(QLabel("확대"))
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(25, 400)
        self.zoom.setValue(100)
        toolbar.addWidget(self.zoom)
        self.zoom_label = QLabel("100%")
        toolbar.addWidget(self.zoom_label)
        self.cancel_draw = QPushButton("그리기 취소")
        self.cancel_draw.clicked.connect(self._cancel_points)
        toolbar.addWidget(self.cancel_draw)
        layout.addLayout(toolbar)
        if task == "detect":
            from widgets.detection_canvas import DetectionCanvas
            self.view = DetectionCanvas(self.pixmap, self)
            self.scene = self.view.scene()
            self.view.box_created.connect(self._create_box)
            self.view.box_edited.connect(self._edit_box)
            self.view.selection_changed.connect(self._select_box)
            self.view.command.connect(self._command)
            self.view.zoom_requested.connect(lambda delta: self.zoom.setValue(self.zoom.value()+delta*10))
        else:
            self.scene = _ImageScene(self)
            self.scene.setSceneRect(0, 0, self.width_px, self.height_px)
            self.scene.point_added.connect(self._add_point)
            self.view = QGraphicsView(self.scene)
        self.view.setMinimumHeight(300)
        layout.addWidget(self.view, 1)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["인덱스", "클래스", "편집"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(180)
        layout.addWidget(self.table)
        if task == "detect":
            layout.removeWidget(self.view)
            layout.removeWidget(self.table)
            self.table.setMaximumHeight(16777215)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.table.cellClicked.connect(lambda row, column: self._select_box(row))
            controls = QHBoxLayout()
            self.mode_buttons = {}
            for mode, label in (("draw", "박스 그리기  B"), ("select", "선택과 이동  D"), ("pan", "화면 이동  H")):
                button = QPushButton(label)
                button.setCheckable(True)
                button.clicked.connect(lambda checked=False, mode=mode: self._set_mode(mode))
                self.mode_buttons[mode] = button
                controls.addWidget(button)
            self.undo_button = QPushButton("실행 취소")
            self.undo_button.clicked.connect(self._undo)
            controls.addWidget(self.undo_button)
            self.redo_button = QPushButton("다시 실행")
            self.redo_button.clicked.connect(self._redo)
            controls.addWidget(self.redo_button)
            duplicate = QPushButton("선택 박스 복제")
            duplicate.clicked.connect(self._duplicate)
            controls.addWidget(duplicate)
            remove = QPushButton("선택 박스 삭제")
            remove.clicked.connect(lambda: self._command("delete"))
            controls.addWidget(remove)
            fit = QPushButton("화면 맞춤  0")
            fit.clicked.connect(self._fit)
            controls.addWidget(fit)
            layout.addLayout(controls)
            self.editor_splitter = QSplitter(Qt.Orientation.Horizontal)
            self.editor_splitter.addWidget(self.view)
            inspector = QWidget()
            inspector_layout = QVBoxLayout(inspector)
            inspector_layout.setContentsMargins(8, 0, 0, 0)
            inspector_layout.addWidget(QLabel("객체 목록"))
            inspector_layout.addWidget(self.table, 1)
            self.editor_splitter.addWidget(inspector)
            self.editor_splitter.setStretchFactor(0, 1)
            self.editor_splitter.setStretchFactor(1, 0)
            self.editor_splitter.setSizes([800, 280])
            self.editor_splitter.setChildrenCollapsible(False)
            layout.addWidget(self.editor_splitter, 1)
            self.cancel_draw.hide()
            self._set_mode("draw")
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        if task == "detect":
            self.view.status_changed.connect(self.status.setText)
        self.save_button = QPushButton("정답 저장")
        self.save_button.clicked.connect(self._save)
        layout.addWidget(self.save_button)
        self._add_navigation(layout, navigation)
        self.zoom.valueChanged.connect(self._zoom)
        self._render()
        QTimer.singleShot(0, self._fit)

    def _checkpoint(self):
        if self.task == "detect":
            self.undo_stack.append(copy.deepcopy(self.rows))
            self.undo_stack = self.undo_stack[-100:]
            self.redo_stack.clear()

    def _undo(self):
        if self.worker or not self.undo_stack:
            return
        self.view.cancel_gesture()
        self.redo_stack.append(copy.deepcopy(self.rows))
        self.rows = self.undo_stack.pop()
        self.selected = -1
        self.dirty = True
        self._render()

    def _redo(self):
        if self.worker or not self.redo_stack:
            return
        self.view.cancel_gesture()
        self.undo_stack.append(copy.deepcopy(self.rows))
        self.rows = self.redo_stack.pop()
        self.selected = -1
        self.dirty = True
        self._render()

    def _set_mode(self, mode):
        self.view.set_mode(mode)
        for key, button in self.mode_buttons.items():
            button.setChecked(key == mode)
            button.setStyleSheet("background: #294b80; border: 1px solid #5590f0;" if key == mode else "")
        self.view.setFocus()

    def _command(self, command):
        if command in ("draw", "select", "pan"):
            self._set_mode(command)
        elif command == "delete" and self.selected >= 0:
            self._remove(self.selected)
        elif command == "undo":
            self._undo()
        elif command == "redo":
            self._redo()
        elif command == "fit":
            self._fit()
        elif command == "save" and self.save_button.isEnabled():
            self._save()
        elif command == "duplicate":
            self._duplicate()
        elif command.startswith("class:"):
            class_id = int(command.split(":")[1])
            if class_id < len(self.names):
                self.class_combo.setCurrentIndex(class_id)
        elif command in {"previous", "next"}:
            self._navigate(-1 if command == "previous" else 1)

    def _duplicate(self):
        if self.worker or not 0 <= self.selected < len(self.rows) or self.task != "detect":
            return
        self._checkpoint()
        row = copy.deepcopy(self.rows[self.selected])
        x, y, width, height = row["coordinates"]
        row["coordinates"] = [max(width/2, min(1-width/2, x+.02)),
                              max(height/2, min(1-height/2, y+.02)), width, height]
        self.rows.append(row)
        self.selected, self.dirty = len(self.rows)-1, True
        self._render()

    def _select_box(self, index):
        self.selected = index
        self.view.set_annotations(self.rows, self.names, index)
        self.table.clearSelection()
        if 0 <= index < len(self.rows):
            self.table.selectRow(index)
            self.table.scrollToItem(self.table.item(index, 0))

    def _create_box(self, coordinates):
        if self.worker or not self.names:
            return
        self._checkpoint()
        self.rows.append({"class_id": self.class_combo.currentIndex(), "coordinates": coordinates})
        self.selected = len(self.rows)-1
        self.dirty = True
        self._render()
        self.status.setText("박스를 만들었습니다. 조절점을 드래그하거나 우측 목록에서 클래스를 바꾸세요.")

    def _edit_box(self, index, coordinates):
        if self.worker or not 0 <= index < len(self.rows):
            return
        self._checkpoint()
        self.rows[index]["coordinates"] = coordinates
        self.dirty = True
        self._render()

    def _add_class(self):
        name, accepted = QInputDialog.getText(self, "클래스 추가", "새 클래스 이름:")
        if not accepted or not name.strip() or self.worker:
            return
        try:
            names = self._add_class_callback(name.strip())
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self.names = list(names)
        self.class_combo.clear()
        self.class_combo.addItems(self.names)
        self.class_combo.setCurrentText(name.strip())
        self.status.setText("클래스를 프로젝트에 저장했습니다. 객체 영역을 지정하세요.")
        self._render()

    def _fit(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.fit_scale = self.view.transform().m11()
        self.zoom.blockSignals(True)
        self.zoom.setValue(100)
        self.zoom.blockSignals(False)
        self.zoom_label.setText("100%")

    def _zoom(self, value):
        self.view.resetTransform()
        scale = getattr(self, "fit_scale", 1) * value / 100
        self.view.scale(scale, scale)
        self.zoom_label.setText(f"{value}%")

    def _add_point(self, x, y):
        if self.worker:
            return
        if not self.names:
            self.status.setText("먼저 클래스 추가 버튼으로 클래스를 등록하세요.")
            return
        if not (0 <= x <= self.width_px and 0 <= y <= self.height_px):
            return
        self.points.extend([x / self.width_px, y / self.height_px])
        if len(self.points) == (4 if self.task == "detect" else 6):
            try:
                if self.task == "detect":
                    x1, y1, x2, y2 = self.points
                    if abs(x2-x1) * self.width_px < 1 or abs(y2-y1) * self.height_px < 1:
                        raise ValueError("박스 너비와 높이는 각각 1픽셀 이상이어야 합니다.")
                    corners = [(x1+x2)/2, (y1+y2)/2, abs(x2-x1), abs(y2-y1)]
                else:
                    corners = rectangle_from_three_points(self.points, (self.width_px, self.height_px))
                self._checkpoint()
                self.rows.append({"class_id": self.class_combo.currentIndex(), "coordinates": corners})
                self.dirty = True
                self.status.setText("")
            except ValueError as exc:
                self.status.setText(str(exc))
            self.points = []
        self._render()

    def _cancel_points(self):
        self.points = []
        self._render()

    def _change_class(self, index, class_id):
        self._checkpoint()
        self.rows[index]["class_id"] = class_id
        self.selected = index
        self.dirty = True
        self._render()

    def _remove(self, index):
        self._checkpoint()
        self.rows.pop(index)
        self.selected = -1
        self.dirty = True
        self._render()

    def _render(self):
        if self.task == "detect":
            self._render_detection()
            return
        self.scene.clear()
        self.scene.addPixmap(self.pixmap)
        self.table.setRowCount(len(self.rows))
        for i, row in enumerate(self.rows):
            color = QColor.fromHsv((row["class_id"] * 67 + 205) % 360, 200, 255)
            pen = QPen(color, 2)
            pen.setCosmetic(True)
            coords = row["coordinates"]
            if self.task == "detect":
                x, y, w, h = coords
                coords = [x-w/2, y-h/2, x+w/2, y-h/2, x+w/2, y+h/2, x-w/2, y+h/2]
            polygon = QPolygonF([QPointF(coords[j] * self.width_px, coords[j+1] * self.height_px)
                                 for j in range(0, 8, 2)])
            fill = QColor(color)
            fill.setAlpha(35)
            self.scene.addPolygon(polygon, pen, QBrush(fill))
            label = self.scene.addText(f"{i+1}: {self.names[row['class_id']]}")
            label.setDefaultTextColor(color)
            label.setPos(polygon[0])
            item = QTableWidgetItem(str(i+1))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, item)
            combo = NoWheelComboBox()
            combo.addItems(self.names)
            combo.setCurrentIndex(row["class_id"])
            combo.currentIndexChanged.connect(lambda value, index=i: self._change_class(index, value))
            self.table.setCellWidget(i, 1, combo)
            remove = QPushButton("삭제")
            remove.clicked.connect(lambda checked=False, index=i: self._remove(index))
            self.table.setCellWidget(i, 2, remove)
        pen = QPen(QColor("#5590F0"), 3)
        pen.setCosmetic(True)
        for i in range(0, len(self.points), 2):
            x, y = self.points[i] * self.width_px, self.points[i+1] * self.height_px
            self.scene.addEllipse(x-3, y-3, 6, 6, pen)
            if i:
                self.scene.addLine(self.points[i-2] * self.width_px, self.points[i-1] * self.height_px, x, y, pen)
        self.save_button.setEnabled((self.dirty or self.save_empty) and not self.points and self.worker is None)
        self.cancel_draw.setEnabled(bool(self.points))

    def _render_detection(self):
        self.view.set_annotations(self.rows, self.names, self.selected)
        self.table.setRowCount(len(self.rows))
        for i, row in enumerate(self.rows):
            item = QTableWidgetItem(str(i+1))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, item)
            combo = NoWheelComboBox()
            combo.addItems(self.names)
            combo.setCurrentIndex(row["class_id"])
            combo.currentIndexChanged.connect(lambda value, index=i: self._change_class(index, value))
            self.table.setCellWidget(i, 1, combo)
            remove = QPushButton("삭제")
            remove.clicked.connect(lambda checked=False, index=i: self._remove(index))
            self.table.setCellWidget(i, 2, remove)
        self.table.resizeRowsToContents()
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(2)
        if 0 <= self.selected < len(self.rows):
            self.table.selectRow(self.selected)
        self.undo_button.setEnabled(bool(self.undo_stack) and self.worker is None)
        self.redo_button.setEnabled(bool(self.redo_stack) and self.worker is None)
        self.save_button.setEnabled((self.dirty or self.save_empty) and not self.points and self.worker is None)
        self.cancel_draw.setEnabled(bool(self.points))

    def _save(self):
        self.worker = _SaveLabels(self._save_callback, self.rows, self)
        self.worker.finished.connect(self._saved)
        self.setEnabled(False)
        self.worker.start()

    def _saved(self):
        worker, self.worker = self.worker, None
        self.setEnabled(True)
        error = worker.error
        worker.deleteLater()
        if error:
            self.navigation_delta = 0
            self.status.setText(error)
            self._render()
        else:
            self.dirty = False
            self.accept()

    def reject(self):
        if self.worker:
            return
        if (self.dirty or self.points) and QMessageBox.question(
                self, "정답 편집", "저장하지 않은 정답 변경을 버리고 닫을까요?") != QMessageBox.StandardButton.Yes:
            return
        super().reject()

    def closeEvent(self, event):
        event.ignore()
        self.reject()
