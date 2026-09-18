"""Direct-manipulation box canvas. All persisted coordinates are normalized coordinates."""
import copy

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QColor, QPen, QBrush, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene


class DetectionCanvas(QGraphicsView):
    box_created = Signal(list)
    box_edited = Signal(int, list)
    selection_changed = Signal(int)
    zoom_requested = Signal(int)
    command = Signal(str)
    status_changed = Signal(str)

    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        scene = QGraphicsScene(self)
        scene.addPixmap(pixmap)
        self.setScene(scene)
        self.image_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
        scene.setSceneRect(self.image_rect)
        self.rows, self.names, self.selected = [], [], -1
        self.mode, self.gesture = 'draw', None
        self.start = self.last = self.hover = None
        self.preview = None
        self.space = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setBackgroundBrush(QColor('#0c1016'))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._cursor()

    def set_annotations(self, rows, names, selected):
        self.rows, self.names = copy.deepcopy(rows), list(names)
        self.selected = selected if 0 <= selected < len(rows) else -1
        self.viewport().update()

    def set_mode(self, mode):
        self.cancel_gesture()
        self.mode = mode
        self._cursor()

    def cancel_gesture(self):
        self.gesture = None
        self.preview = None
        self.viewport().update()

    def _rect(self, row):
        x, y, w, h = row['coordinates']
        iw, ih = self.image_rect.width(), self.image_rect.height()
        return QRectF((x-w/2)*iw, (y-h/2)*ih, w*iw, h*ih)

    def _coordinates(self, rect):
        return [rect.center().x()/self.image_rect.width(), rect.center().y()/self.image_rect.height(),
                rect.width()/self.image_rect.width(), rect.height()/self.image_rect.height()]

    def _clamp(self, point):
        return QPointF(max(0, min(self.image_rect.width(), point.x())),
                       max(0, min(self.image_rect.height(), point.y())))

    @staticmethod
    def handles(rect):
        left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
        x, y = rect.center().x(), rect.center().y()
        return [QPointF(left, top), QPointF(x, top), QPointF(right, top), QPointF(right, y),
                QPointF(right, bottom), QPointF(x, bottom), QPointF(left, bottom), QPointF(left, y)]

    def _handle_at(self, point):
        if self.selected < 0:
            return -1
        radius = 7 / max(self.transform().m11(), .001)
        for i, handle in enumerate(self.handles(self._rect(self.rows[self.selected]))):
            if abs(point.x()-handle.x()) <= radius and abs(point.y()-handle.y()) <= radius:
                return i
        return -1

    def _hit(self, point):
        # Small nested objects remain selectable, irrespective of creation order.
        hits = [i for i, row in enumerate(self.rows) if self._rect(row).contains(point)]
        return min(hits, key=lambda i: self._rect(self.rows[i]).width()*self._rect(self.rows[i]).height()) if hits else -1

    def _cursor(self, point=None):
        shape = Qt.CursorShape.CrossCursor
        if self.space or self.mode == 'pan':
            shape = Qt.CursorShape.ClosedHandCursor if self.gesture == 'pan' else Qt.CursorShape.OpenHandCursor
        elif point is not None and self._handle_at(point) >= 0:
            handle = self._handle_at(point)
            shape = [Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeVerCursor,
                     Qt.CursorShape.SizeBDiagCursor, Qt.CursorShape.SizeHorCursor][handle % 4]
        elif self.mode == 'select':
            shape = Qt.CursorShape.SizeAllCursor if point is not None and self._hit(point) >= 0 else Qt.CursorShape.ArrowCursor
        self.viewport().setCursor(shape)

    def mousePressEvent(self, event):
        self.setFocus()
        point = self.mapToScene(event.position().toPoint())
        self.last = event.position().toPoint()
        if event.button() == Qt.MouseButton.MiddleButton or (
                event.button() == Qt.MouseButton.LeftButton and (self.space or self.mode == 'pan')):
            self.gesture = 'pan'
        elif event.button() == Qt.MouseButton.LeftButton and self.image_rect.contains(point):
            self.start = self._clamp(point)
            handle = self._handle_at(point)
            hit = self._hit(point)
            if handle >= 0:
                self.gesture = 'resize'
                self.handle = handle
                self.original = self._rect(self.rows[self.selected])
            elif self.mode == 'select' or event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.selected = hit
                self.selection_changed.emit(hit)
                if hit >= 0:
                    self.gesture = 'move'
                    self.original = self._rect(self.rows[hit])
                else:
                    self.gesture = 'pan'
            elif self.names:
                self.gesture = 'draw'
            else:
                self.status_changed.emit('먼저 클래스를 추가하세요.')
        else:
            super().mousePressEvent(event)
            return
        self._cursor(point)
        event.accept()

    def mouseMoveEvent(self, event):
        point = self._clamp(self.mapToScene(event.position().toPoint()))
        self.hover = point
        if self.gesture == 'pan':
            delta = event.position().toPoint() - self.last
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-delta.y())
            self.last = event.position().toPoint()
        elif self.gesture == 'draw':
            self.preview = QRectF(self.start, point).normalized()
        elif self.gesture == 'move':
            dx, dy = point.x()-self.start.x(), point.y()-self.start.y()
            dx = max(-self.original.left(), min(self.image_rect.right()-self.original.right(), dx))
            dy = max(-self.original.top(), min(self.image_rect.bottom()-self.original.bottom(), dy))
            self.preview = self.original.translated(dx, dy)
        elif self.gesture == 'resize':
            rect = QRectF(self.original)
            if self.handle in (0, 6, 7):
                rect.setLeft(min(point.x(), rect.right()-1))
            if self.handle in (2, 3, 4):
                rect.setRight(max(point.x(), rect.left()+1))
            if self.handle in (0, 1, 2):
                rect.setTop(min(point.y(), rect.bottom()-1))
            if self.handle in (4, 5, 6):
                rect.setBottom(max(point.y(), rect.top()+1))
            self.preview = rect
        if self.preview is not None:
            self.status_changed.emit(f'{self.preview.width():.0f} × {self.preview.height():.0f} px')
        self._cursor(point)
        self.viewport().update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if not self.gesture:
            super().mouseReleaseEvent(event)
            return
        # Include the release position even if the platform coalesced move events.
        self.mouseMoveEvent(event)
        gesture, rect = self.gesture, self.preview
        self.gesture, self.preview = None, None
        if rect is not None and rect.width() >= 1 and rect.height() >= 1:
            if gesture == 'draw':
                self.box_created.emit(self._coordinates(rect))
            elif gesture in ('move', 'resize') and rect != self.original:
                self.box_edited.emit(self.selected, self._coordinates(rect))
        elif gesture == 'draw':
            self.selected = self._hit(self.start)
            self.selection_changed.emit(self.selected)
        self._cursor(self.hover)
        self.viewport().update()
        event.accept()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_requested.emit(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.space = True
            self._cursor()
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Z:
            self.command.emit('redo' if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 'undo')
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_S:
            self.command.emit('save')
        elif key in (Qt.Key.Key_B, Qt.Key.Key_D, Qt.Key.Key_H):
            self.command.emit({Qt.Key.Key_B:'draw', Qt.Key.Key_D:'select', Qt.Key.Key_H:'pan'}[key])
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.command.emit('delete')
        elif key == Qt.Key.Key_Escape:
            self.cancel_gesture()
            self.command.emit('select')
        elif key == Qt.Key.Key_0:
            self.command.emit('fit')
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.space = False
            self._cursor()
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.space = False
        self.cancel_gesture()
        self._cursor()
        super().focusOutEvent(event)

    def drawForeground(self, painter, rect):
        painter.save()
        painter.setClipRect(self.image_rect)
        scale = max(self.transform().m11(), .001)
        for i, row in enumerate(self.rows):
            box = self.preview if i == self.selected and self.gesture in ('move', 'resize') and self.preview is not None else self._rect(row)
            color = QColor.fromHsv((row['class_id']*67+205)%360, 180, 255)
            pen = QPen(color, 2 if i == self.selected else 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            fill = QColor(color)
            fill.setAlpha(22 if i == self.selected else 6)
            painter.setBrush(QBrush(fill))
            painter.drawRect(box)
            painter.save()
            painter.translate(box.topLeft())
            painter.scale(1/scale, 1/scale)
            text = f"{i+1}  {self.names[row['class_id']]}"
            label = painter.fontMetrics().boundingRect(text).adjusted(-4, -2, 4, 2)
            label.moveTopLeft(label.topLeft()*0)
            painter.fillRect(label, QColor('#162538'))
            painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)
            painter.restore()
            if i == self.selected:
                painter.setBrush(QColor('#E8EAEF'))
                for point in self.handles(box):
                    size = 7/scale
                    painter.drawRect(QRectF(point.x()-size/2, point.y()-size/2, size, size))
        if self.gesture == 'draw' and self.preview is not None:
            pen = QPen(QColor('#5590F0'), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QColor(85,144,240,25))
            painter.drawRect(self.preview)
        if self.mode == 'draw' and self.hover is not None and self.gesture not in ('move', 'resize', 'pan'):
            pen = QPen(QColor(220,230,245,120), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(QPointF(0, self.hover.y()), QPointF(self.image_rect.width(), self.hover.y()))
            painter.drawLine(QPointF(self.hover.x(), 0), QPointF(self.hover.x(), self.image_rect.height()))
        painter.restore()
