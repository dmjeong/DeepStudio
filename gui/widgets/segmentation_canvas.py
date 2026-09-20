"""Original-resolution polygon/brush editor with pan, zoom and vertex handles."""
import copy

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPixmap, QPen, QPolygonF, QPainterPath

from widgets.detection_canvas import DetectionCanvas


class SegmentationCanvas(DetectionCanvas):
    shape_created = Signal(dict)
    shape_edited = Signal(int, dict)
    pending_changed = Signal(bool)

    def __init__(self, pixmap, parent=None):
        super().__init__(pixmap, parent)
        self.mode = "polygon"
        self.class_id, self.brush_size = 1, 20
        self.stroke_erase = False
        self.points, self.stroke, self.preview_shape = [], [], None
        self.document = None
        self.overlay = self.scene().addPixmap(QPixmap())
        self.overlay.setOpacity(.45)
        self.overlay_visible = True

    def set_document(self, document, names, selected=-1):
        self.document, self.names, self.selected = document, list(names), selected
        self.rows = document.shapes
        self._set_overlay_pixels(document.render())
        self.viewport().update()

    def _set_overlay_pixels(self, pixels):
        """Render the actual semantic mask, including an eraser preview."""
        palette = np.zeros((256, 4), dtype=np.uint8)
        for i in range(1, self.document.classes):
            color = QColor.fromHsv((i * 67 + 205) % 360, 180, 255)
            palette[i] = [color.red(), color.green(), color.blue(), 255]
        palette[255] = [255, 255, 255, 110]
        rgba = np.ascontiguousarray(palette[pixels])
        image = QImage(rgba.data, self.document.width, self.document.height,
                       self.document.width * 4, QImage.Format.Format_RGBA8888).copy()
        self.overlay.setPixmap(QPixmap.fromImage(image))

    def set_opacity(self, value):
        self.overlay.setOpacity(value / 100)

    def toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible
        self.overlay.setVisible(self.overlay_visible)

    def cancel_gesture(self):
        super().cancel_gesture()
        self.points, self.stroke, self.preview_shape, self.stroke_erase = [], [], None, False
        if self.document is not None:
            self._set_overlay_pixels(self.document.render())
        self.pending_changed.emit(False)

    def _normalized(self, point):
        return [point.x() / max(1, self.image_rect.width()-1), point.y() / max(1, self.image_rect.height()-1)]

    def _clamp(self, point):
        return QPointF(max(0, min(self.image_rect.width()-1, point.x())),
                       max(0, min(self.image_rect.height()-1, point.y())))

    def _polygon(self, points):
        return QPolygonF([QPointF(points[i] * (self.image_rect.width()-1), points[i+1] * (self.image_rect.height()-1))
                          for i in range(0, len(points), 2)])

    def _hit(self, point):
        for i in reversed(range(len(self.rows))):
            item = self.rows[i]
            if item["kind"] == "polygon" and self._polygon(item["points"]).containsPoint(point, Qt.FillRule.OddEvenFill):
                return i
        return -1

    def _handle_at(self, point):
        if not 0 <= self.selected < len(self.rows) or self.rows[self.selected]["kind"] != "polygon":
            return -1
        radius = 8 / max(self.transform().m11(), .001)
        for i, handle in enumerate(self._polygon(self.rows[self.selected]["points"])):
            if abs(handle.x()-point.x()) <= radius and abs(handle.y()-point.y()) <= radius:
                return i
        return -1

    def _cursor(self, point=None):
        if self.space or self.mode == "pan":
            shape = Qt.CursorShape.ClosedHandCursor if self.gesture == "pan" else Qt.CursorShape.OpenHandCursor
        elif self.mode == "select":
            shape = Qt.CursorShape.SizeAllCursor if point is not None and (self._handle_at(point) >= 0 or self._hit(point) >= 0) else Qt.CursorShape.ArrowCursor
        else:
            shape = Qt.CursorShape.CrossCursor
        self.viewport().setCursor(shape)

    def finish_polygon(self):
        if len(self.points) < 6:
            self.status_changed.emit("다각형은 꼭짓점이 3개 이상 필요합니다.")
            return
        value = {"kind": "polygon", "class_id": self.class_id, "points": list(self.points)}
        try:
            self.document.validate_shape(value)
        except ValueError as exc:
            self.status_changed.emit(str(exc))
            return
        self.points = []
        self.shape_created.emit(value)
        self.pending_changed.emit(False)
        self.viewport().update()

    def mousePressEvent(self, event):
        self.setFocus()
        raw = self.mapToScene(event.position().toPoint())
        self.last = event.position().toPoint()
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and (self.space or self.mode == "pan")):
            self.gesture = "pan"
        elif event.button() == Qt.MouseButton.RightButton:
            self.cancel_gesture()
        elif event.button() == Qt.MouseButton.LeftButton and self.image_rect.contains(raw) and self.names:
            point = self._clamp(raw)
            self.start = point
            if self.mode == "polygon":
                polygon = self._polygon(self.points)
                if len(polygon) >= 3 and (polygon[0]-point).manhattanLength() * self.transform().m11() < 12:
                    self.finish_polygon()
                else:
                    self.points.extend(self._normalized(point))
                    self.pending_changed.emit(True)
            elif self.mode in {"brush", "erase"}:
                self.gesture = "stroke"
                self.stroke = self._normalized(point)
                # Shift temporarily turns the brush into an eraser. It lets
                # the user correct a contour without selecting another tool.
                self.stroke_erase = self.mode == "erase" or bool(
                    event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                self.pending_changed.emit(True)
            elif self.mode == "rectangle":
                self.gesture = "rectangle"
                self.pending_changed.emit(True)
            elif self.mode == "select":
                handle = self._handle_at(point)
                if handle >= 0:
                    self.handle, self.gesture = handle, "vertex"
                else:
                    self.selected = self._hit(point)
                    self.selection_changed.emit(self.selected)
                    self.gesture = "shape_move" if self.selected >= 0 else None
                if self.gesture:
                    self.original_shape = copy.deepcopy(self.rows[self.selected])
                    self.pending_changed.emit(True)
        self._cursor(raw)
        self.viewport().update()
        event.accept()

    def mouseMoveEvent(self, event):
        point = self._clamp(self.mapToScene(event.position().toPoint()))
        self.hover = point
        if self.gesture == "pan":
            delta = event.position().toPoint() - self.last
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-delta.y())
            self.last = event.position().toPoint()
        elif self.gesture == "stroke":
            normalized = self._normalized(point)
            if normalized != self.stroke[-2:]:
                self.stroke.extend(normalized)
            if self.stroke_erase:
                self._set_overlay_pixels(self.document.render({
                    "kind": "stroke", "class_id": 0, "operation": "erase",
                    "points": list(self.stroke), "width": self.brush_size,
                }))
        elif self.gesture == "rectangle":
            x1, y1 = self._normalized(self.start)
            x2, y2 = self._normalized(point)
            self.preview_shape = {"kind": "polygon", "class_id": self.class_id,
                                  "points": [x1, y1, x2, y1, x2, y2, x1, y2]}
        elif self.gesture in {"vertex", "shape_move"}:
            self.preview_shape = copy.deepcopy(self.original_shape)
            values = self.preview_shape["points"]
            x, y = self._normalized(point)
            if self.gesture == "vertex":
                values[2*self.handle:2*self.handle+2] = [x, y]
            else:
                sx, sy = self._normalized(self.start)
                dx = max(-min(values[0::2]), min(1-max(values[0::2]), x-sx))
                dy = max(-min(values[1::2]), min(1-max(values[1::2]), y-sy))
                self.preview_shape["points"] = [v + (dx if i % 2 == 0 else dy) for i, v in enumerate(values)]
        self._cursor(point)
        self.viewport().update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if not self.gesture:
            event.accept()
            return
        self.mouseMoveEvent(event)
        gesture = self.gesture
        self.gesture = None
        shape = self.preview_shape
        if gesture == "stroke":
            shape = {"kind": "stroke", "class_id": 0 if self.stroke_erase else self.class_id,
                     "points": list(self.stroke), "width": self.brush_size}
            if self.stroke_erase:
                shape["operation"] = "erase"
        if shape is not None:
            try:
                shape = self.document.validate_shape(shape)
                if gesture in {"vertex", "shape_move"}:
                    if shape != self.original_shape:
                        self.shape_edited.emit(self.selected, shape)
                elif shape.get("operation") != "erase" or not np.array_equal(
                        self.document.render(), self.document.render(shape)):
                    self.shape_created.emit(shape)
            except ValueError as exc:
                self.status_changed.emit(str(exc))
        self.stroke, self.preview_shape = [], None
        if gesture == "stroke" and self.stroke_erase:
            self._set_overlay_pixels(self.document.render())
        self.stroke_erase = False
        self.pending_changed.emit(bool(self.points))
        self.viewport().update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if self.mode == "polygon" and event.button() == Qt.MouseButton.LeftButton:
            self.finish_polygon()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        self.zoom_requested.emit(1 if event.angleDelta().y() > 0 else -1)
        event.accept()

    def focusOutEvent(self, event):
        # Toolbar/class changes must not discard an unfinished polygon.
        from PySide6.QtWidgets import QGraphicsView
        self.space = False
        self.gesture, self.preview_shape, self.stroke = None, None, []
        self.pending_changed.emit(bool(self.points))
        self._cursor()
        QGraphicsView.focusOutEvent(self, event)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.points:
            self.finish_polygon()
        elif key == Qt.Key.Key_Backspace and self.points:
            self.points = self.points[:-2]
            self.pending_changed.emit(bool(self.points))
            self.viewport().update()
        elif key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight):
            self.command.emit("brush_shrink" if key == Qt.Key.Key_BracketLeft else "brush_grow")
        elif not (event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)) and key in (
                Qt.Key.Key_P, Qt.Key.Key_B, Qt.Key.Key_E, Qt.Key.Key_R, Qt.Key.Key_V, Qt.Key.Key_D, Qt.Key.Key_F):
            self.command.emit({Qt.Key.Key_P: "polygon", Qt.Key.Key_B: "brush", Qt.Key.Key_E: "erase",
                               Qt.Key.Key_R: "rectangle", Qt.Key.Key_V: "select", Qt.Key.Key_D: "select", Qt.Key.Key_F: "overlay"}[key])
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            self.command.emit(f"class:{key-Qt.Key.Key_1}")
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def drawForeground(self, painter, rect):
        painter.save()
        painter.setClipRect(self.image_rect)
        scale = max(self.transform().m11(), .001)
        pen = QPen(QColor("#eef4ff"), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if 0 <= self.selected < len(self.rows) and self.rows[self.selected]["kind"] == "polygon":
            shape = self.preview_shape or self.rows[self.selected]
            polygon = self._polygon(shape["points"])
            painter.drawPolygon(polygon)
            painter.setBrush(QColor("#5590f0"))
            for point in polygon:
                painter.drawRect(QRectF(point.x()-4/scale, point.y()-4/scale, 8/scale, 8/scale))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self.points:
            polygon = self._polygon(self.points)
            painter.drawPolyline(polygon)
            if self.hover is not None:
                painter.drawLine(polygon[-1], self.hover)
            for point in polygon:
                painter.drawEllipse(point, 3/scale, 3/scale)
        if self.preview_shape is not None and self.gesture == "rectangle":
            painter.drawPolygon(self._polygon(self.preview_shape["points"]))
        if self.stroke:
            polygon = self._polygon(self.stroke)
            color = QColor("#ffffff") if self.stroke_erase else QColor.fromHsv((self.class_id*67+205) % 360, 180, 255)
            color.setAlpha(130)
            brush_pen = QPen(color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(brush_pen)
            path = QPainterPath(polygon[0])
            for point in polygon[1:]:
                path.lineTo(point)
            if len(polygon) == 1:
                painter.drawPoint(polygon[0])
            else:
                painter.drawPath(path)
        if self.mode in {"brush", "erase"} and self.hover is not None:
            painter.setPen(pen)
            painter.drawEllipse(self.hover, self.brush_size/2, self.brush_size/2)
        painter.restore()
