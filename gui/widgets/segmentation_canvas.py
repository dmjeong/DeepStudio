"""Original-resolution mask editor with scoped, tiled brush previews."""

import copy

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsItemGroup, QMenu
from widgets.detection_canvas import DetectionCanvas


class SegmentationCanvas(DetectionCanvas):
    shape_created = Signal(dict)
    shape_edited = Signal(int, dict)
    shapes_edited = Signal(object)
    pending_changed = Signal(bool)
    prompt_requested = Signal(dict)

    def __init__(self, pixmap, parent=None):
        super().__init__(pixmap, parent)
        self.mode = "polygon"
        self.class_id, self.brush_size = 1, 20
        self.stroke_erase = False
        self.points, self.stroke, self.preview_shape = [], [], None
        self.document = None
        self.overlay = QGraphicsItemGroup()
        self.scene().addItem(self.overlay)
        self.overlay.setOpacity(0.45)
        self.overlay_visible = True
        self.tiles = {}
        self.erase_scope = "selection"
        self.background_only = False
        self.brush_tip = "circle"
        self._snap_segments = None
        self.brush_object = None
        self.proposal = None
        self.prompt_points = []
        self.vertex = -1
        self._refresh_serial = 0
        self._stroke_dirty = None
        self._view_pixels = self._view_owners = None

    def set_document(self, document, names, selected=-1, region=None):
        self.document, self.names, self.selected = document, list(names), selected
        self.rows = document.shapes
        polygons = [
            np.asarray(s["points"]).reshape(-1, 2)
            * [max(1, document.width - 1), max(1, document.height - 1)]
            for i, s in enumerate(self.rows)
            if s["kind"] == "polygon"
            and not document.objects.get(document.key(i), {}).get("hidden")
        ]
        self._snap_segments = (
            (
                np.concatenate(polygons),
                np.concatenate([np.roll(p, -1, axis=0) for p in polygons]),
            )
            if polygons
            else None
        )
        self.locked = {
            i
            for i in self.object_indices()
            if document.objects.get(document.key(i), {}).get("locked")
        }
        self.hidden = {
            i
            for i in self.object_indices()
            if document.objects.get(document.key(i), {}).get("hidden")
        }
        if region is not None and self._view_pixels is not None:
            x, y, w, h = region
            self._view_pixels[y : y + h, x : x + w] = document.render(
                copy_pixels=False
            )[y : y + h, x : x + w]
            self._view_owners[y : y + h, x : x + w] = document._owners[
                y : y + h, x : x + w
            ]
        else:
            self._view_pixels = document.render()
            self._view_owners = document._owners.copy()
        self._set_overlay_pixels(self._view_pixels, region)
        self._refresh_serial += 1
        self.viewport().update()

    def object_indices(self):
        seen = set()
        result = []
        for i, item in enumerate(self.rows):
            if item.get("operation") == "erase" or item["class_id"] == 0:
                continue
            key = self.document.key(i)
            if key not in seen:
                result.append(i)
                seen.add(key)
        for class_id in self.document.base_classes:
            if class_id != 0:
                result.append(-int(class_id) - 2)
        return result

    def choose(self, indices, primary=None):
        valid = set(self.object_indices())
        self.selected_set = set(indices) & valid - self.hidden
        self.selected = (
            primary
            if primary in self.selected_set
            else next(iter(sorted(self.selected_set)), -1)
        )
        self.selection_changed.emit(self.selected)
        self.selection_set_changed.emit(set(self.selected_set))
        self.vertex = -1
        self.viewport().update()

    def _palette(self):
        palette = np.zeros((256, 4), dtype=np.uint8)
        for i in range(1, self.document.classes):
            color = QColor.fromHsv((i * 67 + 205) % 360, 180, 255)
            palette[i] = [color.red(), color.green(), color.blue(), 255]
        palette[255] = [220, 220, 220, 150]
        return palette

    def _set_overlay_pixels(self, pixels, region=None):
        if self.document is None:
            return
        x, y, w, h = region or (0, 0, self.document.width, self.document.height)
        if not w or not h:
            return
        palette = self._palette()
        size = 256
        for ty in range(y // size, (y + h - 1) // size + 1):
            for tx in range(x // size, (x + w - 1) // size + 1):
                ax, ay = tx * size, ty * size
                cell = pixels[ay : ay + size, ax : ax + size]
                rgba = np.ascontiguousarray(palette[cell])
                if self._view_owners is not None:
                    owners = self._view_owners[ay : ay + size, ax : ax + size]
                    for hidden in self.hidden:
                        rgba[owners == hidden, 3] = 0
                ih, iw = cell.shape
                image = QImage(
                    rgba.data, iw, ih, iw * 4, QImage.Format.Format_RGBA8888
                ).copy()
                tile = self.tiles.get((tx, ty))
                if tile is None:
                    tile = self.scene().addPixmap(QPixmap())
                    self.overlay.addToGroup(tile)
                    tile.setPos(ax, ay)
                    self.tiles[(tx, ty)] = tile
                tile.setPixmap(QPixmap.fromImage(image))

    def set_opacity(self, value):
        self.overlay.setOpacity(value / 100)

    def toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible
        self.overlay.setVisible(self.overlay_visible)

    def cancel_gesture(self):
        was_stroke = bool(getattr(self, "stroke", []))
        super().cancel_gesture()
        self.points, self.stroke, self.preview_shape, self.stroke_erase = (
            [],
            [],
            None,
            False,
        )
        if self.document is not None and was_stroke:
            self._restore_stroke_preview()
        self.pending_changed.emit(False)

    def _normalized(self, point):
        return [
            point.x() / max(1, self.image_rect.width() - 1),
            point.y() / max(1, self.image_rect.height() - 1),
        ]

    def _clamp(self, point):
        return QPointF(
            max(0, min(self.image_rect.width() - 1, point.x())),
            max(0, min(self.image_rect.height() - 1, point.y())),
        )

    def _snap(self, point):
        if not self.snap_enabled or self._snap_segments is None:
            return point
        a, b = self._snap_segments
        p = np.array([point.x(), point.y()])
        vertex = int(np.argmin(((a - p) ** 2).sum(axis=1)))
        radius = 6 / max(0.001, self.transform().m11())
        if np.linalg.norm(a[vertex] - p) <= radius:
            return QPointF(*a[vertex])
        vectors = b - a
        ratio = np.clip(
            ((p - a) * vectors).sum(axis=1)
            / np.maximum((vectors * vectors).sum(axis=1), 1e-20),
            0,
            1,
        )
        projected = a + ratio[:, None] * vectors
        edge = int(np.argmin(((projected - p) ** 2).sum(axis=1)))
        return (
            QPointF(*projected[edge])
            if np.linalg.norm(projected[edge] - p) <= radius
            else point
        )

    def _polygon(self, points):
        return QPolygonF(
            [
                QPointF(
                    points[i] * (self.image_rect.width() - 1),
                    points[i + 1] * (self.image_rect.height() - 1),
                )
                for i in range(0, len(points), 2)
            ]
        )

    def _hit(self, point):
        return (
            self.document.owner_at(round(point.x()), round(point.y()))
            if self.document
            else -1
        )

    def _handle_at(self, point):
        if (
            self.selected < 0
            or self.selected >= len(self.rows)
            or self.selected in self.locked
            or self.rows[self.selected]["kind"] != "polygon"
        ):
            return -1
        radius = 8 / max(self.transform().m11(), 0.001)
        for i, handle in enumerate(self._polygon(self.rows[self.selected]["points"])):
            if (
                abs(handle.x() - point.x()) <= radius
                and abs(handle.y() - point.y()) <= radius
            ):
                return i
        return -1

    def _cursor(self, point=None):
        if self.space or self.mode == "pan":
            shape = (
                Qt.CursorShape.ClosedHandCursor
                if self.gesture == "pan"
                else Qt.CursorShape.OpenHandCursor
            )
        elif self.mode == "select":
            shape = (
                Qt.CursorShape.SizeAllCursor
                if point is not None and self._hit(point) != -1
                else Qt.CursorShape.ArrowCursor
            )
        else:
            shape = Qt.CursorShape.CrossCursor
        self.viewport().setCursor(shape)

    def finish_polygon(self):
        if len(self.points) < 6:
            self.status_changed.emit("다각형은 꼭짓점이 3개 이상 필요합니다.")
            return
        shape = {
            "kind": "polygon",
            "class_id": self.class_id,
            "points": list(self.points),
        }
        try:
            shape = self.document.validate_shape(shape)
        except ValueError as exc:
            self.status_changed.emit(str(exc))
            return
        self.points = []
        self.shape_created.emit(shape)
        self.pending_changed.emit(False)
        self.viewport().update()

    def _stroke_shape(self, points):
        shape = {
            "kind": "stroke",
            "class_id": 0 if self.stroke_erase else self.class_id,
            "points": list(points),
            "width": self.brush_size,
        }
        if self.brush_tip == "square":
            shape["tip"] = "square"
        shape["protected_objects"] = [
            key for key, meta in self.document.objects.items() if meta.get("locked")
        ]
        if self.stroke_erase:
            shape["operation"] = "erase"
            if self.erase_scope == "selection" and self.selected != -1:
                shape["scope_object"] = self.document.key(self.selected)
            elif self.erase_scope != "all":
                shape["scope_class"] = self.class_id
        else:
            if self.brush_object:
                shape["object_id"] = self.brush_object
            if self.background_only:
                shape["background_only"] = True
        return shape

    def _preview_segment(self):
        segment = self._stroke_shape(
            self.stroke[-4:] if len(self.stroke) > 2 else self.stroke
        )
        self.document._apply(
            self._view_pixels, self._view_owners, segment, len(self.rows)
        )
        bounds, _ = self.document.shape_region(segment)
        self._stroke_dirty = (
            QRectF(*bounds).united(QRectF(*self._stroke_dirty))
            if self._stroke_dirty
            else QRectF(*bounds)
        )
        self._stroke_dirty = [
            int(self._stroke_dirty.x()),
            int(self._stroke_dirty.y()),
            int(self._stroke_dirty.width()),
            int(self._stroke_dirty.height()),
        ]
        self._set_overlay_pixels(self._view_pixels, bounds)

    def _restore_stroke_preview(self):
        if self._stroke_dirty:
            x, y, w, h = self._stroke_dirty
            self._view_pixels[y : y + h, x : x + w] = self.document.render(
                copy_pixels=False
            )[y : y + h, x : x + w]
            self._view_owners[y : y + h, x : x + w] = self.document._owners[
                y : y + h, x : x + w
            ]
            self._set_overlay_pixels(self._view_pixels, self._stroke_dirty)
        self._stroke_dirty = None

    def _edge_at(self, point):
        if (
            self.selected < 0
            or self.selected in self.locked
            or self.rows[self.selected]["kind"] != "polygon"
        ):
            return None
        polygon = self._polygon(self.rows[self.selected]["points"])
        best = None
        for i in range(len(polygon)):
            a, b = polygon[i], polygon[(i + 1) % len(polygon)]
            dx, dy = b.x() - a.x(), b.y() - a.y()
            length = dx * dx + dy * dy
            t = max(
                0,
                min(
                    1,
                    ((point.x() - a.x()) * dx + (point.y() - a.y()) * dy)
                    / max(length, 1e-12),
                ),
            )
            projected = QPointF(a.x() + t * dx, a.y() + t * dy)
            distance = (projected - point).manhattanLength() * self.transform().m11()
            if best is None or distance < best[0]:
                best = (distance, i, projected)
        return best if best and best[0] <= 12 else None

    def insert_vertex(self, point):
        edge = self._edge_at(point)
        if not edge:
            return
        _, index, point = edge
        shape = copy.deepcopy(self.rows[self.selected])
        shape["points"][2 * (index + 1) : 2 * (index + 1)] = self._normalized(point)
        self.shape_edited.emit(self.selected, shape)

    def delete_vertex(self):
        if self.selected < 0 or self.selected in self.locked or self.vertex < 0:
            return
        shape = copy.deepcopy(self.rows[self.selected])
        if shape["kind"] != "polygon" or len(shape["points"]) <= 6:
            self.status_changed.emit("다각형은 꼭짓점 3개 이상을 유지해야 합니다.")
            return
        del shape["points"][2 * self.vertex : 2 * self.vertex + 2]
        self.vertex = -1
        self.shape_edited.emit(self.selected, shape)

    def mousePressEvent(self, event):
        if self.mode == "grid_anchor":
            super().mousePressEvent(event)
            return
        self.setFocus()
        raw = self.mapToScene(event.position().toPoint())
        self.last = event.position().toPoint()
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and (self.space or self.mode == "pan")
        ):
            self.gesture = "pan"
        elif event.button() == Qt.MouseButton.RightButton:
            if self.mode == "ai_point" and self.image_rect.contains(raw):
                self.prompt_requested.emit({"point": [raw.x(), raw.y()], "label": 0})
            elif self.mode == "select" and self.selected >= 0:
                menu = QMenu(self)
                add = menu.addAction("여기에 꼭짓점 추가")
                add.setEnabled(self._edge_at(raw) is not None)
                remove = menu.addAction("선택 꼭짓점 삭제")
                remove.setEnabled(self.vertex >= 0 and self.selected not in self.locked)
                add.triggered.connect(lambda: self.insert_vertex(raw))
                remove.triggered.connect(self.delete_vertex)
                menu.exec(event.globalPosition().toPoint())
            else:
                self.cancel_gesture()
        elif (
            event.button() == Qt.MouseButton.LeftButton
            and self.image_rect.contains(raw)
            and self.names
        ):
            point = self._clamp(raw)
            if self.mode in {"polygon", "rectangle"}:
                point = self._snap(point)
            self.start = point
            if self.mode == "polygon":
                polygon = self._polygon(self.points)
                if (
                    len(polygon) >= 3
                    and (polygon[0] - point).manhattanLength() * self.transform().m11()
                    < 12
                ):
                    self.finish_polygon()
                elif (
                    not polygon
                    or (polygon[-1] - point).manhattanLength() * self.transform().m11()
                    >= 2
                ):
                    self.points.extend(self._normalized(point))
                    self.pending_changed.emit(True)
            elif self.mode in {"brush", "erase"}:
                if self.selected in self.locked:
                    self.status_changed.emit(
                        "선택 영역이 잠겨 있습니다. 잠금을 풀거나 새 영역을 선택하세요."
                    )
                    return
                self.gesture = "stroke"
                self.stroke = self._normalized(point)
                self._stroke_dirty = None
                self.stroke_erase = self.mode == "erase" or bool(
                    event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                )
                self._preview_segment()
                self.pending_changed.emit(True)
            elif self.mode in {"rectangle", "ai_box"}:
                self.gesture = self.mode
                self.pending_changed.emit(True)
            elif self.mode == "fill":
                self.command.emit(f"fill:{round(point.x())}:{round(point.y())}")
            elif self.mode == "ai_point":
                self.prompt_requested.emit(
                    {"point": [point.x(), point.y()], "label": 1}
                )
            elif self.mode == "select":
                handle = self._handle_at(point)
                if handle >= 0 and len(self.selected_set) <= 1:
                    self.handle, self.vertex, self.gesture = handle, handle, "vertex"
                    self.original_shape = copy.deepcopy(self.rows[self.selected])
                    self.pending_changed.emit(True)
                else:
                    hit = self._hit(point)
                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        chosen = set(self.selected_set)
                        if hit in chosen:
                            chosen.remove(hit)
                        elif hit != -1:
                            chosen.add(hit)
                        self.choose(chosen, hit)
                    elif hit not in self.selected_set:
                        self.choose({hit} if hit != -1 else set(), hit)
                    if hit >= 0 and hit in self.selected_set and hit not in self.locked:
                        self.gesture = "shape_move"
                        self.original_shape = copy.deepcopy(self.rows[hit])
                        keys = {
                            self.document.key(i)
                            for i in self.selected_set
                            if i >= 0 and i not in self.locked
                        }
                        self.original_shapes = {
                            i: copy.deepcopy(s)
                            for i, s in enumerate(self.rows)
                            if self.document.key(i) in keys
                            or s.get("scope_object") in keys
                        }
                        self.pending_changed.emit(True)
                    elif hit == -1:
                        self.gesture = "marquee"
                        self.preview = None
        self._cursor(raw)
        self.viewport().update()
        event.accept()

    def mouseMoveEvent(self, event):
        point = self._clamp(self.mapToScene(event.position().toPoint()))
        if self.gesture in {"vertex", "rectangle"} or self.mode == "polygon":
            point = self._snap(point)
        self.hover = point
        if self.gesture == "pan":
            delta = event.position().toPoint() - self.last
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self.last = event.position().toPoint()
        elif self.gesture == "stroke":
            normalized = self._normalized(point)
            if normalized != self.stroke[-2:]:
                self.stroke.extend(normalized)
                self._preview_segment()
        elif self.gesture in {"rectangle", "ai_box"}:
            x1, y1 = self._normalized(self.start)
            x2, y2 = self._normalized(point)
            self.preview_shape = {
                "kind": "polygon",
                "class_id": self.class_id,
                "points": [x1, y1, x2, y1, x2, y2, x1, y2],
            }
        elif self.gesture == "vertex":
            self.preview_shape = copy.deepcopy(self.original_shape)
            self.preview_shape["points"][2 * self.handle : 2 * self.handle + 2] = (
                self._normalized(point)
            )
        elif self.gesture == "shape_move":
            sx, sy = self._normalized(self.start)
            x, y = self._normalized(point)
            dx, dy = x - sx, y - sy
            points = []
            for s in self.original_shapes.values():
                if "points" in s:
                    points.extend(zip(s["points"][::2], s["points"][1::2]))
                else:
                    ax, ay, w, h = s["bounds"]
                    points.extend(
                        [
                            (
                                ax / max(1, self.document.width - 1),
                                ay / max(1, self.document.height - 1),
                            ),
                            (
                                (ax + w - 1) / max(1, self.document.width - 1),
                                (ay + h - 1) / max(1, self.document.height - 1),
                            ),
                        ]
                    )
            dx = max(-min(p[0] for p in points), min(1 - max(p[0] for p in points), dx))
            dy = max(-min(p[1] for p in points), min(1 - max(p[1] for p in points), dy))
            self.move_delta = (dx, dy)
            self.preview_shapes = {}
            for index, s in self.original_shapes.items():
                s = copy.deepcopy(s)
                if "points" in s:
                    s["points"] = [
                        v + (dx if j % 2 == 0 else dy)
                        for j, v in enumerate(s["points"])
                    ]
                else:
                    s["bounds"][0] = max(
                        0,
                        min(
                            self.document.width - s["bounds"][2],
                            s["bounds"][0] + round(dx * (self.document.width - 1)),
                        ),
                    )
                    s["bounds"][1] = max(
                        0,
                        min(
                            self.document.height - s["bounds"][3],
                            s["bounds"][1] + round(dy * (self.document.height - 1)),
                        ),
                    )
                self.preview_shapes[index] = s
            self.preview_shape = self.preview_shapes.get(self.selected)
        elif self.gesture == "marquee":
            self.preview = QRectF(self.start, point).normalized()
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
        serial = self._refresh_serial
        if gesture == "stroke":
            shape = self._stroke_shape(self.stroke)
        if gesture == "marquee":
            hits = set()
            for index in self.object_indices():
                bounds = self.document.bounds(index)
                if bounds and self.preview and self.preview.intersects(QRectF(*bounds)):
                    hits.add(index)
            self.choose(hits)
        elif gesture == "ai_box" and shape is not None:
            p = shape["points"]
            self.prompt_requested.emit(
                {
                    "box": [
                        min(p[::2]) * (self.document.width - 1),
                        min(p[1::2]) * (self.document.height - 1),
                        max(p[::2]) * (self.document.width - 1),
                        max(p[1::2]) * (self.document.height - 1),
                    ]
                }
            )
        elif shape is not None:
            try:
                shape = self.document.validate_shape(shape)
                if gesture == "shape_move":
                    changed = {
                        i: self.document.validate_shape(s)
                        for i, s in self.preview_shapes.items()
                        if s != self.rows[i]
                    }
                    if changed:
                        self.shapes_edited.emit(changed)
                elif gesture == "vertex":
                    if shape != self.original_shape:
                        self.shape_edited.emit(self.selected, shape)
                elif shape.get("operation") != "erase" or not np.array_equal(
                    self.document.render(copy_pixels=False), self._view_pixels
                ):
                    self.shape_created.emit(shape)
            except ValueError as exc:
                self.status_changed.emit(str(exc))
        self.stroke, self.preview_shape, self.preview = [], None, None
        self.stroke_erase = False
        if gesture == "stroke" and self._refresh_serial == serial:
            self._restore_stroke_preview()
        self._stroke_dirty = None
        self.pending_changed.emit(bool(self.points))
        self.viewport().update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if self.mode == "polygon" and event.button() == Qt.MouseButton.LeftButton:
            self.finish_polygon()
            event.accept()
        elif self.mode == "select" and event.button() == Qt.MouseButton.LeftButton:
            self.insert_vertex(self.mapToScene(event.position().toPoint()))
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def focusOutEvent(self, event):
        from PySide6.QtWidgets import QGraphicsView

        self.space = False
        if self.gesture:
            self.gesture, self.stroke, self.preview_shape = None, [], None
            if self.document:
                self._restore_stroke_preview()
        self.pending_changed.emit(bool(self.points))
        self._cursor()
        QGraphicsView.focusOutEvent(self, event)

    def keyPressEvent(self, event):
        key = event.key()
        ctrl = bool(
            event.modifiers()
            & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        )
        if ctrl and key == Qt.Key.Key_Z and (self.points or self.gesture):
            if self.points:
                self.points = self.points[:-2]
                self.pending_changed.emit(bool(self.points))
                self.viewport().update()
            else:
                self.cancel_gesture()
        elif key == Qt.Key.Key_Escape:
            if self.points or self.gesture:
                self.cancel_gesture()
            elif self.proposal is not None or self.mode.startswith("ai_"):
                self.command.emit("cancel_proposal")
            else:
                self.choose(set())
        elif not ctrl and key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.points:
            self.finish_polygon()
        elif (
            not ctrl
            and key in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and self.proposal is not None
        ):
            self.command.emit("accept_proposal")
        elif key == Qt.Key.Key_Backspace and self.points:
            self.points = self.points[:-2]
            self.pending_changed.emit(bool(self.points))
            self.viewport().update()
        elif (
            event.modifiers() & Qt.KeyboardModifier.AltModifier
            and key == Qt.Key.Key_Delete
        ):
            self.delete_vertex()
        elif key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight):
            self.command.emit(
                "brush_shrink" if key == Qt.Key.Key_BracketLeft else "brush_grow"
            )
        elif not ctrl and key in (
            Qt.Key.Key_P,
            Qt.Key.Key_B,
            Qt.Key.Key_E,
            Qt.Key.Key_R,
            Qt.Key.Key_V,
            Qt.Key.Key_D,
            Qt.Key.Key_F,
            Qt.Key.Key_O,
        ):
            self.command.emit(
                {
                    Qt.Key.Key_P: "polygon",
                    Qt.Key.Key_B: "brush",
                    Qt.Key.Key_E: "erase",
                    Qt.Key.Key_R: "rectangle",
                    Qt.Key.Key_V: "select",
                    Qt.Key.Key_D: "select",
                    Qt.Key.Key_F: "fill",
                    Qt.Key.Key_O: "overlay",
                }[key]
            )
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def drawForeground(self, painter, rect):
        painter.save()
        painter.setClipRect(self.image_rect)
        scale = max(self.transform().m11(), 0.001)
        pen = QPen(QColor("#e8eef5"), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for index in self.selected_set or (
            {self.selected} if self.selected != -1 else set()
        ):
            if index in self.hidden:
                continue
            shape = (
                self.preview_shape
                if index == self.selected and self.preview_shape
                else self.rows[index]
                if index >= 0
                else None
            )
            if self.gesture == "shape_move":
                shape = getattr(self, "preview_shapes", {}).get(index, shape)
            if shape and shape["kind"] == "polygon":
                polygon = self._polygon(shape["points"])
                painter.drawPolygon(polygon)
                if index not in self.locked and len(self.selected_set) <= 1:
                    for j, point in enumerate(polygon):
                        painter.setBrush(
                            QColor("#ffffff" if j == self.vertex else "#5590f0")
                        )
                        painter.drawRect(
                            QRectF(
                                point.x() - 4 / scale,
                                point.y() - 4 / scale,
                                8 / scale,
                                8 / scale,
                            )
                        )
                    painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                bounds = self.document.bounds(index)
                if bounds:
                    box = QRectF(*bounds)
                    if self.gesture == "shape_move" and index in getattr(
                        self, "original_shapes", {}
                    ):
                        dx, dy = self.move_delta
                        box.translate(
                            dx * (self.document.width - 1),
                            dy * (self.document.height - 1),
                        )
                    painter.drawRect(box)
        if self.points:
            polygon = self._polygon(self.points)
            painter.drawPolyline(polygon)
            if self.hover is not None:
                painter.drawLine(polygon[-1], self.hover)
            for point in polygon:
                painter.drawEllipse(point, 3 / scale, 3 / scale)
        if self.preview_shape is not None and self.gesture in {"rectangle", "ai_box"}:
            painter.drawPolygon(self._polygon(self.preview_shape["points"]))
        if self.gesture == "marquee" and self.preview:
            painter.drawRect(self.preview)
        if self.mode in {"brush", "erase"} and self.hover is not None:
            # The overlay contains the actual changed pixels. Only the cursor
            # outline is painted here; an erase gesture never draws white paint.
            painter.setPen(pen)
            if self.brush_tip == "square":
                painter.drawRect(
                    QRectF(
                        self.hover.x() - self.brush_size / 2,
                        self.hover.y() - self.brush_size / 2,
                        self.brush_size,
                        self.brush_size,
                    )
                )
            else:
                painter.drawEllipse(
                    self.hover, self.brush_size / 2, self.brush_size / 2
                )
        pen = QPen(QColor("#d4c66b"), 1, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for row in self.grid_preview:
            painter.drawPolygon(self._polygon(row["points"]))
        for point, label in self.prompt_points:
            painter.setBrush(QColor("#6ed9a5" if label else "#ff7979"))
            painter.drawEllipse(QPointF(*point), 4 / scale, 4 / scale)
        painter.restore()
