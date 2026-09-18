"""Native-pixel mask editing and a shared-coordinate comparison viewer."""
import base64
from PySide6.QtCore import Qt, QPointF, QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygonF
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
                              QPushButton, QLabel, QComboBox, QSpinBox, QSlider)


class MaskCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.mask = QImage()
        self.backup = QImage()
        self.mask_item = None
        self.tool, self.brush = "brush", 12
        self.length, self.stamp_width, self.angle = 60, 12, 0
        self.drawing = False
        self.points = []
        self.setMinimumHeight(320)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def load(self, image, mask=None):
        pixels = QPixmap(str(image))
        if pixels.isNull():
            raise ValueError("이미지를 불러오지 못했습니다")
        self.scene().clear()
        self.scene().addPixmap(pixels)
        self.setSceneRect(0, 0, pixels.width(), pixels.height())
        self.mask = QImage(pixels.size(), QImage.Format.Format_ARGB32)
        self.mask.fill(Qt.GlobalColor.transparent)
        if mask:
            source = QImage(str(mask))
            if source.size() != pixels.size():
                raise ValueError("마스크와 원본 크기가 다릅니다")
            import numpy as np
            from PIL import Image
            binary = np.array(Image.open(mask).convert("L")) > 127
            rgba = np.full((*binary.shape, 4), 255, dtype="uint8")
            rgba[:, :, 3] = binary.astype("uint8") * 255
            self.mask = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
        self.mask_item = self.scene().addPixmap(QPixmap.fromImage(self.mask))
        self.mask_item.setOpacity(.5)
        self.backup = self.mask.copy()
        self.points = []
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_mask(self):
        if self.mask_item:
            self.mask_item.setPixmap(QPixmap.fromImage(self.mask))

    def mousePressEvent(self, event):
        if self.tool == "pan" or self.mask.isNull():
            super().mousePressEvent(event)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.mapToScene(event.position().toPoint())
        self.backup = self.mask.copy()
        self.start = self.previous = point
        if self.tool == "polygon":
            self.points.append(point)
            return
        if self.tool == "stamp":
            painter = QPainter(self.mask)
            painter.translate(point)
            painter.rotate(self.angle)
            painter.fillRect(-self.length // 2, -self.stamp_width // 2, self.length, self.stamp_width, Qt.GlobalColor.white)
            painter.end()
            self.update_mask()
            return
        self.drawing = True
        self.paint_to(point + QPointF(.01, .01))

    def paint_to(self, point):
        if self.tool == "rect":
            self.mask = self.backup.copy()
        painter = QPainter(self.mask)
        if self.tool == "erase":
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.setPen(QPen(QColor("white"), self.brush, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        if self.tool == "rect":
            from PySide6.QtCore import QRectF
            painter.fillRect(QRectF(self.start, point).normalized(), Qt.GlobalColor.white)
        else:
            painter.drawLine(self.previous, point)
        painter.end()
        self.previous = point
        self.update_mask()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.paint_to(self.mapToScene(event.position().toPoint()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.drawing = False
        super().mouseReleaseEvent(event)

    def finish_polygon(self):
        if len(self.points) >= 3:
            painter = QPainter(self.mask)
            painter.setBrush(Qt.GlobalColor.white)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(QPolygonF(self.points))
            painter.end()
            self.points = []
            self.update_mask()

    def png(self):
        if self.mask.isNull():
            raise ValueError("이미지를 선택하고 불량 영역을 표시하세요")
        result = QImage(self.mask.size(), QImage.Format.Format_RGB888)
        result.fill(Qt.GlobalColor.black)
        painter = QPainter(result)
        painter.drawImage(0, 0, self.mask)
        painter.end()
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        result.save(buffer, "PNG")
        return {"png": base64.b64encode(bytes(data)).decode("ascii")}


class MaskEditor(QWidget):
    def __init__(self, label="불량 영역 편집"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(label))
        row = QHBoxLayout()
        self.canvas = MaskCanvas()
        tools = QComboBox()
        for name, value in (("브러시", "brush"), ("지우개", "erase"), ("사각형", "rect"), ("다각형", "polygon"), ("크기/방향", "stamp"), ("이동", "pan")):
            tools.addItem(name, value)
        def select():
            self.canvas.tool = tools.currentData()
            self.canvas.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if tools.currentData() == "pan" else QGraphicsView.DragMode.NoDrag)
        tools.currentIndexChanged.connect(select)
        row.addWidget(tools)
        for title, attribute, initial, lo, hi in (("브러시 px", "brush", 12, 1, 500), ("길이 px", "length", 60, 1, 4096),
                                                ("폭 px", "stamp_width", 12, 1, 4096), ("회전 °", "angle", 0, -180, 180)):
            row.addWidget(QLabel(title))
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(initial)
            spin.valueChanged.connect(lambda value, key=attribute: setattr(self.canvas, key, value))
            row.addWidget(spin)
        layout.addLayout(row)
        layout.addWidget(self.canvas, 1)
        buttons = QHBoxLayout()
        def clear():
            self.canvas.backup = self.canvas.mask.copy()
            self.canvas.mask.fill(Qt.GlobalColor.transparent)
            self.canvas.update_mask()
        def undo():
            self.canvas.mask = self.canvas.backup.copy()
            self.canvas.update_mask()
        for title, callback in (("다각형 완료", self.canvas.finish_polygon), ("영역 지우기", clear), ("실행 취소", undo),
                                ("확대 +", lambda: self.canvas.scale(1.25, 1.25)), ("축소 -", lambda: self.canvas.scale(.8, .8)),
                                ("화면 맞춤", lambda: self.canvas.fitInView(self.canvas.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio))):
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)

    def load(self, image, mask=None):
        self.canvas.load(image, mask)

    def png(self):
        return self.canvas.png()


class Comparison(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.view = QGraphicsView()
        self.view.setScene(QGraphicsScene(self))
        self.view.setMinimumHeight(320)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        layout.addWidget(self.view)
        row = QHBoxLayout()
        row.addWidget(QLabel("원본 / 생성 비교"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.update_split)
        row.addWidget(self.slider)
        for label, factor in (("확대 +", 1.25), ("축소 -", .8)):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, value=factor: self.view.scale(value, value))
            row.addWidget(button)
        layout.addLayout(row)
        self.original = QPixmap()
        self.overlay = None

    def load(self, original, result):
        self.original = QPixmap(str(original))
        generated = QPixmap(str(result))
        if self.original.isNull() or generated.size() != self.original.size():
            raise ValueError("비교 이미지 크기 오류")
        self.view.scene().clear()
        self.view.scene().addPixmap(generated)
        self.overlay = self.view.scene().addPixmap(self.original)
        self.view.setSceneRect(0, 0, generated.width(), generated.height())
        self.update_split()
        self.view.fitInView(self.view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_split(self):
        if self.overlay:
            width = round(self.original.width() * self.slider.value() / 100)
            self.overlay.setVisible(width > 0)
            if width:
                self.overlay.setPixmap(self.original.copy(0, 0, width, self.original.height()))
