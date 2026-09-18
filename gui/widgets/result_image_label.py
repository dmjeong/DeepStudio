"""원본 이미지와 추론 결과를 분리하여 표시하는 이미지 라벨."""

from PySide6.QtCore import Qt, QRect, QSize, QPointF, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QStyle


class ResultImageLabel(QLabel):
    """크기가 바뀌어도 원본에서 다시 표시하고 결과 배지는 화면 위에 그린다."""

    view_changed = Signal(float, object)

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._source_pixmap = QPixmap()
        self._result_text = ""
        self._zoom = 1.0
        self._pan = QPointF()
        self._drag = None
        self._pixel_source = None
        self.setMouseTracking(True)
        self._result_color = "#5590F0"
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMargin(0)
        self.setIndent(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self):
        # 표시 중인 픽스맵 크기가 레이아웃의 최소 크기를 계속 키우지 않게 한다.
        if not self._source_pixmap.isNull():
            return QSize(160, 160)
        return super().sizeHint()

    def minimumSizeHint(self):
        return QSize(0, 0)

    def setPixmap(self, pixmap: QPixmap):
        """배지가 없는 원본을 보관하고 현재 화면 크기에 맞춰 표시한다."""
        if self._source_pixmap.size() != pixmap.size():
            self._zoom, self._pan = 1.0, QPointF()
        self._source_pixmap = QPixmap(pixmap)
        self._update_scaled_pixmap()

    def set_result(self, text: str = "", color: str = "#5590F0"):
        """이미지를 다시 읽지 않고 결과를 즉시 갱신한다."""
        self._result_text = str(text)
        self._result_color = color if QColor(color).isValid() else "#5590F0"
        self.setToolTip(self._result_text)
        self.update()

    def _reset_image(self):
        self._source_pixmap = QPixmap()
        self._result_text = ""
        self.setToolTip("")

    def setText(self, text: str):
        self._reset_image()
        super().setText(text)

    def clear(self):
        self._reset_image()
        super().clear()

    def _content_rect(self):
        margin = self.margin()
        return self.contentsRect().adjusted(margin, margin, -margin, -margin)

    def _update_scaled_pixmap(self):
        if self._source_pixmap.isNull():
            super().clear()
            self.update()
            return
        rect = self._content_rect()
        if rect.isEmpty():
            return
        ratio = self.devicePixelRatioF()
        target = QSize(max(1, round(rect.width() * ratio)),
                       max(1, round(rect.height() * ratio)))
        scaled = self._source_pixmap.scaled(
            target, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        scaled.setDevicePixelRatio(ratio)
        super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._source_pixmap.isNull():
            self._update_scaled_pixmap()

    def _image_rect(self):
        """QLabel의 정렬과 실제 표시 픽스맵 크기로 이미지 영역을 구한다."""
        rect = self._content_rect()
        if self._source_pixmap.isNull():
            return rect
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return rect
        ratio = pixmap.devicePixelRatioF()
        size = QSize(round(pixmap.width() / ratio), round(pixmap.height() / ratio))
        if self._zoom != 1:
            size = QSize(round(size.width() * self._zoom), round(size.height() * self._zoom))
        aligned = QStyle.alignedRect(self.layoutDirection(), self.alignment(), size, rect)
        return aligned.translated(round(self._pan.x()), round(self._pan.y()))

    def _badge_font(self):
        font = QFont(self.font())
        font.setPointSizeF(10)
        font.setBold(True)
        return font

    def _badge_rect(self):
        if not self._result_text:
            return QRect()
        image_rect = self._image_rect().intersected(self._content_rect())
        max_width = image_rect.width() - 8
        max_height = self._content_rect().bottom() - image_rect.top() - 7
        if max_width <= 12 or max_height <= 8:
            return QRect()
        metrics = QFontMetrics(self._badge_font())
        text = " ".join(self._result_text.splitlines())
        width = min(max_width, metrics.horizontalAdvance(text) + 12)
        height = min(max_height, metrics.height() + 8)
        return QRect(image_rect.left() + 4, image_rect.top() + 4, width, height)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._source_pixmap.isNull() and (self._zoom != 1 or not self._pan.isNull()):
            painter = QPainter(self)
            painter.setClipRect(self._content_rect())
            painter.fillRect(self._content_rect(), QColor("#10161d"))
            painter.drawPixmap(self._image_rect(), self._source_pixmap)
            painter.end()
        badge = self._badge_rect()
        if badge.isEmpty():
            return
        painter = QPainter(self)
        try:
            painter.setClipRect(self._content_rect())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setFont(self._badge_font())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(10, 12, 16, 225))
            painter.drawRoundedRect(badge, 4, 4)
            painter.setPen(QColor(self._result_color))
            text_rect = badge.adjusted(6, 4, -6, -4)
            text = " ".join(self._result_text.splitlines())
            text = painter.fontMetrics().elidedText(
                text, Qt.TextElideMode.ElideRight, text_rect.width())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             text)
        finally:
            painter.end()

    def set_view(self, zoom, pan):
        self._zoom, self._pan = float(zoom), QPointF(pan)
        self.update()

    def set_pixel_source(self, pixels):
        self._pixel_source = pixels

    def wheelEvent(self, event):
        if self._source_pixmap.isNull():
            return super().wheelEvent(event)
        self._zoom = min(16.0, max(1.0, self._zoom * (1.2 if event.angleDelta().y() > 0 else 1 / 1.2)))
        if self._zoom == 1:
            self._pan = QPointF()
        self.view_changed.emit(self._zoom, self._pan)
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag is not None and self._zoom > 1:
            self._pan += event.position() - self._drag
            self._drag = event.position()
            self.view_changed.emit(self._zoom, self._pan)
            self.update()
        rect = self._image_rect()
        if not self._source_pixmap.isNull() and rect.contains(event.position().toPoint()):
            x = min(self._source_pixmap.width() - 1, int((event.position().x() - rect.x()) * self._source_pixmap.width() / rect.width()))
            y = min(self._source_pixmap.height() - 1, int((event.position().y() - rect.y()) * self._source_pixmap.height() / rect.height()))
            pixels = self._pixel_source
            value = pixels[y, x].tolist() if pixels is not None and y < pixels.shape[0] and x < pixels.shape[1] else self._source_pixmap.toImage().pixelColor(x, y).getRgb()[:3]
            self.setToolTip(f"원본 좌표 ({x}, {y}) / 픽셀 {value} / 화면 맞춤 대비 {self._zoom:.2f}배\n휠: 확대, 드래그: 이동, 더블 클릭: 화면 맞춤")
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self._zoom, self._pan = 1.0, QPointF()
        self.view_changed.emit(self._zoom, self._pan)
        self.update()
        event.accept()
