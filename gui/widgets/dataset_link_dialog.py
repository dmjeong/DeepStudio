"""Explicit recovery after an external image rename, with an overlay preview."""

from pathlib import Path

import numpy as np
from core.dataset_editor import scan_dataset, split_root
from core.dataset_links import validate_mask_link
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from widgets.common import NoWheelComboBox


class DatasetLinkDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.selection = None
        self.setWindowTitle("정답 연결 복구")
        self.resize(780, 620)
        layout = QVBoxLayout(self)
        hint = QLabel("탐색기에서 이미지 이름을 바꾸면 정답은 옛 이름으로 남습니다.\n"
                      "이전 정답과 새 이미지를 선택하고 겹친 영역을 확인하세요.\n"
                      "프로젝트 저장은 이 연결을 자동으로 추측하지 않습니다. 원본 정답은 백업됩니다.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(QLabel("이전 이름의 정답 마스크"))
        self.mask_combo = NoWheelComboBox()
        layout.addWidget(self.mask_combo)
        layout.addWidget(QLabel("연결할 새 이미지 (같은 분할, 아직 정답이 없는 이미지)"))
        self.image_combo = NoWheelComboBox()
        layout.addWidget(self.image_combo)
        self.preview = QLabel("마스크와 이미지를 선택하세요.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 220)
        layout.addWidget(self.preview, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)
        buttons = QHBoxLayout()
        buttons.addStretch()
        close = QPushButton("취소")
        close.clicked.connect(self.reject)
        buttons.addWidget(close)
        self.link_button = QPushButton("이 정답 연결")
        self.link_button.setEnabled(False)
        self.link_button.clicked.connect(self.accept)
        buttons.addWidget(self.link_button)
        layout.addLayout(buttons)

        self.index = scan_dataset(project)
        self.mask_combo.addItem("이전 정답 선택", None)
        for row in self.index.get("orphan_masks", []):
            self.mask_combo.addItem(f"{row['split']} / {row['name']}", row)
        self.mask_combo.currentIndexChanged.connect(self._mask_changed)
        self.image_combo.currentIndexChanged.connect(self._preview)
        self._mask_changed()

    def _mask_changed(self):
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        self.image_combo.addItem("새 이미지 선택", None)
        mask = self.mask_combo.currentData()
        if mask:
            for row in self.index["images"]:
                if row["split"] == mask["split"] and not row["annotated"]:
                    relative = Path(row["path"]).relative_to(split_root(self.project, row["split"]))
                    self.image_combo.addItem(str(relative), row["path"])
        self.image_combo.blockSignals(False)
        self._preview()

    def _preview(self):
        self.selection = None
        self.link_button.setEnabled(False)
        self.preview.clear()
        mask, image = self.mask_combo.currentData(), self.image_combo.currentData()
        if not mask or not image:
            self.preview.setText("이전 정답과 새 이미지를 직접 선택하세요.")
            self.details.setText("이름·순서·크기만으로 정답을 자동 연결하지 않습니다.")
            return
        try:
            _, _, pixels = validate_mask_link(self.project, image, mask["path"], mask["split"])
            with Image.open(image) as source:
                original_size = source.size
                source = source.convert("RGB")
                source.thumbnail((720, 380))
                rgb = np.array(source)
            small = np.array(Image.fromarray(pixels).resize(source.size, Image.Resampling.NEAREST))
            # Show only foreground: background remains the original image.
            colors = np.array([[255, 170, 30], [40, 200, 230], [190, 120, 255],
                               [110, 220, 90], [245, 100, 130]], dtype=np.uint8)
            foreground = (small > 0) & (small != 255)
            rgb[foreground] = (rgb[foreground] * .55 + colors[(small[foreground] - 1) % len(colors)] * .45).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)
            shown = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
            self.preview.setPixmap(QPixmap.fromImage(shown))
            ids = [int(i) for i in np.unique(pixels) if i not in (0, 255)]
            classes = ", ".join(self.project.data.class_names[i] for i in ids) or "배경만 있음"
            self.details.setText(f"{mask['path']}\n→ {image}\n"
                                 f"{original_size[0]} × {original_size[1]} | 정답: {classes}\n"
                                 "영역이 해당 이미지의 실제 결함과 맞는지 확인한 뒤 연결하세요.")
            self.selection = (image, mask["path"], mask["split"])
            self.link_button.setEnabled(True)
        except (OSError, ValueError) as exc:
            self.preview.setText("연결할 수 없습니다.")
            self.details.setText(str(exc))
