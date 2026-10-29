"""Shared Windows annotation workspace for axis-aligned boxes and semantic masks."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from widgets.annotation_io import ThumbnailLoader
from widgets.common import NoWheelComboBox


class AnnotationWorkspaceUI:
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel(
            ("Object detection" if self.task == "detect" else "Segmentation")
            + "  /  "
            + Path(self.image_path).name
        )
        title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        title.setMinimumWidth(0)
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(title, 1)
        self.save_state = QLabel("미작업" if self.save_empty else "정답 불러옴")
        header.addWidget(self.save_state)
        self.inspector_toggle = QPushButton("클래스·객체")
        self.inspector_toggle.clicked.connect(
            lambda: self.inspector.setVisible(not self.inspector.isVisible())
        )
        header.addWidget(self.inspector_toggle)
        shortcut_button = QPushButton("단축키")
        shortcut_button.clicked.connect(lambda: self.shortcuts.edit())
        header.addWidget(shortcut_button)
        self.save_button = QPushButton("정답 저장  Ctrl+S")
        self.save_button.clicked.connect(self._save)
        header.addWidget(self.save_button)
        self.review_button = QPushButton("검수 완료 · 다음  Ctrl+Enter")
        self.review_button.clicked.connect(self._review_next)
        header.addWidget(self.review_button)
        layout.addLayout(header)
        self.restore_bar = QWidget()
        restore = QHBoxLayout(self.restore_bar)
        restore.setContentsMargins(0, 0, 0, 0)
        self.restore_label = QLabel()
        restore.addWidget(self.restore_label, 1)
        self.restore_button = QPushButton("초안 복원")
        self.restore_button.clicked.connect(self._restore_draft)
        restore.addWidget(self.restore_button)
        discard = QPushButton("초안 버리기")
        discard.clicked.connect(self._discard_draft)
        restore.addWidget(discard)
        self.restore_bar.hide()
        layout.addWidget(self.restore_bar)
        properties = QGridLayout()
        properties.addWidget(QLabel("클래스"), 0, 0)
        self.class_combo = NoWheelComboBox()
        self.class_combo.addItems(self.names)
        for i in range(len(self.names)):
            self.class_combo.setItemData(i, i)
        if self.task == "segment":
            if self.names:
                self.class_combo.setItemText(0, self.names[0] + " · 배경(자동)")
                self.class_combo.model().item(0).setEnabled(False)
            self.class_combo.addItem("학습 제외 영역 (255)", 255)
        if self.task == "segment" and len(self.names) > 1:
            self.class_combo.setCurrentIndex(1)
        properties.addWidget(self.class_combo, 0, 1)
        properties.setColumnStretch(1, 1)
        self.apply_class_button = QPushButton("선택에 적용")
        self.apply_class_button.clicked.connect(self._apply_class)
        properties.addWidget(self.apply_class_button, 0, 2)
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 1024)
        self.brush_size.setValue(20)
        self.brush_size.setSuffix(" px")
        self.erase_scope = NoWheelComboBox()
        self.erase_scope.addItem("선택 영역 / 활성 클래스", "selection")
        self.erase_scope.addItem("활성 클래스", "class")
        self.erase_scope.addItem("전체 편집 가능 영역", "all")
        self.background_only = QCheckBox("배경에만 그리기")
        self.brush_tip = NoWheelComboBox()
        self.brush_tip.addItem("원형 브러시", "circle")
        self.brush_tip.addItem("사각 브러시", "square")
        self.brush_tip.setVisible(self.task == "segment")
        properties.addWidget(self.brush_tip, 1, 4)
        self.snap = QCheckBox("스냅")
        properties.addWidget(self.snap, 1, 5)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        self.opacity.setMaximumWidth(110)
        for column, widget in enumerate(
            (self.brush_size, self.erase_scope, self.background_only, self.opacity)
        ):
            widget.setVisible(self.task == "segment")
            properties.addWidget(widget, 1, column)
        self.zoom_label = QLabel("100%")
        properties.addWidget(self.zoom_label, 0, 3)
        fit = QPushButton("화면 맞춤 0")
        fit.clicked.connect(self._fit)
        properties.addWidget(fit, 0, 4)
        actual = QPushButton("1:1")
        actual.clicked.connect(self._actual_size)
        properties.addWidget(actual, 0, 5)
        properties.setContentsMargins(0, 0, 0, 0)
        property_widget = QWidget()
        property_widget.setLayout(properties)
        property_scroll = QScrollArea()
        property_scroll.setWidgetResizable(True)
        property_scroll.setWidget(property_widget)
        property_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        property_scroll.setMinimumHeight(68)
        property_scroll.setMaximumHeight(80)
        layout.addWidget(property_scroll)
        self.editor_splitter = QSplitter(Qt.Orientation.Horizontal)
        tool_widget = QWidget()
        tools = QVBoxLayout(tool_widget)
        tools.setContentsMargins(0, 0, 0, 0)
        self.mode_buttons = {}
        modes = [
            ("select", "선택 V"),
            ("draw", "박스 R"),
            ("erase", "클릭 삭제 E"),
            ("pan", "이동 Space"),
        ]
        if self.task == "segment":
            modes = [
                ("select", "선택 V"),
                ("rectangle", "사각형 R"),
                ("polygon", "다각형 P"),
                ("brush", "브러시 B"),
                ("erase", "지우개 E"),
                ("fill", "채우기 F"),
                ("pan", "이동 Space"),
                ("ai_point", "AI 점"),
                ("ai_box", "AI 박스"),
            ]
        for mode, label in modes:
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, m=mode: self._set_mode(m))
            self.mode_buttons[mode] = button
            tools.addWidget(button)
        self.finish_button = QPushButton("다각형 완료 ↵")
        self.finish_button.clicked.connect(self._finish_polygon)
        self.finish_button.setVisible(self.task == "segment")
        tools.addWidget(self.finish_button)
        self.new_region_button = QPushButton("새 영역")
        self.new_region_button.clicked.connect(self._new_region)
        self.new_region_button.setVisible(self.task == "segment")
        tools.addWidget(self.new_region_button)
        self.grid_button = QPushButton("격자 복제 G")
        self.grid_button.clicked.connect(self._grid)
        tools.addWidget(self.grid_button)
        self.undo_button = QPushButton("실행 취소")
        self.redo_button = QPushButton("다시 실행")
        self.undo_button.clicked.connect(self._undo)
        self.redo_button.clicked.connect(self._redo)
        tools.addWidget(self.undo_button)
        tools.addWidget(self.redo_button)
        tools.addStretch()
        hint = QLabel("Space 이동\n휠 확대\nCtrl+Z 취소\nJ / K 탐색")
        hint.setObjectName("annotation-shortcut-hint")
        tools.addWidget(hint)
        tool_widget.setMaximumWidth(132)
        tool_scroll = QScrollArea()
        tool_scroll.setWidgetResizable(True)
        tool_scroll.setWidget(tool_widget)
        tool_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tool_scroll.setMinimumWidth(105)
        tool_scroll.setMaximumWidth(145)
        self.editor_splitter.addWidget(tool_scroll)
        if self.task == "segment":
            from widgets.segmentation_canvas import SegmentationCanvas

            self.view = SegmentationCanvas(self.pixmap, self)
            self.view.shape_created.connect(self._create)
            self.view.shape_edited.connect(self._edit)
            self.view.shapes_edited.connect(self._edit_many)
            self.view.pending_changed.connect(self._pending)
            self.view.prompt_requested.connect(self._ai_prompt)
            self.brush_size.valueChanged.connect(
                lambda v: setattr(self.view, "brush_size", v)
            )
            self.opacity.valueChanged.connect(self.view.set_opacity)
            self.erase_scope.currentIndexChanged.connect(
                lambda _: setattr(
                    self.view, "erase_scope", self.erase_scope.currentData()
                )
            )
            self.background_only.toggled.connect(
                lambda v: setattr(self.view, "background_only", v)
            )
            self.brush_tip.currentIndexChanged.connect(
                lambda _: setattr(self.view, "brush_tip", self.brush_tip.currentData())
            )
        else:
            from widgets.detection_canvas import DetectionCanvas

            self.view = DetectionCanvas(self.pixmap, self)
            self.view.box_created.connect(self._create_box)
            self.view.box_edited.connect(self._edit_box)
            self.view.boxes_edited.connect(self._edit_many)
            self.view.box_deleted.connect(self._remove)
            self.view.pending_changed.connect(self._pending)
        self.scene = self.view.scene()
        self.view.selection_changed.connect(self._select)
        self.snap.toggled.connect(lambda v: setattr(self.view, "snap_enabled", v))
        self.view.selection_set_changed.connect(self._select_many)
        self.view.command.connect(self._command)
        self.view.zoom_requested.connect(self._zoom)
        self.class_combo.currentIndexChanged.connect(self._active_class)
        self.class_combo.activated.connect(self._class_picked)  # 사용자가 콤보에서 고를 때만
        self.editor_splitter.addWidget(self.view)
        self.inspector = QTabWidget()
        self.inspector.setMinimumWidth(355)
        classes = QWidget()
        class_layout = QVBoxLayout(classes)
        self.class_search = QLineEdit()
        self.class_search.setPlaceholderText("클래스 검색")
        self.class_search.textChanged.connect(self._filter_classes)
        class_layout.addWidget(self.class_search)
        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(self._class_list_selected)
        class_layout.addWidget(self.class_list, 1)
        self.add_class_button = QPushButton("클래스 추가")
        self.add_class_button.setEnabled(self._add_class_callback is not None)
        self.add_class_button.clicked.connect(self._add_class)
        class_layout.addWidget(self.add_class_button)
        self.inspector.addTab(classes, "클래스")
        objects = QWidget()
        object_layout = QVBoxLayout(objects)
        self.object_summary = QLabel()
        object_layout.addWidget(self.object_summary)
        self.table = QTableWidget(0, 5)
        # The application's roomy form controls are too large inside 32px rows.
        self.table.setStyleSheet("""
            QTableWidget::item { padding: 3px 4px; }
            QHeaderView::section { padding: 5px 3px; font-size: 11px; }
            QTableWidget QComboBox {
                padding: 1px 18px 1px 5px; min-height: 22px; border-radius: 4px;
            }
            QTableWidget QComboBox::drop-down { width: 18px; }
            QTableWidget QComboBox::down-arrow { margin-right: 4px; }
            QTableWidget QPushButton {
                padding: 0px; min-height: 22px; border-radius: 4px;
            }
            QTableWidget QCheckBox { padding: 0px; margin-left: 10px; }
        """)
        self.table.setHorizontalHeaderLabels(["영역", "클래스", "삭제", "잠금", "표시"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.setColumnWidth(0, 78)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(32)
        for c in (2, 3, 4):
            self.table.setColumnWidth(c, 40)
        self.table.itemSelectionChanged.connect(self._table_selection)
        object_layout.addWidget(self.table, 1)
        edits = QHBoxLayout()
        duplicate = QPushButton("선택 복제")
        duplicate.clicked.connect(self._duplicate)
        edits.addWidget(duplicate)
        remove = QPushButton("선택 삭제")
        remove.clicked.connect(lambda: self._remove())
        edits.addWidget(remove)
        object_layout.addLayout(edits)
        if self.task == "segment":
            split_components = QPushButton("선택 영역의 연결 부분 나누기")
            split_components.clicked.connect(self._split_components)
            object_layout.addWidget(split_components)
        self.empty_confirm = QCheckBox("객체 없음 / 배경만 있음 확인")
        self.empty_confirm.toggled.connect(self._empty_changed)
        object_layout.addWidget(self.empty_confirm)
        self.inspector.addTab(objects, "객체")
        self.history = QListWidget()
        self.inspector.addTab(self.history, "이력")
        self.inspector.setCurrentIndex(1)
        self.ai_panel = QWidget()
        ai = QVBoxLayout(self.ai_panel)
        ai.addWidget(QLabel("로컬 SAM2 · 제안을 확인한 뒤 적용"))
        self.ai_model = NoWheelComboBox()
        for name, key in [
            ("Hiera Tiny", "sam2_hiera_tiny"),
            ("Hiera Small", "sam2_hiera_small"),
            ("Hiera Base+", "sam2_hiera_base_plus"),
            ("Hiera Large", "sam2_hiera_large"),
        ]:
            self.ai_model.addItem(name, key)
        ai.addWidget(self.ai_model)
        self.ai_device = NoWheelComboBox()
        self.ai_device.addItems(["자동", "CPU"])
        ai.addWidget(self.ai_device)
        self.ai_model.currentIndexChanged.connect(self._reset_ai_points)
        self.ai_device.currentIndexChanged.connect(self._reset_ai_points)
        self.ai_status = QLabel(
            "AI 점: 왼쪽 포함 / 오른쪽 제외\nAI 박스: 영역을 드래그"
        )
        self.ai_status.setWordWrap(True)
        ai.addWidget(self.ai_status)
        self.ai_apply = QPushButton("제안 적용")
        self.ai_apply.setEnabled(False)
        self.ai_apply.clicked.connect(self._apply_proposal)
        ai.addWidget(self.ai_apply)
        cancel = QPushButton("제안 취소")
        cancel.clicked.connect(self._cancel_ai)
        ai.addWidget(cancel)
        clear = QPushButton("입력점 초기화")
        clear.clicked.connect(self._reset_ai_points)
        ai.addWidget(clear)
        ai.addStretch()
        if self.task == "segment":
            self.inspector.addTab(self.ai_panel, "AI")
        display = QWidget()
        display_layout = QVBoxLayout(display)
        label = QLabel(
            "원본 확인용 표시 조절입니다. 원본 파일과 학습 정답은 바뀌지 않습니다."
        )
        label.setWordWrap(True)
        display_layout.addWidget(label)
        self.annotation_visible = QCheckBox("정답 오버레이 표시")
        self.annotation_visible.setChecked(True)
        self.annotation_visible.toggled.connect(self._set_annotation_visibility)
        display_layout.addWidget(self.annotation_visible)
        if self.task == "detect":
            labels = QCheckBox("박스 이름 표시 · 작은 박스는 선택 시")
            labels.setChecked(True)
            labels.toggled.connect(self._set_label_visibility)
            display_layout.addWidget(labels)
        self.brightness = QSlider(Qt.Orientation.Horizontal)
        self.brightness.setRange(-100, 100)
        self.contrast = QSlider(Qt.Orientation.Horizontal)
        self.contrast.setRange(-100, 100)
        for label, slider in [("밝기", self.brightness), ("대비", self.contrast)]:
            display_layout.addWidget(QLabel(label))
            display_layout.addWidget(slider)
            slider.valueChanged.connect(lambda _: self._levels_timer.start())
        reset_levels = QPushButton("원본 표시로 복귀")
        reset_levels.clicked.connect(self._reset_levels)
        display_layout.addWidget(reset_levels)
        display_layout.addStretch()
        self.inspector.addTab(display, "보기")
        self.editor_splitter.addWidget(self.inspector)
        self.editor_splitter.setSizes([125, 900, 330])
        self.editor_splitter.setStretchFactor(1, 1)
        self.editor_splitter.setChildrenCollapsible(True)
        layout.addWidget(self.editor_splitter, 1)
        self.proposal_bar = QWidget()
        proposal = QHBoxLayout(self.proposal_bar)
        proposal.setContentsMargins(0, 0, 0, 0)
        self.proposal_label = QLabel()
        proposal.addWidget(self.proposal_label, 1)
        self.proposal_apply = QPushButton("제안 적용 Enter")
        self.proposal_apply.clicked.connect(self._apply_proposal)
        proposal.addWidget(self.proposal_apply)
        cancel_proposal = QPushButton("제안 취소")
        cancel_proposal.clicked.connect(self._cancel_ai)
        proposal.addWidget(cancel_proposal)
        self.proposal_bar.hide()
        layout.addWidget(self.proposal_bar)
        navigation = QHBoxLayout()
        self.previous_button = QPushButton("이전 J")
        self.next_button = QPushButton("다음 K")
        self.previous_button.clicked.connect(lambda: self._navigate(-1))
        self.next_button.clicked.connect(lambda: self._navigate(1))
        navigation.addWidget(self.previous_button)
        self.thumbnail_filter = NoWheelComboBox()
        self.thumbnail_filter.addItems(
            ["근처 전체", "미작업", "작업 중", "초안", "정답 있음", "검수 완료", "오류"]
        )
        self.thumbnail_filter.currentTextChanged.connect(self._filter_thumbnails)
        navigation.addWidget(self.thumbnail_filter)
        self.thumbnails = QListWidget()
        self.thumbnails.setFlow(QListWidget.Flow.LeftToRight)
        self.thumbnails.setMaximumHeight(82)
        self.thumbnails.setWrapping(False)
        self.thumbnails.setIconSize(QSize(64, 48))
        self._thumbnail_items = {}
        if self.image_paths:
            current = self.navigation[0] if self.navigation else 0
            # Keep the strip bounded even for a project with 100,000 images.
            nearby = range(
                max(0, current - 10), min(len(self.image_paths), current + 11)
            )
            for index in nearby:
                path = self.image_paths[index]
                item = QListWidgetItem(Path(path).name)
                item.setData(Qt.ItemDataRole.UserRole, index)
                self.thumbnails.addItem(item)
                self._thumbnail_items[index] = item
                if index == current:
                    self.thumbnails.setCurrentItem(item)
            self.thumbnails.itemDoubleClicked.connect(
                lambda item: self._navigate_to(int(item.data(Qt.ItemDataRole.UserRole)))
            )
        else:
            item = QListWidgetItem(Path(self.image_path).name)
            self.thumbnails.addItem(item)
        navigation.addWidget(self.thumbnails, 1)
        navigation.addWidget(
            QLabel(
                f"{self.navigation[0] + 1} / {self.navigation[1]}"
                if self.navigation
                else "1 / 1"
            )
        )
        navigation.addWidget(self.next_button)
        layout.addLayout(navigation)
        self.status = QLabel(
            "클래스를 고른 뒤 영역을 그리세요. 저장 시 학습용 정답을 확정합니다."
        )
        self.status.setWordWrap(True)
        self.view.status_changed.connect(self.status.setText)
        layout.addWidget(self.status)
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(5, 1600)
        self.zoom.setValue(100)
        self.zoom.valueChanged.connect(self._zoom_value)
        self.zoom.hide()  # Kept for integrations; fit and 1:1 use actual scale.
        self.cancel_draw = QPushButton()
        self.cancel_draw.hide()
        self.cancel_draw.clicked.connect(self.view.cancel_gesture)
        self._active_class(self.class_combo.currentIndex())
        self._set_mode("polygon" if self.task == "segment" else "draw")
        paths = (
            [(i, self.image_paths[i]) for i in self._thumbnail_items]
            if self.image_paths
            else [(0, self.image_path)]
        )
        if not self.image_paths:
            self._thumbnail_items[0] = self.thumbnails.item(0)
        self.thumbnail_worker = ThumbnailLoader(paths, self._thumbnail_state)
        self.thumbnail_worker.ready.connect(self._thumbnail_ready)
        self.thumbnail_worker.start()

    def _thumbnail_ready(self, value):
        index, image, state = value
        item = self._thumbnail_items.get(index)
        if item is not None:
            if not image.isNull():
                item.setIcon(QIcon(QPixmap.fromImage(image)))
            current = self.navigation[0] if self.navigation else 0
            if index == current:
                state = self._current_thumbnail_state
            self._set_thumbnail_state(index, state)

    def _thumbnail_state(self, path):
        record = self.image_records.get(str(Path(path).resolve()), {})
        default = (
            "오류"
            if record.get("error")
            else "정답 있음"
            if record.get("annotated")
            else "미작업"
        )
        if self.revision_provider is None:
            return default
        try:
            value = json.loads(
                self.drafts.path(path, self.task).read_text(encoding="utf-8")
            )
            if (
                value.get("image_revision") != self.drafts.image_revision(path)
                or value.get("names") != self.names
                or value.get("label_revision") != self.revision_provider(path)
            ):
                return default
            return (
                "초안"
                if value.get("state") == "draft"
                else "검수 완료"
                if value.get("reviewed")
                else "정답 있음"
            )
        except (OSError, ValueError, TypeError, AttributeError):
            return default

    def _set_thumbnail_state(self, index, state):
        item = self._thumbnail_items.get(index)
        if item is None:
            return
        path = self.image_paths[index] if self.image_paths else self.image_path
        item.setText(Path(path).name + "\n" + state)
        item.setData(int(Qt.ItemDataRole.UserRole) + 1, state)
        self._filter_thumbnails()

    def _current_state(self, state):
        self._current_thumbnail_state = state
        if hasattr(self, "_thumbnail_items"):
            self._set_thumbnail_state(
                self.navigation[0] if self.navigation else 0, state
            )

    def _filter_thumbnails(self):
        value = self.thumbnail_filter.currentText()
        for item in self._thumbnail_items.values():
            item.setHidden(
                value != "근처 전체"
                and item.data(int(Qt.ItemDataRole.UserRole) + 1) != value
            )
