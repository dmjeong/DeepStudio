"""
Deep Vision Studio — 데이터셋 관리 페이지

사용자 워크플로우:
┌──────────────────────────────────────────────────────────────┐
│ 1. 프로젝트 생성 시 데이터 폴더 자동 생성 완료               │
│ 2. 사용자는 이미지만 추가 (드래그&드롭 or 파일 선택)         │
│ 3. train / val / test 탭에서 각 분할의 이미지 관리           │
│ 4. 클래스별 폴더에 이미지 자동 복사                          │
│ 5. 통계 자동 갱신                                            │
└──────────────────────────────────────────────────────────────┘

폴더 구조 (프로젝트 생성 시 자동):
┌── Classification ──────────┐  ┌── Anomaly ────────────────┐
│ data/                      │  │ data/                     │
│ ├── train/{cls1,cls2,...}/ │  │ ├── train/good/           │
│ ├── val/{cls1,cls2,...}/   │  │ └── test/{good,defect}/   │
│ └── test/{cls1,cls2,...}/  │  └───────────────────────────┘
└────────────────────────────┘
┌── Segmentation ────────────┐  ┌── Detection ──────────────┐
│ data/                      │  │ data/                     │
│ ├── images/{train,val,test}│  │ ├── images/{train,val,test}│
│ └── masks/{train,val,test} │  │ └── labels/{train,val,test}│
└────────────────────────────┘  └───────────────────────────┘
"""

import os
import json
import shutil
import copy
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFrame,
    QFileDialog, QMessageBox, QScrollArea,
    QGroupBox, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QSlider,
    QTabWidget, QInputDialog,
    QTabBar, QMenu, QApplication,
)
from PySide6.QtCore import Qt, Signal, QMimeData, QByteArray, QThread
from PySide6.QtGui import (
    QPixmap, QDragEnterEvent, QDropEvent, QDrag,
)

from core.project import ProjectData, SUPPORTED_TASKS
from core.class_management import ClassManager, move_image_with_sidecars
# 휠 오조작 방지 콤보박스 (분할/클래스 선택이 스크롤 중 바뀌지 않도록)
from widgets.common import NoWheelComboBox


# ── 이미지 확장자 ──
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# ── 내부 드래그&드롭 MIME 타입 ──
#    외부 파일 드롭(URL)과 구분하기 위한 전용 포맷
IMAGE_MIME_TYPE = "application/x-custom_csp-dataset-images"

# ── 분할 목록 (탭 라벨은 텍스트 온리) ──
SPLITS = ["train", "val", "test"]


class _ClassOperationThread(QThread):
    """라벨 검증/백업은 작업 스레드, 확인 대화상자와 화면 갱신은 GUI 스레드."""

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.operation()
        except Exception as exc:
            self.error = str(exc)


class SplitDropTabBar(QTabBar):
    """
    이미지를 드롭할 수 있는 탭 바

    ┌──────────────────────────────────────────────────────────┐
    │  썸네일을 끌어서 탭 헤더 위에 놓으면 해당 분할로 이동      │
    │                                                          │
    │   [ TRAIN] [ VAL] [ TEST]                          │
    │        └── 드래그 중인 이미지를 여기 놓으면 ─┘            │
    │            train → test 로 파일 이동                      │
    └──────────────────────────────────────────────────────────┘
    """

    images_dropped = Signal(int, object)   # (탭 인덱스, QMimeData)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._hover_index = -1

    def dragEnterEvent(self, event):
        """내부 이미지 드래그만 수락"""
        if event.mimeData().hasFormat(IMAGE_MIME_TYPE):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """커서 아래 탭을 하이라이트"""
        if not event.mimeData().hasFormat(IMAGE_MIME_TYPE):
            event.ignore()
            return
        idx = self.tabAt(event.position().toPoint())
        if idx != self._hover_index:
            self._hover_index = idx
            self.update()
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._hover_index = -1
        self.update()

    def dropEvent(self, event):
        """드롭된 탭 인덱스로 이동 요청 발신"""
        idx = self.tabAt(event.position().toPoint())
        self._hover_index = -1
        self.update()
        if idx >= 0 and event.mimeData().hasFormat(IMAGE_MIME_TYPE):
            self.images_dropped.emit(idx, event.mimeData())
            event.acceptProposedAction()
        else:
            event.ignore()


class ImageThumbnail(QFrame):
    """
    이미지 썸네일 위젯 (파일명 + 미리보기)

    지원 인터랙션:
    ┌──────────────────────────────────────────────────────────┐
    │  클릭          → 선택 (Ctrl/Shift 클릭으로 다중 선택)     │
    │  드래그        → 탭 헤더로 끌어 다른 분할로 이동           │
    │  우클릭        → 컨텍스트 메뉴 (분할 이동 / 클래스 변경)   │
    └──────────────────────────────────────────────────────────┘
    """

    clicked = Signal(object, object)        # (self, KeyboardModifiers)
    drag_started = Signal(object)           # (self)
    context_requested = Signal(object, object)   # (self, 전역 QPoint)
    opened = Signal(str)

    # ── 스타일 (일반 / 선택됨) — Premium Dashboard 팔레트 ──
    STYLE_NORMAL = """
        QFrame#thumb {
            background-color: #151920;
            border: 1px solid #1e222c;
            border-radius: 6px;
        }
        QFrame#thumb:hover { border-color: #5590F0; }
    """
    STYLE_SELECTED = """
        QFrame#thumb {
            background-color: #1a2540;
            border: 2px solid #5590F0;
            border-radius: 6px;
        }
    """

    def __init__(self, image_path: str, label: str = "",
                 split: str = "", parent=None, thumbnail=None):
        super().__init__(parent)
        self.image_path = image_path
        self.class_label = label      # 소속 클래스 (classify/anomaly)
        self.split = split            # 소속 분할 (train/val/test)
        self._selected = False
        self._drag_origin = None

        self.setObjectName("thumb")
        self.setFixedSize(140, 176)
        self.setStyleSheet(self.STYLE_NORMAL)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # 우클릭 메뉴를 직접 처리
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # 썸네일 이미지
        img_label = QLabel()
        img_label.setFixedSize(132, 120)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setStyleSheet("border: none; background: transparent;")

        if thumbnail is not None or os.path.isfile(image_path):
            pixmap = QPixmap.fromImage(thumbnail) if thumbnail is not None else QPixmap(image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    132, 120,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                img_label.setPixmap(pixmap)
            else:
                img_label.setText("—")
        else:
            img_label.setText("—")

        layout.addWidget(img_label)

        # 파일명
        name_label = QLabel(os.path.basename(image_path)[:18])
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet(
            "color: #8A92A4; font-size: 10px; border: none; background: transparent;"
        )
        layout.addWidget(name_label)

        # 클래스 배지 (어느 클래스 소속인지 한눈에)
        if label:
            badge = QLabel(label[:16])
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(
                "color: #5590F0; font-size: 10px; font-weight: bold;"
                "border: none; background: transparent;"
            )
            layout.addWidget(badge)

    # ── 선택 상태 ──
    def set_selected(self, selected: bool):
        """선택 상태 변경 → 테두리 스타일 갱신"""
        self._selected = selected
        self.setStyleSheet(
            self.STYLE_SELECTED if selected else self.STYLE_NORMAL
        )

    def is_selected(self) -> bool:
        return self._selected

    # ── 마우스 인터랙션 ──
    def mousePressEvent(self, event):
        """좌클릭 → 선택 / 드래그 시작점 기록"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.position().toPoint()
            self.clicked.emit(self, event.modifiers())
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """격자에서는 확대 조작을 제공하지 않고, 더블 클릭으로 상세 이미지만 연다."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.opened.emit(self.image_path)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        """일정 거리 이상 끌면 드래그 시작"""
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if self._drag_origin is None:
            return
        moved = (event.position().toPoint() - self._drag_origin).manhattanLength()
        if moved < QApplication.startDragDistance():
            return
        self._drag_origin = None
        self.drag_started.emit(self)

    def contextMenuEvent(self, event):
        """우클릭 → 부모에게 메뉴 요청"""
        self.context_requested.emit(self, event.globalPos())


class DatasetImagePreviewDialog(QDialog):
    """데이터셋 원본을 확대해 확인하는 상세 보기 창."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"이미지 보기: {os.path.basename(image_path)}")
        self.resize(900, 700)
        self._source = QPixmap(image_path)

        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("확대"))
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(25, 400)
        self.zoom.setValue(100)
        self.zoom.setAccessibleName("데이터셋 이미지 확대")
        self.zoom.valueChanged.connect(self._refresh)
        toolbar.addWidget(self.zoom, 1)
        self.zoom_label = QLabel("100%")
        toolbar.addWidget(self.zoom_label)
        fit = QPushButton("화면 맞춤")
        fit.clicked.connect(lambda: self.zoom.setValue(100))
        toolbar.addWidget(fit)
        close = QPushButton("닫기")
        close.clicked.connect(self.accept)
        toolbar.addWidget(close)
        layout.addLayout(toolbar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setStyleSheet("background: #0a0c10;")
        self.scroll.setWidget(self.image)
        layout.addWidget(self.scroll, 1)
        self._refresh(100)

    def _refresh(self, value):
        if self._source.isNull():
            self.image.setText("이미지 표시 불가")
            return
        scale = float(value) / 100.0
        size = self._source.size()
        self.image.setPixmap(self._source.scaled(max(1, int(size.width() * scale)),
                                                max(1, int(size.height() * scale)),
                                                Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))
        self.image.adjustSize()
        self.zoom_label.setText(f"{value}%")



class DatasetWidget(QWidget):
    """
    데이터셋 관리 페이지

    핵심 기능:
    - 프로젝트 데이터 폴더 자동 인식
    - 이미지 추가 (파일 선택 → 자동 복사)
    - train / val / test 탭으로 분할 관리
    - 클래스별 이미지 통계
    - 드래그 & 드롭 지원
    """

    about_to_change = Signal()
    project_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._class_worker = None
        self._class_callback = None
        self.setAcceptDrops(True)  # 외부 파일 드래그&드롭 활성화

        # 상단 콤보 ↔ 우측 탭 양방향 동기화 중 재귀 호출 방지 플래그
        self._syncing = False
        self._class_filter = "전체"
        self._filter_project_key = None
        # 선택된 이미지 경로 집합 (분할 이동/삭제 대상)
        self._selected_paths = set()
        # 분할별 썸네일 위젯 목록 {split: [ImageThumbnail, ...]}
        self._thumbs = {s: [] for s in SPLITS}
        # Shift 범위 선택 기준점 (split, index)
        self._last_clicked = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)

        # ── 타이틀 ──
        title = QLabel("데이터셋 관리")
        title.setObjectName("page_title")
        layout.addWidget(title)

        self.task_label = QLabel("프로젝트를 생성하면 데이터 폴더 구조가 자동으로 만들어집니다.")
        self.task_label.setObjectName("page_subtitle")
        self.task_label.setWordWrap(True)
        layout.addWidget(self.task_label)

        # ── 상단: 이미지 추가 컨트롤 ──
        add_group = QGroupBox("이미지 추가")
        add_outer = QVBoxLayout(add_group)
        add_layout = QHBoxLayout()
        add_outer.addLayout(add_layout)

        # 대상 분할 선택 — 우측 탭과 양방향 동기화됨
        add_layout.addWidget(QLabel("대상:"))
        self.split_combo = NoWheelComboBox()
        self.split_combo.addItems(SPLITS)
        self.split_combo.setFixedWidth(100)
        self.split_combo.currentTextChanged.connect(self._on_top_split_changed)
        add_layout.addWidget(self.split_combo)

        # 클래스 선택 (classification/detect 전용) — 탭 필터와 동기화됨
        self.class_label_widget = QLabel("클래스:")
        add_layout.addWidget(self.class_label_widget)
        self.class_combo = NoWheelComboBox()
        self.class_combo.setFixedWidth(150)
        self.class_combo.currentTextChanged.connect(self._on_top_class_changed)
        add_layout.addWidget(self.class_combo)

        add_layout.addStretch()
        add_layout = QHBoxLayout()
        add_outer.addLayout(add_layout)

        # 이미지 추가 버튼
        self.add_files_btn = QPushButton("이미지 추가")
        self.add_files_btn.setProperty("cssClass", "primary")
        self.add_files_btn.clicked.connect(self._add_images)
        add_layout.addWidget(self.add_files_btn)

        # 폴더 추가 버튼
        self.add_folder_btn = QPushButton("폴더에서 추가")
        self.add_folder_btn.clicked.connect(self._add_folder)
        add_layout.addWidget(self.add_folder_btn)

        # 클래스 추가 버튼
        self.add_class_btn = QPushButton("클래스 추가")
        self.add_class_btn.clicked.connect(self._add_class)
        add_layout.addWidget(self.add_class_btn)

        # 클래스 삭제 버튼
        self.del_class_btn = QPushButton("클래스 삭제")
        self.del_class_btn.setStyleSheet("""
            QPushButton {
                color: #E05555;
            }
            QPushButton:hover {
                background-color: #2a1520;
            }
        """)
        self.del_class_btn.clicked.connect(self._delete_class)
        add_layout.addWidget(self.del_class_btn)

        add_layout.addStretch()

        # 드래그&드롭 안내
        drop_label = QLabel(
            "파일을 끌어다 놓아 추가  ·  우클릭 또는 드래그로 분할 간 이동"
        )
        drop_label.setStyleSheet("color: #555D70; font-size: 11px;")
        drop_label.setWordWrap(True)
        add_outer.addWidget(drop_label)
        self.annotation_row = QWidget()
        annotation_layout = QHBoxLayout(self.annotation_row)
        annotation_layout.setContentsMargins(0, 0, 0, 0)
        self.annotate_btn = QPushButton("데이터 티칭 / 정답 그리기")
        self.annotate_btn.clicked.connect(self._edit_selected_annotation)
        annotation_layout.addWidget(self.annotate_btn)
        annotation_hint = QLabel("이미지 선택 후 편집하거나 더블클릭하세요. 클래스는 이미지 전체가 아닌 객체마다 지정합니다.")
        annotation_hint.setWordWrap(True)
        annotation_layout.addWidget(annotation_hint, 1)
        add_outer.addWidget(self.annotation_row)
        self.annotation_row.hide()

        layout.addWidget(add_group)

        # ── 중앙: 탭 (Train / Val / Test) + 통계 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 왼쪽: 통계 패널
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 0, 0, 0)

        stats_group = QGroupBox("데이터 통계")
        stats_inner = QVBoxLayout(stats_group)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(
            ["클래스", "Train", "Val", "Test"]
        )
        # 클래스 이름 열만 남는 폭을 흡수하고, 숫자 열은 내용 폭에 맞춘다
        # (전 열 Stretch 는 클래스명을 "NG_d..." 로 잘라먹는다)
        header = self.stats_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        stats_inner.addWidget(self.stats_table)

        # 요약 정보
        self.summary_label = QLabel("이미지를 추가해 주세요.")
        self.summary_label.setStyleSheet("color: #8A92A4; padding: 8px;")
        stats_inner.addWidget(self.summary_label)

        # 폴더 열기 버튼
        open_folder_btn = QPushButton("데이터 폴더 열기")
        open_folder_btn.clicked.connect(self._open_data_folder)
        stats_inner.addWidget(open_folder_btn)

        stats_layout.addWidget(stats_group)
        splitter.addWidget(stats_widget)

        # 오른쪽: 이미지 브라우저 (탭)
        browser_widget = QWidget()
        browser_layout = QVBoxLayout(browser_widget)
        browser_layout.setContentsMargins(0, 0, 0, 0)

        self.split_tabs = QTabWidget()
        # 탭 헤더를 드롭 타깃으로 교체 → 썸네일을 끌어다 놓아 분할 이동
        self._tab_bar = SplitDropTabBar()
        self._tab_bar.images_dropped.connect(self._on_tab_bar_drop)
        self.split_tabs.setTabBar(self._tab_bar)
        # 탭 전환 → 상단 '대상' 콤보 동기화
        self.split_tabs.currentChanged.connect(self._on_tab_changed)

        # Train / Val / Test 탭 생성
        self.tab_scrolls = {}
        self.tab_grids = {}
        self.tab_grid_widgets = {}
        self.tab_filters = {}
        for split_name in SPLITS:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(4, 4, 4, 4)

            # 클래스 필터 — 상단 '클래스' 콤보와 동기화
            filter_row = QHBoxLayout()
            filter_row.addWidget(QLabel("클래스:"))
            class_filter = NoWheelComboBox()
            class_filter.addItem("전체")
            class_filter.setObjectName(f"filter_{split_name}")
            class_filter.currentTextChanged.connect(
                lambda text, sp=split_name: self._on_filter_changed(sp)
            )
            class_filter.setMaximumWidth(260)
            filter_row.addWidget(class_filter)
            filter_row.addStretch(1)          # 남는 폭은 여백이 흡수
            self.tab_filters[split_name] = class_filter

            # 선택 관련 액션
            #   QSS 의 좌우 padding(20px)까지 감안한 폭 — 이보다 좁으면 라벨이 잘린다
            select_all_btn = QPushButton("전체 선택")
            select_all_btn.setFixedWidth(112)
            select_all_btn.clicked.connect(
                lambda _, sp=split_name: self._select_all(sp)
            )
            filter_row.addWidget(select_all_btn)

            clear_sel_btn = QPushButton("선택 해제")
            clear_sel_btn.setFixedWidth(112)
            clear_sel_btn.clicked.connect(self._clear_selection)
            filter_row.addWidget(clear_sel_btn)

            # 이미지 수 표시
            count_label = QLabel("0장")
            count_label.setObjectName(f"count_{split_name}")
            count_label.setStyleSheet("color: #5590F0; font-weight: bold;")
            filter_row.addWidget(count_label)

            tab_layout.addLayout(filter_row)

            # 스크롤 가능 이미지 그리드
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            grid_widget = QWidget()
            grid_widget.setMinimumWidth(760)
            grid = QGridLayout(grid_widget)
            grid.setSpacing(8)
            grid.setAlignment(Qt.AlignmentFlag.AlignTop)
            scroll.setWidget(grid_widget)
            tab_layout.addWidget(scroll)

            self.tab_scrolls[split_name] = scroll
            self.tab_grids[split_name] = grid
            self.tab_grid_widgets[split_name] = grid_widget

            self.split_tabs.addTab(tab, split_name.upper())

        # 선택 상태 안내
        self.selection_label = QLabel(
            "클릭으로 선택 (Ctrl+클릭 = 다중 선택)  ·  "
            "우클릭 메뉴 또는 탭으로 드래그하여 이동"
        )
        self.selection_label.setStyleSheet(
            "color: #555D70; font-size: 11px; padding: 4px;"
        )
        browser_layout.addWidget(self.split_tabs)
        browser_layout.addWidget(self.selection_label)
        splitter.addWidget(browser_widget)

        splitter.setSizes([350, 650])
        layout.addWidget(splitter, stretch=1)

    def set_project(self, project: ProjectData):
        """프로젝트 설정 → 데이터 폴더 자동 인식 + UI 갱신"""
        project_key = os.path.normcase(os.path.realpath(project.project_dir))
        if self._filter_project_key != project_key:
            self._class_filter = "전체"
        self._filter_project_key = project_key
        self.project = project
        self.annotation_row.setVisible(project.task in ("detect", "segment", "obb"))
        available = ["good", "defect"] if project.task == "anomaly" else project.data.class_names
        if self._class_filter != "전체" and self._class_filter not in available:
            self._class_filter = "전체"
        task_info = SUPPORTED_TASKS.get(project.task, {})
        self.task_label.setText(
            f"Task: {task_info.get('name', '')} — "
            f"{task_info.get('description', '')}\n"
            f"Data: {project.data.root}"
        )

        # 콤보/탭을 프로그램이 갱신하는 동안 동기화 시그널 차단
        self._syncing = True

        # 클래스 콤보박스 업데이트
        self.class_combo.clear()
        if project.task == "anomaly":
            # Anomaly: 고정 클래스
            self.class_combo.addItems(["good", "defect"])
            self.class_label_widget.show()
            self.class_combo.show()
            self.add_class_btn.hide()
            self.del_class_btn.hide()
        elif project.task in ("classify", "detect", "segment", "obb"):
            # Segment도 클래스 번호 관리 가능. 이미지 추가 경로는 클래스와 무관.
            for cls_name in project.data.class_names:
                self.class_combo.addItem(cls_name)
            self.class_label_widget.show()
            self.class_combo.show()
            self.add_class_btn.show()
            self.del_class_btn.show()

        # Anomaly는 val 분할을 쓰지 않음 (train=good만, test=good/defect)
        self.split_combo.clear()
        if project.task == "anomaly":
            self.split_combo.addItems(["train", "test"])
            self.split_tabs.setTabEnabled(SPLITS.index("val"), False)
        else:
            self.split_combo.addItems(SPLITS)
            self.split_tabs.setTabEnabled(SPLITS.index("val"), True)

        # 탭 필터 업데이트
        for split_name in SPLITS:
            filter_combo = self.tab_filters.get(split_name)
            if filter_combo:
                filter_combo.clear()
                filter_combo.addItem("전체")
                if project.task == "anomaly":
                    filter_combo.addItems(["good", "defect"])
                else:
                    for cls_name in project.data.class_names:
                        filter_combo.addItem(cls_name)
                filter_combo.setCurrentText(self._class_filter)

        self._syncing = False

        # 선택 초기화 후 데이터 스캔 & UI 갱신
        self._selected_paths.clear()
        self._scan_all()
        # 상단 콤보를 현재 탭에 맞춰 정렬
        self._on_tab_changed(self.split_tabs.currentIndex())

    # ── 이미지 추가 ──────────────────────────────────
    def _add_images(self):
        """파일 선택 → 프로젝트 데이터 폴더에 복사"""
        if not self._ensure_idle():
            return
        if not self.project or not self.project.data.root:
            QMessageBox.warning(self, "알림", "프로젝트를 먼저 생성해 주세요.")
            return

        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "이미지 선택", "",
            "이미지 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if filepaths:
            self._copy_images_to_project(filepaths)

    def _add_folder(self):
        """폴더 선택 → 폴더 내 모든 이미지를 프로젝트에 복사"""
        if not self._ensure_idle():
            return
        if not self.project or not self.project.data.root:
            QMessageBox.warning(self, "알림", "프로젝트를 먼저 생성해 주세요.")
            return

        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if folder:
            # 폴더 내 모든 이미지 파일 수집
            filepaths = []
            for f in sorted(os.listdir(folder)):
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    filepaths.append(os.path.join(folder, f))
            if filepaths:
                self._copy_images_to_project(filepaths)
            else:
                QMessageBox.information(self, "알림", "선택한 폴더에 이미지 파일이 없습니다.")

    def _copy_images_to_project(self, filepaths: list):
        """이미지 파일을 프로젝트 데이터 폴더에 복사"""
        if not self._ensure_idle():
            return
        split = self.split_combo.currentText()
        task = self.project.task

        # 복사 대상 폴더 결정
        target_dir = self._get_target_dir(split, task)
        if target_dir is None:
            return

        os.makedirs(target_dir, exist_ok=True)

        # 복사 실행
        copied = 0
        skipped = 0
        for src in filepaths:
            fname = os.path.basename(src)
            dst = os.path.join(target_dir, fname)
            if os.path.exists(dst):
                skipped += 1
                continue
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception:
                skipped += 1

        # 결과 알림
        self.summary_label.setText(
            f"{copied}장 추가 완료 ({split}/{self.class_combo.currentText()}) "
            f"| 건너뜀: {skipped}장"
        )

        # 새 Detection 이미지는 정답이 없어 클래스 필터에서 숨겨지지 않게 한다.
        if task in ("detect", "obb") and copied:
            self._set_shared_class_filter("전체")
        # 데이터 다시 스캔
        self._scan_all()

    def _get_target_dir(self, split: str, task: str) -> str:
        """복사 대상 디렉토리 결정"""
        data_root = self.project.data.root
        cls_name = self.class_combo.currentText()

        if task == "classify":
            # data/{split}/{class}/
            if not cls_name:
                QMessageBox.warning(self, "알림", "대상 클래스를 먼저 선택해 주세요.")
                return None
            return os.path.join(data_root, split, cls_name)

        elif task == "segment":
            # data/images/{split}/
            return os.path.join(data_root, "images", split)

        elif task in ("detect", "obb"):
            # data/images/{split}/
            return os.path.join(data_root, "images", split)

        elif task == "anomaly":
            # Anomaly: train→good만, test→good/defect
            if split == "train":
                return os.path.join(data_root, "train", "good")
            else:
                if not cls_name:
                    cls_name = "good"
                return os.path.join(data_root, "test", cls_name)

        return None

    def _ensure_idle(self):
        if not self.project:
            return False
        window = self.window()
        if hasattr(window, "has_active_jobs") and window.has_active_jobs():
            QMessageBox.warning(self, "데이터 변경 불가", "실행 중인 작업 종료 후 변경 가능")
            return False
        return self._class_worker is None

    def _start_class_operation(self, operation, callback, message):
        self.summary_label.setText(message)
        self.setEnabled(False)
        self._class_callback = callback
        self._class_worker = _ClassOperationThread(operation, self)
        self._class_worker.finished.connect(self._on_class_operation_finished)
        self._class_worker.start()

    def _on_class_operation_finished(self):
        worker, callback = self._class_worker, self._class_callback
        self._class_worker = None
        self._class_callback = None
        self.setEnabled(True)
        error, result = worker.error, worker.result
        worker.deleteLater()
        if error is not None:
            QMessageBox.warning(self, "클래스 변경 실패", error)
            self.summary_label.setText("클래스 변경 실패. 오류 내용 확인 필요")
            return
        callback(result)

    def _classes_updated(self, message):
        # 저장이 성공한 모델로 콤보, 모든 분할 필터, 통계를 함께 다시 구성.
        self.set_project(self.project)
        self.summary_label.setText(message)
        self.project_changed.emit(self.project)

    def _add_class(self):
        """클래스 폴더와 프로젝트 설정을 함께 저장."""
        if not self._ensure_idle():
            return
        name, accepted = QInputDialog.getText(self, "클래스 추가", "새 클래스 이름:")
        if not accepted or not name.strip():
            return
        name = name.strip()
        self.about_to_change.emit()
        self._start_class_operation(
            lambda: ClassManager.add(self.project, name),
            lambda _: self._classes_updated(f"클래스 '{name}' 추가 및 저장 완료"),
            "클래스 추가 중...",
        )

    def _delete_class(self):
        """영향 범위 검증 후 확인. 정답 번호와 프로젝트를 함께 변경."""
        if not self._ensure_idle():
            return
        name = self.class_combo.currentText()
        if not name:
            QMessageBox.warning(self, "클래스 삭제 불가", "삭제할 클래스 선택 필요")
            return
        self._start_class_operation(
            lambda: ClassManager.preview_delete(self.project, name),
            self._confirm_class_delete,
            "클래스 삭제 영향 확인 중...",
        )

    def _confirm_class_delete(self, preview):
        if preview.task == "classify":
            details = (f"Train / Val / Test의 폴더 {len(preview.folders)}개, "
                       f"이미지 {preview.image_count}장을 복구용 보관 폴더로 이동합니다.")
        else:
            details = (f"삭제할 정답 {preview.removed_annotations}개, "
                       f"번호를 조정할 정답 {preview.remapped_annotations}개\n"
                       f"변경할 라벨/마스크 파일 {preview.changed_files}개")
            if preview.task == "segment":
                details += (f"\n배경으로 바꿀 픽셀 {preview.removed_pixels:,}개, "
                            f"번호를 조정할 픽셀 {preview.remapped_pixels:,}개")
            details += "\n원본 정답은 보관하고 공유 이미지는 유지합니다."
        answer = QMessageBox.question(
            self, "클래스 삭제",
            f"'{preview.class_name}' 클래스를 삭제하시겠습니까?\n\n{details}\n\n"
            "삭제 후 클래스 번호와 프로젝트를 함께 저장합니다.\n"
            "이전 학습 기록과 모델의 클래스 정보는 유지됩니다.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self.summary_label.setText("클래스 삭제 취소")
            return
        if not self._ensure_idle():
            return
        self.about_to_change.emit()
        self._start_class_operation(
            lambda: ClassManager.delete(self.project, preview.class_name, preview),
            lambda archive: self._classes_updated(
                f"클래스 '{preview.class_name}' 삭제 및 저장 완료. 원본 보관: {archive}"
            ),
            "원본 보관 및 클래스 정답 변경 중...",
        )

    # ── 드래그 & 드롭 ─────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent):
        """드래그 진입 — 이미지 파일인지 확인"""
        if event.mimeData().hasUrls():
            # URL 중 하나라도 이미지면 수락
            for url in event.mimeData().urls():
                ext = os.path.splitext(url.toLocalFile())[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """드롭 — 이미지 파일을 프로젝트에 복사"""
        if not self.project or not self.project.data.root:
            return

        filepaths = []
        for url in event.mimeData().urls():
            filepath = url.toLocalFile()
            ext = os.path.splitext(filepath)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                filepaths.append(filepath)

        if filepaths:
            self._copy_images_to_project(filepaths)
            event.acceptProposedAction()

    # ── 데이터 스캔 ──────────────────────────────────
    def _scan_all(self):
        """스캔 요청만 보내고 결과 인덱스와 썸네일을 비동기로 반영."""
        if not self.project or not self.project.data.root:
            return
        from core.dataset_scan import DatasetScan
        previous = getattr(self, "_scan_worker", None)
        if previous is not None:
            previous.requestInterruption()
        self.summary_label.setText("데이터 탐색 중... 작업 완료 전까지 데이터 변경 대기")
        self._dataset_index = None
        worker = DatasetScan(copy.deepcopy(self.project))
        self._scan_worker = worker
        worker.ready.connect(self._scan_ready)
        worker.failed.connect(lambda message: self.summary_label.setText("데이터 탐색 실패: " + message))
        worker.finished.connect(self._scan_done)
        worker.start()

    def _scan_done(self):
        if self.sender() is self._scan_worker:
            self._scan_worker = None

    def _scan_ready(self, index, thumbnails):
        if self.sender() is not self._scan_worker:
            return
        self._dataset_index = index
        self._dataset_thumbnails = thumbnails
        stats = {}
        for split in index["splits"]:
            for name, count in split["classes"].items():
                stats.setdefault(name, {})[split["split"]] = count
        if not stats:
            stats = {"images": {s["split"]: s["count"] for s in index["splits"]}}
        self._update_stats_table(stats)
        total = len(index["images"])
        self.project.data.image_count = total
        self.summary_label.setText(f"총 {total}장 | {len(index['class_names'])}개 클래스")
        for split in SPLITS:
            self._refresh_browser(split)

    def _update_stats_table(self, stats: dict):
        """통계 테이블 갱신"""
        self.stats_table.setRowCount(len(stats))

        total_all = 0
        for row, (cls_name, counts) in enumerate(stats.items()):
            train_n = counts.get("train", 0)
            val_n = counts.get("val", 0)
            test_n = counts.get("test", 0)
            total_all += train_n + val_n + test_n

            self.stats_table.setItem(row, 0, QTableWidgetItem(cls_name))
            self.stats_table.setItem(row, 1, QTableWidgetItem(str(train_n)))
            self.stats_table.setItem(row, 2, QTableWidgetItem(str(val_n)))
            self.stats_table.setItem(row, 3, QTableWidgetItem(str(test_n)))

        self.summary_label.setText(
            f"총 {total_all}장 | {len(stats)}개 클래스/카테고리"
        )

        # 프로젝트 이미지 수 업데이트
        if self.project:
            self.project.data.image_count = total_all

    # ── 브라우저 갱신 ────────────────────────────────
    def _refresh_browser(self, split_name: str):
        """특정 분할의 이미지 브라우저 갱신"""
        if not self.project or not self.project.data.root:
            return

        idx = SPLITS.index(split_name)
        tab = self.split_tabs.widget(idx)
        if tab is None:
            return

        grid = self.tab_grids[split_name]
        task = self.project.task
        data_root = self.project.data.root

        # 모든 분할은 하나의 클래스 선택을 공유한다.
        filter_class = self._class_filter

        # 이미지 수집
        images = self._collect_images(data_root, split_name, task, filter_class)

        # 기존 위젯 제거
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._thumbs[split_name] = []

        # 썸네일 배치 (최대 200개, 행당 4개)
        display_images = images[:200]
        cols = 4
        for i, (img_path, label) in enumerate(display_images):
            row, col = divmod(i, cols)
            thumb = ImageThumbnail(img_path, label, split=split_name,
                                   thumbnail=getattr(self, "_dataset_thumbnails", {}).get(img_path))
            # 인터랙션 시그널 연결 (선택 / 드래그 / 우클릭 메뉴)
            thumb.clicked.connect(self._on_thumb_clicked)
            thumb.drag_started.connect(self._on_thumb_drag)
            thumb.context_requested.connect(self._on_thumb_context)
            thumb.opened.connect(self._open_image_preview)
            # 이전 선택 상태 복원
            if img_path in self._selected_paths:
                thumb.set_selected(True)
            grid.addWidget(thumb, row, col)
            self._thumbs[split_name].append(thumb)

        if not display_images:
            scope = split_name.upper() if filter_class == "전체" else f"{split_name.upper()} / {filter_class}"
            placeholder = QLabel(
                f"{scope}에 이미지가 없습니다.\n"
                f"위의 '이미지 추가' 버튼을 사용하세요."
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #8A92A4; padding: 40px;")
            grid.addWidget(placeholder, 0, 0)

        # 이미지 수 라벨 갱신 (200장 초과 시 표시 제한 안내)
        count_label = tab.findChild(QLabel, f"count_{split_name}")
        if count_label:
            if len(images) > len(display_images):
                count_label.setText(f"{len(images)}장 (상위 {len(display_images)}장 표시)")
            else:
                count_label.setText(f"{len(images)}장")

        self._update_selection_label()

    def _edit_selected_annotation(self):
        if len(self._selected_paths) != 1:
            QMessageBox.information(self, "데이터 티칭", "편집할 이미지를 한 장 선택하세요.")
            return
        self._open_image_preview(next(iter(self._selected_paths)))

    def _open_image_preview(self, image_path: str):
        """데이터셋 격자의 썸네일을 원본 해상도로 확인한다."""
        if not os.path.isfile(image_path):
            QMessageBox.warning(self, "이미지 보기", "이미지 파일을 찾을 수 없습니다.")
            return
        if self.project and self.project.task in ("detect", "segment", "obb"):
            if not self._ensure_idle():
                return
            from pathlib import Path
            from core.dataset_editor import edit_dataset, read_annotations, sidecars, split_root
            from widgets.obb_annotation import OBBAnnotationDialog
            from widgets.segmentation_annotation import SegmentationAnnotationDialog
            from core.mask_annotations import load_mask_document
            try:
                project = self.project
                split = next(sp for sp in SPLITS if Path(image_path).is_relative_to(split_root(project, sp)))
                records = (getattr(self, "_dataset_index", None) or {}).get("images", [])
                paths = [row["path"] for row in records if row["split"] == split and os.path.isfile(row["path"])]
                if image_path not in paths:
                    paths = [image_path]
                index = paths.index(image_path)
                def add_class(name):
                    self.about_to_change.emit()
                    ClassManager.add(project, name)
                    return project.data.class_names

                while True:
                    image_path = paths[index]
                    if project.task == "segment":
                        document = load_mask_document(project, image_path, split)
                        dialog = SegmentationAnnotationDialog(image_path, project.data.class_names, document,
                            lambda values, path=image_path: edit_dataset(project, "mask_annotations", [path], split=split, annotations=values), self,
                            add_class=add_class, navigation=(index, len(paths)))
                    else:
                        labels = sidecars(project, image_path, split)
                        rows = [row for kind, label in labels if kind == "labels"
                                for row in read_annotations(label, project.task, len(project.data.class_names))]
                        dialog = OBBAnnotationDialog(image_path, project.data.class_names, rows,
                            lambda values, path=image_path: edit_dataset(project, "annotations", [path], split=split, annotations=values), self,
                            task=project.task, add_class=add_class, navigation=(index, len(paths)))
                    dialog.exec()
                    delta = getattr(dialog, "navigation_delta", 0)
                    dialog.deleteLater()
                    if type(delta) is not int or not delta or not 0 <= index + delta < len(paths):
                        break
                    index += delta
                self.set_project(project)
                self.project_changed.emit(project)
            except (OSError, ValueError, StopIteration) as exc:
                QMessageBox.warning(self, "객체 정답 편집", str(exc) or "이미지 분할을 확인할 수 없습니다")
            return
        dialog = DatasetImagePreviewDialog(image_path, self)
        dialog.exec()

    def _collect_images(self, data_root, split_name, task, filter_class):
        index = getattr(self, "_dataset_index", None)
        if index is None:
            return []
        return [(row["path"], ", ".join(row["classes"]) or "미라벨") for row in index["images"]
                if row["split"] == split_name and (filter_class == "전체" or filter_class in row["classes"])]


    # ══════════════════════════════════════════════════
    #  상단 콤보 ↔ 우측 탭 양방향 동기화
    # ══════════════════════════════════════════════════
    #
    #  ┌──────────────────────────────────────────────────────────┐
    #  │  [대상: train ▾] [클래스: cat ▾]   ← 상단 이미지 추가 바 │
    #  │         ▲  │              ▲  │                           │
    #  │         │  ▼              │  ▼                           │
    #  │  [TRAIN][VAL][TEST]  [클래스: cat ▾] ← 우측 브라우저│
    #  └──────────────────────────────────────────────────────────┘
    #  _syncing 플래그로 무한 재귀(A→B→A)를 차단한다.

    def _current_split(self) -> str:
        """현재 선택된 탭의 분할 이름"""
        idx = self.split_tabs.currentIndex()
        return SPLITS[idx] if 0 <= idx < len(SPLITS) else SPLITS[0]

    def _set_combo_text(self, combo, text: str):
        """동기화 플래그를 세운 채 콤보 값 변경 (역방향 시그널 차단)"""
        if combo is None or not text:
            return
        idx = combo.findText(text)
        if idx < 0 or idx == combo.currentIndex():
            return
        prev = self._syncing
        self._syncing = True
        combo.setCurrentIndex(idx)
        self._syncing = prev

    def _on_top_split_changed(self, text: str):
        """상단 대상 변경은 분할만 바꾸고 공통 클래스 선택을 유지한다."""
        if self._syncing or text not in SPLITS:
            return
        self.split_tabs.setCurrentIndex(SPLITS.index(text))

    def _on_tab_changed(self, index: int):
        """분할 전환 시 상단 대상만 동기화하고 공통 필터를 유지한다."""
        if self._syncing or index < 0 or index >= len(SPLITS):
            return
        self._set_combo_text(self.split_combo, SPLITS[index])
        if self._class_filter != "전체":
            self._set_combo_text(self.class_combo, self._class_filter)

    def _on_top_class_changed(self, text: str):
        """상단 클래스 변경도 모든 분할에 적용한다."""
        if not self._syncing and text:
            self._set_shared_class_filter(text)

    def _set_shared_class_filter(self, text: str):
        if not text:
            return
        changed = text != self._class_filter
        self._class_filter = text
        for combo in self.tab_filters.values():
            self._set_combo_text(combo, text)
        if text != "전체":
            self._set_combo_text(self.class_combo, text)
        if changed:
            self._clear_selection()
            for split in SPLITS:
                self._refresh_browser(split)

    def _on_filter_changed(self, split_name: str):
        """어느 분할에서 선택하든 세 브라우저에 같은 클래스 필터 적용."""
        if not self._syncing:
            self._set_shared_class_filter(self.tab_filters[split_name].currentText())

    # ══════════════════════════════════════════════════
    #  썸네일 선택
    # ══════════════════════════════════════════════════

    def _on_thumb_clicked(self, thumb, modifiers):
        """
        썸네일 클릭 선택

        - 단독 클릭 : 기존 선택 해제 후 1장만 선택
        - Ctrl+클릭 : 개별 토글 (다중 선택)
        - Shift+클릭: 직전 클릭 지점부터 범위 선택
        """
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        thumbs = self._thumbs.get(thumb.split, [])
        idx = thumbs.index(thumb) if thumb in thumbs else -1

        if (shift and self._last_clicked
                and self._last_clicked[0] == thumb.split and idx >= 0):
            # 범위 선택
            lo, hi = sorted((self._last_clicked[1], idx))
            for t in thumbs[lo:hi + 1]:
                t.set_selected(True)
                self._selected_paths.add(t.image_path)
        elif ctrl:
            # 개별 토글
            new_state = not thumb.is_selected()
            thumb.set_selected(new_state)
            if new_state:
                self._selected_paths.add(thumb.image_path)
            else:
                self._selected_paths.discard(thumb.image_path)
            self._last_clicked = (thumb.split, idx)
        else:
            # 단독 선택
            self._clear_selection()
            thumb.set_selected(True)
            self._selected_paths.add(thumb.image_path)
            self._last_clicked = (thumb.split, idx)

        self._update_selection_label()

    def _select_all(self, split_name: str):
        """현재 탭에 표시된 이미지를 모두 선택"""
        for t in self._thumbs.get(split_name, []):
            t.set_selected(True)
            self._selected_paths.add(t.image_path)
        self._update_selection_label()

    def _clear_selection(self):
        """전체 선택 해제"""
        for thumbs in self._thumbs.values():
            for t in thumbs:
                if t.is_selected():
                    t.set_selected(False)
        self._selected_paths.clear()
        self._last_clicked = None
        self._update_selection_label()

    def _update_selection_label(self):
        """선택 개수 안내 문구 갱신"""
        n = len(self._selected_paths)
        if n:
            self.selection_label.setText(
                f"{n}장 선택됨 — 우클릭 메뉴 또는 탭 헤더로 드래그하여 이동"
            )
            self.selection_label.setStyleSheet(
                "color: #5590F0; font-size: 11px; padding: 4px;"
            )
        else:
            self.selection_label.setText(
                "썸네일 클릭으로 선택 (Ctrl=토글, Shift=범위)  ·  "
                "우클릭 메뉴 또는 탭으로 드래그하여 이동"
            )
            self.selection_label.setStyleSheet(
                "color: #555D70; font-size: 11px; padding: 4px;"
            )

    def _selected_items(self, fallback_thumb=None) -> list:
        """
        선택된 썸네일 정보 목록

        Returns:
            [(이미지경로, 클래스명, 분할명), ...]
            선택이 없으면 fallback_thumb 1장만 반환
        """
        items = []
        for thumbs in self._thumbs.values():
            for t in thumbs:
                if t.is_selected():
                    items.append((t.image_path, t.class_label, t.split))
        if not items and fallback_thumb is not None:
            items = [(fallback_thumb.image_path,
                      fallback_thumb.class_label,
                      fallback_thumb.split)]
        return items

    # ══════════════════════════════════════════════════
    #  우클릭 컨텍스트 메뉴
    # ══════════════════════════════════════════════════

    def _on_thumb_context(self, thumb, global_pos):
        """썸네일 우클릭 → 분할 이동 / 클래스 변경 / 삭제 메뉴"""
        if not self.project:
            return

        # 선택되지 않은 썸네일을 우클릭하면 그것만 단독 선택
        if not thumb.is_selected():
            self._clear_selection()
            thumb.set_selected(True)
            self._selected_paths.add(thumb.image_path)
            self._update_selection_label()

        items = self._selected_items(thumb)
        if not items:
            return

        task = self.project.task
        menu = QMenu(self)

        header = menu.addAction(f"{len(items)}장 선택됨")
        header.setEnabled(False)
        menu.addSeparator()

        if task in ("detect", "segment", "obb"):
            edit_action = menu.addAction("데이터 티칭 / 정답 그리기")
            edit_action.triggered.connect(lambda: self._open_image_preview(thumb.image_path))
            menu.addSeparator()

        # ── 분할 이동 ──
        allowed = ["train", "test"] if task == "anomaly" else SPLITS
        move_menu = menu.addMenu("분할 이동")
        for sp in allowed:
            act = move_menu.addAction(f"{sp.upper()} 으로 이동")
            act.setEnabled(sp != thumb.split or len(items) > 1)
            act.triggered.connect(
                lambda _=False, target=sp, it=items: self._move_images(it, target)
            )

        # ── 클래스 변경 ──
        class_options = []
        if task == "classify":
            class_options = list(self.project.data.class_names)
        elif task == "anomaly":
            class_options = ["good", "defect"]

        if class_options:
            cls_menu = menu.addMenu("클래스 변경")
            for cls in class_options:
                act = cls_menu.addAction(cls)
                act.triggered.connect(
                    lambda _=False, c=cls, it=items: self._change_class(it, c)
                )

        menu.addSeparator()
        del_act = menu.addAction("삭제")
        del_act.triggered.connect(
            lambda _=False, it=items: self._delete_images(it)
        )

        menu.exec(global_pos)

    # ══════════════════════════════════════════════════
    #  드래그 (썸네일 → 탭 헤더)
    # ══════════════════════════════════════════════════

    def _on_thumb_drag(self, thumb):
        """썸네일 드래그 시작 → 내부 MIME 페이로드 생성"""
        items = self._selected_items(thumb)
        if not items:
            return

        payload = json.dumps([
            {"path": p, "label": lb, "split": sp} for p, lb, sp in items
        ]).encode("utf-8")

        mime = QMimeData()
        mime.setData(IMAGE_MIME_TYPE, QByteArray(payload))

        drag = QDrag(self)
        drag.setMimeData(mime)

        # 드래그 중 미리보기 썸네일
        pm = QPixmap(thumb.image_path)
        if not pm.isNull():
            drag.setPixmap(pm.scaled(
                96, 96,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

        drag.exec(Qt.DropAction.MoveAction)

    def _on_tab_bar_drop(self, tab_index: int, mime):
        """탭 헤더에 드롭 → 해당 분할로 이동"""
        if tab_index < 0 or tab_index >= len(SPLITS):
            return
        target_split = SPLITS[tab_index]

        try:
            payload = json.loads(bytes(mime.data(IMAGE_MIME_TYPE)).decode("utf-8"))
        except Exception:
            return

        items = [
            (d.get("path", ""), d.get("label", ""), d.get("split", ""))
            for d in payload if d.get("path")
        ]
        if items:
            self._move_images(items, target_split)

    # ══════════════════════════════════════════════════
    #  파일 이동 / 클래스 변경 / 삭제
    # ══════════════════════════════════════════════════

    def _dest_dir_for(self, target_split: str, cls_label: str):
        """
        태스크별 이동 대상 디렉토리 결정

        ┌───────────────┬──────────────────────────────────┐
        │ classify      │ data/{split}/{class}/            │
        │ segment/detect│ data/images/{split}/             │
        │ anomaly       │ train→train/good, test→test/{cls}│
        └───────────────┴──────────────────────────────────┘
        """
        task = self.project.task
        root = self.project.data.root

        if task == "classify":
            cls = cls_label
            if not cls and self.project.data.class_names:
                cls = self.project.data.class_names[0]
            if not cls:
                return None
            return os.path.join(root, target_split, cls)

        if task in ("segment", "detect", "obb"):
            return os.path.join(root, "images", target_split)

        if task == "anomaly":
            if target_split == "train":
                return os.path.join(root, "train", "good")
            return os.path.join(root, "test", cls_label or "good")

        return None

    def _sidecar_moves(self, img_path: str, src_split: str, dst_split: str) -> list:
        """
        이미지와 짝을 이루어 함께 이동해야 하는 동반 파일

        Returns:
            [(원본경로, 대상디렉토리, 확장자), ...]
            - segment: masks/{split}/  (같은 파일명 stem의 마스크)
            - detect : labels/{split}/ (같은 stem의 .txt 라벨)
        """
        task = self.project.task
        root = self.project.data.root
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out = []

        if task == "segment":
            src_dir = os.path.join(root, "masks", src_split)
            dst_dir = os.path.join(root, "masks", dst_split)
            if os.path.isdir(src_dir):
                for f in os.listdir(src_dir):
                    if (os.path.splitext(f)[0] == stem
                            and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                            and os.path.isfile(os.path.join(src_dir, f))):
                        out.append((
                            os.path.join(src_dir, f),
                            dst_dir,
                            os.path.splitext(f)[1],
                        ))
        if task in ("detect", "segment", "obb"):
            src = os.path.join(root, "labels", src_split, stem + ".txt")
            if os.path.isfile(src):
                out.append((src, os.path.join(root, "labels", dst_split), ".txt"))

        return out

    @staticmethod
    def _unique_dst(dst: str) -> str:
        """대상 경로가 이미 있으면 _1, _2 ... 접미사를 붙여 충돌 회피"""
        if not os.path.exists(dst):
            return dst
        base, ext = os.path.splitext(dst)
        n = 1
        while os.path.exists(f"{base}_{n}{ext}"):
            n += 1
        return f"{base}_{n}{ext}"

    def _move_images(self, items: list, target_split: str):
        """
        선택 이미지를 다른 분할(train/val/test)로 이동

        마스크(segment) / 라벨(detect) 동반 파일도 stem을 맞춰 함께 이동한다.
        """
        if not self._ensure_idle():
            return
        if not self.project or not items:
            return

        task = self.project.task

        # Anomaly 학습 데이터는 정상(good)만 허용
        if task == "anomaly" and target_split == "train":
            defects = [i for i in items if i[1] == "defect"]
            if defects:
                QMessageBox.warning(
                    self, "이동 불가",
                    "Anomaly 학습 데이터(train)는 정상(good) 이미지만 "
                    "포함할 수 있습니다.\n"
                    f"불량(defect) {len(defects)}장은 제외하고 진행합니다."
                )
                items = [i for i in items if i[1] != "defect"]
                if not items:
                    return

        moved, skipped = 0, 0
        errors = []
        for path, label, src_split in items:
            if src_split == target_split or not os.path.isfile(path):
                skipped += 1
                continue

            dst_dir = self._dest_dir_for(target_split, label)
            if dst_dir is None:
                skipped += 1
                continue

            try:
                reserve_dirs = []
                if task in ("detect", "segment", "obb"):
                    reserve_dirs.append(os.path.join(self.project.data.root, "labels", target_split))
                if task == "segment":
                    reserve_dirs.append(os.path.join(self.project.data.root, "masks", target_split))
                move_image_with_sidecars(
                    path, dst_dir, self._sidecar_moves(path, src_split, target_split),
                    reserve_stem=task in ("detect", "segment", "obb"), reserve_dirs=reserve_dirs,
                )
                moved += 1
            except Exception as exc:
                skipped += 1
                errors.append(f"{os.path.basename(path)}: {exc}")

        self._clear_selection()
        self._scan_all()
        self.summary_label.setText(
            f"{moved}장을 {target_split.upper()}(으)로 이동 / 건너뜀: {skipped}장"
        )
        if errors:
            QMessageBox.warning(self, "일부 이미지 이동 실패", "\n".join(errors[:3]))

        # 이동한 분할 탭으로 자동 전환
        if moved and target_split in SPLITS:
            self.split_tabs.setCurrentIndex(SPLITS.index(target_split))

    def _change_class(self, items: list, new_class: str):
        """선택 이미지를 같은 분할 내에서 다른 클래스로 이동 (라벨 정정)"""
        if not self._ensure_idle():
            return
        if not self.project or not items:
            return

        task = self.project.task
        if task not in ("classify", "anomaly"):
            return

        root = self.project.data.root
        moved, skipped = 0, 0

        for path, label, src_split in items:
            if label == new_class or not os.path.isfile(path):
                skipped += 1
                continue

            if task == "classify":
                dst_dir = os.path.join(root, src_split, new_class)
            else:
                # Anomaly: 클래스 구분은 test 분할에서만 의미가 있음
                if src_split != "test":
                    skipped += 1
                    continue
                dst_dir = os.path.join(root, "test", new_class)

            try:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(path, self._unique_dst(
                    os.path.join(dst_dir, os.path.basename(path))
                ))
                moved += 1
            except Exception:
                skipped += 1

        self._clear_selection()
        self._scan_all()
        self.summary_label.setText(
            f"{moved}장을 '{new_class}' 클래스로 변경  ·  건너뜀: {skipped}장"
        )

    def _delete_images(self, items: list):
        """선택 이미지를 영구 삭제 (동반 파일 포함)"""
        if not self._ensure_idle():
            return
        if not items:
            return

        reply = QMessageBox.question(
            self, "삭제 확인",
            f"선택한 {len(items)}장을 삭제하시겠습니까?\n"
            f"(파일이 디스크에서 영구 삭제됩니다)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        deleted, failed = 0, 0
        for path, label, src_split in items:
            try:
                # 마스크/라벨 등 동반 파일도 함께 삭제
                for s_src, _, _ in self._sidecar_moves(path, src_split, src_split):
                    if os.path.isfile(s_src):
                        os.remove(s_src)
                os.remove(path)
                deleted += 1
            except Exception:
                failed += 1

        self._clear_selection()
        self._scan_all()
        self.summary_label.setText(f"{deleted}장 삭제  ·  실패: {failed}장")

    def _open_data_folder(self):
        """데이터 폴더를 파일 탐색기로 열기"""
        if self.project and self.project.data.root:
            path = self.project.data.root
            if os.path.isdir(path):
                # Windows: explorer, Mac: open, Linux: xdg-open
                import subprocess
                import platform
                if platform.system() == "Windows":
                    os.startfile(path)
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", path])
                else:
                    subprocess.Popen(["xdg-open", path])
