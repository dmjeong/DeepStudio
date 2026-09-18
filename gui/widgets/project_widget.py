"""
Deep Vision Studio — 프로젝트 관리 페이지

기능:
┌────────────────────────────────────────────────────────┐
│ 1. 새 프로젝트 생성 (태스크 선택 + 이름 + 경로)        │
│ 2. 최근 프로젝트 목록                                   │
│ 3. 프로젝트 열기/삭제                                   │
│ 4. 태스크별 설명 카드 (ViDi 스타일)                     │
└────────────────────────────────────────────────────────┘
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFrame,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout, QComboBox, QGridLayout,
    QScrollArea, QSizePolicy, QLayout,
)
from PySide6.QtCore import Qt, Signal, QEvent, QTimer
from PySide6.QtGui import QFont

from core.project import (
    ProjectManager, SUPPORTED_TASKS,
)


class TaskCard(QFrame):
    """Keyboard-accessible task choice with a clear selected state."""

    clicked = Signal(str)
    DETAILS = {
        "classify": ("CLS", "이미지 분류", "이미지 전체를 보고 클래스를 판정합니다."),
        "detect": ("DET", "객체 탐지", "객체의 위치와 크기를 사각 박스로 찾습니다."),
        "obb": ("OBB", "회전 객체 탐지", "기울어진 객체를 회전 박스로 찾습니다."),
        "segment": ("SEG", "세그멘테이션", "대상의 윤곽을 픽셀 단위로 구분합니다."),
        "anomaly": ("AD", "이상 탐지", "정상 이미지와 다른 영역을 찾습니다."),
    }
    COLORS = {
        "classify": "#5590F0", "segment": "#34C759", "detect": "#E5A832",
        "obb": "#19B5AC", "anomaly": "#9B7DFF",
    }

    def __init__(self, task_key: str, parent=None):
        super().__init__(parent)
        self.task_key = task_key
        self._selected = False
        code, title, description = self.DETAILS[task_key]
        self.setObjectName("taskChoice")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(title)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)
        top = QHBoxLayout()
        badge = QLabel(code)
        badge.setStyleSheet(f"color:{self.COLORS[task_key]}; background:#242a34; border:0; border-radius:5px; padding:3px 7px; font-size:11px; font-weight:600;")
        top.addWidget(badge)
        top.addStretch()
        self.indicator = QLabel("○")
        self.indicator.setStyleSheet("background:transparent; border:0; color:#8A92A4; font-size:17px;")
        top.addWidget(self.indicator)
        layout.addLayout(top)
        label = QLabel(title)
        label.setWordWrap(True)
        label.setStyleSheet("background:transparent; border:0; color:#E8EAEF; font-size:15px; font-weight:600;")
        layout.addWidget(label)
        detail = QLabel(description)
        detail.setWordWrap(True)
        detail.setStyleSheet("background:transparent; border:0; color:#8A92A4; font-size:12px;")
        layout.addWidget(detail)
        layout.addStretch()
        for child in self.findChildren(QLabel):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            child.setMinimumWidth(0)
        self.set_selected(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.clicked.emit(self.task_key)
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.clicked.emit(self.task_key)
        else:
            super().keyPressEvent(event)

    def set_selected(self, selected: bool):
        self._selected = selected
        self.indicator.setText("●" if selected else "○")
        border = self.COLORS[self.task_key] if selected else "#2a2f3a"
        background = "#1a2540" if selected else "#151920"
        self.setStyleSheet(f"""
            QFrame#taskChoice {{ background:{background}; border:2px solid {border}; border-radius:12px; }}
            QFrame#taskChoice:hover {{ border-color:{self.COLORS[self.task_key]}; }}
            QFrame#taskChoice:focus {{ border-color:{self.COLORS[self.task_key]}; }}
        """)


class _RecentProjectButton(QPushButton):
    """Long project names must not enlarge the scroll area's minimum width."""

    def __init__(self, text):
        super().__init__(text)
        self._full_text = text
        self.setToolTip(text)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margins = self.contentsMargins()
        width = max(0, self.width() - margins.left() - margins.right() - 12)
        self.setText(self.fontMetrics().elidedText(self._full_text, Qt.TextElideMode.ElideRight, width))


class ProjectWidget(QWidget):
    """프로젝트 관리 페이지"""

    # 프로젝트 생성 완료 시그널
    project_created = Signal(object)  # ProjectData

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_task = None
        self._task_columns = 0
        self._task_layout_pending = False
        self._init_ui()

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea(self)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll)
        self.content = QWidget()
        self.scroll.setWidget(self.content)
        layout = QVBoxLayout(self.content)
        layout.setContentsMargins(24, 16, 24, 20)
        layout.setSpacing(16)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self.content.setStyleSheet("""
            QGroupBox#projectTasks, QGroupBox#projectSettings, QGroupBox#projectRecent {
                margin-top:18px; padding:14px;
            }
            QGroupBox#projectTasks::title, QGroupBox#projectSettings::title,
            QGroupBox#projectRecent::title { left:16px; padding:0 4px; }
        """)

        # ── 타이틀 ──
        title = QLabel("새 프로젝트 생성")
        title.setObjectName("page_title")
        layout.addWidget(title)

        subtitle = QLabel(
            "수행할 비전 태스크를 선택한 뒤 프로젝트를 생성하세요. "
            "태스크별로 최적화된 학습 파이프라인과 모델 헤드가 자동 적용됩니다."
        )
        subtitle.setObjectName("page_subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # ── 태스크 선택 카드 ──
        task_group = QGroupBox("1. 태스크 선택")
        task_group.setObjectName("projectTasks")
        self.task_group = task_group
        task_layout = QGridLayout(task_group)
        self.task_layout = task_layout
        task_layout.setContentsMargins(8, 8, 8, 8)
        task_layout.setSpacing(12)

        self.task_cards = {}
        for index, task_key in enumerate(SUPPORTED_TASKS):
            card = TaskCard(task_key)
            card.clicked.connect(self._on_task_selected)
            self.task_cards[task_key] = card
            task_layout.addWidget(card, index // 3, index % 3)

        for column in range(3):
            task_layout.setColumnStretch(column, 1)
        layout.addWidget(task_group)
        self.scroll.viewport().installEventFilter(self)
        task_group.installEventFilter(self)

        # ── 프로젝트 설정 ──
        config_group = QGroupBox("2. 프로젝트 설정")
        config_group.setObjectName("projectSettings")
        config_layout = QFormLayout(config_group)
        config_layout.setContentsMargins(8, 8, 8, 8)
        config_layout.setHorizontalSpacing(16)
        config_layout.setVerticalSpacing(10)
        config_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        config_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # 프로젝트 이름
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("예: Surface_Defect_Classify")
        config_layout.addRow("프로젝트 이름:", self.name_edit)

        # 저장 경로
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("프로젝트 저장 폴더")
        path_layout.addWidget(self.path_edit)
        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(browse_btn)
        config_layout.addRow("저장 경로:", path_layout)

        # 클래스 이름 입력 (쉼표 구분)
        self.class_edit = QLineEdit()
        self.class_edit.setPlaceholderText("예: OK, NG_scratch, NG_dent (쉼표로 구분)")
        config_layout.addRow("클래스 이름:", self.class_edit)

        # 클래스 안내 라벨
        self.class_hint = QLabel(
            "Classification: 분류할 클래스 이름 입력\n"
            "Anomaly: 입력 불필요 (자동으로 good/defect 생성)"
        )
        self.class_hint.setStyleSheet("color: #555D70; font-size: 11px;")
        self.class_hint.setWordWrap(True)
        config_layout.addRow("", self.class_hint)

        # 입력 채널
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["3 (RGB)", "1 (Grayscale)"])
        self.channel_combo.setCurrentIndex(1)
        config_layout.addRow("입력 채널:", self.channel_combo)

        layout.addWidget(config_group)

        # ── 생성 버튼 ──
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.create_btn = QPushButton("프로젝트 생성")
        self.create_btn.setProperty("cssClass", "primary")
        self.create_btn.setMinimumSize(180, 40)
        self.create_btn.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.create_btn.clicked.connect(self._create_project)
        btn_layout.addWidget(self.create_btn)

        layout.addLayout(btn_layout)

        # ── 최근 프로젝트 ──
        recent_group = QGroupBox("최근 프로젝트")
        recent_group.setObjectName("projectRecent")
        self.recent_layout = QVBoxLayout(recent_group)
        self.recent_layout.setContentsMargins(8, 8, 8, 8)
        self._refresh_recent()
        layout.addWidget(recent_group)

        layout.addStretch()
        self._schedule_task_layout()

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._schedule_task_layout()
        return super().eventFilter(watched, event)

    def _schedule_task_layout(self):
        if not self._task_layout_pending:
            self._task_layout_pending = True
            QTimer.singleShot(0, self._reflow_tasks)

    def _reflow_tasks(self):
        self._task_layout_pending = False
        margins = self.task_layout.contentsMargins()
        page_margins = self.content.layout().contentsMargins()
        group_margins = self.task_group.contentsMargins()
        width = (self.scroll.viewport().width() - page_margins.left() - page_margins.right()
                 - group_margins.left() - group_margins.right() - margins.left() - margins.right())
        if width <= 0:
            return
        gap = self.task_layout.horizontalSpacing()
        minimum = max(240, self.fontMetrics().horizontalAdvance("M") * 24)
        columns = max(1, min(3, (width + gap) // (minimum + gap)))
        if columns != self._task_columns:
            while self.task_layout.count():
                self.task_layout.takeAt(0)
            for column in range(3):
                self.task_layout.setColumnStretch(column, int(column < columns))
            for index, card in enumerate(self.task_cards.values()):
                self.task_layout.addWidget(card, index // columns, index % columns)
            self._task_columns = columns
        card_width = max(1, (width - gap * (columns - 1)) // columns)
        # Wrapped titles and descriptions must fit before the scroll area sizes the page.
        height = max(128, *(max(card.layout().totalHeightForWidth(card_width),
                               card.layout().totalMinimumSize().height())
                            for card in self.task_cards.values()))
        for card in self.task_cards.values():
            card.setMinimumHeight(height)

    def _on_task_selected(self, task_key: str):
        """태스크 카드 선택 → 클래스 입력 안내 업데이트"""
        self._selected_task = task_key
        for key, card in self.task_cards.items():
            card.set_selected(key == task_key)

        # 태스크별 클래스 입력 안내
        hints = {
            "classify": (
                "분류할 클래스 이름을 쉼표(,)로 구분하여 입력하세요.\n"
                "예: OK, NG_scratch, NG_dent"
            ),
            "segment": (
                "세그멘테이션 클래스 이름을 쉼표(,)로 구분하여 입력하세요.\n"
                "예: background, defect, crack"
            ),
            "detect": (
                "탐지할 객체 클래스 이름을 쉼표(,)로 구분하여 입력하세요.\n"
                "예: person, car, bicycle"
            ),
            "obb": (
                "회전 박스로 탐지할 객체 클래스 이름을 쉼표(,)로 구분하여 입력하세요.\n"
                "예: part, scratch, mark"
            ),
            "anomaly": (
                "Anomaly는 클래스 입력 불필요\n"
                "정상(good)/결함(defect) 폴더가 자동으로 생성됩니다."
            ),
        }
        self.class_hint.setText(hints.get(task_key, ""))

        # Anomaly는 클래스 입력 비활성화
        self.class_edit.setEnabled(task_key != "anomaly")
        if task_key == "anomaly":
            self.class_edit.setText("")
            self.class_edit.setPlaceholderText("(자동 생성됨)")

    def _browse_path(self):
        """저장 경로 선택"""
        path = QFileDialog.getExistingDirectory(self, "프로젝트 경로 선택")
        if path:
            self.path_edit.setText(path)

    def _create_project(self):
        """프로젝트 생성"""
        window = self.window()
        if hasattr(window, "_allow_project_change") and not window._allow_project_change():
            return
        # 입력 검증
        if not self._selected_task:
            QMessageBox.warning(self, "알림", "먼저 태스크를 선택해 주세요.")
            return

        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "알림", "프로젝트 이름을 입력해 주세요.")
            return

        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "알림", "프로젝트를 저장할 경로를 선택해 주세요.")
            return

        try:
            ProjectManager.validate_name(name)
        except ValueError as exc:
            QMessageBox.warning(self, "이름 오류", str(exc))
            return
        project_dir = os.path.join(path, name)

        # 클래스 이름 파싱
        class_text = self.class_edit.text().strip()
        if class_text and self._selected_task != "anomaly":
            class_names = [c.strip() for c in class_text.split(",") if c.strip()]
        else:
            class_names = []

        # Classification/Segment/Detect는 클래스 필수
        if self._selected_task in ("classify", "segment", "detect", "obb") and not class_names:
            QMessageBox.warning(
                self, "알림",
                "클래스 이름을 입력해 주세요.\n"
                "예: OK, NG_scratch, NG_dent"
            )
            return

        try:
            # 프로젝트 생성 (폴더 자동 구성 포함)
            project = ProjectManager.create_new(
                name=name,
                task=self._selected_task,
                project_dir=project_dir,
                class_names=class_names,
            )

            # 입력 채널 설정
            in_ch = 3 if self.channel_combo.currentIndex() == 0 else 1
            project.training.in_channels = in_ch

            # 저장
            ProjectManager.save(project)

            # 시그널 emit → MainWindow에서 처리
            self.project_created.emit(project)

            # 생성된 폴더 구조 안내
            task_name = SUPPORTED_TASKS[self._selected_task]['name']
            data_path = os.path.join(project_dir, "data")
            QMessageBox.information(
                self, "성공",
                f"프로젝트가 생성되었습니다.\n\n"
                f"위치: {project_dir}\n"
                f"태스크: {task_name}\n"
                f"클래스: {', '.join(class_names) if class_names else '(자동)'}\n\n"
                f"데이터 폴더 구조가 자동으로 구성되었습니다:\n"
                f"{data_path}\n\n"
                f"Dataset 탭에서 이미지를 추가하세요."
            )
        except Exception as e:
            QMessageBox.critical(
                self, "오류", f"프로젝트 생성 실패:\n{str(e)}"
            )

    def show_create_dialog(self):
        """새 프로젝트 생성 다이얼로그 표시"""
        self.name_edit.setFocus()

    def _refresh_recent(self):
        """최근 프로젝트 목록 갱신"""
        # 기존 위젯 제거
        while self.recent_layout.count():
            item = self.recent_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 최근 프로젝트 검색
        recent = ProjectManager.list_recent(limit=5)
        if not recent:
            no_proj = QLabel("최근 프로젝트가 없습니다. 새로 만들어 보세요.")
            no_proj.setStyleSheet("color: #555D70; padding: 20px;")
            self.recent_layout.addWidget(no_proj)
        else:
            for filepath in recent:
                try:
                    proj = ProjectManager.load(filepath)
                    task_info = SUPPORTED_TASKS.get(proj.task, {})
                    name = task_info.get("name", proj.task)

                    btn = _RecentProjectButton(f"{proj.name}  —  {name}")
                    btn.setStyleSheet("text-align: left; padding: 10px 16px;")
                    btn.clicked.connect(
                        lambda checked, fp=filepath: self._open_recent(fp)
                    )
                    self.recent_layout.addWidget(btn)
                except Exception:
                    pass

    def _open_recent(self, filepath: str):
        """최근 프로젝트 열기"""
        window = self.window()
        if hasattr(window, "_allow_project_change") and not window._allow_project_change():
            return
        try:
            project = ProjectManager.load(filepath)
            self.project_created.emit(project)
        except Exception as e:
            QMessageBox.critical(
                self, "오류", f"프로젝트 열기 실패:\n{str(e)}"
            )
