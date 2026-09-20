"""
Deep Vision Studio — 메인 윈도우

레이아웃 (텍스트 온리 내비게이션 / Premium Dashboard):
┌────────────────────────────────────────────────────────────────┐
│  Deep Vision Studio                          [─] [□] [×]     │
├────────────┬───────────────────────────────────────────────────┤
│            │                                                   │
│  Project   │                                                   │
│  Dataset   │                  Content Area                     │
│  Training  │               (활성 페이지 표시)                    │
│  Inference │                                                   │
│  Export    │                                                   │
│  Defect Gen│                                                   │
├────────────┴───────────────────────────────────────────────────┤
│  Ready                        Device: CPU │ RAM: xxGB          │
└────────────────────────────────────────────────────────────────┘

내비게이션 항목은 아이콘 없이 라벨 텍스트만 사용한다.
선택 상태는 좌측 액센트 바 + 배경 대비로만 표현한다.
"""

import os
import sys
import platform

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QStatusBar,
    QFrame, QMessageBox, QFileDialog, QSizePolicy,
    QButtonGroup, QSpacerItem
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QIcon, QFont, QAction

# 위젯 임포트
from widgets.project_widget import ProjectWidget
from widgets.dataset_widget import DatasetWidget
from widgets.training_widget import TrainingWidget
from widgets.inference_widget import InferenceWidget
from widgets.export_widget import ExportWidget
from widgets.defect_gen_widget import DefectGenWidget
from widgets.model_manager_widget import ModelManagerWidget

from core.project import ProjectData, ProjectManager, SUPPORTED_TASKS
from core.device_manager import get_device_manager
from core.version import APP_VERSION


class MainWindow(QMainWindow):
    """
    Deep Vision Studio 메인 윈도우

    구조:
    - 왼쪽: 네비게이션 사이드바
    - 중앙: 콘텐츠 영역 (QStackedWidget)
    - 하단: 상태바
    """

    def __init__(self):
        super().__init__()
        self.project: ProjectData = None  # 현재 프로젝트
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """UI 초기화"""
        # ── 윈도우 기본 설정 ──
        self.setWindowTitle("Deep Vision Studio")
        self.setMinimumSize(960, 600)
        self.resize(1440, 900)

        # ── 메뉴바 ──
        self._create_menubar()

        # ── 메인 레이아웃 ──
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 사이드바 ──
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)

        # ── 콘텐츠 영역 (스택 위젯) ──
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("content_area")
        main_layout.addWidget(self.content_stack, stretch=1)

        # ── 페이지 위젯 생성 ──
        self.project_page = ProjectWidget()
        self.dataset_page = DatasetWidget()
        self.training_page = TrainingWidget()
        self.inference_page = InferenceWidget()
        self.export_page = ExportWidget()
        self.defect_gen_page = DefectGenWidget()
        self.model_manager_page = ModelManagerWidget()

        self.content_stack.addWidget(self.project_page)    # index 0
        self.content_stack.addWidget(self.dataset_page)     # index 1
        self.content_stack.addWidget(self.training_page)    # index 2
        self.content_stack.addWidget(self.inference_page)    # index 3
        self.content_stack.addWidget(self.export_page)      # index 4
        self.content_stack.addWidget(self.defect_gen_page)   # index 5
        self.content_stack.addWidget(self.model_manager_page) # index 6

        # ── 상태바 ──
        self._create_statusbar()

        # ── 초기 상태: 프로젝트 페이지 ──
        self._navigate_to(0)

    def _create_sidebar(self) -> QFrame:
        """네비게이션 사이드바 생성"""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 앱 타이틀
        title = QLabel("Deep Vision Studio")
        title.setObjectName("app_title")
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 20px;")
        layout.addWidget(title)

        subtitle = QLabel(f"v{APP_VERSION}")
        subtitle.setObjectName("app_subtitle")
        layout.addWidget(subtitle)

        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #1e222c; max-height: 1px;")
        layout.addWidget(line)
        layout.addSpacing(8)

        # 네비게이션 버튼들
        self.nav_buttons = []
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        nav_items = [
            ("Project", "프로젝트 생성 및 관리"),
            ("Dataset", "학습 데이터 관리"),
            ("Training", "모델 학습 및 모니터링"),
            ("Inference", "모델 테스트 및 검증"),
            ("Export", "ONNX 모델 변환"),
            ("Defect Gen", "합성 불량 이미지 생성"),
            ("Settings", "기본 모델과 추가 Docker 모델 관리"),
        ]

        for idx, (text, tooltip) in enumerate(nav_items):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, i=idx: self._navigate_to(i))
            self.nav_group.addButton(btn, idx)
            self.nav_buttons.append(btn)
            layout.addWidget(btn)

        # 스페이서
        layout.addStretch()

        # 하단 정보 (디바이스 상태 표시)
        dm = get_device_manager()
        if dm.cuda_available and dm.gpus:
            gpu = dm.gpus[0]
            device_text = f"GPU\n{gpu.name}"
        else:
            device_text = "CPU Mode"
        info_label = QLabel(f"v{APP_VERSION}\n{device_text}")
        info_label.setObjectName("app_subtitle")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        layout.addSpacing(10)

        return sidebar

    def _create_menubar(self):
        """메뉴바 생성"""
        menubar = self.menuBar()

        # 파일 메뉴
        file_menu = menubar.addMenu("파일(&F)")
        file_menu.addAction("새 프로젝트(&N)", self._new_project, "Ctrl+N")
        file_menu.addAction("열기(&O)", self._open_project, "Ctrl+O")
        file_menu.addAction("저장(&S)", self._save_project, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("종료(&Q)", self.close, "Ctrl+Q")

        # 도움말 메뉴
        help_menu = menubar.addMenu("도움말(&H)")
        help_menu.addAction("정보(&A)", self._show_about)

    def _create_statusbar(self):
        """
        상태바 생성

        표시 정보:
        ┌────────────────────────────────────────────────────────────────┐
        │  준비    GPU: NVIDIA RTX 4090 (24.0 GB) | CUDA 12.x │ ...  │
        │                 또는                                          │
        │  준비    CPU 전용 (GPU 미감지) │ RAM: xxGB │ OS: ...        │
        └────────────────────────────────────────────────────────────────┘
        """
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # 상태 메시지
        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label, stretch=1)

        # 시스템 정보 (디바이스 + RAM + OS)
        import psutil
        cpu_count = os.cpu_count() or 0
        try:
            ram_gb = psutil.virtual_memory().total / (1024**3)
            ram_text = f"{ram_gb:.1f}GB"
        except Exception:
            ram_text = "N/A"

        dm = get_device_manager()
        device_text = dm.get_status_text()

        sys_info = QLabel(
            f"{device_text} │ CPU: {cpu_count}C │ "
            f"RAM: {ram_text} │ "
            f"OS: {platform.system()} {platform.release()}"
        )
        self.statusbar.addPermanentWidget(sys_info)

    # ── 네비게이션 ─────────────────────────────────
    def _navigate_to(self, index: int):
        """페이지 전환"""
        # 프로젝트 없으면 프로젝트 페이지만 허용
        if index not in {0, 6} and self.project is None:
            QMessageBox.information(
                self, "프로젝트 필요",
                "먼저 프로젝트를 생성하거나 열어주세요."
            )
            index = 0

        self.content_stack.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)

    # ── 프로젝트 관리 ──────────────────────────────
    def set_project(self, project: ProjectData):
        """프로젝트 설정 → 모든 페이지에 전파"""
        if not self._allow_project_change():
            return False
        if self.project is not None and self.project is not project:
            try:
                opened_path = ProjectManager.get_active_filepath(project)
                current_path = ProjectManager.get_active_filepath(self.project)
                self._save_current_project()
                if os.path.exists(opened_path) and os.path.samefile(opened_path, current_path):
                    project = ProjectManager.load(opened_path)
            except Exception as exc:
                QMessageBox.critical(self, "저장 오류", f"현재 프로젝트 저장 실패:\n{exc}")
                return False
        self.project = project
        self.setWindowTitle(f"Deep Vision Studio — {project.name}")

        # 각 페이지에 프로젝트 전달
        self.dataset_page.set_project(project)
        self.training_page.set_project(project)
        self.inference_page.set_project(project)
        self.export_page.set_project(project)
        self.defect_gen_page.set_project(project)
        self.model_manager_page.set_project(project)

        self.status_label.setText(
            f"Project: {project.name}  ·  "
            f"Task: {SUPPORTED_TASKS[project.task]['name']}"
        )
        return True

    def has_active_jobs(self):
        """모든 페이지의 실행 중 워커와 종료 결과 반영 대기 상태 확인."""
        if getattr(getattr(self, "inference_page", None), "_inference_worker", None) is not None:
            return True
        if getattr(self.dataset_page, "_class_worker", None) is not None:
            return True
        for page in (self.dataset_page, self.training_page, self.inference_page,
                     self.export_page, self.defect_gen_page):
            if getattr(page, "_run_project", None) is not None:
                return True
            for name, value in vars(page).items():
                if name == "_scan_worker":  # 읽기 전용 스캔은 새 프로젝트에서 취소/교체 가능
                    continue
                running = getattr(value, "isRunning", None)
                if callable(running):
                    try:
                        if running():
                            return True
                    except RuntimeError:  # 이미 해제된 Qt 객체
                        continue
        return False

    def _allow_project_change(self):
        if self.has_active_jobs():
            QMessageBox.warning(self, "작업 실행 중", "실행 중인 작업 완료 또는 중단 후 프로젝트 전환 가능")
            return False
        return True

    def _save_current_project(self):
        class_worker = getattr(self.dataset_page, "_class_worker", None)
        if class_worker is not None:
            raise RuntimeError("클래스 변경 완료 후 저장 가능")
        self.training_page.collect_config()
        return ProjectManager.save(self.project)

    def _new_project(self):
        """새 프로젝트 — ProjectWidget에서 처리"""
        if not self._allow_project_change():
            return
        self._navigate_to(0)
        self.project_page.show_create_dialog()

    def _open_project(self):
        """기존 프로젝트 열기"""
        if not self._allow_project_change():
            return
        filepath, _ = QFileDialog.getOpenFileName(
            self, "프로젝트 열기", "",
            f"Deep Vision Studio 프로젝트 (*{ProjectManager.FILE_EXTENSION});;JSON (*.json);;모든 파일 (*)"
        )
        if filepath:
            try:
                project = ProjectManager.load(filepath)
                if self.set_project(project):
                    self._navigate_to(1)  # 데이터셋 페이지로
            except Exception as e:
                QMessageBox.critical(
                    self, "오류",
                    f"프로젝트 로드 실패:\n{str(e)}"
                )

    def _save_project(self):
        """현재 프로젝트 저장"""
        if self.project is None:
            QMessageBox.information(self, "알림", "열린 프로젝트가 없습니다.")
            return

        try:
            filepath = self._save_current_project()
            self.status_label.setText(f"Saved: {filepath}")
        except Exception as e:
            QMessageBox.critical(
                self, "오류", f"저장 실패:\n{str(e)}"
            )

    def _show_about(self):
        """정보 대화상자"""
        QMessageBox.about(
            self, "Deep Vision Studio",
            "<h2>Deep Vision Studio</h2>"
            f"<p>v{APP_VERSION}</p>"
            "<p>Industrial Vision Training Tool</p>"
            "<hr>"
            "<p><b>Supported Tasks:</b></p>"
            "<ul>"
            "<li>Classification — Image Classification</li>"
            "<li>Segmentation — Semantic Segmentation</li>"
            "<li>Object Detection — Object Detection</li>"
            "<li>Anomaly Detection — Anomaly Detection</li>"
            "</ul>"
            "<p><b>Tools:</b> Synthetic Defect Generator (7 types)</p>"
            "<p><b>Pipeline:</b> Python Training → ONNX → C++ Inference</p>"
        )

    def _connect_signals(self):
        """시그널-슬롯 연결"""
        # ProjectWidget에서 프로젝트 생성 완료 시
        self.project_page.project_created.connect(self._on_project_created)
        if hasattr(self.dataset_page, "about_to_change"):
            self.dataset_page.about_to_change.connect(self.training_page.collect_config)
        if hasattr(self.dataset_page, "project_changed"):
            self.dataset_page.project_changed.connect(self._on_dataset_project_changed)

        # DefectGenWidget에서 불량 이미지 생성 완료 시 → Dataset 자동 새로고침
        self.defect_gen_page.generation_finished.connect(
            self._on_defect_gen_finished
        )
        self.model_manager_page.models_changed.connect(self._on_model_catalog_changed)

    def _on_model_catalog_changed(self):
        """새 Docker 모델을 설치하면 현재 프로젝트의 선택 목록만 새로 고친다."""
        if self.project is not None:
            self.training_page.refresh_model_catalog()

    def _on_project_created(self, project: ProjectData):
        """프로젝트 생성 완료 핸들러"""
        if self.set_project(project):
            self._navigate_to(1)  # 데이터셋 페이지로 이동

    def _on_dataset_project_changed(self, project):
        """저장된 클래스 정의 변경을 종속 페이지에 반영."""
        if project is not self.project:
            return
        self.training_page.set_project(project)
        self.inference_page._clear_model()
        self.inference_page.set_project(project)
        self.export_page.set_project(project)
        self.defect_gen_page.set_project(project)

    def _on_defect_gen_finished(self, count: int):
        """불량 생성 완료 → Dataset 페이지 자동 새로고침"""
        # 생성된 이미지가 있을 때만 스캔 (불필요한 리로드 방지)
        if count > 0 and hasattr(self.dataset_page, "_scan_all"):
            self.dataset_page._scan_all()

    def closeEvent(self, event):
        """종료 시 프로젝트 자동 저장"""
        if self.has_active_jobs():
            QMessageBox.warning(self, "작업 실행 중", "실행 중인 작업 완료 또는 중단 후 종료 가능")
            event.ignore()
            return
        if self.project is not None:
            reply = QMessageBox.question(
                self, "종료 확인",
                "프로젝트를 저장하고 종료하시겠습니까?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                try:
                    self._save_current_project()
                except Exception as exc:
                    QMessageBox.critical(self, "저장 오류", f"저장 실패:\n{exc}")
                    event.ignore()
                    return
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
