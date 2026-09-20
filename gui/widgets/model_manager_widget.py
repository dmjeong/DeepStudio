"""기본 모델과 사용자가 추가한 Docker 모델 팩을 한 화면에서 관리한다."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTreeWidget, QTreeWidgetItem, QFileDialog, QMessageBox, QGroupBox,
)


_TASK_LABELS = {
    "classify": "Classification",
    "anomaly": "Anomaly Detection",
    "detect": "Object Detection",
    "segment": "Segmentation",
}


class ModelManagerWidget(QWidget):
    """모델 추가는 Settings에서만 시작하고, 학습 화면은 선택과 실행에 집중한다."""

    models_changed = Signal()
    project_updated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._init_ui()
        self.refresh_catalog()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("Settings · 모델 관리")
        title.setObjectName("page_title")
        layout.addWidget(title)

        description = QLabel(
            "카탈로그의 모든 모델은 설치본 기본 모델입니다. "
            "추가 모델만 검증된 .dvmodel Docker 팩으로 설치하며, 기본 모델을 대체하지 않습니다."
        )
        description.setWordWrap(True)
        description.setObjectName("text_tertiary")
        layout.addWidget(description)

        catalog_group = QGroupBox("모델 카탈로그")
        catalog_layout = QVBoxLayout(catalog_group)
        self.catalog = QTreeWidget()
        self.catalog.setColumnCount(4)
        self.catalog.setHeaderLabels(["구분", "태스크", "모델", "실행 방식"])
        self.catalog.setRootIsDecorated(False)
        self.catalog.setAlternatingRowColors(False)
        self.catalog.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.catalog.setWordWrap(True)
        self.catalog.setAccessibleName("모델 카탈로그")
        catalog_layout.addWidget(self.catalog)
        self.catalog_notice = QLabel()
        self.catalog_notice.setWordWrap(True)
        self.catalog_notice.setObjectName("text_tertiary")
        catalog_layout.addWidget(self.catalog_notice)
        layout.addWidget(catalog_group, stretch=1)

        add_group = QGroupBox("추가 Docker 모델")
        add_layout = QVBoxLayout(add_group)
        note = QLabel(
            "추가 모델은 학습·추론·ONNX export 구현, 컨테이너 이미지, 라이선스·NOTICE를 포함한 "
            ".dvmodel 파일로 배포합니다. 설치 후 학습 화면의 모델 카탈로그에서 선택하세요."
        )
        note.setWordWrap(True)
        add_layout.addWidget(note)
        self.install_root = QLabel()
        self.install_root.setObjectName("text_tertiary")
        self.install_root.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        add_layout.addWidget(self.install_root)
        buttons = QHBoxLayout()
        self.install_button = QPushButton("모델 추가 (.dvmodel)")
        self.install_button.setToolTip("서명과 체크섬을 검증한 Docker 모델 팩을 사용자 모델 폴더에 설치합니다.")
        self.install_button.clicked.connect(self._install_pack)
        buttons.addWidget(self.install_button)
        self.refresh_button = QPushButton("목록 새로 고침")
        self.refresh_button.clicked.connect(self.refresh_catalog)
        buttons.addWidget(self.refresh_button)
        buttons.addStretch()
        add_layout.addLayout(buttons)
        layout.addWidget(add_group)

    def set_project(self, project):
        self.project = project

    def refresh_catalog(self):
        """Only registry metadata is read; installed pack code is never imported here."""
        from core.model_registry import (builtin_model_specs, default_installed_model_root,
                                         registry_with_installed_packs)

        self.catalog.clear()
        registry, errors = registry_with_installed_packs()
        builtins = {spec.model_id for spec in builtin_model_specs()}
        for spec in registry.list():
            category = "기본 제공" if spec.model_id in builtins else "추가 Docker 모델"
            runtime = "내장" if "windows_native" in spec.runtimes else "Docker 팩"
            item = QTreeWidgetItem([
                category, _TASK_LABELS.get(spec.task, spec.task), spec.display_name,
                runtime,
            ])
            item.setToolTip(2, f"{spec.model_id}\n{spec.notes}".strip())
            self.catalog.addTopLevelItem(item)
        self.catalog.resizeColumnToContents(0)
        self.catalog.resizeColumnToContents(1)
        self.catalog.resizeColumnToContents(3)
        self.install_root.setText(f"설치 위치: {default_installed_model_root()}")
        if errors:
            self.catalog_notice.setText("설치된 모델 팩 일부를 읽지 못했습니다:\n" + "\n".join(errors))
        else:
            self.catalog_notice.setText("기본 제공 모델은 내장 실행 경로를 사용합니다. Docker 팩은 사용자가 추가한 모델에만 사용합니다.")

    def _install_pack(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "추가 Docker 모델 선택", "", "Deep Vision model pack (*.dvmodel)"
        )
        if not filepath:
            return
        try:
            from core.model_registry import default_installed_model_root
            from model_runtime.pack_installer import PackInstaller
            installed = PackInstaller(default_installed_model_root()).install(Path(filepath))
        except Exception as exc:
            QMessageBox.warning(self, "모델 추가 실패", str(exc))
            return
        self.refresh_catalog()
        self.models_changed.emit()
        QMessageBox.information(
            self, "모델 추가 완료",
            f"{installed.model_id} {installed.pack_version}을 설치했습니다.\n"
            "학습 화면의 모델 카탈로그에서 선택하세요.",
        )
