"""기본 모델과 사용자가 추가한 Docker 모델 팩을 한 화면에서 관리한다."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTreeWidget, QTreeWidgetItem, QFileDialog, QMessageBox, QGroupBox, QComboBox,
)


_TASK_LABELS = {
    "classify": "Classification",
    "anomaly": "Anomaly Detection",
    "detect": "Object Detection",
    "segment": "Segmentation",
}


class _Sam2DownloadWorker(QThread):
    """Keep large official checkpoint downloads off the Qt event thread."""

    downloaded = Signal(str)
    failed = Signal(str)

    def __init__(self, model_id: str, cache_dir: Path, parent=None):
        super().__init__(parent)
        self.model_id, self.cache_dir = model_id, cache_dir

    def run(self):
        try:
            from sam2_assets import download_sam2_pretrained
            self.downloaded.emit(str(download_sam2_pretrained(self.model_id, self.cache_dir)))
        except Exception as exc:
            self.failed.emit(str(exc))


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

        sam_group = QGroupBox("SAM2.1 사전학습 가중치")
        sam_layout = QVBoxLayout(sam_group)
        sam_note = QLabel(
            "SAM2는 Meta가 공개한 SAM2.1 Hiera 가중치를 사용합니다. 아래에서 선택한 한 변형만 "
            "공식 저장소에서 내려받아 사용자 모델 캐시에 저장합니다. 다운로드는 백그라운드에서 실행됩니다."
        )
        sam_note.setWordWrap(True)
        sam_layout.addWidget(sam_note)
        sam_buttons = QHBoxLayout()
        self.sam2_variant = QComboBox()
        self.sam2_variant.addItem("SAM2.1 Hiera Tiny", "sam2_hiera_tiny")
        self.sam2_variant.addItem("SAM2.1 Hiera Small", "sam2_hiera_small")
        self.sam2_variant.addItem("SAM2.1 Hiera Base+", "sam2_hiera_base_plus")
        self.sam2_variant.addItem("SAM2.1 Hiera Large", "sam2_hiera_large")
        # currentIndexChanged emits an int. Do not pass it as the optional
        # downloaded-path argument used by the worker completion callback.
        self.sam2_variant.currentIndexChanged.connect(lambda _index: self._update_sam2_apply_state())
        sam_buttons.addWidget(self.sam2_variant)
        self.sam2_download_button = QPushButton("사전학습 가중치 다운로드")
        self.sam2_download_button.clicked.connect(self._download_sam2)
        sam_buttons.addWidget(self.sam2_download_button)
        self.sam2_apply_button = QPushButton("현재 프로젝트에 적용")
        self.sam2_apply_button.setEnabled(False)
        self.sam2_apply_button.clicked.connect(self._apply_sam2_to_project)
        sam_buttons.addWidget(self.sam2_apply_button)
        sam_buttons.addStretch()
        sam_layout.addLayout(sam_buttons)
        self.sam2_status = QLabel("다운로드 후 현재 분할 프로젝트에 적용하면 export에서 바로 사용합니다.")
        self.sam2_status.setObjectName("text_tertiary")
        self.sam2_status.setWordWrap(True)
        sam_layout.addWidget(self.sam2_status)
        layout.addWidget(sam_group)

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
        self._update_sam2_apply_state()

    def _update_sam2_apply_state(self, path: str = ""):
        """Enable the explicit project action only for a downloaded SAM2 weight."""
        from core.model_registry import default_installed_model_root

        candidate = Path(path) if path else (
            default_installed_model_root() / "_builtin_assets" / "sam2" /
            {"sam2_hiera_tiny": "sam2.1_hiera_tiny.pt",
             "sam2_hiera_small": "sam2.1_hiera_small.pt",
             "sam2_hiera_base_plus": "sam2.1_hiera_base_plus.pt",
             "sam2_hiera_large": "sam2.1_hiera_large.pt"}[self.sam2_variant.currentData()]
        )
        valid_project = self.project is not None and self.project.task == "segment"
        self.sam2_apply_button.setEnabled(valid_project and candidate.is_file())
        self.sam2_apply_button.setToolTip(
            "선택한 SAM2 변형과 가중치 경로를 현재 분할 프로젝트에 저장합니다."
            if valid_project else "분할 프로젝트를 먼저 열어주세요."
        )

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

    def _download_sam2(self):
        if getattr(self, "_sam2_worker", None) is not None:
            return
        from core.model_registry import default_installed_model_root
        model_id = self.sam2_variant.currentData()
        cache_dir = default_installed_model_root() / "_builtin_assets" / "sam2"
        self.sam2_download_button.setEnabled(False)
        self.sam2_status.setText(f"{self.sam2_variant.currentText()} 공식 가중치 다운로드 중…")
        worker = _Sam2DownloadWorker(model_id, cache_dir, self)
        self._sam2_worker = worker
        worker.downloaded.connect(self._sam2_downloaded)
        worker.failed.connect(self._sam2_download_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _sam2_downloaded(self, path: str):
        self.sam2_status.setText(f"다운로드 완료: {path}")
        self.sam2_download_button.setEnabled(True)
        self._sam2_worker = None
        self._update_sam2_apply_state(path)

    def _sam2_download_failed(self, message: str):
        self.sam2_status.setText("다운로드 실패: " + message)
        self.sam2_download_button.setEnabled(True)
        self._sam2_worker = None

    def _apply_sam2_to_project(self):
        """Persist an official checkpoint as the selected project's export source.

        This deliberately changes the project model ID too: keeping a SAM2
        weight attached to DeepLab/U-Net would route export through the wrong
        exporter and was the source of a misleading 'downloaded but unused'
        state.
        """
        if self.project is None or self.project.task != "segment":
            self.sam2_status.setText("SAM2 가중치는 분할 프로젝트에만 적용할 수 있습니다.")
            self._update_sam2_apply_state()
            return
        from core.model_registry import default_installed_model_root
        from core.project import ProjectManager
        from sam2_assets import get_sam2_asset

        model_id = self.sam2_variant.currentData()
        checkpoint = (default_installed_model_root() / "_builtin_assets" / "sam2" /
                      get_sam2_asset(model_id).filename).resolve()
        if not checkpoint.is_file():
            self.sam2_status.setText("먼저 선택한 SAM2.1 가중치를 다운로드하세요.")
            self._update_sam2_apply_state()
            return
        self.project.model.model_id = model_id
        self.project.model.pretrained_weights = str(checkpoint)
        self.project.model.pack_path = ""
        ProjectManager.save(self.project)
        self.sam2_status.setText(f"현재 프로젝트에 적용됨: {checkpoint}")
        self.project_updated.emit()
