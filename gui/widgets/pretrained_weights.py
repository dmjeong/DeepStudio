"""사전학습 가중치 관리: 어떤 파일이 있고 없는지, 없으면 어디서 받아 어떻게 넣는지.

빌드 때 받지 못한 가중치(회사망 차단 등)도 다시 빌드하지 않고 넣을 수 있게 한다.

    브라우저로 받은 파일 ──[가중치 가져오기]──▶ 확인(해시·형식) ──▶ 사용자 가중치 폴더 + manifest 기록
                       └─[폴더에 직접 넣기]────────────────────▶ 다음 사용 때 확인
"""

from __future__ import annotations

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from core.paths import ensure_python_path

ensure_python_path()
from builtin_assets import (  # noqa: E402
    ASSET_FILES, SOURCE_URLS, asset_status, import_asset, missing_asset_help, user_asset_root,
)

# 가중치 → 그 가중치로 시작하는 모델 (사용자가 '무엇을 받아야 하는지' 고를 수 있게)
USED_BY = {
    "efficientnet_b0": "EfficientNet-B0 분류",
    "efficientnet_b1": "EfficientNet-B1 분류",
    "resnet18": "ResNet18 분류 · UNet ResNet18 분할",
    "resnet34": "DeepLabV3+ ResNet34 분할",
    "resnet50": "ResNet50 분류",
    "wide_resnet50_2": "PatchCore 이상 탐지 (ImageNet 백본)",
    "convnext_v1_tiny": "ConvNeXt Tiny 분류",
    "dinov2_vits14": "DINOv2 ViT-S/14 분류·분할",
    "dinov2_vitb14": "DINOv2 ViT-B/14 분류·분할",
    "libreyolo_classify_mobilenetv4_small": "MobileNetV4 분류",
    "libreyolo_detect_9t": "LibreYOLO9 탐지",
    "re_detr_v4_small": "RT-DETRv4 S 탐지",
    "re_detr_v4_medium": "RT-DETRv4 M 탐지",
    "re_detr_v4_large": "RT-DETRv4 L 탐지",
    "sam2_hiera_tiny": "SAM2 Tiny 분할 · 라벨링 AI 보조",
    "sam2_hiera_small": "SAM2 Small 분할",
    "sam2_hiera_base_plus": "SAM2 Base+ 분할",
    "sam2_hiera_large": "SAM2 Large 분할",
}
STATUS_TEXT = {"ready": "사용 가능", "missing": "없음", "invalid": "손상·다른 파일"}


def choose_and_import(parent, asset_id: str) -> bool:
    """파일 선택 → 확인 → 사용자 가중치 폴더로 복사. 성공하면 True."""
    from pathlib import Path
    expected = Path(ASSET_FILES[asset_id]).name
    path, _ = QFileDialog.getOpenFileName(
        parent, f"{asset_id} 가중치 파일 선택 ({expected})", "", "PyTorch 가중치 (*.pth *.pt);;모든 파일 (*)")
    if not path:
        return False
    try:
        target = import_asset(asset_id, path)
    except (OSError, RuntimeError, ValueError) as exc:
        QMessageBox.warning(parent, "가중치 가져오기 실패", str(exc))
        return False
    QMessageBox.information(parent, "가중치 가져오기 완료", f"{asset_id} 가중치를 사용할 수 있습니다.\n{target}")
    return True


def open_weights_folder():
    folder = user_asset_root()
    folder.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def offer_weight_import(parent, asset_id: str) -> bool:
    """학습 시작 전에 가중치가 없을 때: 받는 주소와 넣는 곳을 알리고 바로 가져오게 한다. 가져오면 True."""
    box = QMessageBox(QMessageBox.Icon.Warning, "사전학습 가중치 없음", missing_asset_help(asset_id),
                      QMessageBox.StandardButton.Close, parent)
    box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    import_button = box.addButton("가중치 가져오기...", QMessageBox.ButtonRole.AcceptRole)
    copy_button = box.addButton("다운로드 주소 복사", QMessageBox.ButtonRole.ActionRole)
    folder_button = box.addButton("폴더 열기", QMessageBox.ButtonRole.ActionRole)
    while True:
        box.exec()
        clicked = box.clickedButton()
        if clicked is copy_button:
            QGuiApplication.clipboard().setText(SOURCE_URLS[asset_id])
            continue
        if clicked is folder_button:
            open_weights_folder()
            continue
        if clicked is import_button:
            return choose_and_import(parent, asset_id)
        return False


class PretrainedWeightsGroup(QGroupBox):
    """Settings · 모델 관리의 '사전학습 가중치' 표."""

    weights_changed = Signal()
    HEADERS = ["가중치", "사용하는 모델", "상태", "파일 이름"]

    def __init__(self, parent=None):
        super().__init__("사전학습 가중치", parent)
        layout = QVBoxLayout(self)
        note = QLabel(
            "빌드 때 받지 못한 가중치도 다시 빌드하지 않고 넣을 수 있습니다. 브라우저로 '다운로드 주소'의 파일을 받아 "
            "'가중치 가져오기'로 고르세요(다른 PC에서 받아 옮겨도 됩니다). manifest는 필요 없습니다.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setAccessibleName("사전학습 가중치 목록")
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table)
        self.folder_label = QLabel()
        self.folder_label.setObjectName("text_tertiary")
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)
        buttons = QHBoxLayout()
        self.import_button = QPushButton("가중치 가져오기...")
        self.import_button.clicked.connect(self._import_selected)
        self.copy_button = QPushButton("다운로드 주소 복사")
        self.copy_button.clicked.connect(self._copy_url)
        self.folder_button = QPushButton("가중치 폴더 열기")
        self.folder_button.clicked.connect(open_weights_folder)
        self.refresh_button = QPushButton("다시 확인")
        self.refresh_button.clicked.connect(self.refresh)
        for button in (self.import_button, self.copy_button, self.folder_button, self.refresh_button):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.refresh()

    def selected_asset(self):
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def refresh(self):
        """설치본·사용자 폴더·torch 캐시를 다시 확인한다 (큰 파일 해시는 한 번만 계산)."""
        from pathlib import Path
        current = self.selected_asset()
        self.table.setRowCount(len(ASSET_FILES))
        missing = 0
        for row, asset_id in enumerate(ASSET_FILES):
            status, path, problem = asset_status(asset_id)
            missing += status != "ready"
            values = [asset_id, USED_BY.get(asset_id, ""), STATUS_TEXT[status], Path(ASSET_FILES[asset_id]).name]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, asset_id)
                if column == 2:
                    item.setForeground(Qt.GlobalColor.darkGreen if status == "ready" else Qt.GlobalColor.red)
                    item.setToolTip(str(path) if path else problem or "설치본·가중치 폴더·torch 캐시에 없습니다")
                if column == 3:
                    item.setToolTip(SOURCE_URLS[asset_id])
                self.table.setItem(row, column, item)
            if asset_id == current:
                self.table.selectRow(row)
        self.folder_label.setText(
            f"가중치 폴더: {user_asset_root()}  ·  없음 {missing}개 / 전체 {len(ASSET_FILES)}개"
            "  (쓰는 모델의 가중치만 있으면 됩니다)")
        self._selection_changed()

    def _selection_changed(self):
        selected = self.selected_asset() is not None
        self.import_button.setEnabled(selected)
        self.copy_button.setEnabled(selected)

    def _import_selected(self):
        asset_id = self.selected_asset()
        if asset_id and choose_and_import(self, asset_id):
            self.refresh()
            self.weights_changed.emit()

    def _copy_url(self):
        asset_id = self.selected_asset()
        if asset_id:
            QGuiApplication.clipboard().setText(SOURCE_URLS[asset_id])
