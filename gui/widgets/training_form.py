"""학습 기본/고급 설정과 모니터 레이아웃 구성."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy, QPushButton, QLabel, QLineEdit, QFrame, QScrollArea, QGroupBox, QFormLayout, QCheckBox, QSplitter, QProgressBar, QTextEdit, QTabWidget, QTableWidget, QAbstractItemView
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# 마우스 휠로 하이퍼파라미터가 실수로 바뀌는 것을 막는 위젯
from widgets.common import (
    NoWheelSpinBox, NoWheelDoubleSpinBox, NoWheelComboBox,
)


from widgets.training_results import LossChart, MetricChart, LRChart, ConfusionMatrixChart, EvalResultsWidget, ModelCompareWidget, _metric_info_for

class TrainingForm:
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)

        # ── 타이틀 ──
        title = QLabel("모델 학습")
        title.setObjectName("page_title")
        layout.addWidget(title)

        # ── 메인 스플리터 (좌우 비율 드래그로 조절 가능) ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter  # 참조 보관

        # ========== 왼쪽: 설정 패널 ==========
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        # Narrow windows stack panels; each scroll area keeps its controls reachable.
        left_scroll.setMinimumWidth(0)
        left_widget = QWidget()
        self.settings_panel = left_widget
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(12)

        # ── 모델 카탈로그 선택 ──
        # Model IDs are persisted in the project so the same adapter is used
        # by the desktop job, ONNX exporter, and native SDK.  Requested
        # container models remain visible with their status; the start guard
        # explains when their installed pack is required.
        model_group = QGroupBox("모델 카탈로그")
        self.model_group = model_group
        model_layout = QVBoxLayout(model_group)
        model_desc = QLabel("모델 ID는 학습·내보내기·C++/C# 배포 계약을 함께 선택합니다.")
        model_desc.setObjectName("text_tertiary")
        model_desc.setWordWrap(True)
        model_layout.addWidget(model_desc)
        self.model_id_combo = NoWheelComboBox()
        try:
            from core.model_registry import ModelRegistry
            for spec in ModelRegistry.builtin().list():
                self.model_id_combo.addItem(
                    f"{spec.display_name} · {spec.task} · {spec.release_status}",
                    spec.model_id)
        except Exception:
            # A frozen UI can still open a legacy project when the optional
            # catalog module is unavailable; set_project adds its saved ID.
            pass
        self.model_id_combo.currentIndexChanged.connect(self._on_model_id_changed)
        model_layout.addWidget(self.model_id_combo)
        left_layout.addWidget(model_group)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  학습 모드 선택
        #  ┌────────────────────────────────────────┐
        #  │ ① EfficientNet 사전학습 파인튜닝    │
        #  │ ② 이전 학습 모델 이어학습                │
        #  │ ③ Custom CSP 스크래치 학습             │
        #  └────────────────────────────────────────┘
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        mode_group = QGroupBox("학습 모드")
        self.mode_group = mode_group
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(6)

        # 모드 설명 라벨
        mode_desc = QLabel(
            "분류 기본 모델: EfficientNet B0/B1, ImageNet 사전학습 가중치.\n"
            ""
            "기존 Custom CSP 분할은 클래스 인덱스 마스크를 사용합니다."
        )
        mode_desc.setObjectName("text_tertiary")
        mode_desc.setWordWrap(True)
        mode_layout.addWidget(mode_desc)

        # 모드 콤보박스
        self.mode_combo = NoWheelComboBox()
        self.mode_combo.addItem("EfficientNet 사전학습 모델로 시작", "efficientnet_finetune")
        self.mode_combo.addItem("EfficientNet 내 가중치로 추가 학습", "efficientnet_transfer")
        self.mode_combo.addItem("EfficientNet 중단한 학습 재개", "efficientnet_resume")
        self.mode_combo.addItem("Custom CSP (스크래치)", "custom")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        self.mode_help = QLabel()
        self.mode_help.setWordWrap(True)
        mode_layout.addWidget(self.mode_help)

        self.efficientnet_frame = QFrame()
        efficientnet_layout = QFormLayout(self.efficientnet_frame)
        efficientnet_layout.setContentsMargins(0, 4, 0, 0)
        self.efficientnet_model_combo = NoWheelComboBox()
        self.efficientnet_model_combo.addItem("EfficientNet B0 (기본 224 px)", "efficientnet_b0")
        self.efficientnet_model_combo.addItem("EfficientNet B1 (기본 240 px)", "efficientnet_b1")
        self.efficientnet_model_combo.currentIndexChanged.connect(self._on_efficientnet_model_changed)
        efficientnet_layout.addRow("분류 모델:", self.efficientnet_model_combo)
        self.efficientnet_no_decay_check = QCheckBox("BatchNorm과 bias의 weight decay 제외")
        self.efficientnet_no_decay_check.setChecked(True)
        efficientnet_layout.addRow(self.efficientnet_no_decay_check)
        mode_layout.addWidget(self.efficientnet_frame)

        # ── 이전 모델 경로 선택 (모드 ②) ──
        self.resume_frame = QFrame()
        resume_inner = QFormLayout(self.resume_frame)
        resume_inner.setContentsMargins(0, 4, 0, 0)

        resume_row = QHBoxLayout()
        self.resume_edit = QLineEdit()
        self.resume_edit.setPlaceholderText("추가 학습할 가중치 또는 중단 시 last.pt")
        resume_row.addWidget(self.resume_edit)
        resume_browse = QPushButton("...")
        resume_browse.setFixedWidth(40)
        resume_browse.setToolTip("이전 학습 체크포인트 선택")
        resume_browse.clicked.connect(self._browse_resume_weights)
        resume_row.addWidget(resume_browse)
        resume_inner.addRow("모델 경로:", resume_row)
        self.resume_frame.hide()
        mode_layout.addWidget(self.resume_frame)

        # ── 파인튜닝 옵션 (EfficientNet 전이 학습 공통) ──
        # ┌────────────────────────────────────────────────────┐
        # │ [ ] Full Fine-tuning (전체 학습)                    │
        # │ [v] 백본 동결 — 헤드만 학습 (Feature Extraction)     │
        # └────────────────────────────────────────────────────┘
        self.finetune_frame = QFrame()
        ul_ft_layout = QVBoxLayout(self.finetune_frame)
        ul_ft_layout.setContentsMargins(0, 8, 0, 0)
        ul_ft_layout.setSpacing(4)

        ft_label = QLabel("학습 전략:")
        ft_label.setObjectName("section_label")
        ul_ft_layout.addWidget(ft_label)

        self.finetune_freeze_check = QCheckBox(
            "백본 동결 — 헤드만 학습 (Feature Extraction)"
        )
        self.finetune_freeze_check.setToolTip(
            "Full Fine-tuning vs Feature Extraction\n\n"
            "해제 (기본) — Full Fine-tuning\n"
            "  → 백본 + 헤드 전체 파라미터 학습\n"
            "  → 검사 영상에 맞게 특징 조정 가능\n\n"
            "체크 — Feature Extraction\n"
            "  → 백본 가중치 동결, 백본 이후 모듈 학습\n"
            "  → 데이터 적을 때 과적합 방지\n"
            "  → 학습 속도 빠름 (그래디언트 계산 감소)"
        )
        ul_ft_layout.addWidget(self.finetune_freeze_check)

        self.finetune_frame.hide()
        mode_layout.addWidget(self.finetune_frame)

        left_layout.addWidget(mode_group)

        # ── PatchCore 옵션 (Anomaly 태스크 전용) ──
        self.patchcore_group = QGroupBox("이상 탐지 방법")
        pc_layout = QFormLayout(self.patchcore_group)
        pc_layout.setSpacing(6)

        self.anomaly_method_combo = NoWheelComboBox()
        self.anomaly_method_combo.addItem(
            "PatchCore (권장 — 메모리 뱅크 기반)", "patchcore"
        )
        self.anomaly_method_combo.addItem(
            "Reconstruction (재구성 기반)", "reconstruction"
        )
        self.anomaly_method_combo.setToolTip(
            "PatchCore (권장)\n"
            "  사전학습 백본으로 정상 패치 특징을 추출해 메모리 뱅크를 구성합니다.\n"
            "  추론 시 kNN 거리로 이상을 판정하며, 패치별 이상 히트맵을 생성합니다.\n"
            "  백본은 고정하며, 정상 특징을 한 번 추출합니다.\n\n"
            "Reconstruction (재구성 기반)\n"
            "  Custom CSP 백본 + 디코더로 정상 이미지를 재구성합니다.\n"
            "  추론 시 재구성 오차가 큰 영역을 이상으로 판정합니다.\n"
            "  에폭 반복 학습이 필요하며, PatchCore 대비 성능이 낮을 수 있습니다."
        )
        pc_layout.addRow("방법:", self.anomaly_method_combo)

        self.pc_backbone_combo = NoWheelComboBox()
        for label, name in (("Wide ResNet50-2", "wide_resnet50_2"),
                            ("ResNet50", "resnet50"), ("ResNet18", "resnet18")):
            self.pc_backbone_combo.addItem(label, name)
        pc_layout.addRow("사전학습 백본:", self.pc_backbone_combo)
        self.pc_source_combo = NoWheelComboBox()
        self.pc_source_combo.addItem("ImageNet (자동 다운로드 / 캐시)", "imagenet")
        self.pc_source_combo.addItem("로컬 ResNet 백본 가중치", "backbone")
        self.pc_source_combo.addItem("기존 PatchCore 모델", "patchcore")
        pc_layout.addRow("가중치 출처:", self.pc_source_combo)
        pc_weights_row = QHBoxLayout()
        self.pc_weights_edit = QLineEdit()
        self.pc_weights_edit.setPlaceholderText("선택한 출처에 맞는 .pt / .pth 파일")
        pc_weights_row.addWidget(self.pc_weights_edit)
        self.pc_weights_browse = QPushButton("...")
        self.pc_weights_browse.clicked.connect(self._browse_patchcore_weights)
        pc_weights_row.addWidget(self.pc_weights_browse)
        pc_layout.addRow("가중치 파일:", pc_weights_row)
        self.pc_append_check = QCheckBox("기존 정상 특징 유지 후 새 정상 특징 추가")
        self.pc_append_check.setToolTip(
            "기존 PatchCore 모델에서만 사용합니다. 입력 크기가 같아야 하며, "
            "메모리 뱅크 한도가 기존 특징 수보다 커야 합니다.\n"
            "해제하면 저장된 백본으로 새 데이터의 메모리 뱅크를 재구축합니다."
        )
        pc_layout.addRow("", self.pc_append_check)

        self.pc_sampling_spin = NoWheelDoubleSpinBox()
        self.pc_sampling_spin.setRange(0.001, 0.5)
        self.pc_sampling_spin.setDecimals(3)
        self.pc_sampling_spin.setSingleStep(0.005)
        self.pc_sampling_spin.setValue(0.01)
        self.pc_sampling_spin.setToolTip(
            "코어셋 비율: 전체 패치 중 메모리 뱅크에 저장할 비율\n"
            "비율로 구한 수를 후보 수와 메모리 뱅크 한도로 제한합니다.\n"
            "대표 패치 수를 늘리면 저장 공간과 추론 비용이 증가합니다.\n"
            "검출 성능은 별도 검증 데이터로 비교하세요."
        )
        pc_layout.addRow("코어셋 비율:", self.pc_sampling_spin)

        self.pc_neighbors_spin = NoWheelSpinBox()
        self.pc_neighbors_spin.setRange(1, 50)
        self.pc_neighbors_spin.setValue(9)
        self.pc_neighbors_spin.setToolTip(
            "kNN 이웃 수: 이상 스코어 계산에 사용할 최근접 패치 수\n"
            "  9 — 기본값, 안정적인 거리 추정\n"
            "  1~3 — 민감한 감지 (노이즈에 취약)\n"
            "  20+ — 안정적이지만 미세 이상 놓칠 수 있음"
        )
        pc_layout.addRow("kNN 이웃 수:", self.pc_neighbors_spin)

        self.pc_candidates_spin = NoWheelSpinBox()
        self.pc_candidates_spin.setRange(1, 200000)
        self.pc_candidates_spin.setValue(20000)
        pc_layout.addRow("후보 패치 한도:", self.pc_candidates_spin)
        self.pc_bank_spin = NoWheelSpinBox()
        self.pc_bank_spin.setRange(1, 100000)
        self.pc_bank_spin.setValue(4096)
        pc_layout.addRow("메모리 뱅크 한도:", self.pc_bank_spin)
        self.pc_seed_spin = NoWheelSpinBox()
        self.pc_seed_spin.setRange(0, 2147483647)
        self.pc_seed_spin.setValue(0)
        pc_layout.addRow("특징 선택 시드:", self.pc_seed_spin)
        self.pc_crop_check = QCheckBox("중앙 크롭 사용")
        self.pc_crop_check.setToolTip("원본 중앙을 지정한 픽셀 크기로 자른 뒤 모델 입력 크기로 리사이즈합니다. 저장한 모델의 추론에도 같은 크롭을 적용합니다.")
        self.pc_crop_width_spin = NoWheelSpinBox()
        self.pc_crop_height_spin = NoWheelSpinBox()
        for spin, name in ((self.pc_crop_width_spin, "크롭 가로 (px)"), (self.pc_crop_height_spin, "크롭 세로 (px)")):
            spin.setRange(1, 65536)
            spin.setValue(1024)
            spin.setAccessibleName(name)
        self.pc_crop_info = QLabel()
        self.pc_crop_info.setWordWrap(True)
        self.pc_crop_check.toggled.connect(self._on_patchcore_settings_changed)
        self.pc_crop_width_spin.valueChanged.connect(self._on_patchcore_settings_changed)
        self.pc_crop_height_spin.valueChanged.connect(self._on_patchcore_settings_changed)
        self.pc_info_label = QLabel(
            "사전학습 백본을 고정하고 정상 이미지 특징을 저장합니다. "
            "에폭, 학습률, 옵티마이저, 증강은 사용하지 않습니다. "
            "ImageNet은 최초 실행 시 다운로드하며, 오프라인에서는 로컬 가중치를 선택하세요. "
            "후보/뱅크 한도에 도달하면 특징을 샘플링하므로 성능을 검증하세요."
        )
        self.pc_info_label.setWordWrap(True)
        pc_layout.addRow(self.pc_info_label)
        self.anomaly_method_combo.currentIndexChanged.connect(self._on_patchcore_settings_changed)
        self.pc_source_combo.currentIndexChanged.connect(self._on_patchcore_settings_changed)

        self.patchcore_group.hide()  # anomaly 태스크에서만 표시
        left_layout.addWidget(self.patchcore_group)

        # ── 학습 하이퍼파라미터 ──
        hp_group = QGroupBox("하이퍼파라미터")
        hp_layout = QFormLayout(hp_group)
        hp_layout.setSpacing(8)

        self.epochs_spin = NoWheelSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        hp_layout.addRow("에폭:", self.epochs_spin)

        self.batch_spin = NoWheelSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(8)
        hp_layout.addRow("배치 크기:", self.batch_spin)

        self.lr_spin = NoWheelDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(1e-3)
        hp_layout.addRow("학습률:", self.lr_spin)

        self.wd_spin = NoWheelDoubleSpinBox()
        self.wd_spin.setRange(0, 0.1)
        self.wd_spin.setDecimals(5)
        self.wd_spin.setSingleStep(1e-4)
        self.wd_spin.setValue(5e-4)
        hp_layout.addRow("Weight Decay:", self.wd_spin)

        self.optimizer_combo = NoWheelComboBox()
        self.optimizer_combo.addItems(["AdamW", "SGD", "Adam"])
        hp_layout.addRow("옵티마이저:", self.optimizer_combo)

        self.scheduler_combo = NoWheelComboBox()
        self.scheduler_combo.addItems(["Cosine Annealing", "Step", "None"])
        hp_layout.addRow("스케줄러:", self.scheduler_combo)

        self.patience_spin = NoWheelSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(15)
        hp_layout.addRow("Early Stop:", self.patience_spin)
        self.selection_combo = NoWheelComboBox()
        self.selection_combo.currentIndexChanged.connect(self._selection_changed)
        hp_layout.addRow("Best 선정:", self.selection_combo)
        self.selection_info = QLabel()
        self.selection_info.setWordWrap(True)
        hp_layout.addRow(self.selection_info)

        # 클래스 불균형 보정 (자체 모델 및 EfficientNet 분류 학습)
        self.class_weight_combo = NoWheelComboBox()
        self.class_weight_combo.addItem("사용 안 함", "none")
        self.class_weight_combo.addItem("Balanced (역빈도)", "balanced")
        self.class_weight_combo.addItem("Sqrt (완만한 보정)", "sqrt")
        self.class_weight_combo.setToolTip(
            "클래스별 이미지 수가 크게 다를 때 손실에 가중치를 적용합니다.\n"
            "Balanced: 표본이 적은 클래스일수록 큰 가중치 (완전 역빈도)\n"
            "Sqrt: 극단적 불균형에서 학습이 불안정할 때 완만하게 보정\n"
            "자체 모델 및 EfficientNet 분류 학습에 적용됩니다.\n"
            "클래스별 이미지 수가 같으면 두 방식 모두 일반 손실과 같습니다.\n"
            "학습 로그에서 실제 클래스별 표본 수와 적용 가중치를 확인하세요.\n"
            "전체 정확도가 같아도 클래스별 Recall과 F1은 달라질 수 있습니다."
        )
        self.class_weight_label = QLabel("클래스 가중치:")
        hp_layout.addRow(self.class_weight_label, self.class_weight_combo)

        # ── 입력 크기 (32~1024, 32배수 단위) ──
        # 640 권장: 해상도 보존 vs 학습 속도 최적 밸런스
        self.input_size_spin = NoWheelSpinBox()
        self.input_size_spin.setRange(32, 1024)
        self.input_size_spin.setSingleStep(32)
        self.input_size_spin.setValue(224)
        self.input_size_spin.setToolTip(
            "모델에 입력되는 이미지 크기 (32의 배수)\n\n"
            "원본 이미지가 크더라도 이 크기로 자동 리사이즈됩니다.\n\n"
            "  224  — 빠른 학습, 기본 분류에 적합\n"
            "  416  — 디텍션 기본값\n"
            "  512  — 세밀한 분류/세그멘테이션\n"
            "  640  — 권장: 해상도↔속도 최적 밸런스\n"
            "  1024 — 최고 해상도 (batch_size 1~2 필요)\n\n"
            "클수록 GPU 메모리 사용량 급증\n"
            "  batch_size를 함께 줄여야 OOM 방지"
        )
        hp_layout.addRow("입력 크기:", self.input_size_spin)
        self.input_size_spin.valueChanged.connect(self._on_patchcore_settings_changed)
        hp_layout.addRow("", self.pc_crop_check)
        hp_layout.addRow("크롭 가로 (px):", self.pc_crop_width_spin)
        hp_layout.addRow("크롭 세로 (px):", self.pc_crop_height_spin)
        hp_layout.addRow(self.pc_crop_info)

        # ── 디바이스 선택 (GPU/CPU) ──
        from core.device_manager import get_device_manager
        self._device_manager = get_device_manager()

        self.device_combo = NoWheelComboBox()
        for label, value in self._device_manager.get_combo_items():
            self.device_combo.addItem(label, value)
        hp_layout.addRow("디바이스:", self.device_combo)

        # GPU 상태 표시
        self.device_status = QLabel(self._device_manager.get_status_text())
        self.device_status.setObjectName("text_secondary")
        hp_layout.addRow("", self.device_status)

        # AMP (Mixed Precision) 체크박스
        self.amp_check = QCheckBox("Mixed Precision (FP16) — GPU 시 속도↑, 메모리↓")
        self.amp_check.setChecked(True)
        self.amp_check.setEnabled(self._device_manager.cuda_available)
        hp_layout.addRow("", self.amp_check)

        left_layout.addWidget(hp_group)

        # ── 트랜스퍼 러닝 (커스텀 모드 전용) ──
        self.tl_group = QGroupBox("커스텀 모델 트랜스퍼 러닝")
        tl_layout = QFormLayout(self.tl_group)

        # 사전학습 가중치 경로
        weights_row = QHBoxLayout()
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("Custom CSP .pt 파일 (선택)")
        weights_row.addWidget(self.weights_edit)
        weights_browse = QPushButton("...")
        weights_browse.setFixedWidth(40)
        weights_browse.setToolTip("가중치 파일 선택")
        weights_browse.clicked.connect(self._browse_weights)
        weights_row.addWidget(weights_browse)

        # 지정 해제 버튼 — 비우면 스크래치(랜덤 초기화) 학습
        weights_clear = QPushButton("✕")
        weights_clear.setFixedWidth(30)
        weights_clear.setToolTip("가중치 지정 해제 — 스크래치 학습")
        weights_clear.clicked.connect(self.weights_edit.clear)
        weights_row.addWidget(weights_clear)
        tl_layout.addRow("가중치:", weights_row)

        # 백본 동결
        self.freeze_check = QCheckBox("백본 레이어 동결 (헤드만 학습)")
        tl_layout.addRow("", self.freeze_check)

        # 백본 학습률 배수
        self.backbone_lr_spin = NoWheelDoubleSpinBox()
        self.backbone_lr_spin.setRange(0.001, 1.0)
        self.backbone_lr_spin.setDecimals(3)
        self.backbone_lr_spin.setValue(0.1)
        tl_layout.addRow("백본 LR 배수:", self.backbone_lr_spin)

        # 기본 숨김 (초기에는 사용하지 않음)
        self.tl_group.hide()
        left_layout.addWidget(self.tl_group)

        # ── 데이터 증강 ──
        aug_group = QGroupBox("데이터 증강")
        aug_layout = QFormLayout(aug_group)

        self.hflip_spin = NoWheelDoubleSpinBox()
        self.hflip_spin.setRange(0, 1)
        self.hflip_spin.setSingleStep(0.1)
        self.hflip_spin.setValue(0.5)
        aug_layout.addRow("수평 뒤집기:", self.hflip_spin)
        self.unsupported_aug_reset = QPushButton("저장된 미지원 수직 반전과 Mixup을 0으로 변경")
        self.unsupported_aug_reset.clicked.connect(self._reset_unsupported_augmentation)
        self.unsupported_aug_reset.hide()
        aug_layout.addRow(self.unsupported_aug_reset)

        self.rotation_spin = NoWheelDoubleSpinBox()
        self.rotation_spin.setRange(0, 180)
        self.rotation_spin.setValue(15)
        aug_layout.addRow("회전 각도:", self.rotation_spin)

        self.color_jitter_spin = NoWheelDoubleSpinBox()
        self.color_jitter_spin.setRange(0, 1)
        self.color_jitter_spin.setSingleStep(0.05)
        self.color_jitter_spin.setValue(0.2)
        aug_layout.addRow("색상 변화:", self.color_jitter_spin)

        self.advanced_check = QCheckBox("고급 설정: 최적화 상세와 데이터 증강")
        self.advanced_check.setChecked(False)
        aug_group.setVisible(False)
        self.advanced_check.toggled.connect(aug_group.setVisible)
        left_layout.addWidget(self.advanced_check)
        left_layout.addWidget(aug_group)
        self.hp_group, self.aug_group = hp_group, aug_group
        self.debug_group = QGroupBox("선택 레이어 관찰")
        debug_layout = QFormLayout(self.debug_group)
        self.debug_check = QCheckBox("초기 배치의 출력과 역전파 통계 저장")
        self.debug_patterns = QLineEdit("features.0,features.1.*,classifier.1")
        self.debug_patterns.setMaxLength(1024)
        self.debug_patterns.setToolTip("쉼표로 레이어 이름 분리. * 와 ? 패턴 지원. 최대 128개 레이어.")
        self.debug_batches = NoWheelSpinBox()
        self.debug_batches.setRange(1, 10)
        self.debug_default = QPushButton("현재 모델의 추천 패턴")
        self.debug_default.clicked.connect(self._default_debug_patterns)
        self.debug_info = QLabel("관찰한 뒤 전체 학습을 계속합니다. 원본 이미지와 활성값 배열은 저장하지 않습니다.")
        self.debug_info.setWordWrap(True)
        debug_layout.addRow(self.debug_check)
        debug_layout.addRow("레이어 패턴:", self.debug_patterns)
        debug_layout.addRow("관찰 배치 수:", self.debug_batches)
        debug_layout.addRow(self.debug_default)
        debug_layout.addRow(self.debug_info)
        left_layout.addWidget(self.debug_group)
        left_layout.addStretch()

        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # ========== 오른쪽: 모니터링 + 평가 패널 ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)

        # ── 컨트롤 바 ──
        ctrl_bar = QHBoxLayout()

        self.start_btn = QPushButton("학습 시작")
        self.start_btn.setProperty("cssClass", "primary")
        self.start_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.start_btn.setFixedHeight(44)
        self.start_btn.clicked.connect(self._start_training)
        ctrl_bar.addWidget(self.start_btn)

        self.stop_btn = QPushButton("중지")
        self.stop_btn.setProperty("cssClass", "danger")
        self.stop_btn.setFixedHeight(44)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        ctrl_bar.addWidget(self.stop_btn)

        ctrl_bar.addStretch()

        # 현재 상태 표시
        self.status_label = QLabel("대기 중")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setWordWrap(True)
        ctrl_bar.addWidget(self.status_label)

        right_layout.addLayout(ctrl_bar)

        # ── 프로그레스 바 ──
        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_row.addWidget(self.progress_bar, stretch=1)

        self.eta_label = QLabel("ETA: --")
        self.eta_label.setObjectName("text_secondary")
        progress_row.addWidget(self.eta_label)

        right_layout.addLayout(progress_row)

        # ── 메트릭 카드 행 (태스크별 동적) ──
        self.metrics_card_row = QGridLayout()
        self.metric_cards = {}
        # 기본 4개 카드 생성 (태스크 설정 시 갱신)
        for index, (name, default) in enumerate([("Best Metric", "—"), ("Best Epoch", "—"),
                               ("Train Loss", "—"), ("Val Loss", "—")]):
            card = self._create_metric_card(name, default)
            self.metrics_card_row.addWidget(card, index // 2, index % 2)
        right_layout.addLayout(self.metrics_card_row)
        self.best_selection_label = QLabel("Best 선정 결과: 대기")
        self.best_selection_label.setWordWrap(True)
        right_layout.addWidget(self.best_selection_label)
        self.run_identity_label = QLabel("선택된 학습 결과 없음")
        self.run_identity_label.setWordWrap(True)
        self.run_identity_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.run_identity_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right_layout.addWidget(self.run_identity_label)

        # ── 차트 탭 (6개 탭) ──
        self.chart_tabs = QTabWidget()

        # Loss 차트
        self.loss_chart = LossChart()
        self.chart_tabs.addTab(self.loss_chart, "Loss")

        # Metrics 차트
        self.metric_chart = MetricChart()
        self.chart_tabs.addTab(self.metric_chart, "Metrics")

        # Learning Rate 차트
        self.lr_chart = LRChart()
        self.chart_tabs.addTab(self.lr_chart, "LR Schedule")

        # Confusion Matrix
        self.cm_chart = ConfusionMatrixChart()
        self.chart_tabs.addTab(self.cm_chart, "Confusion Matrix")

        # 평가 결과
        self.eval_widget = EvalResultsWidget()
        self.chart_tabs.addTab(self.eval_widget, "평가 결과")

        # 모델 비교
        self.compare_widget = ModelCompareWidget()
        self.chart_tabs.addTab(self.compare_widget, "모델 비교")
        self.debug_table = QTableWidget(0, 9)
        self.debug_table.setHorizontalHeaderLabels(["레이어", "입력 shape", "출력 shape", "최소", "최대", "평균", "출력 유한", "Gradient 평균", "Gradient 유한"])
        self.debug_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.debug_table.setAlternatingRowColors(True)
        self.chart_tabs.addTab(self.debug_table, "레이어 관찰")

        right_layout.addWidget(self.chart_tabs, stretch=1)

        # ── 학습 로그 ──
        log_group = QGroupBox("학습 로그")
        log_inner = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        log_inner.addWidget(self.log_text)
        right_layout.addWidget(log_group)

        right_widget.setMinimumWidth(360)
        right_widget.setMinimumHeight(520)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)

        # ── 스플리터 핸들 스타일 (드래그 가시성 강화) ──
        # ┌─────┬────────────────────────────────────┐
        # │     │← 드래그 핸들 (hover 시 초록색)     │
        # │ 설정 │   실시간 모니터링 + 평가 결과       │
        # │     │                                    │
        # └─────┴────────────────────────────────────┘
        splitter.setHandleWidth(6)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("""
            QSplitter::handle:horizontal {
                background-color: #1e222c;
                border-radius: 2px;
                margin: 4px 0px;
            }
            QSplitter::handle:horizontal:hover {
                background-color: #5590F0;
            }
            QSplitter::handle:horizontal:pressed {
                background-color: #4478D8;
            }
        """)
        splitter.setSizes([560, 840])
        layout.addWidget(splitter, stretch=1)

    def _create_metric_card(self, name: str, value: str) -> QFrame:
        """메트릭 표시 카드 — Ant Design Statistic 스타일"""
        card = QFrame()
        card.setObjectName("metric_card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 12, 16, 12)
        card_layout.setSpacing(4)

        # 라벨 (위) — 작은 회색 텍스트
        display_name = f"{name} (Best)" if name in ("Train Loss", "Val Loss") else name
        name_label = QLabel(display_name)
        name_label.setWordWrap(True)
        name_label.setObjectName("metric_label")
        name_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        card_layout.addWidget(name_label)

        # 값 (아래) — 큰 강조 숫자
        val_label = QLabel(value)
        val_label.setObjectName("metric_value")
        val_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        card_layout.addWidget(val_label)

        self.metric_cards[name] = val_label
        if name in ("Train Loss", "Val Loss"):
            card.setToolTip("베스트 모델로 선정된 에폭의 손실. 전체 에폭 추이는 Loss 그래프에서 확인.")
        return card

    def _rebuild_metric_cards(self, task: str, project=None):
        """태스크에 맞게 메트릭 카드 갱신"""
        # 기존 카드 제거
        while self.metrics_card_row.count():
            item = self.metrics_card_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.metric_cards.clear()

        # 태스크별 카드 구성
        metric_info = _metric_info_for(project or self.project)
        primary_label = metric_info.get("labels", {}).get(
            metric_info.get("primary", ""), "Best Metric"
        )

        cards = [(f"Best {primary_label}", "—"), ("Best Epoch", "—"),
                 ("Train Loss", "—"), ("Val Loss", "—")]

        for index, (name, default) in enumerate(cards):
            card = self._create_metric_card(name, default)
            self.metrics_card_row.addWidget(card, index // 2, index % 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "splitter"):
            orientation = Qt.Vertical if self.width() < 1100 else Qt.Horizontal
            if self.splitter.orientation() != orientation:
                self.splitter.setOrientation(orientation)
                self.splitter.setSizes([300, 600] if orientation == Qt.Vertical else [500, 700])

    # ── 학습 모드 전환 ─────────────────────────────
