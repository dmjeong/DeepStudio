"""
Deep Vision Studio — 추론(테스트) 페이지

기능:
┌──────────────────────────────────────────────────────────────┐
│ 1. 체크포인트 로드                                            │
│ 2. 단일 이미지 추론 + 결과 시각화                              │
│ 3. 배치 이미지 추론                                           │
│ 4. 태스크별 결과 표시:                                        │
│    • Classification → 클래스 + 확률 바                        │
│    • Segmentation  → 마스크 오버레이                           │
│    • Detection     → 바운딩 박스 + 레이블                     │
│    • Anomaly       → 히트맵 오버레이 + 이상 스코어             │
│ 5. Grad-CAM 시각화:                                          │
│    • 모델이 "어디를 보고"판단했는지 히트맵으로 표시             │
│    • 모든 태스크에서 지원 (backbone.stage4 기준)                │
└──────────────────────────────────────────────────────────────┘
"""

import os
import time
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFrame,
    QFileDialog, QMessageBox, QGroupBox,
    QSplitter, QProgressBar, QComboBox, QCheckBox,
    QScrollArea, QGridLayout, QStackedWidget, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer, QSignalBlocker
from PySide6.QtGui import QPixmap, QFont, QImage

import torch

# 프로젝트 모듈 경로 추가 (PyInstaller EXE 호환)
from core.paths import ensure_python_path
ensure_python_path()

from patchcore import PatchCore
from core.project import ProjectData
from core.device_manager import get_device_manager
from core.gradcam import GradCAM
from core.heatmap import render_heatmap
from core.inference_types import InferenceResult, make_anomaly_result  # noqa: F401
from core.inference_engine import InferenceOperations, InferenceEngine
from core.inference_cache import InferencePreviewCache
from core.inference_timing import StageTimer, format_duration, format_result_timing, batch_stage_summary, format_runtime_stages
from widgets.common import NoWheelSpinBox
from widgets.result_image_label import ResultImageLabel
from widgets.inference_review import InferenceReview
from core.inference_review import effective_result






class GridImageLabel(ResultImageLabel):
    """격자 썸네일은 스크롤만 사용하고 확대와 이동은 지원하지 않는다."""

    def wheelEvent(self, event):
        event.ignore()

    def mouseDoubleClickEvent(self, event):
        event.ignore()


class GridCell(QFrame):
    """
    격자 뷰의 개별 셀 — 썸네일 + 추론 결과 오버레이

    구조:
    ┌──────────────────────────┐
    │ ┌─────────────┐          │
    │ │ OK  98.3%   │  ← 결과 │
    │ └─────────────┘          │
    │                          │
    │      [썸네일 이미지]      │
    │                          │
    │  파일명.jpg   ← 하단 캡션 │
    └──────────────────────────┘
    """
    # 셀 클릭 시 인덱스 전달
    clicked = Signal(int)

    # 격자 셀 기본/선택 스타일
    STYLE_NORMAL = """
        GridCell {
            background-color: #151920;
            border: 2px solid #1e222c;
            border-radius: 6px;
        }
    """
    STYLE_SELECTED = """
        GridCell {
            background-color: #1a2030;
            border: 2px solid #5590F0;
            border-radius: 6px;
        }
    """

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self._index = index          # 배치 내 인덱스
        self._result_text = ""       # 오버레이 텍스트 (예: "OK  98.3%")
        self._result_color = "#34C759"  # 결과 텍스트 색상
        self._selected = False

        self.setStyleSheet(self.STYLE_NORMAL)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # 썸네일 이미지 라벨
        self.thumb_label = GridImageLabel()
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setMinimumSize(140, 140)
        self.thumb_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.thumb_label.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(self.thumb_label, stretch=1)

        # 파일명 캡션
        self.name_label = QLabel()
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet(
            "color: #8A92A4; font-size: 10px; border: none;"
        )
        self.name_label.setWordWrap(True)
        layout.addWidget(self.name_label)

    def set_image(self, pixmap: QPixmap):
        """원본 썸네일과 결과를 각각 보관하여 크기 변경에도 다시 표시한다."""
        if pixmap.isNull():
            self.thumb_label.setText("이미지 표시 불가")
        else:
            # 격자에는 표시용 축소본만 보관해 고해상도 배치의 메모리를 제한한다.
            # 상세 보기에서는 파일 원본을 별도로 읽어 처리한다.
            thumbnail = pixmap
            if max(pixmap.width(), pixmap.height()) > 512:
                thumbnail = pixmap.scaled(
                    QSize(512, 512), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
            self.thumb_label.setPixmap(thumbnail)
        self.thumb_label.set_result(self._result_text, self._result_color)

    def set_result(self, text: str, color: str = "#34C759"):
        """이미지 로드 순서와 무관하게 결과 배지를 즉시 갱신한다."""
        self._result_text = text
        self._result_color = color
        self.thumb_label.set_result(text, color)

    def set_selected(self, selected: bool):
        """선택 상태 토글"""
        self._selected = selected
        self.setStyleSheet(
            self.STYLE_SELECTED if selected else self.STYLE_NORMAL
        )

    def mousePressEvent(self, event):
        """셀 클릭 → 시그널 발생"""
        self.clicked.emit(self._index)
        super().mousePressEvent(event)


class ResultCard(QFrame):
    """추론 결과 카드"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #151920;
                border: 1px solid #1e222c;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        # 결과 타이틀
        self.title_label = QLabel("추론 결과")
        self.title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #5590F0; border: none;")
        layout.addWidget(self.title_label)

        # 결과 내용
        self.content_label = QLabel("이미지를 선택한 뒤 추론을 실행해 보세요.")
        self.content_label.setWordWrap(True)
        self.content_label.setStyleSheet("color: #E8EAEF; border: none;")
        layout.addWidget(self.content_label)

        # 확률 바 (Classification용)
        self.prob_widget = QWidget()
        self.prob_layout = QVBoxLayout(self.prob_widget)
        self.prob_layout.setContentsMargins(0, 0, 0, 0)
        self.prob_widget.setStyleSheet("border: none;")
        layout.addWidget(self.prob_widget)

        # 이상 스코어 (Anomaly용)
        self.score_label = QLabel()
        self.score_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; border: none;"
        )
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.hide()
        layout.addWidget(self.score_label)

        layout.addStretch()

        # 추론 시간 표시 (하단)
        self.time_label = QLabel("")
        self.time_label.setStyleSheet(
            "color: #8A92A4; font-size: 11px; border: none; "
            "border-top: 1px solid #1e222c; padding-top: 6px;"
        )
        self.time_label.hide()
        layout.addWidget(self.time_label)

    def show_classification_result(self, class_names, probabilities):
        """분류 결과 표시 — 확률 바"""
        self.title_label.setText("Classification Result")

        # 기존 확률 바 제거
        while self.prob_layout.count():
            item = self.prob_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 모든 클래스 확률 표시. 예측 클래스만 보여주면 운영 중 경계 클래스 확인이 어렵다.
        top_indices = np.argsort(probabilities)[::-1]

        predicted = class_names[top_indices[0]] if top_indices[0] < len(class_names) else f"Class {top_indices[0]}"
        self.content_label.setText(f"예측: {predicted}  ({probabilities[top_indices[0]]:.1%})")

        for idx in top_indices:
            prob = probabilities[idx]
            name = class_names[idx] if idx < len(class_names) else f"Class {idx}"

            row = QHBoxLayout()
            name_label = QLabel(f"{name}")
            name_label.setFixedWidth(120)
            name_label.setStyleSheet("color: #E8EAEF; border: none;")
            row.addWidget(name_label)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(prob * 100))
            bar.setTextVisible(True)
            bar.setFormat(f"{prob:.1%}")
            bar.setFixedHeight(22)
            row.addWidget(bar)

            container = QWidget()
            container.setLayout(row)
            container.setStyleSheet("border: none;")
            self.prob_layout.addWidget(container)

        self.prob_widget.show()
        self.score_label.hide()

    def show_error(self, message):
        self.title_label.setText("추론 오류")
        self.content_label.setText(f"ERROR: {message}")
        self.score_label.hide()
        self.prob_widget.hide()
        self.time_label.hide()

    def show_anomaly_result(self, anomaly_score, threshold=None):
        """보정된 임계값이 없으면 정상/불량 대신 미보정 점수만 표시한다."""
        result = make_anomaly_result("", anomaly_score, threshold)
        self.title_label.setText("Anomaly Detection Result")
        decision = "임계값 미보정, 판정 불가" if threshold is None else (
            "NG" if anomaly_score >= threshold else "OK")
        threshold_text = "없음" if threshold is None else f"{threshold:.6g} (>=)"
        self.content_label.setText(
            f"판정: {decision}\n이상 스코어: {anomaly_score:.6f} (임계값: {threshold_text})")
        self.score_label.setText(f"{anomaly_score:.4f}")
        self.score_label.setStyleSheet(
            f"font-size: 32px; font-weight: bold; color: {result.color}; border: none;")
        self.score_label.show()
        self.prob_widget.hide()

    def show_segmentation_result(self, num_classes, pixel_counts):
        """세그멘테이션 결과 표시"""
        self.title_label.setText("Segmentation Result")

        total = sum(pixel_counts.values())
        info_parts = []
        for cls, count in sorted(pixel_counts.items()):
            pct = count / total * 100 if total > 0 else 0
            info_parts.append(f"Class {cls}: {pct:.1f}%")

        self.content_label.setText("\n".join(info_parts))
        self.prob_widget.hide()
        self.score_label.hide()

    def show_detection_result(self, num_detections):
        """탐지 결과 표시"""
        self.title_label.setText("Detection Result")
        self.content_label.setText(f"탐지된 객체: {num_detections}개")
        self.prob_widget.hide()
        self.score_label.hide()

    def set_infer_time(self, elapsed_sec: float):
        """기존 호출자에게 전체 처리 시간을 명확하게 표시한다."""
        self.time_label.setText(f"⏱ 전체 처리: {format_duration(elapsed_sec)}")
        self.time_label.show()

    def set_inference_times(self, result):
        """계산 단계와 이미지 표시 준비/저장을 포함한 전체 시간을 구분한다."""
        self.time_label.setText(
            f"⏱ {format_result_timing(result)}\n{format_runtime_stages(result)}"
            f"전체 처리: {format_duration(result.elapsed_sec)}")
        self.time_label.setWordWrap(True)
        self.time_label.setToolTip(
            "추론: 이미지 파일 읽기, 전처리, 모델 예측, 수치 후처리.\n"
            "판정에 실제 사용한 엔진을 표시합니다. Grad-CAM은 PyTorch에서 별도로 계산합니다.\n"
            "모델 변환/검증/워밍업은 이미지 처리 전 준비 단계이며 추론 시간에 포함하지 않습니다.\n"
            "Grad-CAM: 추가 전처리, 역전파, 반응 맵과 원본 좌표 복원.\n"
            "두 계산 시간은 이미지 표시와 캐시 저장을 제외하며 GPU 완료를 기다려 측정합니다.\n"
            "전체 처리: 이미지/결과 카드 표시 준비와 캐시 저장 포함. 실제 화면 페인트 대기는 제외합니다.\n"
            "이미지를 다시 선택하면 최초 실행에서 저장한 시간을 표시합니다."
            + ("\n" + (result.details or {})["runtime_warning"] if (result.details or {}).get("runtime_warning") else ""))
        self.time_label.show()


class InferenceWidget(QWidget, InferenceOperations):
    """추론/테스트 페이지"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.model = None
        self._patchcore_model = None   # PatchCore 엔진 (anomaly 전용)
        self.class_names = []
        self._infer_device = torch.device("cpu")  # 추론 디바이스
        self._device_manager = get_device_manager()
        self._gradcam: GradCAM = None  # Grad-CAM 인스턴스
        self._batch_results = {}       # {인덱스: InferenceResult}
        self._current_result = None
        self._active_checkpoint = None
        self._process_model = False
        self._onnx_runtime = None
        self._model_inspection = None
        self._input_size = (224, 224)
        self._normalization = None
        self._anomaly_threshold = None
        self._batch_times = {}         # {인덱스: 추론시간(초)}
        self._stage_timings = {}
        self._last_infer_time = 0.0    # 마지막 단일 추론 시간(초)
        self._grid_cells = []          # GridCell 위젯 목록
        self._view_mode = "single"     # "single" | "grid"
        self._heatmap_cache = None
        self._preview_cache = InferencePreviewCache()
        self._inference_results = {}
        self._current_preview_rgb = None
        self.destroyed.connect(lambda _object=None, cache=self._preview_cache: cache.clear())
        self._heatmap_timer = QTimer(self)
        self._heatmap_timer.setSingleShot(True)
        self._heatmap_timer.setInterval(40)
        self._heatmap_timer.timeout.connect(self._render_cached_heatmap)
        self._inference_worker = None
        self._busy_controls = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(8)

        # ── 타이틀 ──
        title = QLabel("모델 추론 / 테스트")
        title.setObjectName("page_title")
        title.setStyleSheet("font-size: 20px; font-weight: 600; margin: 0; padding: 0;")
        async_row = QHBoxLayout()
        async_row.addWidget(title)
        self.inference_progress = QProgressBar()
        self.inference_progress.setRange(0, 1)
        self.inference_progress.setValue(0)
        self.inference_progress.setMaximumWidth(280)
        self.inference_progress.setMaximumHeight(18)
        async_row.addStretch(1)
        async_row.addWidget(self.inference_progress)
        self.cancel_infer_btn = QPushButton("추론 중단")
        self.cancel_infer_btn.setStyleSheet("padding: 4px 8px; min-height: 20px; font-size: 12px;")
        self.cancel_infer_btn.setEnabled(False)
        self.cancel_infer_btn.clicked.connect(self._cancel_inference)
        async_row.addWidget(self.cancel_infer_btn)
        layout.addLayout(async_row)

        # ── 모델 로드 ──
        self.model_group = QFrame()
        self.model_group.setObjectName("inference_input")
        self.model_group.setStyleSheet("""
            QFrame#inference_input { border: 1px solid #232833; border-radius: 6px; }
            QFrame#inference_input QPushButton, QFrame#inference_input QLineEdit,
            QFrame#inference_input QComboBox { padding: 3px 8px; min-height: 16px; font-size: 12px; }
        """)
        model_layout = QGridLayout(self.model_group)
        model_layout.setContentsMargins(8, 4, 8, 4)
        model_layout.setHorizontalSpacing(6)
        model_layout.setVerticalSpacing(2)
        model_layout.addWidget(QLabel("모델"), 0, 0)
        model_layout.setColumnStretch(1, 1)
        self.ckpt_edit = QLineEdit()
        self.ckpt_edit.setPlaceholderText("체크포인트 경로 (.pt)")
        model_layout.addWidget(self.ckpt_edit, 0, 1)

        ckpt_browse = QPushButton("찾아보기...")
        ckpt_browse.clicked.connect(self._browse_checkpoint)
        model_layout.addWidget(ckpt_browse, 0, 2)

        load_btn = QPushButton("모델 로드")
        load_btn.setProperty("cssClass", "primary")
        load_btn.clicked.connect(self._start_model_inspection)
        model_layout.addWidget(load_btn, 0, 3)
        history_btn = QPushButton("작업 기록")
        history_btn.clicked.connect(self._open_history)
        model_layout.addWidget(history_btn, 0, 5)

        self.model_info = QLabel("모델을 먼저 로드해 주세요.")
        self.model_info.setStyleSheet("color: #8A92A4;")
        self.model_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.model_info.setMinimumWidth(0)
        model_layout.addWidget(self.model_info, 1, 0, 1, 4)

        # ── 추론 디바이스 선택 (CPU/GPU) ──
        self.infer_device_combo = QComboBox()
        for label, value in self._device_manager.get_combo_items():
            # 추론에서는 "자동"대신 기본 CPU로 설정
            self.infer_device_combo.addItem(label, value)
        # CPU를 기본 선택으로 설정 (추론은 CPU가 일반적)
        for i in range(self.infer_device_combo.count()):
            if self.infer_device_combo.itemData(i) == "cpu":
                self.infer_device_combo.setCurrentIndex(i)
                break
        self.infer_device_combo.currentIndexChanged.connect(
            self._on_device_changed
        )
        self.infer_device_combo.setMaximumWidth(190)
        self.infer_device_combo.setToolTip("추론 디바이스")
        model_layout.addWidget(self.infer_device_combo, 0, 4)

        # 디바이스 상태 표시 — 콤보 바로 옆에 붙이고 남는 폭은 여백이 흡수
        self.device_info = QLabel("")
        self.device_info.setStyleSheet("color: #8A92A4; font-size: 11px;")
        self.device_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        model_layout.addWidget(self.device_info, 1, 4, 1, 2)
        self._update_device_info()

        region_row = QHBoxLayout()
        region_row.setContentsMargins(0, 0, 0, 0)
        region_row.setSpacing(6)
        region_row.addWidget(QLabel("입력 영역"))
        self.crop_mode_combo = QComboBox()
        for label, value in (("모델 설정 사용", "model"), ("JSON 설정 사용", "json"), ("원본 전체 사용", "full")):
            self.crop_mode_combo.addItem(label, value)
        self.crop_mode_combo.setAccessibleName("추론 입력 영역")
        region_row.addWidget(self.crop_mode_combo)
        self.crop_json_edit = QLineEdit()
        self.crop_json_edit.setPlaceholderText("크롭 JSON 또는 프로젝트 파일")
        self.crop_json_edit.setAccessibleName("크롭 JSON")
        region_row.addWidget(self.crop_json_edit, 1)
        self.crop_json_browse = QPushButton("JSON 선택")
        self.crop_json_browse.clicked.connect(self._browse_crop_json)
        region_row.addWidget(self.crop_json_browse)
        self.crop_region_info = QLabel()
        self.crop_region_info.setStyleSheet("font-size: 11px;")
        self.crop_region_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        region_row.addWidget(self.crop_region_info, 1)
        model_layout.addLayout(region_row, 2, 0, 1, 6)
        self.crop_mode_combo.currentIndexChanged.connect(self._region_changed)
        self.crop_json_edit.textChanged.connect(self._region_changed)
        self._region_changed()

        layout.addWidget(self.model_group)

        # ── 메인 영역: 이미지 + 결과 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 왼쪽: 이미지 뷰어
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 6, 0)
        image_layout.setSpacing(5)
        image_widget.setStyleSheet("""
            QPushButton, QComboBox, QSpinBox { padding: 4px 6px; min-height: 20px; font-size: 12px; }
            QCheckBox { font-size: 12px; }
        """)

        # 이미지 선택
        img_btn_row = QHBoxLayout()
        select_img_btn = QPushButton("이미지 선택")
        select_img_btn.clicked.connect(self._select_image)
        img_btn_row.addWidget(select_img_btn)

        select_dir_btn = QPushButton("배치 폴더")
        select_dir_btn.clicked.connect(self._select_directory)
        img_btn_row.addWidget(select_dir_btn)

        self.infer_btn = QPushButton("추론 시작")
        self.infer_btn.setProperty("cssClass", "primary")
        self.infer_btn.setEnabled(False)
        self.infer_btn.clicked.connect(self._run_inference)
        img_btn_row.addWidget(self.infer_btn)

        # ── 뷰 모드 토글 (하나씩 보기 / 격자 보기) ──
        img_btn_row.addStretch(1)

        self.single_view_btn = QPushButton("하나씩 보기")
        self.single_view_btn.setCheckable(True)
        self.single_view_btn.setChecked(True)
        self.single_view_btn.setFixedWidth(74)
        self.single_view_btn.setStyleSheet(self._view_btn_style(True))
        self.single_view_btn.clicked.connect(
            lambda: self._switch_view_mode("single")
        )
        img_btn_row.addWidget(self.single_view_btn)

        self.grid_view_btn = QPushButton("격자 보기")
        self.grid_view_btn.setCheckable(True)
        self.grid_view_btn.setChecked(False)
        self.grid_view_btn.setFixedWidth(74)
        self.grid_view_btn.setStyleSheet(self._view_btn_style(False))
        self.grid_view_btn.clicked.connect(
            lambda: self._switch_view_mode("grid")
        )
        img_btn_row.addWidget(self.grid_view_btn)

        image_layout.addLayout(img_btn_row)

        # ── Grad-CAM 옵션 ──
        gradcam_row = QHBoxLayout()
        self.gradcam_checkbox = QCheckBox("Grad-CAM")
        self.gradcam_checkbox.setChecked(True)
        self.gradcam_checkbox.setToolTip(
            "모델이 주목한 영역을 히트맵으로 확인할 수 있습니다.\n"
            "추론 시작 시 체크를 해제하면 Grad-CAM을 계산하지 않습니다.\n"
            "빨간색일수록 강한 반응, 파란색일수록 약한 반응"
        )
        self.gradcam_checkbox.setStyleSheet(
            "QCheckBox { color: #E8EAEF; font-size: 12px; }"
        )
        gradcam_row.addWidget(self.gradcam_checkbox)

        alpha_label = QLabel("강도:")
        alpha_label.setStyleSheet("color: #8A92A4; font-size: 11px;")
        gradcam_row.addWidget(alpha_label)

        self.gradcam_alpha_combo = QComboBox()
        for percent in range(0, 101, 10):
            self.gradcam_alpha_combo.addItem(f"{percent}%", percent / 100)
        self.gradcam_alpha_combo.setCurrentIndex(5)
        self.gradcam_alpha_combo.setToolTip("0%는 원본, 100%는 히트맵 색상만 표시합니다.")
        self.gradcam_alpha_combo.setFixedWidth(66)
        gradcam_row.addWidget(self.gradcam_alpha_combo)
        gradcam_row.addStretch(1)
        image_layout.addLayout(gradcam_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("표시 시작:"))
        self.heatmap_lower_spin = NoWheelSpinBox()
        self.heatmap_lower_spin.setRange(0, 99)
        self.heatmap_lower_spin.setSuffix("%")
        self.heatmap_lower_spin.setValue(0)
        self.heatmap_lower_spin.setToolTip("이 값보다 약한 반응은 숨기고 원본을 표시합니다.")
        range_row.addWidget(self.heatmap_lower_spin)
        range_row.addWidget(QLabel("최대 강조:"))
        self.heatmap_upper_spin = NoWheelSpinBox()
        self.heatmap_upper_spin.setRange(1, 100)
        self.heatmap_upper_spin.setSuffix("%")
        self.heatmap_upper_spin.setValue(100)
        self.heatmap_upper_spin.setToolTip("이 값 이상은 가장 강한 히트맵 색상으로 표시합니다.")
        range_row.addWidget(self.heatmap_upper_spin)
        self.heatmap_reset_btn = QPushButton("초기화")
        self.heatmap_reset_btn.clicked.connect(self._reset_heatmap_display)
        range_row.addWidget(self.heatmap_reset_btn)
        range_row.addStretch(1)
        image_layout.addLayout(range_row)
        range_hint = QLabel(
            "0~100%는 이미지별 반응 강도 범위이며 면적 비율이 아닙니다. "
            "0%부터 표시하면 반응 0도 파란색입니다. 중앙 잘림으로 모델이 보지 않은 영역은 원본으로 남습니다."
        )
        range_hint.setWordWrap(True)
        range_hint.setStyleSheet("color: #8A92A4; font-size: 11px;")
        self.image_label_hint = range_hint.text()
        range_hint.deleteLater()
        self.heatmap_lower_spin.setToolTip(self.image_label_hint)
        self.heatmap_upper_spin.setToolTip(self.image_label_hint)
        self.heatmap_lower_spin.valueChanged.connect(lambda _: self._on_heatmap_range_changed("lower"))
        self.heatmap_upper_spin.valueChanged.connect(lambda _: self._on_heatmap_range_changed("upper"))
        self.gradcam_alpha_combo.currentIndexChanged.connect(self._schedule_heatmap_render)
        self.gradcam_checkbox.toggled.connect(self._on_gradcam_toggled)

        # ── 뷰 스택: 하나씩 보기 / 격자 보기 전환 ──
        self._view_stack = QStackedWidget()

        # [페이지 0] 단일 이미지 뷰 (기존)
        self.image_label = ResultImageLabel("이미지를 불러와 주세요")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(240, 180)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #0a0c10;
                border: 2px dashed #2a2f3a;
                border-radius: 8px;
                color: #8A92A4;
                font-size: 14px;
            }
        """)
        compare_box = QWidget()
        compare_layout = QHBoxLayout(compare_box)
        compare_layout.setContentsMargins(0, 0, 0, 0)
        self.original_label = ResultImageLabel("원본")
        self.original_label.hide()
        compare_layout.addWidget(self.original_label)
        compare_layout.addWidget(self.image_label)
        self.image_label.view_changed.connect(self.original_label.set_view)
        self.original_label.view_changed.connect(self.image_label.set_view)
        self._view_stack.addWidget(compare_box)
        self.compare_check = QCheckBox("원본 비교")
        self.compare_check.toggled.connect(self.original_label.setVisible)
        gradcam_row.addWidget(self.compare_check)
        self.image_label.setToolTip("휠: 확대 / 드래그: 이동 / 더블 클릭: 화면 맞춤 / 마우스 위치: 원본 픽셀\n" + self.image_label_hint)

        # [페이지 1] 격자 뷰 — QScrollArea + QGridLayout
        self._grid_scroll = QScrollArea()
        self._grid_scroll.setWidgetResizable(True)
        self._grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._grid_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #0a0c10;
                border: 2px dashed #2a2f3a;
                border-radius: 8px;
            }
        """)
        self._grid_container = QWidget()
        self._grid_container.setMinimumWidth(620)
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(8, 8, 8, 8)
        self._grid_layout.setSpacing(6)
        self._grid_scroll.setWidget(self._grid_container)
        self._view_stack.addWidget(self._grid_scroll)

        # 기본: 단일 뷰 (페이지 0)
        self._view_stack.setCurrentIndex(0)

        image_layout.addWidget(self._view_stack, stretch=1)

        self.image_path_label = QLabel("")
        self.image_path_label.setStyleSheet("color: #8A92A4; font-size: 11px;")
        self.image_path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        image_layout.addWidget(self.image_path_label)

        # Grad-CAM 상태 라벨
        self.gradcam_info = QLabel("")
        self.gradcam_info.setWordWrap(True)
        self.gradcam_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.gradcam_info.setStyleSheet("color: #E5A832; font-size: 11px;")
        image_layout.addWidget(self.gradcam_info)

        # 태스크에 따라 필요한 결과만 표시한다. 격자에는 이 패널을 표시하지 않는다.
        self.task_details_group = QGroupBox("태스크 결과")
        self.task_details_group.setObjectName("task_details")
        task_details_layout = QVBoxLayout(self.task_details_group)
        task_details_layout.setContentsMargins(6, 6, 6, 6)
        self.task_details_scroll = QScrollArea()
        self.task_details_scroll.setWidgetResizable(True)
        self.task_details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.task_details_scroll.setMaximumHeight(150)
        self.task_details_label = QLabel()
        self.task_details_label.setWordWrap(True)
        self.task_details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.task_details_scroll.setWidget(self.task_details_label)
        task_details_layout.addWidget(self.task_details_scroll)
        self.task_details_group.hide()
        image_layout.addWidget(self.task_details_group)

        splitter.addWidget(image_widget)

        # 오른쪽: 결과 패널
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)

        result_layout.setContentsMargins(0, 0, 0, 0)
        # Keep the presentation adapter for existing engines, outside the visible layout.
        self.result_card = ResultCard(self)
        self.result_card.hide()

        # 배치 결과 리스트
        batch_group = QGroupBox("배치 결과")
        batch_group.setObjectName("inference_batch")
        batch_group.setStyleSheet("""
            QGroupBox#inference_batch { margin-top: 14px; padding: 0; border-radius: 6px; }
            QGroupBox#inference_batch::title { left: 8px; padding: 0; font-size: 11px; }
        """)
        batch_layout = QVBoxLayout(batch_group)
        batch_layout.setContentsMargins(6, 6, 6, 6)
        batch_layout.setSpacing(5)

        # ── 배치 추론 시간 요약 라벨 ──
        self.batch_time_label = QLabel("")
        self.batch_time_label.setStyleSheet(
            "color: #E5A832; font-size: 11px; "
            "background-color: #1a1c24; border-radius: 4px; "
            "padding: 6px 8px;"
        )
        self.batch_time_label.setWordWrap(True)
        self.batch_time_label.setMaximumHeight(44)
        self.batch_time_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.batch_time_label.hide()
        batch_layout.addWidget(self.batch_time_label)

        self.selected_timing_label = QLabel()
        self.selected_timing_label.setWordWrap(True)
        self.selected_timing_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.selected_timing_label.setStyleSheet("color: #B7C9E2; font-size: 11px; padding: 6px 8px;")
        self.selected_timing_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.selected_timing_label.hide()
        batch_layout.addWidget(self.selected_timing_label)

        self.result_review = InferenceReview(self)
        self.result_review.image_selected.connect(self._on_review_image_selected)
        self.result_review.threshold_changed.connect(self._on_review_threshold_changed)
        batch_layout.addWidget(self.result_review)
        result_layout.addWidget(batch_group, 1)

        splitter.addWidget(result_widget)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([430, 650])

        layout.addWidget(splitter, stretch=1)

        # 현재 이미지 경로
        self._current_image = None
        self._batch_images = []

    def _on_device_changed(self, index):
        """기존 모델을 직접 변형하지 않고 새 장치의 후보를 검증한 뒤 교체한다."""
        old_device = self._infer_device
        if self._process_model and self._active_checkpoint:
            self._start_model_inspection(self._active_checkpoint)
            return
        if self._active_checkpoint:
            if not self._load_model(self._active_checkpoint):
                old_index = self.infer_device_combo.findData(str(old_device))
                self.infer_device_combo.blockSignals(True)
                self.infer_device_combo.setCurrentIndex(old_index)
                self.infer_device_combo.blockSignals(False)
        else:
            self._infer_device = self._device_manager.get_device(self.infer_device_combo.currentData())
        self._update_device_info()

    def _update_device_info(self):
        """디바이스 정보 라벨 업데이트"""
        device_label = self._device_manager.get_device_label(self._infer_device)
        self.device_info.setText(device_label)

    def set_project(self, project: ProjectData):
        if getattr(self, "_inference_worker", None) is not None:
            raise RuntimeError("추론 작업 완료 또는 중단 후 변경 가능")
        """프로젝트 설정"""
        if self.project is not project:
            self._batch_images.clear()
            self._current_image = None
            self._clear_model()
            self.image_label.setText("이미지를 불러와 주세요")
            self.image_path_label.clear()
            self.ckpt_edit.clear()
        self.project = project

        # 최근 학습 결과에서 자동으로 체크포인트 탐색
        if project.runs:
            latest = project.runs[-1]
            if latest.checkpoint_path and os.path.isfile(latest.checkpoint_path):
                self.ckpt_edit.setText(latest.checkpoint_path)

    def _browse_checkpoint(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "체크포인트 선택", "",
            "PyTorch 체크포인트 (*.pt *.pth)"
        )
        if filepath:
            self.ckpt_edit.setText(filepath)

    def _clear_results(self):
        if getattr(self, "_inference_worker", None) is not None:
            raise RuntimeError("추론 작업 완료 또는 중단 후 변경 가능")
        self._clear_heatmap_preview(restore_original=True)
        self._preview_cache.clear()
        self._inference_results.clear()
        self._current_preview_rgb = None
        self._current_result = None
        if self._current_image and os.path.isfile(self._current_image):
            self._show_image(self._current_image)
        self.image_label.set_result()
        self._batch_results.clear()
        self._batch_times.clear()
        self.batch_time_label.hide()
        self.selected_timing_label.clear()
        self.selected_timing_label.hide()
        self.result_card.title_label.setText("추론 결과")
        self.result_card.content_label.setText("이미지를 선택한 뒤 추론을 실행해 주세요.")
        self.result_card.prob_widget.hide()
        self.result_card.score_label.hide()
        self.result_card.time_label.hide()
        self.gradcam_info.clear()
        self.task_details_group.hide()
        self.task_details_label.clear()
        self.result_review.set_context(self._batch_images or ([self._current_image] if self._current_image else []), self.project)
        self._refresh_grid()

    def _clear_model(self):
        if getattr(self, "_inference_worker", None) is not None:
            raise RuntimeError("추론 작업 완료 또는 중단 후 변경 가능")
        if self._gradcam is not None:
            self._gradcam.release()
        self.model = self._patchcore_model = self._gradcam = None
        self._onnx_runtime = None
        self._active_checkpoint = None
        self._process_model = False
        self.class_names = []
        self._anomaly_threshold = None
        self._center_crop = None
        self.result_review.set_saved_threshold(None)
        self.infer_btn.setEnabled(False)
        self.model_info.setText("모델을 먼저 로드해 주세요.")
        self._clear_results()
        self._region_changed()

    def _browse_crop_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "크롭 JSON 선택", self.crop_json_edit.text(),
                                             "JSON 및 프로젝트 (*.json *.dvproj);;모든 파일 (*)")
        if path:
            self.crop_json_edit.setText(path)

    def _input_region_request(self):
        from core.inference_region import read_input_region
        return read_input_region(self.crop_mode_combo.currentData(), self.crop_json_edit.text().strip())

    def _region_changed(self):
        from core.inference_region import resolve_input_region, input_region_label
        use_json = self.crop_mode_combo.currentData() == "json"
        self.crop_json_edit.setVisible(use_json)
        self.crop_json_browse.setVisible(use_json)
        try:
            region = resolve_input_region(self._input_region_request(), getattr(self, "_center_crop", None),
                                          self._input_size if self._active_checkpoint else None)
            label = input_region_label(region)
            if region["mode"] == "model" and not self._active_checkpoint:
                label = "모델을 로드하면 저장된 입력 영역을 사용합니다"
            self.crop_region_info.setStyleSheet("color: #8A92A4; font-size: 11px;")
        except ValueError as exc:
            label = str(exc)
            self.crop_region_info.setStyleSheet("color: #E05555; font-size: 11px;")
        self.crop_region_info.setText(label)
        self.crop_region_info.setToolTip(label)

    def _activate_model(self, *, path, device, model=None, patchcore=None,
                        gradcam=None, class_names=(),
                        input_size=(224, 224), normalization=None, threshold=None, info="", process=False,
                        score_normalized=False, center_crop=None):
        from center_crop import validate_center_crop
        center_crop = validate_center_crop(patchcore.center_crop if patchcore is not None else center_crop)
        if self._gradcam is not None:
            self._gradcam.release()
        self.model, self._patchcore_model = model, patchcore
        self._onnx_runtime = None
        self._gradcam = gradcam
        self._infer_device = device
        self._active_checkpoint = path
        self._process_model = process
        self.class_names = list(class_names)
        self._input_size = tuple(input_size)
        self._normalization = normalization
        self._center_crop = center_crop
        self._anomaly_threshold = threshold
        self.result_review.set_saved_threshold(threshold, normalized=score_normalized)
        self._clear_results()
        crop_info = f" | 중앙 크롭: {center_crop['width']}×{center_crop['height']} px" if center_crop else ""
        self.model_info.setText(info + crop_info)
        self.model_info.setStyleSheet("color: #34C759;")
        self.infer_btn.setEnabled(True)
        self.gradcam_checkbox.setEnabled(gradcam is not None)
        self.gradcam_checkbox.setToolTip(
            "모델 출력에 대한 민감도 히트맵. 커스텀 분할에서는 체크를 해제하면 클래스 마스크를 표시합니다." if gradcam is not None else
            "이 엔진의 Grad-CAM은 지원하지 않습니다. PatchCore는 자체 이상 맵을 표시합니다.")
        self._update_device_info()
        self._region_changed()

    def _start_model_inspection(self, checkpoint_path=None):
        """모델 검사와 초기 로드를 별도 프로세스에서 수행한다."""
        from core.desktop_jobs import DesktopJob, desktop_manager
        path = checkpoint_path if isinstance(checkpoint_path, str) else self.ckpt_edit.text().strip()
        if self._model_inspection is not None or self._inference_worker is not None:
            return
        try:
            desktop_manager().require_idle()
            if not os.path.isfile(path):
                raise ValueError("가중치 파일 선택 필요")
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "모델 확인", str(exc))
            return
        self.model_info.setText("모델 검사 중...")
        worker = DesktopJob("inspect_model", {"weights": path, "device": self.infer_device_combo.currentData()}, self)
        self._model_inspection = worker
        self.infer_btn.setEnabled(False)
        worker.failed.connect(self._model_inspection_failed)
        worker.completed.connect(self._model_inspected)
        worker.finished.connect(self._inspection_finished)
        worker.start()

    def _model_inspected(self, job):
        if job["status"] != "completed":
            return
        # Validate the complete result before replacing the previously active model.
        # PatchCore has no classification label list; legacy inspection output may contain null.
        try:
            result = job["output"]
            path, task = result["weights"], result["task"]
            names = result.get("class_names") or []
            if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) for name in names):
                raise ValueError("클래스 목록 형식 오류")
            shape = tuple(int(value) for value in result["input_size"])
            if len(shape) != 2 or min(shape) < 1:
                raise ValueError("모델 입력 크기 오류")
            device = (torch.device(result["device"]) if result.get("device") else
                      self._device_manager.get_device(self.infer_device_combo.currentData()))
            device_label = self._device_manager.get_device_label(device)
            patchcore_info = result.get("patchcore")
            if patchcore_info is not None:
                count = int(patchcore_info["memory_bank_size"])
                if count < 1:
                    raise ValueError("PatchCore 메모리 뱅크가 비어 있습니다")
                info = (f"PatchCore 준비 완료 | 메모리 뱅크: {count:,}개 패치 | "
                        f"백본: {patchcore_info['backbone']} | 디바이스: {device_label}")
            else:
                labels = f" | {len(names)}개 클래스" if names else ""
                info = f"준비 완료 | {task}{labels} | 디바이스: {device_label} | 독립 프로세스 추론"
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            self._model_inspection_failed(str(exc))
            return
        self._activate_model(path=path, device=device, class_names=names,
                             input_size=shape, threshold=result.get("anomaly_threshold"), info=info, process=True,
                             center_crop=(result.get("center_crop") if patchcore_info is None else
                                          patchcore_info.get("center_crop")),
                             score_normalized=patchcore_info is not None)
        self.gradcam_checkbox.setEnabled(patchcore_info is None)
        if patchcore_info is None:
            self.gradcam_checkbox.setToolTip("추론 프로세스에서 지원하는 모델의 Grad-CAM을 계산합니다.")

    def _model_inspection_failed(self, message):
        active = f"\n기존 모델 유지: {self._active_checkpoint}" if self._active_checkpoint else ""
        self.model_info.setText("모델 로드 실패: " + message + active)
        self.model_info.setStyleSheet("color: #E05555;")

    def _inspection_finished(self):
        if self._model_inspection is not None:
            self._model_inspection.deleteLater()
            self._model_inspection = None
        self.infer_btn.setEnabled(bool(self._active_checkpoint))

    def _open_history(self):
        from core.project import ProjectManager
        from widgets.job_history import JobHistoryDialog
        if self._inference_worker is not None:
            return
        dialog = JobHistoryDialog(ProjectManager.get_active_filepath(self.project) if self.project else None, self)
        dialog.open_inference.connect(self._restore_inference_job)
        dialog.exec()

    def _restore_inference_job(self, job_id):
        from core.desktop_jobs import PersistentInferenceCache, desktop_manager, inference_result
        cache = PersistentInferenceCache(desktop_manager().directory(job_id))
        self._clear_results()
        self._preview_cache = cache
        self._batch_images = list(cache.entries)
        self.result_review.set_saved_threshold(None)
        self.result_review.set_context(self._batch_images, self.project)
        self._inference_results = {p: inference_result(row) for p, row in cache.entries.items()}
        self._batch_results = {i: self._inference_results[p] for i, p in enumerate(self._batch_images)}
        for result in self._inference_results.values():
            self.result_review.update_result(result)
        self.gradcam_checkbox.setEnabled(True)
        self._refresh_grid()
        if self._batch_images:
            self._display_cached_result(self._batch_images[0])

    def _load_model(self, checkpoint_path=None):
        if getattr(self, "_inference_worker", None) is not None:
            raise RuntimeError("추론 작업 완료 또는 중단 후 변경 가능")
        """완전히 로드된 후보만 활성화한다. 실패한 후보의 랜덤 가중치는 사용하지 않는다."""
        path = checkpoint_path if isinstance(checkpoint_path, str) else self.ckpt_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "알림", "유효한 체크포인트 파일 경로를 지정해 주세요.")
            return False
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict):
                raise ValueError("체크포인트는 설정과 가중치를 포함한 딕셔너리여야 합니다")
            if checkpoint.get("type") == "patchcore":
                self._load_patchcore_model(checkpoint, path)
            elif isinstance(checkpoint.get("model"), torch.nn.Module) or "train_args" in checkpoint:
                raise ValueError("지원하지 않는 모델 형식입니다. EfficientNet 체크포인트를 선택하세요.")
            else:
                # GUI/CLI 내보내기와 같은 체크포인트 계약을 해석한다.
                from export_onnx import resolve_checkpoint_spec, load_custom_model
                spec = resolve_checkpoint_spec(checkpoint)
                device = self._device_manager.get_device(self.infer_device_combo.currentData())
                threshold = checkpoint.get("anomaly_threshold")
                if threshold is not None:
                    threshold = float(threshold)
                    if not np.isfinite(threshold):
                        raise ValueError("이상 판정 임계값이 유한하지 않습니다")
                if checkpoint.get("threshold_comparator", ">=") != ">=":
                    raise ValueError("미지원 임계값 비교 연산")
                if spec["task"] == "anomaly" and checkpoint.get(
                        "score_definition", "reconstruction_mse_mean") != "reconstruction_mse_mean":
                    raise ValueError("이상 점수 정의 불일치")
                candidate = load_custom_model(checkpoint, spec).to(device).eval()
                try:
                    cam = GradCAM(candidate)
                except (ValueError, AttributeError):
                    cam = None
                preprocessing = spec["preprocessing"]
                normalization = (preprocessing.get("normalize_mean", preprocessing.get("mean")),
                                 preprocessing.get("normalize_std", preprocessing.get("std")))
                if normalization[0] is None or normalization[1] is None:
                    normalization = None
                channel_info = f"입력 {spec['in_channels']}ch"
                if checkpoint.get("engine") == "efficientnet":
                    from efficientnet_contract import channel_description
                    channel_info = channel_description(candidate.checkpoint_config())
                self._activate_model(
                    path=path, device=device, model=candidate, gradcam=cam,
                    class_names=spec["class_names"],
                    input_size=(spec["input_height"], spec["input_width"]),
                    normalization=normalization, threshold=threshold,
                    center_crop=spec["preprocessing"].get("center_crop"),
                    info=f"{checkpoint.get('architecture_name', 'Custom CSP')} 준비 완료 | 태스크: {spec['task']} | 입력: {spec['input_width']}x{spec['input_height']} | "
                         f"{channel_info} | 디바이스: {self._device_manager.get_device_label(device)}")
            return True
        except Exception as exc:
            active = f"\n기존 모델 유지: {self._active_checkpoint}" if self._active_checkpoint else ""
            QMessageBox.critical(self, "모델 로드 오류", f"로드 실패: {exc}{active}")
            if not self._active_checkpoint:
                self.infer_btn.setEnabled(False)
                self.model_info.setText("로드 실패")
            return False

    def _load_patchcore_model(self, checkpoint, ckpt_path):
        device = self._device_manager.get_device(self.infer_device_combo.currentData())
        candidate = PatchCore.load(ckpt_path, device=device)
        info = candidate.get_info()
        if info["memory_bank_size"] <= 0:
            raise ValueError("PatchCore 메모리 뱅크가 비어 있습니다")
        self._activate_model(
            path=ckpt_path, device=device, patchcore=candidate,
            input_size=(candidate.input_size, candidate.input_size),
            threshold=candidate.normalized_threshold, score_normalized=True,
            info=f"PatchCore 준비 완료 | 메모리 뱅크: {info['memory_bank_size']:,}개 패치 | "
                 f"백본: {info.get('backbone', '저장 모델')} | "
                 f"디바이스: {self._device_manager.get_device_label(device)}")


    def _select_image(self):
        if getattr(self, "_inference_worker", None) is not None:
            raise RuntimeError("추론 작업 완료 또는 중단 후 변경 가능")
        """단일 이미지 선택"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "이미지 파일 선택", "",
            "이미지 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if filepath:
            self._batch_images.clear()
            self._clear_results()
            self._current_image = filepath
            self._show_image(filepath)
            self._switch_view_mode("single")

    def _select_directory(self):
        if getattr(self, "_inference_worker", None) is not None:
            raise RuntimeError("추론 작업 완료 또는 중단 후 변경 가능")
        """배치 추론 — 폴더 선택"""
        dirpath = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if dirpath:
            exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
            self._batch_images = [
                os.path.join(dirpath, f)
                for f in sorted(os.listdir(dirpath))
                if f.lower().endswith(exts)
            ]
            self._current_image = None
            self._clear_results()
            if self._batch_images:
                self._current_image = self._batch_images[0]
                self._show_image(self._batch_images[0])

            # 격자 뷰 갱신 (새 폴더 → 결과 없는 썸네일 표시)
            if self._view_mode == "grid":
                self._refresh_grid()

    def _show_image(self, filepath: str):
        """이미지 표시"""
        self._current_preview_rgb = None
        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap)
            self.original_label.setPixmap(pixmap)
            from PIL import Image
            with Image.open(filepath) as source:
                pixels = np.array(source)
            self.image_label.set_pixel_source(pixels)
            self.original_label.set_pixel_source(pixels)
        else:
            self.image_label.setText("이미지 표시 불가")
        result = self._current_result
        if result is not None and result.image_path == filepath:
            self.image_label.set_result(result.summary, result.color)
        else:
            self.image_label.set_result()
        self.image_path_label.setText(filepath)

    def _run_inference(self):
        """시작 버튼은 계산 워커를 실행하고 즉시 이벤트 루프로 복귀한다."""
        if self._inference_worker is not None:
            return
        from core.desktop_jobs import desktop_manager
        try:
            desktop_manager().require_idle()
        except RuntimeError as exc:
            QMessageBox.warning(self, "작업 진행 중", str(exc))
            return
        if not self._process_model and not any(model is not None for model in (self.model, self._patchcore_model)):
            QMessageBox.warning(self, "알림", "모델 미로드")
            return
        paths = list(self._batch_images) or ([self._current_image] if self._current_image else [])
        if not paths:
            QMessageBox.warning(self, "알림", "추론할 이미지 선택 필요")
            return
        try:
            region = self._input_region_request()
        except ValueError as exc:
            QMessageBox.warning(self, "추론 입력 영역", str(exc))
            return
        self._clear_results()
        if self._process_model:
            from core.desktop_jobs import DesktopInferenceJob, PersistentInferenceCache
            from webapp.storage import project_view
            self._preview_cache = PersistentInferenceCache()
            worker = DesktopInferenceJob({"weights": self._active_checkpoint, "images": paths,
                "input_region": region,
                "device": str(self.infer_device_combo.currentData()), "gradcam": self.gradcam_checkbox.isChecked(),
                "project": project_view(self.project) if self.project else {}}, self._preview_cache, self)
        else:
            state = {name: getattr(self, name, None) for name in InferenceEngine.STATE_FIELDS}
            from core.inference_worker import InferenceWorker
            worker = InferenceWorker(InferenceEngine(state, gradcam=self.gradcam_checkbox.isChecked(),
                                                     input_region=region), paths, self._preview_cache, self)
        self._inference_worker = worker
        # 모델 교체나 데이터 선택은 계산 종료까지 잠근다. 결과 탐색은 유지한다.
        self._busy_controls = [(control, control.isEnabled()) for control in
                               self.findChildren(QPushButton) + [self.infer_device_combo, self.ckpt_edit,
                                                                 self.crop_mode_combo, self.crop_json_edit]]
        for control, _ in self._busy_controls:
            control.setEnabled(False)
        self.cancel_infer_btn.setEnabled(True)
        self.inference_progress.setRange(0, len(paths))
        self.inference_progress.setValue(0)
        worker.result_ready.connect(self._on_async_result)
        worker.progress.connect(self._on_inference_progress)
        worker.failed.connect(self._on_inference_failed)
        worker.finished.connect(self._on_inference_finished)
        worker.start()

    def _on_async_result(self, result):
        self._inference_results[result.image_path] = result
        self.result_review.update_result(result)
        display = effective_result(result, self.result_review.threshold)
        if result.image_path in self._batch_images:
            index = self._batch_images.index(result.image_path)
            self._batch_results[index] = result
            self._batch_times[index] = result.elapsed_sec
            if index < len(self._grid_cells):
                self._grid_cells[index].set_result(display.summary, display.color)
        # 사용자가 탐색 중인 이미지를 새 결과가 덮어쓰지 않는다.
        if self._current_image is None or self._current_image == result.image_path:
            self._display_cached_result(result.image_path)

    def _on_inference_progress(self, completed, total):
        self.inference_progress.setValue(completed)
        self.batch_time_label.setText(f"추론 {completed}/{total}장")
        self.batch_time_label.setToolTip(self.batch_time_label.text())
        self.batch_time_label.show()

    def _on_inference_failed(self, message):
        self.gradcam_info.setText("추론 작업 실패: " + message)

    def _cancel_inference(self):
        if self._inference_worker is not None:
            self._inference_worker.stop()
            self.cancel_infer_btn.setEnabled(False)
            self.batch_time_label.setText("현재 이미지 처리 후 중단")

    def _on_inference_finished(self):
        worker = self._inference_worker
        if worker is None:
            return
        if hasattr(worker, "engine"):
            self._onnx_runtime = getattr(worker.engine, "_onnx_runtime", None)
            self._gradcam = worker.engine._gradcam
        for control, enabled in self._busy_controls:
            control.setEnabled(enabled)
        self._busy_controls.clear()
        self.cancel_infer_btn.setEnabled(False)
        results = list(self._inference_results.values())
        errors = sum(result.status == "error" for result in results)
        status = ("실패" if worker.error or (results and errors == len(results)) else
                  "중단" if worker.cancelled else "오류 포함 완료" if errors else "완료")
        self.batch_time_label.setText(
            f"추론 {status}: {len(results)}/{len(worker.paths)}장 | 오류 {errors}장 | 전체 {format_duration(worker.elapsed_sec)}\n"
            f"{batch_stage_summary(results, 'inference')} | {batch_stage_summary(results, 'gradcam')}")
        self.batch_time_label.setToolTip(self.batch_time_label.text())
        self.batch_time_label.show()
        self._refresh_grid()
        if self._current_image in self._inference_results:
            self._display_cached_result(self._current_image)
        self._inference_worker = None
        worker.deleteLater()

    def _run_single_inference(self, image_path: str):
        """실패도 현재 파일의 결과로 반환한다. 시간은 전처리와 표시를 포함한다."""
        from dataclasses import replace
        started = time.perf_counter()
        self._stage_timings = {}
        self.gradcam_enabled = self.gradcam_checkbox.isChecked()
        self._current_image = image_path
        self._current_result = None
        self._current_preview_rgb = None
        self._clear_heatmap_preview()
        self.gradcam_info.clear()
        self.image_label.set_result()
        try:
            if self._patchcore_model is not None:
                result = self._run_patchcore_inference(image_path)
            elif self.model is not None:
                result = self._run_custom_inference(image_path)
            else:
                raise ValueError("모델 미로드")
            if not isinstance(result, InferenceResult):
                raise RuntimeError("추론 결과 형식 오류")
        except Exception as exc:
            self._clear_heatmap_preview()
            result = InferenceResult(image_path, "error", "", "ERROR", "#E05555", error=str(exc))
            self.result_card.show_error(str(exc))
            self.gradcam_info.clear()
            self._show_image(image_path)
        measurements = {}
        for stage in ("inference", "gradcam"):
            timer = self._stage_timings.get(stage)
            if timer is not None:
                measurements[f"{stage}_sec"] = timer.elapsed_sec
                measurements[f"{stage}_status"] = timer.status
        if "gradcam_status" not in measurements and getattr(self, "_gradcam", None) is None:
            measurements["gradcam_status"] = "unsupported"
        result = replace(result, **measurements)
        self._cache_inference_result(result)
        self._render_result_card(result)
        result = replace(result, elapsed_sec=time.perf_counter() - started)
        self._inference_results[result.image_path] = result
        self._present_result(result, render_card=False)
        return result

    def _measure_inference_stage(self, stage):
        timer = StageTimer(self._infer_device)
        self._stage_timings[stage] = timer
        return timer

    def _present_result(self, result: InferenceResult, *, render_card=True):
        """모든 추론 엔진의 결과를 이미지, 배치 목록, 격자에 함께 반영한다."""
        self.result_review.update_result(result)
        raw_result = result
        result = effective_result(result, self.result_review.threshold)
        self._current_result = result
        if render_card or self.result_review.threshold is not None:
            self._render_result_card(result)
        self._last_infer_time = result.elapsed_sec
        self.result_card.set_inference_times(result)
        self.selected_timing_label.setText(
            f"선택 이미지: {os.path.basename(result.image_path)}\n{format_result_timing(result)}\n"
            f"{format_runtime_stages(result).rstrip()}")
        self.selected_timing_label.setToolTip(
            "파일 읽기/디코딩 + 전처리 + 모델 계산 + 수치 후처리의 실제 합계입니다.\n"
            "모델 준비와 화면 표시 시간은 제외하며 Grad-CAM은 별도로 측정합니다.\n"
            "다른 이미지를 선택하면 해당 이미지의 저장된 측정값을 표시합니다."
            + ("\n" + (result.details or {})["runtime_warning"] if (result.details or {}).get("runtime_warning") else ""))
        self.selected_timing_label.show()
        self.image_label.set_result(result.summary, result.color)
        self.image_path_label.setText(
            f"{result.image_path}  |  {format_result_timing(result)}"
        )
        if result.image_path in self._batch_images:
            index = self._batch_images.index(result.image_path)
            self._batch_results[index] = raw_result
            self._batch_times[index] = result.elapsed_sec
            if index < len(self._grid_cells):
                self._grid_cells[index].set_result(result.summary, result.color)
    def _render_result_card(self, result):
        """추론 때 저장한 수치로 결과 카드를 복원한다."""
        details = result.details or {}
        self._update_task_details(result)
        if result.status == "error":
            self.result_card.show_error(result.error)
        elif result.task == "classify" and "probabilities" in details:
            self.result_card.show_classification_result(
                details["class_names"], np.asarray(details["probabilities"]))
        elif result.task == "anomaly" and result.score is not None:
            self.result_card.show_anomaly_result(result.score, result.threshold)
        elif result.task == "segment" and "pixel_counts" in details:
            self.result_card.show_segmentation_result(details["num_classes"], details["pixel_counts"])
        elif "num_detections" in details:
            self.result_card.show_detection_result(details["num_detections"])
            if details.get("candidate_cells"):
                self.result_card.content_label.setText(
                    f"객체 후보 셀: {details['num_detections']}개 (후처리 전)")
        else:
            self.result_card.title_label.setText("추론 결과")
            self.result_card.content_label.setText(result.summary)
            self.result_card.prob_widget.hide()
            self.result_card.score_label.hide()

    def _update_task_details(self, result):
        """선택한 태스크에 맞는 상세 결과를 이미지 패널에 표시한다."""
        if result is None or result.status == "error":
            self.task_details_group.hide()
            return
        details = result.details or {}
        lines = []
        title = "태스크 결과"
        if result.task == "classify" and "probabilities" in details:
            title = "분류 클래스별 확률"
            names = list(details.get("class_names") or [])
            probs = list(details.get("probabilities") or [])
            lines = [f"{names[i] if i < len(names) else f'클래스 {i}'}: {float(value):.1%}"
                     for i, value in enumerate(probs)]
        elif result.task in ("detect", "obb") and "detections" in details:
            title = f"{'OBB 회전 검출' if result.task == 'obb' else '검출 결과'} ({len(details['detections'])}개)"
            names = list(details.get("class_names") or [])
            for item in details["detections"]:
                class_id = item.get("class_id", "?")
                name = names[int(class_id)] if isinstance(class_id, (int, np.integer)) and int(class_id) < len(names) else class_id
                confidence = item.get("confidence", item.get("score", 0))
                geometry = (f" | {item['angle_deg']:.1f}° | {item['width_px']:.1f}×{item['height_px']:.1f} px"
                            if result.task == "obb" else "")
                lines.append(f"{name}: {float(confidence):.1%}{geometry}")
        elif result.task in ("detect", "obb") and details.get("num_detections") is not None:
            title = "검출 결과"
            lines = [f"검출 박스: {int(details['num_detections'])}개"]
        elif result.task == "segment" and "pixel_counts" in details:
            title = "세그멘테이션 픽셀 분포"
            total = sum(int(value) for value in details["pixel_counts"].values())
            lines = [f"클래스 {key}: {int(value):,} px ({int(value) / total:.1%})" if total else f"클래스 {key}: 0 px"
                     for key, value in sorted(details["pixel_counts"].items(), key=lambda item: int(item[0]))]
        elif result.task == "segment" and details.get("num_detections") is not None:
            title = "세그멘테이션 결과"
            lines = [f"분할 객체: {int(details['num_detections'])}개"]
        elif result.task == "anomaly" and result.score is not None:
            title = "이상 탐지 결과"
            threshold = "미보정" if result.threshold is None else f"{result.threshold:.6g}"
            decision = "미보정" if result.threshold is None else (
                "NG" if result.score >= result.threshold else "OK")
            lines = [f"스코어: {result.score:.6f}", f"임계값: {threshold}", f"판정: {decision}"]
        if not lines:
            self.task_details_group.hide()
            return
        self.task_details_group.setTitle(title)
        self.task_details_label.setText("\n".join(lines))
        self.task_details_group.show()

    def _cache_inference_result(self, result):
        """결과 수치는 메모리, 전체 이미지와 맵은 임시 디스크에 저장한다."""
        from PIL import Image
        self._inference_results[result.image_path] = result
        preview = self._current_preview_rgb
        if preview is None:
            try:
                with Image.open(result.image_path) as image:
                    preview = np.array(image.convert("RGB"))
            except OSError:
                preview = None
        try:
            self._preview_cache.put(result.image_path, preview, self._heatmap_cache,
                                    self.gradcam_info.text())
        except (OSError, ValueError) as exc:
            self.gradcam_info.setText(f"결과 이미지 캐시 저장 실패: {exc}. 추론 시작으로 다시 저장해 주세요.")

    def _display_cached_result(self, image_path):
        """목록, 키보드, 격자 선택은 저장된 결과만 표시하고 추론하지 않는다."""
        self._clear_heatmap_preview()
        self._current_image = image_path
        self._current_result = None
        self._current_preview_rgb = None
        result = self._inference_results.get(image_path)
        if result is None:
            self._show_image(image_path)
            self.result_card.title_label.setText("추론 대기")
            self.result_card.content_label.setText("추론 시작 버튼을 눌러 결과를 생성해 주세요.")
            self.result_card.prob_widget.hide()
            self.result_card.score_label.hide()
            self.result_card.time_label.hide()
            self.gradcam_info.clear()
            self.task_details_group.hide()
            return
        try:
            entry = self._preview_cache.get(image_path)
            if entry is None:
                raise OSError("저장된 결과 이미지 없음")
            self.gradcam_info.setText(entry["info"])
            if entry["heatmap"] is not None:
                self._heatmap_cache = entry["heatmap"]
                self._render_cached_heatmap()
            elif entry["preview"] is not None:
                self._display_numpy_image(entry["preview"])
            else:
                self._show_image(image_path)
        except (OSError, ValueError, KeyError) as exc:
            self._clear_heatmap_preview()
            self._show_image(image_path)
            self.gradcam_info.setText(f"결과 이미지 캐시 불러오기 실패: {exc}. 추론 시작으로 다시 저장해 주세요.")
        self._present_result(result)





    def _set_info(self, text):
        self.gradcam_info.setText(text)

    def _clear_heatmap_preview(self, restore_original=False):
        """이전 이미지나 모델의 반응 맵이 다시 표시되지 않도록 폐기한다."""
        self._heatmap_timer.stop()
        cache = self._heatmap_cache
        self._heatmap_cache = None
        if restore_original and cache is not None and cache["image_path"] == self._current_image:
            self._display_numpy_image(cache["original"])

    def _set_heatmap_preview(self, image_path, original, activation, kind, info, base_image=None,
                            valid_mask=None, detections=None):
        """현재 표시 중인 한 장만 메모리에 보관하고 범위를 다시 적용한다."""
        self._heatmap_cache = {
            "image_path": image_path,
            "original": np.asarray(original, dtype=np.uint8).copy(),
            "activation": np.asarray(activation, dtype=np.float32).copy(),
            "kind": kind,
            "info": info,
            "base_image": None if base_image is None else np.asarray(base_image, dtype=np.uint8).copy(),
            "valid_mask": None if valid_mask is None else np.asarray(valid_mask).copy(),
            "detections": detections,
        }
        try:
            self._render_cached_heatmap()
        except Exception:
            self._clear_heatmap_preview()
            raise

    def _on_heatmap_range_changed(self, changed):
        lower, upper = self.heatmap_lower_spin.value(), self.heatmap_upper_spin.value()
        if lower >= upper:
            if changed == "lower":
                with QSignalBlocker(self.heatmap_upper_spin):
                    self.heatmap_upper_spin.setValue(lower + 1)
            else:
                with QSignalBlocker(self.heatmap_lower_spin):
                    self.heatmap_lower_spin.setValue(upper - 1)
        self._schedule_heatmap_render()

    def _reset_heatmap_display(self):
        for control, value in ((self.heatmap_lower_spin, 0), (self.heatmap_upper_spin, 100)):
            with QSignalBlocker(control):
                control.setValue(value)
        with QSignalBlocker(self.gradcam_alpha_combo):
            self.gradcam_alpha_combo.setCurrentIndex(self.gradcam_alpha_combo.findData(0.5))
        self._schedule_heatmap_render()

    def _on_gradcam_toggled(self, checked):
        if self._heatmap_cache is not None:
            self._schedule_heatmap_render()
        elif checked and self._gradcam is not None:
            self.gradcam_info.setText("Grad-CAM을 표시하려면 추론을 실행해 주세요.")

    def _schedule_heatmap_render(self, *_):
        if self._heatmap_cache is not None:
            # 상세 이미지에서 변경을 즉시 확인할 수 있게 전환한다.
            if self._view_mode != "single":
                self._switch_view_mode("single")
            self._heatmap_timer.start()

    def _render_cached_heatmap(self):
        cache = self._heatmap_cache
        if cache is None or cache["image_path"] != self._current_image:
            return
        if cache["kind"] == "Grad-CAM" and not self.gradcam_checkbox.isChecked():
            base = cache["original"] if cache["base_image"] is None else cache["base_image"]
            self._display_numpy_image(base)
            self.gradcam_info.setText("Grad-CAM 숨김")
            return
        alpha = self.gradcam_alpha_combo.currentData()
        lower, upper = self.heatmap_lower_spin.value() / 100, self.heatmap_upper_spin.value() / 100
        mask = cache.get("valid_mask")
        _, overlay = render_heatmap(cache["activation"], cache["original"], alpha, lower, upper,
                                    valid_mask=mask)
        if cache.get("detections") is not None:
            from core.spatial_preview import draw_detection_boxes
            overlay = draw_detection_boxes(overlay, cache["detections"])
        self._display_numpy_image(overlay)
        values = cache["activation"]
        if mask is not None:
            values = values[mask]
        state = ""
        if not values.size:
            state = " | 모델 입력에 포함된 원본 영역 없음"
        elif not np.any(values > 0):
            state = " | 양의 반응 없음: 0%부터 표시하면 파란색"
        elif np.ptp(values) == 0:
            state = " | 반응이 균일하여 위치 구분 불가"
        elif not np.any(values >= lower):
            state = " | 현재 범위에 표시할 반응 없음"
        if mask is not None and not mask.all():
            state += f" | 모델 입력 영역 {mask.mean():.1%}, 중앙 잘림으로 제외된 바깥은 원본 표시"
        self.gradcam_info.setText(
            f"{cache['kind']} | {cache['info']}{state} | 표시 시작 {lower:.0%}"
            f" | 최대 강조 {upper:.0%} | 색상 강도 {alpha:.0%}"
        )

    def _display_numpy_image(self, rgb_array: np.ndarray):
        """
        numpy RGB 배열 → QPixmap 변환 후 이미지 라벨에 표시

        변환 흐름:
        numpy(H,W,3) RGB → QImage(RGB888) → QPixmap → QLabel
        """
        # QImage는 연속 메모리만 올바르게 읽음
        # — 슬라이싱/전치된 배열은 stride가 불규칙하여 깨짐 방지
        from core.image_display import display_rgb
        rgb_array = display_rgb(rgb_array)
        self._current_preview_rgb = rgb_array
        h, w = rgb_array.shape[:2]
        bytes_per_line = rgb_array.strides[0]

        # numpy → QImage (RGB888 포맷)
        qimage = QImage(
            rgb_array.data, w, h, bytes_per_line,
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        cache = self._heatmap_cache
        if cache is not None and cache.get("original") is not None:
            original = np.ascontiguousarray(cache["original"])
            h, w = original.shape[:2]
            qimage = QImage(original.data, w, h, original.strides[0], QImage.Format.Format_RGB888).copy()
            self.original_label.setPixmap(QPixmap.fromImage(qimage))
            self.image_label.set_pixel_source(original)
            self.original_label.set_pixel_source(original)

    def _run_batch_inference(self):
        """
        배치 추론 — 전체 이미지 순회 + 시간 측정

        흐름:
        ┌────────────────────────────────────────────────────────┐
        │ total_t0 = time.perf_counter()                        │
        │ for 이미지 in 폴더:                                    │
        │   ① t0 = perf_counter()                               │
        │   ② _run_single_inference() → 결과 카드 업데이트       │
        │   ③ elapsed = perf_counter() - t0                     │
        │   ④ 결과 텍스트 + 시간 → batch_list 항목에 표시        │
        │ total = perf_counter() - total_t0                     │
        │ → 전체 요약: "50장 | 전체 1.23초 | 평균 24.6ms | FPS" │
        │ → 격자 뷰 갱신 (_refresh_grid)                        │
        └────────────────────────────────────────────────────────┘
        """
        self.result_review.set_context(self._batch_images, self.project)
        self._batch_results.clear()
        self._batch_times.clear()

        total_t0 = time.perf_counter()

        for i, img_path in enumerate(self._batch_images):
            # ── 이미지별 추론 시간 측정 ──
            result = self._run_single_inference(img_path)
            elapsed = result.elapsed_sec
            self._batch_times[i] = elapsed

            self._batch_results[i] = result
        total_elapsed = time.perf_counter() - total_t0
        num_images = len(self._batch_images)

        # ── 배치 요약 라벨 표시 ──────────────────────
        # 전체 시간, 평균 시간, FPS, 최소/최대
        if num_images > 0 and self._batch_times:
            times_list = list(self._batch_times.values())
            avg_time = sum(times_list) / len(times_list)
            min_time = min(times_list)
            max_time = max(times_list)
            fps = num_images / total_elapsed if total_elapsed > 0 else 0

            # 전체 시간 포맷
            if total_elapsed < 1.0:
                total_str = f"{total_elapsed * 1000:.0f} ms"
            else:
                total_str = f"{total_elapsed:.2f} 초"

            self.batch_time_label.setText(
                f"⏱ 배치 추론 완료  |  "
                f"총 {num_images}장 (오류 {sum(r.status == 'error' for r in self._batch_results.values())}장)  |  "
                f"전체: {total_str}  |  "
                f"전체 처리 평균: {avg_time * 1000:.1f} ms/장  |  "
                f"전체 처리량: {fps:.1f} 장/초  |  "
                f"전체 처리 최소/최대: {min_time * 1000:.1f}/{max_time * 1000:.1f} ms\n"
                f"{batch_stage_summary(self._batch_results.values(), 'inference')}  |  "
                f"{batch_stage_summary(self._batch_results.values(), 'gradcam')}"
            )
            self.batch_time_label.setToolTip(self.batch_time_label.text())
            self.batch_time_label.show()
        else:
            self.batch_time_label.hide()

        # 격자 뷰 갱신 (배치 완료 후)
        self._refresh_grid()

    def _extract_current_result(self):
        result = self._current_result
        return (result.summary, result.color) if result is not None else ("미실행", "#8A92A4")

    def _on_review_image_selected(self, image_path):
        """Proxy row positions never determine the underlying image."""
        self._switch_view_mode("single")
        self._display_cached_result(image_path)
        if image_path in self._batch_images:
            self._select_grid_cell(self._batch_images.index(image_path))

    def _on_review_threshold_changed(self):
        # Reuse scores and previews; never run inference or overwrite saved results.
        for index, raw in self._batch_results.items():
            result = effective_result(raw, self.result_review.threshold)
            if index < len(self._grid_cells):
                self._grid_cells[index].set_result(result.summary, result.color)
        if self._current_image in self._inference_results:
            self._present_result(self._inference_results[self._current_image])

    # ── 뷰 모드 전환 메서드 ─────────────────────────────────

    @staticmethod
    def _view_btn_style(active: bool) -> str:
        """뷰 토글 버튼 스타일 — 활성/비활성"""
        if active:
            return """
                QPushButton {
                    background-color: #5590F0;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 11px;
                    font-weight: bold;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #1e222c;
                    color: #8A92A4;
                    border: 1px solid #2a2f3a;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #252a36;
                    color: #E8EAEF;
                }
            """

    def _switch_view_mode(self, mode: str):
        """
        뷰 모드 전환 — 단일 ↔ 격자

        ┌────────────────────────────────────────────┐
        │ "single" → QStackedWidget 페이지 0 (단일)  │
        │ "grid"   → QStackedWidget 페이지 1 (격자)  │
        │                                            │
        │ 격자 전환 시 _refresh_grid() 호출로         │
        │ 현재 배치 결과를 그리드에 반영              │
        └────────────────────────────────────────────┘
        """
        self._view_mode = mode

        # 버튼 상태 업데이트
        is_single = (mode == "single")
        self.single_view_btn.setChecked(is_single)
        self.grid_view_btn.setChecked(not is_single)
        self.single_view_btn.setStyleSheet(self._view_btn_style(is_single))
        self.grid_view_btn.setStyleSheet(self._view_btn_style(not is_single))

        if mode == "single":
            self._view_stack.setCurrentIndex(0)
            # 표시 중인 원본/히트맵과 결과 배지를 그대로 유지한다.
        else:
            self._view_stack.setCurrentIndex(1)
            self._refresh_grid()

    def _refresh_grid(self):
        """
        격자 뷰 갱신 — 배치 이미지를 4열 그리드로 배치

        각 셀 구성:
        ┌───────────────┐
        │ [결과 텍스트]  │ ← QPainter 오버레이
        │   [썸네일]     │ ← QPixmap 스케일링
        │  파일명.jpg    │ ← QLabel 캡션
        └───────────────┘
        """
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._grid_cells.clear()

        if not self._batch_images:
            # 빈 안내 라벨
            empty_label = QLabel("폴더를 선택하고 추론을 실행해 주세요.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setStyleSheet("color: #8A92A4; font-size: 13px;")
            self._grid_layout.addWidget(empty_label, 0, 0, 1, 4)
            return

        # 4열 그리드로 배치
        cols = 4
        for i, img_path in enumerate(self._batch_images):
            row, col = divmod(i, cols)

            cell = GridCell(index=i, parent=self._grid_container)
            cell.name_label.setText(os.path.basename(img_path))

            # 추론 결과가 있으면 오버레이 텍스트 설정
            if i in self._batch_results:
                result = effective_result(self._batch_results[i], self.result_review.threshold)
                cell.set_result(result.summary, result.color)

            # 썸네일 이미지 로드 및 설정
            pixmap = QPixmap()
            try:
                entry = self._preview_cache.get(img_path, thumbnail=True)
                if entry is not None:
                    preview = entry["preview"]
                    cached_map = entry["heatmap"]
                    if cached_map is not None:
                        if cached_map["kind"] == "Grad-CAM" and not self.gradcam_checkbox.isChecked():
                            preview = cached_map["base_image"]
                            if preview is None:
                                preview = cached_map["original"]
                        else:
                            _, preview = render_heatmap(
                                cached_map["activation"], cached_map["original"],
                                self.gradcam_alpha_combo.currentData(),
                                self.heatmap_lower_spin.value() / 100,
                                self.heatmap_upper_spin.value() / 100,
                                valid_mask=cached_map.get("valid_mask"))
                            if cached_map.get("detections") is not None:
                                from core.spatial_preview import draw_detection_boxes
                                preview = draw_detection_boxes(preview, cached_map["detections"])
                    if preview is not None:
                        preview = np.ascontiguousarray(preview)
                        height, width, channels = preview.shape
                        image = QImage(preview.data, width, height, width * channels,
                                       QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(image)
            except (OSError, ValueError, KeyError):
                # 상세 선택 시 캐시 오류를 표시하며 추론을 다시 실행하지 않는다.
                pass
            if pixmap.isNull():
                pixmap = QPixmap(img_path)
            cell.set_image(pixmap)

            # 셀 클릭 → 단일 뷰로 전환 + 상세 표시
            cell.clicked.connect(self._on_grid_cell_clicked)
            self._grid_cells.append(cell)
            self._grid_layout.addWidget(cell, row, col)

    def _on_grid_cell_clicked(self, index: int):
        """격자에서 저장된 원본/히트맵/확률과 기존 추론 시간을 복원한다."""
        if 0 <= index < len(self._batch_images):
            self._switch_view_mode("single")
            self._display_cached_result(self._batch_images[index])
            self._select_grid_cell(index)
            with QSignalBlocker(self.result_review):
                self.result_review.select_path(self._batch_images[index])

    def _select_grid_cell(self, index: int):
        """격자 셀 선택 상태 업데이트 — 하나만 선택"""
        for i, cell in enumerate(self._grid_cells):
            cell.set_selected(i == index)
