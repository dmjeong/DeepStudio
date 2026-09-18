"""학습 차트, 평가 결과와 모델 비교 표시. 작업 시작/중단과 분리."""

import copy
import math
from numbers import Real
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QTableWidget, QTableWidgetItem, QHeaderView
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor

from core.project import ProjectData
from core.metrics import TASK_METRIC_NAMES
from core.training_progress import record_epoch
# 마우스 휠로 하이퍼파라미터가 실수로 바뀌는 것을 막는 위젯


def _finite_metric(value):
    return isinstance(value, Real) and math.isfinite(value)


def _format_metric(value):
    return f"{value:.4f}" if _finite_metric(value) else "N/A"


def _metric_info_for(project=None, results=None):
    """실제 엔진의 지표 의미를 유지하고 계산하지 않은 지표는 만들지 않음."""
    results = results or {}
    task = project.task if project else results.get("task", "classify")
    info = copy.deepcopy(TASK_METRIC_NAMES.get(task, {}))
    if task == "anomaly" and project and project.training.anomaly_method == "reconstruction":
        info["primary"] = "recon_loss"
        info.setdefault("labels", {})["recon_loss"] = "Reconstruction Loss"
    if project and getattr(project.training, "selection_metric", "engine_default") != "engine_default":
        from core.model_selection import selection_policy, LABELS
        from core.training_modes import training_engine_name
        engine = training_engine_name(project.training.training_mode)
        policy = selection_policy(project.training, engine, task)
        info["primary"] = policy.metric
        info.setdefault("labels", {})[policy.metric] = LABELS[policy.metric]
        if policy.metric not in info.setdefault("display", []):
            info["display"].append(policy.metric)
    return info

# ── matplotlib 임포트 (차트용) ──
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use("QtAgg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  다크 테마 차트 색상 팔레트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHART_COLORS = {
    # Premium Dashboard 차트 색상 팔레트
    # ─ 딥 네이비 + 선명한 액센트 ─
    "bg": "#151920",            # 차트 배경 (bg-surface)
    "plot_bg": "#0f1117",       # 플롯 영역 배경 (bg-base)
    "text": "#E8EAEF",          # 텍스트
    "subtext": "#8A92A4",       # 보조 텍스트
    "grid": "#1c2028",          # 격자선 (bg-raised)
    "train_loss": "#5590F0",    # Train loss 라인 (primary blue)
    "val_loss": "#E5A832",      # Val loss 라인 (warning gold)
    "metrics": [                # 메트릭 라인 색상 (최대 6개)
        "#34C759",  # 초록 (primary — accuracy, mIoU, mAP, AUROC)
        "#E05555",  # 빨강 (secondary)
        "#9B7DFF",  # 보라 (tertiary)
        "#E5A832",  # 금색
        "#5590F0",  # 파랑
        "#38BFB0",  # 시안 (teal)
    ],
    "lr": "#E5A832",            # Learning Rate 라인 (금색/warning)
    "cm_cmap": "Blues",         # Confusion Matrix 색상 맵
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  차트 위젯들
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _BaseChart(QFrame):
    """차트 베이스 클래스 — 공통 다크 테마 스타일링"""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(6, 3), facecolor=CHART_COLORS["bg"])
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.ax = self.figure.add_subplot(111)
            self._style_axes(title)
            layout.addWidget(self.canvas)
        else:
            label = QLabel(f"{title}\n(matplotlib 설치 필요)")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {CHART_COLORS['subtext']}; padding: 20px;")
            layout.addWidget(label)

    def _style_axes(self, title: str = ""):
        """다크 테마 축 스타일링"""
        self.ax.set_facecolor(CHART_COLORS["plot_bg"])
        self.ax.tick_params(colors=CHART_COLORS["subtext"], labelsize=9)
        self.ax.spines['bottom'].set_color(CHART_COLORS["grid"])
        self.ax.spines['left'].set_color(CHART_COLORS["grid"])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        if title:
            self.ax.set_title(title, color=CHART_COLORS["text"],
                             fontsize=12, fontweight='bold')

    def _draw(self):
        """안전한 캔버스 갱신"""
        if HAS_MATPLOTLIB:
            self.figure.tight_layout()
            self.canvas.draw_idle()


class LossChart(_BaseChart):
    """
     실시간 Loss 차트

    Train/Val Loss 곡선을 에폭 단위로 실시간 업데이트
    """

    def __init__(self, parent=None):
        super().__init__("Training Loss", parent)
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def update_chart(self, epoch: int, train_loss: float, val_loss: float, *, redraw=True):
        """에폭 데이터 추가 및 차트 갱신"""
        record_epoch(self.epochs, {"train": self.train_losses, "val": self.val_losses},
                     epoch, {"train": train_loss, "val": val_loss})

        if not HAS_MATPLOTLIB or not redraw:
            return

        self.ax.clear()
        self._style_axes("Training Loss")

        epochs = self.epochs
        self.ax.plot(epochs, self.train_losses, '-o',
                    color=CHART_COLORS["train_loss"], markersize=3,
                    linewidth=1.5, label='Train Loss')
        self.ax.plot(epochs, self.val_losses, '-s',
                    color=CHART_COLORS["val_loss"], markersize=3,
                    linewidth=1.5, label='Val Loss')

        self.ax.set_xlabel('Epoch', color=CHART_COLORS["subtext"], fontsize=10)
        self.ax.set_ylabel('Loss', color=CHART_COLORS["subtext"], fontsize=10)
        self.ax.legend(loc='upper right', fontsize=9,
                      facecolor=CHART_COLORS["bg"],
                      edgecolor=CHART_COLORS["grid"],
                      labelcolor=CHART_COLORS["text"])
        self.ax.grid(True, alpha=0.2, color=CHART_COLORS["grid"])
        self._draw()

    def clear(self):
        self.train_losses.clear()
        self.val_losses.clear()
        self.epochs.clear()
        if HAS_MATPLOTLIB:
            self.ax.clear()
            self._style_axes("Training Loss")
            self._draw()


class MetricChart(_BaseChart):
    """
     실시간 Metrics 차트

    태스크별 주요 메트릭 곡선:
    - Classification: Accuracy, F1, Precision, Recall
    - Segmentation: mIoU, Dice, Pixel Accuracy
    - Detection: mAP@0.5, mAP@0.5:0.95, P, R
    - Anomaly: AUROC, F1, Precision, Recall
    """

    def __init__(self, parent=None):
        super().__init__("Metrics", parent)
        self.metrics_history = {}  # {metric_name: [values]}
        self.metric_labels = {}   # {metric_name: display_label}
        self.epochs = []

    def set_task(self, task: str):
        """태스크에 맞는 메트릭 레이블 설정"""
        info = TASK_METRIC_NAMES.get(task, {})
        self.metric_labels = info.get("labels", {})

    def update_metrics(self, epoch: int, metrics: dict, *, redraw=True):
        """메트릭 업데이트"""
        observed = {}
        for key, value in metrics.items():
            # 차트에 표시할 메트릭만 추적 (loss, confusion_matrix 등 제외)
            if key in ("val_loss", "recon_loss", "confusion_matrix", "per_class", "roc_curve") or not _finite_metric(value):
                continue
            observed[key] = value
        record_epoch(self.epochs, self.metrics_history, epoch, observed)

        if not HAS_MATPLOTLIB or not redraw:
            return

        self.ax.clear()
        self._style_axes("Metrics")

        colors = CHART_COLORS["metrics"]
        markers = ['o', 's', '^', 'D', 'v', 'p']

        for idx, (name, values) in enumerate(self.metrics_history.items()):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            epochs = self.epochs
            label = self.metric_labels.get(name, name)
            self.ax.plot(epochs, values, f'-{marker}', color=color,
                        markersize=3, linewidth=1.5, label=label)

        self.ax.set_xlabel('Epoch', color=CHART_COLORS["subtext"], fontsize=10)
        self.ax.set_ylabel('Value', color=CHART_COLORS["subtext"], fontsize=10)
        self.ax.set_ylim(-0.05, 1.05)  # 대부분 0~1 범위
        self.ax.legend(loc='lower right', fontsize=9,
                      facecolor=CHART_COLORS["bg"],
                      edgecolor=CHART_COLORS["grid"],
                      labelcolor=CHART_COLORS["text"])
        self.ax.grid(True, alpha=0.2, color=CHART_COLORS["grid"])
        self._draw()

    def clear(self):
        self.metrics_history.clear()
        self.epochs.clear()
        if HAS_MATPLOTLIB:
            self.ax.clear()
            self._style_axes("Metrics")
            self._draw()


class LRChart(_BaseChart):
    """
     Learning Rate 스케줄 차트

    에폭별 LR 변화를 시각화 (Cosine, Step, etc.)
    """

    def __init__(self, parent=None):
        super().__init__("Learning Rate Schedule", parent)
        self.lr_values = []
        self.epochs = []

    def update_lr(self, epoch: int, lr: float, *, redraw=True):
        """LR 값 추가"""
        record_epoch(self.epochs, {"lr": self.lr_values}, epoch, {"lr": lr})

        if not HAS_MATPLOTLIB or not redraw:
            return

        self.ax.clear()
        self._style_axes("Learning Rate Schedule")

        epochs = self.epochs
        self.ax.plot(epochs, self.lr_values, '-o',
                    color=CHART_COLORS["lr"], markersize=3,
                    linewidth=2.0, label='Learning Rate')
        self.ax.fill_between(epochs, self.lr_values,
                            alpha=0.15, color=CHART_COLORS["lr"])

        self.ax.set_xlabel('Epoch', color=CHART_COLORS["subtext"], fontsize=10)
        self.ax.set_ylabel('LR', color=CHART_COLORS["subtext"], fontsize=10)
        self.ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
        self.ax.legend(loc='upper right', fontsize=9,
                      facecolor=CHART_COLORS["bg"],
                      edgecolor=CHART_COLORS["grid"],
                      labelcolor=CHART_COLORS["text"])
        self.ax.grid(True, alpha=0.2, color=CHART_COLORS["grid"])
        self._draw()

    def clear(self):
        self.lr_values.clear()
        self.epochs.clear()
        if HAS_MATPLOTLIB:
            self.ax.clear()
            self._style_axes("Learning Rate Schedule")
            self._draw()


class ConfusionMatrixChart(_BaseChart):
    """
     Confusion Matrix 시각화

    Classification / Anomaly 태스크용 혼동 행렬 히트맵
    """

    def __init__(self, parent=None):
        super().__init__("Confusion Matrix", parent)
        self._colorbar = None

    def _remove_colorbar(self):
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

    def update_matrix(self, cm: np.ndarray, class_names: list = None):
        """혼동 행렬 히트맵 표시"""
        if not HAS_MATPLOTLIB:
            return

        self._remove_colorbar()
        self.ax.clear()

        n = cm.shape[0]
        if class_names is None:
            class_names = [f"C{i}" for i in range(n)]

        # 히트맵 그리기
        im = self.ax.imshow(cm, interpolation='nearest', cmap='Blues',
                           aspect='auto')

        # 색상 바
        # 기존 색상바 제거 후 다시 그리기
        cbar = self.figure.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        self._colorbar = cbar
        cbar.ax.tick_params(colors=CHART_COLORS["subtext"], labelsize=8)

        # 틱 라벨
        self.ax.set_xticks(range(n))
        self.ax.set_yticks(range(n))

        # 클래스 이름이 길면 잘라서 표시
        short_names = [name[:10] for name in class_names[:n]]
        self.ax.set_xticklabels(short_names, rotation=45, ha='right',
                               fontsize=max(8, 12 - n // 2))
        self.ax.set_yticklabels(short_names,
                               fontsize=max(8, 12 - n // 2))

        # 셀 값 표시
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                color = "white" if val > thresh else "black"
                self.ax.text(j, i, str(int(val)),
                           ha="center", va="center",
                           color=color, fontsize=max(7, 11 - n // 2))

        self.ax.set_xlabel('Predicted', color=CHART_COLORS["subtext"], fontsize=10)
        self.ax.set_ylabel('Actual', color=CHART_COLORS["subtext"], fontsize=10)
        self.ax.set_title('Confusion Matrix', color=CHART_COLORS["text"],
                         fontsize=12, fontweight='bold')
        self.ax.tick_params(colors=CHART_COLORS["subtext"])

        self._draw()

    def clear(self):
        if HAS_MATPLOTLIB:
            self._remove_colorbar()
            self.ax.clear()
            self._style_axes("Confusion Matrix")
            self._draw()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  평가 결과 테이블 위젯
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EvalResultsWidget(QWidget):
    """
     종합 평가 결과 테이블

    구성:
    ┌─────────────────────────────────────────────┐
    │  [요약 메트릭 테이블]                         │
    │  ┌───────────┬───────────┐                  │
    │  │ Accuracy  │  0.9523   │                  │
    │  │ F1 (macro)│  0.9412   │                  │
    │  │ Precision │  0.9380   │                  │
    │  │ Recall    │  0.9445   │                  │
    │  └───────────┴───────────┘                  │
    │                                              │
    │  [클래스별 상세 테이블]                        │
    │  ┌──────────┬──────┬──────┬──────┬─────┐    │
    │  │  Class   │  P   │  R   │  F1  │  N  │    │
    │  ├──────────┼──────┼──────┼──────┼─────┤    │
    │  │  OK      │ .95  │ .97  │ .96  │ 150 │    │
    │  │  NG      │ .92  │ .88  │ .90  │  50 │    │
    │  └──────────┴──────┴──────┴──────┴─────┘    │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # ── 요약 메트릭 ──
        summary_label = QLabel("종합 평가 지표")
        summary_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        summary_label.setStyleSheet("color: #E8EAEF;")
        layout.addWidget(summary_label)

        self.summary_table = QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["지표", "값"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setMaximumHeight(200)
        self.summary_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.summary_table.setAlternatingRowColors(True)
        layout.addWidget(self.summary_table)

        # ── 클래스별 상세 ──
        detail_label = QLabel("클래스별 상세")
        detail_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        detail_label.setStyleSheet("color: #E8EAEF;")
        layout.addWidget(detail_label)

        self.detail_table = QTableWidget(0, 1)
        self.detail_table.setHorizontalHeaderLabels(["Class"])
        self.detail_table.horizontalHeader().setStretchLastSection(True)
        self.detail_table.verticalHeader().setVisible(False)
        self.detail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.detail_table.setAlternatingRowColors(True)
        layout.addWidget(self.detail_table, stretch=1)

        # ── 안내 메시지 ──
        self.placeholder = QLabel(
            "학습 완료 후 종합 평가 결과가 여기에 표시됩니다.\n\n"
            "각 태스크별 유명 지표:\n"
            "• Classification: Accuracy, Precision, Recall, F1\n"
            "• Segmentation: mIoU, Dice Score, Pixel Accuracy\n"
            "• Detection: mAP@0.5, mAP@0.5:0.95\n"
            "• Anomaly: AUROC, F1, Precision, Recall"
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setStyleSheet("color: #555D70; padding: 30px;")
        layout.addWidget(self.placeholder)

        self.summary_table.hide()
        self.detail_table.hide()

    def update_results(self, results: dict):
        """평가 결과를 테이블에 표시"""
        task = results.get("task", "classify")
        metric_info = _metric_info_for(results=results)

        # ── 요약 테이블 ──
        display_keys = metric_info.get("display", [])
        labels = metric_info.get("labels", {})

        self.summary_table.setRowCount(len(display_keys))
        for row, key in enumerate(display_keys):
            label = labels.get(key, key)
            value = results.get(key)

            name_item = QTableWidgetItem(label)
            name_item.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            self.summary_table.setItem(row, 0, name_item)

            val_item = QTableWidgetItem(_format_metric(value))
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # 색상: Premium Dashboard functional colors
            if isinstance(value, (int, float)) and value > 0.8:
                val_item.setForeground(QColor("#34C759"))  # success
            elif isinstance(value, (int, float)) and value > 0.5:
                val_item.setForeground(QColor("#E5A832"))  # warning
            else:
                val_item.setForeground(QColor("#E05555"))  # error
            val_item.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
            self.summary_table.setItem(row, 1, val_item)

        # ── 클래스별 상세 테이블 ──
        per_class = results.get("per_class", [])
        if per_class:
            if task == "classify":
                headers = ["Class", "Precision", "Recall", "F1", "Support"]
                self.detail_table.setColumnCount(5)
                self.detail_table.setHorizontalHeaderLabels(headers)
                self.detail_table.setRowCount(len(per_class))
                for row, pc in enumerate(per_class):
                    self.detail_table.setItem(row, 0, QTableWidgetItem(pc["name"]))
                    self.detail_table.setItem(row, 1, QTableWidgetItem(_format_metric(pc.get("precision"))))
                    self.detail_table.setItem(row, 2, QTableWidgetItem(_format_metric(pc.get("recall"))))
                    self.detail_table.setItem(row, 3, QTableWidgetItem(_format_metric(pc.get("f1"))))
                    self.detail_table.setItem(row, 4, QTableWidgetItem(str(pc['support'])))

            elif task == "segment" and "mask_mAP_50" not in results:
                headers = ["Class", "IoU", "Dice"]
                self.detail_table.setColumnCount(3)
                self.detail_table.setHorizontalHeaderLabels(headers)
                self.detail_table.setRowCount(len(per_class))
                for row, pc in enumerate(per_class):
                    self.detail_table.setItem(row, 0, QTableWidgetItem(pc["name"]))
                    self.detail_table.setItem(row, 1, QTableWidgetItem(_format_metric(pc.get("iou"))))
                    self.detail_table.setItem(row, 2, QTableWidgetItem(_format_metric(pc.get("dice"))))

            elif task in ("detect", "obb") or (task == "segment" and "mask_mAP_50" in results):
                headers = ["Class", "AP@0.5", "AP@.5:.95"]
                self.detail_table.setColumnCount(3)
                self.detail_table.setHorizontalHeaderLabels(headers)
                self.detail_table.setRowCount(len(per_class))
                for row, pc in enumerate(per_class):
                    self.detail_table.setItem(row, 0, QTableWidgetItem(pc["name"]))
                    self.detail_table.setItem(row, 1, QTableWidgetItem(_format_metric(pc.get("ap_50"))))
                    self.detail_table.setItem(row, 2, QTableWidgetItem(_format_metric(pc.get("ap_50_95"))))

            # 상세 테이블 열 너비 자동 조정
            self.detail_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )

        # 표시 전환
        self.placeholder.hide()
        self.summary_table.show()
        self.detail_table.setVisible(bool(per_class))

    def clear(self):
        self.summary_table.setRowCount(0)
        self.detail_table.setRowCount(0)
        self.summary_table.hide()
        self.detail_table.hide()
        self.placeholder.show()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  모델 비교 테이블 위젯
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ModelCompareWidget(QWidget):
    """
     모델 학습 결과 비교

    이전 학습 run들과 현재 결과를 나란히 비교:
    ┌────────┬──────────┬──────────┬──────────┬─────────┐
    │  Run   │ Acc.     │ F1       │ Best Ep. │ Epochs  │
    ├────────┼──────────┼──────────┼──────────┼─────────┤
    │ 현재·최신 │ 0.9523   │ 0.9412   │ 42       │ 100     │
    │ Run 2  │ 0.9201   │ 0.9102   │ 35       │ 80      │
    │ Run 1  │ 0.8874   │ 0.8655   │ 28       │ 50      │
    └────────┴──────────┴──────────┴──────────┴─────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title_label = QLabel("모델 학습 이력 비교")
        title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #E8EAEF;")
        layout.addWidget(title_label)

        self.compare_table = QTableWidget(0, 1)
        self.compare_table.setHorizontalHeaderLabels(["Run"])
        self.compare_table.horizontalHeader().setStretchLastSection(True)
        self.compare_table.verticalHeader().setVisible(False)
        self.compare_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.compare_table.setAlternatingRowColors(True)
        layout.addWidget(self.compare_table, stretch=1)

        # 안내
        self.placeholder = QLabel(
            "여러 학습을 진행하면 결과를 비교할 수 있습니다.\n\n"
            "비교 항목:\n"
            "• 주요 메트릭 (Accuracy, mIoU, mAP, AUROC 등)\n"
            "• Best Epoch / 총 Epoch\n"
            "• 학습 시작/완료 시간\n"
            "• 성능 향상 / 하락 표시"
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setStyleSheet("color: #555D70; padding: 30px;")
        layout.addWidget(self.placeholder)

        self.compare_table.hide()

    def update_comparison(self, project: ProjectData):
        """프로젝트의 모든 학습 기록을 비교 테이블로 표시"""
        runs = project.runs
        if not runs:
            self.clear()
            return

        # 테이블 헤더 구성
        headers = ["Run ID", "상태", "주요 지표", "Best Epoch",
                    "총 Epochs", "시작 시간", "변화"]
        self.compare_table.setColumnCount(len(headers))
        self.compare_table.setHorizontalHeaderLabels(headers)
        self.compare_table.setRowCount(len(runs))

        # 역순 (최신이 위) 표시
        sorted_runs = list(reversed(runs))

        for row, run in enumerate(sorted_runs):
            is_latest = (row == 0)

            # Run ID (최신 run은 텍스트 태그 + 컬러로 강조)
            run_text = f"{run.run_id}  ·  최신" if is_latest else run.run_id
            run_item = QTableWidgetItem(run_text)
            if is_latest:
                run_item.setForeground(QColor("#5590F0"))
                run_item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            self.compare_table.setItem(row, 0, run_item)

            # 상태
            status_text = run.status
            self.compare_table.setItem(row, 1, QTableWidgetItem(status_text))

            # 주요 메트릭
            metric_val = run.best_metric
            metric_text = "N/A" if run.best_metric_name == "unavailable" else _format_metric(metric_val)
            metric_item = QTableWidgetItem(f"{run.best_metric_name}: {metric_text}")
            metric_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            metric_item.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
            self.compare_table.setItem(row, 2, metric_item)

            # Best Epoch
            self.compare_table.setItem(
                row, 3, QTableWidgetItem(str(run.best_epoch))
            )

            # 총 Epochs
            self.compare_table.setItem(
                row, 4, QTableWidgetItem(str(run.epochs_done))
            )

            # 시작 시간
            start_short = run.started_at[:16] if run.started_at else "—"
            self.compare_table.setItem(row, 5, QTableWidgetItem(start_short))

            # 변화 (이전 run 대비)
            previous = sorted_runs[row + 1] if row + 1 < len(sorted_runs) else None
            comparable = (previous is not None and previous.best_metric_name == run.best_metric_name
                          and run.best_metric_name != "unavailable"
                          and previous.status == run.status == "completed"
                          and _finite_metric(metric_val) and _finite_metric(previous.best_metric))
            if comparable:
                delta = metric_val - previous.best_metric
                improvement = -delta if run.best_metric_name in ("recon_loss", "val_loss") else delta
                if improvement > 0.001:
                    change_text = f"개선 {delta:+.4f}"
                    change_item = QTableWidgetItem(change_text)
                    change_item.setForeground(QColor("#34C759"))
                elif improvement < -0.001:
                    change_text = f"하락 {delta:+.4f}"
                    change_item = QTableWidgetItem(change_text)
                    change_item.setForeground(QColor("#E05555"))
                else:
                    change_item = QTableWidgetItem("━ 동일")
                    change_item.setForeground(QColor("#555D70"))
            else:
                change_item = QTableWidgetItem("첫 학습" if previous is None else "비교 불가")
                change_item.setForeground(QColor("#555D70"))

            change_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.compare_table.setItem(row, 6, change_item)

        # 열 너비 자동 조정
        self.compare_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.placeholder.hide()
        self.compare_table.show()

    def clear(self):
        self.compare_table.setRowCount(0)
        self.compare_table.hide()
        self.placeholder.show()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인: 학습 위젯
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
