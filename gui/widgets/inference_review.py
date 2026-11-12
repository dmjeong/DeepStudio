"""Sortable inference records with stable image identity and review-only thresholds."""

from PySide6.QtCore import Qt, Signal, QSortFilterProxyModel, QSignalBlocker, QSettings, QTimer
from PySide6.QtGui import QStandardItem, QStandardItemModel, QColor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QCheckBox, QPushButton, QDoubleSpinBox, QSpinBox, QSlider, QTableView, QAbstractItemView, QHeaderView, QSizePolicy,
    QGroupBox, QTableWidget, QTableWidgetItem)
from core.inference_review import (review_record, review_filter_options, source_class, finite,
                                   normalized_patchcore, result_record)
from core.segment_defects import batch_truth_summary, judge, load_model_rules, rules_of, save_model_rules
from core.inference_timing import format_result_timing, format_runtime_stages


class ReviewFilter(QSortFilterProxyModel):
    def __init__(self, parent):
        super().__init__(parent)
        self.class_name = self.decision = self.search = ""
        self.selected_only = False

    def filterAcceptsRow(self, row, parent):
        model = self.sourceModel()
        def value(col):
            return model.index(row, col, parent).data() or ""
        region_classes = model.index(row, 2, parent).data(Qt.ItemDataRole.UserRole)
        matches_class = (self.class_name in region_classes if region_classes is not None
                         else value(2) == self.class_name)
        return ((not self.class_name or matches_class) and
                (not self.decision or value(4) == self.decision) and
                (not self.search or self.search in str(value(3)).casefold()) and
                (not self.selected_only or model.item(row, 0).checkState() == Qt.CheckState.Checked))


class InferenceReview(QWidget):
    image_selected = Signal(str)
    threshold_changed = Signal()
    HEADERS = ["선택", "인덱스", "클래스", "파일명", "판정", "스코어", "추론 시간 (ms)",
               "mIoU (%)", "Dice (%)", "정답 비교"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.paths, self.raw, self.rows = [], {}, {}
        self.project = None
        self.class_names = []
        self.saved_threshold = None
        self.score_normalized = False
        self._updating = False
        self._bulk = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.setStyleSheet("""
            QPushButton, QComboBox, QLineEdit, QDoubleSpinBox { padding: 4px 6px; min-height: 20px; font-size: 12px; }
            QCheckBox { font-size: 12px; }
            QHeaderView::section { padding: 6px 4px; font-size: 11px; }
            QTableView::item { padding: 3px 5px; }
        """)
        self.threshold_panel = QWidget()
        box = QVBoxLayout(self.threshold_panel)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(3)
        line = QHBoxLayout()
        self.override = QCheckBox("사용자 임계값")
        self.value = QDoubleSpinBox()
        self.value.setAccessibleName("Anomaly 임계값")
        self.value.setDecimals(6)
        self.value.setRange(-1e12, 1e12)
        self.value.setSingleStep(.01)
        self.value.setKeyboardTracking(False)
        # QDoubleSpinBox sizes itself for the longest allowed number, not the
        # current threshold. That forces the whole page wider on Windows.
        self.value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.value.setMinimumWidth(115)
        self.reset = QPushButton("저장된 임계값 복원")
        line.addWidget(self.override)
        line.addWidget(self.value, 1)
        line.addWidget(self.reset)
        box.addLayout(line)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setMaximumHeight(16)
        self.slider.setAccessibleName("Anomaly 임계값 슬라이더")
        slider_row = QHBoxLayout()
        slider_row.addWidget(self.slider, 1)
        self.threshold_note = QLabel("스코어 ≥ 임계값이면 NG. 모델 파일은 변경하지 않습니다.")
        self.threshold_note.setWordWrap(False)
        self.threshold_note.setStyleSheet("font-size: 11px;")
        self.threshold_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        slider_row.addWidget(self.threshold_note, 2)
        box.addLayout(slider_row)
        layout.addWidget(self.threshold_panel)
        # ── 분할 판정 기준: 결함 목록을 다시 걸러 OK/NG를 정한다 (다시 추론하지 않음) ──
        self.segment_panel = QWidget()
        segment = QVBoxLayout(self.segment_panel)
        segment.setContentsMargins(0, 0, 0, 0)
        segment.setSpacing(3)
        rules = QHBoxLayout()
        rules.addWidget(QLabel("판정 기준  결함 면적 ≥"))
        self.min_area = QSpinBox()
        self.min_area.setRange(1, 100_000_000)
        self.min_area.setSuffix(" px")
        self.min_area.setKeyboardTracking(False)
        self.min_area.setAccessibleName("분할 결함 최소 면적")
        self.min_area.setToolTip("이 면적(원본 픽셀)보다 작은 예측 덩어리는 결함으로 세지 않습니다(노이즈 제거).")
        rules.addWidget(self.min_area)
        rules.addWidget(QLabel("점수 ≥"))
        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0, 1)
        self.min_score.setDecimals(2)
        self.min_score.setSingleStep(.05)
        self.min_score.setKeyboardTracking(False)
        self.min_score.setAccessibleName("분할 결함 최소 점수")
        self.min_score.setToolTip("결함 점수 = 결함 픽셀에서 그 클래스 확률의 평균(0~1). 낮은 확신의 결함을 뺍니다.")
        rules.addWidget(self.min_score)
        self.class_rules_btn = QPushButton("클래스별…")
        self.class_rules_btn.setToolTip("crack은 점수 0.85, fm은 0.80처럼 클래스마다 다른 기준을 씁니다.")
        self.zero_alarm_btn = QPushButton("오검출 0 기준 찾기…")
        self.zero_alarm_btn.setToolTip("정상 이미지 폴더를 추론한 뒤 누르세요. 정상에서는 결함이 하나도 남지 않는\n"
                                       "클래스별 기준을 찾고, 라벨 결과가 있으면 검출률도 함께 보여 줍니다.")
        self.save_rules_btn = QPushButton("모델에 저장")
        self.save_rules_btn.setToolTip("지금 기준을 모델 옆(best.rules.json)에 저장합니다.\n"
                                       "ONNX로 내보내면 함께 담기고, C++ SDK가 같은 기준으로 마스크를 정리합니다.")
        rules.addWidget(self.class_rules_btn)
        rules.addWidget(self.zero_alarm_btn)
        rules.addWidget(self.save_rules_btn)
        rules.addStretch(1)
        segment.addLayout(rules)
        self.rules_state = QLabel()
        self.rules_state.setWordWrap(True)
        self.rules_state.setStyleSheet("font-size: 11px;")
        segment.addWidget(self.rules_state)
        self.segment_summary = QLabel()
        self.segment_summary.setWordWrap(True)
        self.segment_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.segment_summary.setStyleSheet("font-size: 11px; color: #B7C9E2;")
        segment.addWidget(self.segment_summary)
        layout.addWidget(self.segment_panel)
        self.segment_panel.hide()
        settings = QSettings("DeepVisionStudio", "Inference")
        self.per_class = {}       # 클래스별 판정 기준 {이름: {"min_area", "min_score"}}
        self.model_path = None    # 판정 기준을 함께 저장할 분할 모델 (best.pt)
        self._saved_rules = None  # 모델 옆에 저장된 기준 (없으면 None)
        try:
            self.min_area.setValue(int(settings.value("segment/min_area", 1)))
            self.min_score.setValue(float(settings.value("segment/min_score", 0.0)))
            import json
            self.per_class = rules_of({}, {"per_class": json.loads(settings.value("segment/per_class", "{}") or "{}")}
                                      ).get("per_class", {})
        except (TypeError, ValueError):
            pass
        filters = QHBoxLayout()
        self.class_filter = QComboBox()
        self.class_filter.addItem("전체 클래스", "")
        self.class_filter.setToolTip("데이터셋 클래스(정답)로 필터링합니다. 모델 예측은 판정 필터를 사용하세요.")
        self.decision_filter = QComboBox()
        self.decision_filter.addItem("전체 판정", "")
        self.decision_filter.setToolTip("현재 모델의 예측 클래스 또는 실제 판정값으로 필터링합니다.")
        self.search = QLineEdit()
        self.search.setPlaceholderText("파일명 검색")
        self.search.setAccessibleName("결과 파일명 검색")
        filters.addWidget(self.class_filter)
        filters.addWidget(self.decision_filter)
        filters.addWidget(self.search, 1)
        layout.addLayout(filters)
        selection = QHBoxLayout()
        self.selected_only = QCheckBox("선택한 행만 보기")
        self.select_visible = QPushButton("표시 행 선택")
        self.clear_selection = QPushButton("선택 해제")
        selection.addWidget(self.selected_only)
        selection.addWidget(self.select_visible)
        selection.addWidget(self.clear_selection)
        layout.addLayout(selection)
        self.model = QStandardItemModel(0, len(self.HEADERS), self)
        self.model.setHorizontalHeaderLabels(self.HEADERS)
        self.proxy = ReviewFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for col, width in enumerate((38, 52, 90, 130, 55, 88, 105, 80, 80, 160)):
            self.table.setColumnWidth(col, width)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.selectionModel().currentRowChanged.connect(self._current_changed)
        layout.addWidget(self.table, 1)
        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        self.segmentation_group = QGroupBox("전체 배치 클래스별 평가 (필터와 무관)")
        segmentation_layout = QVBoxLayout(self.segmentation_group)
        self.segmentation_summary = QLabel()
        self.segmentation_summary.setTextFormat(Qt.TextFormat.PlainText)
        self.segmentation_summary.setWordWrap(True)
        segmentation_layout.addWidget(self.segmentation_summary)
        self.segmentation_table = QTableWidget(0, 4)
        self.segmentation_table.setHorizontalHeaderLabels(["클래스", "IoU", "Dice", "정답 픽셀 수"])
        self.segmentation_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.segmentation_table.verticalHeader().hide()
        self.segmentation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.segmentation_table.setMaximumHeight(165)
        segmentation_layout.addWidget(self.segmentation_table)
        layout.addWidget(self.segmentation_group)
        self._update_segmentation()
        for control in (self.class_filter, self.decision_filter):
            control.currentIndexChanged.connect(self._filter)
        self.search.textChanged.connect(self._filter)
        self.selected_only.toggled.connect(self._filter)
        self.select_visible.clicked.connect(self._select_visible)
        self.clear_selection.clicked.connect(self._clear_selection)
        self.model.itemChanged.connect(self._item_changed)
        self.override.toggled.connect(self._threshold_changed)
        self.value.valueChanged.connect(self._threshold_changed)
        self.slider.valueChanged.connect(self._slider_changed)
        self.reset.clicked.connect(lambda: self.override.setChecked(False))
        self.min_area.valueChanged.connect(self._segment_rules_changed)
        self.min_score.valueChanged.connect(self._segment_rules_changed)
        self.class_rules_btn.clicked.connect(self._edit_class_rules)
        self.zero_alarm_btn.clicked.connect(self._find_zero_alarm_rules)
        self.save_rules_btn.clicked.connect(self.save_rules_to_model)
        # 배치 요약(OK/NG 수·정답 비교)은 결과가 연달아 들어올 때 한 번만 다시 계산한다 (이미지마다 전체를 다시 세지 않음).
        self._segment_timer = QTimer(self)
        self._segment_timer.setSingleShot(True)
        self._segment_timer.setInterval(150)
        self._segment_timer.timeout.connect(self._sync_segment)
        self.set_saved_threshold(None)

    @property
    def segment_rules(self):
        """분할 결함 판정 기준. 다음 추론에도 같은 기준을 쓰고, 지난 결과는 이 기준으로 다시 판정한다."""
        return rules_of({}, {"min_area": self.min_area.value(), "min_score": round(self.min_score.value(), 4),
                             "per_class": self.per_class})

    def _segment_rules_changed(self, *_):
        import json
        settings = QSettings("DeepVisionStudio", "Inference")
        settings.setValue("segment/min_area", self.min_area.value())
        settings.setValue("segment/min_score", self.min_score.value())
        settings.setValue("segment/per_class", json.dumps(self.per_class, ensure_ascii=False))
        self._show_rules_state()
        self._threshold_changed()

    def set_rules(self, rules):
        """기준 전체(기본 + 클래스별)를 한 번에 바꾸고 결과를 한 번만 다시 판정한다."""
        rules = rules_of({}, rules)
        with QSignalBlocker(self.min_area), QSignalBlocker(self.min_score):
            self.min_area.setValue(rules["min_area"])
            self.min_score.setValue(rules["min_score"])
        self.per_class = dict(rules.get("per_class") or {})
        self._segment_rules_changed()

    def set_model_path(self, path):
        """분할 모델을 불러오면: 모델 옆에 저장된 기준이 있으면 그것으로 시작한다(내보내기·C++과 같은 기준)."""
        self.model_path = path or None
        self._saved_rules = load_model_rules(path) if path else None
        if self._saved_rules is not None:
            self.set_rules(self._saved_rules)
        self._show_rules_state()

    def save_rules_to_model(self):
        if not self.model_path:
            return None
        target = save_model_rules(self.model_path, self.segment_rules)
        self._saved_rules = self.segment_rules
        self._show_rules_state()
        return target

    def _show_rules_state(self):
        per_class = " · ".join(f"{name} 면적≥{rule['min_area']} 점수≥{rule['min_score']:.2f}"
                               for name, rule in sorted(self.per_class.items()))
        text = f"클래스별: {per_class}" if per_class else "모든 클래스에 같은 기준"
        if self.model_path:
            same = self._saved_rules == self.segment_rules
            text += (" — 모델에 저장됨 (내보내기에 포함)" if same else
                     " — <b>모델에 저장 안 됨</b>: 내보내기·C++에는 저장된 기준이 들어갑니다" if self._saved_rules else
                     " — 모델에 저장 안 됨: 내보내기에는 판정 기준 없이 들어갑니다")
            self.save_rules_btn.setEnabled(not same)
        else:
            self.save_rules_btn.setEnabled(False)
        self.rules_state.setText(text)

    def _segment_details(self):
        return [r.details for r in self.raw.values() if r.task == "segment" and "defects" in (r.details or {})]

    def _edit_class_rules(self):
        from widgets.judgement_rules_dialog import ClassRulesDialog
        details = self._segment_details()
        names = (details[0].get("class_names") if details else None) or self.class_names
        dialog = ClassRulesDialog(names, self.segment_rules, self)
        if dialog.exec() == 1:  # 확인
            self.set_rules(dialog.result_rules())

    def _find_zero_alarm_rules(self):
        from PySide6.QtWidgets import QMessageBox
        from widgets.judgement_rules_dialog import ZeroAlarmDialog
        details = self._segment_details()
        if not details:
            QMessageBox.information(self, "오검출 0 기준 찾기",
                                    "먼저 정상 이미지 폴더를 분할 모델로 추론하세요(라벨 필요 없음).\n"
                                    "라벨 있는 검증 데이터도 평가해 두면 기준을 올릴 때 놓치는 결함 수를 함께 봅니다.")
            return
        dialog = ZeroAlarmDialog(details, self.segment_rules, can_save=bool(self.model_path), parent=self)
        code = dialog.exec()
        if code in (1, ZeroAlarmDialog.SAVE):  # 적용 / 적용하고 저장
            self.set_rules(dialog.result_rules())
            if code == ZeroAlarmDialog.SAVE:
                self.save_rules_to_model()

    @property
    def threshold(self):
        return self.value.value() if self.override.isChecked() else None

    def set_saved_threshold(self, value, *, normalized=False):
        self.score_normalized = normalized
        self.saved_threshold = float(value) if finite(value) else None
        with QSignalBlocker(self.override), QSignalBlocker(self.value):
            self.override.setChecked(False)
            self.value.setRange(0, 1) if normalized else self.value.setRange(-1e12, 1e12)
            self.value.setValue(self.saved_threshold if self.saved_threshold is not None else .5)
        self._threshold_changed()

    def set_context(self, paths, project=None, *, class_names=()):
        self.project = project
        self.class_names = list(class_names)
        self.paths, self.raw, self.rows = [], {}, {}
        # Structural row signals must reach the proxy; only suppress view selection callbacks.
        with QSignalBlocker(self.table.selectionModel()):
            self.model.removeRows(0, self.model.rowCount())
            for path in paths:
                self._add_path(path)
        self._update_filters()
        self._sync_threshold()
        self._sync_segment()
        self._filter()
        self._update_segmentation()

    def set_project(self, project):
        self.project = project
        self._update_filters()
        self._sync_threshold()
        self._update_segmentation()
        self._sync_segment()

    def _add_path(self, path):
        if path in self.rows:
            return
        row = self.model.rowCount()
        self.rows[path] = row
        self.paths.append(path)
        items = [QStandardItem() for _ in self.HEADERS]
        items[0].setCheckable(True)
        items[0].setData(path, Qt.ItemDataRole.UserRole)
        for col, value in ((1, row + 1), (2, source_class(path, self.project) or "—"),
                           (3, path.replace("\\", "/").rsplit("/", 1)[-1]), (4, "대기")):
            items[col].setData(value, Qt.ItemDataRole.DisplayRole)
        items[3].setToolTip(path)
        self.model.appendRow(items)

    def update_result(self, result):
        self.raw[result.image_path] = result
        self._updating = True
        try:
            self._add_path(result.image_path)
            row = self.rows[result.image_path]
            record = review_record(result_record(result), self.threshold, self.segment_rules)
            values = {2: record["source_class"] or source_class(result.image_path, self.project) or "—",
                      4: record["decision"], 5: record["display_score"],
                      6: round(result.inference_sec * 1000, 3) if finite(result.inference_sec) else None}
            evaluation = (result.details or {}).get("segmentation_evaluation") or {}
            evaluated = evaluation.get("status") == "ok"
            region_classes = None
            if result.task == "segment":
                region_classes = [name for name, count in zip(evaluation.get("class_names", []),
                                                               evaluation.get("gt_area", [])) if count > 0]
                values[2] = ", ".join(region_classes) or "—"
            self.model.item(row, 2).setData(region_classes, Qt.ItemDataRole.UserRole)
            values.update({7: round(evaluation["mIoU"] * 100, 2) if evaluated else None,
                           8: round(evaluation["dice_score"] * 100, 2) if evaluated else None,
                           9: "완료" if evaluated else evaluation.get("reason", "정답 비교 미실행") if result.task == "segment" else ""})
            for col, value in values.items():
                self.model.item(row, col).setData(value, Qt.ItemDataRole.DisplayRole)
            self.model.item(row, 4).setForeground(QColor(record.get("color", "#5590F0")))
            self.model.item(row, 4).setToolTip(record.get("error") or record.get("summary", ""))
            self.model.item(row, 5).setToolTip(record["score_description"])
            from core.segmentation_evaluation import evaluation_lines
            for col in (7, 8, 9):
                self.model.item(row, col).setToolTip("\n".join(evaluation_lines(evaluation)))
            self.model.item(row, 6).setToolTip(
                f"{format_result_timing(result)}\n{format_runtime_stages(result).rstrip()}"
                + ("\n" + (result.details or {})["runtime_warning"] if (result.details or {}).get("runtime_warning") else ""))
        finally:
            self._updating = False
        if not self._bulk:
            self._update_filters()
            self._sync_threshold()
            self._count()
            self._update_segmentation()

    def _update_segmentation(self):
        from core.segmentation_evaluation import aggregate_evaluations, percent
        segmentation = (getattr(self.project, "task", "") == "segment"
                        or any(result.task == "segment" for result in self.raw.values()))
        for col in (7, 8, 9):
            self.table.setColumnHidden(col, not segmentation)
        if getattr(self, "_segmentation_mode", None) != segmentation:
            header = self.table.horizontalHeader()
            order = (0, 1, 2, 3, 7, 8, 9, 4, 5, 6) if segmentation else range(len(self.HEADERS))
            for visual, logical in enumerate(order):
                header.moveSection(header.visualIndex(logical), visual)
            self._segmentation_mode = segmentation
        self.table.setColumnHidden(5, segmentation)
        self.model.setHeaderData(2, Qt.Orientation.Horizontal, "정답 영역 클래스" if segmentation else "클래스")
        self.class_filter.setToolTip("정답 마스크에 해당 클래스 픽셀이 있는 이미지로 필터링합니다." if segmentation
                                    else "데이터셋 클래스(정답)로 필터링합니다. 모델 예측은 판정 필터를 사용하세요.")
        self.segmentation_group.setVisible(segmentation)
        self.segmentation_table.setRowCount(0)
        if not segmentation:
            return
        summary = aggregate_evaluations([(result.details or {}).get("segmentation_evaluation") or {}
                                         for result in self.raw.values()])
        text = f"정답 비교 {summary['evaluated']} / 처리 {len(self.raw)} / 전체 {len(self.paths)}장"
        if summary["status"] == "ok":
            text += (f" · mIoU {percent(summary['mIoU'])} · Dice {percent(summary['dice_score'])}"
                     f" · 픽셀 정확도 {percent(summary['pixel_accuracy'])}\n"
                     "픽셀 수 합산 · 배경 포함 · 정답 픽셀 없는 클래스/255 제외")
        else:
            text += "\n" + summary.get("reason", "프로젝트에 저장한 정답 마스크/폴리곤이 있어야 IoU·Dice를 계산합니다.")
        self.segmentation_summary.setText(text)
        self.segmentation_table.setRowCount(len(summary["per_class"]))
        for index, row in enumerate(summary["per_class"]):
            for col, value in enumerate((row["name"], percent(row["iou"]), percent(row["dice"]),
                                         f"{summary['gt_area'][index]:,}")):
                item = QTableWidgetItem(value)
                if row["iou"] is None:
                    item.setToolTip("정답 픽셀 없음 — 평균에서 제외")
                self.segmentation_table.setItem(index, col, item)


    def _sync_segment(self):
        """분할 결과가 있으면 판정 기준과 배치 요약(OK/NG 수, 정답 비교)을 보여준다."""
        results = [r for r in self.raw.values() if r.task == "segment" and "defects" in (r.details or {})]
        self.segment_panel.setVisible(bool(results) or getattr(self.project, "task", "") == "segment")
        if not results:
            self.segment_summary.setText("추론하면 결함마다 클래스 · 위치 · 면적 · 점수를 보고 OK/NG를 판정합니다.")
            return
        decisions = [judge(r.details, self.segment_rules)["decision"] for r in results]
        text = f"NG {decisions.count('NG')}장 · OK {decisions.count('OK')}장"
        names = (results[0].details or {}).get("class_names") or self.class_names
        summary = batch_truth_summary([r.details for r in results], self.segment_rules, names)
        if summary:
            classes = " · ".join(f"{row['class_name']} {row['iou']:.2f}" for row in summary["per_class"])
            miou = "—" if summary["miou"] is None else f"{summary['miou']:.3f}"
            text += (f"\n정답 비교 {summary['images']}장: mIoU {miou} ({classes or '결함 없음'})"
                     f"\n결함 검출 {summary['found']}/{summary['truth_defects']} · 놓침 {summary['missed']}"
                     f" · 오검출 {summary['false_positives']} · OK/NG 판정 일치 {summary['decision_accuracy']:.0%}")
        else:
            text += "\n정답 비교: 데이터셋 이미지(정답 마스크가 있는 이미지)를 추론하면 IoU와 놓침·오검출을 계산합니다."
        self.segment_summary.setText(text)

    def _update_filters(self):
        rows = []
        for path in self.paths:
            index = self.rows[path]
            result = self.raw.get(path)
            row = {"task": result.task, "details": result.details} if result is not None else {}
            row.update(source_class=(result.source_class if result else "") or source_class(path, self.project),
                       decision=self.model.item(index, 4).text())
            rows.append(row)
        options = review_filter_options(rows, project=self.project, class_names=self.class_names)
        for combo, label, values in ((self.class_filter, "전체 클래스", options["classes"]),
                                     (self.decision_filter, "전체 판정", options["decisions"])):
            current = combo.currentData()
            with QSignalBlocker(combo):
                combo.clear()
                combo.addItem(label, "")
                for name in values:
                    combo.addItem(name or "—", name or "—")
                combo.setCurrentIndex(max(0, combo.findData(current)))
        # A disappeared choice must also clear the proxy's previous predicate.
        self._filter()

    def _sync_threshold(self):
        results = [r for r in self.raw.values() if r.task == "anomaly"]
        scored = [r for r in results if r.status != "error" and finite(r.score)]
        normalized = self.score_normalized or (bool(scored) and all(normalized_patchcore(result_record(r)) for r in scored))
        with QSignalBlocker(self.value):
            self.value.setRange(0, 1) if normalized else self.value.setRange(-1e12, 1e12)
        self.threshold_panel.setVisible(bool(results) or getattr(self.project, "task", "") == "anomaly" or self.saved_threshold is not None)
        saved = sorted({float(r.threshold) for r in results if finite(r.threshold)})
        if not self.override.isChecked():
            with QSignalBlocker(self.value):
                self.value.setValue(saved[0] if len(saved) == 1 else self.saved_threshold if self.saved_threshold is not None else .5)
        values = [float(r.score) for r in results if finite(r.score)] + saved + [self.value.value()]
        low, high = min(values), max(values)
        margin = max((high - low) * .1, .01)
        self._slider_range = (0.0, 1.0) if normalized else (low - margin, high + margin)
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(1000 * (self.value.value() - self._slider_range[0]) / (self._slider_range[1] - self._slider_range[0])))
        self.value.setEnabled(self.override.isChecked())
        self.slider.setEnabled(self.override.isChecked())
        note = ", ".join(f"{v:g}" for v in saved) or (f"{self.saved_threshold:g}" if self.saved_threshold is not None else "미보정")
        self.threshold_note.setText(("0~1 | " if normalized else "") + f"스코어 ≥ 임계값: NG | 저장값: {note}")
        meaning = "0에 가까울수록 정상 학습 데이터와 유사, 1에 가까울수록 멂. 불량 확률이 아닙니다.\n" if normalized else ""
        self.threshold_note.setToolTip(meaning + self.threshold_note.text() + "\n임계값 미만: OK, 이상: NG. 모델 파일과 저장된 원본 결과는 변경하지 않습니다.")
        self.slider.setToolTip(meaning + "임계값 미만: OK | 임계값 이상: NG")

    def _threshold_changed(self, *_):
        self._bulk = True
        try:
            for result in list(self.raw.values()):
                self.update_result(result)
        finally:
            self._bulk = False
        self._update_filters()
        self._sync_threshold()
        self._count()
        self._sync_segment()
        self.threshold_changed.emit()

    def _slider_changed(self, value):
        low, high = self._slider_range
        self.value.setValue(low + (high - low) * value / 1000)

    def _filter(self, *_):
        self.proxy.class_name = self.class_filter.currentData() or ""
        self.proxy.decision = self.decision_filter.currentData() or ""
        self.proxy.search = self.search.text().casefold().strip()
        self.proxy.selected_only = self.selected_only.isChecked()
        self.proxy.invalidateFilter()
        self._count()

    def _count(self):
        checked = sum(self.model.item(i, 0).checkState() == Qt.CheckState.Checked for i in range(self.model.rowCount()))
        self.count_label.setText(f"표시 {self.proxy.rowCount()} / 전체 {self.model.rowCount()} | 선택 {checked}")

    def _item_changed(self, item):
        if not self._updating and item.column() == 0:
            self._filter()

    def _select_visible(self):
        indexes = [self.proxy.mapToSource(self.proxy.index(i, 0)).row() for i in range(self.proxy.rowCount())]
        for index in indexes:
            self.model.item(index, 0).setCheckState(Qt.CheckState.Checked)

    def _clear_selection(self):
        for i in range(self.model.rowCount()):
            self.model.item(i, 0).setCheckState(Qt.CheckState.Unchecked)

    def _current_changed(self, current, _previous):
        if current.isValid():
            row = self.proxy.mapToSource(current).row()
            self.image_selected.emit(self.model.item(row, 0).data(Qt.ItemDataRole.UserRole))

    def select_path(self, path):
        if path in self.rows:
            index = self.proxy.mapFromSource(self.model.index(self.rows[path], 1))
            if index.isValid():
                self.table.setCurrentIndex(index)
                self.table.scrollTo(index)
