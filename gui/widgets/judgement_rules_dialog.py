"""분할 판정 기준 대화상자: 클래스별 기준 편집과 '오검출 0 기준 찾기'.

    정상 트레이 폴더 추론 ─▶ [오검출 0 기준 찾기] ─▶ 클래스별 (면적, 점수) 추천 ─▶ [적용하고 모델에 저장]
                                  │                                              │
                     라벨 있는 검증 결과가 함께 있으면 검출률도 비교            best.rules.json
                                                                       (ONNX 내보내기가 함께 담고, C++ SDK가 적용)
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
                               QCheckBox, QSpinBox, QDoubleSpinBox, QDialogButtonBox, QPushButton, QHeaderView)

from core.segment_defects import class_rule, rules_of, zero_alarm_rules


class ClassRulesDialog(QDialog):
    """클래스마다 '기본 기준 사용' 또는 자기 기준(최소 면적·최소 점수)."""

    def __init__(self, class_names, rules, parent=None):
        super().__init__(parent)
        self.setWindowTitle("클래스별 판정 기준")
        self.resize(560, 360)
        self.rules = rules_of({}, rules)
        layout = QVBoxLayout(self)
        note = QLabel("체크한 클래스만 자기 기준을 씁니다. 나머지는 기본 기준"
                      f"(면적 ≥ {self.rules['min_area']}px, 점수 ≥ {self.rules['min_score']:.2f})을 따릅니다.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["클래스", "자기 기준", "면적 ≥ (px)", "점수 ≥"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        per_class = self.rules.get("per_class") or {}
        self._rows = []
        for name in [n for n in class_names if n and n.lower() != "background"]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            use = QCheckBox()
            use.setChecked(name in per_class)
            area, score = QSpinBox(), QDoubleSpinBox()
            area.setRange(1, 100_000_000)
            score.setRange(0, 1)
            score.setDecimals(2)
            score.setSingleStep(.05)
            area.setValue(class_rule(self.rules, name)[0])
            score.setValue(class_rule(self.rules, name)[1])
            for widget in (area, score):
                widget.setEnabled(use.isChecked())
                use.toggled.connect(widget.setEnabled)
            self.table.setCellWidget(row, 1, use)
            self.table.setCellWidget(row, 2, area)
            self.table.setCellWidget(row, 3, score)
            self._rows.append((name, use, area, score))
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_rules(self):
        per_class = {name: {"min_area": area.value(), "min_score": round(score.value(), 4)}
                     for name, use, area, score in self._rows if use.isChecked()}
        return rules_of({}, {"min_area": self.rules["min_area"], "min_score": self.rules["min_score"],
                             "per_class": per_class})


class ZeroAlarmDialog(QDialog):
    """지금 결과로 '정상에서는 결함이 하나도 안 남는' 클래스별 기준을 찾아 보여 준다."""

    SAVE = 2  # 적용하고 모델에 저장

    def __init__(self, details_list, rules, *, can_save=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("오검출 0 기준 찾기")
        self.resize(820, 470)
        self.details_list, self.base_rules = list(details_list), rules_of({}, rules)
        self._can_save = can_save
        self.result = None
        layout = QVBoxLayout(self)
        intro = QLabel("지금 결과 목록에서, <b>정상 이미지에는 결함이 하나도 남지 않는</b> 클래스별 기준(면적·점수)을 찾습니다. "
                       "라벨(정답 마스크)이 있는 결과가 있으면 그 결함을 가장 많이 남기는 기준을 고릅니다.<br>"
                       "정상 트레이 폴더를 추론한 뒤 '검증 데이터 평가'까지 해 두면 두 가지를 함께 봅니다.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.unlabeled_normal = QCheckBox("정답 라벨이 없는 이미지는 모두 정상(불량 없음)으로 봄")
        self.unlabeled_normal.setChecked(True)
        self.unlabeled_normal.setToolTip("정상 트레이 폴더만 추론했다면 켜 두세요. 불량이 섞인 폴더라면 끄세요.")
        self.unlabeled_normal.toggled.connect(self._compute)
        layout.addWidget(self.unlabeled_normal)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["클래스", "지금 기준 (면적·점수)", "추천 기준", "정상 오검출 (지금 → 추천)",
                                              "라벨 결함 검출 (지금 → 추천)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.apply_button = QPushButton("적용")
        self.apply_button.setToolTip("결과 목록의 판정만 이 기준으로 다시 합니다.")
        self.save_button = QPushButton("적용하고 모델에 저장")
        self.save_button.setToolTip("모델 옆(best.rules.json)에 저장합니다. ONNX로 내보내면 함께 담기고 C++ SDK가 적용합니다.")
        self.save_button.setEnabled(can_save)
        close = QPushButton("닫기")
        self.apply_button.clicked.connect(self.accept)
        self.save_button.clicked.connect(lambda: self.done(self.SAVE))
        close.clicked.connect(self.reject)
        for button in (self.apply_button, self.save_button, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.save_button.setDefault(can_save)
        self._compute()

    def _compute(self, *_):
        self.result = zero_alarm_rules(self.details_list, self.base_rules,
                                       unlabeled_are_normal=self.unlabeled_normal.isChecked())
        rows = self.result["classes"]
        self.table.setRowCount(len(rows))

        def pair(value):
            return "—" if value is None else f"≥ {value[0]}px · ≥ {value[1]:.2f}"

        for index, row in enumerate(rows):
            proposed = row["proposed"]
            cells = [row["class_name"], pair(row["current"]),
                     pair(proposed) if proposed else "점수로 못 거름",
                     f"{row['false_now']} → {row['false_after']}" if proposed else f"{row['false_now']} → ?",
                     (f"{row['found_now']}/{row['truth']} → {row['found_after']}/{row['truth']}"
                      if row["truth"] and proposed else "라벨 없음" if not row["truth"] else "—")]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 4 and row["truth"] and proposed and row["found_after"] < row["truth"]:
                    item.setToolTip("기준을 올려 놓치는 결함입니다. 대개 흐린(확신이 낮은) 결함입니다.")
                self.table.setItem(index, column, item)
        normal, labeled, bound = self.result["normal_images"], self.result["labeled_images"], self.result["bound"]
        lines = [f"정상 이미지 {normal}장 · 라벨 이미지 {labeled}장으로 계산했습니다."]
        if not rows:
            lines.append("결과에 결함 덩어리가 없습니다. 지금 기준으로 이미 오검출이 0입니다.")
        if bound is not None:
            lines.append(f"정상 {normal}장에서 오검출 0 → 이미지 한 장이 오검출을 낼 확률은 95% 신뢰도로 {bound:.1%} 이하입니다.")
        if normal < 100:
            lines.append("<b>정상 이미지가 적습니다.</b> 오검출 0을 믿으려면 수백 장이 필요합니다(300장이면 약 1%). "
                         "정상 트레이는 라벨링이 필요 없으니 많이 모아 추론하세요.")
        if not self.result["complete"]:
            lines.append("<b>일부 클래스는 기준으로 거를 수 없는 오검출이 남습니다</b>(점수 0.99 이상). "
                         "그 이미지를 정상으로 라벨링해 다시 학습하세요.")
        self.summary.setText("<br>".join(lines))
        has_proposal = any(row["proposed"] for row in rows)
        self.apply_button.setEnabled(has_proposal)
        self.save_button.setEnabled(self._can_save and has_proposal)

    def result_rules(self):
        return self.result["rules"]
