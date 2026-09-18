"""로컬 EXE 작업 기록 열기, 내보내기와 보관 기간별 정리."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
                               QPushButton, QFileDialog, QMessageBox, QSpinBox, QLabel, QHeaderView)
from core.desktop_jobs import desktop_manager


class JobHistoryDialog(QDialog):
    open_inference = Signal(str)

    def __init__(self, project_path=None, parent=None):
        super().__init__(parent)
        self.project_path = project_path
        self.setWindowTitle("작업 기록과 저장 공간")
        self.resize(950, 580)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["종류", "상태", "실행 시간", "크기 (MB)", "프로젝트"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        for label, callback in (("추론 결과 열기", self.open_result), ("ZIP 내보내기", self.export),
                                ("선택 기록 삭제", self.delete), ("새로고침", self.reload)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        self.days = QSpinBox()
        self.days.setRange(1, 3650)
        self.days.setValue(30)
        self.days.setSuffix("일")
        row.addWidget(self.days)
        old = QPushButton("기간이 지난 기록 선택")
        old.clicked.connect(self.select_old)
        row.addWidget(old)
        layout.addLayout(row)
        self.reload()

    def reload(self):
        usage = desktop_manager().storage()
        self.records = [j for j in usage["jobs"] if not self.project_path or j.get("project_path") == self.project_path]
        self.records.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        self.summary.setText(f"전체 작업 결과 {usage['bytes'] / 1048576:.1f} MB | 삭제는 작업 기록과 미리보기에만 적용")
        self.table.setRowCount(len(self.records))
        for i, job in enumerate(self.records):
            for c, value in enumerate((job["kind"], job["status"], job["created_at"], f"{job['bytes'] / 1048576:.1f}", job.get("project_path") or "없음")):
                self.table.setItem(i, c, QTableWidgetItem(value))

    def selected(self):
        return [self.records[index.row()] for index in self.table.selectionModel().selectedRows()]

    def open_result(self):
        jobs = self.selected()
        if len(jobs) == 1 and jobs[0]["kind"] == "infer":
            self.open_inference.emit(jobs[0]["id"])
            self.accept()

    def export(self):
        jobs = self.selected()
        if len(jobs) != 1:
            return
        path, _ = QFileDialog.getSaveFileName(self, "작업 기록 내보내기", f"studio-{jobs[0]['id']}.zip", "ZIP (*.zip)")
        if path:
            try:
                source = desktop_manager().export_archive(jobs[0]["id"])
                if Path(path).resolve() != source.resolve():
                    shutil.copy2(source, path)
            except (OSError, RuntimeError) as exc:
                QMessageBox.warning(self, "내보내기 실패", str(exc))

    def delete(self):
        jobs = self.selected()
        if jobs and QMessageBox.question(self, "기록 삭제", f"선택 기록 {len(jobs)}개와 미리보기 삭제?") == QMessageBox.StandardButton.Yes:
            try:
                desktop_manager().delete([j["id"] for j in jobs])
                self.reload()
            except (OSError, RuntimeError) as exc:
                QMessageBox.warning(self, "삭제 실패", str(exc))

    def select_old(self):
        from PySide6.QtCore import QItemSelectionModel
        self.table.clearSelection()
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.days.value())
        for i, job in enumerate(self.records):
            if job["created_at"] and datetime.fromisoformat(job["created_at"]) < cutoff and job["id"] != desktop_manager().active_id:
                self.table.selectionModel().select(self.table.model().index(i, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
