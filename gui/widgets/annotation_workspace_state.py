"""Shared Windows annotation workspace for axis-aligned boxes and semantic masks."""

from __future__ import annotations

import copy

import numpy as np
from core.mask_annotations import MaskDocument
from widgets.annotation_io import AnnotationWrite


class AnnotationWorkspaceState:
    def _capture(self):
        if self.task == "detect":
            return {
                "items": copy.deepcopy(self.rows),
                "ids": list(self.ids),
                "objects": copy.deepcopy(self.object_meta),
            }
        document = copy.copy(self.document)
        document.shapes = copy.deepcopy(self.document.shapes)
        document.objects = copy.deepcopy(self.document.objects)
        document.invalidate()
        document._pixels = None
        document._owners = None
        return {"document": document}

    def _serialized(self, state):
        if self.task == "segment":
            return state["document"].payload()
        return state

    def _apply_state(self, state):
        if self.task == "segment":
            self.document = state["document"]
            self.rows = self.document.shapes
            self.object_meta = self.document.objects
        else:
            self.rows = state["items"]
            self.ids = state["ids"]
            self.object_meta = state["objects"]
        self.selected = -1
        self.selected_indices = set()
        self.view.selected_set = set()

    def _checkpoint(self, label="영역 수정"):
        self.undo_stack.append((label, self._capture()))
        self.undo_stack = self.undo_stack[-100:]
        self.redo_stack.clear()

        # Immutable base pixels are shared. Histories never retain rendered
        # full-size pixel/ownership caches, and editable data has a byte budget.
        def cost(state):
            shapes = (
                state["document"].shapes if self.task == "segment" else state["items"]
            )
            return sum(
                256
                + len(s.get("points", s.get("coordinates", []))) * 32
                + len(s.get("mask_png", ""))
                for s in shapes
            )

        budget = sum(cost(state) for _, state in self.undo_stack)
        while len(self.undo_stack) > 1 and budget > 64 * 1024 * 1024:
            _, removed = self.undo_stack.pop(0)
            budget -= cost(removed)

    def _undo(self):
        if getattr(self.view, "points", []) or self.view.gesture:
            if getattr(self.view, "points", []):
                self.view.points = self.view.points[:-2]
                self._pending(bool(self.view.points))
                self.view.viewport().update()
            else:
                self.view.cancel_gesture()
            return
        if not self.undo_stack:
            return
        label, state = self.undo_stack.pop()
        self.redo_stack.append((label, self._capture()))
        self._apply_state(state)
        self._changed()

    def _redo(self):
        if not self.redo_stack:
            return
        self.view.cancel_gesture()
        label, state = self.redo_stack.pop()
        self.undo_stack.append((label, self._capture()))
        self._apply_state(state)
        self._changed()

    def _remember_preferences(self):
        self.preferences.update(
            mode=self.view.mode,
            class_name=self.class_combo.currentText(),
            brush_size=self.brush_size.value(),
            opacity=self.opacity.value(),
            brush_tip=self.brush_tip.currentData(),
            snap=self.snap.isChecked(),
        )

    def _restore_preferences(self):
        name = self.preferences.get("class_name")
        if name in self.names:
            self.class_combo.setCurrentText(name)
        self.brush_size.setValue(self.preferences.get("brush_size", 20))
        self.opacity.setValue(self.preferences.get("opacity", 45))
        self.brush_tip.setCurrentIndex(
            max(0, self.brush_tip.findData(self.preferences.get("brush_tip", "circle")))
        )
        self.snap.setChecked(self.preferences.get("snap", False))
        mode = self.preferences.get("mode")
        if mode in self.mode_buttons:
            self._set_mode(mode)

    def _load_draft(self):
        try:
            value = self.drafts.load(
                self.image_path, self.task, self._baseline, self.names
            )
        except (OSError, ValueError, TypeError) as exc:
            self.restore_label.setText(str(exc))
            self.restore_button.hide()
            self.restore_bar.show()
            return
        if not value:
            return
        if value.get("state") == "saved":
            try:
                self._apply_state(self._deserialize(value["payload"]))
                self.reviewed = bool(value.get("reviewed"))
                self._current_state("검수 완료" if self.reviewed else "정답 있음")
                self.save_state.setText("검수 완료" if self.reviewed else "정답 불러옴")
            except (ValueError, KeyError, TypeError):
                pass
            return
        self._restorable = value
        self.view.setEnabled(False)
        self.restore_label.setText(
            "복구할 편집 초안이 있습니다. 정답 파일은 변경되지 않았습니다."
        )
        self.restore_bar.show()

    def _deserialize(self, payload):
        if self.task == "segment":
            document = MaskDocument.from_payload(
                payload, self.width_px, self.height_px, len(self.names)
            )
            document.render()  # Validate operation references before replacing the active document.
            return {"document": document}
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise TypeError("박스 초안 형식 오류")
        rows = payload["items"]
        ids = payload.get("ids", [])
        if (
            not isinstance(ids, list)
            or any(not isinstance(k, str) or not 1 <= len(k) <= 128 for k in ids)
            or len(ids) != len(rows)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("박스 식별자 오류")
        for row in rows:
            p = row.get("coordinates", [])
            cid = row.get("class_id")
            if (
                len(p) != 4
                or not np.isfinite(p).all()
                or type(cid) is not int
                or not 0 <= cid < len(self.names)
                or min(p[2:]) <= 0
                or min(p) < 0
                or max(p) > 1
            ):
                raise ValueError("박스 초안 좌표·클래스 오류")
        return {
            "items": copy.deepcopy(rows),
            "ids": list(ids),
            "objects": MaskDocument._validate_objects(payload.get("objects", {})),
        }

    def _restore_draft(self):
        if not self._restorable:
            return
        try:
            state = self._deserialize(self._restorable["payload"])
            self._checkpoint("초안 복원")
            self._apply_state(state)
            pending = self._restorable.get("pending", {})
            self._changed()
            if self.task == "segment" and pending.get("points"):
                points = pending["points"]
                if len(points) % 2 or any(
                    type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1
                    for v in points
                ):
                    raise ValueError("미완성 다각형 좌표 오류")
                self.view.points = list(points)
                self._pending(True)
                self.view.viewport().update()
            self.restore_bar.hide()
            self.view.setEnabled(True)
            self._restorable = None
        except (ValueError, KeyError, TypeError) as exc:
            self.status.setText("초안 복원 실패: " + str(exc))

    def _discard_draft(self):
        try:
            self.drafts.remove(self.image_path, self.task)
        except OSError as exc:
            self.status.setText(str(exc))
            return
        self._restorable = None
        self.restore_bar.hide()
        self.view.setEnabled(True)

    def _write_draft(self):
        if self.worker or getattr(self, "_save_queued", None) is not None:
            self._draft_timer.start()
            return
        if self.draft_worker:
            self._draft_timer.start()
            return
        if self._restorable:
            self.status.setText("복구할 초안을 복원하거나 버리기를 선택하세요.")
            return
        state = self._capture()
        baseline = self._baseline
        names = list(self.names)
        pending = {"points": list(getattr(self.view, "points", []))}
        revision = self.revision
        self.draft_worker = AnnotationWrite(
            lambda: self.drafts.save(
                self.image_path,
                self.task,
                baseline,
                self._serialized(state),
                names=names,
                pending=pending,
                label_revision=self.label_revision,
                state="draft" if self.dirty or self.pending else "saved",
            ),
            self,
        )
        self.draft_worker.finished.connect(
            lambda: self._draft_written(revision, pending)
        )
        self.draft_worker.start()

    def _draft_written(self, revision, pending):
        worker, self.draft_worker = self.draft_worker, None
        if worker.error:
            self.save_state.setText("초안 저장 실패")
            self.status.setText(worker.error)
        elif self.revision == revision:
            self.save_state.setText("초안 저장됨 · 정답 미확정")
            self._current_state("초안")
        if (
            self._draft_navigation
            and not worker.error
            and (
                revision != self.revision
                or pending.get("points", []) != getattr(self.view, "points", [])
            )
        ):
            worker.deleteLater()
            self._write_draft()
            return
        if self._draft_navigation and not worker.error:
            self.navigation_delta = self._draft_navigation
            self.accept()
        self._draft_navigation = 0
        worker.deleteLater()

    def _save(self):
        if (
            self.worker
            or getattr(self, "_save_queued", None) is not None
            or self.pending
            or self.points
            or not self.names
        ):
            return
        self._draft_timer.stop()
        self._save_revision = self.revision
        state = self._capture()
        if self.draft_worker:
            self._save_queued = state
            self.draft_worker.finished.connect(self._save_after_draft)
            self._buttons()
            return
        self._start_save(state)

    def _save_after_draft(self):
        state = getattr(self, "_save_queued", None)
        self._save_queued = None
        if state is not None:
            self._start_save(state)

    def _start_save(self, state):
        self._saving_state = state

        def save():
            payload = self._serialized(state)
            result = self._save_callback(
                payload if self.task == "segment" else payload["items"]
            )
            label_revision = (
                result.get("annotation_revision")
                if isinstance(result, dict)
                else self.label_revision
            )
            return (self._label_fingerprint(state), payload, label_revision)

        self.worker = AnnotationWrite(save, self)
        self.worker.finished.connect(self._saved)
        self.worker.start()
        self.save_state.setText("정답 저장 중")
        self._buttons()

    def _saved(self):
        worker, self.worker = self.worker, None
        if worker.error:
            self.navigation_delta = 0
            self.dirty = True
            self._save_review = False
            self.save_state.setText("정답 저장 실패")
            self.status.setText(worker.error)
            self._saving_state = None
            self._buttons()
            worker.deleteLater()
            return
        self._baseline, payload, self.label_revision = worker.value
        self._baseline_state = self._saving_state
        self._saving_state = None
        self._restorable = None
        self.restore_bar.hide()
        unchanged = self.revision == self._save_revision
        self.reviewed = bool(self._save_review and unchanged)
        self.dirty = not unchanged
        self.save_empty = False
        if self.task == "segment":
            self.document.persisted = True
        try:
            if unchanged:
                self.drafts.save(
                    self.image_path,
                    self.task,
                    self._baseline,
                    payload,
                    names=self.names,
                    state="saved",
                    reviewed=self._save_review,
                    label_revision=self.label_revision,
                )
            else:
                self._draft_timer.start()
        except OSError as exc:
            self.status.setText("정답은 저장됐지만 편집 속성 저장 실패: " + str(exc))
        self.save_state.setText(
            "검수 완료"
            if self._save_review and unchanged
            else "정답 저장됨"
            if unchanged
            else "새 수정 있음 · 정답 저장 필요"
        )
        self._current_state(
            "검수 완료" if self.reviewed else "정답 있음" if unchanged else "작업 중"
        )
        if self.navigation_delta and unchanged:
            self.accept()
        elif self.navigation_delta:
            self.navigation_delta = 0
            self.status.setText(
                "저장 중 새 수정이 생겨 이동하지 않았습니다. 다시 저장하세요."
            )
        self._save_review = False
        self._buttons()
        worker.deleteLater()

    def _review_next(self):
        if self.pending or self.worker:
            return
        self._save_review = True
        self.navigation_delta = (
            1 if self.navigation and self.navigation[0] + 1 < self.navigation[1] else 0
        )
        self._save()

    def accept(self):
        if self.worker:
            return
        self._draft_timer.stop()
        self._closing = True
        if self.ai_worker:
            self._cancel_ai()
            self.ai_worker.finished.connect(self.accept)
            self.status.setText("AI 작업을 취소한 뒤 이동합니다.")
            return
        if self.draft_worker:
            self.draft_worker.finished.connect(self.accept)
            return
        super().accept()

    def _navigate_to(self, index):
        if self.navigation and index != self.navigation[0]:
            self._navigate(index - self.navigation[0])

    def _navigate(self, delta):
        self._remember_preferences()
        if (
            not self.navigation
            or self.worker
            or not 0 <= self.navigation[0] + delta < self.navigation[1]
        ):
            return
        # Never turn an untouched image into a negative training example merely
        # because the user browsed past it. Incomplete work travels as a draft.
        if self.pending or (not self._indices() and not self.empty_confirm.isChecked()):
            self._draft_navigation = delta
            self._write_draft()
            return
        self.navigation_delta = delta
        if self.dirty:
            self._save()
        else:
            self.accept()

    def reject(self):
        self._remember_preferences()
        self._draft_navigation = 0
        if self.worker:
            self.status.setText("정답 저장이 끝난 뒤 닫을 수 있습니다.")
            return
        if self.ai_worker:
            self._closing = True
            self._cancel_ai()
            self.status.setText("AI 작업을 취소하고 있습니다. 완료 후 닫습니다.")
            self.ai_worker.finished.connect(self.reject)
            return
        if self.draft_worker:
            self.draft_worker.finished.connect(self.reject)
            return
        self._draft_timer.stop()
        if self.dirty or self.pending:
            # Closing preserves a draft, never silently discards the user's work.
            self._closing = True
            self._write_draft()
            if self.draft_worker:
                self.draft_worker.finished.connect(self._close_after_draft)
                return
        super().reject()

    def _close_after_draft(self):
        if self.draft_worker:
            return
        if self.save_state.text() == "초안 저장 실패":
            self._closing = False
            self.status.setText(
                "초안을 저장하지 못해 창을 유지했습니다. 정답을 저장하거나 다시 시도하세요."
            )
            return
        super().reject()

    def closeEvent(self, event):
        event.ignore()
        self.reject()
